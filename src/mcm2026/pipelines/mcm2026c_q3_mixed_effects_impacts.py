from __future__ import annotations

# Ref: docs/spec/task.md
# Ref: docs/spec/architecture.md

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from mcm2026.core import paths
from mcm2026.data import io


@dataclass(frozen=True)
class Q3Outputs:
    impact_coeffs_csv: Path


def _read_weekly_panel() -> pd.DataFrame:
    return io.read_table(paths.processed_data_dir() / "dwts_weekly_panel.csv")


def _read_season_features() -> pd.DataFrame:
    return io.read_table(paths.processed_data_dir() / "dwts_season_features.csv")


def _read_q1_posterior_summary() -> pd.DataFrame:
    return io.read_table(paths.predictions_dir() / "mcm2026c_q1_fan_vote_posterior_summary.csv")


def _q1_sd_from_p05_p95(p05: pd.Series, p95: pd.Series) -> pd.Series:
    # Approximate sd from 5% and 95% quantiles assuming Normal:
    # (q95 - q05) / (2 * z0.95) where z0.95 ~= 1.64485.
    z = 1.6448536269514722
    sd = (p95.astype(float) - p05.astype(float)) / (2.0 * z)
    sd = sd.where(np.isfinite(sd), 0.0)
    sd = sd.clip(lower=0.0)
    return sd


def _rss(x: pd.Series) -> float:
    arr = pd.to_numeric(x, errors="coerce").fillna(0.0).astype(float).to_numpy()
    return float(np.sqrt(np.sum(arr * arr)))


def _build_season_level_dataset(
    weekly: pd.DataFrame,
    season_features: pd.DataFrame,
    q1_post: pd.DataFrame,
) -> pd.DataFrame:
    weekly = weekly.copy()
    weekly["active_flag"] = weekly["active_flag"].astype(bool)

    # Season-level judges outcome (technical line): mean judge_score_pct across active weeks.
    w_active = weekly.loc[weekly["active_flag"]].copy()
    judges_agg = (
        w_active.groupby(["season", "celebrity_name"], sort=True)
        .agg(
            judge_score_pct_mean=("judge_score_pct", "mean"),
            n_weeks_active=("week", "nunique"),
        )
        .reset_index()
    )

    # Fan line: use Q1 percent mechanism only; aggregate to season-level mean.
    q1p = q1_post.loc[q1_post["mechanism"].astype(str) == "percent"].copy()
    q1p = q1p.merge(
        w_active[["season", "week", "celebrity_name"]],
        how="inner",
        on=["season", "week", "celebrity_name"],
    )

    q1p["fan_index_sd"] = _q1_sd_from_p05_p95(q1p["fan_vote_index_p05"], q1p["fan_vote_index_p95"])

    fan_agg = (
        q1p.groupby(["season", "celebrity_name"], sort=True)
        .agg(
            fan_vote_index_mean=("fan_vote_index_mean", "mean"),
            fan_share_mean=("fan_share_mean", "mean"),
            fan_index_sd_rss=("fan_index_sd", _rss),
            n_weeks_q1=("week", "nunique"),
        )
        .reset_index()
    )

    # Convert RSS uncertainty across weeks into an sd of the mean.
    fan_agg["fan_vote_index_sd_mean"] = fan_agg["fan_index_sd_rss"].astype(float) / fan_agg["n_weeks_q1"].astype(float)
    fan_agg["fan_vote_index_sd_mean"] = fan_agg["fan_vote_index_sd_mean"].replace([np.inf, -np.inf], 0.0).fillna(0.0)

    # Join static season features + outcomes.
    df = season_features.merge(judges_agg, how="left", on=["season", "celebrity_name"])
    df = df.merge(fan_agg, how="left", on=["season", "celebrity_name"])

    # Minimal cleaning.
    df = df.dropna(subset=["judge_score_pct_mean", "fan_vote_index_mean", "pro_name", "celebrity_industry"]).copy()

    df["age"] = df["celebrity_age_during_season"].astype(float)
    df["age_sq"] = df["age"] ** 2

    pop = pd.to_numeric(df["state_population_2020"], errors="coerce").fillna(0.0)
    df["log_state_pop"] = np.log1p(pop)

    df["is_us"] = df["is_us"].astype(bool).astype(int)

    # Stabilize categorical levels.
    df["industry"] = df["celebrity_industry"].astype(str)
    df["pro_name"] = df["pro_name"].astype(str)
    df["season"] = df["season"].astype(int)

    return df


def _fit_mixedlm_or_ols(df: pd.DataFrame, *, y_col: str, rng: np.random.Generator | None = None) -> tuple[str, object, list[str]]:
    # MixedLM with pro random intercept and season variance component.
    # Fallback to OLS with pro/season fixed effects if MixedLM fails.
    formula = f"{y_col} ~ age + age_sq + C(industry) + is_us + log_state_pop"

    try:
        model = smf.mixedlm(
            formula,
            df,
            groups=df["pro_name"],
            re_formula="1",
            vc_formula={"season": "0 + C(season)"},
        )
        # deterministic-ish initialization
        start_params = None
        if rng is not None:
            # small random perturbation to avoid singular start in some edge cases
            start_params = None
        res = model.fit(method="lbfgs", maxiter=200, disp=False, start_params=start_params)
        fe_names = list(res.model.exog_names)
        return "mixedlm", res, fe_names
    except Exception:
        # Heavy but robust fallback.
        ols_formula = formula + " + C(pro_name) + C(season)"
        res = smf.ols(ols_formula, df).fit()

        def keep_term(term: str) -> bool:
            if term.startswith("C(pro_name)"):
                return False
            if term.startswith("C(season)"):
                return False
            return True

        fe_names = [t for t in list(res.params.index) if keep_term(t)]
        return "ols", res, fe_names


def _extract_fixed_effect_table(
    *,
    outcome: str,
    model_kind: str,
    res: object,
    fe_names: list[str],
    n_obs: int,
    n_refits: int,
    suffix: str,
) -> pd.DataFrame:
    rows: list[dict] = []

    if model_kind == "mixedlm":
        params = getattr(res, "fe_params", None)
        if params is None:
            params = res.params

        # MixedLM provides bse_fe for fixed effects.
        bse = getattr(res, "bse_fe", None)
        if bse is None:
            bse = res.bse

        pvals = getattr(res, "pvalues", pd.Series(dtype=float))

        for term in fe_names:
            est = float(params[term])
            if hasattr(bse, "index") and term in bse.index:
                se = float(bse[term])
            else:
                se = float("nan")
            ci_low = est - 1.96 * se
            ci_high = est + 1.96 * se
            p = float(pvals[term]) if term in getattr(pvals, "index", []) else float("nan")
            rows.append(
                {
                    "outcome": outcome,
                    "model": model_kind,
                    "term": term,
                    "estimate": est,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "std_err": se,
                    "p_value": p,
                    "n_obs": int(n_obs),
                    "n_refits": int(n_refits),
                    "note": suffix,
                }
            )

    else:
        params = res.params
        bse = res.bse
        pvals = res.pvalues
        for term in fe_names:
            est = float(params[term])
            se = float(bse[term])
            ci_low = float(res.conf_int().loc[term, 0])
            ci_high = float(res.conf_int().loc[term, 1])
            p = float(pvals[term])
            rows.append(
                {
                    "outcome": outcome,
                    "model": model_kind,
                    "term": term,
                    "estimate": est,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "std_err": se,
                    "p_value": p,
                    "n_obs": int(n_obs),
                    "n_refits": int(n_refits),
                    "note": suffix,
                }
            )

    return pd.DataFrame(rows)


def run(*, n_refits: int = 15, seed: int = 20260130) -> Q3Outputs:
    paths.ensure_dirs()

    weekly = _read_weekly_panel()
    season_features = _read_season_features()
    q1_post = _read_q1_posterior_summary()

    df = _build_season_level_dataset(weekly, season_features, q1_post)

    rng = np.random.default_rng(int(seed))

    # 1) Judges line: single fit.
    model_kind_j, res_j, fe_names_j = _fit_mixedlm_or_ols(df, y_col="judge_score_pct_mean", rng=rng)
    out_j = _extract_fixed_effect_table(
        outcome="judge_score_pct_mean",
        model_kind=model_kind_j,
        res=res_j,
        fe_names=fe_names_j,
        n_obs=len(df),
        n_refits=1,
        suffix="judge_line",
    )

    # 2) Fans line: refit with posterior uncertainty propagation.
    # Sample season-level fan_vote_index around mean using sd-of-mean from Q1.
    coeff_draws: list[pd.DataFrame] = []
    for k in range(int(n_refits)):
        y = df["fan_vote_index_mean"].astype(float).to_numpy(copy=True)
        sd = df["fan_vote_index_sd_mean"].astype(float).to_numpy(copy=True)
        eps = rng.normal(0.0, 1.0, size=len(df))
        y_draw = y + sd * eps

        df_k = df.copy()
        df_k["fan_vote_index_draw"] = y_draw

        model_kind_f, res_f, fe_names_f = _fit_mixedlm_or_ols(df_k, y_col="fan_vote_index_draw", rng=rng)
        tab_f = _extract_fixed_effect_table(
            outcome="fan_vote_index_mean",
            model_kind=model_kind_f,
            res=res_f,
            fe_names=fe_names_f,
            n_obs=len(df_k),
            n_refits=int(n_refits),
            suffix=f"fan_line_refit_{k+1}",
        )
        coeff_draws.append(tab_f)

    fan_all = pd.concat(coeff_draws, ignore_index=True)

    # Aggregate across refits into coefficient intervals.
    grp = fan_all.groupby(["outcome", "model", "term"], sort=True, dropna=False)
    out_f = (
        grp["estimate"]
        .agg(
            estimate="median",
            ci_low=lambda s: float(np.quantile(s.astype(float), 0.05)),
            ci_high=lambda s: float(np.quantile(s.astype(float), 0.95)),
        )
        .reset_index()
    )
    out_f["std_err"] = float("nan")
    out_f["p_value"] = float("nan")
    out_f["n_obs"] = int(len(df))
    out_f["n_refits"] = int(n_refits)
    out_f["note"] = "fan_line_posterior_refit"

    out = pd.concat([out_j, out_f], ignore_index=True)

    out_fp = paths.tables_dir() / "mcm2026c_q3_impact_analysis_coeffs.csv"
    io.write_csv(out, out_fp)

    return Q3Outputs(impact_coeffs_csv=out_fp)


def main() -> int:
    out = run()
    print(f"Wrote: {out.impact_coeffs_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
