from __future__ import annotations

# Ref: docs/spec/task.md
# Ref: docs/spec/architecture.md
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from mcm2026.core import paths
from mcm2026.data import io


def _softmax_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x - np.nanmax(x, axis=1, keepdims=True)
    ex = np.exp(x)
    ex[~np.isfinite(ex)] = 0.0
    denom = np.sum(ex, axis=1, keepdims=True)
    denom = np.where(denom > 0, denom, 1.0)
    return ex / denom


@dataclass(frozen=True)
class Q1Outputs:
    posterior_summary_csv: Path
    uncertainty_summary_csv: Path


def _config_path() -> Path:
    return paths.repo_root() / "src" / "mcm2026" / "config" / "config.yaml"


def _load_config() -> dict:
    fp = _config_path()
    if not fp.exists():
        return {}

    text = fp.read_text(encoding="utf-8")
    if not text.strip():
        return {}

    cfg = yaml.safe_load(text)
    return cfg if isinstance(cfg, dict) else {}


def _get_q1_params_from_config() -> tuple[float, float, int, int]:
    cfg = _load_config()
    node = cfg.get("dwts", {}).get("q1", {}) if isinstance(cfg, dict) else {}

    alpha = float(node.get("alpha", 0.5))
    tau = float(node.get("tau", 0.03))
    m = int(node.get("prior_draws_m", 2000))
    r = int(node.get("posterior_resample_r", 500))

    return alpha, tau, m, r


def _read_weekly_panel() -> pd.DataFrame:
    fp = paths.processed_data_dir() / "dwts_weekly_panel.csv"
    return io.read_table(fp)


def _weighted_ess(w: np.ndarray) -> float:
    s = float(w.sum())
    if not np.isfinite(s) or s <= 0:
        return 0.0

    w = w / s
    return float(1.0 / np.sum(np.square(w)))


def _stable_exp_weights(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x - np.nanmax(x)
    w = np.exp(x)
    w[~np.isfinite(w)] = 0.0
    return w


def _prob_set_without_replacement_from_probs(p: np.ndarray, selected_idx: np.ndarray) -> float | np.ndarray:
    p = np.asarray(p, dtype=float)
    selected_idx = np.asarray(selected_idx, dtype=int)

    if selected_idx.size == 0:
        return 1.0

    k = int(selected_idx.size)
    if k == 1:
        return p[..., int(selected_idx[0])]

    if k == 2:
        a = int(selected_idx[0])
        b = int(selected_idx[1])
        pa = p[..., a]
        pb = p[..., b]
        denom_a = np.clip(1.0 - pa, 1e-12, None)
        denom_b = np.clip(1.0 - pb, 1e-12, None)
        return pa * (pb / denom_a) + pb * (pa / denom_b)

    raise ValueError("Only supports k in {0,1,2} for vectorized probs")


def _prob_set_without_replacement(weights: np.ndarray, selected_idx: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=float)
    selected_idx = np.asarray(selected_idx, dtype=int)

    n = len(weights)
    if n == 0:
        return 1.0

    if selected_idx.size == 0:
        return 1.0

    if np.any(selected_idx < 0) or np.any(selected_idx >= n):
        return 0.0

    if len(set(selected_idx.tolist())) != selected_idx.size:
        return 0.0

    k = int(selected_idx.size)
    if k > 6:
        return float("nan")

    total_w = float(weights.sum())
    if total_w <= 0 or not np.isfinite(total_w):
        return 0.0

    if k == 1:
        i = int(selected_idx[0])
        return float(weights[i] / total_w)

    if k == 2:
        a = int(selected_idx[0])
        b = int(selected_idx[1])
        wa = float(weights[a])
        wb = float(weights[b])
        if wa <= 0 or wb <= 0:
            return 0.0
        denom_a = total_w - wa
        denom_b = total_w - wb
        if denom_a <= 0 or denom_b <= 0:
            return 0.0
        p_ab = (wa / total_w) * (wb / denom_a)
        p_ba = (wb / total_w) * (wa / denom_b)
        return float(p_ab + p_ba)

    if k == 3:
        a = int(selected_idx[0])
        b = int(selected_idx[1])
        c = int(selected_idx[2])
        wa = float(weights[a])
        wb = float(weights[b])
        wc = float(weights[c])
        if wa <= 0 or wb <= 0 or wc <= 0:
            return 0.0

        def p3(x1: float, x2: float, x3: float) -> float:
            denom1 = total_w - x1
            denom2 = total_w - x1 - x2
            if denom1 <= 0 or denom2 <= 0:
                return 0.0
            return (x1 / total_w) * (x2 / denom1) * (x3 / denom2)

        return float(
            p3(wa, wb, wc)
            + p3(wa, wc, wb)
            + p3(wb, wa, wc)
            + p3(wb, wc, wa)
            + p3(wc, wa, wb)
            + p3(wc, wb, wa)
        )

    raise RuntimeError("unreachable")


def _seed_for(season: int, week: int, mechanism: str, *, seed_base: int | None = None) -> int:
    mech = 1 if mechanism == "percent" else 2
    base = 0 if seed_base is None else int(seed_base)
    return base * 1_000_000 + int(season) * 1000 + int(week) * 10 + mech


def _summarize_samples(x: np.ndarray) -> tuple[float, float, float, float]:
    x = np.asarray(x, dtype=float)
    return (
        float(np.mean(x)),
        float(np.quantile(x, 0.5)),
        float(np.quantile(x, 0.05)),
        float(np.quantile(x, 0.95)),
    )


def _logit(p: np.ndarray, *, eps: float = 1e-9) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p) - np.log1p(-p)


def _infer_one_week(
    df_week: pd.DataFrame,
    *,
    mechanism: str,
    alpha: float,
    tau: float,
    m: int,
    r: int,
    seed_base: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    df_week = df_week.copy()
    df_week["active_flag"] = df_week["active_flag"].astype(bool)

    df_active = df_week.loc[df_week["active_flag"]].copy()
    n_active = int(len(df_active))

    exit_mask = df_active["eliminated_this_week"].astype(bool) | df_active["withdrew_this_week"].astype(bool)
    df_exit = df_active.loc[exit_mask].copy()
    n_exit = int(len(df_exit))

    if n_active == 0:
        summary = {
            "n_active": 0,
            "n_exit": n_exit,
            "alpha": alpha,
            "tau": tau,
            "m": m,
            "r": r,
            "ess": 0.0,
            "ess_ratio": 0.0,
            "evidence": float("nan"),
        }
        out = df_week.assign(
            mechanism=mechanism,
            fan_share_mean=pd.NA,
            fan_share_median=pd.NA,
            fan_share_p05=pd.NA,
            fan_share_p95=pd.NA,
            fan_vote_index_mean=pd.NA,
            fan_vote_index_median=pd.NA,
            fan_vote_index_p05=pd.NA,
            fan_vote_index_p95=pd.NA,
        )
        return out, summary

    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0,1)")
    if not (tau > 0.0):
        raise ValueError("tau must be > 0")
    if m <= 0 or r <= 0:
        raise ValueError("m and r must be positive")

    season = int(df_active["season"].iloc[0])
    week = int(df_active["week"].iloc[0])

    rng = np.random.default_rng(_seed_for(season, week, mechanism, seed_base=seed_base))
    pF = rng.dirichlet(alpha=np.ones(n_active, dtype=float), size=m)

    if n_exit == 0:
        like = np.ones(m, dtype=float)
    else:
        exit_names = df_exit["celebrity_name"].astype(str).tolist()
        name_to_pos = {str(nm): i for i, nm in enumerate(df_active["celebrity_name"].astype(str).tolist())}
        exit_idx = np.array([name_to_pos[nm] for nm in exit_names if nm in name_to_pos], dtype=int)

        if mechanism == "percent":
            pJ = df_active["judge_score_pct"].to_numpy(dtype=float)
            combined = alpha * pJ[None, :] + (1.0 - alpha) * pF
            p_elim = _softmax_rows((-combined) / tau)

            if n_exit <= 2:
                like = _prob_set_without_replacement_from_probs(p_elim, exit_idx)
            else:
                like = np.empty(m, dtype=float)
                score = (-combined) / tau
                for i in range(m):
                    w = _stable_exp_weights(score[i])
                    like[i] = _prob_set_without_replacement(w, exit_idx)

        elif mechanism == "rank":
            rJ = df_active["judge_rank"].to_numpy(dtype=float)
            rF = np.argsort(np.argsort(-pF, axis=1), axis=1).astype(float) + 1.0

            combined_rank = alpha * rJ[None, :] + (1.0 - alpha) * rF
            p_elim = _softmax_rows(combined_rank / tau)

            if n_exit <= 2:
                like = _prob_set_without_replacement_from_probs(p_elim, exit_idx)
            else:
                like = np.empty(m, dtype=float)
                score = combined_rank / tau
                for i in range(m):
                    w = _stable_exp_weights(score[i])
                    like[i] = _prob_set_without_replacement(w, exit_idx)

        else:
            raise ValueError(f"Unknown mechanism: {mechanism}")

    like = np.asarray(like, dtype=float)
    like[~np.isfinite(like)] = 0.0

    evidence = float(np.mean(like))

    wsum = float(like.sum())
    if wsum <= 0 or not np.isfinite(wsum):
        w = np.ones(m, dtype=float) / float(m)
    else:
        w = like / wsum

    ess = _weighted_ess(w)
    ess_ratio = float(ess / float(m)) if m > 0 else 0.0

    idx = rng.choice(np.arange(m), size=r, replace=True, p=w)
    post = pF[idx]

    share_stats = np.apply_along_axis(_summarize_samples, 0, post)
    share_mean = share_stats[0]
    share_med = share_stats[1]
    share_p05 = share_stats[2]
    share_p95 = share_stats[3]

    logit_post = _logit(post)
    idx_stats = np.apply_along_axis(_summarize_samples, 0, logit_post)
    idx_mean = idx_stats[0]
    idx_med = idx_stats[1]
    idx_p05 = idx_stats[2]
    idx_p95 = idx_stats[3]

    out_active = df_active[[
        "season",
        "week",
        "celebrity_name",
        "judge_score_pct",
        "judge_rank",
        "eliminated_this_week",
        "withdrew_this_week",
    ]].copy()

    out_active = out_active.assign(
        mechanism=mechanism,
        fan_share_mean=share_mean,
        fan_share_median=share_med,
        fan_share_p05=share_p05,
        fan_share_p95=share_p95,
        fan_vote_index_mean=idx_mean,
        fan_vote_index_median=idx_med,
        fan_vote_index_p05=idx_p05,
        fan_vote_index_p95=idx_p95,
    )

    summary = {
        "season": season,
        "week": week,
        "mechanism": mechanism,
        "n_active": n_active,
        "n_exit": n_exit,
        "alpha": alpha,
        "tau": tau,
        "m": m,
        "r": r,
        "ess": float(ess),
        "ess_ratio": float(ess_ratio),
        "evidence": float(evidence),
    }

    return out_active, summary


def _safe_rank_corr(a: pd.Series, b: pd.Series) -> float:
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    mask = a.notna() & b.notna()
    if int(mask.sum()) < 2:
        return float("nan")
    ra = a.loc[mask].rank(method="average")
    rb = b.loc[mask].rank(method="average")
    xa = ra.to_numpy(dtype=float)
    xb = rb.to_numpy(dtype=float)
    if xa.size < 2 or xb.size < 2:
        return float("nan")
    if float(np.nanstd(xa)) == 0.0 or float(np.nanstd(xb)) == 0.0:
        return float("nan")
    return float(np.corrcoef(xa, xb)[0, 1])


def _q1_week_level_diagnostics(posterior: pd.DataFrame, *, alpha: float, tau: float) -> pd.DataFrame:
    rows: list[dict] = []

    for (season, week, mechanism), g in posterior.groupby(["season", "week", "mechanism"], sort=True, dropna=False):
        gg = g.copy()
        gg["eliminated_this_week"] = gg["eliminated_this_week"].astype(bool)
        gg["withdrew_this_week"] = gg["withdrew_this_week"].astype(bool)
        gg = gg.sort_values("celebrity_name", kind="mergesort")

        n_active = int(len(gg))
        obs = gg.loc[gg["eliminated_this_week"] | gg["withdrew_this_week"], "celebrity_name"].astype(str).tolist()
        obs = sorted(obs)
        k = int(len(obs))

        fan = pd.to_numeric(gg["fan_share_mean"], errors="coerce")
        judge_pct = pd.to_numeric(gg["judge_score_pct"], errors="coerce")
        judge_rank = pd.to_numeric(gg["judge_rank"], errors="coerce")

        if n_active > 0 and fan.isna().any():
            fan = fan.fillna(1.0 / float(n_active))
            s0 = float(fan.sum(skipna=True))
            if np.isfinite(s0) and s0 > 0:
                fan = fan / s0

        fan_sum = float(fan.sum(skipna=True))
        fan_min = float(fan.min(skipna=True)) if n_active > 0 else float("nan")
        fan_max = float(fan.max(skipna=True)) if n_active > 0 else float("nan")

        w_share = pd.to_numeric(gg["fan_share_p95"], errors="coerce") - pd.to_numeric(gg["fan_share_p05"], errors="coerce")
        w_index = pd.to_numeric(gg["fan_vote_index_p95"], errors="coerce") - pd.to_numeric(
            gg["fan_vote_index_p05"], errors="coerce"
        )

        width_share_mean = float(w_share.mean(skipna=True)) if not w_share.empty else float("nan")
        width_index_mean = float(w_index.mean(skipna=True)) if not w_index.empty else float("nan")

        pred = []
        match_pred = pd.NA
        obs_exit_prob = float("nan")

        if k > 0 and n_active > 0:
            names = gg["celebrity_name"].astype(str).to_numpy()
            name_to_pos = {str(nm): i for i, nm in enumerate(names.tolist())}
            obs_idx = np.array([name_to_pos[nm] for nm in obs if nm in name_to_pos], dtype=int)

            if str(mechanism) == "percent":
                score = alpha * judge_pct.astype(float).to_numpy() + (1.0 - alpha) * fan.astype(float).to_numpy()
                order = np.lexsort((names, score))
                pred = sorted([str(x) for x in names[order[:k]]])
                p = _softmax_rows(((-score) / float(tau))[None, :])[0]
                if k <= 2:
                    obs_exit_prob = float(_prob_set_without_replacement_from_probs(p, obs_idx))
                else:
                    w = _stable_exp_weights((-score) / float(tau))
                    obs_exit_prob = float(_prob_set_without_replacement(w, obs_idx))
            else:
                fr = (-fan.astype(float)).rank(method="average", ascending=True).to_numpy(dtype=float)
                comb = alpha * judge_rank.astype(float).to_numpy() + (1.0 - alpha) * fr
                order = np.lexsort((names, -comb))
                pred = sorted([str(x) for x in names[order[:k]]])
                p = _softmax_rows(((comb) / float(tau))[None, :])[0]
                if k <= 2:
                    obs_exit_prob = float(_prob_set_without_replacement_from_probs(p, obs_idx))
                else:
                    w = _stable_exp_weights((comb) / float(tau))
                    obs_exit_prob = float(_prob_set_without_replacement(w, obs_idx))

            match_pred = int(pred == obs) if k > 0 else pd.NA

        if np.isfinite(obs_exit_prob):
            obs_exit_prob = float(np.clip(obs_exit_prob, 0.0, 1.0))

        rows.append(
            {
                "season": int(season),
                "week": int(week),
                "mechanism": str(mechanism),
                "n_active": int(n_active),
                "n_exit": int(k),
                "fan_share_sum": float(fan_sum),
                "fan_share_min": float(fan_min),
                "fan_share_max": float(fan_max),
                "fan_share_width_mean": float(width_share_mean),
                "fan_index_width_mean": float(width_index_mean),
                "judge_fan_rank_corr": _safe_rank_corr(judge_pct, fan),
                "observed_exit": "|".join(obs) if obs else "",
                "pred_exit": "|".join(pred) if pred else "",
                "match_pred": match_pred,
                "observed_exit_prob_at_posterior_mean": float(obs_exit_prob),
            }
        )

    return pd.DataFrame(rows)


def _q1_mechanism_sensitivity(posterior: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (season, week), g in posterior.groupby(["season", "week"], sort=True, dropna=False):
        p = g.loc[g["mechanism"].astype(str) == "percent", ["celebrity_name", "fan_share_mean"]].copy()
        r = g.loc[g["mechanism"].astype(str) == "rank", ["celebrity_name", "fan_share_mean"]].copy()

        if p.empty or r.empty:
            continue

        p = p.rename(columns={"fan_share_mean": "fan_share_percent"})
        r = r.rename(columns={"fan_share_mean": "fan_share_rank"})

        m = p.merge(r, how="outer", on=["celebrity_name"])
        m["fan_share_percent"] = pd.to_numeric(m["fan_share_percent"], errors="coerce").fillna(0.0)
        m["fan_share_rank"] = pd.to_numeric(m["fan_share_rank"], errors="coerce").fillna(0.0)

        sp = float(m["fan_share_percent"].sum())
        sr = float(m["fan_share_rank"].sum())
        if sp > 0:
            m["fan_share_percent"] = m["fan_share_percent"] / sp
        if sr > 0:
            m["fan_share_rank"] = m["fan_share_rank"] / sr

        tv = 0.5 * float(np.abs(m["fan_share_percent"].to_numpy() - m["fan_share_rank"].to_numpy()).sum())
        corr = _safe_rank_corr(m["fan_share_percent"], m["fan_share_rank"])

        rows.append(
            {
                "season": int(season),
                "week": int(week),
                "tv_distance": float(tv),
                "rank_corr": float(corr),
                "n_contestants": int(len(m)),
            }
        )

    return pd.DataFrame(rows)


def run(
    *,
    alpha: float | None = None,
    tau: float | None = None,
    prior_draws_m: int | None = None,
    posterior_resample_r: int | None = None,
    seed_base: int | None = None,
    output_posterior_path: Path | None = None,
    output_uncertainty_path: Path | None = None,
) -> Q1Outputs:
    paths.ensure_dirs()

    cfg_alpha, cfg_tau, cfg_m, cfg_r = _get_q1_params_from_config()
    alpha = cfg_alpha if alpha is None else float(alpha)
    tau = cfg_tau if tau is None else float(tau)
    m = cfg_m if prior_draws_m is None else int(prior_draws_m)
    r = cfg_r if posterior_resample_r is None else int(posterior_resample_r)

    df = _read_weekly_panel()

    weekly_rows: list[pd.DataFrame] = []
    uncertainty_rows: list[dict] = []

    for mechanism in ["percent", "rank"]:
        for (_, _), df_week in df.groupby(["season", "week"], sort=True, dropna=False):
            out_week, summary = _infer_one_week(
                df_week,
                mechanism=mechanism,
                alpha=alpha,
                tau=tau,
                m=m,
                r=r,
                seed_base=seed_base,
            )
            weekly_rows.append(out_week)
            uncertainty_rows.append(summary)

    posterior = pd.concat(weekly_rows, ignore_index=True)
    uncertainty = pd.DataFrame(uncertainty_rows)

    out_pred = (
        paths.predictions_dir() / "mcm2026c_q1_fan_vote_posterior_summary.csv"
        if output_posterior_path is None
        else Path(output_posterior_path)
    )
    out_unc = (
        paths.tables_dir() / "mcm2026c_q1_uncertainty_summary.csv"
        if output_uncertainty_path is None
        else Path(output_uncertainty_path)
    )

    io.write_csv(posterior, out_pred)
    io.write_csv(uncertainty, out_unc)

    if output_posterior_path is None and output_uncertainty_path is None:
        diag = _q1_week_level_diagnostics(posterior, alpha=float(alpha), tau=float(tau))
        io.write_csv(diag, paths.tables_dir() / "mcm2026c_q1_error_diagnostics_week.csv")

        sens = _q1_mechanism_sensitivity(posterior)
        io.write_csv(sens, paths.tables_dir() / "mcm2026c_q1_mechanism_sensitivity_week.csv")

    return Q1Outputs(posterior_summary_csv=out_pred, uncertainty_summary_csv=out_unc)


def main() -> int:
    out = run()
    print(f"Wrote: {out.posterior_summary_csv}")
    print(f"Wrote: {out.uncertainty_summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
