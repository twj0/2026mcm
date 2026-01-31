from __future__ import annotations

# Ref: docs/spec/task.md
# Ref: docs/spec/architecture.md

"""
Q4: New Voting System Design and Evaluation

IMPORTANT MODELING ASSUMPTIONS AND LIMITATIONS:
1. This module evaluates voting mechanisms within the Q1-identifiable fan strength space.
   It does NOT predict real-world champions but assesses mechanism trade-offs given Q1 constraints.

2. Q1 fan strength inference is based on weekly elimination constraints + judge scores.
   It may underestimate "external mobilization" cases (e.g., Bobby Bones S27) where 
   organized fan campaigns exceed what weekly constraints can identify.

3. TPI (Technical Protection Index) now uses season-average judge percentile for robustness,
   rather than final-week ranking which has small sample size issues.

4. Fan vs Uniform Contrast measures difference between realistic fan distribution and 
   uniform baseline - this is a controlled experiment metric, not direct "fan influence".

5. Robustness testing uses multiple outlier multipliers (2x, 5x, 10x) as stress tests,
   not realistic probability estimates.

For extreme cases like Bobby Bones, the framework identifies them as "identifiability 
limitations" requiring external information, rather than model failures.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from mcm2026.core import paths
from mcm2026.data import io


@dataclass(frozen=True)
class Q4Outputs:
    new_system_metrics_csv: Path


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


def _get_alpha_from_config() -> float:
    cfg = _load_config()
    node = cfg.get("dwts", {}).get("q1", {}) if isinstance(cfg, dict) else {}
    return float(node.get("alpha", 0.5))


def _get_q4_params_from_config() -> tuple[str, int, int, list[float]]:
    cfg = _load_config()
    node = cfg.get("dwts", {}).get("q4", {}) if isinstance(cfg, dict) else {}

    mech = str(node.get("fan_source_mechanism", "percent"))
    if mech not in {"percent", "rank"}:
        mech = "percent"

    n_sims = int(node.get("n_sims", 50))
    if n_sims <= 0:
        n_sims = 50

    seed = int(node.get("seed", 20260130))

    outlier_mults_raw = node.get("outlier_mults", [2.0, 5.0, 10.0])
    outlier_mults: list[float] = []
    if isinstance(outlier_mults_raw, (list, tuple)):
        for x in outlier_mults_raw:
            try:
                outlier_mults.append(float(x))
            except Exception:
                continue
    if not outlier_mults:
        outlier_mults = [2.0, 5.0, 10.0]

    return mech, n_sims, seed, outlier_mults


def _get_q4_optional_grids_from_config() -> tuple[list[float] | None, list[str] | None, list[int] | None]:
    cfg = _load_config()
    node = cfg.get("dwts", {}).get("q4", {}) if isinstance(cfg, dict) else {}

    alpha_grid_raw = node.get("alpha_grid", None)
    alpha_grid: list[float] | None = None
    if isinstance(alpha_grid_raw, (list, tuple)):
        tmp: list[float] = []
        for x in alpha_grid_raw:
            try:
                tmp.append(float(x))
            except Exception:
                continue
        if tmp:
            alpha_grid = tmp

    mechs_raw = node.get("mechanisms", None)
    mechanisms: list[str] | None = None
    if isinstance(mechs_raw, (list, tuple)):
        tmp2 = [str(x) for x in mechs_raw]
        if tmp2:
            mechanisms = tmp2

    seasons_raw = node.get("seasons", None)
    seasons: list[int] | None = None
    if isinstance(seasons_raw, (list, tuple)):
        tmp3: list[int] = []
        for x in seasons_raw:
            try:
                tmp3.append(int(x))
            except Exception:
                continue
        if tmp3:
            seasons = tmp3

    return alpha_grid, mechanisms, seasons


def _read_weekly_panel() -> pd.DataFrame:
    return io.read_table(paths.processed_data_dir() / "dwts_weekly_panel.csv")


def _read_q1_posterior_summary() -> pd.DataFrame:
    return io.read_table(paths.predictions_dir() / "mcm2026c_q1_fan_vote_posterior_summary.csv")


def _safe_entropy(counts: dict[str, int]) -> float:
    if not counts:
        return float("nan")
    total = float(sum(counts.values()))
    if total <= 0:
        return float("nan")
    p = np.asarray([v / total for v in counts.values()], dtype=float)
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def _calculate_season_tpi(champion: str, season_week_map: dict[int, pd.DataFrame], weeks: list[int]) -> float:
    """
    Calculate Technical Protection Index (TPI) as season-average judge percentile.
    This is more robust than using only final week ranking.
    """
    judge_percentiles = []
    
    for week in weeks:
        week_data = season_week_map[week]
        active_data = week_data.loc[week_data["active_flag"].astype(bool)].copy()
        
        if champion not in set(active_data["celebrity_name"].astype(str).tolist()):
            continue
            
        # Calculate judge percentile for this week
        champ_row = active_data.loc[active_data["celebrity_name"].astype(str) == champion]
        if len(champ_row) == 0:
            continue
            
        champ_judge_pct = float(champ_row["judge_score_pct"].iloc[0])
        all_judge_pcts = active_data["judge_score_pct"].astype(float).to_numpy()
        
        # Calculate percentile (higher score = higher percentile)
        percentile = float(np.mean(all_judge_pcts <= champ_judge_pct))
        judge_percentiles.append(percentile)
    
    if not judge_percentiles:
        return float("nan")
    
    return float(np.mean(judge_percentiles))


def _sample_fan_share(
    df: pd.DataFrame,
    rng: np.random.Generator,
    *,
    eps: float = 1e-9,
) -> np.ndarray:
    # Sample per-contestant positive weights via log-normal using mean and (p05,p95) width.
    # Then renormalize to a simplex.
    z = 1.6448536269514722

    p_mean = pd.to_numeric(df["fan_share_mean"], errors="coerce").to_numpy(dtype=float)

    if "fan_share_p05" in df.columns:
        p05 = pd.to_numeric(df["fan_share_p05"], errors="coerce").to_numpy(dtype=float)
    else:
        p05 = np.full(len(df), np.nan, dtype=float)

    if "fan_share_p95" in df.columns:
        p95 = pd.to_numeric(df["fan_share_p95"], errors="coerce").to_numpy(dtype=float)
    else:
        p95 = np.full(len(df), np.nan, dtype=float)

    n = len(p_mean)
    if n == 0:
        return np.asarray([], dtype=float)

    if not np.isfinite(p_mean).all():
        p_mean = np.where(np.isfinite(p_mean), p_mean, np.nan)

    if np.all(~np.isfinite(p_mean)):
        return np.ones(n, dtype=float) / float(n)

    p_mean = np.where(np.isfinite(p_mean), p_mean, np.nan)
    # Fallback missing means to uniform then renormalize.
    if np.any(np.isnan(p_mean)):
        p_mean = np.where(np.isnan(p_mean), 1.0, p_mean)

    # Approximate sigma in log space.
    p05 = np.where(np.isfinite(p05), p05, np.nan)
    p95 = np.where(np.isfinite(p95), p95, np.nan)

    mu = np.log(np.clip(p_mean, eps, 1.0))

    sigma = np.zeros(n, dtype=float)
    ok = np.isfinite(p05) & np.isfinite(p95)
    if np.any(ok):
        lo = np.log(np.clip(p05[ok], eps, 1.0))
        hi = np.log(np.clip(p95[ok], eps, 1.0))
        sigma_ok = (hi - lo) / (2.0 * z)
        sigma_ok = np.clip(sigma_ok, 0.0, 5.0)
        sigma[ok] = sigma_ok

    w = np.exp(mu + sigma * rng.normal(0.0, 1.0, size=n))
    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    s = float(w.sum())
    if not np.isfinite(s) or s <= 0:
        return np.ones(n, dtype=float) / float(n)

    return w / s


def _compress_fan_share(pf: np.ndarray, *, kind: str, eps: float = 1e-12) -> np.ndarray:
    pf = np.asarray(pf, dtype=float)
    n = len(pf)
    if n == 0:
        return pf

    pf = np.clip(pf, 0.0, 1.0)

    if kind == "sqrt":
        g = np.sqrt(pf)
    elif kind == "log":
        g = np.log(pf + eps)
        g = g - np.nanmin(g)
        g = np.where(np.isfinite(g), g, 0.0)
    elif kind == "cap":
        # winsorize at 90th percentile
        cap = float(np.quantile(pf, 0.9))
        g = np.minimum(pf, cap)
    else:
        g = pf

    g = np.where(np.isfinite(g) & (g > 0), g, 0.0)
    s = float(g.sum())
    if s <= 0 or not np.isfinite(s):
        return np.ones(n, dtype=float) / float(n)
    return g / s


def _fan_rank_from_share(pf: np.ndarray) -> np.ndarray:
    # Rank 1 is best (highest share). Ties get average rank.
    s = pd.Series(pf)
    return (-s).rank(method="average", ascending=True).to_numpy(dtype=float)


def _select_eliminated(
    df_active: pd.DataFrame,
    pf: np.ndarray,
    *,
    mechanism: str,
    alpha: float,
    k: int,
    w_dyn: tuple[float, float] = (0.35, 0.65),
    week_index: int = 0,
    n_weeks: int = 1,
) -> list[str]:
    if k <= 0:
        return []

    names = df_active["celebrity_name"].astype(str).to_numpy()
    pj = pd.to_numeric(df_active["judge_score_pct"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    pj = np.where(np.isfinite(pj) & (pj >= 0), pj, 0.0)
    pj_sum = float(pj.sum())
    if pj_sum > 0:
        pj = pj / pj_sum
    else:
        pj = np.ones(len(pj), dtype=float) / float(len(pj))

    if mechanism == "percent":
        score = alpha * pj + (1.0 - alpha) * pf
        order = np.lexsort((names, score))
        return [str(x) for x in names[order[:k]]]

    if mechanism == "rank":
        jr = pd.to_numeric(df_active["judge_rank"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        fr = _fan_rank_from_share(pf)
        comb = alpha * jr + (1.0 - alpha) * fr
        # larger is worse
        order = np.lexsort((names, -comb))
        return [str(x) for x in names[order[:k]]]

    if mechanism == "percent_sqrt":
        pf2 = _compress_fan_share(pf, kind="sqrt")
        score = alpha * pj + (1.0 - alpha) * pf2
        order = np.lexsort((names, score))
        return [str(x) for x in names[order[:k]]]

    if mechanism == "percent_log":
        pf2 = _compress_fan_share(pf, kind="log")
        score = alpha * pj + (1.0 - alpha) * pf2
        order = np.lexsort((names, score))
        return [str(x) for x in names[order[:k]]]

    if mechanism == "percent_cap":
        pf2 = _compress_fan_share(pf, kind="cap")
        score = alpha * pj + (1.0 - alpha) * pf2
        order = np.lexsort((names, score))
        return [str(x) for x in names[order[:k]]]

    if mechanism == "dynamic_weight":
        w_min, w_max = w_dyn
        if n_weeks <= 1:
            w = alpha
        else:
            w = w_min + (w_max - w_min) * (float(week_index) / float(n_weeks - 1))
        score = w * pj + (1.0 - w) * pf
        order = np.lexsort((names, score))
        return [str(x) for x in names[order[:k]]]

    if mechanism == "percent_judge_save":
        # Only defined for single elimination.
        if k != 1:
            score = alpha * pj + (1.0 - alpha) * pf
            order = np.lexsort((names, score))
            return [str(x) for x in names[order[:k]]]

        score = alpha * pj + (1.0 - alpha) * pf
        order = np.lexsort((names, score))
        bottom2 = [str(x) for x in names[order[:2]]]
        if len(bottom2) != 2:
            return bottom2

        df_idx = df_active.set_index("celebrity_name", drop=False)
        a = df_idx.loc[bottom2[0]]
        b = df_idx.loc[bottom2[1]]
        ja = float(a["judge_score_total"])
        jb = float(b["judge_score_total"])

        if ja < jb:
            return [bottom2[0]]
        if jb < ja:
            return [bottom2[1]]
        return [sorted(bottom2)[0]]

    raise ValueError(f"Unknown mechanism: {mechanism}")


def _select_champion(
    df_final: pd.DataFrame,
    pf: np.ndarray,
    *,
    mechanism: str,
    alpha: float,
) -> str:
    df = df_final.copy()
    names = df["celebrity_name"].astype(str).to_numpy()

    pj = pd.to_numeric(df["judge_score_pct"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    pj = np.where(np.isfinite(pj) & (pj >= 0), pj, 0.0)
    pj_sum = float(pj.sum())
    if pj_sum > 0:
        pj = pj / pj_sum
    else:
        pj = np.ones(len(pj), dtype=float) / float(len(pj))

    if mechanism == "rank":
        jr = pd.to_numeric(df["judge_rank"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        fr = _fan_rank_from_share(pf)
        comb = alpha * jr + (1.0 - alpha) * fr
        order = np.lexsort((names, comb))
        return str(names[order[0]])

    if mechanism == "percent_sqrt":
        pf = _compress_fan_share(pf, kind="sqrt")
    elif mechanism == "percent_log":
        pf = _compress_fan_share(pf, kind="log")
    elif mechanism == "percent_cap":
        pf = _compress_fan_share(pf, kind="cap")

    score = alpha * pj + (1.0 - alpha) * pf
    order = np.lexsort((names, -score))
    return str(names[order[0]])


def _simulate_one(
    season_week_map: dict[int, pd.DataFrame],
    weeks: list[int],
    *,
    mechanism: str,
    alpha: float,
    rng: np.random.Generator,
    use_uniform_fans: bool = False,
    outlier_mult: float | None = None,
) -> str:
    active: set[str] = set()

    # Initialize from first week.
    df0 = season_week_map[weeks[0]]
    df0 = df0.loc[df0["active_flag"].astype(bool)].copy()
    active = set(df0["celebrity_name"].astype(str).tolist())

    n_weeks = len(weeks)

    for wi, w in enumerate(weeks):
        dfw = season_week_map[w]
        dfw = dfw.loc[dfw["active_flag"].astype(bool)].copy()

        # sync active set to observed active pool (safer than purely simulated state)
        observed_active = set(dfw["celebrity_name"].astype(str).tolist())
        active = active.intersection(observed_active)

        if not active:
            break

        df_active = dfw.loc[dfw["celebrity_name"].astype(str).isin(active)].copy()
        df_active = df_active.sort_values("celebrity_name", kind="mergesort")

        # Forced withdrawals happen regardless of mechanism.
        withdrew = (
            df_active.loc[df_active["withdrew_this_week"].astype(bool), "celebrity_name"].astype(str).tolist()
        )
        for name in withdrew:
            active.discard(name)

        if not active:
            break

        df_active = df_active.loc[df_active["celebrity_name"].astype(str).isin(active)].copy()
        df_active = df_active.sort_values("celebrity_name", kind="mergesort")

        k_elim = int(df_active["eliminated_this_week"].astype(bool).sum())
        if k_elim >= len(df_active):
            k_elim = max(int(len(df_active) - 1), 0)
        if k_elim <= 0:
            continue

        if use_uniform_fans:
            pf = np.ones(len(df_active), dtype=float) / float(len(df_active))
        else:
            pf = _sample_fan_share(df_active, rng)

        if outlier_mult is not None and len(df_active) >= 2:
            # Inflate the fan share for the lowest judge-score contestant.
            pj = pd.to_numeric(df_active["judge_score_pct"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            pj = np.where(np.isfinite(pj), pj, 0.0)
            worst = int(np.lexsort((df_active["celebrity_name"].astype(str).to_numpy(), pj))[0])
            pf = pf.copy()
            pf[worst] = pf[worst] * float(outlier_mult)
            s = float(pf.sum())
            if s > 0:
                pf = pf / s

        eliminated = _select_eliminated(
            df_active,
            pf,
            mechanism=mechanism,
            alpha=alpha,
            k=k_elim,
            week_index=wi,
            n_weeks=n_weeks,
        )
        for name in eliminated:
            active.discard(name)

        if len(active) <= 1:
            break

    # Final winner selection among remaining active contestants using final week data.
    final_week = weeks[-1]
    dff = season_week_map[final_week]
    dff = dff.loc[dff["active_flag"].astype(bool)].copy()
    dff = dff.loc[dff["celebrity_name"].astype(str).isin(active)].copy()
    dff = dff.sort_values("celebrity_name", kind="mergesort")

    if len(dff) == 0:
        # Fallback: deterministic name.
        if active:
            return sorted(list(active))[0]

        dff_all = season_week_map[final_week]
        dff_all = dff_all.loc[dff_all["active_flag"].astype(bool)].copy()
        if len(dff_all) == 0:
            return ""
        dff_all = dff_all.sort_values(["judge_score_pct", "celebrity_name"], ascending=[False, True], kind="mergesort")
        return str(dff_all["celebrity_name"].iloc[0])

    if len(dff) == 1:
        return str(dff["celebrity_name"].iloc[0])

    if use_uniform_fans:
        pf_final = np.ones(len(dff), dtype=float) / float(len(dff))
    else:
        pf_final = _sample_fan_share(dff, rng)

    if outlier_mult is not None and len(dff) >= 2:
        pj = pd.to_numeric(dff["judge_score_pct"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        pj = np.where(np.isfinite(pj), pj, 0.0)
        worst = int(np.lexsort((dff["celebrity_name"].astype(str).to_numpy(), pj))[0])
        pf_final = pf_final.copy()
        pf_final[worst] = pf_final[worst] * float(outlier_mult)
        s = float(pf_final.sum())
        if s > 0:
            pf_final = pf_final / s

    return _select_champion(dff, pf_final, mechanism=mechanism, alpha=alpha)


def run(
    *,
    n_sims: int | None = None,
    seed: int | None = None,
    alpha: float | None = None,
    outlier_mults: list[float] | None = None,
    fan_source_mechanism: str | None = None,
    mechanisms: list[str] | None = None,
    seasons: list[int] | None = None,
    output_path: Path | None = None,
) -> Q4Outputs:
    paths.ensure_dirs()

    alpha_cfg = _get_alpha_from_config()
    alpha = alpha_cfg if alpha is None else float(alpha)

    mech_cfg, n_sims_cfg, seed_cfg, outlier_mults_cfg = _get_q4_params_from_config()
    fan_source_mechanism = mech_cfg if fan_source_mechanism is None else str(fan_source_mechanism)
    if fan_source_mechanism not in {"percent", "rank"}:
        fan_source_mechanism = "percent"

    n_sims = int(n_sims_cfg) if n_sims is None else int(n_sims)
    if n_sims <= 0:
        n_sims = int(n_sims_cfg)

    seed = int(seed_cfg) if seed is None else int(seed)
    
    # Default outlier multipliers for robustness stress testing
    if outlier_mults is None:
        outlier_mults = list(outlier_mults_cfg)

    alpha_grid_cfg, mechanisms_cfg, seasons_cfg = _get_q4_optional_grids_from_config()
    if mechanisms is not None:
        mechanisms_cfg = list(mechanisms)
    if seasons is not None:
        seasons_cfg = list(seasons)
    alphas = [alpha] if alpha_grid_cfg is None else list(alpha_grid_cfg)

    weekly = _read_weekly_panel()
    q1 = _read_q1_posterior_summary()

    q1p = q1.loc[q1["mechanism"].astype(str) == str(fan_source_mechanism)].copy()

    key_cols = [
        "season",
        "week",
        "celebrity_name",
        "fan_share_mean",
        "fan_share_p05",
        "fan_share_p95",
    ]
    q1p = q1p[key_cols]

    df = weekly.merge(q1p, how="left", on=["season", "week", "celebrity_name"])

    n_seasons = int(df["season"].nunique())
    print(
        "Q4: evaluating design space | "
        f"seasons={n_seasons} | n_sims={int(n_sims)} | outlier_mults={list(outlier_mults)}"
    )

    allowed_mechs = {
        "percent",
        "rank",
        "percent_judge_save",
        "percent_sqrt",
        "percent_log",
        "dynamic_weight",
        "percent_cap",
    }
    if mechanisms_cfg is None:
        mechanisms = [
            "percent",
            "rank",
            "percent_judge_save",
            "percent_sqrt",
            "percent_log",
            "dynamic_weight",
            "percent_cap",
        ]
    else:
        mechanisms = [m for m in mechanisms_cfg if m in allowed_mechs]
        if not mechanisms:
            mechanisms = [
                "percent",
                "rank",
                "percent_judge_save",
                "percent_sqrt",
                "percent_log",
                "dynamic_weight",
                "percent_cap",
            ]

    rng_master = np.random.default_rng(int(seed))

    rows: list[dict] = []

    if seasons_cfg is not None:
        df = df.loc[df["season"].astype(int).isin(set(int(x) for x in seasons_cfg))].copy()
        n_seasons = int(df["season"].nunique())

    for season_i, (season, g) in enumerate(df.groupby("season", sort=True), start=1):
        print(f"Q4: season {int(season)} ({season_i}/{n_seasons})")
        g = g.copy()
        g["week"] = g["week"].astype(int)
        season_weeks = sorted(g["week"].unique().tolist())
        if not season_weeks:
            continue

        season_week_map = {int(w): gg.copy() for w, gg in g.groupby("week", sort=True)}

        final_week = season_weeks[-1]
        dff = season_week_map[final_week]
        dff = dff.loc[dff["active_flag"].astype(bool)].copy()
        n_finalists = int(dff["celebrity_name"].nunique())
        if n_finalists <= 0:
            n_finalists = int(g.loc[g["active_flag"].astype(bool), "celebrity_name"].nunique())

        top_judge_final = ""
        if len(dff) > 0:
            dff2 = dff.sort_values(["judge_score_pct", "celebrity_name"], ascending=[False, True], kind="mergesort")
            top_judge_final = str(dff2["celebrity_name"].iloc[0])

        base_seeds = [int(rng_master.integers(0, 2**31 - 1)) for _ in range(int(n_sims))]

        for alpha_eval in alphas:
            for mech in mechanisms:
                champ_counts: dict[str, int] = {}
                tpi_vals: list[float] = []
                fan_vs_uniform_vals: list[int] = []

                for base_seed in base_seeds:
                    rng_base = np.random.default_rng(base_seed)
                    rng_u = np.random.default_rng(base_seed + 1)

                    champ = _simulate_one(
                        season_week_map,
                        season_weeks,
                        mechanism=mech,
                        alpha=float(alpha_eval),
                        rng=rng_base,
                        use_uniform_fans=False,
                        outlier_mult=None,
                    )

                    champ_u = _simulate_one(
                        season_week_map,
                        season_weeks,
                        mechanism=mech,
                        alpha=float(alpha_eval),
                        rng=rng_u,
                        use_uniform_fans=True,
                        outlier_mult=None,
                    )

                    champ_counts[champ] = champ_counts.get(champ, 0) + 1
                    fan_vs_uniform_vals.append(int(champ != champ_u))

                    if champ:
                        season_tpi = _calculate_season_tpi(champ, season_week_map, season_weeks)
                        if not np.isnan(season_tpi):
                            tpi_vals.append(season_tpi)

                champ_mode = max(champ_counts.items(), key=lambda kv: kv[1])[0] if champ_counts else ""
                champ_mode_prob = float(champ_counts.get(champ_mode, 0) / float(n_sims)) if n_sims > 0 else float("nan")
                champ_entropy = _safe_entropy(champ_counts)
                tpi_season_avg = float(np.mean(tpi_vals)) if tpi_vals else float("nan")
                fan_vs_uniform_contrast = float(np.mean(fan_vs_uniform_vals)) if fan_vs_uniform_vals else float("nan")

                for outlier_mult in outlier_mults:
                    robust_fail_vals: list[int] = []

                    for base_seed in base_seeds:
                        rng_out = np.random.default_rng(base_seed + 2)
                        champ_out = _simulate_one(
                            season_week_map,
                            season_weeks,
                            mechanism=mech,
                            alpha=float(alpha_eval),
                            rng=rng_out,
                            use_uniform_fans=False,
                            outlier_mult=float(outlier_mult),
                        )
                        robust_fail_vals.append(int(top_judge_final != "" and champ_out != top_judge_final))

                    rows.append(
                        {
                            "season": int(season),
                            "mechanism": mech,
                            "alpha": float(alpha_eval),
                            "n_sims": int(n_sims),
                            "n_finalists": int(n_finalists),
                            "top_judge_final": top_judge_final,
                            "champion_mode": champ_mode,
                            "champion_mode_prob": champ_mode_prob,
                            "champion_entropy": champ_entropy,
                            "tpi_season_avg": tpi_season_avg,
                            "fan_vs_uniform_contrast": fan_vs_uniform_contrast,
                            "robust_fail_rate": float(np.mean(robust_fail_vals)) if robust_fail_vals else float("nan"),
                            "outlier_mult": float(outlier_mult),
                        }
                    )

    out = pd.DataFrame(rows)

    out_fp = (paths.tables_dir() / "mcm2026c_q4_new_system_metrics.csv") if output_path is None else Path(output_path)
    io.write_csv(out, out_fp)

    return Q4Outputs(new_system_metrics_csv=out_fp)


def main() -> int:
    out = run()
    print(f"Wrote: {out.new_system_metrics_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
