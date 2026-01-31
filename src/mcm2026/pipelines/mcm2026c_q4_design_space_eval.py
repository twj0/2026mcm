from __future__ import annotations

# Ref: docs/spec/task.md
# Ref: docs/spec/architecture.md

"""
Q4：新投票系统设计与评估

重要的建模假设和限制：
1. 该模块评估 Q1 可识别粉丝强度空间内的投票机制。它不会预测现实世界的冠军，而是评估给定第一季度约束的机制权衡。
2、Q1粉丝实力推算基于周淘汰限制+评委评分。它可能会低估“外部动员”案例（例如，Bobby Bones S27），其中有组织的粉丝活动超出了每周限制所能识别的范围。
3. TPI（技术保护指数）现在使用季节平均判断百分位来保证稳健性，而不是最后一周的排名，后者存在样本量较小的问题。
4. 扇形与均匀对比度衡量实际扇形分布与均匀分布之间的差异统一基线 - 这是一个受控实验指标，而不是直接的“粉丝影响”。
5. 稳健性测试使用多个离群值乘数（2x、5x、10x）作为压力测试，不现实的概率估计。对于像 Bobby Bones 这样的极端情况，该框架将它们识别为“可识别性”限制”需要外部信息，而不是模型失败。
"""

from dataclasses import dataclass
from pathlib import Path
import zlib

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


def _get_q4_params_from_config() -> tuple[str, int, int, list[float], int]:
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

    bootstrap_b = int(node.get("bootstrap_b", 200))
    if bootstrap_b < 0:
        bootstrap_b = 0

    return mech, n_sims, seed, outlier_mults, bootstrap_b


def _get_q4_sigma_scales_from_config() -> list[float] | None:
    cfg = _load_config()
    node = cfg.get("dwts", {}).get("q4", {}) if isinstance(cfg, dict) else {}

    raw = node.get("sigma_scales", None)
    if not isinstance(raw, (list, tuple)):
        return None

    out: list[float] = []
    for x in raw:
        try:
            v = float(x)
        except Exception:
            continue
        if v > 0:
            out.append(v)
    return out if out else None


def _get_q4_robustness_attacks_from_config() -> tuple[bool, list[str], str | None, int, float, float]:
    cfg = _load_config()
    node = cfg.get("dwts", {}).get("q4", {}) if isinstance(cfg, dict) else {}
    raw = node.get("robustness_attacks", None)
    if not isinstance(raw, dict):
        return False, [], None, 3, 0.0, 0.0

    enabled = bool(raw.get("enabled", False))

    strategies_raw = raw.get("strategies", [])
    strategies: list[str] = []
    if isinstance(strategies_raw, (list, tuple)):
        for x in strategies_raw:
            s = str(x).strip()
            if s:
                strategies.append(s)

    fixed_contestant_raw = raw.get("fixed_contestant", "")
    fixed_contestant = str(fixed_contestant_raw).strip() if fixed_contestant_raw is not None else ""
    fixed_contestant_out = fixed_contestant if fixed_contestant else None

    bottom_k = int(raw.get("bottom_k", 3))
    if bottom_k < 2:
        bottom_k = 2

    add_delta = float(raw.get("add_delta", 0.0))
    if not np.isfinite(add_delta) or add_delta < 0:
        add_delta = 0.0

    redistribute_frac = float(raw.get("redistribute_frac", 0.0))
    if not np.isfinite(redistribute_frac) or redistribute_frac < 0:
        redistribute_frac = 0.0
    if redistribute_frac > 0.95:
        redistribute_frac = 0.95

    return enabled, strategies, fixed_contestant_out, bottom_k, add_delta, redistribute_frac


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
    sigma_scale: float = 1.0,
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

    try:
        sigma_scale = float(sigma_scale)
    except Exception:
        sigma_scale = 1.0
    if not np.isfinite(sigma_scale) or sigma_scale <= 0:
        sigma_scale = 1.0
    sigma = sigma * sigma_scale

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
    sigma_scale: float = 1.0,
    rng: np.random.Generator,
    use_uniform_fans: bool = False,
    outlier_mult: float | None = None,
    outlier_target_mode: str = "worst_judge",
    outlier_fixed_name: str | None = None,
    outlier_bottom_k: int = 3,
    outlier_rng: np.random.Generator | None = None,
    outlier_attack_mode: str = "mult",
    outlier_add_delta: float = 0.0,
    outlier_redistribute_frac: float = 0.0,
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
            pf = _sample_fan_share(df_active, rng, sigma_scale=sigma_scale)

        if outlier_mult is not None and len(df_active) >= 2:
            names = df_active["celebrity_name"].astype(str).to_numpy()
            pj = pd.to_numeric(df_active["judge_score_pct"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            pj = np.where(np.isfinite(pj), pj, 0.0)

            target: int | None = None
            mode = str(outlier_target_mode)
            if mode == "fixed" and outlier_fixed_name is not None:
                hits = np.flatnonzero(names == str(outlier_fixed_name))
                if hits.size > 0:
                    target = int(hits[0])
            elif mode == "random_bottom_k":
                k = int(outlier_bottom_k)
                if k < 2:
                    k = 2
                k = min(k, int(len(names)))
                order = np.lexsort((names, pj))
                pool = order[:k]
                rng_sel = outlier_rng if outlier_rng is not None else rng
                if pool.size > 0:
                    target = int(rng_sel.choice(pool))
            else:
                target = int(np.lexsort((names, pj))[0])

            if target is not None:
                pf = pf.copy()
                mode_a = str(outlier_attack_mode)

                if mode_a == "add":
                    scale = max(float(outlier_mult) - 1.0, 0.0)
                    pf[target] = pf[target] + float(outlier_add_delta) * scale
                elif mode_a == "redistribute":
                    scale = max(float(outlier_mult) - 1.0, 0.0)
                    frac = float(outlier_redistribute_frac) * scale
                    if frac > 0.95:
                        frac = 0.95
                    if frac > 0:
                        pf_other = 1.0 - float(pf[target])
                        pf[target] = float(pf[target]) + frac * pf_other
                        if pf_other > 0:
                            keep = 1.0 - frac
                            for j in range(len(pf)):
                                if j != target:
                                    pf[j] = pf[j] * keep
                else:
                    pf[target] = pf[target] * float(outlier_mult)

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
        pf_final = _sample_fan_share(dff, rng, sigma_scale=sigma_scale)

    if outlier_mult is not None and len(dff) >= 2:
        names_f = dff["celebrity_name"].astype(str).to_numpy()
        pj = pd.to_numeric(dff["judge_score_pct"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        pj = np.where(np.isfinite(pj), pj, 0.0)

        target_f: int | None = None
        mode_f = str(outlier_target_mode)
        if mode_f == "fixed" and outlier_fixed_name is not None:
            hits = np.flatnonzero(names_f == str(outlier_fixed_name))
            if hits.size > 0:
                target_f = int(hits[0])
        elif mode_f == "random_bottom_k":
            k = int(outlier_bottom_k)
            if k < 2:
                k = 2
            k = min(k, int(len(names_f)))
            order = np.lexsort((names_f, pj))
            pool = order[:k]
            rng_sel = outlier_rng if outlier_rng is not None else rng
            if pool.size > 0:
                target_f = int(rng_sel.choice(pool))
        else:
            target_f = int(np.lexsort((names_f, pj))[0])

        if target_f is not None:
            pf_final = pf_final.copy()
            mode_af = str(outlier_attack_mode)

            if mode_af == "add":
                scale = max(float(outlier_mult) - 1.0, 0.0)
                pf_final[target_f] = pf_final[target_f] + float(outlier_add_delta) * scale
            elif mode_af == "redistribute":
                scale = max(float(outlier_mult) - 1.0, 0.0)
                frac = float(outlier_redistribute_frac) * scale
                if frac > 0.95:
                    frac = 0.95
                if frac > 0:
                    pf_other = 1.0 - float(pf_final[target_f])
                    pf_final[target_f] = float(pf_final[target_f]) + frac * pf_other
                    if pf_other > 0:
                        keep = 1.0 - frac
                        for j in range(len(pf_final)):
                            if j != target_f:
                                pf_final[j] = pf_final[j] * keep
            else:
                pf_final[target_f] = pf_final[target_f] * float(outlier_mult)

            s = float(pf_final.sum())
            if s > 0:
                pf_final = pf_final / s

    return _select_champion(dff, pf_final, mechanism=mechanism, alpha=alpha)


def _mix_seed(*parts: object) -> int:
    payload = "|".join([str(x) for x in parts]).encode("utf-8")
    return int(zlib.crc32(payload) & 0x7FFFFFFF)


def _bootstrap_mean_ci(
    values: np.ndarray,
    rng: np.random.Generator,
    *,
    alpha: float = 0.05,
    b: int = 200,
) -> tuple[float, float]:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    n = int(v.size)
    if n <= 1 or b <= 0:
        return float("nan"), float("nan")

    idx = rng.integers(0, n, size=(int(b), n))
    means = np.mean(v[idx], axis=1)
    lo, hi = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lo), float(hi)


def run(
    *,
    n_sims: int | None = None,
    seed: int | None = None,
    alpha: float | None = None,
    outlier_mults: list[float] | None = None,
    bootstrap_b: int | None = None,
    sigma_scales: list[float] | None = None,
    fan_source_mechanism: str | None = None,
    mechanisms: list[str] | None = None,
    seasons: list[int] | None = None,
    output_path: Path | None = None,
) -> Q4Outputs:
    paths.ensure_dirs()

    alpha_cfg = _get_alpha_from_config()
    alpha = alpha_cfg if alpha is None else float(alpha)

    mech_cfg, n_sims_cfg, seed_cfg, outlier_mults_cfg, bootstrap_b_cfg = _get_q4_params_from_config()
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

    bootstrap_b = int(bootstrap_b_cfg) if bootstrap_b is None else int(bootstrap_b)
    if bootstrap_b < 0:
        bootstrap_b = int(bootstrap_b_cfg)

    sigma_scales_cfg = _get_q4_sigma_scales_from_config()
    if sigma_scales is None:
        sigma_scales = [1.0] if sigma_scales_cfg is None else list(sigma_scales_cfg)
    sigma_scales = [float(x) for x in sigma_scales if np.isfinite(float(x)) and float(x) > 0]
    if not sigma_scales:
        sigma_scales = [1.0]

    (
        attacks_enabled,
        attack_strategies,
        attack_fixed_name,
        attack_bottom_k,
        attack_add_delta,
        attack_redistribute_frac,
    ) = _get_q4_robustness_attacks_from_config()

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
        f"seasons={n_seasons} | n_sims={int(n_sims)} | outlier_mults={list(outlier_mults)} | bootstrap_b={int(bootstrap_b)}"
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
            for sigma_scale in sigma_scales:
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
                            sigma_scale=float(sigma_scale),
                            rng=rng_base,
                            use_uniform_fans=False,
                            outlier_mult=None,
                        )

                        champ_u = _simulate_one(
                            season_week_map,
                            season_weeks,
                            mechanism=mech,
                            alpha=float(alpha_eval),
                            sigma_scale=float(sigma_scale),
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
                    champ_mode_prob = (
                        float(champ_counts.get(champ_mode, 0) / float(n_sims)) if n_sims > 0 else float("nan")
                    )
                    champ_entropy = _safe_entropy(champ_counts)

                    fan_vs_uniform_contrast = float(np.mean(fan_vs_uniform_vals)) if fan_vs_uniform_vals else float("nan")
                    if np.isfinite(fan_vs_uniform_contrast) and n_sims > 0:
                        fan_vs_uniform_contrast_se = float(
                            np.sqrt(fan_vs_uniform_contrast * (1.0 - fan_vs_uniform_contrast) / float(n_sims))
                        )
                    else:
                        fan_vs_uniform_contrast_se = float("nan")

                    if np.isfinite(champ_mode_prob) and n_sims > 0:
                        champion_mode_prob_se = float(
                            np.sqrt(champ_mode_prob * (1.0 - champ_mode_prob) / float(n_sims))
                        )
                    else:
                        champion_mode_prob_se = float("nan")

                    tpi_arr = np.asarray(tpi_vals, dtype=float)
                    tpi_arr = tpi_arr[np.isfinite(tpi_arr)]
                    tpi_n = int(tpi_arr.size)
                    tpi_season_avg = float(np.mean(tpi_arr)) if tpi_n > 0 else float("nan")
                    tpi_std = float(np.std(tpi_arr, ddof=1)) if tpi_n > 1 else float("nan")
                    if tpi_n > 0:
                        tpi_p05, tpi_p95 = [float(x) for x in np.quantile(tpi_arr, [0.05, 0.95])]
                    else:
                        tpi_p05, tpi_p95 = float("nan"), float("nan")

                    rng_boot = np.random.default_rng(_mix_seed(seed, season, mech, alpha_eval, sigma_scale))
                    tpi_boot_p025, tpi_boot_p975 = _bootstrap_mean_ci(
                        tpi_arr, rng_boot, alpha=0.05, b=int(bootstrap_b)
                    )

                    for outlier_mult in outlier_mults:
                        robust_fail_vals: list[int] = []

                        robust_fail_rate_fixed = float("nan")
                        robust_fail_rate_fixed_se = float("nan")
                        robust_fail_rate_bottomk = float("nan")
                        robust_fail_rate_bottomk_se = float("nan")

                        robust_fail_rate_add = float("nan")
                        robust_fail_rate_add_se = float("nan")
                        robust_fail_rate_redist = float("nan")
                        robust_fail_rate_redist_se = float("nan")

                        robust_fail_vals_fixed: list[int] = []
                        robust_fail_vals_bottomk: list[int] = []
                        robust_fail_vals_add: list[int] = []
                        robust_fail_vals_redist: list[int] = []

                        for base_seed in base_seeds:
                            rng_out = np.random.default_rng(base_seed + 2)
                            champ_out = _simulate_one(
                                season_week_map,
                                season_weeks,
                                mechanism=mech,
                                alpha=float(alpha_eval),
                                sigma_scale=float(sigma_scale),
                                rng=rng_out,
                                use_uniform_fans=False,
                                outlier_mult=float(outlier_mult),
                            )
                            robust_fail_vals.append(int(top_judge_final != "" and champ_out != top_judge_final))

                            if attacks_enabled:
                                strat = set(attack_strategies)

                                if "fixed" in strat and attack_fixed_name is not None:
                                    rng_fx = np.random.default_rng(base_seed + 2)
                                    champ_fx = _simulate_one(
                                        season_week_map,
                                        season_weeks,
                                        mechanism=mech,
                                        alpha=float(alpha_eval),
                                        sigma_scale=float(sigma_scale),
                                        rng=rng_fx,
                                        use_uniform_fans=False,
                                        outlier_mult=float(outlier_mult),
                                        outlier_target_mode="fixed",
                                        outlier_fixed_name=str(attack_fixed_name),
                                    )
                                    robust_fail_vals_fixed.append(int(top_judge_final != "" and champ_fx != top_judge_final))

                                if "random_bottom_k" in strat:
                                    rng_bk = np.random.default_rng(base_seed + 2)
                                    rng_sel = np.random.default_rng(base_seed + 2002)
                                    champ_bk = _simulate_one(
                                        season_week_map,
                                        season_weeks,
                                        mechanism=mech,
                                        alpha=float(alpha_eval),
                                        sigma_scale=float(sigma_scale),
                                        rng=rng_bk,
                                        use_uniform_fans=False,
                                        outlier_mult=float(outlier_mult),
                                        outlier_target_mode="random_bottom_k",
                                        outlier_bottom_k=int(attack_bottom_k),
                                        outlier_rng=rng_sel,
                                    )
                                    robust_fail_vals_bottomk.append(
                                        int(top_judge_final != "" and champ_bk != top_judge_final)
                                    )

                                if "add" in strat and attack_add_delta > 0:
                                    rng_add = np.random.default_rng(base_seed + 2)
                                    champ_add = _simulate_one(
                                        season_week_map,
                                        season_weeks,
                                        mechanism=mech,
                                        alpha=float(alpha_eval),
                                        sigma_scale=float(sigma_scale),
                                        rng=rng_add,
                                        use_uniform_fans=False,
                                        outlier_mult=float(outlier_mult),
                                        outlier_attack_mode="add",
                                        outlier_add_delta=float(attack_add_delta),
                                    )
                                    robust_fail_vals_add.append(int(top_judge_final != "" and champ_add != top_judge_final))

                                if "redistribute" in strat and attack_redistribute_frac > 0:
                                    rng_rd = np.random.default_rng(base_seed + 2)
                                    champ_rd = _simulate_one(
                                        season_week_map,
                                        season_weeks,
                                        mechanism=mech,
                                        alpha=float(alpha_eval),
                                        sigma_scale=float(sigma_scale),
                                        rng=rng_rd,
                                        use_uniform_fans=False,
                                        outlier_mult=float(outlier_mult),
                                        outlier_attack_mode="redistribute",
                                        outlier_redistribute_frac=float(attack_redistribute_frac),
                                    )
                                    robust_fail_vals_redist.append(int(top_judge_final != "" and champ_rd != top_judge_final))

                        robust_fail_rate = float(np.mean(robust_fail_vals)) if robust_fail_vals else float("nan")
                        if np.isfinite(robust_fail_rate) and n_sims > 0:
                            robust_fail_rate_se = float(
                                np.sqrt(robust_fail_rate * (1.0 - robust_fail_rate) / float(n_sims))
                            )
                        else:
                            robust_fail_rate_se = float("nan")

                        if attacks_enabled and robust_fail_vals_fixed:
                            robust_fail_rate_fixed = float(np.mean(robust_fail_vals_fixed))
                            if np.isfinite(robust_fail_rate_fixed) and n_sims > 0:
                                robust_fail_rate_fixed_se = float(
                                    np.sqrt(robust_fail_rate_fixed * (1.0 - robust_fail_rate_fixed) / float(n_sims))
                                )

                        if attacks_enabled and robust_fail_vals_bottomk:
                            robust_fail_rate_bottomk = float(np.mean(robust_fail_vals_bottomk))
                            if np.isfinite(robust_fail_rate_bottomk) and n_sims > 0:
                                robust_fail_rate_bottomk_se = float(
                                    np.sqrt(robust_fail_rate_bottomk * (1.0 - robust_fail_rate_bottomk) / float(n_sims))
                                )

                        if attacks_enabled and robust_fail_vals_add:
                            robust_fail_rate_add = float(np.mean(robust_fail_vals_add))
                            if np.isfinite(robust_fail_rate_add) and n_sims > 0:
                                robust_fail_rate_add_se = float(
                                    np.sqrt(robust_fail_rate_add * (1.0 - robust_fail_rate_add) / float(n_sims))
                                )

                        if attacks_enabled and robust_fail_vals_redist:
                            robust_fail_rate_redist = float(np.mean(robust_fail_vals_redist))
                            if np.isfinite(robust_fail_rate_redist) and n_sims > 0:
                                robust_fail_rate_redist_se = float(
                                    np.sqrt(robust_fail_rate_redist * (1.0 - robust_fail_rate_redist) / float(n_sims))
                                )

                        rows.append(
                            {
                                "season": int(season),
                                "mechanism": mech,
                                "alpha": float(alpha_eval),
                                "sigma_scale": float(sigma_scale),
                                "n_sims": int(n_sims),
                                "n_finalists": int(n_finalists),
                                "top_judge_final": top_judge_final,
                                "champion_mode": champ_mode,
                                "champion_mode_prob": champ_mode_prob,
                                "champion_mode_prob_se": champion_mode_prob_se,
                                "champion_entropy": champ_entropy,
                                "tpi_season_avg": tpi_season_avg,
                                "tpi_n": int(tpi_n),
                                "tpi_std": tpi_std,
                                "tpi_p05": tpi_p05,
                                "tpi_p95": tpi_p95,
                                "tpi_boot_p025": tpi_boot_p025,
                                "tpi_boot_p975": tpi_boot_p975,
                                "fan_vs_uniform_contrast": fan_vs_uniform_contrast,
                                "fan_vs_uniform_contrast_se": fan_vs_uniform_contrast_se,
                                "robust_fail_rate": robust_fail_rate,
                                "robust_fail_rate_se": robust_fail_rate_se,
                                "robust_fail_rate_fixed": robust_fail_rate_fixed,
                                "robust_fail_rate_fixed_se": robust_fail_rate_fixed_se,
                                "robust_fail_rate_random_bottom_k": robust_fail_rate_bottomk,
                                "robust_fail_rate_random_bottom_k_se": robust_fail_rate_bottomk_se,
                                "robust_fail_rate_add": robust_fail_rate_add,
                                "robust_fail_rate_add_se": robust_fail_rate_add_se,
                                "robust_fail_rate_redistribute": robust_fail_rate_redist,
                                "robust_fail_rate_redistribute_se": robust_fail_rate_redist_se,
                                "outlier_mult": float(outlier_mult),
                            }
                        )

    out = pd.DataFrame(rows)

    out_fp = (paths.tables_dir() / "mcm2026c_q4_new_system_metrics.csv") if output_path is None else Path(output_path)
    io.write_csv(out, out_fp)

    if output_path is None:
        def _fmt_float_token(x: float) -> str:
            s = ("%g" % float(x)).replace(".", "p")
            return s

        outlier_tok = "-".join(_fmt_float_token(x) for x in outlier_mults)
        sigma_tok = "-".join(_fmt_float_token(x) for x in sigma_scales)

        atk_suffix = ""
        if attacks_enabled:
            parts: list[str] = []
            strat = set(attack_strategies)
            if "fixed" in strat:
                nm = (str(attack_fixed_name) if attack_fixed_name is not None else "")
                safe = "".join([c for c in nm if c.isalnum()])
                safe = safe[:16] if safe else "NA"
                parts.append(f"fix{safe}")
            if "random_bottom_k" in strat:
                parts.append(f"bk{int(attack_bottom_k)}")
            if "add" in strat and attack_add_delta > 0:
                parts.append(f"add{_fmt_float_token(float(attack_add_delta))}")
            if "redistribute" in strat and attack_redistribute_frac > 0:
                parts.append(f"red{_fmt_float_token(float(attack_redistribute_frac))}")
            atk_suffix = "_atk" + ("-".join(parts) if parts else "on")

        param_name = (
            "mcm2026c_q4_new_system_metrics_"
            f"{fan_source_mechanism}_"
            f"{int(n_sims)}_"
            f"{int(seed)}_"
            f"{outlier_tok}_"
            f"{int(bootstrap_b)}_"
            f"{sigma_tok}"
            f"{atk_suffix}.csv"
        )
        io.write_csv(out, paths.tables_dir() / param_name)

    return Q4Outputs(new_system_metrics_csv=out_fp)


def main() -> int:
    out = run()
    print(f"Wrote: {out.new_system_metrics_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
