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


@dataclass(frozen=True)
class Q2Outputs:
    mechanism_comparison_csv: Path


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


def _get_q2_params_from_config() -> tuple[str, bool]:
    cfg = _load_config()
    node = cfg.get("dwts", {}).get("q2", {}) if isinstance(cfg, dict) else {}

    mech = str(node.get("fan_source_mechanism", "percent"))
    if mech not in {"percent", "rank"}:
        mech = "percent"

    count_withdraw = bool(node.get("count_withdraw_as_exit", True))
    return mech, count_withdraw


def _read_weekly_panel() -> pd.DataFrame:
    fp = paths.processed_data_dir() / "dwts_weekly_panel.csv"
    return io.read_table(fp)


def _read_q1_posterior_summary() -> pd.DataFrame:
    fp = paths.predictions_dir() / "mcm2026c_q1_fan_vote_posterior_summary.csv"
    return io.read_table(fp)


def _fan_rank_from_share(p: pd.Series) -> pd.Series:
    # Rank 1 is best (highest share). Ties get average rank.
    return (-p).rank(method="average", ascending=True)


def _select_eliminated_percent(
    df_active: pd.DataFrame,
    *,
    alpha: float,
    k: int,
) -> list[str]:
    if k <= 0:
        return []

    df = df_active.copy()
    df["combined"] = alpha * df["judge_score_pct"].astype(float) + (1.0 - alpha) * df["fan_share"].astype(float)

    df = df.sort_values(["combined", "celebrity_name"], ascending=[True, True], kind="mergesort")
    return df["celebrity_name"].astype(str).head(k).tolist()


def _select_eliminated_rank(
    df_active: pd.DataFrame,
    *,
    alpha: float,
    k: int,
) -> list[str]:
    if k <= 0:
        return []

    df = df_active.copy()
    df["fan_rank"] = _fan_rank_from_share(df["fan_share"].astype(float))
    df["combined_rank"] = alpha * df["judge_rank"].astype(float) + (1.0 - alpha) * df["fan_rank"].astype(float)

    # Larger combined_rank is worse.
    df = df.sort_values(["combined_rank", "celebrity_name"], ascending=[False, True], kind="mergesort")
    return df["celebrity_name"].astype(str).head(k).tolist()


def _judge_save_from_bottom2(
    df_active: pd.DataFrame,
    bottom2: list[str],
) -> str | None:
    if len(bottom2) != 2:
        return None

    df = df_active.set_index("celebrity_name", drop=False)
    if bottom2[0] not in df.index or bottom2[1] not in df.index:
        return None

    a = df.loc[bottom2[0]]
    b = df.loc[bottom2[1]]

    # Judges vote to eliminate the worse dancer (lower judge total).
    ja = float(a["judge_score_total"])
    jb = float(b["judge_score_total"])

    if ja < jb:
        return str(a["celebrity_name"])
    if jb < ja:
        return str(b["celebrity_name"])

    return sorted([str(a["celebrity_name"]), str(b["celebrity_name"])])[0]


def _week_level_comparison(
    df_week: pd.DataFrame,
    q1_week: pd.DataFrame,
    *,
    alpha: float,
    fan_source_mechanism: str,
    count_withdraw_as_exit: bool,
) -> dict:
    df_week = df_week.copy()
    df_week["active_flag"] = df_week["active_flag"].astype(bool)

    df_active = df_week.loc[df_week["active_flag"]].copy()
    if df_active.empty:
        return {}

    season = int(df_active["season"].iloc[0])
    week = int(df_active["week"].iloc[0])

    # Merge fan share mean (by config-selected mechanism) for active contestants.
    q1 = q1_week.loc[q1_week["mechanism"].astype(str) == str(fan_source_mechanism)].copy()
    q1 = q1[["season", "week", "celebrity_name", "fan_share_mean"]].rename(columns={"fan_share_mean": "fan_share"})

    df_active = df_active.merge(q1, how="left", on=["season", "week", "celebrity_name"])

    missing_fan_share_n = int(df_active["fan_share"].isna().sum())
    missing_fan_share_any = int(missing_fan_share_n > 0)

    # Safety: if something is missing, fall back to uniform.
    if df_active["fan_share"].isna().any():
        n = len(df_active)
        df_active["fan_share"] = df_active["fan_share"].fillna(1.0 / float(n))

    s = float(df_active["fan_share"].sum())
    if np.isfinite(s) and s > 0:
        df_active["fan_share"] = df_active["fan_share"] / s

    exit_mask = df_active["eliminated_this_week"].astype(bool)
    if bool(count_withdraw_as_exit):
        exit_mask = exit_mask | df_active["withdrew_this_week"].astype(bool)
    observed = sorted(df_active.loc[exit_mask, "celebrity_name"].astype(str).tolist())
    k = int(len(observed))

    pred_percent = sorted(_select_eliminated_percent(df_active, alpha=alpha, k=k))
    pred_rank = sorted(_select_eliminated_rank(df_active, alpha=alpha, k=k))

    match_percent = int(pred_percent == observed) if k > 0 else pd.NA
    match_rank = int(pred_rank == observed) if k > 0 else pd.NA

    diff_percent_rank = int(pred_percent != pred_rank) if k > 0 else pd.NA

    # Judge save variant: only meaningful when single exit.
    pred_percent_js = None
    pred_rank_js = None
    match_percent_js = pd.NA
    match_rank_js = pd.NA

    if k == 1 and len(df_active) >= 2:
        bottom2_percent = _select_eliminated_percent(df_active, alpha=alpha, k=2)
        bottom2_rank = _select_eliminated_rank(df_active, alpha=alpha, k=2)

        pred_percent_js = _judge_save_from_bottom2(df_active, bottom2_percent)
        pred_rank_js = _judge_save_from_bottom2(df_active, bottom2_rank)

        match_percent_js = int(sorted([pred_percent_js]) == observed) if pred_percent_js is not None else 0
        match_rank_js = int(sorted([pred_rank_js]) == observed) if pred_rank_js is not None else 0

    def _mean_fan_share(names: list[str]) -> float:
        if not names:
            return float("nan")
        sub = df_active.loc[df_active["celebrity_name"].astype(str).isin(names), "fan_share"].astype(float)
        return float(sub.mean()) if not sub.empty else float("nan")

    def _mean_judge_pct(names: list[str]) -> float:
        if not names:
            return float("nan")
        sub = df_active.loc[df_active["celebrity_name"].astype(str).isin(names), "judge_score_pct"].astype(float)
        return float(sub.mean()) if not sub.empty else float("nan")

    return {
        "season": season,
        "week": week,
        "n_active": int(len(df_active)),
        "n_exit": k,
        "alpha": float(alpha),
        "fan_source_mechanism": str(fan_source_mechanism),
        "count_withdraw_as_exit": int(bool(count_withdraw_as_exit)),
        "observed_exit": "|".join(observed) if observed else "",
        "pred_exit_percent": "|".join(pred_percent) if pred_percent else "",
        "pred_exit_rank": "|".join(pred_rank) if pred_rank else "",
        "pred_exit_percent_judge_save": pred_percent_js or "",
        "pred_exit_rank_judge_save": pred_rank_js or "",
        "match_percent": match_percent,
        "match_rank": match_rank,
        "match_percent_judge_save": match_percent_js,
        "match_rank_judge_save": match_rank_js,
        "diff_percent_rank": diff_percent_rank,
        "mean_fan_share_observed": _mean_fan_share(observed),
        "mean_fan_share_pred_percent": _mean_fan_share(pred_percent),
        "mean_fan_share_pred_rank": _mean_fan_share(pred_rank),
        "mean_judge_pct_observed": _mean_judge_pct(observed),
        "mean_judge_pct_pred_percent": _mean_judge_pct(pred_percent),
        "mean_judge_pct_pred_rank": _mean_judge_pct(pred_rank),
        "missing_fan_share_any": int(missing_fan_share_any),
        "missing_fan_share_n": int(missing_fan_share_n),
    }


def _aggregate_season_level(week_level: pd.DataFrame, *, alpha: float, fan_source_mechanism: str, count_withdraw_as_exit: bool) -> pd.DataFrame:
    def _rate(series: pd.Series) -> float:
        s = series.dropna()
        return float(s.mean()) if not s.empty else float("nan")

    season_rows: list[dict] = []
    for season, g in week_level.groupby("season", sort=True):
        exit_weeks = g.loc[g["n_exit"] > 0].copy()
        single_exit_weeks = g.loc[g["n_exit"] == 1].copy()

        season_rows.append(
            {
                "season": int(season),
                "alpha": float(alpha),
                "fan_source_mechanism": str(fan_source_mechanism),
                "count_withdraw_as_exit": int(bool(count_withdraw_as_exit)),
                "n_weeks": int(len(g)),
                "n_exit_weeks": int(len(exit_weeks)),
                "n_single_exit_weeks": int(len(single_exit_weeks)),
                "n_multi_exit_weeks": int((g["n_exit"] >= 2).sum()),
                "match_rate_percent": _rate(exit_weeks["match_percent"]),
                "match_rate_rank": _rate(exit_weeks["match_rank"]),
                "match_rate_percent_judge_save": _rate(single_exit_weeks["match_percent_judge_save"]),
                "match_rate_rank_judge_save": _rate(single_exit_weeks["match_rank_judge_save"]),
                "diff_weeks_percent_vs_rank": int((exit_weeks["diff_percent_rank"] == 1).sum()),
                "mean_fan_share_observed": float(exit_weeks["mean_fan_share_observed"].mean())
                if not exit_weeks.empty
                else float("nan"),
                "mean_fan_share_pred_percent": float(exit_weeks["mean_fan_share_pred_percent"].mean())
                if not exit_weeks.empty
                else float("nan"),
                "mean_fan_share_pred_rank": float(exit_weeks["mean_fan_share_pred_rank"].mean())
                if not exit_weeks.empty
                else float("nan"),
                "mean_judge_pct_observed": float(exit_weeks["mean_judge_pct_observed"].mean())
                if not exit_weeks.empty
                else float("nan"),
                "mean_judge_pct_pred_percent": float(exit_weeks["mean_judge_pct_pred_percent"].mean())
                if not exit_weeks.empty
                else float("nan"),
                "mean_judge_pct_pred_rank": float(exit_weeks["mean_judge_pct_pred_rank"].mean())
                if not exit_weeks.empty
                else float("nan"),
                "missing_fan_share_week_rate": float(pd.to_numeric(g.get("missing_fan_share_any"), errors="coerce").fillna(0.0).mean()),
                "missing_fan_share_total": int(pd.to_numeric(g.get("missing_fan_share_n"), errors="coerce").fillna(0.0).sum()),
            }
        )

    return pd.DataFrame(season_rows)


def run(
    *,
    alpha: float | None = None,
    fan_source_mechanism: str | None = None,
    count_withdraw_as_exit: bool | None = None,
    output_path: Path | None = None,
) -> Q2Outputs:
    paths.ensure_dirs()

    alpha_cfg = _get_alpha_from_config()
    alpha = alpha_cfg if alpha is None else float(alpha)

    fan_source_mechanism_cfg, count_withdraw_as_exit_cfg = _get_q2_params_from_config()
    fan_source_mechanism = fan_source_mechanism_cfg if fan_source_mechanism is None else str(fan_source_mechanism)
    if fan_source_mechanism not in {"percent", "rank"}:
        fan_source_mechanism = "percent"
    count_withdraw_as_exit = (
        bool(count_withdraw_as_exit_cfg) if count_withdraw_as_exit is None else bool(count_withdraw_as_exit)
    )

    weekly = _read_weekly_panel()
    q1 = _read_q1_posterior_summary()

    def _build_week_level(*, mech: str) -> pd.DataFrame:
        rows: list[dict] = []
        for (season, week), df_week in weekly.groupby(["season", "week"], sort=True, dropna=False):
            q1_week = q1.loc[(q1["season"] == season) & (q1["week"] == week)]
            row = _week_level_comparison(
                df_week,
                q1_week,
                alpha=alpha,
                fan_source_mechanism=str(mech),
                count_withdraw_as_exit=count_withdraw_as_exit,
            )
            if row:
                rows.append(row)
        return pd.DataFrame(rows)

    week_level_cfg = _build_week_level(mech=str(fan_source_mechanism))
    out = _aggregate_season_level(
        week_level_cfg,
        alpha=float(alpha),
        fan_source_mechanism=str(fan_source_mechanism),
        count_withdraw_as_exit=bool(count_withdraw_as_exit),
    )

    out_fp = (paths.tables_dir() / "mcm2026c_q2_mechanism_comparison.csv") if output_path is None else Path(output_path)
    io.write_csv(out, out_fp)

    if output_path is None:
        week_level_percent = _build_week_level(mech="percent")
        week_level_rank = _build_week_level(mech="rank")

        io.write_csv(week_level_percent, paths.tables_dir() / "mcm2026c_q2_week_level_comparison_percent.csv")
        io.write_csv(week_level_rank, paths.tables_dir() / "mcm2026c_q2_week_level_comparison_rank.csv")

        season_percent = _aggregate_season_level(
            week_level_percent,
            alpha=float(alpha),
            fan_source_mechanism="percent",
            count_withdraw_as_exit=bool(count_withdraw_as_exit),
        ).rename(columns={
            "match_rate_percent": "match_rate_percent_fan_percent",
            "match_rate_rank": "match_rate_rank_fan_percent",
            "match_rate_percent_judge_save": "match_rate_percent_judge_save_fan_percent",
            "match_rate_rank_judge_save": "match_rate_rank_judge_save_fan_percent",
            "diff_weeks_percent_vs_rank": "diff_weeks_percent_vs_rank_fan_percent",
            "missing_fan_share_week_rate": "missing_fan_share_week_rate_fan_percent",
            "missing_fan_share_total": "missing_fan_share_total_fan_percent",
        })

        season_rank = _aggregate_season_level(
            week_level_rank,
            alpha=float(alpha),
            fan_source_mechanism="rank",
            count_withdraw_as_exit=bool(count_withdraw_as_exit),
        ).rename(columns={
            "match_rate_percent": "match_rate_percent_fan_rank",
            "match_rate_rank": "match_rate_rank_fan_rank",
            "match_rate_percent_judge_save": "match_rate_percent_judge_save_fan_rank",
            "match_rate_rank_judge_save": "match_rate_rank_judge_save_fan_rank",
            "diff_weeks_percent_vs_rank": "diff_weeks_percent_vs_rank_fan_rank",
            "missing_fan_share_week_rate": "missing_fan_share_week_rate_fan_rank",
            "missing_fan_share_total": "missing_fan_share_total_fan_rank",
        })

        keep_base = ["season", "alpha", "count_withdraw_as_exit", "n_weeks", "n_exit_weeks", "n_single_exit_weeks", "n_multi_exit_weeks"]
        s1 = season_percent[[c for c in season_percent.columns if c in keep_base or c.endswith("_fan_percent")]].copy()
        s2 = season_rank[[c for c in season_rank.columns if c in keep_base or c.endswith("_fan_rank")]].copy()
        sens = s1.merge(s2, how="outer", on=keep_base)

        if "match_rate_rank_fan_percent" in sens.columns and "match_rate_rank_fan_rank" in sens.columns:
            sens["delta_match_rate_rank"] = (
                pd.to_numeric(sens["match_rate_rank_fan_percent"], errors="coerce")
                - pd.to_numeric(sens["match_rate_rank_fan_rank"], errors="coerce")
            )
        if "match_rate_percent_judge_save_fan_percent" in sens.columns and "match_rate_percent_judge_save_fan_rank" in sens.columns:
            sens["delta_match_rate_percent_judge_save"] = (
                pd.to_numeric(sens["match_rate_percent_judge_save_fan_percent"], errors="coerce")
                - pd.to_numeric(sens["match_rate_percent_judge_save_fan_rank"], errors="coerce")
            )

        io.write_csv(sens, paths.tables_dir() / "mcm2026c_q2_fan_source_sensitivity.csv")

    return Q2Outputs(mechanism_comparison_csv=out_fp)


def main() -> int:
    out = run()
    print(f"Wrote: {out.mechanism_comparison_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
