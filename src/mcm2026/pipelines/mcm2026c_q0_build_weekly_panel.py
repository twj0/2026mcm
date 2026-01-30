from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from mcm2026.core import paths
from mcm2026.data import io


_SCORE_COL_RE = re.compile(r"^week(?P<week>\d+)_judge(?P<judge>\d+)_score$")


@dataclass(frozen=True)
class Q0Outputs:
    weekly_panel_csv: Path
    season_features_csv: Path


def _official_dwts_path() -> Path:
    return paths.repo_root() / "mcm2026c" / "2026_MCM_Problem_C_Data.csv"


def load_dwts_official(path: Path | None = None) -> pd.DataFrame:
    fp = _official_dwts_path() if path is None else path
    return pd.read_csv(fp, na_values=["N/A"], keep_default_na=True)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "ballroom_partner": "pro_name",
        "celebrity_homecountry/region": "celebrity_homecountry_region",
    }
    return df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})


def _score_columns(df: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for c in df.columns:
        if _SCORE_COL_RE.match(c):
            cols.append(c)
    return cols


def _parse_results_week(s: str | float | None) -> int | None:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return None
    text = str(s)
    m = re.search(r"Week\s+(\d+)", text)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def build_season_features(df_wide_raw: pd.DataFrame) -> pd.DataFrame:
    df_wide = _normalize_columns(df_wide_raw).copy()

    keep_cols = [
        "season",
        "celebrity_name",
        "pro_name",
        "celebrity_industry",
        "celebrity_homestate",
        "celebrity_homecountry_region",
        "celebrity_age_during_season",
        "results",
        "placement",
    ]

    df = df_wide[[c for c in keep_cols if c in df_wide.columns]].copy()

    for c in ["celebrity_homestate", "celebrity_homecountry_region", "celebrity_industry", "results"]:
        if c in df.columns:
            df[c] = df[c].replace({"": pd.NA}).astype("string")

    if "celebrity_age_during_season" in df.columns:
        df["celebrity_age_during_season"] = pd.to_numeric(df["celebrity_age_during_season"], errors="coerce")

    if "placement" in df.columns:
        df["placement"] = pd.to_numeric(df["placement"], errors="coerce")

    if "celebrity_homecountry_region" in df.columns:
        df["is_us"] = df["celebrity_homecountry_region"].fillna("").eq("United States")
    else:
        df["is_us"] = False

    state_pop_fp = paths.raw_data_dir() / "us_census_2020_state_population.csv"
    if state_pop_fp.exists() and "celebrity_homestate" in df.columns:
        pop = pd.read_csv(state_pop_fp)
        pop = pop.rename(columns={"NAME": "state_name", "P1_001N": "state_population_2020"})
        pop = pop[["state_name", "state_population_2020"]].copy()

        fixes = {
            "Washington D.C.": "District of Columbia",
            "Washington, D.C.": "District of Columbia",
            "New Hamshire": "New Hampshire",
        }
        homestate = df["celebrity_homestate"].replace(fixes)
        df = df.assign(_homestate_norm=homestate)

        df = df.merge(pop, how="left", left_on="_homestate_norm", right_on="state_name")
        df = df.drop(columns=["state_name", "_homestate_norm"], errors="ignore")

        df.loc[~df["is_us"], "state_population_2020"] = pd.NA

    df = df.sort_values(["season", "celebrity_name"], kind="mergesort").reset_index(drop=True)
    return df


def build_weekly_panel(df_wide_raw: pd.DataFrame) -> pd.DataFrame:
    df_wide = _normalize_columns(df_wide_raw).copy()

    score_cols = _score_columns(df_wide)
    id_cols = [
        "season",
        "celebrity_name",
        "pro_name",
        "results",
        "placement",
    ]
    id_cols = [c for c in id_cols if c in df_wide.columns]

    melted = df_wide.melt(id_vars=id_cols, value_vars=score_cols, var_name="score_key", value_name="judge_score")

    wk = melted["score_key"].str.extract(r"^week(?P<week>\d+)_judge(?P<judge>\d+)_score$")
    melted["week"] = pd.to_numeric(wk["week"], errors="coerce")
    melted["judge"] = pd.to_numeric(wk["judge"], errors="coerce")
    melted["judge_score"] = pd.to_numeric(melted["judge_score"], errors="coerce")

    agg = (
        melted.groupby(["season", "week", "celebrity_name", "pro_name", "results", "placement"], dropna=False)[
            "judge_score"
        ]
        .sum(min_count=1)
        .reset_index()
        .rename(columns={"judge_score": "judge_score_total"})
    )

    agg = agg.dropna(subset=["week", "judge_score_total"]).copy()

    agg["week"] = agg["week"].astype(int)

    last_active = (
        agg.loc[agg["judge_score_total"] > 0]
        .groupby(["season", "celebrity_name"], dropna=False)["week"]
        .max()
        .reset_index()
        .rename(columns={"week": "last_active_week"})
    )

    agg = agg.merge(last_active, how="left", on=["season", "celebrity_name"])
    agg = agg.loc[agg["last_active_week"].notna() & (agg["week"] <= agg["last_active_week"])].copy()

    agg["active_flag"] = agg["judge_score_total"] > 0

    exit_info = df_wide[["season", "celebrity_name", "results"]].copy()
    res = exit_info["results"].astype("string")
    exit_info["exit_type"] = "unknown"
    exit_info.loc[res.str.contains("Withdrew", na=False), "exit_type"] = "withdrew"
    exit_info.loc[res.str.contains("Eliminated", na=False), "exit_type"] = "eliminated"
    exit_info["elimination_week"] = exit_info["results"].map(_parse_results_week)

    agg = agg.merge(
        exit_info[["season", "celebrity_name", "exit_type", "elimination_week"]],
        how="left",
        on=["season", "celebrity_name"],
    )

    agg["elimination_week"] = pd.to_numeric(agg["elimination_week"], errors="coerce").astype("Int64")
    agg["exit_week_inferred"] = pd.NA
    mask_exit = agg["exit_type"].isin(["eliminated", "withdrew"])
    agg.loc[mask_exit, "exit_week_inferred"] = agg.loc[mask_exit, "elimination_week"]

    mask_fallback = mask_exit & (agg["exit_week_inferred"].isna() | (agg["exit_week_inferred"] > agg["last_active_week"]))
    agg.loc[mask_fallback, "exit_week_inferred"] = agg.loc[mask_fallback, "last_active_week"]
    agg["exit_week_inferred"] = pd.to_numeric(agg["exit_week_inferred"], errors="coerce").astype("Int64")

    agg["eliminated_this_week"] = False
    mask_elim = (agg["exit_type"] == "eliminated") & agg["exit_week_inferred"].notna() & (agg["week"] == agg["exit_week_inferred"])
    agg.loc[mask_elim, "eliminated_this_week"] = True

    agg["withdrew_this_week"] = False
    mask_wd = (agg["exit_type"] == "withdrew") & agg["exit_week_inferred"].notna() & (agg["week"] == agg["exit_week_inferred"])
    agg.loc[mask_wd, "withdrew_this_week"] = True

    week_totals = (
        agg.loc[agg["active_flag"]]
        .groupby(["season", "week"], dropna=False)["judge_score_total"]
        .sum()
        .reset_index()
        .rename(columns={"judge_score_total": "season_week_judge_total"})
    )

    agg = agg.merge(week_totals, how="left", on=["season", "week"])
    agg["judge_score_pct"] = agg["judge_score_total"] / agg["season_week_judge_total"]

    agg.loc[~agg["active_flag"], ["judge_score_pct"]] = pd.NA

    agg["judge_rank"] = (
        agg.groupby(["season", "week"], dropna=False)["judge_score_total"]
        .rank(method="average", ascending=False)
        .where(agg["active_flag"], pd.NA)
    )

    n_active = (
        agg.groupby(["season", "week"], dropna=False)["active_flag"]
        .sum()
        .reset_index()
        .rename(columns={"active_flag": "n_active"})
    )
    agg = agg.merge(n_active, how="left", on=["season", "week"])

    agg = agg.sort_values(["season", "week", "celebrity_name"], kind="mergesort").reset_index(drop=True)
    return agg


def write_processed(weekly_panel: pd.DataFrame, season_features: pd.DataFrame) -> Q0Outputs:
    out_dir = paths.processed_data_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    weekly_fp = out_dir / "dwts_weekly_panel.csv"
    features_fp = out_dir / "dwts_season_features.csv"

    io.write_csv(weekly_panel, weekly_fp)
    io.write_csv(season_features, features_fp)

    return Q0Outputs(weekly_panel_csv=weekly_fp, season_features_csv=features_fp)


def run() -> Q0Outputs:
    df = load_dwts_official()
    season_features = build_season_features(df)
    weekly_panel = build_weekly_panel(df)
    return write_processed(weekly_panel=weekly_panel, season_features=season_features)


def main() -> int:
    paths.ensure_dirs()
    out = run()
    print(f"Wrote: {out.weekly_panel_csv}")
    print(f"Wrote: {out.season_features_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
