from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from mcm2026.core import paths
from mcm2026.data import io


@dataclass(frozen=True)
class Q0SanityOutputs:
    season_week_report_csv: Path
    contestant_report_csv: Path


def _read_weekly_panel() -> pd.DataFrame:
    fp = paths.processed_data_dir() / "dwts_weekly_panel.csv"
    return io.read_table(fp)


def _season_week_checks(df: pd.DataFrame, *, tol: float) -> pd.DataFrame:
    df = df.copy()

    df["active_flag"] = df["active_flag"].astype(bool)

    g = df.groupby(["season", "week"], dropna=False)

    rows = []
    for (season, week), part in g:
        n_rows = int(part.shape[0])

        n_active_sum = int(part["active_flag"].sum())
        n_active_nunique = int(part["n_active"].nunique(dropna=False))
        n_active_reported = int(part["n_active"].iloc[0]) if n_active_nunique == 1 else int(part["n_active"].mode().iloc[0])

        pct_sum_active = float(part.loc[part["active_flag"], "judge_score_pct"].sum())
        pct_abs_err = float(abs(pct_sum_active - 1.0)) if n_active_sum > 0 else float("nan")

        n_pct_nan_active = int(part.loc[part["active_flag"], "judge_score_pct"].isna().sum())

        n_elim_flags = int(part["eliminated_this_week"].fillna(False).astype(bool).sum())
        n_withdrew_flags = int(part["withdrew_this_week"].fillna(False).astype(bool).sum())

        issues: list[str] = []
        if n_active_nunique != 1:
            issues.append("n_active_not_constant")
        if n_active_reported != n_active_sum:
            issues.append("n_active_mismatch")
        if n_active_sum > 0 and (pct_abs_err > tol):
            issues.append("judge_score_pct_sum_not_1")
        if n_pct_nan_active > 0:
            issues.append("judge_score_pct_nan_on_active")

        rows.append(
            {
                "season": season,
                "week": week,
                "n_rows": n_rows,
                "n_active_sum": n_active_sum,
                "n_active_reported": n_active_reported,
                "n_active_nunique": n_active_nunique,
                "pct_sum_active": pct_sum_active,
                "pct_abs_err": pct_abs_err,
                "n_pct_nan_active": n_pct_nan_active,
                "n_eliminated_flags": n_elim_flags,
                "n_withdrew_flags": n_withdrew_flags,
                "issues": "|".join(issues),
            }
        )

    out = pd.DataFrame(rows)
    out = out.sort_values(["season", "week"], kind="mergesort").reset_index(drop=True)
    return out


def _contestant_checks(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["eliminated_this_week"] = df["eliminated_this_week"].fillna(False).astype(bool)
    df["withdrew_this_week"] = df["withdrew_this_week"].fillna(False).astype(bool)

    g = df.groupby(["season", "celebrity_name"], dropna=False)

    rows = []
    for (season, name), part in g:
        results = str(part["results"].iloc[0])
        last_active_week_val = int(part["last_active_week"].iloc[0]) if "last_active_week" in part.columns else None

        if "exit_type" in part.columns:
            exit_type = str(part["exit_type"].iloc[0])
        else:
            exit_type = "unknown"
            if "Withdrew" in results:
                exit_type = "withdrew"
            if "Eliminated" in results:
                exit_type = "eliminated"

        elim_week_val = None
        if "elimination_week" in part.columns:
            elim_week = part["elimination_week"].dropna().unique().tolist()
            elim_week_val = int(elim_week[0]) if len(elim_week) == 1 else None

        inferred_exit_week = None
        if "exit_week_inferred" in part.columns:
            ex = part["exit_week_inferred"].dropna().unique().tolist()
            inferred_exit_week = int(ex[0]) if len(ex) == 1 else None
        else:
            if exit_type in {"eliminated", "withdrew"}:
                inferred_exit_week = elim_week_val
                if inferred_exit_week is None and last_active_week_val is not None:
                    inferred_exit_week = last_active_week_val
                if (
                    inferred_exit_week is not None
                    and last_active_week_val is not None
                    and inferred_exit_week > last_active_week_val
                ):
                    inferred_exit_week = last_active_week_val

        n_elim_flag = int(part["eliminated_this_week"].sum())
        n_withdrew_flag = int(part["withdrew_this_week"].sum())

        issues: list[str] = []

        if exit_type == "eliminated":
            if n_elim_flag != 1:
                issues.append("expected_eliminated_flag_once")
            if n_withdrew_flag != 0:
                issues.append("unexpected_withdrew_flag")
        elif exit_type == "withdrew":
            if n_withdrew_flag != 1:
                issues.append("expected_withdrew_flag_once")
            if n_elim_flag != 0:
                issues.append("unexpected_eliminated_flag")
        else:
            if n_elim_flag != 0:
                issues.append("unexpected_eliminated_flag")
            if n_withdrew_flag != 0:
                issues.append("unexpected_withdrew_flag")

        if inferred_exit_week is not None:
            if exit_type == "eliminated":
                if not part.loc[part["week"] == inferred_exit_week, "eliminated_this_week"].any():
                    issues.append("eliminated_flag_not_on_exit_week_inferred")
            if exit_type == "withdrew":
                if not part.loc[part["week"] == inferred_exit_week, "withdrew_this_week"].any():
                    issues.append("withdrew_flag_not_on_exit_week_inferred")

        if (
            elim_week_val is not None
            and last_active_week_val is not None
            and elim_week_val <= last_active_week_val
            and inferred_exit_week is not None
            and inferred_exit_week != elim_week_val
        ):
            issues.append("exit_week_inferred_mismatch_results_week")

        rows.append(
            {
                "season": season,
                "celebrity_name": name,
                "results": results,
                "exit_type": exit_type,
                "elimination_week": elim_week_val,
                "exit_week_inferred": inferred_exit_week,
                "last_active_week": last_active_week_val,
                "n_rows": int(part.shape[0]),
                "n_eliminated_flags": n_elim_flag,
                "n_withdrew_flags": n_withdrew_flag,
                "issues": "|".join(issues),
            }
        )

    out = pd.DataFrame(rows)
    out = out.sort_values(["season", "celebrity_name"], kind="mergesort").reset_index(drop=True)
    return out


def run(*, tol: float = 1e-6) -> Q0SanityOutputs:
    paths.ensure_dirs()

    df = _read_weekly_panel()

    season_week_report = _season_week_checks(df, tol=tol)
    contestant_report = _contestant_checks(df)

    out1 = paths.tables_dir() / "mcm2026c_q0_sanity_season_week.csv"
    out2 = paths.tables_dir() / "mcm2026c_q0_sanity_contestant.csv"

    io.write_csv(season_week_report, out1)
    io.write_csv(contestant_report, out2)

    return Q0SanityOutputs(season_week_report_csv=out1, contestant_report_csv=out2)


def main() -> int:
    out = run()

    sw = pd.read_csv(out.season_week_report_csv)
    ct = pd.read_csv(out.contestant_report_csv)

    sw_bad = sw[sw["issues"].fillna("") != ""]
    ct_bad = ct[ct["issues"].fillna("") != ""]

    max_err = float(np.nanmax(sw["pct_abs_err"].to_numpy())) if not sw.empty else float("nan")

    print(f"Wrote: {out.season_week_report_csv}")
    print(f"Wrote: {out.contestant_report_csv}")
    print(f"Season-week issues: {len(sw_bad)} / {len(sw)} (max pct_abs_err={max_err:.3e})")
    print(f"Contestant issues: {len(ct_bad)} / {len(ct)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
