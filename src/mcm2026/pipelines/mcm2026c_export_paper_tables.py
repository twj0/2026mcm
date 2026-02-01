from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from mcm2026.core import paths
from mcm2026.data import io


@dataclass(frozen=True)
class PaperTableOutputs:
    q1_high_uncertainty_weeks_csv: Path
    q1_error_cases_csv: Path
    q2_divergence_summary_csv: Path
    q4_recommendation_summary_csv: Path


def _export_q1_high_uncertainty_weeks(*, top_k: int = 12) -> pd.DataFrame:
    src = io.read_table(paths.tables_dir() / "mcm2026c_q1_uncertainty_summary.csv")

    df = src.loc[(src["mechanism"].astype(str) == "percent") & (src["n_exit"].astype(int) >= 2)].copy()
    df = df.sort_values(["ess_ratio", "evidence", "season", "week"], ascending=[True, True, True, True])

    df = df.head(int(top_k))
    df = df.rename(columns={"n_active": "nactive", "n_exit": "nexit", "ess_ratio": "essratio"})
    return df[["season", "week", "nactive", "nexit", "essratio", "evidence"]].reset_index(drop=True)


def _export_q1_error_cases(*, top_k: int = 12) -> pd.DataFrame:
    src = io.read_table(paths.tables_dir() / "mcm2026c_q1_error_diagnostics_week.csv")

    df = src.loc[(src["mechanism"].astype(str) == "percent") & (src["n_exit"].astype(int) >= 2)].copy()
    df = df.sort_values(
        ["observed_exit_prob_at_posterior_mean", "judge_fan_rank_corr", "season", "week"],
        ascending=[True, True, True, True],
    )

    df = df.head(int(top_k))
    df = df.rename(
        columns={
            "n_active": "nactive",
            "judge_fan_rank_corr": "corr",
            "observed_exit": "observedexit",
            "observed_exit_prob_at_posterior_mean": "pexit",
        }
    )
    return df[["season", "week", "nactive", "corr", "observedexit", "pexit"]].reset_index(drop=True)


def _export_q2_divergence_summary(*, top_k: int = 12) -> pd.DataFrame:
    src = io.read_table(paths.tables_dir() / "mcm2026c_q2_mechanism_comparison.csv")

    df = src.loc[src["fan_source_mechanism"].astype(str) == "percent"].copy()
    df = df.sort_values(["diff_weeks_percent_vs_rank", "match_rate_rank", "season"], ascending=[False, True, True])

    df = df.head(int(top_k))
    df = df.rename(
        columns={
            "n_weeks": "nweeks",
            "diff_weeks_percent_vs_rank": "diffweeks",
            "match_rate_percent": "matchpercent",
            "match_rate_rank": "matchrank",
            "match_rate_percent_judge_save": "matchpercentjs",
        }
    )
    return df[["season", "nweeks", "diffweeks", "matchpercent", "matchrank", "matchpercentjs"]].reset_index(
        drop=True
    )


def _export_q4_recommendation_summary() -> pd.DataFrame:
    src = io.read_table(paths.tables_dir() / "mcm2026c_q4_sensitivity_summary.csv")

    df = src.loc[
        (src["fan_source_mechanism"].astype(str) == "percent")
        & (src["mechanism"].astype(str) == "rank")
        & (src["alpha"].astype(float) == 0.5)
        & (src["sigma_scale"].astype(float) == 2.0)
    ].copy()

    df = df.sort_values(["outlier_mult"], ascending=[True])
    df = df.rename(columns={"outlier_mult": "outlier_mult"})

    return df[
        [
            "outlier_mult",
            "mechanism",
            "tpi_mean",
            "tpi_ci95_low",
            "tpi_ci95_high",
            "fan_mean",
            "fan_ci95_low",
            "fan_ci95_high",
            "robust_fail_mean",
            "robust_fail_ci95_low",
            "robust_fail_ci95_high",
        ]
    ].reset_index(drop=True)


def run(*, top_k: int = 12) -> PaperTableOutputs:
    out_dir = paths.tables_dir()

    q1_hi = _export_q1_high_uncertainty_weeks(top_k=top_k)
    q1_err = _export_q1_error_cases(top_k=top_k)
    q2_div = _export_q2_divergence_summary(top_k=top_k)
    q4_rec = _export_q4_recommendation_summary()

    fp_q1_hi = out_dir / "paper_q1_high_uncertainty_weeks.csv"
    fp_q1_err = out_dir / "paper_q1_error_cases.csv"
    fp_q2_div = out_dir / "paper_q2_divergence_summary.csv"
    fp_q4_rec = out_dir / "paper_q4_recommendation_summary.csv"

    io.write_csv(q1_hi, fp_q1_hi)
    io.write_csv(q1_err, fp_q1_err)
    io.write_csv(q2_div, fp_q2_div)
    io.write_csv(q4_rec, fp_q4_rec)

    return PaperTableOutputs(
        q1_high_uncertainty_weeks_csv=fp_q1_hi,
        q1_error_cases_csv=fp_q1_err,
        q2_divergence_summary_csv=fp_q2_div,
        q4_recommendation_summary_csv=fp_q4_rec,
    )


def main() -> int:
    paths.ensure_dirs()
    out = run()
    print(f"Wrote: {out.q1_high_uncertainty_weeks_csv}")
    print(f"Wrote: {out.q1_error_cases_csv}")
    print(f"Wrote: {out.q2_divergence_summary_csv}")
    print(f"Wrote: {out.q4_recommendation_summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
