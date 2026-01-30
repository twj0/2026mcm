from __future__ import annotations

# Ref: docs/spec/task.md
# Ref: docs/spec/architecture.md

import sys
from pathlib import Path

import pandas as pd


def _bootstrap_imports() -> None:
    repo_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(repo_root / "src"))


def _list_raw_files(raw_dir: Path) -> list[Path]:
    if not raw_dir.exists():
        return []

    exts = {".csv", ".tsv", ".txt", ".xlsx", ".xls"}
    return sorted([p for p in raw_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])


def main() -> int:
    _bootstrap_imports()

    from mcm2026.core import paths
    from mcm2026.data import audit, io
    from mcm2026.pipelines.mcm2026c_q0_build_weekly_panel import run as run_q0
    from mcm2026.pipelines.mcm2026c_q1_smc_fan_vote import run as run_q1
    from mcm2026.pipelines.mcm2026c_q2_counterfactual_simulation import run as run_q2
    from mcm2026.pipelines.mcm2026c_q3_mixed_effects_impacts import run as run_q3
    from mcm2026.pipelines.mcm2026c_q4_design_space_eval import run as run_q4

    paths.ensure_dirs()
    paths.raw_data_dir().mkdir(parents=True, exist_ok=True)

    raw_files = _list_raw_files(paths.raw_data_dir())
    if not raw_files:
        print("No files found under data/raw/. Continuing with official DWTS preprocessing.")

    summary_rows: list[dict] = []
    for fp in raw_files:
        try:
            df = io.read_table(fp)
        except Exception as e:
            summary_rows.append({"file": fp.name, "error": str(e)})
            continue

        s = audit.audit_summary_dict(df)
        summary_rows.append({"file": fp.name, **s})

        col_audit = audit.audit_columns(df)
        io.write_csv(col_audit, paths.tables_dir() / f"raw_{fp.stem}_columns.csv")

    summary = pd.DataFrame(summary_rows)
    io.write_csv(summary, paths.tables_dir() / "raw_audit_summary.csv")

    if raw_files:
        print(f"Audited {len(raw_files)} raw file(s). See outputs/tables/raw_audit_summary.csv")

    q0_out = run_q0()
    print(f"Built processed dataset: {q0_out.weekly_panel_csv}")
    print(f"Built processed dataset: {q0_out.season_features_csv}")

    q1_out = run_q1()
    print(f"Wrote: {q1_out.posterior_summary_csv}")
    print(f"Wrote: {q1_out.uncertainty_summary_csv}")

    q2_out = run_q2()
    print(f"Wrote: {q2_out.mechanism_comparison_csv}")

    q3_out = run_q3()
    print(f"Wrote: {q3_out.impact_coeffs_csv}")

    q4_out = run_q4()
    print(f"Wrote: {q4_out.new_system_metrics_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
