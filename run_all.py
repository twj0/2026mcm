from __future__ import annotations

# Ref: docs/spec/task.md
# Ref: docs/spec/architecture.md

import sys
from pathlib import Path

import pandas as pd
import yaml


def _bootstrap_imports() -> None:
    repo_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(repo_root / "src"))


def _list_raw_files(raw_dir: Path) -> list[Path]:
    if not raw_dir.exists():
        return []

    exts = {".csv", ".tsv", ".txt", ".xlsx", ".xls"}
    return sorted([p for p in raw_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])


def main() -> int:
    run_showcase = "--showcase" in sys.argv

    repo_root = Path(__file__).resolve().parent
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

    print("Running Q0 (build weekly panel)...")
    q0_out = run_q0()
    print(f"Built processed dataset: {q0_out.weekly_panel_csv}")
    print(f"Built processed dataset: {q0_out.season_features_csv}")

    print("Running Q1 (fan vote inference)...")
    q1_out = run_q1()
    print(f"Wrote: {q1_out.posterior_summary_csv}")
    print(f"Wrote: {q1_out.uncertainty_summary_csv}")

    print("Running Q2 (counterfactual mechanism simulation)...")
    q2_out = run_q2()
    print(f"Wrote: {q2_out.mechanism_comparison_csv}")

    print("Running Q3 (mixed effects impacts)...")
    q3_out = run_q3()
    print(f"Wrote: {q3_out.impact_coeffs_csv}")

    print("Running Q4 (new system design space eval)...")
    q4_out = run_q4()
    print(f"Wrote: {q4_out.new_system_metrics_csv}")

    if run_showcase:
        cfg = {}
        cfg_path = repo_root / "src" / "mcm2026" / "config" / "config.yaml"
        if cfg_path.exists():
            text = cfg_path.read_text(encoding="utf-8")
            if text.strip():
                loaded = yaml.safe_load(text)
                cfg = loaded if isinstance(loaded, dict) else {}

        sc = cfg.get("showcase", {}) if isinstance(cfg, dict) else {}
        sc_enabled = bool(sc.get("enabled", False))
        if not sc_enabled:
            print("Showcase disabled in config (showcase.enabled=false).")
            return 0

        sc_seed = int(sc.get("seed", 20260130))
        sc_out_dir = Path(str(sc.get("output_dir", "outputs/tables/showcase")))
        if not sc_out_dir.is_absolute():
            sc_out_dir = repo_root / sc_out_dir

        q1_sc = sc.get("q1", {}) if isinstance(sc, dict) else {}
        q2_sc = sc.get("q2", {}) if isinstance(sc, dict) else {}
        q3_sc = sc.get("q3", {}) if isinstance(sc, dict) else {}
        q4_sc = sc.get("q4", {}) if isinstance(sc, dict) else {}

        print("Running showcase pipelines (appendix-only)...")

        if bool(q1_sc.get("enabled", False)):
            from mcm2026.pipelines.showcase.mcm2026c_q1_ml_elimination_baselines import run as run_sc_q1
            from mcm2026.pipelines.showcase.mcm2026c_showcase_q1_sensitivity import run as run_sc_q1_sens

            sc_q1_out = run_sc_q1(seed=sc_seed, output_dir=sc_out_dir)
            print(f"Wrote: {sc_q1_out.cv_metrics_csv}")
            print(f"Wrote: {sc_q1_out.cv_summary_csv}")

            sc_q1_sens_out = run_sc_q1_sens(
                seed=sc_seed,
                output_dir=sc_out_dir,
                alpha_grid=q1_sc.get("alpha_grid", None),
                tau_grid=q1_sc.get("tau_grid", None),
                prior_draws_m_grid=q1_sc.get("prior_draws_m_grid", None),
                posterior_resample_r_grid=q1_sc.get("posterior_resample_r_grid", None),
                max_runs=q1_sc.get("max_runs", None),
            )
            print(f"Wrote: {sc_q1_sens_out.sensitivity_summary_csv}")

        if bool(q2_sc.get("enabled", False)):
            from mcm2026.pipelines.showcase.mcm2026c_showcase_q2_grid import run as run_sc_q2

            sc_q2_out = run_sc_q2(
                seed=sc_seed,
                output_dir=sc_out_dir,
                fan_source_mechanism_grid=q2_sc.get("fan_source_mechanism_grid", None),
                count_withdraw_as_exit_grid=q2_sc.get("count_withdraw_as_exit_grid", None),
            )
            print(f"Wrote: {sc_q2_out.grid_csv}")

        if bool(q3_sc.get("enabled", False)):
            from mcm2026.pipelines.showcase.mcm2026c_q3_ml_fan_index_baselines import run as run_sc_q3
            from mcm2026.pipelines.showcase.mcm2026c_showcase_q3_refit_grid import run as run_sc_q3_refits

            sc_q3_out = run_sc_q3(seed=sc_seed, output_dir=sc_out_dir)
            print(f"Wrote: {sc_q3_out.cv_metrics_csv}")
            print(f"Wrote: {sc_q3_out.cv_summary_csv}")

            sc_q3_ref_out = run_sc_q3_refits(
                seed=sc_seed,
                output_dir=sc_out_dir,
                n_refits_grid=q3_sc.get("n_refits_grid", None),
                max_runs=q3_sc.get("max_runs", None),
            )
            print(f"Wrote: {sc_q3_ref_out.grid_csv}")

        if bool(q4_sc.get("enabled", False)):
            from mcm2026.pipelines.showcase.mcm2026c_showcase_q4_sensitivity import run as run_sc_q4

            sc_q4_out = run_sc_q4(
                seed=sc_seed,
                output_dir=sc_out_dir,
                alpha_grid=q4_sc.get("alpha_grid", None),
                n_sims_grid=q4_sc.get("n_sims_grid", None),
                outlier_mults_grid=q4_sc.get("outlier_mults_grid", None),
                mechanisms=q4_sc.get("mechanisms", None),
                seasons=q4_sc.get("seasons", None),
                max_runs=q4_sc.get("max_runs", None),
            )
            print(f"Wrote: {sc_q4_out.grid_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
