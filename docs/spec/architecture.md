---
description: 系统架构
---

# Architecture

## Directory Layout

The repository uses a simple, reproducible structure:

- `data/raw/`: original problem attachments (read-only)
- `data/processed/`: cleaned datasets generated from `raw/`
- `src/mcm2026/`: main reusable code area (structured subpackages: `core/`, `data/`, `models/`, `validation/`, `pipelines/`)
- `run_all.py`: preferred entry point to regenerate the main outputs used by the paper
- `outputs/`: generated artifacts
  - `figures/`: png/pdf
  - `tables/`: csv/tex
  - `predictions/`: csv/parquet
- `paper/`: LaTeX paper

## Design Rules (KISS)

- Prefer small pure functions over heavy abstraction layers.
- Keep I/O (reading/writing) in `run_all.py` and a small set of helpers.
- Ensure no data leakage for time series / “decision at time t” problems.

## DWTS (2026C) Pipelines

DWTS work should live under `src/mcm2026/pipelines/` following the naming convention:

- `mcm2026c_q0_build_weekly_panel.py`
  - Build the canonical season-week-celebrity panel from `mcm2026c/2026_MCM_Problem_C_Data.csv`.
  - Derive judge totals, judge percent, judge ranks, active flags, elimination indicators.

- `mcm2026c_q1_*` (multi-method, one mainline + optional comparisons)
  - Mainline should estimate fan vote **share/index** per contestant-week under Percent and Rank mechanisms.
  - Optional comparison/appendix pipelines may include alternative inference and DL baselines, but must not change main results.

- `mcm2026c_q2_*`
  - Counterfactual simulation: rerun seasons under alternate rule (Rank vs Percent), and Judge Save variant.
  - Produce season-level and case-level comparison tables/figures.

- `mcm2026c_q3_*`
  - Impact analysis: celebrity attributes + pro dancer effects on judge scores and estimated fan vote indices.
  - Prefer interpretable models; mixed-effects models are acceptable.

- `mcm2026c_q4_*`
  - Define metrics (fairness/excitement/robustness) and evaluate proposed new system via simulation.

## Canonical Datasets (Single Source of Truth)

To keep the analysis consistent across questions, produce and reuse:

- `data/processed/dwts_weekly_panel.(csv|parquet)`
  - Grain: `season, week, celebrity_name`
  - Contains judge score aggregates + rule-specific intermediates.

- `data/processed/dwts_season_features.(csv|parquet)`
  - Grain: `season, celebrity_name`
  - Static attributes and optional stable reference joins (e.g., state population).
  - Popularity proxies from Google Trends / Wikipedia Pageviews are excluded from main analysis in the current repository state.

## Output Artifacts

- All generated artifacts must be written under `outputs/`.
- Naming should follow `mcm2026c_q<k>_*` for tables/figures/predictions.
- Outputs used by the paper should be derivable by running `run_all.py`.

