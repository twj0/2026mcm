---
description: 任务
---

# 2026 MCM Problem C (DWTS) - Global Task Spec

## Goal

- Build a reproducible modeling workflow for DWTS seasons 1–34 to:
  - **Q1** Estimate *relative* fan voting strength per contestant-week (fan vote share / index) consistent with weekly eliminations, with uncertainty.
  - **Q2** Compare Rank vs Percent combining rules across seasons via counterfactual simulation, including controversy cases and the Judge Save (bottom-2 judge decision) variant.
  - **Q3** Analyze impacts of celebrity attributes and pro dancers on judge scores and fan votes (separately), emphasizing interpretability and uncertainty.
  - **Q4** Propose and justify an alternative voting system (fairness / excitement) using metric-based evaluation and simulation.

## Inputs

- **Provided (core)**:
  - `mcm2026c/2026_MCM_Problem_C_Data.csv`
- **Local external data (not used in main pipeline; appendix/audit only)**:
  - `data/raw/dwts_wikipedia_pageviews.csv`
  - `data/raw/dwts_google_trends.csv`
- **Local reference data (allowed, optional)**:
  - `data/raw/us_census_2020_state_population.csv`

## Outputs

- **Processed datasets** (written to `data/processed/`):
  - `dwts_weekly_panel.(csv|parquet)` (season-week-celebrity panel, includes judge totals, ranks/percents, active/elimination flags)
  - `dwts_season_features.(csv|parquet)` (season-celebrity features; optional stable reference joins allowed, but mainline does not depend on external popularity proxies)

- **Main artifacts** (written to `outputs/`):
  - `outputs/predictions/mcm2026c_q1_fan_vote_posterior_summary.csv`
  - `outputs/tables/mcm2026c_q1_uncertainty_summary.csv`
  - `outputs/tables/mcm2026c_q2_mechanism_comparison.csv`
  - `outputs/tables/mcm2026c_q3_impact_analysis_coeffs.csv`
  - `outputs/tables/mcm2026c_q4_new_system_metrics.csv`
  - Optional comparison artifacts may be added using the pattern `mcm2026c_q<k>_<method>_*.csv` (e.g., DL vs baseline), without changing the main artifacts above.
  - Figures (png/pdf) supporting Q1–Q4 in `outputs/figures/`

## Acceptance Criteria

- `run_all.py` can reproduce the core tables/figures deterministically given the same inputs.
- All outputs are written under `outputs/` and `data/processed/` (no manual edits of generated artifacts).
- Methods do **not** claim absolute vote counts; Q1 outputs are fan vote **share/index** with uncertainty.
- Main results do not depend on Google Trends or Wikipedia Pageviews external datasets; any mention is limited to audit/appendix discussion.

