# 2026 MCM Problem C (DWTS)
Reproducible modeling pipelines for COMAP MCM/ICM 2026, Problem C: Dancing with the Stars (DWTS) – Q0–Q4 end‑to‑end workflows with single-command regeneration.

## Overview
This repository implements a clean, reproducible workflow to study DWTS elimination mechanisms and propose a robust alternative. The code converts the official wide table into canonical modeling datasets (Q0), infers relative fan voting strength with uncertainty (Q1), compares rule variants via counterfactual simulation (Q2), analyzes feature impacts with mixed/OLS models (Q3), and evaluates new voting mechanisms under multiple metrics and stress tests (Q4).

The project emphasizes determinism and transparency: one command regenerates all main artifacts under outputs/, with configuration centralized in a single YAML file. Optional “showcase” modules (deep learning baselines, sensitivity grids) are available for appendix-only exploration without altering the mainline results.

## Technology Stack
- Language/Runtime: Python 3.11 (see .python-version)
- Package/Env: uv (pyproject.toml + uv.lock)
- Frameworks/Libs: numpy, pandas, scipy, scikit-learn, statsmodels, matplotlib, seaborn, plotly, pyyaml, pyarrow, tqdm
- Optional groups: dl (torch via cu124/cpu), web (requests/bs4/pytrends), notebook (jupyterlab/ipykernel), opt (cvxpy/pulp)
- Quality/Dev: ruff (lint+format), pytest (+pytest-cov), pre-commit (optional)
- Build/Packaging: hatchling (for wheel packaging, packages=src/mcm2026)

## Project Structure
```
.
├─ src/mcm2026/                # Reusable code (installable package)
│  ├─ core/                    # Paths & project conventions (paths.py)
│  ├─ data/                    # Data IO + auditing (io.py, audit.py)
│  ├─ pipelines/               # Main pipelines (Q0–Q4)
│  │  ├─ mcm2026c_q0_build_weekly_panel.py
│  │  ├─ mcm2026c_q1_smc_fan_vote.py
│  │  ├─ mcm2026c_q2_counterfactual_simulation.py
│  │  ├─ mcm2026c_q3_mixed_effects_impacts.py
│  │  ├─ mcm2026c_q4_design_space_eval.py
│  │  └─ showcase/             # Appendix-only pipelines (do not change main artifacts)
│  ├─ config/config.yaml       # Central configuration for Q1–Q4 + showcase
│  └─ __init__.py
│
├─ run_all.py                  # One-click orchestrator: Q0→Q1→Q2→Q3→Q4 (+showcase gating)
├─ mcm2026c/                   # Official problem statement + data CSV
│  ├─ 2026_MCM_Problem_C_Data.csv
│  └─ README.txt (etc.)
│
├─ data/
│  ├─ raw/                     # Read-only inputs + optional external proxies
│  └─ processed/               # Canonical outputs from Q0 (weekly_panel, season_features)
│
├─ outputs/
│  ├─ figures/
│  ├─ predictions/             # Q1 posterior summary
│  └─ tables/                  # Q0 sanity, Q1 uncertainty, Q2/Q3/Q4 tables (+showcase subdir)
│
├─ scripts/                    # Optional data fetchers (external proxies)
│  ├─ fetch_dwts_google_trends.py
│  ├─ fetch_dwts_wikipedia_pageviews.py
│  └─ fetch_us_state_population_2020.py
│
├─ tests/                      # Minimal smoke tests
│  └─ test_smoke.py
│
├─ docs/
│  ├─ project_structure.md     # Chinese overview of layout & artifacts
│  ├─ conventions.md           # Naming & pipeline conventions
│  └─ spec/                    # Task/architecture/target specs
│
├─ paper/                      # Writing assets (Markdown draft + LaTeX)
│  ├─ draft.md (Chinese draft, no images)
│  └─ main.tex + figures/
│
├─ pyproject.toml              # Project metadata, deps (default dev group), uv sources
├─ uv.lock                     # Locked dependency graph for uv
├─ README.md                   # Quickstart & artifact index
└─ (no LICENSE file found)
```

## Key Features
- Reproducible, single-command regeneration of all core outputs
- Canonical datasets (Q0) ensuring consistent downstream use
- Bayesian/importance-sampling style fan-vote inference with uncertainty (Q1)
- Counterfactual simulations comparing Percent vs Rank and Judge Save (Q2)
- Mixed-effects/OLS analysis for interpretable impact estimates (Q3)
- Mechanism design space exploration with multiple metrics and stress tests (Q4)
- Centralized, documented configuration (config.yaml) + optional showcase grids

## Getting Started

### Prerequisites
- Python 3.11
- uv (https://docs.astral.sh/uv/) installed on your system

### Installation
```bash
# Create/sync environment (default dev group)
uv sync
# Or include all optional groups (dl/web/notebook/opt)
uv sync --all-groups
```

### Usage
```bash
# One-click mainline regeneration (Q0→Q4)
uv run python run_all.py

# Include appendix-only showcase (requires enabling in config)
uv run python run_all.py --showcase
```
Main artifacts will be written to outputs/ (see “Main Artifacts” below). Canonical processed datasets (Q0) go to data/processed/.

## Development

### Available Scripts
- Lint: `uv run ruff check .`
- Format: `uv run ruff format .`
- Tests: `uv run pytest`
- End-to-end smoke (mainline): `uv run python run_all.py`
- Optional external data fetchers (appendix/audit only):
  - `uv run python scripts/fetch_dwts_wikipedia_pageviews.py`
  - `uv run python scripts/fetch_dwts_google_trends.py`
  - `uv run python scripts/fetch_us_state_population_2020.py`

### Development Workflow
1. Sync env with uv (optionally all groups if needed)
2. Implement or modify pipelines under src/mcm2026/pipelines/
3. Keep configuration in src/mcm2026/config/config.yaml
4. Run ruff + pytest locally
5. Regenerate outputs: `uv run python run_all.py`
6. Update paper/draft.md and docs/ as needed

## Configuration
Central config: src/mcm2026/config/config.yaml
- dwts.q1 (fan vote inference)
  - alpha: float (judge vs fans weight, default 0.5)
  - tau: float (soft constraint temperature, default 0.03)
  - prior_draws_m: int (Dirichlet prior samples, e.g., 2000)
  - posterior_resample_r: int (resamples for posterior summaries, e.g., 500)
- dwts.q2 (counterfactual comparison)
  - fan_source_mechanism: {percent, rank}
  - count_withdraw_as_exit: bool
- dwts.q3 (impact analysis)
  - fan_source_mechanism: {percent, rank}
  - n_refits: int (propagate Q1 uncertainty)
  - seed: int
- dwts.q4 (design space evaluation)
  - fan_source_mechanism: {percent, rank}
  - n_sims: int, seed: int
  - outlier_mults: [2.0, 5.0, 10.0] (stress tests)
  - Optional grids: alpha_grid, mechanisms, seasons
- showcase (appendix-only; disabled by default)
  - enabled: bool, seed, output_dir
  - q1/q2/q3/q4: sensitivity grids and toggles

## Architecture
High-level dataflow:
- Q0 (mcm2026c_q0_build_weekly_panel):
  - Read official wide CSV → derive weekly panel and season features
  - Compute judge aggregates, ranks/percents, active flags, exit markers
- Q1 (mcm2026c_q1_smc_fan_vote):
  - Infer per-week fan vote share/index with uncertainty under Percent/Rank
  - Write posterior summary (predictions/) and uncertainty (tables/)
- Q2 (mcm2026c_q2_counterfactual_simulation):
  - Compare mechanisms per week/season (incl. Judge Save variant)
- Q3 (mcm2026c_q3_mixed_effects_impacts):
  - MixedLM/OLS on interpretable covariates (season-level join of Q1 + features)
- Q4 (mcm2026c_q4_design_space_eval):
  - Simulate seasons with multiple mechanisms/alphas; output fairness/excitement/robustness metrics, incl. stress tests
- Orchestration: run_all.py wires Q0→Q4, ensures directories, reads optional config, and gates showcase.
- Canonical datasets: data/processed/dwts_weekly_panel.csv, data/processed/dwts_season_features.csv

## Main Artifacts (by run_all.py)
- Q0 sanity: outputs/tables/mcm2026c_q0_sanity_season_week.csv
- Q0 sanity: outputs/tables/mcm2026c_q0_sanity_contestant.csv
- Q1 posterior: outputs/predictions/mcm2026c_q1_fan_vote_posterior_summary.csv
- Q1 uncertainty: outputs/tables/mcm2026c_q1_uncertainty_summary.csv
- Q2 comparison: outputs/tables/mcm2026c_q2_mechanism_comparison.csv
- Q3 impacts: outputs/tables/mcm2026c_q3_impact_analysis_coeffs.csv
- Q4 design metrics: outputs/tables/mcm2026c_q4_new_system_metrics.csv
- Showcase (appendix-only): outputs/tables/showcase/*

## Contributing
- Style: ruff (line length 100; Python target 3.11)
- Tests: add/extend pytest under tests/
- Repro: changes that affect outputs must be re-generated via run_all.py
- Avoid manual edits to outputs/ and data/processed/
- Document meaningful config changes in config.yaml comments and/or docs/

## License
No LICENSE file was found in the repository. If you intend to open-source, add a LICENSE in the project root; otherwise, treat this code as “all rights reserved” within your team context.

---
This AGENTS.md summarizes the architecture and workflows for fast onboarding and reproducible execution. Update it when pipelines or configuration change.
