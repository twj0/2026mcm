# Conventions

## Naming

- `archive-contest/mcmYYYYc/`
  - `statement.md`: original Problem C statement (markdown)
  - `data/`: provided datasets for that year
  - `notes_*.md`: optional notes/context

- `src/mcm2026/pipelines/`
  - One file per sub-question / deliverable.
  - Filename format: `mcmYYYYc_q<k>_<verb>_<object>.py`
  - If multiple methods are implemented for the same question, include a **method tag** in the filename (after `q<k>`), and treat non-mainline methods as comparison/appendix pipelines:
    - examples:
      - `mcm2026c_q1_smc_fan_vote.py`
      - `mcm2026c_q1_rank_plackett_luce_mcmc.py`
      - `mcm2026c_q1_dl_elimination_transformer.py`
      - `mcm2026c_q3_mixed_effects_impacts.py`
      - `mcm2026c_q3_dl_fanvote_mlp.py`
      - `mcm2026c_q4_multiobjective_pareto_search.py`
    - examples:
      - `mcm2023c_q1_predict_medals.py`
      - `mcm2024c_q2_optimize_teams.py`
      - `mcm2025c_q3_events_impact.py`
      - `mcm2026c_q4_*.py`
      

- `src/mcm2026/models/`
  - Reusable model implementations (not per-question).
  - Prefer naming by model family: `baseline_ml.py`, `baseline_dl.py`, `poisson.py`, `quantile.py`, etc.

- `outputs/`
  - `tables/`: `mcmYYYYc_q<k>_*.csv`
  - `figures/`: `mcmYYYYc_q<k>_*.png`
  - `predictions/`: `mcmYYYYc_q<k>_*.csv`
  - If multiple methods exist, include the same method tag in artifact names:
    - `mcm2026c_q1_smc_*.csv`, `mcm2026c_q1_transformer_*.csv`, `mcm2026c_q3_mixed_effects_*.csv`, `mcm2026c_q4_pareto_*.png`

## Pipeline structure

Each pipeline module should be runnable and reproducible.

- `load_*()`: read inputs
- `build_features_*()`: feature engineering / cleaning
- `fit_*()`: call `mcm2026.models.*`
- `evaluate_*()`: metrics + uncertainty
- `write_outputs_*()`: write to `outputs/`
- `main()` / `run()`: orchestrate
