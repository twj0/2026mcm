---
# /docs/spec/target.md
description: 目标
---

# Target / Constraints

## Tech Stack

- **OS**: Windows
- **Python**: 3.11 (see `.python-version`)
- **Package manager**: `uv` (`pyproject.toml` + `uv.lock`)
- **Primary libs**: numpy / pandas / scipy / scikit-learn / statsmodels
- **Plotting**: matplotlib / seaborn / plotly
- **Quality**: ruff (lint + format), pytest

## Reproducibility Rules

- Any figure/table used in the paper must be reproducible by running `run_all.py`.
- Outputs must be written to `outputs/` (generated artifacts should not be edited by hand).
- Respect the problem statement data policy (external data is allowed but must be fully documented).

## Data Policy (DWTS)

- The problem statement permits adding additional information/data, but all sources must be documented.
- External data is treated as **optional** signals for priors/features/explanations, not a replacement for rule-based constraints.
- External datasets may be present in `data/raw/`, but they are **not required** for the main solution.
- Audit conclusion (current repository state):
  - `data/raw/dwts_google_trends.csv` is excluded from main analysis (coverage limited to seasons 1–20; unstable; values not reliably comparable across contestants).
  - `data/raw/dwts_wikipedia_pageviews.csv` is excluded from main analysis (currently only covers season 27; insufficient for cross-season analysis).

## Modeling Constraints (DWTS)

- Do not claim absolute vote totals; outputs should be fan vote **share/index** with uncertainty.
- Avoid decision-time leakage: when modeling a week `t`, do not use information that would only be known after week `t` (unless explicitly flagged as post-hoc analysis).

## Showtime (Optional)

- Deep learning track: enable `dl` group (PyTorch wheels via cu124 on non-mac platforms).
- Web scraping: enable `web` group only if the problem statement allows external data.

## Definition of Done (for coding tasks)

- Code can be executed in a clean environment created by `uv sync` (default groups) or `uv sync --all-groups`.
- A single entry script can reproduce key outputs (figures/tables/predictions) deterministically given the same input data.