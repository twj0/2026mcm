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
- Respect the problem statement data policy (external data only if explicitly allowed).

## Showtime (Optional)

- Deep learning track: enable `dl` group (PyTorch wheels via cu124 on non-mac platforms).
- Web scraping: enable `web` group only if the problem statement allows external data.

## Definition of Done (for coding tasks)

- Code can be executed in a clean environment created by `uv sync` (default groups) or `uv sync --all-groups`.
- A single entry script can reproduce key outputs (figures/tables/predictions) deterministically given the same input data.