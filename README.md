# Problem C

## Quickstart (Python)

- Python version: `3.11` (see `.python-version`)
- Package manager: `uv`

Recommended workflow:

1. Create / sync environment

    - default (dev): `uv sync`
    - include all optional groups: `uv sync --all-groups`

2. Smoke run

    - `uv run python run_all.py`

3. Run lint / tests (optional but recommended)

    - `uv run ruff check .`
    - `uv run pytest`

## Directory Conventions

- `data/`
  - `raw/`: original attachments from the problem statement (read-only)
  - `processed/`: cleaned datasets generated from `raw/`
- `src/`: reusable code
  - `src/mcm2026/core/`: paths / project conventions
  - `src/mcm2026/data/`: data IO + auditing
  - `src/mcm2026/models/`: baseline ML/DL models
  - `src/mcm2026/validation/`: sanity checks / validation
  - `src/mcm2026/pipelines/`: reproducible pipelines (optional)
- `outputs/`: generated figures/tables/predictions for the paper
- `paper/`: LaTeX paper (template from `examples/MCM-Latex-template`)

Note: `outputs/` is for reproducible generation. Copy/insert finalized figures into `paper/figures/` as needed.

## Optional "Showtime" Dependency Groups

- Deep learning (PyTorch CUDA 12.4): `uv sync --group dl` (installs PyTorch with CUDA 12.4 via `uv sources`)
- Web scraping (only if allowed by data policy): `uv sync --group web`

## AI Workflows

- Use `/mcm` for the end-to-end contest workflow.
- Use `/spec` when you want spec-driven implementation via `docs/spec/*`.
