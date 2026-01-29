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

