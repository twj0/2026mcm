from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def data_dir() -> Path:
    return repo_root() / "data"


def raw_data_dir() -> Path:
    return data_dir() / "raw"


def processed_data_dir() -> Path:
    return data_dir() / "processed"


def outputs_dir() -> Path:
    return repo_root() / "outputs"


def figures_dir() -> Path:
    return outputs_dir() / "figures"


def tables_dir() -> Path:
    return outputs_dir() / "tables"


def predictions_dir() -> Path:
    return outputs_dir() / "predictions"


def ensure_dirs() -> None:
    for p in [processed_data_dir(), figures_dir(), tables_dir(), predictions_dir()]:
        p.mkdir(parents=True, exist_ok=True)
