from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path, na_values=["N/A"], keep_default_na=True)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t", na_values=["N/A"], keep_default_na=True)
    if suffix in {".txt"}:
        return pd.read_csv(path, sep=None, engine="python", na_values=["N/A"], keep_default_na=True)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)

    raise ValueError(f"Unsupported file type: {suffix}")


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
