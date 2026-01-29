from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd


@dataclass(frozen=True)
class AuditSummary:
    rows: int
    cols: int
    missing_cells: int
    missing_ratio: float


def audit_dataframe(df: pd.DataFrame) -> AuditSummary:
    rows, cols = df.shape
    missing_cells = int(df.isna().sum().sum())
    total_cells = max(rows * cols, 1)
    return AuditSummary(
        rows=rows,
        cols=cols,
        missing_cells=missing_cells,
        missing_ratio=missing_cells / total_cells,
    )


def audit_columns(df: pd.DataFrame) -> pd.DataFrame:
    missing = df.isna().mean().rename("missing_ratio")
    dtype = df.dtypes.astype(str).rename("dtype")
    out = pd.concat([dtype, missing], axis=1).reset_index(names="column")
    return out.sort_values(["missing_ratio", "column"], ascending=[False, True])


def audit_summary_dict(df: pd.DataFrame) -> dict:
    return asdict(audit_dataframe(df))
