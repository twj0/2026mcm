from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class AuditSummary:
    n_rows: int
    n_cols: int
    n_missing: int
    missing_ratio: float


def audit_dataframe(df: pd.DataFrame) -> AuditSummary:
    n_rows, n_cols = df.shape
    n_missing = int(df.isna().sum().sum())
    denom = max(1, n_rows * n_cols)
    missing_ratio = float(n_missing / denom)
    return AuditSummary(
        n_rows=n_rows,
        n_cols=n_cols,
        n_missing=n_missing,
        missing_ratio=missing_ratio,
    )


def audit_summary_dict(df: pd.DataFrame) -> dict:
    s = audit_dataframe(df)
    return {
        "n_rows": s.n_rows,
        "n_cols": s.n_cols,
        "n_missing": s.n_missing,
        "missing_ratio": s.missing_ratio,
    }


def audit_columns(df: pd.DataFrame) -> pd.DataFrame:
    miss = df.isna().mean().rename("missing_ratio")
    dtypes = df.dtypes.astype(str).rename("dtype")
    out = pd.concat([miss, dtypes], axis=1).reset_index().rename(columns={"index": "column"})
    return out.sort_values(["missing_ratio", "column"], ascending=[False, True], kind="mergesort").reset_index(
        drop=True
    )
