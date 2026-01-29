from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    issues: list[str]


def validate_series_basic(name: str, s: pd.Series) -> ValidationResult:
    issues: list[str] = []

    if s.isna().any():
        issues.append(f"{name}: contains NaN")

    arr = s.to_numpy()
    if np.isinf(arr).any():
        issues.append(f"{name}: contains inf")

    if s.dtype.kind in {"f", "i"}:
        q01 = float(np.nanquantile(arr, 0.01))
        q99 = float(np.nanquantile(arr, 0.99))
        if abs(q99 - q01) == 0:
            issues.append(f"{name}: near-constant")

    return ValidationResult(ok=len(issues) == 0, issues=issues)


def validate_probability(name: str, s: pd.Series) -> ValidationResult:
    issues: list[str] = []

    r = validate_series_basic(name, s)
    issues.extend(r.issues)

    if ((s < 0) | (s > 1)).any():
        issues.append(f"{name}: outside [0, 1]")

    return ValidationResult(ok=len(issues) == 0, issues=issues)


def validate_non_negative(name: str, s: pd.Series) -> ValidationResult:
    issues: list[str] = []

    r = validate_series_basic(name, s)
    issues.extend(r.issues)

    if (s < 0).any():
        issues.append(f"{name}: contains negative values")

    return ValidationResult(ok=len(issues) == 0, issues=issues)
