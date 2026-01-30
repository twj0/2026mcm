from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass(frozen=True)
class FitResult:
    pipeline: Pipeline
    metrics: dict[str, float]


def _infer_columns(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    return numeric_cols, categorical_cols


def _make_preprocessor(
    X: pd.DataFrame,
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> ColumnTransformer:
    if numeric_features is None or categorical_features is None:
        num_cols, cat_cols = _infer_columns(X)
        numeric_features = num_cols if numeric_features is None else numeric_features
        categorical_features = cat_cols if categorical_features is None else categorical_features

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, list(numeric_features)),
            ("cat", categorical_pipe, list(categorical_features)),
        ],
        remainder="drop",
    )


def fit_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    X_valid: pd.DataFrame | None = None,
    y_valid: pd.Series | np.ndarray | None = None,
    *,
    alpha: float = 1.0,
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
    random_state: int = 0,
) -> FitResult:
    pre = _make_preprocessor(X_train, numeric_features=numeric_features, categorical_features=categorical_features)
    model = Ridge(alpha=alpha, random_state=random_state)

    pipe = Pipeline(steps=[("preprocess", pre), ("model", model)])
    pipe.fit(X_train, y_train)

    metrics: dict[str, float] = {}
    if X_valid is not None and y_valid is not None:
        pred = pipe.predict(X_valid)
        metrics["rmse"] = float(mean_squared_error(y_valid, pred, squared=False))
        metrics["r2"] = float(r2_score(y_valid, pred))

    return FitResult(pipeline=pipe, metrics=metrics)


def fit_classification(
    X_train: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    X_valid: pd.DataFrame | None = None,
    y_valid: pd.Series | np.ndarray | None = None,
    *,
    C: float = 1.0,
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
    max_iter: int = 2000,
) -> FitResult:
    pre = _make_preprocessor(X_train, numeric_features=numeric_features, categorical_features=categorical_features)
    model = LogisticRegression(C=C, max_iter=max_iter)

    pipe = Pipeline(steps=[("preprocess", pre), ("model", model)])
    pipe.fit(X_train, y_train)

    metrics: dict[str, float] = {}
    if X_valid is not None and y_valid is not None:
        pred = pipe.predict(X_valid)
        metrics["accuracy"] = float(accuracy_score(y_valid, pred))

        try:
            proba = pipe.predict_proba(X_valid)[:, 1]
            metrics["roc_auc"] = float(roc_auc_score(y_valid, proba))
        except Exception:
            pass

    return FitResult(pipeline=pipe, metrics=metrics)