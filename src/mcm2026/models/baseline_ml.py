from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass(frozen=True)
class FitResult:
    model: object
    metrics: dict


def _split_columns(df: pd.DataFrame, features: list[str]) -> tuple[list[str], list[str]]:
    X = df[features]
    numeric = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical = [c for c in X.columns if c not in numeric]
    return numeric, categorical


def fit_regression(df: pd.DataFrame, target: str, features: list[str]) -> FitResult:
    y = df[target].to_numpy()
    numeric, categorical = _split_columns(df, features)

    pre = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("ohe", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )

    model = Pipeline(steps=[("pre", pre), ("est", Ridge(alpha=1.0))])
    model.fit(df[features], y)
    pred = model.predict(df[features])

    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "mae": float(mean_absolute_error(y, pred)),
        "r2": float(r2_score(y, pred)),
    }
    return FitResult(model=model, metrics=metrics)


def fit_classification(df: pd.DataFrame, target: str, features: list[str]) -> FitResult:
    y = df[target].to_numpy()
    numeric, categorical = _split_columns(df, features)

    pre = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("ohe", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )

    model = Pipeline(
        steps=[
            ("pre", pre),
            ("est", LogisticRegression(max_iter=2000)),
        ]
    )
    model.fit(df[features], y)
    pred = model.predict(df[features])

    metrics = {
        "accuracy": float(accuracy_score(y, pred)),
    }
    return FitResult(model=model, metrics=metrics)
