from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from mcm2026.core import paths
from mcm2026.data import io
from mcm2026.pipelines import mcm2026c_q3_mixed_effects_impacts as q3_main


@dataclass(frozen=True)
class Q3BaselineOutputs:
    cv_metrics_csv: Path
    cv_summary_csv: Path


def _make_preprocessor(*, numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
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


def _fit_and_score(
    *,
    train: pd.DataFrame,
    test: pd.DataFrame,
    y_col: str,
    numeric_features: list[str],
    categorical_features: list[str],
    model_kind: str,
    random_state: int,
) -> dict:
    X_train = train[numeric_features + categorical_features]
    y_train = train[y_col].astype(float)

    X_test = test[numeric_features + categorical_features]
    y_test = test[y_col].astype(float)

    pre = _make_preprocessor(numeric_features=numeric_features, categorical_features=categorical_features)

    if model_kind == "ridge":
        model = Ridge(alpha=1.0, random_state=int(random_state))
    elif model_kind == "mlp":
        model = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            alpha=1e-4,
            max_iter=1500,
            early_stopping=True,
            random_state=int(random_state),
        )
    else:
        raise ValueError(f"Unknown model_kind: {model_kind}")

    pipe = Pipeline(steps=[("preprocess", pre), ("model", model)])
    pipe.fit(X_train, y_train)

    pred = pipe.predict(X_test)

    rmse = float(np.sqrt(mean_squared_error(y_test, pred))) if len(y_test) else float("nan")
    r2 = float(r2_score(y_test, pred)) if len(y_test) else float("nan")

    return {
        "rmse": rmse,
        "r2": r2,
    }


def run(
    *,
    seed: int = 20260130,
    fan_source_mechanism: str | None = None,
    max_test_seasons: int | None = None,
) -> Q3BaselineOutputs:
    paths.ensure_dirs()

    mech_cfg, _, _ = q3_main._get_q3_params_from_config()
    fan_source_mechanism = mech_cfg if fan_source_mechanism is None else str(fan_source_mechanism)
    if fan_source_mechanism not in {"percent", "rank"}:
        fan_source_mechanism = "percent"

    weekly = q3_main._read_weekly_panel()
    season_features = q3_main._read_season_features()
    q1_post = q3_main._read_q1_posterior_summary()

    df = q3_main._build_season_level_dataset(
        weekly,
        season_features,
        q1_post,
        fan_source_mechanism=fan_source_mechanism,
    )

    df = df.copy()
    df["season"] = df["season"].astype(int)

    y_col = "fan_vote_index_mean"

    numeric_features = [
        "age",
        "age_sq",
        "is_us",
        "log_state_pop",
        "n_weeks_active",
        "n_weeks_q1",
    ]
    categorical_features = [
        "industry",
        "pro_name",
    ]

    seasons = sorted(df["season"].unique().tolist())
    if max_test_seasons is not None:
        seasons = seasons[: int(max_test_seasons)]

    rows: list[dict] = []
    for season_test in seasons:
        train = df.loc[df["season"] != int(season_test)].copy()
        test = df.loc[df["season"] == int(season_test)].copy()

        for model_kind in ["ridge", "mlp"]:
            metrics = _fit_and_score(
                train=train,
                test=test,
                y_col=y_col,
                numeric_features=numeric_features,
                categorical_features=categorical_features,
                model_kind=model_kind,
                random_state=int(seed),
            )

            rows.append(
                {
                    "season_test": int(season_test),
                    "model": model_kind,
                    "fan_source_mechanism": str(fan_source_mechanism),
                    "n_train": int(len(train)),
                    "n_test": int(len(test)),
                    **metrics,
                }
            )

    out = pd.DataFrame(rows)

    out_dir = paths.tables_dir() / "showcase"
    out_fp = out_dir / "mcm2026c_q3_ml_fan_index_baselines_cv.csv"
    io.write_csv(out, out_fp)

    metric_cols = [
        "rmse",
        "r2",
    ]
    summary = (
        out.groupby(["fan_source_mechanism", "model"], sort=True)
        .agg(
            n_folds=("season_test", "nunique"),
            n_test_total=("n_test", "sum"),
            **{f"{c}_mean": (c, "mean") for c in metric_cols},
            **{f"{c}_std": (c, "std") for c in metric_cols},
        )
        .reset_index()
    )
    summary_fp = out_dir / "mcm2026c_q3_ml_fan_index_baselines_cv_summary.csv"
    io.write_csv(summary, summary_fp)

    return Q3BaselineOutputs(cv_metrics_csv=out_fp, cv_summary_csv=summary_fp)


def main() -> int:
    out = run()
    print(f"Wrote: {out.cv_metrics_csv}")
    print(f"Wrote: {out.cv_summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
