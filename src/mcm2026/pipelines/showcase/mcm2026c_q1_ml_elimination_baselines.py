from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from mcm2026.core import paths
from mcm2026.data import io


@dataclass(frozen=True)
class Q1BaselineOutputs:
    cv_metrics_csv: Path
    cv_summary_csv: Path


def _read_weekly_panel() -> pd.DataFrame:
    return io.read_table(paths.processed_data_dir() / "dwts_weekly_panel.csv")


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["season"] = df["season"].astype(int)
    df["week"] = df["week"].astype(int)

    df["judge_score_pct"] = pd.to_numeric(df["judge_score_pct"], errors="coerce").fillna(0.0)
    df["judge_score_total"] = pd.to_numeric(df["judge_score_total"], errors="coerce").fillna(0.0)
    df["judge_rank"] = pd.to_numeric(df["judge_rank"], errors="coerce").fillna(0.0)
    df["n_active"] = pd.to_numeric(df["n_active"], errors="coerce").fillna(0.0)
    df["season_week_judge_total"] = pd.to_numeric(df["season_week_judge_total"], errors="coerce").fillna(0.0)

    df = df.sort_values(["season", "celebrity_name", "week"], kind="mergesort")

    g = df.groupby(["season", "celebrity_name"], sort=False)

    df["weeks_seen_prev"] = g.cumcount().astype(int)

    def _cummean_prev(s: pd.Series) -> pd.Series:
        return s.shift(1).expanding().mean()

    def _cumstd_prev(s: pd.Series) -> pd.Series:
        return s.shift(1).expanding().std()

    df["judge_pct_cummean_prev"] = g["judge_score_pct"].apply(_cummean_prev).reset_index(level=[0, 1], drop=True)
    df["judge_pct_cumstd_prev"] = g["judge_score_pct"].apply(_cumstd_prev).reset_index(level=[0, 1], drop=True)

    df["judge_pct_cummean_prev"] = df["judge_pct_cummean_prev"].fillna(df["judge_score_pct"])
    df["judge_pct_cumstd_prev"] = df["judge_pct_cumstd_prev"].fillna(0.0)

    w = df.groupby(["season", "week"], sort=False)
    mu = w["judge_score_pct"].transform("mean")
    sd = w["judge_score_pct"].transform("std").fillna(0.0)
    df["judge_pct_z"] = np.where(sd > 0, (df["judge_score_pct"] - mu) / sd, 0.0)

    df["judge_rank_norm"] = np.where(df["n_active"] > 0, df["judge_rank"] / df["n_active"], 0.0)
    df["judge_pct_delta_prevmean"] = df["judge_score_pct"] - df["judge_pct_cummean_prev"]

    df["pro_name"] = df["pro_name"].astype(str)

    return df


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
    y_train = train[y_col].astype(int)

    X_test = test[numeric_features + categorical_features]
    y_test = test[y_col].astype(int)

    pre = _make_preprocessor(numeric_features=numeric_features, categorical_features=categorical_features)

    if model_kind == "logreg":
        model = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs",
        )
    elif model_kind == "mlp":
        model = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            alpha=1e-4,
            max_iter=1000,
            early_stopping=True,
            random_state=int(random_state),
        )
    else:
        raise ValueError(f"Unknown model_kind: {model_kind}")

    pipe = Pipeline(steps=[("preprocess", pre), ("model", model)])
    pipe.fit(X_train, y_train)

    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    prevalence = float(y_test.mean()) if len(y_test) else float("nan")

    roc_auc = float("nan")
    avg_prec = float("nan")
    if y_test.nunique(dropna=True) >= 2:
        roc_auc = float(roc_auc_score(y_test, proba))
        avg_prec = float(average_precision_score(y_test, proba))

    out = {
        "roc_auc": roc_auc,
        "average_precision": avg_prec,
        "accuracy": float(accuracy_score(y_test, pred)) if len(y_test) else float("nan"),
        "brier": float(brier_score_loss(y_test, proba)) if len(y_test) else float("nan"),
        "log_loss": float(log_loss(y_test, proba, labels=[0, 1])) if len(y_test) else float("nan"),
        "prevalence": prevalence,
    }

    return out


def run(*, seed: int = 20260130, max_test_seasons: int | None = None) -> Q1BaselineOutputs:
    paths.ensure_dirs()

    weekly = _read_weekly_panel()

    df = weekly.loc[weekly["active_flag"].astype(bool)].copy()
    df = df.loc[~df["withdrew_this_week"].astype(bool)].copy()

    df["y_eliminated"] = df["eliminated_this_week"].astype(bool).astype(int)

    df = _build_features(df)

    numeric_features = [
        "week",
        "judge_score_total",
        "judge_score_pct",
        "judge_pct_z",
        "judge_rank",
        "judge_rank_norm",
        "n_active",
        "season_week_judge_total",
        "weeks_seen_prev",
        "judge_pct_cummean_prev",
        "judge_pct_cumstd_prev",
        "judge_pct_delta_prevmean",
    ]
    categorical_features = [
        "pro_name",
    ]

    seasons = sorted(df["season"].unique().tolist())
    if max_test_seasons is not None:
        seasons = seasons[: int(max_test_seasons)]

    rows: list[dict] = []
    for season_test in seasons:
        train = df.loc[df["season"] != int(season_test)].copy()
        test = df.loc[df["season"] == int(season_test)].copy()

        for model_kind in ["logreg", "mlp"]:
            metrics = _fit_and_score(
                train=train,
                test=test,
                y_col="y_eliminated",
                numeric_features=numeric_features,
                categorical_features=categorical_features,
                model_kind=model_kind,
                random_state=int(seed),
            )

            rows.append(
                {
                    "season_test": int(season_test),
                    "model": model_kind,
                    "n_train": int(len(train)),
                    "n_test": int(len(test)),
                    **metrics,
                }
            )

    out = pd.DataFrame(rows)

    out_dir = paths.tables_dir() / "showcase"
    out_fp = out_dir / "mcm2026c_q1_ml_elimination_baselines_cv.csv"
    io.write_csv(out, out_fp)

    metric_cols = [
        "roc_auc",
        "average_precision",
        "accuracy",
        "brier",
        "log_loss",
        "prevalence",
    ]
    summary = (
        out.groupby(["model"], sort=True)
        .agg(
            n_folds=("season_test", "nunique"),
            n_test_total=("n_test", "sum"),
            **{f"{c}_mean": (c, "mean") for c in metric_cols},
            **{f"{c}_std": (c, "std") for c in metric_cols},
        )
        .reset_index()
    )
    summary_fp = out_dir / "mcm2026c_q1_ml_elimination_baselines_cv_summary.csv"
    io.write_csv(summary, summary_fp)

    return Q1BaselineOutputs(cv_metrics_csv=out_fp, cv_summary_csv=summary_fp)


def main() -> int:
    out = run()
    print(f"Wrote: {out.cv_metrics_csv}")
    print(f"Wrote: {out.cv_summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
