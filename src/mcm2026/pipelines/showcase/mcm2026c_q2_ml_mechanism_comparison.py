from __future__ import annotations

# Ref: docs/spec/task.md
# Ref: docs/spec/architecture.md

"""
Q2 Machine Learning Showcase: Advanced Mechanism Comparison Analysis

This module implements machine learning approaches for analyzing counterfactual simulation results.
Instead of simple statistical comparisons, we use ML to:
1. Predict mechanism performance from simulation parameters
2. Classify which mechanisms are most robust under different conditions
3. Identify key factors that drive mechanism differences

The purpose is to demonstrate:
1. ML-driven analysis of simulation results
2. Feature importance analysis for mechanism selection
3. Predictive modeling of counterfactual scenarios
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

from mcm2026.core import paths
from mcm2026.data import io
from mcm2026.pipelines import mcm2026c_q2_counterfactual_simulation as q2_main


@dataclass(frozen=True)
class Q2MLOutputs:
    mechanism_prediction_csv: Path
    feature_importance_csv: Path
    performance_analysis_csv: Path


def _as_float_list(x: object, default: list[float]) -> list[float]:
    """Convert to float list with fallback."""
    if isinstance(x, (list, tuple)):
        out: list[float] = []
        for v in x:
            try:
                out.append(float(v))
            except Exception:
                continue
        return out if out else list(default)
    return list(default)


def _as_str_list(x: object, default: list[str]) -> list[str]:
    """Convert to string list with fallback."""
    if isinstance(x, (list, tuple)):
        return [str(v) for v in x]
    return list(default)


def _prepare_ml_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Prepare features for ML analysis of mechanism comparison results."""
    
    # Create features from simulation parameters and results
    features_df = df.copy()
    
    # Numerical features from simulation results
    numerical_features = [
        "match_rate_percent",
        "match_rate_rank", 
        "n_exit_weeks",
        "n_weeks",
        "diff_weeks_percent_vs_rank",
        "mean_fan_share_observed",
        "mean_judge_pct_observed",
    ]
    
    # Add derived features
    if "n_exit_weeks" in df.columns and "n_weeks" in df.columns:
        features_df["exit_rate"] = (
            pd.to_numeric(df["n_exit_weeks"], errors="coerce") / 
            pd.to_numeric(df["n_weeks"], errors="coerce").replace(0, 1)
        )
        numerical_features.append("exit_rate")
    
    if "diff_weeks_percent_vs_rank" in df.columns and "n_exit_weeks" in df.columns:
        features_df["controversy_rate"] = (
            pd.to_numeric(df["diff_weeks_percent_vs_rank"], errors="coerce") / 
            pd.to_numeric(df["n_exit_weeks"], errors="coerce").replace(0, 1)
        )
        numerical_features.append("controversy_rate")
    
    # Categorical features
    categorical_features = []
    if "fan_source_mechanism" in df.columns:
        categorical_features.append("fan_source_mechanism")
    if "count_withdraw_as_exit" in df.columns:
        categorical_features.append("count_withdraw_as_exit")
    
    # Season-level aggregation features if available
    if "season" in df.columns and len(df) > 1:
        # Add season era feature
        features_df["season_era"] = pd.cut(
            pd.to_numeric(df["season"], errors="coerce"),
            bins=[0, 10, 20, 30, 40],
            labels=["early", "mid", "late", "recent"],
        ).astype(str)
        categorical_features.append("season_era")
    
    # Filter numerical features to only include those that exist
    numerical_features = [f for f in numerical_features if f in features_df.columns]
    categorical_features = [f for f in categorical_features if f in features_df.columns]
    
    return features_df, numerical_features, categorical_features


def _train_mechanism_predictor(
    df: pd.DataFrame,
    target_col: str,
    numerical_features: list[str],
    categorical_features: list[str],
    *,
    task_type: str = "classification",
) -> tuple[Any, dict[str, float], pd.DataFrame]:
    """Train ML model to predict mechanism performance."""
    
    # Prepare features
    X_num = df[numerical_features].fillna(0)
    X_cat = pd.get_dummies(df[categorical_features], prefix_sep="_") if categorical_features else pd.DataFrame()
    
    if len(X_cat.columns) > 0:
        X = pd.concat([X_num, X_cat], axis=1)
    else:
        X = X_num
    
    y = df[target_col]
    
    # Handle missing values
    X = X.fillna(0)
    
    if task_type == "classification":
        # Encode target if string
        if y.dtype == "object":
            le = LabelEncoder()
            y = le.fit_transform(y)
        
        # Train Random Forest classifier
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=20260130,
        )
        model.fit(X, y)
        
        # Cross-validation scores
        cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
        
        # Predictions for analysis
        y_pred = model.predict(X)
        
        metrics = {
            "cv_accuracy_mean": float(cv_scores.mean()),
            "cv_accuracy_std": float(cv_scores.std()),
            "train_accuracy": float(accuracy_score(y, y_pred)),
        }
        
    else:  # regression
        # Train Random Forest regressor
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=20260130,
        )
        model.fit(X, y)
        
        # Cross-validation scores
        cv_scores = cross_val_score(model, X, y, cv=5, scoring="r2")
        
        # Predictions for analysis
        y_pred = model.predict(X)
        
        metrics = {
            "cv_r2_mean": float(cv_scores.mean()),
            "cv_r2_std": float(cv_scores.std()),
            "train_r2": float(r2_score(y, y_pred)),
            "train_rmse": float(np.sqrt(mean_squared_error(y, y_pred))),
        }
    
    # Feature importance
    feature_importance = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_,
        "target": target_col,
        "task_type": task_type,
    }).sort_values("importance", ascending=False)
    
    return model, metrics, feature_importance


def run(
    *,
    seed: int = 20260130,
    fan_source_mechanisms: object = None,
    count_withdraw_options: object = None,
    output_dir: Path | None = None,
) -> Q2MLOutputs:
    """Run ML analysis of Q2 counterfactual simulation results."""
    
    paths.ensure_dirs()
    
    # Set random seed
    np.random.seed(seed)
    
    # Default parameters
    mechanisms = _as_str_list(fan_source_mechanisms, ["percent", "rank"])
    withdraw_options = [True, False] if count_withdraw_options is None else list(count_withdraw_options)
    
    # Collect simulation results
    all_results = []
    
    for mech in mechanisms:
        for count_withdraw in withdraw_options:
            print(f"Running Q2 simulation: mechanism={mech}, count_withdraw={count_withdraw}")
            
            try:
                out = q2_main.run(
                    fan_source_mechanism=str(mech),
                    count_withdraw_as_exit=bool(count_withdraw),
                )
                
                # Read results
                df = io.read_table(out.mechanism_comparison_csv)
                df["fan_source_mechanism"] = str(mech)
                df["count_withdraw_as_exit"] = bool(count_withdraw)
                
                all_results.append(df)
                
            except Exception as e:
                print(f"  Error: {e}")
                continue
    
    if not all_results:
        raise ValueError("No simulation results collected")
    
    # Combine all results
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # Prepare ML features
    features_df, numerical_features, categorical_features = _prepare_ml_features(combined_df)
    
    # ML Analysis 1: Predict best mechanism from simulation parameters
    mechanism_models = {}
    mechanism_metrics = {}
    all_feature_importance = []
    
    # Task 1: Predict which mechanism has higher match_rate_percent
    if "match_rate_percent" in features_df.columns:
        # Create binary target: is this the best mechanism for this scenario?
        features_df["is_best_mechanism"] = False
        
        for group_cols in [["count_withdraw_as_exit"], ["season_era"]]:
            if all(col in features_df.columns for col in group_cols):
                for group_vals, group_df in features_df.groupby(group_cols):
                    if len(group_df) > 1:
                        best_idx = group_df["match_rate_percent"].idxmax()
                        features_df.loc[best_idx, "is_best_mechanism"] = True
        
        # Train classifier
        model, metrics, importance = _train_mechanism_predictor(
            features_df,
            "is_best_mechanism",
            numerical_features,
            categorical_features,
            task_type="classification",
        )
        
        mechanism_models["best_mechanism_classifier"] = model
        mechanism_metrics["best_mechanism_classifier"] = metrics
        all_feature_importance.append(importance)
    
    # Task 2: Predict match_rate_percent directly
    if "match_rate_percent" in features_df.columns:
        model, metrics, importance = _train_mechanism_predictor(
            features_df,
            "match_rate_percent",
            numerical_features,
            categorical_features,
            task_type="regression",
        )
        
        mechanism_models["match_rate_percent_regressor"] = model
        mechanism_metrics["match_rate_percent_regressor"] = metrics
        all_feature_importance.append(importance)
    
    # Task 3: Predict controversy_rate
    if "controversy_rate" in features_df.columns:
        model, metrics, importance = _train_mechanism_predictor(
            features_df,
            "controversy_rate", 
            numerical_features,
            categorical_features,
            task_type="regression",
        )
        
        mechanism_models["controversy_rate_regressor"] = model
        mechanism_metrics["controversy_rate_regressor"] = metrics
        all_feature_importance.append(importance)
    
    # Save results
    out_dir = (paths.tables_dir() / "showcase") if output_dir is None else Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Mechanism prediction results
    prediction_results = []
    for model_name, metrics in mechanism_metrics.items():
        prediction_results.append({
            "model_name": model_name,
            "seed": int(seed),
            **metrics,
        })
    
    prediction_df = pd.DataFrame(prediction_results)
    prediction_fp = out_dir / "mcm2026c_q2_ml_mechanism_prediction.csv"
    io.write_csv(prediction_df, prediction_fp)
    
    # Feature importance
    if all_feature_importance:
        importance_df = pd.concat(all_feature_importance, ignore_index=True)
        importance_fp = out_dir / "mcm2026c_q2_ml_feature_importance.csv"
        io.write_csv(importance_df, importance_fp)
    else:
        importance_fp = out_dir / "mcm2026c_q2_ml_feature_importance.csv"
        io.write_csv(pd.DataFrame(), importance_fp)
    
    # Performance analysis by mechanism
    performance_analysis = []
    
    if "fan_source_mechanism" in combined_df.columns:
        for mech, mech_df in combined_df.groupby("fan_source_mechanism"):
            analysis = {
                "mechanism": str(mech),
                "n_simulations": len(mech_df),
                "match_rate_percent_mean": float(mech_df["match_rate_percent"].mean()) if "match_rate_percent" in mech_df.columns else float("nan"),
                "match_rate_percent_std": float(mech_df["match_rate_percent"].std()) if "match_rate_percent" in mech_df.columns else float("nan"),
                "match_rate_rank_mean": float(mech_df["match_rate_rank"].mean()) if "match_rate_rank" in mech_df.columns else float("nan"),
                "match_rate_rank_std": float(mech_df["match_rate_rank"].std()) if "match_rate_rank" in mech_df.columns else float("nan"),
                "controversy_rate_mean": float(mech_df["controversy_rate"].mean()) if "controversy_rate" in mech_df.columns else float("nan"),
                "controversy_rate_std": float(mech_df["controversy_rate"].std()) if "controversy_rate" in mech_df.columns else float("nan"),
            }
            performance_analysis.append(analysis)
    
    performance_df = pd.DataFrame(performance_analysis)
    performance_fp = out_dir / "mcm2026c_q2_ml_performance_analysis.csv"
    io.write_csv(performance_df, performance_fp)
    
    return Q2MLOutputs(
        mechanism_prediction_csv=prediction_fp,
        feature_importance_csv=importance_fp,
        performance_analysis_csv=performance_fp,
    )


def main() -> int:
    out = run()
    print(f"Wrote: {out.mechanism_prediction_csv}")
    print(f"Wrote: {out.feature_importance_csv}")
    print(f"Wrote: {out.performance_analysis_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())