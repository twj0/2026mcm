from __future__ import annotations

# Ref: docs/spec/task.md
# Ref: docs/spec/architecture.md

"""
Q4 Machine Learning Showcase: Advanced System Design Optimization

This module implements machine learning approaches for optimizing the new voting system design.
Instead of manual parameter tuning, we use ML to:
1. Predict optimal system parameters from historical data
2. Classify which system configurations are most robust
3. Optimize multi-objective trade-offs (fairness vs entertainment vs predictability)

The purpose is to demonstrate:
1. ML-driven system design optimization
2. Multi-objective optimization with Pareto frontiers
3. Predictive modeling for system parameter selection
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

from mcm2026.core import paths
from mcm2026.data import io
from mcm2026.pipelines import mcm2026c_q4_design_space_eval as q4_main


@dataclass(frozen=True)
class Q4MLOutputs:
    parameter_optimization_csv: Path
    pareto_frontier_csv: Path
    robustness_analysis_csv: Path
    feature_importance_csv: Path


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


def _prepare_q4_ml_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Prepare features for ML analysis of Q4 system design results."""
    
    features_df = df.copy()
    
    # Numerical features from system parameters and results
    numerical_features = [
        "alpha",
        "n_sims", 
        "outlier_mult",  # Correct column name
        "champion_mode_prob",
        "champion_entropy",
        "tpi_season_avg",
        "fan_vs_uniform_contrast",
        "robust_fail_rate",
    ]
    
    # Add derived features for better ML performance
    if "champion_mode_prob" in df.columns and "champion_entropy" in df.columns:
        features_df["champion_predictability"] = (
            pd.to_numeric(df["champion_mode_prob"], errors="coerce") * 
            (1 - pd.to_numeric(df["champion_entropy"], errors="coerce"))
        )
        numerical_features.append("champion_predictability")
    
    if "tpi_season_avg" in df.columns and "fan_vs_uniform_contrast" in df.columns:
        features_df["system_balance"] = (
            pd.to_numeric(df["tpi_season_avg"], errors="coerce") * 
            pd.to_numeric(df["fan_vs_uniform_contrast"], errors="coerce")
        )
        numerical_features.append("system_balance")
    
    if "robust_fail_rate" in df.columns:
        features_df["robustness_score"] = 1 - pd.to_numeric(df["robust_fail_rate"], errors="coerce")
        numerical_features.append("robustness_score")
    
    # Categorical features
    categorical_features = []
    if "mechanism" in df.columns:
        categorical_features.append("mechanism")
    if "season" in df.columns:
        # Convert season to categorical bins
        features_df["season_era"] = pd.cut(
            pd.to_numeric(df["season"], errors="coerce"),
            bins=[0, 10, 20, 30, 40],
            labels=["early", "mid", "late", "recent"],
        ).astype(str)
        categorical_features.append("season_era")
    
    # Parameter interaction features
    if "alpha" in df.columns and "outlier_mult" in df.columns:
        features_df["alpha_outlier_interaction"] = (
            pd.to_numeric(df["alpha"], errors="coerce") * 
            pd.to_numeric(df["outlier_mult"], errors="coerce")
        )
        numerical_features.append("alpha_outlier_interaction")
    
    # Filter features to only include those that exist
    numerical_features = [f for f in numerical_features if f in features_df.columns]
    categorical_features = [f for f in categorical_features if f in features_df.columns]
    
    return features_df, numerical_features, categorical_features


def _compute_pareto_frontier(df: pd.DataFrame, objectives: list[str]) -> pd.DataFrame:
    """Compute Pareto frontier for multi-objective optimization."""
    
    # Extract objective values
    obj_values = df[objectives].fillna(0).values
    
    # Find Pareto optimal points
    is_pareto = np.ones(len(obj_values), dtype=bool)
    
    for i in range(len(obj_values)):
        if is_pareto[i]:
            # Check if point i is dominated by any other point
            for j in range(len(obj_values)):
                if i != j and is_pareto[j]:
                    # Point j dominates point i if j is better in all objectives
                    dominates = all(obj_values[j, k] >= obj_values[i, k] for k in range(len(objectives)))
                    strictly_better = any(obj_values[j, k] > obj_values[i, k] for k in range(len(objectives)))
                    
                    if dominates and strictly_better:
                        is_pareto[i] = False
                        break
    
    # Return Pareto optimal points
    pareto_df = df[is_pareto].copy()
    pareto_df["is_pareto_optimal"] = True
    
    return pareto_df


def _train_system_optimizer(
    df: pd.DataFrame,
    target_cols: list[str],
    numerical_features: list[str],
    categorical_features: list[str],
) -> tuple[Any, dict[str, float], pd.DataFrame]:
    """Train ML model to optimize system parameters."""
    
    # Prepare features
    X_num = df[numerical_features].fillna(0)
    X_cat = pd.get_dummies(df[categorical_features], prefix_sep="_") if categorical_features else pd.DataFrame()
    
    if len(X_cat.columns) > 0:
        X = pd.concat([X_num, X_cat], axis=1)
    else:
        X = X_num
    
    # Handle missing values
    X = X.fillna(0)
    
    # Multi-output targets
    y = df[target_cols].fillna(0)
    
    # Train multi-output Random Forest
    base_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=3,
        min_samples_leaf=1,
        random_state=20260130,
    )
    
    model = MultiOutputRegressor(base_model)
    model.fit(X, y)
    
    # Cross-validation scores for each target
    metrics = {}
    for i, target in enumerate(target_cols):
        cv_scores = cross_val_score(
            RandomForestRegressor(**base_model.get_params()),
            X, y.iloc[:, i],
            cv=5, scoring="r2"
        )
        
        y_pred = model.predict(X)[:, i]
        
        metrics[f"{target}_cv_r2_mean"] = float(cv_scores.mean())
        metrics[f"{target}_cv_r2_std"] = float(cv_scores.std())
        metrics[f"{target}_train_r2"] = float(r2_score(y.iloc[:, i], y_pred))
        metrics[f"{target}_train_rmse"] = float(np.sqrt(mean_squared_error(y.iloc[:, i], y_pred)))
    
    # Feature importance (average across all targets)
    if hasattr(model.estimators_[0], 'feature_importances_'):
        avg_importance = np.mean([est.feature_importances_ for est in model.estimators_], axis=0)
        
        feature_importance = pd.DataFrame({
            "feature": X.columns,
            "importance": avg_importance,
            "targets": "_".join(target_cols),
        }).sort_values("importance", ascending=False)
    else:
        feature_importance = pd.DataFrame()
    
    return model, metrics, feature_importance


def run(
    *,
    seed: int = 20260130,
    alphas: object = None,
    n_sims_list: object = None,
    outlier_multipliers: object = None,
    mechanisms: object = None,
    max_runs: int | None = None,
    output_dir: Path | None = None,
) -> Q4MLOutputs:
    """Run ML optimization analysis of Q4 system design."""
    
    paths.ensure_dirs()
    
    # Set random seed
    np.random.seed(seed)
    
    # Default parameters
    alpha_list = _as_float_list(alphas, [0.3, 0.5, 0.7])
    n_sims_grid = [10, 20, 50] if n_sims_list is None else list(n_sims_list)
    outlier_mults = _as_float_list(outlier_multipliers, [2.0, 5.0, 10.0])
    mech_list = _as_str_list(mechanisms, ["percent", "rank", "percent_judge_save"])
    
    # Use existing Q4 results instead of running new simulations
    print("Using existing Q4 results from outputs/tables/mcm2026c_q4_new_system_metrics.csv")
    
    try:
        existing_results_path = paths.tables_dir() / "mcm2026c_q4_new_system_metrics.csv"
        combined_df = io.read_table(existing_results_path)
        print(f"Loaded {len(combined_df)} rows from existing Q4 results")
    except Exception as e:
        print(f"Could not load existing Q4 results: {e}")
        print("Running limited Q4 simulations...")
        
        # Fallback: run limited simulations
        all_results = []
        run_count = 0
        
        for alpha in alpha_list[:2]:  # Limit to 2 alphas
            for n_sims in n_sims_grid[:1]:  # Limit to 1 n_sims
                for outlier_mult in outlier_mults[:2]:  # Limit to 2 outliers
                    for mech in mech_list[:2]:  # Limit to 2 mechanisms
                        if max_runs is not None and run_count >= max_runs:
                            break
                        
                        print(f"Running Q4 simulation: alpha={alpha}, n_sims={n_sims}, outlier={outlier_mult}, mech={mech}")
                        
                        try:
                            out = q4_main.run(
                                alpha=alpha,
                                n_sims=n_sims,
                                outlier_mults=[outlier_mult],
                                fan_source_mechanism=mech,
                            )
                            
                            # Read results
                            df = io.read_table(out.new_system_metrics_csv)
                            df["run_id"] = run_count
                            
                            all_results.append(df)
                            run_count += 1
                            
                        except Exception as e:
                            print(f"  Error: {e}")
                            continue
        
        if not all_results:
            raise ValueError("No Q4 simulation results collected")
        
        # Combine all results
        combined_df = pd.concat(all_results, ignore_index=True)
    
    # Prepare ML features
    features_df, numerical_features, categorical_features = _prepare_q4_ml_features(combined_df)
    
    # Define optimization objectives
    objectives = [
        "champion_mode_prob",    # Predictability (higher is more predictable)
        "robustness_score",      # Robustness (higher is more robust)
        "tpi_season_avg",        # Fan influence (higher means more fan influence)
    ]
    
    # Filter objectives that exist in data
    available_objectives = [obj for obj in objectives if obj in features_df.columns]
    
    if not available_objectives:
        raise ValueError("No optimization objectives found in data")
    
    # ML Analysis 1: Multi-objective parameter optimization
    parameter_features = ["alpha", "n_sims", "outlier_mult"]  # Correct column name
    available_param_features = [f for f in parameter_features if f in numerical_features]
    
    optimization_results = {}
    all_feature_importance = []
    
    if available_param_features and available_objectives:
        model, metrics, importance = _train_system_optimizer(
            features_df,
            available_objectives,
            available_param_features,
            categorical_features,
        )
        
        optimization_results["multi_objective_optimizer"] = metrics
        all_feature_importance.append(importance)
    
    # ML Analysis 2: Pareto frontier analysis
    if len(available_objectives) >= 2:
        pareto_df = _compute_pareto_frontier(features_df, available_objectives)
    else:
        pareto_df = pd.DataFrame()  # Empty if not enough objectives
    
    # ML Analysis 3: Robustness prediction
    if "robust_fail_rate" in features_df.columns:
        robustness_model, robustness_metrics, robustness_importance = _train_system_optimizer(
            features_df,
            ["robust_fail_rate"],
            numerical_features,
            categorical_features,
        )
        
        optimization_results["robustness_predictor"] = robustness_metrics
        all_feature_importance.append(robustness_importance)
    
    # Save results
    out_dir = (paths.tables_dir() / "showcase") if output_dir is None else Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Parameter optimization results
    optimization_df = pd.DataFrame([
        {"model_name": model_name, "seed": int(seed), **metrics}
        for model_name, metrics in optimization_results.items()
    ])
    optimization_fp = out_dir / "mcm2026c_q4_ml_parameter_optimization.csv"
    io.write_csv(optimization_df, optimization_fp)
    
    # Pareto frontier
    pareto_fp = out_dir / "mcm2026c_q4_ml_pareto_frontier.csv"
    io.write_csv(pareto_df, pareto_fp)
    
    # Robustness analysis
    robustness_analysis = []
    
    if "mechanism" in features_df.columns:
        for mech, mech_df in features_df.groupby("mechanism"):
            analysis = {
                "mechanism": str(mech),
                "n_configurations": len(mech_df),
                "avg_robustness": float(mech_df["robustness_score"].mean()) if "robustness_score" in mech_df.columns else float("nan"),
                "avg_predictability": float(mech_df["champion_mode_prob"].mean()) if "champion_mode_prob" in mech_df.columns else float("nan"),
                "avg_fan_influence": float(mech_df["tpi_season_avg"].mean()) if "tpi_season_avg" in mech_df.columns else float("nan"),
                "pareto_optimal_count": int(len(pareto_df[pareto_df["mechanism"] == str(mech)])) if "is_pareto_optimal" in pareto_df.columns else 0,
            }
            robustness_analysis.append(analysis)
    
    robustness_df = pd.DataFrame(robustness_analysis)
    robustness_fp = out_dir / "mcm2026c_q4_ml_robustness_analysis.csv"
    io.write_csv(robustness_df, robustness_fp)
    
    # Feature importance
    if all_feature_importance:
        importance_df = pd.concat(all_feature_importance, ignore_index=True)
        importance_fp = out_dir / "mcm2026c_q4_ml_feature_importance.csv"
        io.write_csv(importance_df, importance_fp)
    else:
        importance_fp = out_dir / "mcm2026c_q4_ml_feature_importance.csv"
        io.write_csv(pd.DataFrame(), importance_fp)
    
    return Q4MLOutputs(
        parameter_optimization_csv=optimization_fp,
        pareto_frontier_csv=pareto_fp,
        robustness_analysis_csv=robustness_fp,
        feature_importance_csv=importance_fp,
    )


def main() -> int:
    # Run with reduced parameters for testing
    out = run(
        alphas=[0.3, 0.5, 0.7],
        n_sims_list=[10, 20],
        outlier_multipliers=[2.0, 5.0],
        mechanisms=["percent", "rank"],
        max_runs=8,  # Limit for testing
    )
    print(f"Wrote: {out.parameter_optimization_csv}")
    print(f"Wrote: {out.pareto_frontier_csv}")
    print(f"Wrote: {out.robustness_analysis_csv}")
    print(f"Wrote: {out.feature_importance_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())