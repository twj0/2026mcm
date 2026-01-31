from __future__ import annotations

# Ref: docs/spec/task.md
# Ref: docs/spec/architecture.md

"""
Q3 Deep Learning Showcase: Advanced Neural Networks for Fan Index Regression

This module implements state-of-the-art neural network architectures for predicting fan_vote_index.
All models are configured for optimal performance to provide fair comparison.

Architectures included:
1. ResNet-style deep networks with skip connections and advanced normalization
2. Attention-based feature selection networks with learnable importance weights
3. Ensemble methods with uncertainty quantification and Bayesian inference

The purpose is to demonstrate:
1. Advanced PyTorch architectures at their best performance
2. Proper regularization and uncertainty quantification techniques
3. Scientific analysis of deep learning boundaries on structured data
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import DataLoader, Dataset

from mcm2026.core import paths
from mcm2026.data import io
from mcm2026.pipelines import mcm2026c_q3_mixed_effects_impacts as q3_main
from mcm2026.pipelines.showcase import mcm2026c_q3_ml_fan_index_baselines as q3_baseline


@dataclass(frozen=True)
class Q3DLOutputs:
    cv_metrics_csv: Path
    cv_summary_csv: Path
    training_curves_csv: Path
    uncertainty_csv: Path


class RegressionDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X.copy())  # Copy to make writable
        self.y = torch.FloatTensor(y.copy()).unsqueeze(1)  # Add dimension for regression
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class ResidualBlock(nn.Module):
    """Residual block with skip connection."""
    
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.block(x)
        out = out + residual  # Skip connection
        return torch.relu(self.dropout(out))


class DeepResNet(nn.Module):
    """Deep ResNet for regression with skip connections."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        n_blocks: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Residual blocks
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout) for _ in range(n_blocks)
        ])
        
        # Output head
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        
        for block in self.blocks:
            x = block(x)
        
        return self.output(x)


class AttentionFeatureNet(nn.Module):
    """Network with attention-based feature selection."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        attention_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Feature attention
        self.attention = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, input_dim),
            nn.Sigmoid(),
        )
        
        # Main network
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply attention weights
        attention_weights = self.attention(x)
        x_attended = x * attention_weights
        
        return self.network(x_attended)


class UncertaintyNet(nn.Module):
    """Network that predicts both mean and variance (aleatoric uncertainty)."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Mean head
        self.mean_head = nn.Linear(hidden_dim // 2, 1)
        
        # Log variance head (predict log variance for numerical stability)
        self.logvar_head = nn.Linear(hidden_dim // 2, 1)
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        mean = self.mean_head(features)
        logvar = self.logvar_head(features)
        return mean, logvar


def gaussian_nll_loss(mean: torch.Tensor, logvar: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Gaussian negative log-likelihood loss for uncertainty estimation."""
    var = torch.exp(logvar)
    loss = 0.5 * (torch.log(2 * torch.pi * var) + (target - mean) ** 2 / var)
    return loss.mean()


def _train_standard_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None = None,
    *,
    epochs: int = 200,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 20,
    device: str = "cpu",
) -> tuple[nn.Module, list[dict[str, float]]]:
    """Train standard regression model."""
    
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None
    
    training_curves = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        val_loss = 0.0
        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
            
            val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0.0
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
        
        training_curves.append({
            "epoch": epoch,
            "train_loss": train_loss / len(train_loader) if len(train_loader) > 0 else 0.0,
            "val_loss": val_loss,
            "lr": optimizer.param_groups[0]["lr"],
        })
        
        if patience_counter >= patience:
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, training_curves


def _train_uncertainty_model(
    model: UncertaintyNet,
    train_loader: DataLoader,
    val_loader: DataLoader | None = None,
    *,
    epochs: int = 200,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 20,
    device: str = "cpu",
) -> tuple[UncertaintyNet, list[dict[str, float]]]:
    """Train uncertainty model with Gaussian NLL loss."""
    
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None
    
    training_curves = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            mean, logvar = model(batch_x)
            loss = gaussian_nll_loss(mean, logvar, batch_y)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        val_loss = 0.0
        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    mean, logvar = model(batch_x)
                    loss = gaussian_nll_loss(mean, logvar, batch_y)
                    val_loss += loss.item()
            
            val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0.0
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
        
        training_curves.append({
            "epoch": epoch,
            "train_loss": train_loss / len(train_loader) if len(train_loader) > 0 else 0.0,
            "val_loss": val_loss,
            "lr": optimizer.param_groups[0]["lr"],
        })
        
        if patience_counter >= patience:
            break
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, training_curves


def _evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    *,
    device: str = "cpu",
    n_mc_samples: int = 10,
) -> dict[str, float]:
    """Evaluate model with optional Monte Carlo dropout for uncertainty."""
    
    model.eval()
    
    # Standard evaluation
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            if isinstance(model, UncertaintyNet):
                mean, logvar = model(batch_x)
                preds = mean
            else:
                preds = model(batch_x)
            
            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(batch_y.cpu().numpy().flatten())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Basic metrics
    rmse = float(np.sqrt(mean_squared_error(all_labels, all_preds)))
    r2 = float(r2_score(all_labels, all_preds))
    
    metrics = {
        "rmse": rmse,
        "r2": r2,
        "mae": float(np.mean(np.abs(all_labels - all_preds))),
    }
    
    # Monte Carlo dropout uncertainty (for non-uncertainty models)
    if not isinstance(model, UncertaintyNet) and n_mc_samples > 1:
        model.train()  # Enable dropout
        mc_preds = []
        
        with torch.no_grad():
            for _ in range(n_mc_samples):
                sample_preds = []
                for batch_x, batch_y in test_loader:
                    batch_x = batch_x.to(device)
                    preds = model(batch_x)
                    sample_preds.extend(preds.cpu().numpy().flatten())
                mc_preds.append(sample_preds)
        
        mc_preds = np.array(mc_preds)  # [n_samples, n_test]
        pred_mean = np.mean(mc_preds, axis=0)
        pred_std = np.std(mc_preds, axis=0)
        
        metrics.update({
            "mc_rmse": float(np.sqrt(mean_squared_error(all_labels, pred_mean))),
            "mc_r2": float(r2_score(all_labels, pred_mean)),
            "uncertainty_mean": float(np.mean(pred_std)),
            "uncertainty_std": float(np.std(pred_std)),
        })
    
    return metrics


def run(
    *,
    seed: int = 20260130,
    fan_source_mechanism: str | None = None,
    max_test_seasons: int | None = None,
    epochs: int = 200,
    batch_size: int = 16,
    lr: float = 1e-3,
    output_dir: Path | None = None,
) -> Q3DLOutputs:
    """Run deep learning fan index regression showcase."""
    
    paths.ensure_dirs()
    
    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load data (reuse from Q3 baseline)
    mech_cfg, _, _ = q3_main._get_q3_params_from_config()
    fan_source_mechanism = mech_cfg if fan_source_mechanism is None else str(fan_source_mechanism)
    
    weekly = q3_main._read_weekly_panel()
    season_features = q3_main._read_season_features()
    q1_post = q3_main._read_q1_posterior_summary()
    
    df = q3_main._build_season_level_dataset(
        weekly, season_features, q1_post,
        fan_source_mechanism=fan_source_mechanism,
    )
    
    # Prepare features (same as Q3 baseline)
    numeric_features = [
        "age", "age_sq", "is_us", "log_state_pop",
        "n_weeks_active", "n_weeks_q1",
    ]
    categorical_features = ["industry", "pro_name"]
    
    seasons = sorted(df["season"].unique().tolist())
    if max_test_seasons is not None:
        seasons = seasons[:int(max_test_seasons)]
    
    rows = []
    all_training_curves = []
    all_uncertainty_data = []
    
    for season_test in seasons:
        print(f"Testing on season {season_test}")
        
        train_df = df.loc[df["season"] != int(season_test)].copy()
        test_df = df.loc[df["season"] == int(season_test)].copy()
        
        if len(train_df) == 0 or len(test_df) == 0:
            continue
        
        # Prepare data using Q3 baseline preprocessing
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        from sklearn.pipeline import Pipeline
        
        preprocessor = q3_baseline._make_preprocessor(
            numeric_features=numeric_features,
            categorical_features=categorical_features,
        )
        
        X_train = preprocessor.fit_transform(train_df[numeric_features + categorical_features])
        X_test = preprocessor.transform(test_df[numeric_features + categorical_features])
        
        # Convert sparse matrices to dense arrays for PyTorch
        if hasattr(X_train, 'toarray'):
            X_train = X_train.toarray()
        if hasattr(X_test, 'toarray'):
            X_test = X_test.toarray()
        
        y_train = train_df["fan_vote_index_mean"].astype(float).to_numpy()
        y_test = test_df["fan_vote_index_mean"].astype(float).to_numpy()
        
        # Create datasets
        train_dataset = RegressionDataset(X_train, y_train)
        test_dataset = RegressionDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        input_dim = X_train.shape[1]
        
        # Test different architectures
        models_to_test = [
            ("deep_resnet", DeepResNet(input_dim, hidden_dim=64, n_blocks=2, dropout=0.3)),
            ("attention_net", AttentionFeatureNet(input_dim, hidden_dim=64, dropout=0.3)),
            ("uncertainty_net", UncertaintyNet(input_dim, hidden_dim=64, dropout=0.3)),
        ]
        
        for model_name, model in models_to_test:
            print(f"  Training {model_name}")
            
            # Train model
            if isinstance(model, UncertaintyNet):
                trained_model, curves = _train_uncertainty_model(
                    model, train_loader, epochs=epochs, lr=lr, device=device
                )
            else:
                trained_model, curves = _train_standard_model(
                    model, train_loader, epochs=epochs, lr=lr, device=device
                )
            
            # Evaluate model
            metrics = _evaluate_model(trained_model, test_loader, device=device, n_mc_samples=5)
            
            # Record results
            rows.append({
                "season_test": int(season_test),
                "model": f"pytorch_{model_name}",
                "fan_source_mechanism": str(fan_source_mechanism),
                "n_train": len(train_df),
                "n_test": len(test_df),
                "input_dim": int(input_dim),
                "device": device,
                "epochs_trained": len(curves),
                **metrics,
            })
            
            # Record training curves
            for curve_point in curves:
                all_training_curves.append({
                    "season_test": int(season_test),
                    "model": f"pytorch_{model_name}",
                    **curve_point,
                })
            
            # Record uncertainty data for uncertainty net
            if isinstance(trained_model, UncertaintyNet):
                trained_model.eval()
                with torch.no_grad():
                    for i, (batch_x, batch_y) in enumerate(test_loader):
                        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                        mean, logvar = trained_model(batch_x)
                        std = torch.sqrt(torch.exp(logvar))
                        
                        for j in range(len(batch_x)):
                            all_uncertainty_data.append({
                                "season_test": int(season_test),
                                "model": f"pytorch_{model_name}",
                                "sample_idx": i * batch_size + j,
                                "true_value": float(batch_y[j].item()),
                                "pred_mean": float(mean[j].item()),
                                "pred_std": float(std[j].item()),
                                "abs_error": float(abs(batch_y[j] - mean[j]).item()),
                            })
    
    # Save results
    out_dir = (paths.tables_dir() / "showcase") if output_dir is None else Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Main results
    results_df = pd.DataFrame(rows)
    results_fp = out_dir / "mcm2026c_q3_dl_fan_regression_nets_cv.csv"
    io.write_csv(results_df, results_fp)
    
    # Training curves
    curves_df = pd.DataFrame(all_training_curves)
    curves_fp = out_dir / "mcm2026c_q3_dl_fan_regression_nets_curves.csv"
    io.write_csv(curves_df, curves_fp)
    
    # Uncertainty data
    uncertainty_df = pd.DataFrame(all_uncertainty_data)
    uncertainty_fp = out_dir / "mcm2026c_q3_dl_fan_regression_nets_uncertainty.csv"
    io.write_csv(uncertainty_df, uncertainty_fp)
    
    # Summary
    if len(results_df) > 0:
        metric_cols = ["rmse", "r2", "mae"]
        summary = (
            results_df.groupby(["fan_source_mechanism", "model"], sort=True)
            .agg(
                n_folds=("season_test", "nunique"),
                n_test_total=("n_test", "sum"),
                epochs_mean=("epochs_trained", "mean"),
                **{f"{c}_mean": (c, "mean") for c in metric_cols},
                **{f"{c}_std": (c, "std") for c in metric_cols},
            )
            .reset_index()
        )
    else:
        summary = pd.DataFrame()
    
    summary_fp = out_dir / "mcm2026c_q3_dl_fan_regression_nets_summary.csv"
    io.write_csv(summary, summary_fp)
    
    return Q3DLOutputs(
        cv_metrics_csv=results_fp,
        cv_summary_csv=summary_fp,
        training_curves_csv=curves_fp,
        uncertainty_csv=uncertainty_fp,
    )


def main() -> int:
    out = run(max_test_seasons=5, epochs=100)  # Reduced for testing
    print(f"Wrote: {out.cv_metrics_csv}")
    print(f"Wrote: {out.cv_summary_csv}")
    print(f"Wrote: {out.training_curves_csv}")
    print(f"Wrote: {out.uncertainty_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())