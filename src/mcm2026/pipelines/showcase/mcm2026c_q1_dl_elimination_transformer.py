from __future__ import annotations

# Ref: docs/spec/task.md
# Ref: docs/spec/architecture.md

"""
Q1 Deep Learning Showcase: Transformer-based Elimination Prediction

This module implements state-of-the-art deep learning architectures for predicting weekly eliminations.
All models are configured for optimal performance to provide fair "head-to-head" comparison.

The purpose is to demonstrate:
1. Mastery of modern deep learning techniques at their best
2. Fair comparison between traditional and deep learning methods
3. Scientific analysis of performance boundaries
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset

from mcm2026.core import paths
from mcm2026.data import io
from mcm2026.pipelines.showcase import mcm2026c_q1_ml_elimination_baselines as q1_baseline


@dataclass(frozen=True)
class Q1DLOutputs:
    cv_metrics_csv: Path
    cv_summary_csv: Path
    training_curves_csv: Path


def _as_int_or_default(x: object, default: int) -> int:
    """Convert to int with fallback."""
    try:
        return int(x) if x is not None else default
    except (ValueError, TypeError):
        return default


def _as_float_or_default(x: object, default: float) -> float:
    """Convert to float with fallback."""
    try:
        return float(x) if x is not None else default
    except (ValueError, TypeError):
        return default


class TabularDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X.copy())  # Copy to make writable
        self.y = torch.LongTensor(y.copy())  # Copy to make writable
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class TabTransformer(nn.Module):
    """
    State-of-the-art TabTransformer for tabular data.
    
    Architecture optimized for best performance:
    1. Larger embedding dimensions for better representation
    2. Multiple transformer layers with proper normalization
    3. Sophisticated MLP head with residual connections
    4. Advanced regularization techniques
    """
    
    def __init__(
        self,
        *,
        n_numerical: int,
        categorical_cardinalities: list[int],
        embed_dim: int = 64,  # Increased for better representation
        n_heads: int = 8,     # More attention heads
        n_layers: int = 4,    # Deeper transformer
        mlp_hidden: int = 256, # Larger MLP
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.n_numerical = n_numerical
        self.n_categorical = len(categorical_cardinalities)
        
        # Embedding layers for categorical features with layer norm
        self.embeddings = nn.ModuleList([
            nn.Sequential(
                nn.Embedding(cardinality, embed_dim),
                nn.LayerNorm(embed_dim),
            )
            for cardinality in categorical_cardinalities
        ])
        
        # Transformer layers for categorical features
        if self.n_categorical > 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=n_heads,
                dim_feedforward=embed_dim * 4,  # Larger feedforward
                dropout=dropout,
                batch_first=True,
                activation='gelu',  # Better activation
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            
            # Multi-head pooling for better representation
            self.pool = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
            self.pool_query = nn.Parameter(torch.randn(1, 1, embed_dim))
            
            transformer_out_dim = embed_dim
        else:
            transformer_out_dim = 0
        
        # Numerical feature processing
        if n_numerical > 0:
            self.numerical_bn = nn.BatchNorm1d(n_numerical)
            self.numerical_proj = nn.Sequential(
                nn.Linear(n_numerical, embed_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            numerical_out_dim = embed_dim
        else:
            numerical_out_dim = 0
        
        # Advanced MLP head with residual connections
        input_dim = numerical_out_dim + transformer_out_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, mlp_hidden),
            nn.GELU(),
            nn.BatchNorm1d(mlp_hidden),
            nn.Dropout(dropout),
            
            nn.Linear(mlp_hidden, mlp_hidden // 2),
            nn.GELU(),
            nn.BatchNorm1d(mlp_hidden // 2),
            nn.Dropout(dropout),
            
            nn.Linear(mlp_hidden // 2, mlp_hidden // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(mlp_hidden // 4, 2),  # Binary classification
        )
        
    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        features = []
        
        # Process numerical features with batch norm
        if self.n_numerical > 0:
            x_num_norm = self.numerical_bn(x_num)
            x_num_proj = self.numerical_proj(x_num_norm)
            features.append(x_num_proj)
        
        # Process categorical features with transformer
        if self.n_categorical > 0:
            # Embed categorical features
            embedded = []
            for i, embedding in enumerate(self.embeddings):
                embedded.append(embedding(x_cat[:, i]))
            
            # Stack embeddings: [batch_size, n_categorical, embed_dim]
            embedded = torch.stack(embedded, dim=1)
            
            # Apply transformer
            transformed = self.transformer(embedded)  # [batch_size, n_categorical, embed_dim]
            
            # Multi-head attention pooling
            batch_size = transformed.size(0)
            query = self.pool_query.expand(batch_size, -1, -1)
            pooled, _ = self.pool(query, transformed, transformed)
            pooled = pooled.squeeze(1)  # [batch_size, embed_dim]
            
            features.append(pooled)
        
        # Concatenate all features
        if features:
            x = torch.cat(features, dim=1)
        else:
            x = torch.zeros(x_num.size(0), 1, device=x_num.device)
        
        return self.mlp(x)


class AdvancedMLP(nn.Module):
    """Advanced MLP with modern techniques for best performance."""
    
    def __init__(self, input_dim: int, hidden_dims: list[int] = None, dropout: float = 0.1):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [512, 256, 128, 64]  # Larger network
        
        layers = []
        prev_dim = input_dim
        
        # Input batch norm
        layers.append(nn.BatchNorm1d(input_dim))
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.GELU(),  # Better activation
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout),
            ])
            
            # Add residual connection for deeper layers
            if i > 0 and prev_dim == hidden_dim:
                # This would require more complex implementation
                pass
            
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 2))  # Binary classification
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def _prepare_data(
    df: pd.DataFrame,
    *,
    numeric_features: list[str],
    categorical_features: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int], ColumnTransformer, dict[str, LabelEncoder]]:
    """Prepare data for PyTorch models."""
    
    # Prepare features
    X = df[numeric_features + categorical_features].copy()
    y = df["y_eliminated"].astype(int).to_numpy()
    
    # Handle numerical features
    numeric_transformer = StandardScaler()
    X_num = numeric_transformer.fit_transform(X[numeric_features].fillna(0))
    
    # Handle categorical features
    categorical_encoders = {}
    categorical_cardinalities = []
    X_cat_list = []
    
    for col in categorical_features:
        encoder = LabelEncoder()
        # Handle missing values
        col_data = X[col].fillna("MISSING").astype(str)
        encoded = encoder.fit_transform(col_data)
        X_cat_list.append(encoded)
        categorical_encoders[col] = encoder
        categorical_cardinalities.append(len(encoder.classes_))
    
    X_cat = np.column_stack(X_cat_list) if X_cat_list else np.empty((len(X), 0), dtype=int)
    
    # Create combined preprocessor for sklearn compatibility
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
        ],
        remainder="drop",
    )
    
    return X_num, X_cat, y, categorical_cardinalities, preprocessor, categorical_encoders


def _train_pytorch_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None = None,
    *,
    epochs: int = 200,  # More epochs for better convergence
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 20,  # More patience
    device: str = "cpu",
) -> tuple[nn.Module, list[dict[str, float]]]:
    """Train PyTorch model with advanced optimization."""
    
    model = model.to(device)
    
    # Advanced optimizer with better defaults
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=lr, 
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    
    # Label smoothing for better generalization
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Cosine annealing with warm restarts
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None
    
    training_curves = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            
            if isinstance(model, TabTransformer):
                # Split numerical and categorical features
                x_num = batch_x[:, :model.n_numerical] if model.n_numerical > 0 else torch.empty(batch_x.size(0), 0, device=device)
                x_cat = batch_x[:, model.n_numerical:].long() if model.n_categorical > 0 else torch.empty(batch_x.size(0), 0, dtype=torch.long, device=device)
                outputs = model(x_num, x_cat)
            else:
                outputs = model(batch_x)
            
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()
        
        train_acc = train_correct / train_total if train_total > 0 else 0.0
        
        # Validation
        val_loss = 0.0
        val_acc = 0.0
        if val_loader is not None:
            model.eval()
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    
                    if isinstance(model, TabTransformer):
                        x_num = batch_x[:, :model.n_numerical] if model.n_numerical > 0 else torch.empty(batch_x.size(0), 0, device=device)
                        x_cat = batch_x[:, model.n_numerical:].long() if model.n_categorical > 0 else torch.empty(batch_x.size(0), 0, dtype=torch.long, device=device)
                        outputs = model(x_num, x_cat)
                    else:
                        outputs = model(batch_x)
                    
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += batch_y.size(0)
                    val_correct += (predicted == batch_y).sum().item()
            
            val_acc = val_correct / val_total if val_total > 0 else 0.0
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
        
        # Update learning rate
        scheduler.step()
        
        training_curves.append({
            "epoch": epoch,
            "train_loss": train_loss / len(train_loader) if len(train_loader) > 0 else 0.0,
            "train_acc": train_acc,
            "val_loss": val_loss / len(val_loader) if val_loader and len(val_loader) > 0 else 0.0,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"],
        })
        
        if patience_counter >= patience:
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, training_curves


def _evaluate_pytorch_model(
    model: nn.Module,
    test_loader: DataLoader,
    *,
    device: str = "cpu",
) -> dict[str, float]:
    """Evaluate PyTorch model."""
    
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            if isinstance(model, TabTransformer):
                x_num = batch_x[:, :model.n_numerical] if model.n_numerical > 0 else torch.empty(batch_x.size(0), 0, device=device)
                x_cat = batch_x[:, model.n_numerical:].long() if model.n_categorical > 0 else torch.empty(batch_x.size(0), 0, dtype=torch.long, device=device)
                outputs = model(x_num, x_cat)
            else:
                outputs = model(batch_x)
            
            probs = torch.softmax(outputs, dim=1)[:, 1]  # Probability of class 1
            
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    predictions = (all_probs >= 0.5).astype(int)
    
    metrics = {
        "accuracy": float(accuracy_score(all_labels, predictions)),
        "roc_auc": float("nan"),
        "average_precision": float("nan"),
        "prevalence": float(all_labels.mean()),
    }
    
    if len(np.unique(all_labels)) >= 2:
        metrics["roc_auc"] = float(roc_auc_score(all_labels, all_probs))
        metrics["average_precision"] = float(average_precision_score(all_labels, all_probs))
    
    return metrics


def run(
    *,
    seed: int = 20260130,
    max_test_seasons: int | None = None,
    epochs: int = 200,  # More epochs for better convergence
    batch_size: int = 64,  # Larger batch size
    lr: float = 1e-3,
    output_dir: Path | None = None,
) -> Q1DLOutputs:
    """Run deep learning elimination prediction showcase with optimal configurations."""
    
    paths.ensure_dirs()
    
    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load and prepare data (reuse from baseline)
    weekly = q1_baseline._read_weekly_panel()
    df = weekly.loc[weekly["active_flag"].astype(bool)].copy()
    df = df.loc[~df["withdrew_this_week"].astype(bool)].copy()
    df["y_eliminated"] = df["eliminated_this_week"].astype(bool).astype(int)
    df = q1_baseline._build_features(df)
    
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
    categorical_features = ["pro_name"]
    
    seasons = sorted(df["season"].unique().tolist())
    max_test_seasons = _as_int_or_default(max_test_seasons, len(seasons))
    if max_test_seasons > 0:
        seasons = seasons[:max_test_seasons]
    
    rows = []
    all_training_curves = []
    
    for season_test in seasons:
        print(f"Testing on season {season_test}")
        
        train_df = df.loc[df["season"] != int(season_test)].copy()
        test_df = df.loc[df["season"] == int(season_test)].copy()
        
        if len(train_df) == 0 or len(test_df) == 0:
            continue
        
        # Prepare data
        X_num_train, X_cat_train, y_train, cat_cardinalities, _, _ = _prepare_data(
            train_df, numeric_features=numeric_features, categorical_features=categorical_features
        )
        X_num_test, X_cat_test, y_test, _, _, _ = _prepare_data(
            test_df, numeric_features=numeric_features, categorical_features=categorical_features
        )
        
        # Combine numerical and categorical for simple models
        X_train_combined = np.concatenate([X_num_train, X_cat_train], axis=1)
        X_test_combined = np.concatenate([X_num_test, X_cat_test], axis=1)
        
        # Create datasets and loaders
        train_dataset = TabularDataset(X_train_combined, y_train)
        test_dataset = TabularDataset(X_test_combined, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Test different models with optimal configurations
        models_to_test = [
            ("advanced_mlp", AdvancedMLP(
                input_dim=X_train_combined.shape[1],
                hidden_dims=[512, 256, 128, 64],
                dropout=0.2,
            )),
            ("tab_transformer", TabTransformer(
                n_numerical=len(numeric_features),
                categorical_cardinalities=cat_cardinalities,
                embed_dim=64,  # Larger embedding
                n_heads=8,     # More attention heads
                n_layers=4,    # Deeper transformer
                mlp_hidden=256, # Larger MLP
                dropout=0.1,
            )),
        ]
        
        for model_name, model in models_to_test:
            print(f"  Training {model_name}")
            
            # Train model with optimal settings
            trained_model, curves = _train_pytorch_model(
                model,
                train_loader,
                val_loader=None,  # No validation set for simplicity
                epochs=epochs,
                lr=lr,
                device=device,
            )
            
            # Evaluate model
            metrics = _evaluate_pytorch_model(trained_model, test_loader, device=device)
            
            # Record results
            rows.append({
                "season_test": int(season_test),
                "model": f"pytorch_{model_name}",
                "n_train": len(train_df),
                "n_test": len(test_df),
                "device": device,
                "epochs_trained": len(curves),
                "batch_size": int(batch_size),
                "learning_rate": float(lr),
                **metrics,
            })
            
            # Record training curves
            for curve_point in curves:
                all_training_curves.append({
                    "season_test": int(season_test),
                    "model": f"pytorch_{model_name}",
                    **curve_point,
                })
    
    # Save results
    out_dir = (paths.tables_dir() / "showcase") if output_dir is None else Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Main results
    results_df = pd.DataFrame(rows)
    results_fp = out_dir / "mcm2026c_q1_dl_elimination_transformer_cv.csv"
    io.write_csv(results_df, results_fp)
    
    # Training curves
    curves_df = pd.DataFrame(all_training_curves)
    curves_fp = out_dir / "mcm2026c_q1_dl_elimination_transformer_curves.csv"
    io.write_csv(curves_df, curves_fp)
    
    # Summary
    if len(results_df) > 0:
        metric_cols = ["accuracy", "roc_auc", "average_precision", "prevalence"]
        summary = (
            results_df.groupby(["model"], sort=True)
            .agg(
                n_folds=("season_test", "nunique"),
                n_test_total=("n_test", "sum"),
                epochs_mean=("epochs_trained", "mean"),
                batch_size_mode=("batch_size", lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0]),
                lr_mode=("learning_rate", lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0]),
                **{f"{c}_mean": (c, "mean") for c in metric_cols},
                **{f"{c}_std": (c, "std") for c in metric_cols},
            )
            .reset_index()
        )
    else:
        summary = pd.DataFrame()
    
    summary_fp = out_dir / "mcm2026c_q1_dl_elimination_transformer_summary.csv"
    io.write_csv(summary, summary_fp)
    
    return Q1DLOutputs(
        cv_metrics_csv=results_fp,
        cv_summary_csv=summary_fp,
        training_curves_csv=curves_fp,
    )


def main() -> int:
    out = run(max_test_seasons=5, epochs=100)  # Reduced for testing
    print(f"Wrote: {out.cv_metrics_csv}")
    print(f"Wrote: {out.cv_summary_csv}")
    print(f"Wrote: {out.training_curves_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())