from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DlFitResult:
    model: object
    metrics: dict


def fit_mlp_regression(
    df: pd.DataFrame,
    target: str,
    features: list[str],
    epochs: int = 50,
    lr: float = 1e-3,
    hidden: int = 64,
) -> DlFitResult:
    try:
        import torch
        import torch.nn as nn
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "torch is not installed. Enable the dl group: uv sync --group dl"
        ) from e

    X = df[features].to_numpy(dtype=np.float32)
    y = df[target].to_numpy(dtype=np.float32).reshape(-1, 1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xt = torch.from_numpy(X).to(device)
    yt = torch.from_numpy(y).to(device)

    model = nn.Sequential(
        nn.Linear(xt.shape[1], hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, 1),
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(int(epochs)):
        opt.zero_grad(set_to_none=True)
        pred = model(xt)
        loss = loss_fn(pred, yt)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        pred = model(xt).detach().cpu().numpy().reshape(-1)

    rmse = float(np.sqrt(np.mean((pred - y.reshape(-1)) ** 2)))
    return DlFitResult(model=model, metrics={"rmse": rmse, "device": str(device)})
