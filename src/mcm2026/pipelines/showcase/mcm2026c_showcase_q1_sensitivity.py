from __future__ import annotations

# Ref: docs/spec/task.md
# Ref: docs/spec/architecture.md

from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from mcm2026.data import io
from mcm2026.pipelines import mcm2026c_q1_smc_fan_vote as q1_main


@dataclass(frozen=True)
class Q1SensitivityOutputs:
    sensitivity_summary_csv: Path


def _as_float_list(x: object, default: list[float]) -> list[float]:
    if isinstance(x, (list, tuple)):
        out: list[float] = []
        for v in x:
            try:
                out.append(float(v))
            except Exception:
                continue
        return out if out else list(default)
    return list(default)


def _as_int_list(x: object, default: list[int]) -> list[int]:
    if isinstance(x, (list, tuple)):
        out: list[int] = []
        for v in x:
            try:
                out.append(int(v))
            except Exception:
                continue
        return out if out else list(default)
    return list(default)


def run(
    *,
    seed: int = 20260130,
    output_dir: Path | None = None,
    alpha_grid: object = None,
    tau_grid: object = None,
    prior_draws_m_grid: object = None,
    posterior_resample_r_grid: object = None,
    max_runs: int | None = None,
) -> Q1SensitivityOutputs:
    out_dir = Path("outputs/tables/showcase") if output_dir is None else Path(output_dir)

    alphas = _as_float_list(alpha_grid, [0.3, 0.5, 0.7])
    taus = _as_float_list(tau_grid, [0.02, 0.03, 0.05])
    ms = _as_int_list(prior_draws_m_grid, [1000, 2000])
    rs = _as_int_list(posterior_resample_r_grid, [300, 500])

    tmp_pred = out_dir / "_tmp_q1_posterior.csv"
    tmp_unc = out_dir / "_tmp_q1_uncertainty.csv"

    rows: list[dict] = []
    combos = list(product(alphas, taus, ms, rs))
    if max_runs is not None and max_runs > 0:
        combos = combos[: int(max_runs)]

    for alpha, tau, m, r in combos:
        out = q1_main.run(
            alpha=float(alpha),
            tau=float(tau),
            prior_draws_m=int(m),
            posterior_resample_r=int(r),
            output_posterior_path=tmp_pred,
            output_uncertainty_path=tmp_unc,
        )

        unc = io.read_table(out.uncertainty_summary_csv)

        for mechanism, g in unc.groupby("mechanism", sort=True):
            ess_ratio = pd.to_numeric(g["ess_ratio"], errors="coerce")
            evidence = pd.to_numeric(g["evidence"], errors="coerce")

            rows.append(
                {
                    "seed": int(seed),
                    "alpha": float(alpha),
                    "tau": float(tau),
                    "prior_draws_m": int(m),
                    "posterior_resample_r": int(r),
                    "mechanism": str(mechanism),
                    "n_weeks": int(len(g)),
                    "ess_ratio_mean": float(ess_ratio.mean()),
                    "ess_ratio_min": float(ess_ratio.min()),
                    "ess_ratio_p10": float(np.nanquantile(ess_ratio.to_numpy(), 0.10)),
                    "ess_ratio_lt_0p1": float(np.mean(ess_ratio.to_numpy() < 0.1)),
                    "evidence_mean": float(evidence.mean()),
                    "evidence_p10": float(np.nanquantile(evidence.to_numpy(), 0.10)),
                }
            )

    df = pd.DataFrame(rows)
    out_fp = out_dir / "mcm2026c_showcase_q1_sensitivity_summary.csv"
    io.write_csv(df, out_fp)

    return Q1SensitivityOutputs(sensitivity_summary_csv=out_fp)


def main() -> int:
    out = run()
    print(f"Wrote: {out.sensitivity_summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
