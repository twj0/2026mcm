from __future__ import annotations

# Ref: docs/spec/task.md
# Ref: docs/spec/architecture.md

from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from mcm2026.data import io
from mcm2026.pipelines import mcm2026c_q4_design_space_eval as q4_main


@dataclass(frozen=True)
class Q4SensitivityOutputs:
    grid_csv: Path


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


def _as_list_of_float_lists(x: object, default: list[list[float]]) -> list[list[float]]:
    if isinstance(x, (list, tuple)):
        out: list[list[float]] = []
        for v in x:
            if isinstance(v, (list, tuple)):
                tmp: list[float] = []
                for y in v:
                    try:
                        tmp.append(float(y))
                    except Exception:
                        continue
                if tmp:
                    out.append(tmp)
        return out if out else [list(v) for v in default]
    return [list(v) for v in default]


def _as_str_list(x: object, default: list[str]) -> list[str]:
    if isinstance(x, (list, tuple)):
        out = [str(v) for v in x]
        return out if out else list(default)
    return list(default)


def _as_int_list_or_none(x: object) -> list[int] | None:
    if x is None:
        return None
    if isinstance(x, (list, tuple)):
        out: list[int] = []
        for v in x:
            try:
                out.append(int(v))
            except Exception:
                continue
        return out if out else None
    return None


def run(
    *,
    seed: int = 20260130,
    output_dir: Path | None = None,
    alpha_grid: object = None,
    n_sims_grid: object = None,
    outlier_mults_grid: object = None,
    mechanisms: object = None,
    seasons: object = None,
    max_runs: int | None = None,
) -> Q4SensitivityOutputs:
    out_dir = Path("outputs/tables/showcase") if output_dir is None else Path(output_dir)

    alphas = _as_float_list(alpha_grid, [0.3, 0.5, 0.7])
    n_sims_list = _as_int_list(n_sims_grid, [10, 20, 50])
    outlier_sets = _as_list_of_float_lists(outlier_mults_grid, [[2.0], [2.0, 5.0], [2.0, 5.0, 10.0]])

    mech_list = _as_str_list(mechanisms, ["percent", "rank", "percent_judge_save", "percent_sqrt", "percent_log", "percent_cap", "dynamic_weight"]) if mechanisms is not None else None
    season_list = _as_int_list_or_none(seasons)

    combos = list(product(alphas, n_sims_list, range(len(outlier_sets))))
    if max_runs is not None and max_runs > 0:
        combos = combos[: int(max_runs)]

    tmp = out_dir / "_tmp_q4_metrics.csv"

    rows: list[dict] = []
    for alpha, n_sims, idx in combos:
        outlier_mults = outlier_sets[int(idx)]

        out = q4_main.run(
            seed=int(seed),
            alpha=float(alpha),
            n_sims=int(n_sims),
            outlier_mults=list(outlier_mults),
            mechanisms=mech_list,
            seasons=season_list,
            output_path=tmp,
        )

        df = io.read_table(out.new_system_metrics_csv)

        for mech, g in df.groupby("mechanism", sort=True):
            rows.append(
                {
                    "seed": int(seed),
                    "alpha": float(alpha),
                    "n_sims": int(n_sims),
                    "outlier_mults": "+".join([str(float(x)) for x in outlier_mults]),
                    "mechanism": str(mech),
                    "n_rows": int(len(g)),
                    "champion_mode_prob_mean": float(pd.to_numeric(g["champion_mode_prob"], errors="coerce").mean()),
                    "champion_entropy_mean": float(pd.to_numeric(g["champion_entropy"], errors="coerce").mean()),
                    "tpi_season_avg_mean": float(pd.to_numeric(g["tpi_season_avg"], errors="coerce").mean()),
                    "fan_vs_uniform_contrast_mean": float(
                        pd.to_numeric(g["fan_vs_uniform_contrast"], errors="coerce").mean()
                    ),
                    "robust_fail_rate_mean": float(pd.to_numeric(g["robust_fail_rate"], errors="coerce").mean()),
                }
            )

    out_df = pd.DataFrame(rows)
    out_fp = out_dir / "mcm2026c_showcase_q4_sensitivity_grid.csv"
    io.write_csv(out_df, out_fp)

    return Q4SensitivityOutputs(grid_csv=out_fp)


def main() -> int:
    out = run(max_runs=6)
    print(f"Wrote: {out.grid_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
