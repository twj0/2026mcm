from __future__ import annotations

# Ref: docs/spec/task.md
# Ref: docs/spec/architecture.md

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from mcm2026.data import io
from mcm2026.pipelines import mcm2026c_q3_mixed_effects_impacts as q3_main


@dataclass(frozen=True)
class Q3RefitGridOutputs:
    grid_csv: Path


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
    n_refits_grid: object = None,
    max_runs: int | None = None,
) -> Q3RefitGridOutputs:
    out_dir = Path("outputs/tables/showcase") if output_dir is None else Path(output_dir)

    n_refits_list = _as_int_list(n_refits_grid, [10, 15, 30, 50])
    n_refits_list = [n for n in n_refits_list if n > 0]
    if max_runs is not None and max_runs > 0:
        n_refits_list = n_refits_list[: int(max_runs)]

    tmp = out_dir / "_tmp_q3_coeffs.csv"

    rows: list[dict] = []
    for n_refits in n_refits_list:
        out = q3_main.run(n_refits=int(n_refits), seed=int(seed), output_path=tmp)
        df = io.read_table(out.impact_coeffs_csv)

        w = pd.to_numeric(df["ci_high"], errors="coerce") - pd.to_numeric(df["ci_low"], errors="coerce")
        w = w.astype(float)

        for outcome, g in df.groupby("outcome", sort=True):
            ww = w.loc[g.index]
            rows.append(
                {
                    "seed": int(seed),
                    "n_refits": int(n_refits),
                    "fan_source_mechanism": str(q3_main._get_q3_params_from_config()[0]),
                    "outcome": str(outcome),
                    "model": "|".join(sorted(set(g["model"].astype(str).tolist()))),
                    "n_terms": int(len(g)),
                    "ci_width_mean": float(np.nanmean(ww.to_numpy())),
                    "ci_width_median": float(np.nanmedian(ww.to_numpy())),
                    "ci_width_p90": float(np.nanquantile(ww.to_numpy(), 0.90)),
                }
            )

    out_df = pd.DataFrame(rows)
    out_fp = out_dir / "mcm2026c_showcase_q3_refit_grid.csv"
    io.write_csv(out_df, out_fp)

    return Q3RefitGridOutputs(grid_csv=out_fp)


def main() -> int:
    out = run()
    print(f"Wrote: {out.grid_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
