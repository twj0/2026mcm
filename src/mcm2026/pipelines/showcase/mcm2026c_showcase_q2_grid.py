from __future__ import annotations

# Ref: docs/spec/task.md
# Ref: docs/spec/architecture.md

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from mcm2026.data import io
from mcm2026.pipelines import mcm2026c_q2_counterfactual_simulation as q2_main


@dataclass(frozen=True)
class Q2GridOutputs:
    grid_csv: Path


def _as_str_list(x: object, default: list[str]) -> list[str]:
    if isinstance(x, (list, tuple)):
        out = [str(v) for v in x]
        return out if out else list(default)
    return list(default)


def _as_bool_list(x: object, default: list[bool]) -> list[bool]:
    if isinstance(x, (list, tuple)):
        out: list[bool] = []
        for v in x:
            out.append(bool(v))
        return out if out else list(default)
    return list(default)


def run(
    *,
    seed: int = 20260130,
    output_dir: Path | None = None,
    fan_source_mechanism_grid: object = None,
    count_withdraw_as_exit_grid: object = None,
) -> Q2GridOutputs:
    out_dir = Path("outputs/tables/showcase") if output_dir is None else Path(output_dir)

    mechs = _as_str_list(fan_source_mechanism_grid, ["percent", "rank"])
    mechs = [m for m in mechs if m in {"percent", "rank"}] or ["percent"]

    withdraws = _as_bool_list(count_withdraw_as_exit_grid, [True, False])

    alpha = q2_main._get_alpha_from_config()

    rows: list[pd.DataFrame] = []
    tmp = out_dir / "_tmp_q2_grid.csv"

    for mech in mechs:
        for count_withdraw in withdraws:
            out = q2_main.run(
                alpha=float(alpha),
                fan_source_mechanism=str(mech),
                count_withdraw_as_exit=bool(count_withdraw),
                output_path=tmp,
            )
            df = io.read_table(out.mechanism_comparison_csv)
            df["seed"] = int(seed)
            rows.append(df)

    grid = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame([])
    out_fp = out_dir / "mcm2026c_showcase_q2_grid.csv"
    io.write_csv(grid, out_fp)

    return Q2GridOutputs(grid_csv=out_fp)


def main() -> int:
    out = run()
    print(f"Wrote: {out.grid_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
