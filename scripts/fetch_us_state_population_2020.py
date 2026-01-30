"""Fetch U.S. state population data (2020 Decennial Census, PL 94-171) via Census Data API.

This script downloads the latest available 2020 total population by state and saves it to
`data/raw/` for downstream feature engineering (e.g., mapping `celebrity_homestate` to a
population proxy).

Data source:
- https://api.census.gov/data/2020/dec/pl.html
- API request used: https://api.census.gov/data/2020/dec/pl?get=NAME,P1_001N&for=state:*

Note:
- This script intentionally uses `requests` as requested.
- If `requests` is not installed in your environment, install it first (e.g. `pip install requests`).
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

API_URL = "https://api.census.gov/data/2020/dec/pl?get=NAME,P1_001N&for=state:*"
CENSUS_HEADERS = {
    "User-Agent": "2026mcm-mcm2026/0.1 (MCM modeling; requests)",
    "Accept": "application/json",
}
DEFAULT_OUTPUT_FILENAME = "us_census_2020_state_population.csv"
DEFAULT_META_FILENAME = "us_census_2020_state_population.meta.json"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def fetch_state_population(timeout_seconds: float, max_retries: int = 3) -> list[dict[str, Any]]:
    """Fetch population by state from Census Data API.

    Returns a list of rows like:
      {"NAME": "California", "P1_001N": 39538223, "state": "06"}

    Key detail:
    - `P1_001N` is total population (2020).
    - `state` is a 2-digit FIPS code (string).
    """

    try:
        import requests
    except ModuleNotFoundError as e:  # pragma: no cover
        raise ModuleNotFoundError(
            "Missing dependency 'requests'. Install it first, e.g. `pip install requests`."
        ) from e

    session = requests.Session()
    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = session.get(API_URL, timeout=timeout_seconds, headers=CENSUS_HEADERS)
            resp.raise_for_status()

            payload = resp.json()
            if not isinstance(payload, list) or len(payload) < 2:
                raise ValueError("Unexpected Census API payload shape")

            # Expected shape: [ [header...], [row...], [row...], ... ]
            header = payload[0]
            rows = payload[1:]
            if not isinstance(header, list) or any(k not in header for k in ["NAME", "P1_001N", "state"]):
                raise ValueError("Unexpected Census API header")

            out: list[dict[str, Any]] = []
            for r in rows:
                row = dict(zip(header, r, strict=True))
                row["P1_001N"] = int(row["P1_001N"])
                row["state"] = str(row["state"]).zfill(2)
                out.append(row)

            return out
        except Exception as err:  # keep broad: network, JSON, schema, etc.
            last_err = err
            if attempt < max_retries:
                # small backoff to reduce hammering the endpoint
                time.sleep(0.8 * attempt)
                continue
            raise

    # Unreachable, but keeps type-checkers happy.
    if last_err is not None:
        raise last_err
    return []


def write_outputs(rows: list[dict[str, Any]], output_dir: Path, overwrite: bool) -> tuple[Path, Path]:
    """Write CSV + a small metadata JSON file into output_dir."""

    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / DEFAULT_OUTPUT_FILENAME
    meta_path = output_dir / DEFAULT_META_FILENAME

    if not overwrite and (csv_path.exists() or meta_path.exists()):
        raise FileExistsError(
            f"Output already exists. Use --overwrite to replace. Existing: {csv_path} / {meta_path}"
        )

    # Keep columns explicit for stable downstream merges.
    # - NAME matches the homestate names used in many datasets.
    # - state is the 2-digit FIPS code.
    # - P1_001N is the 2020 total population.
    import pandas as pd

    df = pd.DataFrame(rows)[["NAME", "state", "P1_001N"]]
    df["state"] = df["state"].astype(str).str.zfill(2)
    df["P1_001N"] = df["P1_001N"].astype(int)
    df = df.sort_values(["NAME"]).reset_index(drop=True)

    df.to_csv(csv_path, index=False, encoding="utf-8")

    meta = {
        "source": "U.S. Census Bureau - Census Data API",
        "dataset": "2020 Decennial Census: Redistricting Data (PL 94-171)",
        "api_url": API_URL,
        "fetched_at_utc": _utc_now_iso(),
        "fields": {"NAME": "State name", "state": "State FIPS", "P1_001N": "Total population"},
        "notes": "Use NAME to merge with celebrity_homestate when homecountry/region == United States.",
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return csv_path, meta_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch US state population (2020 Census, PL 94-171) and save to data/raw."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory. Default: <repo_root>/data/raw",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="HTTP timeout in seconds (default: 30).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max retries for transient network/API errors (default: 3).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files if present.",
    )

    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    output_dir = Path(args.output_dir) if args.output_dir else (repo_root / "data" / "raw")

    rows = fetch_state_population(timeout_seconds=args.timeout, max_retries=args.max_retries)
    csv_path, meta_path = write_outputs(rows=rows, output_dir=output_dir, overwrite=args.overwrite)

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {meta_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
