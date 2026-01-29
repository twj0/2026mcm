from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


def _bootstrap_imports() -> None:
    repo_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(repo_root / "src"))


def _list_raw_files(raw_dir: Path) -> list[Path]:
    if not raw_dir.exists():
        return []

    exts = {".csv", ".tsv", ".txt", ".xlsx", ".xls"}
    return sorted([p for p in raw_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])


def main() -> int:
    _bootstrap_imports()

    from mcm2026.core import paths
    from mcm2026.data import audit, io

    paths.ensure_dirs()
    paths.raw_data_dir().mkdir(parents=True, exist_ok=True)

    raw_files = _list_raw_files(paths.raw_data_dir())
    if not raw_files:
        print("Repo ready. Put attachments under data/raw/ (csv/tsv/xlsx), then rerun run_all.py.")
        return 0

    summary_rows: list[dict] = []
    for fp in raw_files:
        try:
            df = io.read_table(fp)
        except Exception as e:
            summary_rows.append({"file": fp.name, "error": str(e)})
            continue

        s = audit.audit_summary_dict(df)
        summary_rows.append({"file": fp.name, **s})

        col_audit = audit.audit_columns(df)
        io.write_csv(col_audit, paths.tables_dir() / f"raw_{fp.stem}_columns.csv")

    summary = pd.DataFrame(summary_rows)
    io.write_csv(summary, paths.tables_dir() / "raw_audit_summary.csv")

    print(f"Audited {len(raw_files)} raw file(s). See outputs/tables/raw_audit_summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
