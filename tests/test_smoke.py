import sys
from pathlib import Path


def test_import():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    import mcm2026

    assert mcm2026.__name__ == "mcm2026"
