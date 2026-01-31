"""测试基础依赖是否正确安装"""

import sys
from pathlib import Path

import pytest


def _version_tuple(version_str):
    """将版本字符串转换为元组以便比较"""
    return tuple(map(int, version_str.split(".")[:3]))


def test_core_dependencies():
    """测试核心科学计算依赖"""
    import numpy as np
    import pandas as pd
    import scipy
    from sklearn import __version__ as sklearn_version
    import statsmodels.api as sm

    # 验证版本
    assert _version_tuple(np.__version__) >= (1, 26, 0)
    assert _version_tuple(pd.__version__) >= (2, 1, 0)
    assert _version_tuple(scipy.__version__) >= (1, 10, 0)
    assert _version_tuple(sklearn_version) >= (1, 3, 0)


def test_visualization_dependencies():
    """测试可视化依赖"""
    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly

    assert _version_tuple(matplotlib.__version__) >= (3, 7, 0)
    assert _version_tuple(sns.__version__) >= (0, 13, 0)
    assert _version_tuple(plotly.__version__) >= (5, 14, 0)


def test_io_dependencies():
    """测试数据IO依赖"""
    import openpyxl
    import pyarrow
    import yaml

    assert _version_tuple(openpyxl.__version__) >= (3, 1, 0)
    # pyarrow版本可能是 "14.0.0" 或 "14.0.1"
    major_version = int(pyarrow.__version__.split(".")[0])
    assert major_version >= 14


def test_utility_dependencies():
    """测试工具依赖"""
    from tqdm import tqdm

    # 简单测试tqdm是否可用
    items = list(tqdm(range(10), disable=True))
    assert len(items) == 10


def test_mcm2026_package():
    """测试mcm2026包是否可导入"""
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    import mcm2026
    from mcm2026.core import paths
    from mcm2026.data import audit, io

    assert mcm2026.__name__ == "mcm2026"
    assert callable(paths.repo_root)
    assert callable(audit.audit_summary_dict)
    assert callable(io.read_table)
