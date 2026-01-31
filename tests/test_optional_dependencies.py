"""测试可选依赖组"""

import pytest


def _version_tuple(version_str):
    """将版本字符串转换为元组以便比较"""
    return tuple(map(int, version_str.split(".")[:3]))


def test_ml_dependencies():
    """测试机器学习依赖（ml组）"""
    try:
        import xgboost as xgb
    except ImportError:
        pytest.skip("XGBoost not installed (optional ml group)")

    assert _version_tuple(xgb.__version__) >= (2, 0, 0)
    print(f"\nXGBoost version: {xgb.__version__}")


def test_optimization_dependencies():
    """测试优化依赖（opt组）"""
    try:
        import cvxpy as cp
        import pulp
    except ImportError:
        pytest.skip("Optimization packages not installed (optional opt group)")

    print(f"\nCVXPY version: {cp.__version__}")
    print(f"PuLP version: {pulp.__version__}")

    # 简单测试cvxpy
    x = cp.Variable()
    prob = cp.Problem(cp.Minimize(x), [x >= 1])
    prob.solve()
    assert abs(x.value - 1.0) < 1e-6


def test_web_dependencies():
    """测试网络爬虫依赖（web组）"""
    try:
        import requests
        from bs4 import BeautifulSoup
        import bs4
        from pytrends.request import TrendReq
    except ImportError:
        pytest.skip("Web scraping packages not installed (optional web group)")

    print(f"\nRequests version: {requests.__version__}")
    print(f"BeautifulSoup4 version: {bs4.__version__}")


def test_notebook_dependencies():
    """测试Jupyter依赖（notebook组）"""
    try:
        import ipykernel
        import jupyterlab
    except ImportError:
        pytest.skip("Notebook packages not installed (optional notebook group)")

    print(f"\nipykernel version: {ipykernel.__version__}")
    print(f"jupyterlab version: {jupyterlab.__version__}")


def test_dev_dependencies():
    """测试开发依赖（dev组，默认安装）"""
    import pytest as pytest_module
    from IPython import __version__ as ipython_version

    try:
        import ruff
    except ImportError:
        # ruff可能作为命令行工具安装，不一定有Python包
        pytest.skip("Ruff not available as Python package")

    assert _version_tuple(pytest_module.__version__) >= (8, 0, 0)
    print(f"\nPytest version: {pytest_module.__version__}")
    print(f"IPython version: {ipython_version}")
