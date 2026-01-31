"""测试项目结构和路径配置"""

import sys
from pathlib import Path

import pytest


@pytest.fixture
def repo_root():
    """获取仓库根目录"""
    return Path(__file__).resolve().parents[1]


def test_directory_structure(repo_root):
    """测试目录结构是否符合预期"""
    # 必须存在的目录
    required_dirs = [
        "src/mcm2026",
        "src/mcm2026/core",
        "src/mcm2026/data",
        "src/mcm2026/pipelines",
        "src/mcm2026/pipelines/showcase",
        "data",
        "outputs",
        "paper",
        "docs",
        "tests",
    ]

    for dir_path in required_dirs:
        full_path = repo_root / dir_path
        assert full_path.exists(), f"Required directory not found: {dir_path}"
        assert full_path.is_dir(), f"Path exists but is not a directory: {dir_path}"


def test_key_files_exist(repo_root):
    """测试关键文件是否存在"""
    required_files = [
        "pyproject.toml",
        ".python-version",
        "README.md",
        "run_all.py",
        "src/mcm2026/__init__.py",
        "src/mcm2026/core/paths.py",
        "src/mcm2026/data/io.py",
        "src/mcm2026/data/audit.py",
        "src/mcm2026/config/config.yaml",
        "CLAUDE.md",
    ]

    for file_path in required_files:
        full_path = repo_root / file_path
        assert full_path.exists(), f"Required file not found: {file_path}"
        assert full_path.is_file(), f"Path exists but is not a file: {file_path}"


def test_paths_module(repo_root):
    """测试paths模块功能"""
    sys.path.insert(0, str(repo_root / "src"))

    from mcm2026.core import paths

    # 测试路径函数
    assert paths.repo_root().exists()
    assert paths.data_dir() == paths.repo_root() / "data"
    assert paths.raw_data_dir() == paths.repo_root() / "data" / "raw"
    assert paths.processed_data_dir() == paths.repo_root() / "data" / "processed"
    assert paths.outputs_dir() == paths.repo_root() / "outputs"
    assert paths.figures_dir() == paths.repo_root() / "outputs" / "figures"
    assert paths.tables_dir() == paths.repo_root() / "outputs" / "tables"
    assert paths.predictions_dir() == paths.repo_root() / "outputs" / "predictions"


def test_ensure_dirs(repo_root, tmp_path):
    """测试目录创建功能"""
    sys.path.insert(0, str(repo_root / "src"))

    from mcm2026.core import paths

    # 在实际环境中测试（应该不会报错）
    paths.ensure_dirs()

    # 验证目录已创建
    assert paths.processed_data_dir().exists()
    assert paths.figures_dir().exists()
    assert paths.tables_dir().exists()
    assert paths.predictions_dir().exists()


def test_python_version(repo_root):
    """测试Python版本是否符合要求"""
    python_version_file = repo_root / ".python-version"
    assert python_version_file.exists()

    version = python_version_file.read_text().strip()
    assert version == "3.11", f"Expected Python 3.11, got {version}"

    # 检查当前运行的Python版本
    import sys

    current_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    assert current_version == "3.11", f"Running Python {current_version}, expected 3.11"


def test_config_file(repo_root):
    """测试配置文件是否可读取"""
    sys.path.insert(0, str(repo_root / "src"))

    import yaml

    config_path = repo_root / "src" / "mcm2026" / "config" / "config.yaml"
    assert config_path.exists()

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 验证关键配置项
    assert "dwts" in config
    assert "q1" in config["dwts"]
    assert "q2" in config["dwts"]
    assert "q3" in config["dwts"]
    assert "q4" in config["dwts"]
    assert "showcase" in config

    # 验证Q1关键参数
    q1_config = config["dwts"]["q1"]
    assert "alpha" in q1_config
    assert "tau" in q1_config
    assert "prior_draws_m" in q1_config
    assert "posterior_resample_r" in q1_config

    print(f"\nConfig loaded successfully:")
    print(f"  alpha: {q1_config['alpha']}")
    print(f"  tau: {q1_config['tau']}")
    print(f"  prior_draws_m: {q1_config['prior_draws_m']}")
    print(f"  showcase.enabled: {config['showcase']['enabled']}")
