"""测试系统信息和环境配置"""

import platform
import sys

import pytest


def test_python_version():
    """测试Python版本"""
    version_info = sys.version_info
    print(f"\nPython version: {version_info.major}.{version_info.minor}.{version_info.micro}")
    print(f"Python implementation: {platform.python_implementation()}")
    print(f"Python compiler: {platform.python_compiler()}")

    assert version_info.major == 3
    assert version_info.minor == 11


def test_platform_info():
    """测试平台信息"""
    print(f"\nPlatform: {platform.platform()}")
    print(f"System: {platform.system()}")
    print(f"Release: {platform.release()}")
    print(f"Machine: {platform.machine()}")
    print(f"Processor: {platform.processor()}")


def test_gpu_info():
    """测试GPU信息（如果可用）"""
    try:
        import torch
    except ImportError:
        pytest.skip("PyTorch not installed")

    if not torch.cuda.is_available():
        print("\nNo CUDA GPU detected")
        return

    print(f"\n{'='*60}")
    print("GPU Information")
    print(f"{'='*60}")

    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")

    for i in range(torch.cuda.device_count()):
        print(f"\nGPU {i}:")
        print(f"  Name: {torch.cuda.get_device_name(i)}")

        props = torch.cuda.get_device_properties(i)
        print(f"  Compute capability: {props.major}.{props.minor}")
        print(f"  Total memory: {props.total_memory / 1024**3:.2f} GB")
        print(f"  Multi-processor count: {props.multi_processor_count}")

        # 内存使用情况
        if torch.cuda.is_available():
            mem_allocated = torch.cuda.memory_allocated(i) / 1024**3
            mem_reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"  Memory allocated: {mem_allocated:.2f} GB")
            print(f"  Memory reserved: {mem_reserved:.2f} GB")


def test_numpy_blas_info():
    """测试NumPy BLAS配置"""
    import numpy as np

    print(f"\nNumPy version: {np.__version__}")
    print(f"NumPy configuration:")

    try:
        config = np.__config__.show()
        print(config)
    except Exception:
        print("  (BLAS info not available)")


def test_package_versions():
    """打印所有关键包的版本"""
    packages = [
        "numpy",
        "pandas",
        "scipy",
        "sklearn",
        "statsmodels",
        "matplotlib",
        "seaborn",
        "plotly",
        "torch",
        "xgboost",
    ]

    print(f"\n{'='*60}")
    print("Package Versions")
    print(f"{'='*60}")

    for pkg_name in packages:
        try:
            if pkg_name == "sklearn":
                import sklearn

                version = sklearn.__version__
            else:
                pkg = __import__(pkg_name)
                version = pkg.__version__
            print(f"{pkg_name:20s} {version}")
        except ImportError:
            print(f"{pkg_name:20s} (not installed)")
        except AttributeError:
            print(f"{pkg_name:20s} (version unknown)")


def test_memory_info():
    """测试系统内存信息"""
    try:
        import psutil
    except ImportError:
        pytest.skip("psutil not installed")

    mem = psutil.virtual_memory()
    print(f"\nSystem Memory:")
    print(f"  Total: {mem.total / 1024**3:.2f} GB")
    print(f"  Available: {mem.available / 1024**3:.2f} GB")
    print(f"  Used: {mem.used / 1024**3:.2f} GB")
    print(f"  Percentage: {mem.percent}%")


def test_cpu_info():
    """测试CPU信息"""
    try:
        import psutil
    except ImportError:
        pytest.skip("psutil not installed")

    print(f"\nCPU Information:")
    print(f"  Physical cores: {psutil.cpu_count(logical=False)}")
    print(f"  Logical cores: {psutil.cpu_count(logical=True)}")
    print(f"  CPU frequency: {psutil.cpu_freq().current:.2f} MHz")
