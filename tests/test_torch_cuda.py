"""测试PyTorch和CUDA配置"""

import sys
import platform

import pytest


def test_torch_import():
    """测试PyTorch是否可导入"""
    try:
        import torch
    except ImportError:
        pytest.skip("PyTorch not installed (optional dependency)")

    assert torch.__version__ is not None
    print(f"\nPyTorch version: {torch.__version__}")


def test_torch_cuda_availability():
    """测试CUDA是否可用"""
    try:
        import torch
    except ImportError:
        pytest.skip("PyTorch not installed (optional dependency)")

    cuda_available = torch.cuda.is_available()
    print(f"\nCUDA available: {cuda_available}")

    if cuda_available:
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"  - Compute capability: {props.major}.{props.minor}")
            print(f"  - Total memory: {props.total_memory / 1024**3:.2f} GB")
    else:
        # 在非CUDA环境下，应该使用CPU版本
        if platform.system() != "Darwin":  # 非macOS
            print("Warning: CUDA not available, using CPU version")


def test_torch_basic_operations():
    """测试PyTorch基本操作"""
    try:
        import torch
    except ImportError:
        pytest.skip("PyTorch not installed (optional dependency)")

    # CPU测试
    x = torch.randn(3, 3)
    y = torch.randn(3, 3)
    z = x + y
    assert z.shape == (3, 3)

    # 如果CUDA可用，测试GPU操作
    if torch.cuda.is_available():
        x_gpu = x.cuda()
        y_gpu = y.cuda()
        z_gpu = x_gpu + y_gpu
        assert z_gpu.shape == (3, 3)
        assert z_gpu.is_cuda

        # 测试CPU-GPU数据传输
        z_cpu = z_gpu.cpu()
        assert not z_cpu.is_cuda


def test_torch_cuda_version_match():
    """测试PyTorch CUDA版本是否匹配预期"""
    try:
        import torch
    except ImportError:
        pytest.skip("PyTorch not installed (optional dependency)")

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    cuda_version = torch.version.cuda
    print(f"\nPyTorch CUDA version: {cuda_version}")

    # 根据pyproject.toml，非macOS应该使用CUDA 12.4
    if platform.system() != "Darwin":
        # 检查是否是CUDA 12.x版本
        if cuda_version:
            major_version = int(cuda_version.split(".")[0])
            assert major_version == 12, f"Expected CUDA 12.x, got {cuda_version}"
            print(f"✓ CUDA version matches expected (12.x)")


def test_torch_device_selection():
    """测试设备选择逻辑"""
    try:
        import torch
    except ImportError:
        pytest.skip("PyTorch not installed (optional dependency)")

    # 测试自动设备选择
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nSelected device: {device}")

    x = torch.randn(10, 10, device=device)
    assert x.device.type == device.type

    # 测试多GPU情况
    if torch.cuda.device_count() > 1:
        print(f"Multiple GPUs detected: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            device_i = torch.device(f"cuda:{i}")
            x_i = torch.randn(10, 10, device=device_i)
            assert x_i.device.index == i
