---
# Tests
---
这个目录包含了用于验证项目配置和依赖的测试套件。

## 测试文件说明

### 1. `test_dependencies.py`
测试基础依赖是否正确安装：
- 核心科学计算库（numpy, pandas, scipy, scikit-learn, statsmodels）
- 可视化库（matplotlib, seaborn, plotly）
- 数据IO库（openpyxl, pyarrow, pyyaml）
- 工具库（tqdm）
- mcm2026包本身

### 2. `test_optional_dependencies.py`
测试可选依赖组：
- `ml` 组：XGBoost
- `opt` 组：CVXPY, PuLP
- `web` 组：requests, BeautifulSoup, pytrends
- `notebook` 组：ipykernel, jupyterlab
- `dev` 组：pytest, IPython, ruff

### 3. `test_torch_cuda.py`
测试PyTorch和CUDA配置：
- PyTorch是否可导入
- CUDA是否可用
- CUDA版本是否匹配（预期CUDA 12.x）
- GPU信息（数量、名称、显存）
- 基本张量操作（CPU和GPU）
- 设备选择逻辑

### 4. `test_project_structure.py`
测试项目结构和路径配置：
- 目录结构是否完整
- 关键文件是否存在
- `paths`模块功能
- Python版本检查
- 配置文件读取

### 5. `test_system_info.py`
测试系统信息和环境：
- Python版本和平台信息
- GPU详细信息（如果可用）
- NumPy BLAS配置
- 所有关键包的版本
- 系统内存和CPU信息（需要psutil）

## 运行测试

### 运行所有测试
```bash
uv run pytest
```

### 运行特定测试文件
```bash
# 测试基础依赖
uv run pytest tests/test_dependencies.py -v

# 测试PyTorch和CUDA
uv run pytest tests/test_torch_cuda.py -v

# 测试项目结构
uv run pytest tests/test_project_structure.py -v

# 测试系统信息（显示详细输出）
uv run pytest tests/test_system_info.py -v -s
```

### 运行特定测试函数
```bash
# 只测试GPU信息
uv run pytest tests/test_system_info.py::test_gpu_info -v -s

# 只测试CUDA可用性
uv run pytest tests/test_torch_cuda.py::test_torch_cuda_availability -v -s
```

### 显示打印输出
```bash
# -s 参数会显示print输出
uv run pytest tests/test_system_info.py -v -s
```

### 跳过可选依赖测试
如果你没有安装某些可选依赖，pytest会自动跳过相关测试。例如：
```bash
# 如果没有安装PyTorch，相关测试会被跳过
uv run pytest tests/test_torch_cuda.py -v
```

## 测试覆盖率

```bash
# 生成覆盖率报告
uv run pytest --cov=mcm2026 --cov-report=html

# 查看报告
# 打开 htmlcov/index.html
```

## 常见问题

### PyTorch CUDA测试失败
如果你在没有GPU的机器上运行，CUDA相关测试会被跳过或显示警告，这是正常的。

### 可选依赖测试被跳过
如果你没有安装某个可选依赖组，相关测试会被跳过。要安装可选依赖：
```bash
uv sync --group dl    # PyTorch
uv sync --group ml    # XGBoost
uv sync --group opt   # 优化库
uv sync --group web   # 网络爬虫
```

### 系统信息测试需要psutil
如果要运行完整的系统信息测试，需要安装psutil：
```bash
uv add --dev psutil
```

## CI/CD集成

这些测试可以集成到CI/CD流程中：
```yaml
# .github/workflows/test.yml 示例
- name: Run tests
  run: |
    uv sync
    uv run pytest tests/test_dependencies.py
    uv run pytest tests/test_project_structure.py
```
