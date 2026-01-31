---
# .ipynb文件的使用方法
---

## 1. 什么是 `.ipynb` 文件？

`.ipynb` 是 Jupyter Notebook 的文件格式。你可以把它理解成一种“可执行的实验讲义”，它把三类东西放在同一个文件里：

- **Markdown 文本**：用来写解释、假设、结论、注意事项（适合论文手阅读）
- **代码单元格（Code cell）**：可以逐格执行，边跑边看结果
- **输出（Output）**：表格预览、图像、日志等会直接显示在下面

在我们这个项目里，Notebook 的定位是：

- 给队友（尤其是论文手）一个“从问题→建模→输出→怎么解读”的可视化入口
- 不要求你读完所有 pipeline 源码，也能理解我们每一题在做什么


## 2. 通用使用方法：怎么打开/怎么运行？

你可以用下面任意一种方式打开 `.ipynb`：

### 2.1 用 IDE（推荐）

- 在 Windsurf / VSCode / PyCharm 等 IDE 里直接打开 `.ipynb`
- 按 cell 顺序执行（一般是从上往下）

### 2.2 用 JupyterLab/Notebook

1) 在项目根目录安装并进入环境

```bash
uv sync
```

2) 启动 Jupyter（如果你安装了 notebook 相关依赖组）

```bash
uv run jupyter lab
```

### 2.3 Notebook 的基本操作（所有人都用得到）

- **运行一个 cell**：点击 cell 左侧运行按钮，或者 Shift+Enter
- **重跑全部**：从上到下依次运行（建议第一次跑就这样做）
- **看输出**：每个 cell 的输出会紧跟在 cell 下方


## 3. 我们项目里的 notebooks：怎么使用、怎么跑？

本目录下的 notebooks 对应 Q0–Q4 主线。每个 notebook 都强调：

- 口语化解释（给论文手/新队友）
- 主线代码的可复现运行方式
- 关键输出表怎么读、关键图怎么解释


### 3.1 环境准备（第一次跑必看）

在项目根目录执行：

```bash
uv sync
```

如果你只是跑主线（Q0–Q4），默认依赖通常足够。


### 3.2 推荐运行顺序

如果你想从头理解整个项目，推荐顺序：

1) `Q0_Data_Preprocessing.ipynb`
2) `Q1_Fan_Vote_Inference.ipynb`
3) `Q2_Mechanism_Comparison.ipynb`
4) `Q3_Impact_Analysis.ipynb`
5) `Q4_New_System_Design.ipynb`

原因：Q1/Q2/Q3/Q4 依赖 Q0 的 processed 数据；而 Q2/Q3/Q4 又依赖 Q1 的推断结果。


### 3.3 “一键运行”怎么用？

每个 notebook 的前面都放了一个 **一键运行（one-click）** 的 cell（Markdown + code）。

它的设计目标是：

- 论文手/新队友不需要理解所有细节
- 直接跑这一格就能把该题的主要输出生成出来，并在 notebook 里看到关键表/关键图

一键运行通常会：

- 自动补齐依赖链（例如 Q4 会先跑 Q0→Q1→Q4）
- 生成输出 CSV 到 `outputs/tables/` 或 `outputs/predictions/`
- 调用可视化脚本生成图到 `outputs/figures/q*/tiff` 和 `outputs/figures/q*/eps`
- 在 notebook 里 **优先显示 TIFF**（更适合 notebook 预览；EPS 通常用于论文排版）


### 3.4 输出在哪里？怎么找？

- 主表 / 统计表：`outputs/tables/`
- 预测/推断结果：`outputs/predictions/`
- 图：`outputs/figures/q*/tiff`（预览） 与 `outputs/figures/q*/eps`（论文用）


### 3.5 Q4 特别说明：误差分析 & 灵敏度检测怎么跑？

Q4 的关键配置在：

- `src/mcm2026/config/config.yaml` 的 `dwts.q4` 节点

常用参数：

- `n_sims`：Monte Carlo 模拟次数（越大误差越小，但越慢）
- `bootstrap_b`：TPI 的 bootstrap 次数（用于 CI；越大越稳，但越慢）
- `sigma_scales`：灵敏度检测参数（控制 Q1→Q4 不确定性传播强度；建议如 `[0.5, 1.0, 2.0]`）
- `outlier_mults`：压力测试强度（如 2x/5x/10x）
- `robustness_attacks.enabled`：扩展攻击（默认关闭，打开会变慢）

Q4 现在会输出两张主线表：

- `mcm2026c_q4_new_system_metrics.csv`：逐 season 的主表（用于细看/可视化）
- `mcm2026c_q4_sensitivity_summary.csv`：跨 season 聚合的敏感性汇总表（更适合写论文“总体结论”）


## 4. 常见问题（Troubleshooting）

### 4.1 ImportError / ModuleNotFoundError

一般原因：你没有在项目根目录运行，或者 notebook 没把 `src/` 加到 `sys.path`。

建议：

- 先运行每个 notebook 顶部的“路径初始化” cell（会自动把 `src` 加进去）
- 或者确保你的工作目录在项目根目录

### 4.2 File not found：找不到 outputs / tables / predictions

一般原因：上游 pipeline 还没跑。

建议：

- 直接运行该 notebook 的“一键运行” cell
- 或者在根目录跑一次全链路：

```bash
uv run python run_all.py
```

### 4.3 图在 notebook 里不显示（EPS/TIFF）

我们在 notebook 里 **优先显示 TIFF**。

- 如果你只看到了 EPS 路径提示，通常是因为 TIFF 还没生成，或文件名不在候选列表中。
- EPS 主要用于论文排版，notebook 里不一定能稳定预览（取决于系统环境）。

建议：

- 先确认 `outputs/figures/q*/tiff/` 是否有文件
- 如果有但没被展示，检查文件名与 notebook 候选列表是否一致

