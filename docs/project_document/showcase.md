# Showcase（附录/炫技模块）

本文件用于统一说明本仓库的“showcase（附录/炫技）”代码与产物口径。

## 1. 定位

- Showcase 仅用于“现代方法对照/失败分析/加分点展示”。
- Showcase **不改变** Q0–Q4 主线脚本的输出文件名与内容。
- Showcase 的结论默认不进入主线结论；如要写进论文，应放入“附录/扩展实验”并明确其局限性。

## 2. 复现原则

- 所有 showcase 产物写入 `outputs/tables/showcase/`（未来如需图片，则写入 `outputs/figures/showcase/`）。
- 不引入新的外生数据作为主线输入；若探索外生数据，必须在文中注明覆盖不足/可比性问题，且不得进入主线。

## 3. 目录结构

- 代码目录：`src/mcm2026/pipelines/showcase/`
- 输出目录：`outputs/tables/showcase/`

## 4. 当前已实现的 Showcase 实验

### 4.1 Q1：淘汰预测（LogReg vs MLP；LOSOCV）

- 脚本：`src/mcm2026/pipelines/showcase/mcm2026c_q1_ml_elimination_baselines.py`
- 输出：
  - `outputs/tables/showcase/mcm2026c_q1_ml_elimination_baselines_cv.csv`
  - `outputs/tables/showcase/mcm2026c_q1_ml_elimination_baselines_cv_summary.csv`

说明：该任务是预测视角对照，不替代 Q1 的“反推 fan vote share/index”主线。

### 4.2 Q3：fan_vote_index 回归（Ridge vs MLP；LOSOCV）

- 脚本：`src/mcm2026/pipelines/showcase/mcm2026c_q3_ml_fan_index_baselines.py`
- 输出：
  - `outputs/tables/showcase/mcm2026c_q3_ml_fan_index_baselines_cv.csv`
  - `outputs/tables/showcase/mcm2026c_q3_ml_fan_index_baselines_cv_summary.csv`

说明：该任务用于展示“tabular 回归”对照线，主线仍以可解释模型（混合效应/层级）为准。

## 5. 如何运行

- 推荐入口：`uv run python run_all.py --showcase`
- 或单独运行：
  - `uv run python -m mcm2026.pipelines.showcase.mcm2026c_q1_ml_elimination_baselines`
  - `uv run python -m mcm2026.pipelines.showcase.mcm2026c_q3_ml_fan_index_baselines`

## 6. 论文写法建议（附录口径）

- 先给出“为何做 showcase”：展示现代方法能力与正确的失败诊断。
- 只引用 `cv_summary.csv` 的汇总指标，不在主线过度展开。
- 若 DL 不优于传统方法，强调“小数据 + 强结构约束场景下传统/层级模型更稳健”。
