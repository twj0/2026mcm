# Project Structure (DWTS / MCM 2026 Problem C)

本文件用于快速理解本仓库的结构、主线产物、以及“一条命令复现”的入口。

## 1. 一句话总览

- `data/raw/`：题目给定数据（只读）与可选外生数据（主线不依赖）。
- `data/processed/`：Q0 预处理后的“唯一真源”数据表。
- `src/mcm2026/pipelines/`：Q0–Q4 主线可复现流水线脚本。
- `outputs/`：论文所需表格/预测结果（可复现生成）。
- `docs/`：建模路线、Q1–Q4 方案文档、审计报告（内部）。
- `paper/`：论文相关文件（当前先用 `paper/draft.md` 写中文草稿）。

## 2. 关键入口（复现）

- 一键复现（推荐）：

```bash
uv run python run_all.py
```

该命令会依次运行：Q0 → Q1 → Q2 → Q3 → Q4，并写出本仓库主线 CSV 产物。

## 3. 主线数据（Q0 产物）

- `data/processed/dwts_weekly_panel.csv`
  - 周级面板：`season, week, celebrity` 粒度。
  - 含评委分、rank/percent 相关字段、active 标记、淘汰/退赛事件等。
- `data/processed/dwts_season_features.csv`
  - 赛季级特征：`season, celebrity` 粒度。

## 4. 主线流水线脚本（Q0–Q4）

- Q0 预处理
  - `src/mcm2026/pipelines/mcm2026c_q0_build_weekly_panel.py`
- Q1 反推粉丝投票强度（后验分布）
  - `src/mcm2026/pipelines/mcm2026c_q1_smc_fan_vote.py`
- Q2 机制对比与反事实（rank vs percent + judge save）
  - `src/mcm2026/pipelines/mcm2026c_q2_counterfactual_simulation.py`
- Q3 特征影响分析（混合效应/层级模型）
  - `src/mcm2026/pipelines/mcm2026c_q3_mixed_effects_impacts.py`
- Q4 新系统设计与评估（机制族 + 多指标 + 压力测试）
  - `src/mcm2026/pipelines/mcm2026c_q4_design_space_eval.py`

## 5. 主线输出（论文表格/证据链）

- Q0
  - `outputs/tables/mcm2026c_q0_sanity_season_week.csv`
  - `outputs/tables/mcm2026c_q0_sanity_contestant.csv`
- Q1
  - `outputs/predictions/mcm2026c_q1_fan_vote_posterior_summary.csv`
  - `outputs/tables/mcm2026c_q1_uncertainty_summary.csv`
- Q2
  - `outputs/tables/mcm2026c_q2_mechanism_comparison.csv`
- Q3
  - `outputs/tables/mcm2026c_q3_impact_analysis_coeffs.csv`
- Q4
  - `outputs/tables/mcm2026c_q4_new_system_metrics.csv`

## 6. 文档入口（读者视角）

- 建模总路线：`docs/project_document/plan.md`
- 分题方案：
  - `docs/project_document/Q1.md`
  - `docs/project_document/Q2.md`
  - `docs/project_document/Q3.md`
  - `docs/project_document/Q4.md`
- 写作口径备忘录：`docs/project_document/writing_reference_memo.md`

## 7. 论文草稿

- 中文草稿（Markdown，无图片）：`paper/draft.md`

后续转 LaTeX 时，再将图表插入 `paper/main.tex` / `paper/figures/`。
