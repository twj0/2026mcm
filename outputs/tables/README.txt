---
# 数据说明
---

本目录包含论文的“可复现表格证据”。原则：

1. 表格数据都来自流水线输出（Q0–Q4），禁止手工改写。
2. 论文中表格建议用 `csvsimple` 从 CSV 直接读取插入（避免复制粘贴出错）。
3. 论文路径约定：在 `paper/texfile/*.tex` 内引用本目录文件时使用 `../outputs/tables/<file>.csv`。

## 论文专用（paper_*.csv）

这些表是为了在论文“误差分析 / 灵敏度分析”中更紧凑地展示关键证据，从主表聚合得到：

- **paper_q1_high_uncertainty_weeks.csv**
  - **用途**：列出 Q1 在 Percent 机制下最不稳定的赛季-周（ESS ratio 低、evidence 低），用于说明哪些周是误差高风险输入。
  - **论文位置**：`paper/texfile/6SensitivityAndErrorAnalysis.tex` 表 \ref{tab:q1_high_uncertainty_weeks}。
  - **来源**：`mcm2026c_q1_uncertainty_summary.csv` 过滤/排序。

- **paper_q1_error_cases.csv**
  - **用途**：列出 Q1 在发生淘汰的周里，观众-评委相关性与“淘汰选手后验退出概率”的案例，用于展示模型在低可识别周的偏差模式。
  - **论文位置**：`paper/texfile/6SensitivityAndErrorAnalysis.tex` 表 \ref{tab:q1_error_cases}。
  - **来源**：`mcm2026c_q1_error_diagnostics_week.csv`（Percent 机制）。

- **paper_q2_divergence_summary.csv**
  - **用途**：按 Percent vs Rank 的“分歧周数”排序，筛出最能区分机制的赛季，作为 Q2 误差/敏感性验证的重点样本。
  - **论文位置**：`paper/texfile/6SensitivityAndErrorAnalysis.tex` 表 \ref{tab:q2_divergence_summary}。
  - **来源**：`mcm2026c_q2_mechanism_comparison.csv`。

- **paper_q4_recommendation_summary.csv**
  - **用途**：在不同压力强度（outlier_mult）下给出“推荐机制”的汇总，并附带 TPI/Fan/Robust 的均值与区间，用于支撑 Q4 的决策建议与鲁棒性描述。
  - **论文位置（建议）**：`paper/texfile/7ModelEvaluation.tex` 或 Q4 章节（若单独写 Q4 讨论段）。
  - **来源**：`mcm2026c_q4_sensitivity_summary.csv`。

## 主流水线表（mcm2026c_*.csv）

这些表用于“追溯与复核”（不一定全部放进正文，但保证证据可查）：

- **mcm2026c_q0_sanity_season_week.csv / mcm2026c_q0_sanity_contestant.csv**
  - **用途**：Q0 数据审计与一致性检查（缺失、周数、选手数、淘汰标记）。
  - **论文位置（可选）**：数据预处理章节末尾/附录。

- **mcm2026c_q1_uncertainty_summary.csv**
  - **用途**：Q1 采样有效样本量（ESS）与证据强度等诊断，用于解释不确定性来源。
  - **论文位置**：`6SensitivityAndErrorAnalysis.tex`（建议用 paper_q1_high_uncertainty_weeks.csv 更紧凑）。

- **mcm2026c_q1_error_diagnostics_week.csv**
  - **用途**：Q1 的周级误差诊断与一致性核验字段（observed/pred exit 等）。
  - **论文位置**：误差分析（建议用 paper_q1_error_cases.csv）。

- **mcm2026c_q2_mechanism_comparison.csv / mcm2026c_q2_week_level_comparison_*.csv**
  - **用途**：Q2 的赛季级/周级机制对比，用于解释“分歧来自哪里”。
  - **论文位置**：敏感性分析 + 反事实对比（正文/附录均可）。

- **mcm2026c_q3_impact_analysis_coeffs.csv / mcm2026c_q3_fan_refit_stability.csv / mcm2026c_q3_dataset_diagnostics.csv**
  - **用途**：Q3 的系数、重拟合稳定性、数据诊断，用于论证 Q3 结论的稳健性。
  - **论文位置**：Q3 章节 + `6SensitivityAndErrorAnalysis.tex`。

- **mcm2026c_q4_new_system_metrics.csv / mcm2026c_q4_sensitivity_summary.csv**
  - **用途**：Q4 多目标指标与压力测试汇总；支持推荐与权衡图。
  - **论文位置**：Q4 图表（trade-off / robustness / recommendation）。

## LaTeX 插表模板（csvsimple）

已在 `paper/main.tex` 启用：`\usepackage{csvsimple}`。

推荐写法（避免 CSV 列名含下划线导致宏名非法）：

1. 在 `\csvreader{...}{<列名>=\SafeMacro,...}{...}` 中显式做列名映射。
2. 表头用 `booktabs`：`\toprule / \midrule / \bottomrule`。