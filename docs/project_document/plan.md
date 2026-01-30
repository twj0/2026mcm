# 2026 MCM Problem C（DWTS）建模路线与可行性确认

## 0. 可行性结论（对齐 Gemini 评价）

本题的关键是“反推粉丝投票”这一**不可辨识（identifiability）**问题：官方数据只给了评委分与历史结果，不给绝对票数。

- **我们能可靠得到的量**：每周/每季的**相对粉丝投票强度**（fan vote share / fan vote index）及其不确定性区间。
- **我们无法唯一确定的量**：绝对投票人数（例如“百万票”），除非引入额外外生信息或强先验。

这在 MCM 场景下是完全可接受的：题目要求“estimate / analyze / propose”，并未要求还原绝对票数。我们会在文中把“粉丝票估计”明确表述为**比例/指数 + 区间**，并用稳健性验证支撑可信度。

外生数据不是“解题必需”。在当前仓库审计结论下：
- `dwts_google_trends.csv` 与 `dwts_wikipedia_pageviews.csv` 主线弃用（覆盖/可比性不足），仅作为附录的失败尝试记录。
- `us_census_2020_state_population.csv` 可作为稳定可复现的参考变量（可选），但不作为 Q1 反推的决定性输入。

## 1. 数据与工程落地（必须先做）

### 1.1 原始数据

- 官方数据：`mcm2026c/2026_MCM_Problem_C_Data.csv`
- 外生数据（审计/附录，仅记录）：
  - Wikipedia pageviews（主线弃用）：`data/raw/dwts_wikipedia_pageviews.csv`
  - Google Trends（主线弃用）：`data/raw/dwts_google_trends.csv`
  - 2020 州人口（可选参考）：`data/raw/us_census_2020_state_population.csv`
    - 审计结论：两份人气 proxy（Trends/Pageviews）当前质量不足，主线不使用，仅作为附录/失败尝试记录（覆盖不足、抓取不稳定、可比性存疑等）。

### 1.2 建模输入表（建议作为“唯一真源”）

#### A. weekly panel（核心）

粒度：`season, week, celebrity`

字段建议：
- `season`
- `week`
- `celebrity_name`
- `pro_name`（ballroom_partner）
- `judge_score_total`（该周合计；或保留 `judge1..judgeK`）
- `judge_score_pct`（百分比法用：该周个人分 / 全体分之和）
- `judge_rank`（排名法用：该周按分排序得到名次，平分要处理）
- `active_flag`（该周是否仍在赛）
- `eliminated_this_week`（该周是否淘汰）

输出位置建议：`data/processed/dwts_weekly_panel.parquet`（或 csv）

#### B. season features（解释与回归）

粒度：`season, celebrity`

字段建议：
- 官方静态特征：`celebrity_industry, celebrity_age_during_season, celebrity_homestate, ...`
- 可选参考变量（稳定可复现）：
  - `state_population_2020`
- 可选内生派生特征（仅由官方数据计算）：
  - 历史均值/趋势（rolling mean / slope）、周内名次波动等

输出位置建议：`data/processed/dwts_season_features.parquet`

### 1.3 Pipeline 文件命名（按 conventions）

建议按你们的约定：`src/mcm2026/pipelines/mcm2026c_q<k>_<verb>_<object>.py`

- 主线脚本放 `src/mcm2026/pipelines/`；对照/炫技脚本放 `src/mcm2026/pipelines/showcase/`；对应产物写入 `outputs/*/showcase/`。
- Showcase（附录/炫技）统一说明文档：`docs/project_document/showcase.md`。

- `mcm2026c_q0_build_weekly_panel.py`（数据落地）
- Q1（主线/对照/炫技）：
  - `mcm2026c_q1_smc_fan_vote.py`
  - `mcm2026c_q1_rejection_hard_constraints.py`
  - `mcm2026c_q1_rank_plackett_luce_mcmc.py`
  - `mcm2026c_q1_dl_elimination_transformer.py`
- Q2（主线/对照/炫技）：
  - `mcm2026c_q2_counterfactual_simulation.py`
  - `mcm2026c_q2_tau_sensitivity.py`
  - `mcm2026c_q2_controversy_case_studies.py`
  - `mcm2026c_q2_agentic_ablation_runner.py`
- Q3（主线/对照/炫技）：
  - `mcm2026c_q3_mixed_effects_impacts.py`
  - `mcm2026c_q3_ridge_baseline.py`
  - `mcm2026c_q3_posterior_refit_kfold.py`
  - `mcm2026c_q3_dl_fanvote_mlp.py`
- Q4（候选机制库 + Pareto 评估）：
  - `mcm2026c_q4_design_space_eval.py`
  - `mcm2026c_q4_rule_*.py`
  - `mcm2026c_q4_multiobjective_pareto_search.py`

## 2. 第一题：估算粉丝投票（核心：逆向约束 + 采样/优化）

### 2.1 目标与输出

**目标**：对每个赛季每周，估计每位选手的粉丝投票份额 `P_fan(i)`（或排名 `R_fan(i)`）的分布。

**主要输出**：
- `E[P_fan(i)]`、`median[P_fan(i)]`、`CI_95%`（或分位数区间）
- `fan_vote_index`：把份额映射为便于比较的指数（例如 `logit(P_fan)` 或标准化 z-score）
- `certainty`：确定性指标（例如后验熵、可行解比例、区间宽度）

### 2.2 关键假设（最小集合）

- 每周的淘汰由当周规则（rank-based / percent-based）决定；允许以“噪声”形式放宽（见 2.4）。
- 粉丝票只需要**相对比例**，不需要绝对人数。
- 同一周投票总量被归一化：`sum_i P_fan(i) = 1`。

### 2.3 约束构造

#### 百分比法（Percentage）

- 已知：`P_judge(i) = judge_score_total(i) / sum_j judge_score_total(j)`
- 未知：`P_fan(i)`，且 `P_fan(i) >= 0, sum P_fan = 1`
- 总分：`T(i) = P_judge(i) + P_fan(i)`
- 淘汰约束（硬约束版）：实际淘汰者 `e` 满足：`T(e) = min_i T(i)`

#### 排名法（Rank）

- 已知：`R_judge(i)`（按评委分排名）
- 未知：`R_fan(i)`（粉丝票排名，为 1..n 的排列）
- 总排名：`S(i) = R_judge(i) + R_fan(i)`
- 淘汰约束（硬约束版）：`S(e) = max_i S(i)`

### 2.4 求解方法（推荐：Monte Carlo/SMC；备选：优化）

#### 方法 A：Monte Carlo + 拒绝采样（易实现、可解释）

- 百分比法：
  - 先验：`P_fan ~ Dirichlet(alpha)`
  - `alpha` 默认全 1（均匀）；可做不同先验强度的敏感性分析，但主线不依赖外生人气 proxy
  - 抽样 `P_fan`，计算 `T(i)`，保留满足淘汰约束的样本
  - 得到 `P_fan(i)` 的后验样本集合

- 排名法：
  - 先验：对 `R_fan` 取均匀随机排列
  - 计算 `S(i)`，保留满足淘汰约束的排列
  - 统计 `R_fan` 的后验分布（可转换成“相对票强度”）

#### 方法 B：软约束（更现实，也更稳健）

硬约束可能导致某些周可行解很少（或与现实不完全一致）。可将淘汰规则建成概率：

- 令 `Pr(eliminate=i) ∝ exp(-T(i)/tau)`（百分比法）或 `exp(S(i)/tau)`（排名法）
- 用 `tau` 控制噪声：`tau→0` 接近硬淘汰
- 通过重要性采样/SMC 得到后验

#### 方法 C：最优化（用于“点估计”）

构造一个目标让“预测的全季名次/淘汰周”与真实最接近：
- 决策变量：每周 `P_fan(i)` 或参数化 `P_fan(i)=softmax(z_i)`
- 约束：`P_fan` simplex
- 目标：最小化淘汰/名次的损失（例如 hinge loss）

推荐在论文主线用 A/B（输出区间、可解释），优化作为补充。

### 2.5 外生数据如何并入（让反推更“有自信”）

- 州人口（可选参考）：
  - 可作为弱解释变量/敏感性对照（例如加入 Q3 回归），但不作为 Q1 反推的决定性信息。

注意：主线反推与机制对比仅依赖官方数据（评委分 + 淘汰结果）。

### 2.6 验证与可解释性输出

- 每周：可行样本数、`certainty`（熵/区间宽度）
- 全季：
  - 预测冠军/进入决赛概率（由后验样本产生）
  - “争议指数”：评委排名与粉丝票排名分歧程度（Spearman 相关等）

## 3. 第二题：两种规则比较 + 争议案例 + judge save（核心：反事实模拟）

### 3.1 目标与输出

- 把同一季的数据放在两套规则下重算：
  - 淘汰者概率变化、冠军概率变化
  - 粉丝权重敏感性（人气断层在 percent 法下更强）

**输出建议**：
- “机制差异表”：每季在两规则下冠军变化率、淘汰周变化率
- “争议案例复盘”：例如 Bobby Bones
- 引入 judge save（评委二选一）的结果变化

### 3.2 模拟方式

基于第一题得到的后验样本：
- 对每个后验样本（每周一组 `P_fan` 或 `R_fan`），在另一套规则下重新计算淘汰/晋级
- 统计反事实下的概率分布（而不是给单一结论）

### 3.3 judge save 的建模

在每周淘汰前加入：
- 先按总分选出 bottom-2
- 评委从 bottom-2 中救一个

评委救人规则可取：
- 纯按评委分救高者（最小假设）
- 或引入噪声（以体现“节目效果/裁判偏好”）

## 4. 第三题：选手特征 + 专业舞伴影响（核心：回归/混合效应/层级）

### 4.1 两条线并行（分别解释“技术”和“人气”）

- 技术线（评委分）：
  - 因变量：`judge_score_pct`（周）或 `season_avg_judge_pct`（季）
- 人气线（粉丝票）：
  - 因变量：第一题输出的 `fan_vote_index`（周/季）

### 4.2 特征工程

- 选手：年龄、行业、home state
- 舞伴（pro）：
  - 固定效应/随机效应（建议 random intercept）
  - 可加“历史战绩”特征：例如过往进入决赛次数（需要从本数据计算即可，不必外抓）
- 可选参考变量：州人口（仅对美国选手，且不作为决定性输入）

### 4.3 建模选择

- 快速 baseline（附录/showcase，对照线）：`src/mcm2026/pipelines/showcase/mcm2026c_q3_ml_fan_index_baselines.py`

- 论文更强方案（推荐）：statsmodels 混合效应
  - `fan_vote_index ~ age + industry + state_population_2020 + (1|pro) + (1|season)`
  - 输出系数、置信区间、显著性与解释

### 4.4 验证

- 按 season 分组的交叉验证（leave-one-season-out）
- 与 baseline（Ridge）/不做不确定性传播的版本对比（看解释力/泛化是否提升）

## 5. 第四题：提出新系统（核心：机制设计 + 指标 + 仿真）

### 5.1 定义“更好/更公平”的量化指标

建议至少包含：
- **专业性保护**：冠军的评委排名分位数（越靠前越好）
- **人气表达**：粉丝票对结果的边际影响（不能为 0，否则失去互动）
- **鲁棒性**：对极端粉丝票（断层）不敏感（避免“低技术高票”压倒性夺冠）
- **可解释/可执行**：规则简单，观众能理解

### 5.2 候选规则模板（可选其一作为主推荐）

- 非线性压缩粉丝票：
  - `P_fan' = softmax(log(P_fan + eps) / k)` 或 `sqrt(P_fan)`
  - 直觉：降低“人气断层”的压制力

- 动态权重：
  - 前期粉丝权重大，后期评委权重提高

- 技术门槛 + 粉丝加成：
  - 评委分低于阈值不能进决赛（阈值可由历史分布定）

- judge save 常态化：
  - 每周 bottom-2 由评委救一个

### 5.3 证明方式（必须仿真）

- 用第一题后验样本做仿真：
  - 在旧规则 vs 新规则下跑完整季
  - 汇总指标与争议案例（如 Bobby Bones）
- 给出 trade-off 图表：公平性提升 vs 观众投票影响保留

## 6. 交付物（对应论文/备忘录）

### 6.1 表格（outputs/tables）

- 每周 `fan_vote_index` 与区间
- 机制对比的冠军/淘汰变化概率
- 特征影响回归表（含 pro 效应）
- 新机制指标对比表

### 6.2 图（outputs/figures）

- 每季争议指数时间序列
- 案例季（如 27）在不同机制下冠军概率条形图
- pro 随机效应/固定效应可视化
- 新机制 trade-off 曲线

### 6.3 出图计划（LaTeX 阶段的“图谱清单”，不影响主线代码）

说明：现阶段主线先保证 CSV 产物与叙事闭环；图在 LaTeX 阶段生成并插入。每张图都需要同时回答一个“评委会问的问题”。

1. **Fig-Q0-1：数据覆盖与事件类型概览（sanity）**
   - 数据源：`outputs/tables/mcm2026c_q0_sanity_season_week.csv`、`outputs/tables/mcm2026c_q0_sanity_contestant.csv`
   - 画法：条形图/表格化摘要（缺失/冲突为 0 的结论要显式写出）
   - 想回答：预处理是否可靠、是否存在大规模结构性缺失。

2. **Fig-Q1-1：不确定性强弱分布（ESS / evidence）**
   - 数据源：`outputs/tables/mcm2026c_q1_uncertainty_summary.csv`
   - 画法：
     - `ess_ratio` 的直方图/箱线图（可按 mechanism 分组）
     - `evidence` 的直方图/箱线图
   - 想回答：哪些周可辨识性强/弱，主线推断是否“合理但不过度自信”。

3. **Fig-Q1-2：示例赛季的 fan_share / fan_vote_index 时间序列（带区间）**
   - 数据源：`outputs/predictions/mcm2026c_q1_fan_vote_posterior_summary.csv`
   - 列：`season, week, celebrity_name, mechanism, fan_share_mean, fan_share_p05, fan_share_p95, fan_vote_index_mean, fan_vote_index_p05, fan_vote_index_p95`
   - 画法：选 1 个“正常季” + 1 个“争议季”各画 1 张（折线 + ribbon 区间）
   - 想回答：粉丝强度随周次的动态是否符合直觉（淘汰前后变化、不确定性随 active 数量变化）。

4. **Fig-Q2-1：机制一致性与分歧周（Percent vs Rank）**
   - 数据源：`outputs/tables/mcm2026c_q2_mechanism_comparison.csv`
   - 画法：
     - 每季 `match_rate_percent / match_rate_rank` 条形图
     - `diff_weeks_percent_vs_rank` 条形图（指出分歧周占比）
   - 想回答：两种公开说法在历史数据上的“可兼容程度”，以及分歧集中在哪些季/周。

5. **Fig-Q3-1：固定效应系数森林图（两条线：技术 vs 人气）**
   - 数据源：`outputs/tables/mcm2026c_q3_impact_analysis_coeffs.csv`
   - 画法：对 `judge_score_pct_mean` 与 `fan_vote_index_mean` 各画一张 forest plot（`estimate` + CI）
   - 想回答：哪些特征在“技术线”有效、哪些在“人气线”有效（以及不确定性是否足够诚实）。

6. **Fig-Q4-1：新机制 trade-off 散点图（Pareto 视角）**
   - 数据源：`outputs/tables/mcm2026c_q4_new_system_metrics.csv`
   - 列：`mechanism, outlier_mult, tpi_season_avg, fan_vs_uniform_contrast, robust_fail_rate`
   - 画法：
     - 横轴 `fan_vs_uniform_contrast`，纵轴 `tpi_season_avg`，点颜色/大小编码 `robust_fail_rate`
     - 按 `outlier_mult` 分面（2x/5x/10x）
   - 想回答：规则不是“谁更好”，而是 trade-off；压力测试档位越极端，机制排序是否发生变化。

7. **Fig-Q4-2：Bobby Bones（S27）压力测试冠军概率**
   - 数据源：`outputs/tables/mcm2026c_q4_new_system_metrics.csv`
   - 画法：S27 子集，按 mechanism×outlier_mult 的 `champion_mode_prob` 柱状图/热力图
   - 想回答：外部动员型极端案例在何种压力档位下出现，以及为何我们把它当作“风控边界”而不是主线失败。

## 7. 我们的“自信来源”（写给评委看的）

- 反推不是拍脑袋：
  - 约束来自规则 + 已知淘汰/名次
  - 输出是分布与区间，并给出确定性指标
- 外生数据审计有据可查：
  - 人口数据可复现、带元数据
  - 人气 proxy（Trends/Pageviews）已完成审计并明确主线弃用；附录可展示失败原因与风险控制
- 稳健性：
  - 机制切换点（如 season 28）当作参数做敏感性分析
  - 硬/软约束两版对照

## 8. 炫技（加分）模块（但仍保持可复现与可解释）

- **不确定性可视化**：
  - Q1 输出每周每人的后验区间宽度/熵（certainty）热图，展示“哪些周/哪些人可辨识性更强”。
- **反事实仿真 + 争议复盘**：
  - 用 Q1 后验样本在 Rank/Percent/Judge Save 下重跑整季，输出冠军概率与淘汰路径概率（而非单次模拟）。
- **层级/混合效应（解释 pro dancer）**：
  - 对 pro dancer 做随机效应，给出效应分布与区间，并与 baseline 线性/岭回归对照。
- **机制设计的指标化 trade-off 曲线**：
  - 为新规则定义“技术保护/人气表达/鲁棒性/可解释性”指标，给出旧规则 vs 新规则的对比表与 trade-off 图。
- **深度学习/大模型概念的对照实验（附录，可失败）**：
  - 目的：展示我们掌握新方法，但不让主线依赖它；即使效果不如传统方法，也能给出“为何失败/为何过拟合/传统方法为何更强”的证据。
  - 数据与合规：仅使用官方数据构造的 `weekly panel` / `season features`（不引入额外外部语料与爬虫数据）。
  - 任务选择（任选其一或两者都做）：
    - Q1：把“当周是否被淘汰/进入 bottom-k”作为分类任务（输入为当周评委分相关特征 + 历史统计特征），用小型 MLP/TabTransformer 风格网络做对照。
    - Q3：对 `fan_vote_index`（来自 Q1 后验均值/样本）做回归，对比 Ridge/混合效应 vs 小型 MLP 的泛化能力。
  - 对应的“Test-Time Compute”落地方式（不使用不可控的 LLM 推理）：
    - 多随机种子/多初始化集成（ensemble）+ 校准（温度缩放）作为推断时加计算的稳健化手段。
  - 复现要求：固定随机种子、记录超参数与训练曲线；结论以分组交叉验证（leave-one-season-out）为准。
- **LLM / Agent 的现代化应用场景（工具化，不作为主线输入）**：
  - 目的：提升写作与工程效率，展示“我们会用大模型”，但不把 LLM 产出当作建模数据，避免不可复现与数据政策争议。
  - 可用场景（任选）：
    - 规则形式化与边界条件审计：让 LLM 将题面规则转成可执行的伪代码/测试用例清单，用来检查实现是否覆盖“双淘汰/无淘汰/退赛/Judge Save”。
    - Agentic 实验编排：让 Agent 自动跑 ablation（不同 tau、不同机制假设、不同先验强度、不同 CV 划分），并汇总成表格（核心计算仍是我们自己的代码）。
    - “论文一致性检查”RAG：对本仓库文档（题面+spec+Q1–Q4 计划）做本地检索问答，检查口径一致、避免自相矛盾与时间泄漏表述。
    - 结果叙事生成（可控模板）：用 LLM 根据固定模板生成图表 caption / 风险清单 / 解释段落草稿（最终人工审核）。
    - 机制候选生成（冻结候选集）：用 LLM 提出若干新投票规则候选（公式/参数范围），然后在仿真里做确定性评估；论文中冻结候选列表以保证可复现。

---

## 附：下一步最小可执行顺序

1. `mcm2026c_q0_build_weekly_panel.py`：把官方 wide 表转成 weekly panel（含 rank/pct）
2. `mcm2026c_q1_smc_fan_vote.py`：主线反推（Percent/Rank + soft constraint），输出后验与 certainty
3. `mcm2026c_q2_counterfactual_simulation.py`：用 Q1 后验做反事实 + judge save
4. `mcm2026c_q3_mixed_effects_impacts.py`：主线解释（混合效应 + 不确定性传播）
5. `mcm2026c_q4_design_space_eval.py`：候选机制库 + 指标化评估 + trade-off
