# 2026 MCM/ICM Problem C（DWTS）中文草稿（Markdown，无图片）

> 说明：本草稿用于中文写作与结构对齐（不直接插入图片/表格）；后续转 LaTeX 时按 `paper/texfile/*.tex` 的正文结构插入。
>
> 本草稿章节顺序严格对齐 `paper/texfile/`：
>
> - `0abstract.tex` → Abstract
> - `1ProblemRestatement.tex` → Introduction
> - `2ProblemAnalysis.tex` → Problem Analysis
> - `3AssumptionAndNotations.tex` → Assumptions & Notations
> - `4DataPreprocessing.tex` → Data Preprocessing
> - `5ModelBuilding.tex` → Model Building and Solution
> - `6SensitivityAndErrorAnalysis.tex` → Sensitivity & Error Analysis
> - `7ModelEvaluation.tex` → Model Evaluation & Conclusion
> - `8Reference.tex` → References
> - `9Appendix.tex` → Appendix
> - `10AIToolDeclaration.tex` → Report on Use of AI
>
> 本仓库所有主线结论均可由 `uv run python run_all.py`（或分别运行 Q1/Q2/Q3/Q4 pipeline）复现生成，对应 CSV 产物位于 `outputs/`.

## 0 Abstract（摘要 / Summary Sheet）

我们研究《Dancing with the Stars》(DWTS) 中“评委分 + 观众投票”合成淘汰机制下，如何在缺少真实投票数据的前提下：

1. 反推出每周各选手的相对粉丝投票强度，并量化不确定性（Q1）。
2. 在两种已知机制（Percent vs Rank）与“Judge Save”变体下进行反事实比较，解释争议赛季（Q2）。
3. 分析选手特征与职业舞伴（pro dancer）对“技术线（评委表现）”与“人气线（粉丝强度）”的影响差异（Q3）。
4. 设计更稳健的新投票结合系统，通过多指标评价与压力测试给出可执行建议（Q4）。

## 1 Introduction（引言）

### 1.1 Background（背景与动机）

DWTS 的淘汰机制同时依赖评委打分与观众投票，但官方并不公布真实投票数。题目要求在这一信息不完整的现实约束下，建立可解释、可复现的模型：

- 在不引入外部“人气 proxy”作为主线输入的前提下，反推出相对粉丝强度，并给出不确定性；
- 在不同公开叙事（Percent vs Rank）下对赛制效果做反事实比较；
- 在“技术线 vs 人气线”两条维度上解释影响因素；
- 给出更稳健的新机制建议，并用压力测试公开说明模型边界。

### 1.2 Restatement of the Problem（问题重述）

题目要求我们完成：

1. **Q1**：估计每周每位选手的相对粉丝投票强度，并量化不确定性.
2. **Q2**：比较 Rank 与 Percent 两类合成机制，并讨论 Judge Save 变体与争议案例.
3. **Q3**：分析选手特征与职业舞伴（pro dancer）对“技术线（评委表现）”与“人气线（粉丝强度）”的影响差异.
4. **Q4**：设计更稳健的新投票结合系统，并用多指标评价与压力测试给出建议.

### 1.3 Our Work（我们的工作概览）

我们的策略是“主线强解释 + 支线强加分”：

- **主线（Mainline）**：以最小主观性（KISS）实现可解释统计模型与可复现工程流水线；所有核心结论均从 `outputs/` 的 CSV 产物可追溯.
- **支线（Showcase / Appendix-only）**：提供更大规模的参数网格敏感性、机器学习 baseline 与自动化实验编排，作为附录增强证据，不改变主线口径.

## 2 Problem Analysis（问题分析）

### 2.1 Analysis of Question One

Q1 的核心难点是 **identifiability**：没有真实票数，只能识别“与淘汰结果一致的相对粉丝强度”。因此主线输出定位为：

- `fan_share`：每周在赛集合上的相对份额（simplex 上求和为 1）；
- `fan_vote_index`：便于跨周比较的人气指数（对份额做 logit 变换），并提供区间.

我们将“淘汰规则”转换为软约束概率，并用重要性采样/重采样得到后验分布.

### 2.2 Analysis of Question Two

Q2 的关键在于：Percent 与 Rank 虽然都被宣传为“50/50”，但其实际偏好不同.

- Percent 会放大份额差距（极端票数更容易支配）.
- Rank 更接近“名次投票”，对极端动员更稳健，但可能牺牲粉丝表达.

因此 Q2 需要把“机制差异”量化为可复现指标，并可定位到具体周（week-level）解释误差来源.

### 2.3 Analysis of Question Three

Q3 的关键是：避免把“技术”和“人气”混为一个黑盒；同时粉丝线来自 Q1 后验，必须传播不确定性.

我们用两条 outcome：

- 技术线：`judge_score_pct_mean`（赛季聚合）；
- 人气线：`fan_vote_index_mean`（赛季聚合），并用重复拟合传播 Q1 不确定性.

### 2.4 Analysis of Question Four

Q4 是机制设计（mechanism design）：我们不追求“预测现实冠军”，而是评估不同机制在多个目标上的 trade-off，并通过压力测试透明表达“外部动员型极端案例”的风险边界.

## 3 Assumptions and Justifications（假设与合理性）

为保证主线可执行、可复现、且不引入不必要主观性，我们采用以下最小假设：

1. **数据可信**：题目给定的评委分、结果与选手信息真实可靠.
2. **周内在赛集合可由面板确定**：以 `active_flag` 为当周在赛集合的判定依据.
3. **淘汰规则可用软约束近似**：用温度参数 `τ` 表达现实中的噪声与制作安排（硬规则的扰动）.
4. **Rank 口径的粉丝名次可由份额排序近似**：这是 KISS 近似；其敏感性将通过 percent vs rank 对比显式披露.
5. **主线不引入外部人气 proxy**：避免覆盖不足与可比性风险；极端动员情景通过 Q4 压力测试显式讨论.

### 3.1 Notations（符号与记号）

为避免口径漂移，我们统一符号如下（均为主线用法）：

| 记号 | 含义 | 备注 |
|---|---|---|
| `s` | season（赛季） | `s=1..34` |
| `t` | week（周） | 每季周数不同 |
| `i` | contestant（选手） | `celebrity_name` |
| `A_{s,t}` | 当周仍在赛选手集合 | 由 `active_flag` 决定 |
| `J_{s,t,i}` | 当周评委总分 | `judge_score_total` |
| `pJ_{s,t,i}` | 当周评委份额 | `judge_score_pct`，在 `A_{s,t}` 内归一化 |
| `rJ_{s,t,i}` | 当周评委名次 | `judge_rank`（从 1 开始，越小越好） |
| `pF_{s,t,i}` | 当周粉丝投票份额（未知） | Q1 反推对象，满足 `sum_{i in A} pF=1` |
| `fan_vote_index` | 粉丝强度指数 | 主线采用 `logit(pF)` 的后验统计量 |
| `α` | 评委权重 | 主线默认 `α=0.5`（见 `config.yaml`） |
| `τ` | 软约束温度参数 | `τ` 越小越接近“硬淘汰”，主线默认 `τ=0.03` |
| `TPI` | 技术保护指数 | Q4 主线实现为 `tpi_season_avg`（冠军赛季平均评委分位数） |
| `fan_vs_uniform_contrast` | “粉丝 vs 均匀”对照差异率 | Q4 主线实现的粉丝表达指标（受控对照实验） |
| `robust_fail_rate` | 压力测试失败率 | Q4 中“极端动员”下冠军偏离技术 top1 的频率 |

---

## 4 Data Preprocessing（数据与预处理，Q0）

### 4.1 数据规模与主线产物概览（可直接写进论文）

- **Q0 处理后周级面板**：`dwts_weekly_panel.csv` 为 `2777 × 18`.
- **Q0 赛季特征表**：`dwts_season_features.csv` 为 `421 × 11`.
- **Q1 不确定性汇总行数**：`mcm2026c_q1_uncertainty_summary.csv` 共 `670` 行（season-week-mechanism 粒度）.
- **Q4 评价表行数**：`mcm2026c_q4_new_system_metrics.csv` 共 `714` 行（34 seasons × 7 mechanisms × 3 outlier 档位）.

这些数量可作为“数据与工程落地已闭环”的硬证据.

---

### 4.2 数据来源与数据政策

- 官方数据：`mcm2026c/2026_MCM_Problem_C_Data.csv`.
- 外生数据：本仓库包含 Google Trends/Wikipedia pageviews/州人口等文件，但**主线不依赖人气 proxy**（覆盖不足或可比性风险），仅将其作为审计记录或附录可能性.

### 4.3 统一“真源表”（Canonical Tables）

我们将官方 wide 表整理为两张建模输入表：

- 周级面板：`data/processed/dwts_weekly_panel.csv`
  - 粒度：`season, week, celebrity_name`.
  - 核心字段：`judge_score_total, judge_score_pct, judge_rank, active_flag, eliminated_this_week, withdrew_this_week` 等.
- 赛季特征表：`data/processed/dwts_season_features.csv`
  - 粒度：`season, celebrity_name`.
  - 包含选手静态特征与舞伴信息，用于 Q3.

### 4.4 质量控制与现实一致性检查（Sanity & Value Range）

我们对处理后的面板做一致性审计，确保核心建模字段（评委分/份额/排名等）可用；缺失主要集中在“文本不可解析、淘汰后周次不再评分”等结构性字段.

sanity check 表位于：

- `outputs/tables/mcm2026c_q0_sanity_season_week.csv`
- `outputs/tables/mcm2026c_q0_sanity_contestant.csv`

为避免“模型算出来但不符合现实口径”的风险，我们对主线输入/输出做了最小但关键的数值一致性检查（均可由产物 CSV 复现）：

- **评委份额归一化**：在每个 `season-week` 的在赛集合内，`judge_score_pct` 求和应为 1（浮点误差量级约 `1e-15`）.
- **粉丝份额在 simplex 上**：在每个 `season-week-mechanism` 内，Q1 的 `fan_share_mean` 求和应为 1（浮点误差量级约 `1e-15`），且分位数落在 `[0,1]`.
- **概率型指标值域**：Q1 诊断中的 `observed_exit_prob_at_posterior_mean` 严格落在 `[0,1]`；Q4 指标如 `robust_fail_rate` 等均在 `[0,1]`.

## 5 Model Building and Solution（模型建立与求解）

### 5.1 Model Establishment and Solution of Question One（Q1：反推粉丝投票强度）

题目给定 DWTS 多赛季数据：每周评委打分（多名评委）、选手静态信息、以及最终结果/名次文本. 题目要求在此基础上完成四项任务：

- **Q1**：估计每周每位选手的粉丝投票强度（相对份额/指数）并量化不确定性.
- **Q2**：比较 Rank 与 Percent 两类合成机制，进行反事实模拟，并分析争议案例与 Judge Save 变体.
- **Q3**：分析选手特征与职业舞伴（pro dancer）对“技术线（评委表现）”与“人气线（粉丝强度）”的影响差异.
- **Q4**：设计更稳健的新投票结合系统，通过多指标评价与压力测试给出可执行建议.

我们采用“传统主线 + 现代化加分点（仅附录）”的策略：主线以可解释、可复现的统计建模为核心；深度学习/自动化实验编排作为可选附录，不作为主线输入.

---

### 5.1.1 输入与输出（Q1 产物索引）

Q1 的主线产物为：

- `outputs/predictions/mcm2026c_q1_fan_vote_posterior_summary.csv`
- `outputs/tables/mcm2026c_q1_uncertainty_summary.csv`

为保证论文“误差/一致性/敏感性”段落可直接落地，我们在本次增强后，Q1 额外输出：

- `outputs/tables/mcm2026c_q1_error_diagnostics_week.csv`（week-level 误差与一致性诊断）
- `outputs/tables/mcm2026c_q1_mechanism_sensitivity_week.csv`（percent vs rank 的周级敏感性）

### 5.1.2 核心思想：可辨识性与输出形式

由于真实投票数不可观测，本题可稳定识别的是：在给定赛制与评委分的前提下，能够解释历史淘汰结果的**相对粉丝强度**.

我们输出两种量：

- `fan_share`：当周粉丝投票份额（simplex 上，和为 1）.
- `fan_vote_index`：对份额做 logit 变换得到的指数（便于跨周比较），并提供区间.

### 5.1.3 机制建模：Percent vs Rank

对于每个赛季-周，设当周仍在赛选手集合为 `A`.

- Percent（份额相加）：
  - 已知评委份额 `pJ_i`, 未知粉丝份额 `pF_i`.
  - 合成得分 `T_i = α·pJ_i + (1-α)·pF_i`.
- Rank（名次相加）：
  - 已知评委名次 `rJ_i`.
  - 我们用 `pF` 的排序近似粉丝名次 `rF_i`（KISS 做法）.
  - 合成名次 `S_i = α·rJ_i + (1-α)·rF_i`.

其中 `α` 为评委权重.

### 5.1.4 软约束似然与重要性采样/重采样

硬约束“最低者必淘汰”在现实中可能被制作安排、噪声等扰动. 我们采用温度参数 `τ` 将淘汰规则转为概率（soft constraint）：

- Percent：`Pr(eliminate=i) = softmax(-T_i/τ)`
- Rank：`Pr(eliminate=i) = softmax(S_i/τ)`

对每周从 Dirichlet 先验采样 `pF`, 计算观测淘汰集合的似然权重，进行重要性重采样，得到 `pF` 的后验样本，并汇总均值/中位数/5%-95% 分位数.

主要配置位于：`src/mcm2026/config/config.yaml`（主线默认 `α=0.5, τ=0.03`）.

### 5.1.5 产物与验证

- 后验汇总（主输出）：
  - `outputs/predictions/mcm2026c_q1_fan_vote_posterior_summary.csv`
- 不确定性汇总（每周 ESS/证据等）：
  - `outputs/tables/mcm2026c_q1_uncertainty_summary.csv`

我们使用 ESS、证据（平均似然）等量作为“可辨识性/信息量”指标，并在后续问题（Q3/Q4）中通过抽样传播不确定性.

### 5.1.6 Q1 算法步骤（伪代码）

下面用伪代码描述主线 Q1（对每个 season-week 独立推断，KISS）：

```text
For each mechanism in {percent, rank}:
  For each (season s, week t):
    1) 取当周仍在赛集合 A_{s,t}
    2) 从先验采样 m 次：pF^(m) ~ Dirichlet(1)
    3) 计算合成得分/名次：
         percent: T_i = α·pJ_i + (1-α)·pF_i
         rank:    S_i = α·rJ_i + (1-α)·rF_i  (rF 由 pF 排序近似)
    4) 用温度 τ 把淘汰规则转成概率（soft constraint），得到权重 like^(m)
    5) 归一化权重 w^(m)，计算 ESS 与 evidence
    6) 按 w^(m) 重采样 r 次，得到后验样本集合
    7) 汇总后验：mean / median / p05 / p95
```

### 5.1.7 Q1 诊断与可视化（不插图但给出图位描述）

- 周级误差/一致性诊断：`outputs/tables/mcm2026c_q1_error_diagnostics_week.csv`
  - `match_pred`：后验均值回放淘汰集合是否匹配观测
  - `observed_exit_prob_at_posterior_mean`：观测淘汰集合在后验均值下的概率（越小越难解释）
  - `fan_share_width_mean` / `fan_index_width_mean`：不确定性强度
- 周级机制敏感性：`outputs/tables/mcm2026c_q1_mechanism_sensitivity_week.csv`
  - `tv_distance`：percent vs rank 份额分布差异
  - `rank_corr`：两机制排序一致性

图位（自然语言描述，不插图）：

- 此处应插入：Q1 不确定性热力图（`outputs/figures/q1/eps/q1_uncertainty_heatmap.eps`），颜色表示区间宽度或 ESS（信息量）。

---

### 5.2 Model Establishment and Solution of Question Two（Q2：机制对比与反事实）

Q2 的目标不是“拟合历史”本身，而是把两种公开叙事（Percent vs Rank）在淘汰路径上的差异变成可复现的指标，并讨论 Judge Save 纠偏变体。

#### 5.2.1 输入与输出

- 主输出：`outputs/tables/mcm2026c_q2_mechanism_comparison.csv`
- 本次增强额外输出：
  - `outputs/tables/mcm2026c_q2_week_level_comparison_percent.csv`
  - `outputs/tables/mcm2026c_q2_week_level_comparison_rank.csv`
  - `outputs/tables/mcm2026c_q2_fan_source_sensitivity.csv`

#### 5.2.2 关键口径与“误差定位”

- `match_rate_percent` 接近 1 更像**一致性检查**：因为 Q1 的 percent 后验本身由观测淘汰约束驱动。
- rank 机制往往产生更大波动：这是“名次口径”对周内结构噪声更敏感的体现。
- week-level 表可以定位“哪些周导致差异/误差”，并区分：无淘汰周、退赛周、双淘汰周等结构事件。

图位（自然语言描述，不插图）：

- 此处应插入：Q2 机制对比图（见 `outputs/figures/q2/eps/`），展示 percent vs rank 的差异周数、以及 Judge Save 的纠偏效果。

---

### 5.3 Model Establishment and Solution of Question Three（Q3：特征影响分析）

Q3 将结果拆成两条“可解释线”：

- 技术线：`judge_score_pct_mean`
- 人气线：`fan_vote_index_mean`（来自 Q1 后验推断）

并通过 pro dancer 的层级结构（跨季复现）吸收“舞伴效应”。

#### 5.3.1 输入与输出

- 主输出：`outputs/tables/mcm2026c_q3_impact_analysis_coeffs.csv`
- 本次增强额外输出：
  - `outputs/tables/mcm2026c_q3_dataset_diagnostics.csv`
  - `outputs/tables/mcm2026c_q3_fan_refit_coeff_draws.csv`
  - `outputs/tables/mcm2026c_q3_fan_refit_stability.csv`
  - `outputs/tables/mcm2026c_q3_fan_source_sensitivity_quick_ols.csv`

#### 5.3.2 不确定性传播（主线亮点之一）

人气线的因变量来自 Q1 推断而非直接观测，因此我们不把它当作确定值，而是用重复 refit 的方式传播不确定性：

- 每次 refit 对 `fan_vote_index` 做一次随机扰动抽样，再拟合一次模型
- 用 refit 的系数分布形成区间，并用 `sign_consistency` 衡量方向稳定性

图位（自然语言描述，不插图）：

- 此处应插入：Q3 系数稳定性/区间图（见 `outputs/figures/q3/eps/`），展示 refit 后系数区间与方向一致率。

---

### 5.4 Model Establishment and Solution of Question Four（Q4：新系统设计与评估）

Q4 的核心是机制设计（mechanism design）：在 Q1 可识别的粉丝强度空间内，比较机制族在多目标上的 trade-off。

#### 5.4.1 目标分解与指标

- 技术保护：`tpi_season_avg`
- 粉丝表达：`fan_vs_uniform_contrast`（受控对照：真实粉丝 vs 均匀粉丝）
- 鲁棒性：`robust_fail_rate`（压力测试下失败率）
- 稳定性：`champion_mode_prob`、`champion_entropy`

#### 5.4.2 输出

- 主输出：`outputs/tables/mcm2026c_q4_new_system_metrics.csv`

图位（自然语言描述，不插图）：

- 此处应插入：Q4 多目标 trade-off 图（见 `outputs/figures/q4/`），横轴技术保护，纵轴粉丝表达，点大小/颜色表示鲁棒性或冠军熵。

---

## 6 Sensitivity Analysis and Error Analysis（灵敏度与误差分析）

### 6.1 Sensitivity Analysis（灵敏度分析）

主线灵敏度分析遵循“**不改主线口径，只做可复现对照**”的原则：

1. **Q1：Percent vs Rank 的推断敏感性**
   - 产物：`outputs/tables/mcm2026c_q1_mechanism_sensitivity_week.csv`
   - 指标：`tv_distance`（分布差异）、`rank_corr`（排序一致性）
   - 写法：挑选 `tv_distance` 较大、`rank_corr` 较低的周作为案例，解释两种公开叙事在“淘汰边缘选手”上可能产生不同推断.

2. **Q2：fan_source_mechanism 切换敏感性（赛季级）**
   - 产物：`outputs/tables/mcm2026c_q2_fan_source_sensitivity.csv`
   - 写法：用 `delta_match_rate_*` 与 Judge Save 相关差异，说明“机制口径变化会系统性改变反事实结论”.

3. **Q3：percent vs rank 的解释敏感性（快速 OLS 对照）**
   - 产物：`outputs/tables/mcm2026c_q3_fan_source_sensitivity_quick_ols.csv`
   - 写法：对比关键项（如 `is_us`、人口/年龄变量）的符号与显著性是否一致；并强调主线结论更关注“方向一致性 + 区间”，而非追求大量显著项.

4. **支线（showcase）作为“加分型敏感性网格”**
   - 产物（示例）：`outputs/tables/showcase/mcm2026c_showcase_q1_sensitivity_summary.csv`
   - 写法：说明我们在附录中对 `α/τ/m/r` 做了更大网格扫描，用于回答“参数选择是否稳健”，但主线不依赖这些外部实验.

### 6.2 Error Analysis for Question 1（Q1 误差分析）

Q1 的“误差”主要来自可辨识性限制与赛制噪声，而非数值计算不稳定.

- 周级一致性诊断：`outputs/tables/mcm2026c_q1_error_diagnostics_week.csv`
  - `match_pred`：后验均值下的规则回放是否匹配观测
  - `observed_exit_prob_at_posterior_mean`：观测淘汰集合在后验均值下的概率（越小越难解释）
  - `fan_share_width_mean`：区间越宽，说明信息量更低、误差更大

### 6.3 Error Analysis for Question 2（Q2 误差分析）

Q2 的误差定位到 week-level 更有解释力：

- 产物：
  - `mcm2026c_q2_week_level_comparison_percent.csv`
  - `mcm2026c_q2_week_level_comparison_rank.csv`

结构性缺失说明：当 `n_exit=0`（无淘汰周）时，相关预测字段为空是合理现象，不应视为数据错误.

### 6.4 Error Analysis for Question 3（Q3 误差分析）

Q3 的误差分析核心是“不确定性传播是否稳定”：

- 系数 refit 明细：`mcm2026c_q3_fan_refit_coeff_draws.csv`
- 稳定性汇总：`mcm2026c_q3_fan_refit_stability.csv`
  - `iqr`：系数波动范围
  - `sign_consistency`：符号一致率（用于写“稳健性”）

图位（自然语言描述，不插图）：

- 此处应插入：Q3 不确定性传播图（`outputs/figures/q3/eps/q3_uncertainty_propagation.eps`），展示 refit 后各系数区间宽度与方向稳定性.

---

## 7 Model Evaluation and Conclusion（模型评价与结论）

### 7.1 Advantages of the Model（优点）

1. **可复现**：一键生成 `data/processed/` 与 `outputs/`, 每个结论都能回溯到 CSV.
2. **可解释**：主线遵循 KISS, 避免不必要黑盒；Q1–Q4 的每一步都有清晰口径.
3. **不确定性传播**：Q3/Q4 不把 Q1 当成“确定输入”，而是显式传播后验不确定性.
4. **误差与敏感性可写**：新增 Q1/Q2/Q3 诊断表，使论文的 Error/Sensitivity 章节有可复现硬证据.
5. **支线加分但不污染主线**：showcase 提供网格敏感性与 ML baseline 作为附录，凸显工程能力与稳健性.

### 7.2 Disadvantages of the Model（不足与局限）

1. **可辨识性限制**：不声称恢复真实票数，只能估计相对强度.
2. **Rank 口径近似**：粉丝名次用份额排序近似，虽可复现但并非唯一可能.
3. **外部动员极端案例**：主线不引入外部人气 proxy, 需用压力测试刻画风险边界.

### 7.3 Conclusion（结论与建议）

我们建议制作方在现有“评委 + 粉丝”的框架内，优先考虑能抑制极端票数支配的新机制（如对粉丝份额做非线性压缩再合成，或加入 Judge Save 纠偏），并将压力测试结果作为节目风控工具.

---

## 8 References（参考与资料来源）

说明：本题不要求外部数据；以下链接仅用于赛制口径说明与背景引用，主线模型不以其为输入.

- ABC（投票合成说明）：https://abc.com/news/04b80298-dc11-47c0-9f91-adc58c4440b9/category/1074633
- Entertainment Weekly（制作人解释合成方式）：https://ew.com/dwts-producer-reveals-how-scores-and-votes-are-calculated-11857082
- E! Online（50/50 叙事）：https://www.eonline.com/news/1423264/dancing-with-the-stars-eliminations-scores-and-votes-explained
- TVLine（Judge Save 规则解释）：https://www.tvline.com/news/dancing-with-the-stars-judges-save-eliminated-rule-change-explained-1235168027/

---

## 9 Appendix（附录：复现指南与代码索引）

### 9.1 环境

- Python 3.11
- 包管理：`uv`

### 9.2 主线结果表索引

- Q1：`outputs/predictions/mcm2026c_q1_fan_vote_posterior_summary.csv`
- Q1 不确定性：`outputs/tables/mcm2026c_q1_uncertainty_summary.csv`
- Q1 诊断：`outputs/tables/mcm2026c_q1_error_diagnostics_week.csv`, `outputs/tables/mcm2026c_q1_mechanism_sensitivity_week.csv`
- Q2：`outputs/tables/mcm2026c_q2_mechanism_comparison.csv`
- Q2 诊断：`outputs/tables/mcm2026c_q2_week_level_comparison_percent.csv`, `outputs/tables/mcm2026c_q2_week_level_comparison_rank.csv`, `outputs/tables/mcm2026c_q2_fan_source_sensitivity.csv`
- Q3：`outputs/tables/mcm2026c_q3_impact_analysis_coeffs.csv`
- Q3 诊断：`outputs/tables/mcm2026c_q3_dataset_diagnostics.csv`, `outputs/tables/mcm2026c_q3_fan_refit_coeff_draws.csv`, `outputs/tables/mcm2026c_q3_fan_refit_stability.csv`, `outputs/tables/mcm2026c_q3_fan_source_sensitivity_quick_ols.csv`
- Q4：`outputs/tables/mcm2026c_q4_new_system_metrics.csv`

### 9.3 一键复现命令

```bash
uv sync
uv run python run_all.py
```

### 9.4 主线代码索引（正文引用口径）

- Q0：`src/mcm2026/pipelines/mcm2026c_q0_build_weekly_panel.py`
- Q1：`src/mcm2026/pipelines/mcm2026c_q1_smc_fan_vote.py`
- Q2：`src/mcm2026/pipelines/mcm2026c_q2_counterfactual_simulation.py`
- Q3：`src/mcm2026/pipelines/mcm2026c_q3_mixed_effects_impacts.py`
- Q4：`src/mcm2026/pipelines/mcm2026c_q4_design_space_eval.py`

### 9.5 支线（showcase）如何“衬托主线”

支线代码位于 `src/mcm2026/pipelines/showcase/`, 其产物写入 `outputs/tables/showcase/`.

论文写法建议：

- 正文只引用主线产物（确保页数与叙事聚焦）.
- 附录引用 showcase 表作为“额外稳健性/敏感性证据”与“现代化实验编排能力”的展示.

## 10 Report on Use of AI（AI 工具使用说明）

本项目在编码与写作阶段使用了 AI 辅助以提高效率，但所有模型口径与结论链条均以可复现代码与 CSV 产物为准：

- **用途**：协助梳理主线 pipeline、补全 Q1–Q3 误差/敏感性诊断输出、生成论文草稿结构与表述建议.
- **约束**：不引入题外数据作为主线输入；不在无代码与产物支撑的情况下直接给出“数值结论”.
