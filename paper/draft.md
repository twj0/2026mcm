
# 2026 MCM/ICM Problem C（DWTS）中文草稿（Markdown，无图片）

> 说明：本草稿用于中文写作与结构对齐；图表将在后续 LaTeX 阶段插入。
>
> 本仓库所有主线结论均可由 `uv run python run_all.py` 复现生成，对应 CSV 产物位于 `outputs/`。

## 摘要（面向评委/制作人的一页式要点）

我们研究《Dancing with the Stars》(DWTS) 中“评委分 + 观众投票”合成淘汰机制下，如何在缺少真实投票数据的前提下：

1. 反推出每周各选手的相对粉丝投票强度，并量化不确定性（Q1）。
2. 在两种已知机制（Percent vs Rank）与“Judge Save”变体下进行反事实比较，解释争议赛季（Q2）。
3. 分析选手特征与职业舞伴（pro dancer）对“技术线（评委分）”与“人气线（粉丝强度）”的影响差异（Q3）。
4. 设计更稳健的新投票结合系统，通过多指标评价与压力测试给出可执行建议（Q4）。

核心挑战是可辨识性（identifiability）：官方数据仅提供评委分与结果，不给投票数。因此我们主线输出的是**相对粉丝份额/指数及其区间**，而非绝对票数。我们采用基于赛制约束的贝叶斯/重要性采样框架，使推断与历史淘汰“概率一致”，并将不确定性向下游任务传播。

最终我们给出：

- `outputs/predictions/mcm2026c_q1_fan_vote_posterior_summary.csv`：Q1 每季每周每人的粉丝份额与指数（均值/分位数）。
- `outputs/tables/mcm2026c_q2_mechanism_comparison.csv`：Q2 机制对比与 Judge Save 反事实指标。
- `outputs/tables/mcm2026c_q3_impact_analysis_coeffs.csv`：Q3 混合效应模型的关键系数与不确定性.
- `outputs/tables/mcm2026c_q4_new_system_metrics.csv`：Q4 新机制族在多指标与多档压力测试下的评价表.

我们推荐制作方优先考虑“对粉丝份额做非线性压缩后再合成”的机制族（如 `percent_log`），并将压力测试结果作为节目风控工具：在极端动员情境下，新机制相较旧机制更能降低“低技术高票完全支配结果”的风险，同时保留观众参与感.

---

## 0. 符号与记号（Notation）

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

## 0.1 数据规模与主线产物概览（可直接写进论文）

- **Q0 处理后周级面板**：`dwts_weekly_panel.csv` 为 `2777 × 18`。
- **Q0 赛季特征表**：`dwts_season_features.csv` 为 `421 × 11`。
- **Q1 不确定性汇总行数**：`mcm2026c_q1_uncertainty_summary.csv` 共 `670` 行（season-week-mechanism 粒度）。
- **Q4 评价表行数**：`mcm2026c_q4_new_system_metrics.csv` 共 `714` 行（34 seasons × 7 mechanisms × 3 outlier 档位）。

这些数量可作为“数据与工程落地已闭环”的硬证据。

---

## 1. 问题重述与建模目标

题目给定 DWTS 多赛季数据：每周评委打分（多名评委）、选手静态信息、以及最终结果/名次文本。题目要求在此基础上完成四项任务：

- **Q1**：估计每周每位选手的粉丝投票强度（相对份额/指数）并量化不确定性.
- **Q2**：比较 Rank 与 Percent 两类合成机制，进行反事实模拟，并分析争议案例与 Judge Save 变体.
- **Q3**：分析选手特征与职业舞伴（pro dancer）对“技术线（评委分）”与“人气线（粉丝强度）”的影响差异.
- **Q4**：设计更稳健的新投票结合系统，通过多指标评价与压力测试给出可执行建议.

我们采用“传统主线 + 现代化加分点（仅附录）”的策略：主线以可解释、可复现的统计建模为核心；深度学习/自动化实验编排作为可选附录，不作为主线输入.

---

## 2. 数据与预处理（Q0）

### 2.1 数据来源与数据政策

- 官方数据：`mcm2026c/2026_MCM_Problem_C_Data.csv`.
- 外生数据：本仓库包含 Google Trends/Wikipedia pageviews/州人口等文件，但**主线不依赖人气 proxy**（覆盖不足或可比性风险），仅将其作为审计记录或附录可能性.

### 2.2 统一“真源表”

我们将官方 wide 表整理为两张建模输入表：

- 周级面板：`data/processed/dwts_weekly_panel.csv`
  - 粒度：`season, week, celebrity_name`.
  - 核心字段：`judge_score_total, judge_score_pct, judge_rank, active_flag, eliminated_this_week, withdrew_this_week` 等.
- 赛季特征表：`data/processed/dwts_season_features.csv`
  - 粒度：`season, celebrity_name`.
  - 包含选手静态特征与舞伴信息，用于 Q3.

### 2.3 质量控制与“结构性缺失”表述

我们对处理后的面板做一致性审计，确保核心建模字段（评委分/份额/排名等）可用；缺失主要集中在“文本不可解析、淘汰后周次不再评分”等结构性字段. 相关 sanity check 表位于：

- `outputs/tables/mcm2026c_q0_sanity_season_week.csv`
- `outputs/tables/mcm2026c_q0_sanity_contestant.csv`

---

## 3. Q1：反推粉丝投票强度（Fan Vote Estimation）

### 3.1 可辨识性与输出形式

由于真实投票数不可观测，本题可稳定识别的是：在给定赛制与评委分的前提下，能够解释历史淘汰结果的**相对粉丝强度**.

我们输出两种量：

- `fan_share`：当周粉丝投票份额（simplex 上，和为 1）.
- `fan_vote_index`：对份额做 logit 变换得到的指数（便于跨周比较），并提供区间.

### 3.2 机制建模：Percent vs Rank

对于每个赛季-周，设当周仍在赛选手集合为 `A`.

- Percent（份额相加）：
  - 已知评委份额 `pJ_i`，未知粉丝份额 `pF_i`.
  - 合成得分 `T_i = α·pJ_i + (1-α)·pF_i`.
- Rank（名次相加）：
  - 已知评委名次 `rJ_i`.
  - 我们用 `pF` 的排序近似粉丝名次 `rF_i`（KISS 做法）.
  - 合成名次 `S_i = α·rJ_i + (1-α)·rF_i`.

其中 `α` 为评委权重.

### 3.3 软约束似然与重要性采样/重采样

硬约束“最低者必淘汰”在现实中可能被制作安排、噪声等扰动. 我们采用温度参数 `τ` 将淘汰规则转为概率（soft constraint）：

- Percent：`Pr(eliminate=i) = softmax(-T_i/τ)`
- Rank：`Pr(eliminate=i) = softmax(S_i/τ)`

对每周从 Dirichlet 先验采样 `pF`，计算观测淘汰集合的似然权重，进行重要性重采样，得到 `pF` 的后验样本，并汇总均值/中位数/5%-95% 分位数.

主要配置位于：`src/mcm2026/config/config.yaml`（主线默认 `α=0.5, τ=0.03`）.

### 3.4 产物与验证

- 后验汇总（主输出）：
  - `outputs/predictions/mcm2026c_q1_fan_vote_posterior_summary.csv`
- 不确定性汇总（每周 ESS/证据等）：
  - `outputs/tables/mcm2026c_q1_uncertainty_summary.csv`

我们使用 ESS、证据（平均似然）等量作为“可辨识性/信息量”指标，并在后续问题（Q3/Q4）中通过抽样传播不确定性.

### 3.5 Q1 算法步骤（伪代码）

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

### 3.6 Q1 关键数值摘要（从输出表抽取）

在主线默认参数下（`α=0.5, τ=0.03, m=2000, r=500`），Q1 不确定性汇总统计如下（来自 `mcm2026c_q1_uncertainty_summary.csv`）：

- `ess_ratio`（有效样本比率）：
  - 均值约 `0.520`
  - 5% 分位约 `0.032`
  - 中位数约 `0.491`
  - 95% 分位约 `1.000`
- `evidence`（平均似然/一致性强度）：
  - 均值约 `0.359`
  - 5% 分位约 `0.016`
  - 中位数约 `0.206`
  - 95% 分位约 `1.000`

解释口径：`ess_ratio` 越高表示该周“淘汰约束对 pF 的可辨识性越强”；`evidence` 越低通常意味着该周存在更强的随机性/多淘汰/退赛等复杂情形，或赛制噪声更难用单一 `τ` 捕捉.

---

## 4. Q2：机制对比与反事实（Rank vs Percent + Judge Save）

### 4.1 反事实框架

Q2 以 Q1 的后验样本为输入：对每个赛季重复模拟整季淘汰过程，分别在不同机制下统计冠军/淘汰路径的概率差异. 由于输入是分布，我们输出的是概率型结论而非确定性断言.

### 4.2 Judge Save 变体

Judge Save 的最小可执行定义：当周先确定 bottom-2，再由评委淘汰评委分更低者（等价于“在 bottom-2 内做技术纠偏”）.

### 4.3 产物

- 赛季级机制对比表：
  - `outputs/tables/mcm2026c_q2_mechanism_comparison.csv`

表中包含机制差异的汇总指标，用于支撑“哪种机制更偏向粉丝/更保护技术”的讨论.

### 4.4 Q2 关键数值摘要（可直接写进论文）

从 `mcm2026c_q2_mechanism_comparison.csv` 汇总（34 个赛季）：

- `match_rate_percent`：均值与各分位均为 `1.000`.
  - 解释口径：这更像**一致性检查**而非外部验证；因为 Q1 的 `percent` 后验本身就是由观测淘汰约束驱动得到，因而“在样本内回放”几乎必然匹配.
- `match_rate_rank`：
  - 均值约 `0.848`
  - 中位数约 `0.875`
  - 5% 分位约 `0.571`，95% 分位约 `1.000`
  - 解释口径：rank 机制对“名次打破平分、周内结构噪声”等更敏感，因此出现较大波动.
- `match_rate_percent_judge_save`：均值约 `0.477`（中位数 `0.500`）.
- `match_rate_rank_judge_save`：均值约 `0.594`（中位数 `0.625`）.
- `diff_weeks_percent_vs_rank`：
  - 均值约 `1.147`
  - 中位数 `1.000`
  - 95% 分位约 `3.000`

这部分结论可用于引出 Q4：制作方在“观众参与感 vs 技术保护”的权衡中，改变合成规则会系统性改变淘汰路径.

---

## 5. Q3：特征影响分析（Celebrities + Pro Dancers）

### 5.1 两条因变量：技术线 vs 人气线

为避免将“技术”和“人气”混为一个黑盒，我们分别建模：

- 技术线：基于 `judge_score_pct` 聚合得到的赛季级表现.
- 人气线：基于 Q1 的 `fan_vote_index` 聚合得到的赛季级人气强度（并传播不确定性）.

### 5.2 混合效应模型与层级结构

pro dancer 会跨季重复出现，属于典型层级结构. 我们使用混合效应（Mixed Effects）建模 pro 的随机效应，并同时吸收赛季差异.

### 5.3 不确定性传播

粉丝线因变量来自 Q1 后验，因此我们用“后验重复拟合”的方式得到系数区间：从 Q1 的区间构造近似噪声，对同一模型重复拟合并汇总.

### 5.4 产物

- 系数与区间表：
  - `outputs/tables/mcm2026c_q3_impact_analysis_coeffs.csv`

该表用于在论文中解释：哪些特征更影响技术线、哪些更影响人气线，以及 pro dancer 的平均加成方向.

### 5.5 Q3 结果写法建议（结合输出表的“能写进论文的句子”）

从 `mcm2026c_q3_impact_analysis_coeffs.csv` 可见：

- 我们对两条 outcome 分别拟合：
  - `judge_score_pct_mean`
  - `fan_vote_index_mean`
- **技术线（judge_score_pct_mean）**中，存在一组稳定且可解释的显著项：
  - 例如 `is_us` 对技术线为正且显著（`estimate ≈ 0.0898, p ≈ 0.003`），可解释为“美国本土选手在评分体系/节目叙事中更可能获得更稳定的评委表现”.
  - 部分行业项对技术线存在显著负向（例如 `Model` 相关项在输出中呈显著负向），可作为“行业类型与技术表现差异”的案例描述.
- **人气线（fan_vote_index_mean）**中，许多项的区间较宽，显著性不稳定：
  - 解释口径应强调：人气线的因变量来自 Q1 后验推断，天然带不确定性；因此我们更关注方向一致性与区间，而非追求大量显著性.
  - 例如 `is_us` 在人气线的估计为正（输出中 `estimate ≈ 1.39` 且区间不跨 0），可作为“人气/观众基础更偏向本土”的结构性信号，但仍需在局限性中说明其并不代表真实投票数.

---

## 6. Q4：新系统设计与多指标评估（Mechanism Design）

### 6.1 设计目标

我们将“更好”的目标分解为可量化维度：

- **技术保护**：技术强者不应被断层人气轻易淘汰.
- **粉丝表达**：观众投票必须对结果有实质影响.
- **鲁棒性**：面对极端动员/断层票，应尽量避免系统性失真.
- **可执行与可解释**：规则简单、参数可调、便于对观众说明.

### 6.2 机制候选

在保留“评委 + 粉丝”两端信息的前提下，我们评估以下可落地机制族（示例）：

- `percent`：基线.
- `rank`：基线.
- `percent_judge_save`：制度化纠偏.
- `percent_sqrt` / `percent_log`：对粉丝份额做非线性压缩后再归一化合成.
- `percent_cap`：对粉丝份额做封顶（winsorize/cap）.
- `dynamic_weight`：随周数逐渐提高评委权重.

### 6.3 关键建模假设与“现实对齐”边界

本模块评估机制的范围是**Q1 可识别的粉丝强度空间**：

- 我们并不声称“预测现实冠军”，而是评估：在给定（可由淘汰约束识别的）粉丝强度不确定性下，不同机制在多目标上的 trade-off.
- 对极端动员型案例（如 S27 Bobby Bones），Q1 可能低估其真实投票动员强度；因此我们在 Q4 通过压力测试显式讨论此类情景.

### 6.4 指标体系（实现口径）

Q4 输出表以赛季-机制为单位，关键指标包括：

- `tpi_season_avg`：技术保护指数（冠军的赛季平均评委分位数），避免只看决赛周造成的小样本问题.
- `fan_vs_uniform_contrast`：受控对照指标（真实粉丝分布 vs 均匀粉丝分布）下冠军是否改变的频率，用于衡量“粉丝端信息是否实质影响结果”.
- `robust_fail_rate`：压力测试下“技术 top-1 被非 top-1 冠军替代”的频率.
- `champion_mode_prob`、`champion_entropy`：稳定性/不确定性（冠军分布是否过度随机）.

### 6.5 多档压力测试与争议赛季复盘

我们对“极端动员”设置多档 outlier 倍数（2×/5×/10×）进行 stress test. 输出表可直接用于定位类似 Bobby Bones 的赛季：

- 产物：`outputs/tables/mcm2026c_q4_new_system_metrics.csv`
- 示例：在 `season=27, mechanism=percent_log, outlier_mult=10` 场景下，Bobby Bones 出现为冠军众数且概率约 `0.28`. 该结论支持我们将其作为“识别边界 + 压力测试可覆盖”的叙事，而非模型完全失败.

### 6.6 Q4 指标口径对齐：FanImpact vs fan_vs_uniform_contrast

在早期文档中我们曾用 `FanImpact` 表述“粉丝票对结果的边际影响”. 严格来说，若要实现“微扰灵敏度”的数学定义，需要对 `pF` 做小扰动并估计 `Pr(win)` 的变化率.

但在当前主线实现中，我们采用更稳健、可复现且更好解释的受控对照实验：

- **真实粉丝分布**：来自 Q1 后验采样.
- **均匀粉丝分布**：作为“无粉丝信息”基线.

因此主线输出的 `fan_vs_uniform_contrast` 应被解读为：**粉丝端信息是否会系统性改变冠军归属**，而不是“局部导数意义上的敏感度”. 论文写法建议：将 `FanImpact` 作为概念目标，主线用 `fan_vs_uniform_contrast` 实现其“是否有影响”的可证据化版本；微扰灵敏度可在附录作为扩展.

### 6.7 Q4 算法步骤（伪代码）

Q4 的核心是“整季仿真 + 多目标汇总”，其流程可写为：

```text
For each season s:
  1) 按周读取该 season 的面板，并建立 week -> DataFrame 的映射
  2) For each mechanism m:
       For each outlier_mult in {2,5,10}:
         Repeat n_sims 次：
           a) 从 Q1 后验为每周采样 pF_{t,*}
           b) 按机制 m 逐周淘汰，得到冠军 champ
           c) 同一随机种子下用“均匀粉丝”再跑一次，得到 champ_uniform
           d) 在 outlier_mult 压力测试下再跑一次，得到 champ_outlier
         汇总：champion_mode_prob / entropy / tpi_season_avg / fan_vs_uniform_contrast / robust_fail_rate
```

### 6.8 Q4 关键数值摘要（跨赛季均值，用于写“trade-off”段落）

我们将 `mcm2026c_q4_new_system_metrics.csv` 按 `(mechanism, outlier_mult)` 聚合（跨赛季平均），得到一个可直接写进论文的 trade-off 证据（以下为 `outlier_mult=2` 时的代表性结果）：

| outlier_mult | mechanism | `tpi_season_avg`（均值） | `fan_vs_uniform_contrast`（均值） | `robust_fail_rate`（均值） |
|---:|---|---:|---:|---:|
| 2 | `rank` | 0.846 | 0.397 | 0.448 |
| 2 | `percent` | 0.767 | 0.674 | 0.733 |
| 2 | `percent_judge_save` | 0.796 | 0.631 | 0.722 |
| 2 | `percent_log` | 0.746 | 0.700 | 0.782 |

解释口径：

- `rank` 在技术保护与鲁棒性上更强，但粉丝表达（对照差异）明显更弱.
- `percent_log` 强化了粉丝端影响，但技术保护与鲁棒性有所下降；它更像“节目效果/观众主权”导向.
- `percent_judge_save` 显著提高技术保护（相对 `percent`），同时粉丝表达不至于归零，可作为制作方更容易接受的折中方案.

争议赛季复盘（S27 Bobby Bones）：在 `percent_log` 且 `outlier_mult=10` 的压力测试场景下，Bobby Bones 成为冠军众数且概率约 `0.28`. 该结论支持我们将其定位为“外部动员型极端案例”：在不引入外部人气 proxy 的主线限制下，我们用压力测试透明地刻画其风险边界.

---

## 7. 讨论：局限性与稳健性

1. **可辨识性限制**：仅凭评委分与淘汰结果，我们只能识别“与历史淘汰兼容的相对粉丝强度”，无法恢复绝对票数，也可能低估外部动员型极端案例.
2. **赛制细节不完全可得**：无淘汰/双淘汰/退赛等事件我们以可复现规则落点处理；论文中需明确这是最小主观性处理.
3. **外生人气 proxy 主线弃用**：为保证可比性与合规性，主线不引入 Google Trends / Wikipedia pageviews 等人气 proxy；这会牺牲对少量“外部流量驱动”赛季的解释能力，但可通过压力测试与附录补充讨论.
4. **参数敏感性**：`α` 与 `τ` 影响 Q1–Q4 的推断强度与“意外淘汰”容忍度；主线使用默认值并建议在附录做敏感性分析.

---

## 8. 复现指南（写进论文附录/README 的版本）

### 8.1 环境

- Python 3.11
- 包管理：`uv`

### 8.2 一键复现

```bash
uv sync
uv run python run_all.py
```

运行后会生成（或覆盖）`data/processed/` 与 `outputs/` 下的主线 CSV 产物。

### 8.3 主线结果表索引

- Q1：`outputs/predictions/mcm2026c_q1_fan_vote_posterior_summary.csv`
- Q1 不确定性：`outputs/tables/mcm2026c_q1_uncertainty_summary.csv`
- Q2：`outputs/tables/mcm2026c_q2_mechanism_comparison.csv`
- Q3：`outputs/tables/mcm2026c_q3_impact_analysis_coeffs.csv`
- Q4：`outputs/tables/mcm2026c_q4_new_system_metrics.csv`

---

## 9. 参考链接（写作时择 2–3 个引用即可）

- ABC（投票合成说明）：https://abc.com/news/04b80298-dc11-47c0-9f91-adc58c4440b9/category/1074633
- Entertainment Weekly（2025，制作人解释合成方式）：https://ew.com/dwts-producer-reveals-how-scores-and-votes-are-calculated-11857082
- E! Online（2025，50/50 叙事）：https://www.eonline.com/news/1423264/dancing-with-the-stars-eliminations-scores-and-votes-explained
- TVLine（Judge Save 解释）：https://www.tvline.com/news/dancing-with-the-stars-judges-save-eliminated-rule-change-explained-1235168027/
