# 写论文/写代码参考备忘录（审核口径 + 现实机制对齐）

## 0. 目的

本备忘录用于：

- 写论文时提供**可直接引用**的“事实口径 / 审计措辞 / 局限性表述”。
- 写代码时提供**字段定义**与**一致性约束**，避免口径漂移。

说明：`docs/report/` 下的多份审核报告为内部材料，其中部分表述偏“营销式绝对化”（如“零缺失/完美”）。论文中建议使用本备忘录给出的替代措辞。

---

## 1. Q0 处理后数据：事实口径（可直接写进论文/代码注释）

### 1.1 核心产物

- `data/processed/dwts_weekly_panel.csv`：`2777 × 18`
- `data/processed/dwts_season_features.csv`：`421 × 11`

### 1.2 关键字段（weekly panel）

- **评委相关**：`judge_score_total`, `season_week_judge_total`, `judge_score_pct`, `judge_rank`
- **时间索引**：`season`, `week`
- **选手索引**：`celebrity_name`, `pro_name`
- **状态/事件**：`active_flag`, `exit_type`, `elimination_week`, `exit_week_inferred`, `eliminated_this_week`, `withdrew_this_week`

### 1.3 缺失值：必须解释为“结构性缺失”

当前整体缺失率（所有列综合）：

- weekly panel：约 `4.54%`
- season features：约 `2.42%`

关键解释：

- `judge_score_total / judge_score_pct / judge_rank` 等**核心训练特征无缺失**。
- `elimination_week` 与 `exit_week_inferred` 存在缺失：
  - 许多选手 `results` 并非 “Eliminated Week X/Withdrew Week X” 这种可解析文本。
  - 对 `exit_type == unknown` 的选手，我们不强行推断退出周，因此 `exit_week_inferred` 为 NA。

### 1.4 事件类型计数（contestant-season 粒度）

以 `weekly[['season','celebrity_name','exit_type']].drop_duplicates()` 统计：

- `exit_type == eliminated`：298
- `exit_type == withdrew`：10
- `exit_type == unknown`：113（通常对应：决赛名次/未淘汰文本等）

> 注意：这不是“每季淘汰人数”的直接统计，而是 contestant-season 粒度的类别。

---

## 2. 方案B（推断退出周）：定义、合理性、风险与论文措辞

### 2.1 定义（可写进方法/数据预处理章节）

我们引入：

- `exit_type ∈ {eliminated, withdrew, unknown}`
- `exit_week_inferred`（可空 Int）

推断规则（只对 `exit_type != unknown` 生效）：

- 若 `results` 可解析出 `elimination_week = k` 且 `k <= last_active_week`，则 `exit_week_inferred = k`。
- 若 `results` 无周次（如仅 `Withdrew`）或出现冲突（`k > last_active_week`），则回退：`exit_week_inferred = last_active_week`。

事件标记：

- `eliminated_this_week = (exit_type == eliminated) & (week == exit_week_inferred)`
- `withdrew_this_week = (exit_type == withdrew) & (week == exit_week_inferred)`

### 2.2 合理性（为什么用 last_active_week）

- 题目数据中存在 `Withdrew` 但不含周次的记录；若不补全，反事实仿真（Q2/Q4）会出现“事件无法落在某一周”的断裂。
- 也存在 “文本写 Elimination Week k 但当周评分为 0” 的冲突情况。以评分序列为准回退到 `last_active_week` 是一个**可复现**、**最小主观性**的处理。

### 2.3 风险与缓解（必须写进局限性/稳健性）

- **风险**：真实退出可能发生在最后一次有效评分之后的“信息周/制作安排周”，题面数据无法验证。
- **缓解**：
  - 同时保留 `elimination_week`（文本可解析时）与 `exit_week_inferred`（模型/仿真用），并在论文中透明披露。
  - 在 Q2/Q4 做敏感性：将 `exit_week_inferred` 作 `±1` 周扰动，检验机制比较是否稳健。

### 2.4 推荐论文措辞（替换“完美/零问题”）

- 推荐：
  - “关键训练特征（评委总分/份额/排名）无缺失，且通过一致性校验；少量缺失集中在结构性字段（如退出周次的文本不可解析）。”
- 不建议：
  - “零缺失、零残缺、完美一致”。

---

## 3. “审核报告”常见夸张表述：建议改句清单（用于写论文）

以下是内部报告常见句式与建议替换。

### 3.1 “零缺失/完美清洁度”

- 建议替换为：
  - “核心建模特征无缺失；缺失主要为结构性缺失（淘汰后周次不再有评分、文本不可解析字段等），不影响主线建模。”

### 3.2 “所有业务逻辑完全一致”

- 建议替换为：
  - “在采用 `exit_week_inferred` 作为事件落点的定义下，事件标记与面板时间轴一致；当文本周次缺失或与评分冲突时，使用可复现规则回退。”

### 3.3 “州人口匹配率 43/45”

- 建议替换为：
  - “对美国选手且 homestate 非空的样本，州人口可完整匹配；非美国或 homestate 缺失样本不强行匹配。”

### 3.4 “需要扩展外生数据（动态获取人气）”

- 建议替换为：
  - “主线不使用 Google Trends / Wikipedia pageviews 等人气 proxy；如作为附录探索，需明确其覆盖不足与可比性问题，且不进入主线结论。”

---

## 4. 现实机制对齐：可引用来源与如何映射到 Q2/Q4

### 4.1 现实机制要点（写进 Q2/Q4 引言/假设）

- DWTS 的淘汰是 **评委评分 + 观众投票** 的合成结果。
- 制作方会在“专业性/公平感”和“观众参与感/节目效果”之间权衡。

### 4.2 可引用来源（写论文时建议引用 2-3 个即可）

- ABC（投票说明，明确“judges’ scores + viewer votes”合成，但不公开细节）：
  - https://abc.com/news/04b80298-dc11-47c0-9f91-adc58c4440b9/category/1074633
- Entertainment Weekly（2025，制作人解释“50/50 effectively”且用 ranking points 合并）：
  - https://ew.com/dwts-producer-reveals-how-scores-and-votes-are-calculated-11857082
- E! Online（2025，“50% judges scores + 50% viewer votes”的解释，百分比/pie 叙事）：
  - https://www.eonline.com/news/1423264/dancing-with-the-stars-eliminations-scores-and-votes-explained
- TVLine（2024，解释 judge save 与“should be about America”的观众主权动因）：
  - https://www.tvline.com/news/dancing-with-the-stars-judges-save-eliminated-rule-change-explained-1235168027/

### 4.3 映射到我们的 Q2/Q4

- **Q2（机制对比）**：
  - 将“评委权重”设为敏感性参数 `alpha`（例如 0.4–0.6 区间），对比 rank / percentage 合并方式。
- **Q4（新系统设计）**：
  - 现实中 judge save 的引入/取消反映两类目标冲突：
    - 公平/专业性（防止“低技术高人气”极端结果）
    - 观众主权/参与感（“America decides”）
  - 建议把这两类目标显式写进多目标评价：公平性、鲁棒性、可解释性、参与感（或“节目效果/争议度”）。

---

## 5. 写代码时的落地提醒（避免口径漂移）

- 所有主线脚本读取 `data/processed/dwts_weekly_panel.csv` / `dwts_season_features.csv`，不要直接从 raw wide 表推特征。
- 事件落点使用 `exit_week_inferred`（若要使用 `elimination_week`，必须先检查是否 NA，且处理冲突）。
- 任何引入外生数据的尝试必须是“附录/炫技模块”，并在输出文件名中体现 method tag。

- 附录/炫技（showcase）脚本统一放在：`src/mcm2026/pipelines/showcase/`。
- 附录/炫技（showcase）产物统一写入：`outputs/tables/showcase/`（以及未来可能的 `outputs/figures/showcase/`）。
