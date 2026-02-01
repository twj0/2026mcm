# Publication-grade Visualization Plan (Sci/Nature-style)

## 0. Scope & goals

- **Goal**: produce a small set of **high-density, self-explanatory** figures suitable for a formal paper (Sci/Nature-like): consistent design system, strong hierarchy, and figures that “speak by themselves”.
- **Quantity control**: **4–5 figures per question** (Q1–Q4). Extra plots are **appendix/showcase** only.
- **Reproducibility constraint**:
  - **Mainline figures**: only use
    - `data/processed/*.csv`
    - `outputs/tables/*.csv`
    - `outputs/predictions/*.csv`
  - **Showcase policy (baseline / contrast)**:
    - `outputs/tables/showcase/*.csv` may be used **inside mainline figures** as a **baseline/contrast** to highlight why the mainline approach is stronger.
    - It must **not replace** mainline evidence or the primary numeric claims; treat it as “comparison layer”.
    - In captions/labels, explicitly mark it as **Showcase baseline**.

## 0.1 Decisions (fixed for this paper)

- **Style direction**: **Style B+C fusion** (Sci modern + fancy, still academic)
  - base: minimal, clean, high-contrast typography
  - add-ons: small UI-like callouts, subtle outlines, richer but controlled cues
- **Figure language**: **English-only**

## 1. Figure design system (house style)

### 1.1 Visual hierarchy

- **Title**: short, statement-like (what the figure proves).
- **Subtitle**: 1 line describing dataset/condition (mechanism/season range) and key parameters (alpha/tau if relevant).
- **Panel labels**: `A`, `B`, `C`, … at top-left of each subplot.
- **Annotations**: only for the top 1–3 takeaways (avoid clutter); use light callout boxes.

**B+C fusion details**:

- Callouts should look “UI-like” but restrained:
  - rounded box, very light fill, thin border
  - small pointer/arrow only when necessary
- Prefer **inset panels** for comparisons (e.g., Showcase baseline), so the main plot stays clean.

### 1.2 Color system (consistent across Q1–Q4)

**Mechanism colors** (fixed mapping; never change per-figure):

- `percent`: blue
- `rank`: orange
- `percent_judge_save`: green
- `percent_sqrt`: red
- `percent_log`: purple
- `percent_cap`: magenta
- `dynamic_weight`: neutral gray

**Semantic colors**:

- “good/optimal”: green
- “risk/failure”: red
- “uncertainty/weak evidence”: amber
- “reference/baseline”: gray

### 1.3 Typography & layout

- Serif for body (`Times New Roman` with fallback), STIX math.
- Compact but readable: base font ~9.5pt, titles ~10.5pt.
- Thin grid, no top/right spines, dark text.
- Prefer **single-column** (6×4 in) or **double-column** (12×4.5 in). For multi-panel, use shared axes & aligned margins.

### 1.4 Information density patterns (recommended)

- Combine related ideas into **multi-panel composites** instead of many single plots.
- Use **small multiples** (facet by outlier_mult or mechanism) to show robustness.
- Use **errorbars/intervals** and **calibration overlays** (data uncertainty is part of the story).
- Use **ranked ordering** (sorted bars/forests) so reader can scan quickly.

**B+C fusion patterns**:

- Use **compact side legends** and “key takeaway chips” (mini badges) to reduce verbose text.
- Use **consistent glyph grammar**:
  - filled circle = mainline estimate
  - open circle = showcase baseline
  - x marker = eliminated / failure
  - thin band = uncertainty/CI

### 1.5 Export policy

- Final: `tiff` (300 dpi), `eps`, `pdf`.
- Add lightweight `png` preview for quick iteration.

## 2. Figure shortlist (4–5 per question)

Below is the recommended “paper-body” set. Each item includes: purpose, plot type, data sources, and key encoding.

### Q1 (Fan vote inference & uncertainty)

**Q1-F1. Uncertainty atlas (A/B panels)**
- **Purpose**: show where inference is reliable and where it is weak.
- **Plot**: 2 heatmaps (A: ESS ratio, B: evidence), overlay elimination-week markers.
- **Data**: `outputs/tables/mcm2026c_q1_uncertainty_summary.csv` (filter `mechanism=percent` for mainline).
- **Encoding**: diverging cmap for ESS, sequential for evidence.

**Q1-F2. Posterior intervals: “hard week vs easy week”**
- **Purpose**: show what “uncertainty” means at contestant-level.
- **Plot**: two stacked interval plots (mean + [p05,p95]) sorted by mean; mark eliminated.
- **Data**: `outputs/predictions/mcm2026c_q1_fan_vote_posterior_summary.csv` + uncertainty summary.

**Q1-F3. Technical vs popularity separation (scatter + marginal density)**
- **Purpose**: motivate Q3/Q4: judges ≠ fans.
- **Plot**: scatter (judge_score_pct vs fan_share_mean), with eliminated highlighted; add marginal KDEs if space.
- **Data**: `data/processed/dwts_weekly_panel.csv` + Q1 posterior.

**Q1-F4. Mechanism sensitivity “at a glance” (Percent vs Rank)**
- **Purpose**: show how much the inferred fan share differs between mechanisms.
- **Plot**: distribution of TV distance / rank-corr across weeks; add top-k outlier weeks annotation.
- **Data**: `outputs/tables/mcm2026c_q1_mechanism_sensitivity_week.csv`.

**Optional appendix (Q1-A1)**
- Statistical vs ML baseline comparison (keep as appendix).

**Showcase-as-baseline option (if you want it in the mainline)**:

- Fold a small inset into **Q1-F4**:
  - “Showcase baseline” performance summary as a small bar or dot plot
  - the main plot remains the mainline sensitivity distribution

### Q2 (Counterfactual mechanism comparison)

**Q2-F1. Season-level divergence summary (A/B)**
- **Purpose**: how often Percent vs Rank change eliminations.
- **Plot**: histogram of divergent weeks (A) + season scatter of divergence rate (B).
- **Data**: `outputs/tables/mcm2026c_q2_mechanism_comparison.csv`.

**Q2-F2. Judge Save impact (multi-bar + delta)**
- **Purpose**: quantify Judge Save effect on match rate.
- **Plot**: grouped bars for match rates; annotate average delta.
- **Data**: same as above.

**Q2-F3. Controversy dashboard heatmap**
- **Purpose**: identify controversy seasons; highlight S27.
- **Plot**: standardized metric heatmap.
- **Data**: same as above.

**Q2-F4. Micro-level divergence map (week heatmap)**
- **Purpose**: where differences occur within seasons/weeks.
- **Plot**: season×week heatmap of diff flags or divergence score.
- **Data**: `outputs/tables/mcm2026c_q2_week_level_comparison_percent.csv` if present.

**Optional appendix (Q2-A1)**
- ML mechanism prediction.

**Showcase-as-baseline option (mainline)**:

- If you include ML mechanism prediction, it should be an **inset baseline** in **Q2-F1** (not a standalone full figure).

### Q3 (Impact factors: judge-line vs fan-line)

**Q3-F1. Dual forest plot (judge vs fan)**
- **Purpose**: core narrative: same covariate acts differently on two lines.
- **Plot**: paired forest (two columns share y), key terms + top industries.
- **Data**: `outputs/tables/mcm2026c_q3_impact_analysis_coeffs.csv`.

**Q3-F2. Effect size comparison (ranked bars)**
- **Purpose**: easiest summary for readers.
- **Plot**: side-by-side bars for shared key terms; sorted by |fan| or |judge|.
- **Data**: same.

**Q3-F3. Age effect curves (nonlinear)**
- **Purpose**: show interpretability of age/age^2.
- **Plot**: curves over age range for both lines; include confidence band if available.
- **Data**: same.

**Q3-F4. Uncertainty propagation / stability (bubble)**
- **Purpose**: show Q1 uncertainty propagation into Q3 results.
- **Plot**: bubble: sign-consistency vs IQR, size=|median|; annotate top terms.
- **Data**: `outputs/tables/mcm2026c_q3_fan_refit_stability.csv`.

**Optional appendix (Q3-A1)**
- ML/DL baseline comparison, refit grid.

**Showcase-as-baseline option (mainline)**:

- In **Q3-F1** (forest) or **Q3-F4** (stability), add a small inset showing “ML baseline predictive score summary” as contrast.

### Q4 (New mechanism design)

**Q4-F1. Trade-off map (small multiples by outlier_mult)**
- **Purpose**: core decision: TPI vs fan-expression vs robustness.
- **Plot**: per-outlier panel; x=fan_vs_uniform_contrast, y=tpi; size=robustness; errorbars if available.
- **Data**: `outputs/tables/mcm2026c_q4_new_system_metrics.csv`.

**Q4-F2. Robustness curves (with CI)**
- **Purpose**: how quickly each mechanism degrades under pressure.
- **Plot**: fail rate vs outlier_mult with band.
- **Data**: same.

**Q4-F3. Champion uncertainty (entropy + mode prob)**
- **Purpose**: entertainment vs determinism.
- **Plot**: boxplots or ridge-style; plus scatter entropy vs TPI.
- **Data**: same.

**Q4-F4. Decision guide (clean infographic)**
- **Purpose**: policy recommendation “ready to paste into paper”.
- **Plot**: minimal decision tree + small metric table.
- **Data**: same (aggregate at baseline outlier_mult).

**Optional appendix (Q4-A1)**
- 3D Pareto frontier; ML feature importance.

**Showcase-as-baseline option (mainline)**:

- Keep 3D Pareto as appendix, but you may add a **2D inset** (showcase Pareto points) into **Q4-F1** to visually justify your selected mechanism.

## 3. Implementation rules (code-level)

### 3.1 Single source of truth

- All rcParams / colors / colormaps come from `src/mcm2026/visualizations/config.py`.
- Each `q*_visualizations.py` should **not hardcode** colors except for rare, semantic reasons.

### 3.2 Multi-panel composition utilities (recommended)

- Provide helper(s) for:
  - panel labels
  - consistent legends
  - consistent axis formatting
  - consistent annotation box style

### 3.3 Figure naming

- Keep stable filenames matching docs:
  - `q1_uncertainty_heatmap`, `q1_fan_share_intervals`, `q1_judge_vs_fan_scatter`, `q1_mechanism_sensitivity_overview`
  - `q2_mechanism_difference_distribution`, `q2_judge_save_impact`, `q2_controversial_seasons_heatmap`, `q2_week_level_divergence_heatmap`
  - `q3_judge_vs_fan_forest_plot`, `q3_effect_size_comparison`, `q3_age_effect_curves`, `q3_refit_stability_bubble`
  - `q4_mechanism_tradeoff_scatter`, `q4_robustness_curves`, `q4_champion_uncertainty_analysis`, `q4_mechanism_recommendation`

## 4. Implementation milestones (to keep changes controlled)

1. **Style system**: centralize palettes, colormaps, panel labels, and callout helpers in `VisualizationConfig`.
2. **Reduce figure set**: keep only the 4–5 core figures per question.
3. **Showcase baseline**: integrate as insets/overlays (explicitly labeled), not as standalone figures.
4. **One-command regeneration**: a single runner that regenerates Q1–Q4 figures into `outputs/figures/`.

**Style direction**: Fused Style B+C (slightly stronger contrasts, subtle outlines, clearer annotations, small UI-like callouts, slightly richer palette, more visual cues).

**Figure labels**: English-only.

**Showcase-as-baseline policy**: Showcase baseline can appear in mainline as comparison, not replacing mainline evidence.
