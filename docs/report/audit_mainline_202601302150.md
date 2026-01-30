# Mainline Audit Report (DWTS / MCM 2026 C)

- Timestamp: `202601302150`
- Scope:
  - `src/mcm2026/config/config.yaml`
  - Pipelines: `mcm2026c_q1_smc_fan_vote.py`, `mcm2026c_q2_counterfactual_simulation.py`, `mcm2026c_q3_mixed_effects_impacts.py`, `mcm2026c_q4_design_space_eval.py`
  - Core data: `mcm2026c/2026_MCM_Problem_C_Data.csv`
  - Processed data: `data/processed/dwts_weekly_panel.csv`, `data/processed/dwts_season_features.csv`
  - Main artifacts: `outputs/tables/mcm2026c_q{1..4}_*.csv`, `outputs/predictions/mcm2026c_q1_fan_vote_posterior_summary.csv`
  - Problem statement: `mcm2026c/2026_MCM_Problem_C.md`
  - Project docs: `docs/spec/task.md`, `docs/project_document/Q1.md`–`Q4.md`

## 1. Executive Summary

- **Overall**: The mainline design is consistent with the COMAP prompt: Q1 infers *relative* fan strength (share/index) with uncertainty; Q2 compares rank vs percent; Q3 separates judges vs fans lines; Q4 evaluates alternative mechanisms via simulation and metrics.
- **Reproducibility**: All reviewed pipelines are deterministic given the same inputs/config. Q1 uses a deterministic per-week seed function; Q3/Q4 use configured seeds.
- **Key risks found**:
  - **[Q4 output/config mismatch]** Current `outputs/tables/mcm2026c_q4_new_system_metrics.csv` reflects a **smoke run** (`n_sims=2`, `outlier_mult=2.0` only), but `config.yaml` specifies `n_sims=50` and `outlier_mults=[2.0,5.0,10.0]`. This is not a code bug, but it means the artifact currently in `outputs/tables/` is **not the intended mainline run**.
  - **[Q1 identifiability low weeks]** There are **71/670** season-week-mechanism rows with `ess_ratio < 0.1` (minimum ~`0.0005`). This is expected in hard-to-identify weeks (multi-exit / strong constraints), but it should be explicitly documented as “low identifiability weeks”.
  - **[Q2 is not independent validation]** `match_rate_percent = 1.0` across seasons is plausible because Q2 consumes Q1 posterior means that were inferred *using the elimination constraint* (soft constraint). This should be presented as **internal consistency** (not prediction).
  - **[Q3 MixedLM fallback]** Q3 currently uses **OLS** for both judge line and fan line (MixedLM fell back). This is acceptable and more stable, but it should be described as “MixedLM attempted; fell back due to instability” to avoid overclaiming hierarchical inference.

## 2. Configuration Audit (`src/mcm2026/config/config.yaml`)

- **[Parsing correctness]** `yaml.safe_load` succeeds; config keys detected: `q1`, `q2`, `q3`, `q4`.
- **[Values used by code]**
  - Q1 reads `alpha`, `tau`, `prior_draws_m`, `posterior_resample_r`.
  - Q2 reads `q1.alpha`, plus `q2.fan_source_mechanism`, `q2.count_withdraw_as_exit`.
  - Q3 reads `q3.fan_source_mechanism`, `q3.n_refits`, `q3.seed`.
  - Q4 reads `q4.fan_source_mechanism`, `q4.n_sims`, `q4.seed`, `q4.outlier_mults`, plus optional `alpha_grid/mechanisms/seasons`.
- **[Doc/comment safety]** Comments are verbose but do not affect parsing. Current file is valid YAML.

## 3. Data & Preprocessing Consistency (Prompt vs Processed Data)

- **[Prompt alignment]** Official data is the wide table with `weekX_judgeY_score` and N/A/0 semantics. The pipeline operates on a derived weekly panel, which is consistent with the problem statement and project spec.
- **[Processed weekly panel sanity]**
  - `outputs/tables/mcm2026c_q0_sanity_season_week.csv` shows `judge_score_pct` sums to ~1.0 per season-week (floating error ~1e-15).
  - Exit count checks:
    - `weekly_panel exit_cnt max = 3`
    - `weeks exit_cnt > 2 = 2`
  - This matches Q1’s implementation limits (k up to 3 is supported).

## 4. Q1 Audit — Fan Vote Estimation (`mcm2026c_q1_smc_fan_vote.py`)

### 4.1 Modeling logic & prompt compliance

- **[What Q1 estimates]** Relative fan share `fan_share_*` and a stabilized index `fan_vote_index_*` (logit space), **not absolute votes** (prompt-compliant).
- **[Mechanisms]** Implements both:
  - `percent`: combines `judge_score_pct` and sampled `pF` shares.
  - `rank`: combines `judge_rank` and derived `fan_rank`.
- **[Soft constraint]** Uses `softmax` likelihood controlled by `tau`, consistent with the written spec.
- **[Edge cases]**
  - `n_exit=0`: likelihood is set to 1.0 (no constraint week). This is reasonable, but means those weeks convey no elimination information.
  - Multi-exit: exact set probability is computed via a small-k without-replacement model; supported up to k=6 (non-vectorized) and k<=2 vectorized.

### 4.2 Stability signals in outputs

From `outputs/tables/mcm2026c_q1_uncertainty_summary.csv`:

- **[Coverage]** `670` rows = seasons 1–34, both mechanisms.
- **[ESS ratio]**
  - `ess_ratio < 0.1`: `71` rows
  - `ess_ratio` median ~`0.491`
  - `ess_ratio` minimum ~`0.0005`

**Interpretation**: the pipeline correctly exposes identifiability variation via ESS. For the writeup, highlight low-ESS weeks as “high uncertainty / weakly identifiable”.

### 4.3 Recommendations (mainline-safe)

- **[Report-level]** In the paper, explicitly state that `ESS_ratio` is a diagnostic for identifiability; show a histogram or at least summary stats.
- **[Config-level]** If you want fewer low-ESS weeks, increase `prior_draws_m` (e.g., 2000 → 5000) and/or consider a slightly larger `tau` (but that changes the posterior).

## 5. Q2 Audit — Mechanism Comparison (`mcm2026c_q2_counterfactual_simulation.py`)

### 5.1 What it does

- Pulls Q1 posterior summary, selects `fan_share_mean` for the configured mechanism (`q2.fan_source_mechanism`).
- Computes predicted eliminated set each week under:
  - `percent` combine
  - `rank` combine
  - `judge_save` variants for single-exit weeks

### 5.2 What the current outputs mean (important)

From `outputs/tables/mcm2026c_q2_mechanism_comparison.csv`:

- **[Average match rates]**
  - `match_rate_percent` mean: `1.0`
  - `match_rate_rank` mean: `~0.848`
  - `match_rate_percent_judge_save` mean: `~0.477`
  - `match_rate_rank_judge_save` mean: `~0.594`
- **[Percent vs Rank difference]** `22 / 34` seasons have at least one week where percent vs rank prediction differs.

**Interpretation caution**:
- `match_rate_percent=1.0` should be described as “Q1-derived posterior mean produces eliminations consistent with the observed eliminations under the same combine rule”, i.e., an **internal consistency** check.
- It should NOT be presented as independent validation of true fan voting.

### 5.3 Recommendations

- **[Paper-level]** Frame Q2 as: “Given the fan-strength distributions consistent with eliminations (Q1), percent vs rank create different counterfactual eliminations / finalists in many seasons.”
- **[Optional robustness]** If you want Q2 to be less tautological, use posterior sampling (multiple draws) rather than only `fan_share_mean`.

## 6. Q3 Audit — Impact Modeling (`mcm2026c_q3_mixed_effects_impacts.py`)

### 6.1 Dataset construction

- Builds a season-level dataset by aggregating weekly panel to:
  - `judge_score_pct_mean` (technical line)
  - `fan_vote_index_mean` (fan line, from Q1)
- Drops rows missing required fields; current `n_obs=421` (matches `dwts_season_features.csv`).

### 6.2 Model stability & recent hardening

- **[MixedLM fallback]** The pipeline tries MixedLM and falls back to OLS when unstable (non-converged / singular covariance / invalid `cov_re`).
- **[Refit consistency]** Fan-line refits are forced to use the same model kind after the first refit draw.

Current artifact check (`outputs/tables/mcm2026c_q3_impact_analysis_coeffs.csv`):
- `judge_line` model: `ols`
- `fan_line_posterior_refit` model: `ols`

This is acceptable and stable, but should be stated clearly as “MixedLM attempted but unstable on this dataset”.

### 6.3 Noted minor data-quality issue (non-fatal)

- **[Industry label duplication]** There are both `Social Media Personality` and `Social media personality` categories in coefficients. This is a data cleaning issue (case normalization) and can be fixed upstream, but would change outputs.

## 7. Q4 Audit — New System Evaluation (`mcm2026c_q4_design_space_eval.py`)

### 7.1 Mechanisms & metrics

- Evaluates mechanisms: `percent`, `rank`, `percent_judge_save`, `percent_sqrt`, `percent_log`, `percent_cap`, `dynamic_weight`.
- Key metrics per `season × mechanism × alpha × outlier_mult`:
  - `champion_mode_prob`, `champion_entropy`
  - `tpi_season_avg`
  - `fan_vs_uniform_contrast`
  - `robust_fail_rate` (currently defined vs `top_judge_final`)

### 7.2 Stability fix applied (edge case)

- **[Fixed]** Prevented a pathological case where a week’s `k_elim` could eliminate all remaining active contestants in simulation, producing blank `champion_mode`.

### 7.3 Current artifact mismatch (must address before final submission)

- `config.yaml` expects:
  - `n_sims: 50`
  - `outlier_mults: [2.0, 5.0, 10.0]`
- Current `outputs/tables/mcm2026c_q4_new_system_metrics.csv` shows:
  - `n_sims = 2` and only `outlier_mult = 2.0`

**Conclusion**: This file is currently a smoke-run artifact. Before paper finalization, regenerate mainline outputs via `run_all.py` (or rerun Q4 with no overrides) so the artifact reflects your intended settings.

### 7.4 Recommended metric enhancement (defer unless you approve mainline column changes)

- **[Candidate]** Add extra robustness comparators:
  - Fail vs baseline mechanism champion
  - Fail vs true champion (from `placement==1` in season features)
- This would add new columns to the mainline Q4 table; do only if you accept a main artifact schema change.

## 8. Clear Action Items (minimal + safe)

- **[Must do]** Regenerate `mcm2026c_q4_new_system_metrics.csv` using the config-specified `n_sims/outlier_mults` (current artifact is a smoke run).
- **[Should do]** In the writeup, explicitly state:
  - Q1 identifiability varies; show ESS diagnostics.
  - Q2 match_rate_percent is internal consistency, not independent validation.
  - Q3 MixedLM may be unstable; OLS fallback used for stability.
- **[Nice to have]** Normalize industry labels upstream to avoid duplicated categories in Q3.

## 9. Completion Status

- Mainline code review: completed
- Data/prompt alignment check: completed
- Main remaining issue is **artifact freshness** (Q4 table not matching config), not algorithm correctness.
