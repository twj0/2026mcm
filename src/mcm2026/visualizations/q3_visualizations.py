"""
Q3 Visualization Module: Impact Factor Analysis and Judge vs Fan Line Comparison

This module implements all visualization functions for Q3 analysis including:
- Judge vs Fan coefficients forest plots
- Effect size comparisons
- Age effect curves
- Industry category impact heatmaps
- Mixed effects vs ML comparisons
- Uncertainty propagation effects
- Pro dancer random effects
- Interaction effects analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Tuple, List, Optional, Dict
import warnings
warnings.filterwarnings('ignore')

try:
    from .config import VisualizationConfig, create_output_directories, save_figure_with_config
except ImportError:  # pragma: no cover
    from mcm2026.visualizations.config import (
        VisualizationConfig,
        create_output_directories,
        save_figure_with_config,
    )


def create_q3_judge_vs_fan_forest_plot(
    coeffs_data: pd.DataFrame,
    output_dirs: Dict[str, Path],
    config: VisualizationConfig
) -> None:
    """
    Create forest plot comparing judge vs fan line coefficients.
    
    Args:
        coeffs_data: DataFrame with coefficient estimates
        output_dir: Directory to save figures
        figsize: Figure size tuple
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=config.get_figure_size('large_figure'), sharey=True)
    
    # Select key terms for comparison
    key_terms = ['age', 'age_sq', 'is_us', 'log_state_pop', 'n_weeks_active']
    
    # Add industry terms if available
    industry_terms = [term for term in coeffs_data['term'].unique() 
                     if term.startswith('C(industry)[T.') and 
                     term in ['C(industry)[T.Actor]', 'C(industry)[T.Athlete]', 'C(industry)[T.Singer]']]
    key_terms.extend(industry_terms)
    
    # Left plot: Judge line (technical)
    judge_data = coeffs_data[coeffs_data['outcome'] == 'judge_score_pct_mean']
    judge_key = judge_data[judge_data['term'].isin(key_terms)].sort_values('estimate')
    
    if len(judge_key) > 0:
        y_pos = np.arange(len(judge_key))
        ax1.errorbar(judge_key['estimate'], y_pos, 
                    xerr=[judge_key['estimate'] - judge_key['ci_low'],
                          judge_key['ci_high'] - judge_key['estimate']],
                    fmt='o', capsize=4, capthick=1.6, color=config.get_color('primary'), markersize=7)
        
        # Add significance markers
        for i, (_, row) in enumerate(judge_key.iterrows()):
            if row['p_value'] < 0.001:
                ax1.text(row['ci_high'] + 0.01, i, '***', va='center', fontweight='bold')
            elif row['p_value'] < 0.01:
                ax1.text(row['ci_high'] + 0.01, i, '**', va='center', fontweight='bold')
            elif row['p_value'] < 0.05:
                ax1.text(row['ci_high'] + 0.01, i, '*', va='center', fontweight='bold')
        
        ax1.axvline(x=0, color=config.get_color('muted'), linestyle='--', alpha=0.85, linewidth=1.2)
        
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels([term.replace('C(industry)[T.', '').replace(']', '') 
                            for term in judge_key['term']])
        ax1.set_xlabel('Coefficient estimate')
        ax1.set_title('Judge line (technical)', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
    
    # Right plot: Fan line (popularity)
    fan_data = coeffs_data[coeffs_data['outcome'] == 'fan_vote_index_mean']
    fan_key = fan_data[fan_data['term'].isin(key_terms)].sort_values('estimate')
    
    if len(fan_key) > 0:
        y_pos = np.arange(len(fan_key))
        ax2.errorbar(fan_key['estimate'], y_pos, 
                    xerr=[fan_key['estimate'] - fan_key['ci_low'],
                          fan_key['ci_high'] - fan_key['estimate']],
                    fmt='o', capsize=4, capthick=1.6, color=config.get_color('danger'), markersize=7)
        
        # Add significance markers
        for i, (_, row) in enumerate(fan_key.iterrows()):
            if row['p_value'] < 0.001:
                ax2.text(row['ci_high'] + 0.01, i, '***', va='center', fontweight='bold')
            elif row['p_value'] < 0.01:
                ax2.text(row['ci_high'] + 0.01, i, '**', va='center', fontweight='bold')
            elif row['p_value'] < 0.05:
                ax2.text(row['ci_high'] + 0.01, i, '*', va='center', fontweight='bold')
        
        ax2.axvline(x=0, color=config.get_color('muted'), linestyle='--', alpha=0.85, linewidth=1.2)
        
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels([term.replace('C(industry)[T.', '').replace(']', '') 
                            for term in fan_key['term']])
        ax2.set_xlabel('Coefficient estimate')
        ax2.set_title('Fan line (popularity)', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()

    save_figure_with_config(fig, 'q3_judge_vs_fan_forest_plot', output_dirs, config)


def create_q3_effect_size_comparison(
    coeffs_data: pd.DataFrame,
    output_dirs: Dict[str, Path],
    config: VisualizationConfig
) -> None:
    """
    Create effect size comparison plot for key factors.
    
    Args:
        coeffs_data: DataFrame with coefficient estimates
        output_dir: Directory to save figures
        figsize: Figure size tuple
    """
    fig, ax = plt.subplots(figsize=config.get_figure_size('single_column'))
    
    # Get common terms between judge and fan lines
    judge_data = coeffs_data[coeffs_data['outcome'] == 'judge_score_pct_mean']
    fan_data = coeffs_data[coeffs_data['outcome'] == 'fan_vote_index_mean']
    
    judge_terms = set(judge_data['term'])
    fan_terms = set(fan_data['term'])
    common_terms = list(judge_terms & fan_terms)
    
    # Filter to key terms
    key_terms = [term for term in common_terms if term in 
                ['age', 'is_us', 'log_state_pop', 'n_weeks_active'] or 
                term.startswith('C(industry)[T.')]
    key_terms = key_terms[:8]  # Limit to avoid crowding
    
    if len(key_terms) > 0:
        judge_effects = []
        fan_effects = []
        
        for term in key_terms:
            judge_coef = judge_data[judge_data['term'] == term]['estimate']
            fan_coef = fan_data[fan_data['term'] == term]['estimate']
            
            if len(judge_coef) > 0 and len(fan_coef) > 0:
                judge_effects.append(judge_coef.iloc[0])
                fan_effects.append(fan_coef.iloc[0])
            else:
                judge_effects.append(0)
                fan_effects.append(0)
        
        x = np.arange(len(key_terms))
        width = 0.35
        
        # Create bars
        bars1 = ax.bar(x - width/2, judge_effects, width, 
                      label='Judge line', color=config.get_color('primary'), alpha=0.82, edgecolor=config.get_color('text'), linewidth=0.35)
        bars2 = ax.bar(x + width/2, fan_effects, width, 
                      label='Fan line', color=config.get_color('danger'), alpha=0.78, edgecolor=config.get_color('text'), linewidth=0.35)
        
        # Add value labels
        for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
            height1 = bar1.get_height()
            height2 = bar2.get_height()
            ax.text(bar1.get_x() + bar1.get_width()/2., height1 + 0.01 if height1 >= 0 else height1 - 0.01,
                   f'{height1:.3f}', ha='center', va='bottom' if height1 >= 0 else 'top', fontsize=10, color=config.get_color('text'))
            ax.text(bar2.get_x() + bar2.get_width()/2., height2 + 0.01 if height2 >= 0 else height2 - 0.01,
                   f'{height2:.3f}', ha='center', va='bottom' if height2 >= 0 else 'top', fontsize=10, color=config.get_color('text'))
        
        ax.set_xlabel('Term')
        ax.set_ylabel('Coefficient estimate')
        ax.set_title('Effect size comparison (judge vs fan)', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([term.replace('C(industry)[T.', '').replace(']', '') 
                           for term in key_terms], rotation=45, ha='right')
        ax.legend()
        ax.axhline(y=0, color=config.get_color('muted'), linestyle='-', alpha=0.55)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()

    save_figure_with_config(fig, 'q3_effect_size_comparison', output_dirs, config)


def create_q3_age_effect_curves(
    coeffs_data: pd.DataFrame,
    output_dirs: Dict[str, Path],
    config: VisualizationConfig
) -> None:
    """
    Create age effect curves showing non-linear relationships.
    
    Args:
        coeffs_data: DataFrame with coefficient estimates
    """
    fig, ax = plt.subplots(figsize=config.get_figure_size('single_column'))
    
    # Extract age coefficients
    judge_data = coeffs_data[coeffs_data['outcome'] == 'judge_score_pct_mean']
    fan_data = coeffs_data[coeffs_data['outcome'] == 'fan_vote_index_mean']
    
    # Get age and age_sq coefficients
    judge_age = judge_data[judge_data['term'] == 'age']['estimate']
    judge_age_sq = judge_data[judge_data['term'] == 'age_sq']['estimate']
    fan_age = fan_data[fan_data['term'] == 'age']['estimate']
    fan_age_sq = fan_data[fan_data['term'] == 'age_sq']['estimate']
    
    if len(judge_age) > 0 and len(judge_age_sq) > 0 and len(fan_age) > 0 and len(fan_age_sq) > 0:
        age_range = np.linspace(18, 65, 100)
        
        # Calculate age effects
        judge_age_effect = judge_age.iloc[0] * age_range + judge_age_sq.iloc[0] * (age_range ** 2)
        fan_age_effect = fan_age.iloc[0] * age_range + fan_age_sq.iloc[0] * (age_range ** 2)
        
        # Plot curves
        ax.plot(age_range, judge_age_effect, '-', linewidth=2.6, label='Judge line', alpha=0.9, color=config.get_color('primary'))
        ax.plot(age_range, fan_age_effect, '-', linewidth=2.6, label='Fan line', alpha=0.9, color=config.get_color('danger'))
        
        # Calculate and mark optimal ages
        if judge_age_sq.iloc[0] != 0:
            judge_optimal_age = -judge_age.iloc[0] / (2 * judge_age_sq.iloc[0])
            if 18 <= judge_optimal_age <= 65:
                ax.axvline(x=judge_optimal_age, color=config.get_color('primary'), linestyle='--', alpha=0.8,
                          label=f'Judge line optimal age: {judge_optimal_age:.1f} years')
        
        if len(fan_age) > 0 and len(fan_age_sq) > 0 and fan_age_sq.iloc[0] != 0:
            fan_optimal_age = -fan_age.iloc[0] / (2 * fan_age_sq.iloc[0])
            if 18 <= fan_optimal_age <= 65:
                ax.axvline(x=fan_optimal_age, color=config.get_color('danger'), linestyle='--', alpha=0.8,
                          label=f'Fan line optimal age: {fan_optimal_age:.1f} years')
        
        ax.set_xlabel('Age')
        ax.set_ylabel('Predicted effect (relative)')
        ax.set_title('Nonlinear age effect: judge vs fan line', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()

    save_figure_with_config(fig, 'q3_age_effect_curves', output_dirs, config)


def create_q3_industry_impact_heatmap(
    coeffs_data: pd.DataFrame,
    output_dirs: Dict[str, Path],
    config: VisualizationConfig
) -> None:
    """
    Create industry category impact heatmap.
    
    Args:
        coeffs_data: DataFrame with coefficient estimates
    """
    fig, ax = plt.subplots(figsize=config.get_figure_size('single_column'))
    
    # Extract industry coefficients
    industry_terms = [term for term in coeffs_data['term'].unique() 
                     if term.startswith('C(industry)[T.')]
    
    if len(industry_terms) > 0:
        judge_data = coeffs_data[coeffs_data['outcome'] == 'judge_score_pct_mean']
        fan_data = coeffs_data[coeffs_data['outcome'] == 'fan_vote_index_mean']
        
        industry_matrix = []
        industry_labels = []
        
        for term in industry_terms[:10]:  # Limit to avoid crowding
            industry_name = term.replace('C(industry)[T.', '').replace(']', '')
            judge_coef = judge_data[judge_data['term'] == term]['estimate']
            fan_coef = fan_data[fan_data['term'] == term]['estimate']
            
            if len(judge_coef) > 0 and len(fan_coef) > 0:
                industry_matrix.append([judge_coef.iloc[0], fan_coef.iloc[0]])
                industry_labels.append(industry_name)
        
        if len(industry_matrix) > 0:
            industry_matrix = np.array(industry_matrix)
            
            # Create heatmap
            vmax = float(np.nanquantile(np.abs(industry_matrix).ravel(), 0.95)) if industry_matrix.size else 0.1
            vmax = vmax if np.isfinite(vmax) and vmax > 0 else 0.1
            im = ax.imshow(industry_matrix.T, cmap=config.get_cmap('corr'), aspect='auto', vmin=-vmax, vmax=vmax)
            
            # Set labels
            ax.set_xticks(range(len(industry_labels)))
            ax.set_xticklabels(industry_labels, rotation=45, ha='right')
            ax.set_yticks([0, 1])
            ax.set_yticklabels(['Judge line', 'Fan line'])
            
            # Add value annotations
            for i in range(len(industry_labels)):
                for j in range(2):
                    text = ax.text(i, j, f'{industry_matrix[i, j]:.3f}',
                                  ha="center", va="center", color=config.get_color('text'), fontweight='bold')
            
            # Add colorbar
            cbar = plt.colorbar(im)
            cbar.set_label('Coefficient estimate')
            
            ax.set_title('Industry effects: judge vs fan line', fontweight='bold')
    
    plt.tight_layout()

    save_figure_with_config(fig, 'q3_industry_impact_heatmap', output_dirs, config)


def create_q3_mixed_effects_vs_ml(
    ml_summary: pd.DataFrame,
    dl_summary: pd.DataFrame,
    output_dirs: Dict[str, Path],
    config: VisualizationConfig
) -> None:
    """
    Create mixed effects vs ML comparison (showcase).
    
    Args:
        output_dir: Directory to save figures
        figsize: Figure size tuple
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=config.get_figure_size('double_column'))

    ml_perf = ml_summary.copy()
    ml_perf['source'] = 'ML'
    dl_perf = dl_summary.copy()
    dl_perf['source'] = 'DL'

    perf = pd.concat([ml_perf, dl_perf], ignore_index=True)
    perf = perf.dropna(subset=['r2_mean', 'rmse_mean']).copy()
    perf['model'] = perf['model'].astype(str)

    perf_sorted = perf.sort_values('r2_mean', ascending=False).head(10)
    ax1.barh(perf_sorted['model'], perf_sorted['r2_mean'], color=config.get_color('muted'), alpha=0.85)
    ax1.set_xlabel('R² (mean)')
    ax1.set_title('Showcase: Fan-index prediction (R²)', fontweight='bold')
    ax1.grid(True, alpha=0.3)

    perf_sorted2 = perf.sort_values('rmse_mean', ascending=True).head(10)
    ax2.barh(perf_sorted2['model'], perf_sorted2['rmse_mean'], color=config.get_color('danger'), alpha=0.75)
    ax2.set_xlabel('RMSE (mean)')
    ax2.set_title('Showcase: Fan-index prediction (RMSE)', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()

    save_figure_with_config(fig, 'q3_mixed_effects_vs_ml', output_dirs, config)


def create_q3_uncertainty_propagation(
    refit_grid: pd.DataFrame,
    output_dirs: Dict[str, Path],
    config: VisualizationConfig
) -> None:
    """
    Create uncertainty propagation effects plot.
    
    Args:
        output_dir: Directory to save figures
        figsize: Figure size tuple
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=config.get_figure_size('double_column'))

    grid = refit_grid.copy()
    grid = grid[grid['fan_source_mechanism'] == 'percent'].copy() if 'fan_source_mechanism' in grid.columns else grid
    grid = grid.dropna(subset=['n_refits', 'ci_width_mean', 'outcome']).copy()

    for outcome, ax in [('judge_score_pct_mean', ax1), ('fan_vote_index_mean', ax2)]:
        sub = grid[grid['outcome'] == outcome].sort_values('n_refits')
        if len(sub) == 0:
            continue
        ax.plot(sub['n_refits'], sub['ci_width_mean'], 'o-', linewidth=2, markersize=6, color=config.get_color('primary'))
        ax.set_xlabel('n_refits')
        ax.set_ylabel('Mean CI width')
        ax.set_title(f'Showcase: CI width vs n_refits ({outcome})', fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()

    save_figure_with_config(fig, 'q3_uncertainty_propagation', output_dirs, config)


def create_q3_refit_stability_bubble(
    stability_data: pd.DataFrame,
    showcase_baseline: pd.DataFrame | None,
    output_dirs: Dict[str, Path],
    config: VisualizationConfig,
) -> None:
    df = stability_data.copy()
    if df.empty:
        return

    df = df[df["outcome"].astype(str) == "fan_vote_index_mean"].copy()
    if df.empty:
        return

    df["iqr"] = pd.to_numeric(df["iqr"], errors="coerce")
    df["sign_consistency"] = pd.to_numeric(df["sign_consistency"], errors="coerce")
    df["estimate_median"] = pd.to_numeric(df["estimate_median"], errors="coerce")
    df = df.dropna(subset=["iqr", "sign_consistency", "estimate_median"]).copy()
    if df.empty:
        return

    def _clean_term(t: str) -> str:
        s = str(t)
        if s.startswith("C(industry)[T."):
            return s.replace("C(industry)[T.", "")[:-1]
        return s

    df["term_clean"] = df["term"].astype(str).map(_clean_term)
    df["term_group"] = np.where(df["term"].astype(str).str.startswith("C(industry)[T."), "industry", "core")

    df = df.sort_values("iqr", ascending=False).head(24).copy()

    fig, ax = plt.subplots(figsize=config.get_figure_size("double_column"))

    colors = {"core": config.get_color('primary'), "industry": config.get_color('secondary')}

    size = np.abs(df["estimate_median"].to_numpy(dtype=float))
    size = 40.0 + 220.0 * (size / (np.nanmax(size) if np.nanmax(size) > 0 else 1.0))

    ax.scatter(
        df["sign_consistency"].to_numpy(dtype=float),
        df["iqr"].to_numpy(dtype=float),
        s=size,
        c=[colors.get(g, config.get_color('muted')) for g in df["term_group"].astype(str)],
        alpha=0.78,
        linewidths=0.6,
        edgecolors=config.get_color('text'),
    )

    for _, r in df.head(8).iterrows():
        ax.annotate(
            str(r["term_clean"]),
            (float(r["sign_consistency"]), float(r["iqr"])),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=8,
            color=config.get_color('text'),
        )

    ax.set_xlabel("Sign consistency across refits")
    ax.set_ylabel("Coefficient uncertainty (IQR: q95 - q05)")
    ax.set_title("Q3 fan-line refit stability (direction vs uncertainty)", fontweight="bold")
    ax.set_xlim(0.5, 1.02)
    ax.set_ylim(bottom=0.0)

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=colors["core"], markeredgecolor=config.get_color('text'), markersize=8, label="Core terms"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=colors["industry"], markeredgecolor=config.get_color('text'), markersize=8, label="Industry terms"),
    ]
    ax.legend(handles=handles, loc="upper left")

    if showcase_baseline is not None and not showcase_baseline.empty:
        try:
            inset = ax.inset_axes([0.62, 0.64, 0.36, 0.30])
            inset.set_title('Showcase baseline', fontsize=9.0, fontweight='bold')
            sb = showcase_baseline.copy()
            sb['r2_mean'] = pd.to_numeric(sb.get('r2_mean', np.nan), errors='coerce')
            sb['rmse_mean'] = pd.to_numeric(sb.get('rmse_mean', np.nan), errors='coerce')

            rows: list[tuple[str, float]] = []
            if 'model' in sb.columns and 'r2_mean' in sb.columns:
                best = sb.sort_values('r2_mean', ascending=False).head(1)
                if len(best) and np.isfinite(float(best['r2_mean'].iloc[0])):
                    rows.append((f"ML best R² ({str(best['model'].iloc[0])})", float(best['r2_mean'].iloc[0])))

            if 'dl_r2_mean' in sb.columns:
                rows.append(('DL best R²', float(sb['dl_r2_mean'].iloc[0])))

            if rows:
                names = [a for a, _ in rows]
                vals = [b for _, b in rows]
                y0 = np.arange(len(vals))
                inset.barh(y0, vals, color=config.get_color('muted'), alpha=0.85)
                inset.set_yticks(y0)
                inset.set_yticklabels(names, fontsize=8.0)
                inset.set_xlim(-1.0, 1.0)
                inset.grid(True, alpha=0.25)
                for i, v in enumerate(vals):
                    if np.isfinite(v):
                        inset.text(float(v) + 0.03, i, f"{float(v):.2f}", va='center', fontsize=8.0)

                config.add_callout(ax, 'Showcase shown as contrast only', loc='lower left', kind='note')
        except Exception:
            pass

    plt.tight_layout()
    save_figure_with_config(fig, "q3_refit_stability_bubble", output_dirs, config)


def generate_all_q3_visualizations(
    data_dir: Path,
    output_dirs: Dict[str, Path],
    config: VisualizationConfig,
    showcase: bool = False,
    mode: str = 'paper'
) -> None:
    """
    Generate all Q3 visualizations.
    
    Args:
        data_dir: Directory containing input data files
        output_dir: Directory to save output figures
    """
    print("🎨 Generating Q3 visualizations...")
    
    # Load data
    try:
        coeffs_data = pd.read_csv(data_dir / 'outputs' / 'tables' / 'mcm2026c_q3_impact_analysis_coeffs.csv')
        
        print(f"✅ Loaded data: {len(coeffs_data)} coefficient records")

        config.apply_matplotlib_style()
        
        mode = str(mode).strip().lower()

        baseline_df = None
        try:
            ml_path = data_dir / 'outputs' / 'tables' / 'showcase' / 'mcm2026c_q3_ml_fan_index_baselines_cv_summary.csv'
            dl_path = data_dir / 'outputs' / 'tables' / 'showcase' / 'mcm2026c_q3_dl_fan_regression_nets_summary.csv'
            if ml_path.exists():
                baseline_df = pd.read_csv(ml_path)
                if dl_path.exists():
                    dl = pd.read_csv(dl_path)
                    if 'r2_mean' in dl.columns:
                        baseline_df = baseline_df.assign(dl_r2_mean=float(dl['r2_mean'].iloc[0]))
        except Exception:
            baseline_df = None

        # Generate visualizations (paper mode = 4 core figures)
        create_q3_judge_vs_fan_forest_plot(coeffs_data, output_dirs, config)
        print("✅ Created judge vs fan forest plot")
        
        create_q3_effect_size_comparison(coeffs_data, output_dirs, config)
        print("✅ Created effect size comparison")
        
        create_q3_age_effect_curves(coeffs_data, output_dirs, config)
        print("✅ Created age effect curves")

        stability_path = data_dir / 'outputs' / 'tables' / 'mcm2026c_q3_fan_refit_stability.csv'
        if stability_path.exists():
            stab = pd.read_csv(stability_path)
            create_q3_refit_stability_bubble(stab, baseline_df, output_dirs, config)
            print("✅ Created refit stability bubble plot")

        if mode != 'paper':
            create_q3_industry_impact_heatmap(coeffs_data, output_dirs, config)
            print("✅ Created industry impact heatmap")

        if showcase and mode != 'paper':
            ml_summary = pd.read_csv(
                data_dir
                / 'outputs'
                / 'tables'
                / 'showcase'
                / 'mcm2026c_q3_ml_fan_index_baselines_cv_summary.csv'
            )
            dl_summary = pd.read_csv(
                data_dir
                / 'outputs'
                / 'tables'
                / 'showcase'
                / 'mcm2026c_q3_dl_fan_regression_nets_summary.csv'
            )
            create_q3_mixed_effects_vs_ml(ml_summary, dl_summary, output_dirs, config)
            print("✅ Created showcase ML/DL baseline comparison")

            refit_grid = pd.read_csv(
                data_dir
                / 'outputs'
                / 'tables'
                / 'showcase'
                / 'mcm2026c_showcase_q3_refit_grid.csv'
            )
            create_q3_uncertainty_propagation(refit_grid, output_dirs, config)
            print("✅ Created showcase uncertainty propagation")
        
        print(f"🎉 Q3 visualizations completed! Saved to {output_dirs['tiff']} and {output_dirs['eps']}")
        
    except Exception as e:
        print(f"❌ Error generating Q3 visualizations: {e}")
        raise

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Generate Q3 figures (TIFF + EPS).')
    parser.add_argument('--data-dir', type=Path, default=Path('.'), help='Project root directory')
    parser.add_argument(
        '--ini',
        type=Path,
        default=None,
        help='Optional visualization ini file path (font/dpi overrides)',
    )
    parser.add_argument('--showcase', action='store_true', help='Also generate appendix-only figures')
    parser.add_argument('--mode', type=str, default='paper', help='paper (4 core figs) or full')
    args = parser.parse_args()

    config = VisualizationConfig.from_ini(args.ini) if args.ini is not None else VisualizationConfig()
    output_structure = create_output_directories(args.data_dir / 'outputs' / 'figures', ['Q3'])

    generate_all_q3_visualizations(
        args.data_dir,
        output_structure['Q3'],
        config,
        showcase=args.showcase,
        mode=str(args.mode),
    )