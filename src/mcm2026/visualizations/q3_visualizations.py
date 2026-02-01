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
import matplotlib.patheffects as pe
try:
    import seaborn as sns
except Exception:
    sns = None
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
    
    stroke_fc = str(config.callout_bbox(kind='note').get('facecolor', '#ffffff'))

    if len(judge_key) > 0:
        y_pos = np.arange(len(judge_key))
        for yi in y_pos:
            if int(yi) % 2 == 0:
                ax1.axhspan(float(yi) - 0.5, float(yi) + 0.5, color=config.get_color('muted'), alpha=0.06, zorder=0)

        ax1.scatter(
            judge_key['estimate'],
            y_pos,
            s=120,
            c=config.get_color('primary'),
            alpha=0.14,
            linewidths=0.0,
            zorder=1,
        )
        ax1.errorbar(judge_key['estimate'], y_pos, 
                    xerr=[judge_key['estimate'] - judge_key['ci_low'],
                          judge_key['ci_high'] - judge_key['estimate']],
                    fmt='o', capsize=4, capthick=1.6, color=config.get_color('primary'), markersize=7, zorder=3)
        
        # Add significance markers
        for i, (_, row) in enumerate(judge_key.iterrows()):
            if row['p_value'] < 0.001:
                t = ax1.text(row['ci_high'] + 0.01, i, '***', va='center', fontweight='bold')
                t.set_path_effects([pe.withStroke(linewidth=2.4, foreground=stroke_fc)])
            elif row['p_value'] < 0.01:
                t = ax1.text(row['ci_high'] + 0.01, i, '**', va='center', fontweight='bold')
                t.set_path_effects([pe.withStroke(linewidth=2.4, foreground=stroke_fc)])
            elif row['p_value'] < 0.05:
                t = ax1.text(row['ci_high'] + 0.01, i, '*', va='center', fontweight='bold')
                t.set_path_effects([pe.withStroke(linewidth=2.4, foreground=stroke_fc)])
        
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
        for yi in y_pos:
            if int(yi) % 2 == 0:
                ax2.axhspan(float(yi) - 0.5, float(yi) + 0.5, color=config.get_color('muted'), alpha=0.06, zorder=0)

        ax2.scatter(
            fan_key['estimate'],
            y_pos,
            s=120,
            c=config.get_color('danger'),
            alpha=0.14,
            linewidths=0.0,
            zorder=1,
        )
        ax2.errorbar(fan_key['estimate'], y_pos, 
                    xerr=[fan_key['estimate'] - fan_key['ci_low'],
                          fan_key['ci_high'] - fan_key['estimate']],
                    fmt='o', capsize=4, capthick=1.6, color=config.get_color('danger'), markersize=7, zorder=3)
        
        # Add significance markers
        for i, (_, row) in enumerate(fan_key.iterrows()):
            if row['p_value'] < 0.001:
                t = ax2.text(row['ci_high'] + 0.01, i, '***', va='center', fontweight='bold')
                t.set_path_effects([pe.withStroke(linewidth=2.4, foreground=stroke_fc)])
            elif row['p_value'] < 0.01:
                t = ax2.text(row['ci_high'] + 0.01, i, '**', va='center', fontweight='bold')
                t.set_path_effects([pe.withStroke(linewidth=2.4, foreground=stroke_fc)])
            elif row['p_value'] < 0.05:
                t = ax2.text(row['ci_high'] + 0.01, i, '*', va='center', fontweight='bold')
                t.set_path_effects([pe.withStroke(linewidth=2.4, foreground=stroke_fc)])
        
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
    config: VisualizationConfig,
    fan_refit_draws: Optional[pd.DataFrame] = None,
) -> None:
    """
    Create age effect curves showing non-linear relationships.
    
    Args:
        coeffs_data: DataFrame with coefficient estimates
    """
    w, h = config.get_figure_size('single_column')
    fig, (ax_j, ax_f) = plt.subplots(
        2,
        1,
        figsize=(float(w), float(h) * 1.30),
        sharex=True,
        gridspec_kw={'hspace': 0.14, 'height_ratios': [1.0, 1.0]},
    )

    stroke_fc = str(config.callout_bbox(kind='note').get('facecolor', '#ffffff'))
    age_range = np.linspace(18.0, 65.0, 240)
    center_age = 35.0

    # Extract age coefficients
    judge_data = coeffs_data[coeffs_data['outcome'] == 'judge_score_pct_mean'].copy()
    fan_data = coeffs_data[coeffs_data['outcome'] == 'fan_vote_index_mean'].copy()

    for c in ['estimate', 'std_err', 'ci_low', 'ci_high']:
        if c in judge_data.columns:
            judge_data[c] = pd.to_numeric(judge_data[c], errors='coerce')
        if c in fan_data.columns:
            fan_data[c] = pd.to_numeric(fan_data[c], errors='coerce')

    def _get_row(d: pd.DataFrame, term: str) -> Optional[pd.Series]:
        s = d[d['term'].astype(str) == str(term)]
        if len(s) == 0:
            return None
        return s.iloc[0]

    def _vertex(b1: float, b2: float) -> Optional[float]:
        if not np.isfinite(b1) or not np.isfinite(b2) or float(b2) == 0.0:
            return None
        x = -float(b1) / (2.0 * float(b2))
        if 18.0 <= x <= 65.0:
            return float(x)
        return None

    def _curve(b1: float, b2: float) -> np.ndarray:
        y = float(b1) * age_range + float(b2) * (age_range ** 2)
        y0 = float(b1) * center_age + float(b2) * (center_age ** 2)
        return y - y0

    rng = np.random.default_rng(2026)

    def _draws_from_row(row: Optional[pd.Series], n: int) -> np.ndarray:
        if row is None:
            return np.array([], dtype=float)

        mu = float(row.get('estimate', np.nan))
        if not np.isfinite(mu):
            return np.array([], dtype=float)

        se = float(row.get('std_err', np.nan))
        ci_low = float(row.get('ci_low', np.nan))
        ci_high = float(row.get('ci_high', np.nan))

        sd = se if np.isfinite(se) and se > 0 else np.nan
        if not np.isfinite(sd) and np.isfinite(ci_low) and np.isfinite(ci_high) and (ci_high > ci_low):
            sd = float(ci_high - ci_low) / 3.92

        if not np.isfinite(sd) or sd <= 0:
            return np.full(n, mu, dtype=float)
        return rng.normal(mu, sd, size=int(n)).astype(float)

    def _band(b1_draws: np.ndarray, b2_draws: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        if len(b1_draws) == 0 or len(b2_draws) == 0:
            return None
        n = int(min(len(b1_draws), len(b2_draws)))
        if n < 5:
            return None

        b1 = b1_draws[:n].astype(float)
        b2 = b2_draws[:n].astype(float)
        curves = np.vstack([_curve(float(a), float(b)) for a, b in zip(b1, b2)])
        lo = np.nanpercentile(curves, 10, axis=0)
        hi = np.nanpercentile(curves, 90, axis=0)
        return lo.astype(float), hi.astype(float)

    j_age = _get_row(judge_data, 'age')
    j_age_sq = _get_row(judge_data, 'age_sq')
    f_age = _get_row(fan_data, 'age')
    f_age_sq = _get_row(fan_data, 'age_sq')

    if j_age is None or j_age_sq is None or f_age is None or f_age_sq is None:
        for ax in (ax_j, ax_f):
            ax.axis('off')
        ax_j.text(0.5, 0.5, 'Age terms not found', ha='center', va='center', transform=ax_j.transAxes)
    else:
        judge_b1 = float(j_age.get('estimate', np.nan))
        judge_b2 = float(j_age_sq.get('estimate', np.nan))
        fan_b1 = float(f_age.get('estimate', np.nan))
        fan_b2 = float(f_age_sq.get('estimate', np.nan))

        judge_line = _curve(judge_b1, judge_b2)
        fan_line = _curve(fan_b1, fan_b2)

        j_band = _band(_draws_from_row(j_age, 500), _draws_from_row(j_age_sq, 500))

        fan_b1_draws = np.array([], dtype=float)
        fan_b2_draws = np.array([], dtype=float)
        if fan_refit_draws is not None and isinstance(fan_refit_draws, pd.DataFrame) and not fan_refit_draws.empty:
            try:
                d = fan_refit_draws.copy()
                d = d[(d.get('outcome', '').astype(str) == 'fan_vote_index_mean')].copy()
                d['term'] = d['term'].astype(str)
                d['note'] = d.get('note', '').astype(str)
                d['estimate'] = pd.to_numeric(d.get('estimate', np.nan), errors='coerce')
                d = d[d['term'].isin(['age', 'age_sq'])].dropna(subset=['estimate']).copy()
                if not d.empty:
                    piv = d.pivot_table(index='note', columns='term', values='estimate', aggfunc='mean')
                    piv = piv.dropna(subset=['age', 'age_sq']).copy()
                    if not piv.empty:
                        fan_b1_draws = piv['age'].to_numpy(dtype=float)
                        fan_b2_draws = piv['age_sq'].to_numpy(dtype=float)
            except Exception:
                fan_b1_draws = np.array([], dtype=float)
                fan_b2_draws = np.array([], dtype=float)

        if len(fan_b1_draws) == 0 or len(fan_b2_draws) == 0:
            fan_b1_draws = _draws_from_row(f_age, 500)
            fan_b2_draws = _draws_from_row(f_age_sq, 500)

        f_band = _band(fan_b1_draws, fan_b2_draws)

        for ax in (ax_j, ax_f):
            ax.axvspan(22, 45, color=config.get_color('muted'), alpha=0.06, zorder=0)
            ax.axvline(center_age, color=config.get_color('muted'), linewidth=1.0, alpha=0.55, linestyle='--', zorder=0)
            ax.grid(True, alpha=0.22)

        if j_band is not None:
            ax_j.fill_between(age_range, j_band[0], j_band[1], color=config.get_color('primary'), alpha=0.12, linewidth=0.0, zorder=1)
        if f_band is not None:
            ax_f.fill_between(age_range, f_band[0], f_band[1], color=config.get_color('danger'), alpha=0.12, linewidth=0.0, zorder=1)

        ax_j.plot(age_range, judge_line, '-', linewidth=4.4, label=None, alpha=0.20, color=config.get_color('primary'), zorder=2)
        ax_j.plot(age_range, judge_line, '-', linewidth=2.6, label='Judge line', alpha=0.92, color=config.get_color('primary'), zorder=3)
        ax_f.plot(age_range, fan_line, '-', linewidth=4.4, label=None, alpha=0.20, color=config.get_color('danger'), zorder=2)
        ax_f.plot(age_range, fan_line, '-', linewidth=2.6, label='Fan line', alpha=0.92, color=config.get_color('danger'), zorder=3)

        j_opt = _vertex(judge_b1, judge_b2)
        f_opt = _vertex(fan_b1, fan_b2)

        if j_opt is not None:
            ax_j.axvline(x=j_opt, color=config.get_color('primary'), linestyle='--', alpha=0.75, linewidth=1.6, zorder=4)
            ytxt = float(np.nanmax(judge_line)) if np.isfinite(float(np.nanmax(judge_line))) else 0.0
            t = ax_j.text(j_opt, ytxt, f"  turning point ≈ {j_opt:.1f}", ha='left', va='bottom', fontsize=9.2, color=config.get_color('primary'))
            t.set_path_effects([pe.withStroke(linewidth=2.4, foreground=stroke_fc)])

        if f_opt is not None:
            ax_f.axvline(x=f_opt, color=config.get_color('danger'), linestyle='--', alpha=0.75, linewidth=1.6, zorder=4)
            ytxt = float(np.nanmax(fan_line)) if np.isfinite(float(np.nanmax(fan_line))) else 0.0
            t = ax_f.text(f_opt, ytxt, f"  turning point ≈ {f_opt:.1f}", ha='left', va='bottom', fontsize=9.2, color=config.get_color('danger'))
            t.set_path_effects([pe.withStroke(linewidth=2.4, foreground=stroke_fc)])

        ax_j.set_ylabel('Effect (centered at age 35)')
        ax_f.set_ylabel('Effect (centered at age 35)')
        ax_f.set_xlabel('Age')

        ax_j.set_title('Nonlinear age effect (judge line)', fontweight='bold')
        ax_f.set_title('Nonlinear age effect (fan line)', fontweight='bold')

        ax_j.legend(loc='upper right', frameon=False)
        ax_f.legend(loc='upper right', frameon=False)

        ax_j.text(
            0.02,
            0.04,
            'Shaded band = uncertainty (CI / refit draws)',
            transform=ax_j.transAxes,
            ha='left',
            va='bottom',
            fontsize=9.0,
            bbox=config.callout_bbox(kind='note'),
        )

    plt.tight_layout()

    save_figure_with_config(fig, 'q3_age_effect_curves', output_dirs, config)


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

        fan_refit_draws = None
        try:
            refit_path = data_dir / 'outputs' / 'tables' / 'mcm2026c_q3_fan_refit_coeff_draws.csv'
            if refit_path.exists():
                fan_refit_draws = pd.read_csv(refit_path)
        except Exception:
            fan_refit_draws = None

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
        
        create_q3_age_effect_curves(coeffs_data, output_dirs, config, fan_refit_draws=fan_refit_draws)
        print("✅ Created age effect curves")

        stability_path = data_dir / 'outputs' / 'tables' / 'mcm2026c_q3_fan_refit_stability.csv'
        if stability_path.exists():
            stab = pd.read_csv(stability_path)
            fn = globals().get('create_q3_refit_stability_bubble', None)
            if callable(fn):
                fn(stab, baseline_df, output_dirs, config)
                print("✅ Created refit stability bubble plot")
            else:
                print("⚠️ Skipping refit stability bubble (function not available)")

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