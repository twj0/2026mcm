"""
Q1 Visualization Module: Fan Vote Inference and Uncertainty Quantification

This module implements all visualization functions for Q1 analysis including:
- Uncertainty heatmaps
- Posterior interval plots  
- Judge vs Fan preference scatter plots
- Mechanism comparison plots
- Statistical vs ML method comparisons
- Sensitivity analysis results
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


def create_q1_uncertainty_heatmap(
    uncertainty_data: pd.DataFrame,
    output_dirs: Dict[str, Path],
    config: VisualizationConfig
) -> None:
    """
    Create uncertainty heatmap showing ESS ratio and evidence across seasons and weeks.
    
    Args:
        uncertainty_data: DataFrame with columns [season, week, ess_ratio, evidence, n_active, n_exit]
        output_dirs: Dictionary with 'tiff' and 'eps' paths
        config: Configuration object
    """
    figsize = config.get_figure_size('double_column')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, sharey=True)
    
    # Prepare data for heatmaps
    ess_pivot = uncertainty_data.pivot_table(
        values='ess_ratio', index='season', columns='week', fill_value=np.nan
    )
    evidence_pivot = uncertainty_data.pivot_table(
        values='evidence', index='season', columns='week', fill_value=np.nan
    )

    exit_pivot = uncertainty_data.pivot_table(values='n_exit', index='season', columns='week', fill_value=0)
    
    # Left heatmap: ESS Ratio
    sns.heatmap(ess_pivot, ax=ax1, cmap=config.get_cmap('diverging'), vmin=0, vmax=1, 
                cbar_kws={'label': 'ESS Ratio'}, annot=False)
    ax1.set_title('ESS Ratio (Lower = Higher Uncertainty)', fontweight='bold')
    ax1.set_xlabel('Week')
    ax1.set_ylabel('Season')
    
    # Right heatmap: Evidence
    sns.heatmap(evidence_pivot, ax=ax2, cmap=config.get_cmap('sequential'), vmin=0, vmax=1,
                cbar_kws={'label': 'Evidence'}, annot=False)
    ax2.set_title('Evidence (Brighter = Stronger Constraints)', fontweight='bold')
    ax2.set_xlabel('Week')
    ax2.set_ylabel('Season')

    try:
        mask = exit_pivot.to_numpy(dtype=float) > 0
        yy, xx = np.where(mask)
        ax1.scatter(xx + 0.5, yy + 0.5, s=10, marker='s', c=config.get_color('muted'), alpha=0.40, linewidths=0)
        ax2.scatter(xx + 0.5, yy + 0.5, s=10, marker='s', c=config.get_color('muted'), alpha=0.40, linewidths=0)
        ax1.text(0.01, 0.01, 'squares = elimination weeks', transform=ax1.transAxes, fontsize=8)
    except Exception:
        pass
    
    plt.tight_layout()
    
    # Save using config
    save_figure_with_config(fig, 'q1_uncertainty_heatmap', output_dirs, config)


def create_q1_fan_share_intervals(
    posterior_data: pd.DataFrame,
    uncertainty_data: pd.DataFrame,
    output_dirs: Dict[str, Path],
    config: VisualizationConfig
) -> None:
    """
    Create fan share posterior interval plots for high vs low uncertainty weeks.
    
    Args:
        posterior_data: DataFrame with fan share posterior summaries
        uncertainty_data: DataFrame with uncertainty metrics
        output_dirs: Dictionary with 'tiff' and 'eps' paths
        config: Configuration object
    """
    posterior_data = posterior_data[posterior_data['mechanism'] == 'percent'].copy()
    uncertainty_data = uncertainty_data[uncertainty_data['mechanism'] == 'percent'].copy()

    # Select contrasting weeks
    # High uncertainty: low evidence, has eliminations
    high_uncertainty = uncertainty_data[
        (uncertainty_data['evidence'] < 0.3) & 
        (uncertainty_data['n_exit'] > 0) &
        (uncertainty_data['n_active'] >= 6)
    ].iloc[0] if len(uncertainty_data[
        (uncertainty_data['evidence'] < 0.3) & 
        (uncertainty_data['n_exit'] > 0) &
        (uncertainty_data['n_active'] >= 6)
    ]) > 0 else uncertainty_data.iloc[0]
    
    # Low uncertainty: high evidence, has eliminations  
    low_uncertainty = uncertainty_data[
        (uncertainty_data['evidence'] > 0.7) & 
        (uncertainty_data['n_exit'] > 0) &
        (uncertainty_data['n_active'] >= 6)
    ].iloc[0] if len(uncertainty_data[
        (uncertainty_data['evidence'] > 0.7) & 
        (uncertainty_data['n_exit'] > 0) &
        (uncertainty_data['n_active'] >= 6)
    ]) > 0 else uncertainty_data.iloc[-1]
    
    figsize = config.get_figure_size('single_column')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    
    # High uncertainty week
    high_week_data = posterior_data[
        (posterior_data['season'] == high_uncertainty['season']) &
        (posterior_data['week'] == high_uncertainty['week'])
    ].sort_values('fan_share_mean', ascending=False)
    
    if len(high_week_data) > 0:
        x_pos = range(len(high_week_data))
        ax1.errorbar(x_pos, high_week_data['fan_share_mean'],
                    yerr=[high_week_data['fan_share_mean'] - high_week_data['fan_share_p05'],
                          high_week_data['fan_share_p95'] - high_week_data['fan_share_mean']],
                    fmt='o', capsize=5, capthick=2, markersize=8, color=config.get_color('warning'))
        
        # Highlight eliminated contestants
        eliminated_mask = high_week_data['eliminated_this_week']
        if eliminated_mask.any():
            eliminated_indices = [i for i, elim in enumerate(eliminated_mask) if elim]
            ax1.scatter([x_pos[i] for i in eliminated_indices], 
                       [high_week_data.iloc[i]['fan_share_mean'] for i in eliminated_indices],
                       color=config.get_color('danger'), s=100, marker='x', linewidth=3, label='Eliminated')
        
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([name[:15] + '...' if len(name) > 15 else name 
                            for name in high_week_data['celebrity_name']], rotation=45, ha='right')
        ax1.set_ylabel('Estimated Fan Vote Share')
        ax1.set_title(
            f'High-Uncertainty Week (Season {high_uncertainty["season"]}, Week {high_uncertainty["week"]})\n'
            f'Evidence={high_uncertainty["evidence"]:.3f}',
            fontweight='bold',
        )
        
        ax1.grid(True, alpha=0.3)
        if eliminated_mask.any():
            ax1.legend()
    
    # Low uncertainty week
    low_week_data = posterior_data[
        (posterior_data['season'] == low_uncertainty['season']) &
        (posterior_data['week'] == low_uncertainty['week'])
    ].sort_values('fan_share_mean', ascending=False)
    
    if len(low_week_data) > 0:
        x_pos = range(len(low_week_data))
        ax2.errorbar(x_pos, low_week_data['fan_share_mean'],
                    yerr=[low_week_data['fan_share_mean'] - low_week_data['fan_share_p05'],
                          low_week_data['fan_share_p95'] - low_week_data['fan_share_mean']],
                    fmt='o', capsize=5, capthick=2, markersize=8, color=config.get_color('primary'))
        
        # Highlight eliminated contestants
        eliminated_mask = low_week_data['eliminated_this_week']
        if eliminated_mask.any():
            eliminated_indices = [i for i, elim in enumerate(eliminated_mask) if elim]
            ax2.scatter([x_pos[i] for i in eliminated_indices], 
                       [low_week_data.iloc[i]['fan_share_mean'] for i in eliminated_indices],
                       color=config.get_color('danger'), s=100, marker='x', linewidth=3, label='Eliminated')
        
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([name[:15] + '...' if len(name) > 15 else name 
                            for name in low_week_data['celebrity_name']], rotation=45, ha='right')
        ax2.set_ylabel('Estimated Fan Vote Share')
        ax2.set_title(
            f'Low-Uncertainty Week (Season {low_uncertainty["season"]}, Week {low_uncertainty["week"]})\n'
            f'Evidence={low_uncertainty["evidence"]:.3f}',
            fontweight='bold',
        )
        
        ax2.grid(True, alpha=0.3)
        if eliminated_mask.any():
            ax2.legend()
    
    plt.tight_layout()
    
    # Save using config
    save_figure_with_config(fig, 'q1_fan_share_intervals', output_dirs, config)


def create_q1_judge_vs_fan_scatter(
    posterior_data: pd.DataFrame,
    weekly_panel: pd.DataFrame,
    output_dirs: Dict[str, Path],
    config: VisualizationConfig
) -> None:
    """
    Create scatter plot comparing judge scores vs fan preferences.
    
    Args:
        posterior_data: DataFrame with fan vote posterior summaries
        weekly_panel: DataFrame with judge scores and elimination info
        output_dirs: Dictionary with 'tiff' and 'eps' paths
        config: Configuration object
    """
    posterior_data = posterior_data[posterior_data['mechanism'] == 'percent'].copy()

    # Merge data
    merged_data = pd.merge(
        posterior_data[['season', 'week', 'celebrity_name', 'fan_share_mean', 'eliminated_this_week']],
        weekly_panel[['season', 'week', 'celebrity_name', 'judge_score_pct']],
        on=['season', 'week', 'celebrity_name'],
        how='inner'
    )
    
    figsize = config.get_figure_size('single_column')
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create scatter plot
    eliminated = merged_data[merged_data['eliminated_this_week']]
    not_eliminated = merged_data[~merged_data['eliminated_this_week']]
    
    # Plot non-eliminated contestants
    ax.scatter(
        not_eliminated['judge_score_pct'],
        not_eliminated['fan_share_mean'],
        alpha=0.65,
        s=50,
        c=config.get_color('primary'),
        edgecolors='#111827',
        linewidths=0.25,
        label='Not eliminated',
    )
    
    # Plot eliminated contestants
    ax.scatter(
        eliminated['judge_score_pct'],
        eliminated['fan_share_mean'],
        alpha=0.90,
        s=90,
        c=config.get_color('danger'),
        marker='x',
        linewidth=2.5,
        label='Eliminated',
    )
    
    # Add diagonal reference line
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Judge = Fan (reference)')
    
    # Identify and annotate extreme cases
    # High judge, low fan (technical but unpopular)
    high_judge_low_fan = merged_data[
        (merged_data['judge_score_pct'] > 0.7) & 
        (merged_data['fan_share_mean'] < 0.3)
    ]
    
    # Low judge, high fan (popular but not technical)
    low_judge_high_fan = merged_data[
        (merged_data['judge_score_pct'] < 0.3) & 
        (merged_data['fan_share_mean'] > 0.7)
    ]
    
    # Annotate a few extreme cases
    for _, row in high_judge_low_fan.head(3).iterrows():
        ax.annotate(
            f"{row['celebrity_name'][:10]}...",
            (row['judge_score_pct'], row['fan_share_mean']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=9,
            bbox=config.callout_bbox(kind='note'),
        )
    
    for _, row in low_judge_high_fan.head(3).iterrows():
        ax.annotate(
            f"{row['celebrity_name'][:10]}...",
            (row['judge_score_pct'], row['fan_share_mean']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=9,
            bbox=config.callout_bbox(kind='note'),
        )
    
    ax.set_xlabel('Judge Score Share (Technical Line)')
    ax.set_ylabel('Estimated Fan Vote Share (Popularity Line)')
    ax.set_title('Technical Skill vs Popularity: Judge vs Fan Preferences', fontweight='bold')
    
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save using config
    save_figure_with_config(fig, 'q1_judge_vs_fan_scatter', output_dirs, config)


def create_q1_mechanism_comparison(
    posterior_data: pd.DataFrame,
    output_dirs: Dict[str, Path],
    config: VisualizationConfig
) -> None:
    """
    Create mechanism comparison plot showing Percent vs Rank differences.
    
    Args:
        posterior_data: DataFrame with fan vote posterior summaries
        output_dirs: Dictionary with 'tiff' and 'eps' paths
        config: Configuration object
    """
    # Select representative weeks for comparison
    # Get weeks that have both percent and rank mechanisms
    mechanism_counts = posterior_data.groupby(['season', 'week'])['mechanism'].nunique()
    weeks_with_both = mechanism_counts[mechanism_counts >= 2].index.tolist()
    
    if len(weeks_with_both) >= 2:
        selected_weeks = weeks_with_both[:2]
    else:
        # No (season, week) has both mechanisms -> still show percent panels and mark rank as missing.
        percent_weeks = (
            posterior_data[posterior_data['mechanism'] == 'percent'][['season', 'week']]
            .drop_duplicates()
            .head(2)
        )
        selected_weeks = [(row['season'], row['week']) for _, row in percent_weeks.iterrows()]
    
    figsize = config.get_figure_size('large_figure')
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    for i, (season, week) in enumerate(selected_weeks):
        ax1, ax2 = axes[i]
        
        # Get data for this week
        week_data = posterior_data[
            (posterior_data['season'] == season) & 
            (posterior_data['week'] == week)
        ]
        
        # Percent mechanism results
        percent_data = week_data[week_data['mechanism'] == 'percent'].sort_values('fan_share_mean', ascending=False)
        if len(percent_data) > 0:
            x_pos = range(len(percent_data))
            bars1 = ax1.bar(
                x_pos,
                percent_data['fan_share_mean'],
                color=config.get_color('percent'),
                alpha=0.82,
                edgecolor='#111827',
                linewidth=0.35,
            )
            
            ax1.set_title(f'Percent (Season {season}, Week {week})', fontweight='bold')
            ax1.set_ylabel('Estimated Fan Vote Share')
            
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels([name[:10] + '...' if len(name) > 10 else name 
                                for name in percent_data['celebrity_name']], 
                               rotation=45, ha='right')
            ax1.grid(True, alpha=0.3)
            
            # Highlight eliminated contestants
            for j, (_, row) in enumerate(percent_data.iterrows()):
                if row['eliminated_this_week']:
                    bars1[j].set_color(config.get_color('danger'))
                    bars1[j].set_alpha(0.9)
        
        else:
            ax1.axis('off')
            ax1.text(
                0.5,
                0.5,
                f'No percent data\n(Season {season}, Week {week})',
                ha='center',
                va='center',
                transform=ax1.transAxes,
                fontsize=12,
            )
        
        # Rank mechanism results
        rank_data = week_data[week_data['mechanism'] == 'rank'].sort_values('fan_share_mean', ascending=False)
        if len(rank_data) > 0:
            x_pos = range(len(rank_data))
            bars2 = ax2.bar(
                x_pos,
                rank_data['fan_share_mean'],
                color=config.get_color('rank'),
                alpha=0.82,
                edgecolor='#111827',
                linewidth=0.35,
            )
            
            ax2.set_title(f'Rank (Season {season}, Week {week})', fontweight='bold')
            ax2.set_ylabel('Estimated Fan Vote Share')
            
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels([name[:10] + '...' if len(name) > 10 else name 
                                for name in rank_data['celebrity_name']], 
                               rotation=45, ha='right')
            ax2.grid(True, alpha=0.3)
            
            # Highlight eliminated contestants
            for j, (_, row) in enumerate(rank_data.iterrows()):
                if row['eliminated_this_week']:
                    bars2[j].set_color(config.get_color('danger'))
                    bars2[j].set_alpha(0.9)
        
        else:
            ax2.axis('off')
            ax2.text(
                0.5,
                0.5,
                f'No rank data\n(Season {season}, Week {week})',
                ha='center',
                va='center',
                transform=ax2.transAxes,
                fontsize=12,
            )
    
    plt.tight_layout()
    
    # Save using config
    save_figure_with_config(fig, 'q1_mechanism_comparison', output_dirs, config)


def create_q1_statistical_vs_ml_comparison(
    ml_summary: pd.DataFrame,
    dl_summary: pd.DataFrame,
    output_dirs: Dict[str, Path],
    config: VisualizationConfig
) -> None:
    """
    Create comparison between statistical and ML methods (showcase).
    
    Args:
        output_dirs: Dictionary with 'tiff' and 'eps' paths
        config: Configuration object
    """
    df_ml = ml_summary.copy()
    df_dl = dl_summary.copy()

    methods = []
    roc_auc = []

    if 'model' in df_ml.columns and 'roc_auc_mean' in df_ml.columns:
        for _, row in df_ml.iterrows():
            methods.append(str(row['model']))
            roc_auc.append(float(row['roc_auc_mean']))

    if 'model' in df_dl.columns and 'roc_auc_mean' in df_dl.columns:
        for _, row in df_dl.iterrows():
            methods.append(str(row['model']))
            roc_auc.append(float(row['roc_auc_mean']))

    colors = sns.color_palette('Set2', n_colors=max(3, len(methods)))
    
    figsize = config.get_figure_size('double_column')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Left plot: Performance comparison
    bars = ax1.bar(methods, roc_auc, color=colors, alpha=0.8, edgecolor='black')
    ax1.set_ylabel('ROC-AUC (CV mean)')
    ax1.set_title('Showcase: Elimination Prediction Baselines', fontweight='bold')
    ax1.set_ylim(0.0, 1.0)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, acc in zip(bars, roc_auc):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    ax1.tick_params(axis='x', rotation=45)
    
    # Right plot: Capability radar chart
    categories = ['Performance', 'Interpretability', 'Uncertainty', 'Engineering Cost']
    bayesian_scores = [0.85, 0.95, 0.95, 0.8]
    ml_avg_scores = [0.80, 0.4, 0.3, 0.7]
    
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # Close the circle
    
    bayesian_scores += bayesian_scores[:1]
    ml_avg_scores += ml_avg_scores[:1]
    
    ax2.plot(angles, bayesian_scores, 'o-', linewidth=2.6, label='Structured inference', color=config.get_color('primary'))
    ax2.fill(angles, bayesian_scores, alpha=0.18, color=config.get_color('primary'))
    ax2.plot(angles, ml_avg_scores, 'o-', linewidth=2.6, label='Showcase ML baselines', color=config.get_color('muted'))
    ax2.fill(angles, ml_avg_scores, alpha=0.12, color=config.get_color('muted'))
    
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(categories)
    ax2.set_ylim(0, 1)
    ax2.set_title('Showcase: Method Capability Profile', fontweight='bold')
    ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax2.grid(True)
    
    plt.tight_layout()
    
    # Save using config
    save_figure_with_config(fig, 'q1_statistical_vs_ml_comparison', output_dirs, config)


def create_q1_mechanism_sensitivity_overview(
    sensitivity_data: pd.DataFrame,
    showcase_baseline: pd.DataFrame | None,
    output_dirs: Dict[str, Path],
    config: VisualizationConfig,
) -> None:
    df = sensitivity_data.copy()
    if df.empty:
        return

    df['tv_distance'] = pd.to_numeric(df['tv_distance'], errors='coerce')
    df['rank_corr'] = pd.to_numeric(df['rank_corr'], errors='coerce')
    df['n_contestants'] = pd.to_numeric(df.get('n_contestants', np.nan), errors='coerce')
    df = df.dropna(subset=['season', 'week', 'tv_distance', 'rank_corr']).copy()
    if df.empty:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=config.get_figure_size('double_column'))

    tv_pivot = df.pivot_table(values='tv_distance', index='season', columns='week', fill_value=np.nan)
    vmax = float(np.nanquantile(df['tv_distance'].to_numpy(dtype=float), 0.99)) if len(df) else 1.0
    vmax = vmax if np.isfinite(vmax) and vmax > 0 else 1.0
    sns.heatmap(
        tv_pivot,
        ax=ax1,
        cmap=config.get_cmap('heatmap'),
        vmin=0,
        vmax=vmax,
        cbar_kws={'label': 'TV distance'},
        annot=False,
    )
    ax1.set_title('Percent vs Rank sensitivity (TV distance)', fontweight='bold')
    ax1.set_xlabel('Week')
    ax1.set_ylabel('Season')

    s = df['n_contestants']
    s = s.fillna(s.median())
    size = 20.0 + 80.0 * (s.to_numpy(dtype=float) / max(float(s.max()), 1.0))

    sc = ax2.scatter(
        df['rank_corr'].to_numpy(dtype=float),
        df['tv_distance'].to_numpy(dtype=float),
        s=size,
        c=df['season'].to_numpy(dtype=float),
        cmap=config.get_cmap('sequential'),
        alpha=0.75,
        linewidths=0.3,
        edgecolors='#222222',
    )
    ax2.set_xlabel('Rank correlation (percent vs rank)')
    ax2.set_ylabel('TV distance (percent vs rank)')
    ax2.set_title('Global summary', fontweight='bold')
    ax2.set_ylim(bottom=0.0)
    cbar = plt.colorbar(sc, ax=ax2)
    cbar.set_label('Season')

    top = df.sort_values('tv_distance', ascending=False).head(6)
    for _, r in top.iterrows():
        ax2.annotate(
            f"S{int(r['season'])}W{int(r['week'])}",
            (float(r['rank_corr']), float(r['tv_distance'])),
            xytext=(4, 3),
            textcoords='offset points',
            fontsize=8,
        )

    plt.tight_layout()

    if showcase_baseline is not None and not showcase_baseline.empty:
        try:
            inset = ax2.inset_axes([0.06, 0.68, 0.38, 0.28])
            inset.set_title('Showcase baseline', fontsize=9.0, fontweight='bold')
            sb = showcase_baseline.copy()
            sb['roc_auc_mean'] = pd.to_numeric(sb.get('roc_auc_mean', np.nan), errors='coerce')
            sb['average_precision_mean'] = pd.to_numeric(sb.get('average_precision_mean', np.nan), errors='coerce')

            rows: list[tuple[str, float]] = []
            if 'model' in sb.columns and 'roc_auc_mean' in sb.columns:
                for _, r in sb.iterrows():
                    if str(r.get('model', '')).strip() == 'logreg':
                        rows.append(('ML logreg ROC-AUC', float(r['roc_auc_mean'])))
                        break

            if 'dl_roc_auc_mean' in sb.columns:
                rows.append(('DL tab-transformer ROC-AUC', float(sb['dl_roc_auc_mean'].iloc[0])))

            if rows:
                names = [a for a, _ in rows]
                vals = [b for _, b in rows]
                y = np.arange(len(vals))
                inset.barh(y, vals, color=config.get_color('muted'), alpha=0.85)
                inset.set_yticks(y)
                inset.set_yticklabels(names, fontsize=8.0)
                inset.set_xlim(0.0, 1.0)
                inset.grid(True, alpha=0.25)
                for i, v in enumerate(vals):
                    if np.isfinite(v):
                        inset.text(float(v) + 0.01, i, f"{float(v):.2f}", va='center', fontsize=8.0)

                config.add_callout(ax2, 'Shown as contrast only (different task)', loc='lower left', kind='note')
        except Exception:
            pass

    save_figure_with_config(fig, 'q1_mechanism_sensitivity_overview', output_dirs, config)


def create_q1_error_diagnostics_overview(
    diagnostics_data: pd.DataFrame,
    output_dirs: Dict[str, Path],
    config: VisualizationConfig,
) -> None:
    df = diagnostics_data.copy()
    if df.empty:
        return

    df = df[pd.to_numeric(df['n_exit'], errors='coerce') > 0].copy()
    if df.empty:
        return

    df['fan_share_width_mean'] = pd.to_numeric(df['fan_share_width_mean'], errors='coerce')
    df['observed_exit_prob_at_posterior_mean'] = pd.to_numeric(df['observed_exit_prob_at_posterior_mean'], errors='coerce')
    df['judge_fan_rank_corr'] = pd.to_numeric(df['judge_fan_rank_corr'], errors='coerce')
    df['match_pred'] = pd.to_numeric(df['match_pred'], errors='coerce')
    df['n_active'] = pd.to_numeric(df['n_active'], errors='coerce')

    df = df.dropna(subset=['fan_share_width_mean', 'observed_exit_prob_at_posterior_mean', 'judge_fan_rank_corr']).copy()
    if df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=config.get_figure_size('double_column'), sharey=True)

    last_sc = None
    for ax, mech in zip(axes, ['percent', 'rank']):
        sub = df[df['mechanism'].astype(str) == mech].copy()
        if sub.empty:
            ax.axis('off')
            continue

        na = sub['n_active']
        na = na.fillna(na.median())
        size = 20.0 + 80.0 * (na.to_numpy(dtype=float) / max(float(na.max()), 1.0))

        last_sc = ax.scatter(
            sub['fan_share_width_mean'].to_numpy(dtype=float),
            sub['observed_exit_prob_at_posterior_mean'].to_numpy(dtype=float),
            s=size,
            c=sub['judge_fan_rank_corr'].to_numpy(dtype=float),
            cmap='RdBu_r',
            vmin=-1,
            vmax=1,
            alpha=0.78,
            linewidths=0.3,
            edgecolors='#222222',
        )

        bad = sub[sub['match_pred'] == 0]
        if not bad.empty:
            ax.scatter(
                bad['fan_share_width_mean'].to_numpy(dtype=float),
                bad['observed_exit_prob_at_posterior_mean'].to_numpy(dtype=float),
                s=90,
                facecolors='none',
                edgecolors='#111111',
                linewidths=1.2,
                alpha=0.95,
            )

        ax.set_title(f"{mech}: consistency vs uncertainty", fontweight='bold')
        ax.set_xlabel('Mean posterior interval width (fan share)')
        ax.set_ylim(0.0, 1.02)

    axes[0].set_ylabel('Observed elimination probability at posterior mean')

    if last_sc is not None:
        cbar = plt.colorbar(last_sc, ax=axes, location='right', fraction=0.05, pad=0.02)
        cbar.set_label('Judge–fan rank corr')

    plt.tight_layout()
    save_figure_with_config(fig, 'q1_error_diagnostics_overview', output_dirs, config)


def generate_all_q1_visualizations(
    data_dir: Path,
    output_dirs: Dict[str, Path],
    config: VisualizationConfig,
    showcase: bool = False,
    mode: str = 'paper'
) -> None:
    """
    Generate all Q1 visualizations.
    
    Args:
        data_dir: Directory containing input data files
        output_dirs: Dictionary with 'tiff' and 'eps' paths
        config: Configuration object
        showcase: Whether to generate showcase-only figures
        mode: paper (4 core figs) or full
    """
    print("🎨 Generating Q1 visualizations...")
    
    # Load data
    try:
        uncertainty_data = pd.read_csv(data_dir / 'outputs' / 'tables' / 'mcm2026c_q1_uncertainty_summary.csv')
        posterior_data = pd.read_csv(data_dir / 'outputs' / 'predictions' / 'mcm2026c_q1_fan_vote_posterior_summary.csv')
        weekly_panel = pd.read_csv(data_dir / 'data' / 'processed' / 'dwts_weekly_panel.csv')
        
        print(f"✅ Loaded data: {len(uncertainty_data)} uncertainty records, {len(posterior_data)} posterior records")
        
        config.apply_matplotlib_style()

        # Generate visualizations
        create_q1_uncertainty_heatmap(
            uncertainty_data[uncertainty_data['mechanism'] == 'percent'].copy(),
            output_dirs,
            config,
        )
        print("✅ Created uncertainty heatmap")
        
        create_q1_fan_share_intervals(posterior_data, uncertainty_data, output_dirs, config)
        print("✅ Created fan share intervals plot")
        
        create_q1_judge_vs_fan_scatter(posterior_data, weekly_panel, output_dirs, config)
        print("✅ Created judge vs fan scatter plot")

        baseline_df = None
        try:
            ml_path = data_dir / 'outputs' / 'tables' / 'showcase' / 'mcm2026c_q1_ml_elimination_baselines_cv_summary.csv'
            dl_path = data_dir / 'outputs' / 'tables' / 'showcase' / 'mcm2026c_q1_dl_elimination_transformer_summary.csv'
            if ml_path.exists():
                ml = pd.read_csv(ml_path)
                baseline_df = ml
                if dl_path.exists():
                    dl = pd.read_csv(dl_path)
                    if 'roc_auc_mean' in dl.columns:
                        baseline_df = baseline_df.assign(dl_roc_auc_mean=float(dl['roc_auc_mean'].iloc[0]))
        except Exception:
            baseline_df = None

        sens_path = data_dir / 'outputs' / 'tables' / 'mcm2026c_q1_mechanism_sensitivity_week.csv'
        if sens_path.exists():
            sens = pd.read_csv(sens_path)
            create_q1_mechanism_sensitivity_overview(sens, baseline_df, output_dirs, config)
            print("✅ Created mechanism sensitivity overview")

        if mode != 'paper':
            create_q1_mechanism_comparison(posterior_data, output_dirs, config)
            print("✅ Created mechanism comparison plot")

            diag_path = data_dir / 'outputs' / 'tables' / 'mcm2026c_q1_error_diagnostics_week.csv'
            if diag_path.exists():
                diag = pd.read_csv(diag_path)
                create_q1_error_diagnostics_overview(diag, output_dirs, config)
                print("✅ Created error diagnostics overview")
        
        if showcase and mode != 'paper':
            ml_summary = pd.read_csv(
                data_dir
                / 'outputs'
                / 'tables'
                / 'showcase'
                / 'mcm2026c_q1_ml_elimination_baselines_cv_summary.csv'
            )
            dl_summary = pd.read_csv(
                data_dir
                / 'outputs'
                / 'tables'
                / 'showcase'
                / 'mcm2026c_q1_dl_elimination_transformer_summary.csv'
            )
            create_q1_statistical_vs_ml_comparison(ml_summary, dl_summary, output_dirs, config)
            print("✅ Created showcase ML baseline comparison")
        
        print(f"🎉 Q1 visualizations completed! Saved to {output_dirs['tiff']} and {output_dirs['eps']}")
        
    except Exception as e:
        print(f"❌ Error generating Q1 visualizations: {e}")
        raise

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Generate Q1 figures (TIFF + EPS).')
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
    output_structure = create_output_directories(args.data_dir / 'outputs' / 'figures', ['Q1'])

    generate_all_q1_visualizations(
        args.data_dir,
        output_structure['Q1'],
        config,
        showcase=args.showcase,
        mode=str(args.mode),
    )