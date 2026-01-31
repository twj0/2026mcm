"""
Q4 Visualization Module: New Voting Mechanism Design and Evaluation

This module implements all visualization functions for Q4 analysis including:
- Mechanism trade-off scatter plots
- Robustness curves
- Champion uncertainty analysis
- Seasonal variation analysis
- Pareto frontier plots
- ML feature importance analysis
- Prediction validation
- Mechanism recommendation decision tree
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


# Set publication-quality style
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 11,
    'font.family': 'serif',
    'figure.dpi': 300
})

def create_q4_mechanism_tradeoff_scatter(
    metrics_data: pd.DataFrame,
    output_dirs: Dict[str, Path],
    config: VisualizationConfig
) -> None:

    """
    Create mechanism trade-off scatter plot across different outlier levels.
    
    Args:
        metrics_data: DataFrame with mechanism performance metrics
        output_dir: Directory to save figures
        figsize: Figure size tuple
    """
    outlier_levels = sorted(metrics_data['outlier_mult'].unique())
    fig, axes = plt.subplots(1, len(outlier_levels), figsize=(6 * len(outlier_levels), 6))
    if len(outlier_levels) == 1:
        axes = [axes]

    mechanisms = sorted(metrics_data['mechanism'].unique())
    palette = sns.color_palette('tab10', n_colors=max(len(mechanisms), 3))
    colors = {m: palette[i % len(palette)] for i, m in enumerate(mechanisms)}

    for i, outlier_mult in enumerate(outlier_levels):
        ax = axes[i]
        data_subset = metrics_data[metrics_data['outlier_mult'] == outlier_mult]

        # Group by mechanism and calculate averages
        mechanism_avg = data_subset.groupby('mechanism').agg({
            'tpi_season_avg': 'mean',
            'fan_vs_uniform_contrast': 'mean',
            'robust_fail_rate': 'mean'
        }).reset_index()

        has_fan_se = 'fan_vs_uniform_contrast_se' in data_subset.columns
        has_tpi_boot = 'tpi_boot_p025' in data_subset.columns and 'tpi_boot_p975' in data_subset.columns
        has_tpi_std = 'tpi_std' in data_subset.columns and 'tpi_n' in data_subset.columns

        fan_xerr: dict[str, float] = {}
        tpi_yerr: dict[str, float] = {}
        if has_fan_se:
            for mech in mechanisms:
                s = data_subset.loc[data_subset['mechanism'] == mech, 'fan_vs_uniform_contrast_se']
                s = s[np.isfinite(s)]
                fan_xerr[mech] = float(1.96 * np.sqrt(np.mean(np.square(s)))) if len(s) else 0.0

        if has_tpi_boot:
            for mech in mechanisms:
                lo = data_subset.loc[data_subset['mechanism'] == mech, 'tpi_boot_p025']
                hi = data_subset.loc[data_subset['mechanism'] == mech, 'tpi_boot_p975']
                lo = lo[np.isfinite(lo)]
                hi = hi[np.isfinite(hi)]
                if len(lo) and len(hi):
                    se = np.mean((hi.to_numpy() - lo.to_numpy()) / (2.0 * 1.96))
                    tpi_yerr[mech] = float(1.96 * se)
                else:
                    tpi_yerr[mech] = 0.0
        elif has_tpi_std:
            for mech in mechanisms:
                std = data_subset.loc[data_subset['mechanism'] == mech, 'tpi_std']
                n = data_subset.loc[data_subset['mechanism'] == mech, 'tpi_n']
                std = std[np.isfinite(std)]
                n = n[np.isfinite(n)]
                if len(std) and len(n):
                    se = np.mean(std.to_numpy() / np.sqrt(np.maximum(n.to_numpy(), 1.0)))
                    tpi_yerr[mech] = float(1.96 * se)
                else:
                    tpi_yerr[mech] = 0.0

        # Create scatter plot with size representing robustness
        for _, row in mechanism_avg.iterrows():
            mech = row['mechanism']
            size = (1 - row['robust_fail_rate']) * 300 + 50  # Higher robustness = larger point

            ax.scatter(
                row['fan_vs_uniform_contrast'],
                row['tpi_season_avg'],
                s=size,
                c=colors.get(mech, 'gray'),
                alpha=0.7,
                label=mech if i == 0 else "",
            )  # Only label in first subplot

            xerr = fan_xerr.get(mech, 0.0)
            yerr = tpi_yerr.get(mech, 0.0)
            if (xerr and np.isfinite(xerr)) or (yerr and np.isfinite(yerr)):
                ax.errorbar(
                    row['fan_vs_uniform_contrast'],
                    row['tpi_season_avg'],
                    xerr=xerr if xerr and np.isfinite(xerr) else None,
                    yerr=yerr if yerr and np.isfinite(yerr) else None,
                    fmt='none',
                    ecolor=colors.get(mech, 'gray'),
                    elinewidth=1,
                    alpha=0.35,
                    capsize=2,
                )

            # Add mechanism labels
            ax.annotate(mech.replace('_', '\n'), 
                       (row['fan_vs_uniform_contrast'], row['tpi_season_avg']),
                       xytext=(5, 5), textcoords='offset points', fontsize=9)

        ax.set_xlabel('Fan expression strength (fan vs uniform contrast)')
        ax.set_ylabel('Technical Protection Index (TPI)')
        ax.set_title(f'Stress test: outlier_mult={outlier_mult}', fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Add ideal region
        ideal_rect = plt.Rectangle((0.6, 0.7), 0.3, 0.2, 
                                  fill=False, edgecolor='gold', linewidth=2, linestyle='--')
        ax.add_patch(ideal_rect)
        ax.text(0.75, 0.8, 'Ideal region', ha='center', va='center', 
               bbox=dict(boxstyle='round', facecolor='gold', alpha=0.3))

        # Set consistent axis limits
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()

    save_figure_with_config(fig, 'q4_mechanism_tradeoff_scatter', output_dirs, config)

def create_q4_robustness_curves(
    metrics_data: pd.DataFrame,
    output_dirs: Dict[str, Path],
    config: VisualizationConfig
) -> None:

    """
    Create robustness curves showing performance under stress tests.
    
    Args:
        metrics_data: DataFrame with mechanism performance metrics
        output_dir: Directory to save figures
        figsize: Figure size tuple
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=config.get_figure_size('double_column'))

    mechanisms = sorted(metrics_data['mechanism'].unique())
    outlier_values = sorted(metrics_data['outlier_mult'].unique())
    palette = sns.color_palette('tab10', n_colors=max(len(mechanisms), 3))
    colors = {m: palette[i % len(palette)] for i, m in enumerate(mechanisms)}

    # Left plot: Robustness failure rate curves
    for mech in mechanisms:
        mech_data = metrics_data[metrics_data['mechanism'] == mech]
        fail_rates = []
        fail_rate_band = []

        has_mc_se = 'robust_fail_rate_se' in mech_data.columns

        for outlier in outlier_values:
            subset = mech_data[mech_data['outlier_mult'] == outlier]
            if len(subset) > 0:
                m = float(subset['robust_fail_rate'].mean())
                fail_rates.append(m)

                if has_mc_se and 'robust_fail_rate_se' in subset.columns:
                    se_mc = subset['robust_fail_rate_se']
                    se_mc = se_mc[np.isfinite(se_mc)]
                    se_mc_agg = float(np.sqrt(np.mean(np.square(se_mc)))) if len(se_mc) else 0.0

                    n = int(subset['robust_fail_rate'].notna().sum())
                    se_between = float(subset['robust_fail_rate'].std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0

                    band = float(1.96 * np.sqrt(se_between * se_between + se_mc_agg * se_mc_agg))
                    fail_rate_band.append(band)
                else:
                    fail_rate_band.append(float(subset['robust_fail_rate'].std()))
            else:
                fail_rates.append(0)
                fail_rate_band.append(0)

        ax1.plot(outlier_values, fail_rates, 'o-', label=mech, 
                linewidth=2, markersize=8, color=colors.get(mech, 'gray'))
        ax1.fill_between(outlier_values, 
                        np.array(fail_rates) - np.array(fail_rate_band),
                        np.array(fail_rates) + np.array(fail_rate_band),
                        alpha=0.2, color=colors.get(mech, 'gray'))

    ax1.set_xlabel('Stress test intensity (outlier_mult)')
    ax1.set_ylabel('Robustness fail rate')
    ax1.set_title('Robustness vs stress intensity', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)

    # Right plot: Robustness ranking changes
    robustness_ranks = {}
    for outlier in outlier_values:
        subset = metrics_data[metrics_data['outlier_mult'] == outlier]
        mech_avg = subset.groupby('mechanism')['robust_fail_rate'].mean().sort_values()
        for rank, mech in enumerate(mech_avg.index):
            if mech not in robustness_ranks:
                robustness_ranks[mech] = []
            robustness_ranks[mech].append(rank + 1)

    for mech, ranks in robustness_ranks.items():
        if len(ranks) == len(outlier_values):
            ax2.plot(outlier_values, ranks, 'o-', label=mech, 
                    linewidth=2, markersize=8, color=colors.get(mech, 'gray'))

    ax2.set_xlabel('Stress test intensity (outlier_mult)')
    ax2.set_ylabel('Robustness rank (1=best)')
    ax2.set_title('Robustness rank change', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.invert_yaxis()  # Lower rank is better

    plt.tight_layout()

    save_figure_with_config(fig, 'q4_robustness_curves', output_dirs, config)

def create_q4_champion_uncertainty_analysis(
    metrics_data: pd.DataFrame,
    output_dirs: Dict[str, Path],
    config: VisualizationConfig
) -> None:

    """
    Create champion uncertainty analysis plots.
    
    Args:
        metrics_data: DataFrame with mechanism performance metrics
        output_dir: Directory to save figures
        figsize: Figure size tuple
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=config.get_figure_size('large_figure'))

    mechanism_order = sorted(metrics_data['mechanism'].unique())
    palette = sns.color_palette('tab10', n_colors=max(len(mechanism_order), 3))
    colors = {m: palette[i % len(palette)] for i, m in enumerate(mechanism_order)}

    # Subplot 1: Champion entropy distribution
    entropy_data = [metrics_data[metrics_data['mechanism'] == mech]['champion_entropy'] 
                   for mech in mechanism_order if mech in metrics_data['mechanism'].values]
    mechanism_labels = [mech for mech in mechanism_order if mech in metrics_data['mechanism'].values]

    box_plot = ax1.boxplot(entropy_data, labels=mechanism_labels, patch_artist=True)
    for patch, mech in zip(box_plot['boxes'], mechanism_labels):
        patch.set_facecolor(colors.get(mech, 'gray'))
        patch.set_alpha(0.7)

    ax1.set_ylabel('Champion entropy')
    ax1.set_title('Outcome randomness (champion entropy)', fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)

    # Subplot 2: Champion mode probability distribution
    mode_prob_data = [metrics_data[metrics_data['mechanism'] == mech]['champion_mode_prob'] 
                     for mech in mechanism_labels]

    box_plot2 = ax2.boxplot(mode_prob_data, labels=mechanism_labels, patch_artist=True)
    for patch, mech in zip(box_plot2['boxes'], mechanism_labels):
        patch.set_facecolor(colors.get(mech, 'gray'))
        patch.set_alpha(0.7)

    ax2.set_ylabel('Champion mode probability')
    ax2.set_title('Outcome concentration (mode probability)', fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)

    # Subplot 3: Uncertainty vs Technical Protection scatter
    for mech in mechanism_labels:
        mech_data = metrics_data[metrics_data['mechanism'] == mech]
        ax3.scatter(mech_data['champion_entropy'], mech_data['tpi_season_avg'],
                   c=colors.get(mech, 'gray'), label=mech, alpha=0.6, s=50)

    ax3.set_xlabel('Champion entropy')
    ax3.set_ylabel('Technical Protection Index (TPI)')
    ax3.set_title('Randomness vs technical protection', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Subplot 4: Ideal region analysis
    ideal_entropy_range = (0.5, 0.8)  # Moderate randomness
    ideal_tpi_threshold = 0.7  # High technical protection

    ideal_rates = []
    for mech in mechanism_labels:
        mech_data = metrics_data[metrics_data['mechanism'] == mech]
        ideal_count = len(mech_data[
            (mech_data['champion_entropy'] >= ideal_entropy_range[0]) &
            (mech_data['champion_entropy'] <= ideal_entropy_range[1]) &
            (mech_data['tpi_season_avg'] >= ideal_tpi_threshold)
        ])
        ideal_rate = ideal_count / len(mech_data) if len(mech_data) > 0 else 0
        ideal_rates.append(ideal_rate)

    bars = ax4.bar(mechanism_labels, ideal_rates, 
                   color=[colors.get(mech, 'gray') for mech in mechanism_labels], alpha=0.7)
    ax4.set_ylabel('Share in ideal region')
    ax4.set_title('Share in ideal region (moderate randomness + high TPI)', fontweight='bold')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)

    # Add value labels
    for bar, rate in zip(bars, ideal_rates):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{rate:.2%}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()

    save_figure_with_config(fig, 'q4_champion_uncertainty_analysis', output_dirs, config)

def create_q4_seasonal_variation_analysis(
    metrics_data: pd.DataFrame,
    output_dirs: Dict[str, Path],
    config: VisualizationConfig
) -> None:

    """
    Create seasonal variation analysis plots.
    
    Args:
        metrics_data: DataFrame with mechanism performance metrics
        output_dir: Directory to save figures
        figsize: Figure size tuple
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=config.get_figure_size('large_figure'))

    seasons = sorted(metrics_data['season'].unique())
    mechanisms_subset = sorted(metrics_data['mechanism'].unique())

    # Upper plot: TPI coefficient of variation heatmap
    tpi_cv_matrix = []
    for season in seasons:
        season_data = metrics_data[metrics_data['season'] == season]
        season_cvs = []
        for mech in mechanisms_subset:
            mech_data = season_data[season_data['mechanism'] == mech]
            if len(mech_data) > 0:
                mean_val = mech_data['tpi_season_avg'].mean()
                std_val = mech_data['tpi_season_avg'].std()
                if mean_val == 0 or np.isnan(mean_val):
                    season_cvs.append(0)
                else:
                    cv = std_val / mean_val
                    season_cvs.append(float(cv) if not np.isnan(cv) else 0)
            else:
                season_cvs.append(0)

        tpi_cv_matrix.append(season_cvs)

    tpi_cv_matrix = np.array(tpi_cv_matrix)

    # Create heatmap
    im1 = ax1.imshow(tpi_cv_matrix.T, cmap='YlOrRd', aspect='auto')
    ax1.set_xticks(range(0, len(seasons), 5))
    ax1.set_xticklabels(seasons[::5])
    ax1.set_yticks(range(len(mechanisms_subset)))
    ax1.set_yticklabels(mechanisms_subset)
    ax1.set_xlabel('Season')
    ax1.set_title('TPI coefficient of variation (CV) by season', fontweight='bold')

    # Add colorbar
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('CV')

    # Lower plot: Mechanism consistency analysis
    consistency_scores = {}
    for mech in mechanisms_subset:
        mech_data = metrics_data[metrics_data['mechanism'] == mech]
        if len(mech_data) > 0:
            # Calculate cross-season consistency (inverse of standard deviation)
            tpi_std = mech_data.groupby('season')['tpi_season_avg'].mean().std()
            fan_std = mech_data.groupby('season')['fan_vs_uniform_contrast'].mean().std()
            robust_std = mech_data.groupby('season')['robust_fail_rate'].mean().std()

            tpi_consistency = 1 / (tpi_std + 0.01)  # Add small constant to avoid division by zero
            fan_consistency = 1 / (fan_std + 0.01)
            robust_consistency = 1 / (robust_std + 0.01)

            consistency_scores[mech] = [tpi_consistency, fan_consistency, robust_consistency]

    # Create grouped bar chart
    x = np.arange(len(mechanisms_subset))

    width = 0.25
    metrics = ['TPI consistency', 'Fan expression consistency', 'Robustness consistency']

    colors_metrics = ['steelblue', 'lightcoral', 'lightgreen']

    for i, metric in enumerate(metrics):
        values = [consistency_scores.get(mech, [0, 0, 0])[i] for mech in mechanisms_subset]
        ax2.bar(x + i*width, values, width, label=metric, alpha=0.8, color=colors_metrics[i])

    ax2.set_xlabel('Mechanism')
    ax2.set_ylabel('Consistency score (higher is better)')
    ax2.set_title('Cross-season consistency', fontweight='bold')

    ax2.set_xticks(x + width)
    ax2.set_xticklabels(mechanisms_subset, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    save_figure_with_config(fig, 'q4_seasonal_variation_analysis', output_dirs, config)

def create_q4_pareto_frontier(
    pareto_data: pd.DataFrame,
    output_dirs: Dict[str, Path],
    config: VisualizationConfig
) -> None:

    """
    Create Pareto frontier plot for multi-objective optimization (showcase).
    
    Args:
        output_dir: Directory to save figures
        figsize: Figure size tuple
    """
    df = pareto_data.copy()
    required = ['tpi_season_avg', 'fan_vs_uniform_contrast', 'robust_fail_rate']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f'Missing required columns for Pareto plot: {missing}')

    df['robustness_score'] = 1.0 - df['robust_fail_rate'].astype(float)
    if 'is_pareto_optimal' in df.columns:
        is_pareto = (
            df['is_pareto_optimal']
            .astype(str)
            .str.strip()
            .str.lower()
            .isin(['true', '1', 'yes'])
        )
    else:
        is_pareto = None

    fig = plt.figure(figsize=config.get_figure_size('large_figure'))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(
        df['tpi_season_avg'].astype(float),
        df['fan_vs_uniform_contrast'].astype(float),
        df['robustness_score'].astype(float),
        c='lightgray',
        alpha=0.4,
        s=30,
        label='All configurations',
    )

    if is_pareto is not None and is_pareto.any():
        pareto_df = df[is_pareto].copy()
        ax.scatter(
            pareto_df['tpi_season_avg'].astype(float),
            pareto_df['fan_vs_uniform_contrast'].astype(float),
            pareto_df['robustness_score'].astype(float),
            c='red',
            alpha=0.9,
            s=80,
            label='Pareto-optimal',
        )

    ax.set_xlabel('TPI')
    ax.set_ylabel('Fan expression')
    ax.set_zlabel('Robustness (1 - fail rate)')
    ax.set_title('Showcase: Pareto frontier (3D)', fontweight='bold')
    ax.legend()

    ax.view_init(elev=20, azim=45)

    plt.tight_layout()

    save_figure_with_config(fig, 'q4_pareto_frontier', output_dirs, config)

def create_q4_mechanism_recommendation(
    metrics_data: pd.DataFrame,
    output_dirs: Dict[str, Path],
    config: VisualizationConfig
) -> None:

    """
    Create mechanism recommendation decision tree.
    
    Args:
        output_dir: Directory to save figures
        figsize: Figure size tuple
    """
    fig, ax = plt.subplots(figsize=config.get_figure_size('large_figure'))

    # Decision tree structure
    from matplotlib.patches import FancyBboxPatch

    # Decision nodes
    decisions = {
        'root': {'pos': (0.5, 0.9), 'text': 'Producer priority?', 'size': (0.2, 0.08)},
        'fairness': {'pos': (0.2, 0.7), 'text': 'Fairness-first', 'size': (0.15, 0.06)},
        'entertainment': {'pos': (0.8, 0.7), 'text': 'Entertainment-first', 'size': (0.15, 0.06)},
        'balance': {'pos': (0.5, 0.7), 'text': 'Balanced', 'size': (0.15, 0.06)},
    }

    # Recommendation nodes
    recommendations = {
        'rank': {'pos': (0.1, 0.5), 'text': 'Rank\nHigh TPI', 'color': 'lightblue'},
        'percent_judge_save': {'pos': (0.5, 0.5), 'text': 'Percent + Judge Save\nBalanced', 'color': 'lightgreen'},
        'percent': {'pos': (0.9, 0.5), 'text': 'Percent\nHigh fan expression', 'color': 'lightcoral'},
    }

    # Draw decision nodes
    for node, props in decisions.items():
        bbox = FancyBboxPatch((props['pos'][0] - props['size'][0]/2, 
                              props['pos'][1] - props['size'][1]/2),
                             props['size'][0], props['size'][1],
                             boxstyle="round,pad=0.01", 
                             facecolor='lightyellow', edgecolor='black')
        ax.add_patch(bbox)
        ax.text(props['pos'][0], props['pos'][1], props['text'], 
               ha='center', va='center', fontsize=10)

    # Draw recommendation nodes
    for node, props in recommendations.items():
        bbox = FancyBboxPatch((props['pos'][0] - 0.08, props['pos'][1] - 0.04),
                             0.16, 0.08,
                             boxstyle="round,pad=0.01", 
                             facecolor=props['color'], edgecolor='black')
        ax.add_patch(bbox)
        ax.text(props['pos'][0], props['pos'][1], props['text'], 
               ha='center', va='center', fontsize=10)

    # Draw connections
    connections = [
        (decisions['root']['pos'], decisions['fairness']['pos']),
        (decisions['root']['pos'], decisions['balance']['pos']),
        (decisions['root']['pos'], decisions['entertainment']['pos']),
        (decisions['fairness']['pos'], recommendations['rank']['pos']),
        (decisions['balance']['pos'], recommendations['percent_judge_save']['pos']),
        (decisions['entertainment']['pos'], recommendations['percent']['pos']),
    ]

    for start, end in connections:
        ax.plot([start[0], end[0]], [start[1], end[1]], 'k-', alpha=0.7, linewidth=2)

    df = metrics_data.copy()
    df = df[df['outlier_mult'] == sorted(df['outlier_mult'].unique())[0]].copy()
    perf = df.groupby('mechanism').agg({
        'tpi_season_avg': 'mean',
        'fan_vs_uniform_contrast': 'mean',
        'robust_fail_rate': 'mean',
    }).reset_index()
    perf['robustness'] = 1.0 - perf['robust_fail_rate']

    table_mechs = ['rank', 'percent_judge_save', 'percent']
    table_data = []
    for mech in table_mechs:
        row = perf[perf['mechanism'] == mech]
        if len(row) == 0:
            continue
        r = row.iloc[0]
        table_data.append([mech, f"{r['tpi_season_avg']:.2f}", f"{r['fan_vs_uniform_contrast']:.2f}", f"{r['robustness']:.2f}"])

    table = ax.table(cellText=table_data,
                    colLabels=['Mechanism', 'TPI', 'Fan expression', 'Robustness'],

                    cellLoc='center',
                    loc='lower center',
                    bbox=[0.2, 0.1, 0.6, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('DWTS mechanism selection guide (data-informed)', fontsize=16, fontweight='bold')
    ax.axis('off')

    plt.tight_layout()

    save_figure_with_config(fig, 'q4_mechanism_recommendation', output_dirs, config)

def create_q4_ml_feature_importance(
    feature_importance: pd.DataFrame,
    output_dirs: Dict[str, Path],
    config: VisualizationConfig,
) -> None:
    df = feature_importance.copy()
    if 'targets' in df.columns:
        df = df[df['targets'].astype(str).str.contains('tpi', case=False, na=False)]
    df = df.sort_values('importance', ascending=False).head(15)

    fig, ax = plt.subplots(figsize=config.get_figure_size('single_column'))
    ax.barh(df['feature'].astype(str), df['importance'].astype(float), color='steelblue', alpha=0.85)
    ax.set_xlabel('Importance')
    ax.set_title('Showcase: Feature importance (Q4 meta-model)', fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_figure_with_config(fig, 'q4_ml_feature_importance', output_dirs, config)

def generate_all_q4_visualizations(
    data_dir: Path,
    output_dirs: Dict[str, Path],
    config: VisualizationConfig,
    showcase: bool = False
) -> None:

    """
    Generate all Q4 visualizations.
    
    Args:
        data_dir: Directory containing input data files
        output_dir: Directory to save output figures
    """
    print("🎨 Generating Q4 visualizations...")

    # Load data
    try:
        metrics_data = pd.read_csv(data_dir / 'outputs' / 'tables' / 'mcm2026c_q4_new_system_metrics.csv')

        attack_cols = [
            'robust_fail_rate_fixed',
            'robust_fail_rate_random_bottom_k',
            'robust_fail_rate_add',
            'robust_fail_rate_redistribute',
        ]
        present_attack_cols = [c for c in attack_cols if c in metrics_data.columns]
        if present_attack_cols:
            has_any = metrics_data[present_attack_cols].notna().any().any()
            if bool(has_any):
                df_tmp = metrics_data.copy()
                cols = ['robust_fail_rate'] + present_attack_cols
                v = df_tmp[cols].apply(pd.to_numeric, errors='coerce')
                df_tmp['robust_fail_rate'] = v.max(axis=1, skipna=True)

                if 'n_sims' in df_tmp.columns:
                    p = pd.to_numeric(df_tmp['robust_fail_rate'], errors='coerce')
                    n = pd.to_numeric(df_tmp['n_sims'], errors='coerce')
                    ok = (n > 0) & p.notna()
                    se = pd.Series(np.nan, index=df_tmp.index, dtype=float)
                    se.loc[ok] = np.sqrt(p.loc[ok] * (1.0 - p.loc[ok]) / n.loc[ok])
                    df_tmp['robust_fail_rate_se'] = se

                metrics_data = df_tmp

        print(f"✅ Loaded data: {len(metrics_data)} metrics records")

        config.apply_matplotlib_style()

        # Generate visualizations
        create_q4_mechanism_tradeoff_scatter(metrics_data, output_dirs, config)
        print("✅ Created mechanism trade-off scatter plot")

        create_q4_robustness_curves(metrics_data, output_dirs, config)
        print("✅ Created robustness curves")

        create_q4_champion_uncertainty_analysis(metrics_data, output_dirs, config)
        print("✅ Created champion uncertainty analysis")

        create_q4_seasonal_variation_analysis(metrics_data, output_dirs, config)
        print("✅ Created seasonal variation analysis")

        create_q4_mechanism_recommendation(metrics_data, output_dirs, config)
        print("✅ Created mechanism recommendation decision tree")

        if showcase:
            pareto_data = pd.read_csv(
                data_dir
                / 'outputs'
                / 'tables'
                / 'showcase'
                / 'mcm2026c_q4_ml_pareto_frontier.csv'
            )
            create_q4_pareto_frontier(pareto_data, output_dirs, config)
            print("✅ Created showcase Pareto frontier plot")

            feature_importance = pd.read_csv(
                data_dir
                / 'outputs'
                / 'tables'
                / 'showcase'
                / 'mcm2026c_q4_ml_feature_importance.csv'
            )
            create_q4_ml_feature_importance(feature_importance, output_dirs, config)
            print("✅ Created showcase feature importance")

        print(f"🎉 Q4 visualizations completed! Saved to {output_dirs['tiff']} and {output_dirs['eps']}")

    except Exception as e:
        print(f"❌ Error generating Q4 visualizations: {e}")
        raise

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Generate Q4 figures (TIFF + EPS).')
    parser.add_argument('--data-dir', type=Path, default=Path('.'), help='Project root directory')
    parser.add_argument(
        '--ini',
        type=Path,
        default=None,
        help='Optional visualization ini file path (font/dpi overrides)',
    )
    parser.add_argument('--showcase', action='store_true', help='Also generate appendix-only figures')
    args = parser.parse_args()

    config = VisualizationConfig.from_ini(args.ini) if args.ini is not None else VisualizationConfig()
    output_structure = create_output_directories(args.data_dir / 'outputs' / 'figures', ['Q4'])

    generate_all_q4_visualizations(args.data_dir, output_structure['Q4'], config, showcase=args.showcase)