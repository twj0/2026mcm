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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Prepare data for heatmaps
    ess_pivot = uncertainty_data.pivot_table(
        values='ess_ratio', index='season', columns='week', fill_value=np.nan
    )
    evidence_pivot = uncertainty_data.pivot_table(
        values='evidence', index='season', columns='week', fill_value=np.nan
    )
    
    # Left heatmap: ESS Ratio
    sns.heatmap(ess_pivot, ax=ax1, cmap='RdYlBu_r', vmin=0, vmax=1, 
                cbar_kws={'label': 'ESS Ratio'}, annot=False)
    ax1.set_title('ESS Ratio (Lower = Higher Uncertainty)', fontweight='bold')
    ax1.set_xlabel('Week')
    ax1.set_ylabel('Season')
    
    # Right heatmap: Evidence
    sns.heatmap(evidence_pivot, ax=ax2, cmap='viridis', vmin=0, vmax=1,
                cbar_kws={'label': 'Evidence'}, annot=False)
    ax2.set_title('Evidence (Brighter = Stronger Constraints)', fontweight='bold')
    ax2.set_xlabel('Week')
    ax2.set_ylabel('Season')
    
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
                    fmt='o', capsize=5, capthick=2, markersize=8, color='steelblue')
        
        # Highlight eliminated contestants
        eliminated_mask = high_week_data['eliminated_this_week']
        if eliminated_mask.any():
            eliminated_indices = [i for i, elim in enumerate(eliminated_mask) if elim]
            ax1.scatter([x_pos[i] for i in eliminated_indices], 
                       [high_week_data.iloc[i]['fan_share_mean'] for i in eliminated_indices],
                       color='red', s=100, marker='x', linewidth=3, label='Eliminated')
        
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
                    fmt='o', capsize=5, capthick=2, markersize=8, color='darkgreen')
        
        # Highlight eliminated contestants
        eliminated_mask = low_week_data['eliminated_this_week']
        if eliminated_mask.any():
            eliminated_indices = [i for i, elim in enumerate(eliminated_mask) if elim]
            ax2.scatter([x_pos[i] for i in eliminated_indices], 
                       [low_week_data.iloc[i]['fan_share_mean'] for i in eliminated_indices],
                       color='red', s=100, marker='x', linewidth=3, label='Eliminated')
        
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
    ax.scatter(not_eliminated['judge_score_pct'], not_eliminated['fan_share_mean'],
              alpha=0.6, s=50, c='steelblue', label='Not eliminated')
    
    # Plot eliminated contestants
    ax.scatter(eliminated['judge_score_pct'], eliminated['fan_share_mean'],
              alpha=0.8, s=80, c='red', marker='x', linewidth=2, label='Eliminated')
    
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
        ax.annotate(f"{row['celebrity_name'][:10]}...", 
                   (row['judge_score_pct'], row['fan_share_mean']),
                   xytext=(5, 5), textcoords='offset points', fontsize=9,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))
    
    for _, row in low_judge_high_fan.head(3).iterrows():
        ax.annotate(f"{row['celebrity_name'][:10]}...", 
                   (row['judge_score_pct'], row['fan_share_mean']),
                   xytext=(5, 5), textcoords='offset points', fontsize=9,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))
    
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
            bars1 = ax1.bar(x_pos, percent_data['fan_share_mean'], 
                           color='lightblue', alpha=0.8, edgecolor='black')
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
                    bars1[j].set_color('red')
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
            bars2 = ax2.bar(x_pos, rank_data['fan_share_mean'], 
                           color='lightcoral', alpha=0.8, edgecolor='black')
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
                    bars2[j].set_color('darkred')
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
    
    ax2.plot(angles, bayesian_scores, 'o-', linewidth=3, label='Structured Inference', color='gold')
    ax2.fill(angles, bayesian_scores, alpha=0.25, color='gold')
    ax2.plot(angles, ml_avg_scores, 'o-', linewidth=3, label='ML Baselines', color='lightblue')
    ax2.fill(angles, ml_avg_scores, alpha=0.25, color='lightblue')
    
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(categories)
    ax2.set_ylim(0, 1)
    ax2.set_title('Showcase: Method Capability Profile', fontweight='bold')
    ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax2.grid(True)
    
    plt.tight_layout()
    
    # Save using config
    save_figure_with_config(fig, 'q1_statistical_vs_ml_comparison', output_dirs, config)

def generate_all_q1_visualizations(
    data_dir: Path,
    output_dirs: Dict[str, Path],
    config: VisualizationConfig,
    showcase: bool = False
) -> None:
    """
    Generate all Q1 visualizations.
    
    Args:
        data_dir: Directory containing input data files
        output_dirs: Dictionary with 'tiff' and 'eps' paths
        config: Configuration object
        showcase: Whether to generate showcase-only figures
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
        
        create_q1_mechanism_comparison(posterior_data, output_dirs, config)
        print("✅ Created mechanism comparison plot")

        if showcase:
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
    args = parser.parse_args()

    config = VisualizationConfig.from_ini(args.ini) if args.ini is not None else VisualizationConfig()
    output_structure = create_output_directories(args.data_dir / 'outputs' / 'figures', ['Q1'])

    generate_all_q1_visualizations(args.data_dir, output_structure['Q1'], config, showcase=args.showcase)