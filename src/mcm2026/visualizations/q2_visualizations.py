"""
Q2 Visualization Module: Counterfactual Mechanism Comparison

This module implements all visualization functions for Q2 analysis including:
- Mechanism difference distribution plots
- Judge Save impact comparisons
- Controversial seasons heatmaps
- Fan-judge divergence analysis
- Bobby Bones case study
- ML mechanism prediction results
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


def create_q2_mechanism_difference_distribution(
    comparison_data: pd.DataFrame,
    output_dirs: Dict[str, Path],
    config: VisualizationConfig
) -> None:
    """
    Create mechanism difference distribution plot.
    
    Args:
        comparison_data: DataFrame with mechanism comparison results
        output_dirs: Dictionary with 'tiff' and 'eps' paths
        config: Configuration object
    """
    figsize = config.get_figure_size('double_column')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Left plot: Difference weeks distribution
    diff_weeks = comparison_data['diff_weeks_percent_vs_rank']
    ax1.hist(diff_weeks, bins=range(0, int(diff_weeks.max()) + 2), 
             alpha=0.7, color='steelblue', edgecolor='black')
    ax1.set_xlabel('Number of Weeks with Different Elimination')
    ax1.set_ylabel('Number of Seasons')
    ax1.set_title('Percent vs Rank: Distribution of Divergent Weeks', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add statistics text
    mean_diff = diff_weeks.mean()
    std_diff = diff_weeks.std()
    ax1.text(0.7, 0.8, f'Mean: {mean_diff:.1f} weeks\nSD: {std_diff:.1f} weeks', 
             transform=ax1.transAxes, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Right plot: Difference rate vs season
    if 'n_exit_weeks' in comparison_data.columns:
        diff_rate = comparison_data['diff_weeks_percent_vs_rank'] / comparison_data['n_exit_weeks']
        diff_rate = diff_rate.fillna(0)  # Handle division by zero
    else:
        # Estimate exit weeks if not available
        diff_rate = comparison_data['diff_weeks_percent_vs_rank'] / 8  # Assume average 8 weeks
    
    scatter = ax2.scatter(comparison_data['season'], diff_rate, 
                         alpha=0.7, s=60, c=comparison_data['season'], cmap='viridis')
    ax2.set_xlabel('Season')
    ax2.set_ylabel('Divergence Rate (Divergent Weeks / Exit Weeks)')
    ax2.set_title('Season-level Divergence Rate', fontweight='bold')
    
    # Add mean line
    mean_rate = diff_rate.mean()
    ax2.axhline(y=mean_rate, color='red', linestyle='--', 
               label=f'Mean: {mean_rate:.2%}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('Season')
    
    plt.tight_layout()
    
    # Save using config
    save_figure_with_config(fig, 'q2_mechanism_difference_distribution', output_dirs, config)


def create_q2_judge_save_impact(
    comparison_data: pd.DataFrame,
    output_dirs: Dict[str, Path],
    config: VisualizationConfig
) -> None:
    """
    Create Judge Save impact comparison plot.
    
    Args:
        comparison_data: DataFrame with mechanism comparison results
        output_dirs: Dictionary with 'tiff' and 'eps' paths
        config: Configuration object
    """
    figsize = config.get_figure_size('large_figure')
    fig, ax = plt.subplots(figsize=figsize)
    
    seasons = comparison_data['season']
    x = np.arange(len(seasons))
    width = 0.2
    
    # Four groups of bars
    bars1 = ax.bar(x - 1.5*width, comparison_data['match_rate_percent'], width, 
                   label='Percent (Standard)', color='lightblue', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x - 0.5*width, comparison_data['match_rate_percent_judge_save'], width,
                   label='Percent + Judge Save', color='darkblue', alpha=0.8, edgecolor='black')
    bars3 = ax.bar(x + 0.5*width, comparison_data['match_rate_rank'], width,
                   label='Rank (Standard)', color='lightcoral', alpha=0.8, edgecolor='black')
    bars4 = ax.bar(x + 1.5*width, comparison_data['match_rate_rank_judge_save'], width,
                   label='Rank + Judge Save', color='darkred', alpha=0.8, edgecolor='black')
    
    ax.set_xlabel('Season')
    ax.set_ylabel('Match Rate')
    ax.set_title('Impact of Judge Save on Matching Rate', fontweight='bold')
    ax.set_xticks(x[::2])  # Show every other season to avoid crowding
    ax.set_xticklabels(seasons.iloc[::2])
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Add average impact annotations
    percent_delta = (
        comparison_data['match_rate_percent_judge_save'] - comparison_data['match_rate_percent']
    ).mean()
    rank_delta = (
        comparison_data['match_rate_rank_judge_save'] - comparison_data['match_rate_rank']
    ).mean()
    
    ax.text(0.02, 0.98, f'Average Δ (Judge Save - Standard):\nPercent: {percent_delta:.2%}\nRank: {rank_delta:.2%}', 
           transform=ax.transAxes, va='top',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    plt.tight_layout()
    
    save_figure_with_config(fig, 'q2_judge_save_impact', output_dirs, config)


def create_q2_controversial_seasons_heatmap(
    comparison_data: pd.DataFrame,
    output_dirs: Dict[str, Path],
    config: VisualizationConfig
) -> None:
    """
    Create controversial seasons identification heatmap.
    
    Args:
        comparison_data: DataFrame with mechanism comparison results
        output_dirs: Dictionary with 'tiff' and 'eps' paths
        config: Configuration object
    """
    fig, ax = plt.subplots(figsize=config.get_figure_size('single_column'))
    
    # Calculate controversy metrics
    controversy_metrics = pd.DataFrame({
        'Season': comparison_data['season'],
        'Divergent weeks': comparison_data['diff_weeks_percent_vs_rank'],
        'Match rate (Percent)': comparison_data['match_rate_percent'],
        'Match rate (Rank)': comparison_data['match_rate_rank'],
        'Judge Save impact': (comparison_data['match_rate_percent'] - 
                        comparison_data['match_rate_percent_judge_save']),
    })
    
    # Add fan-judge divergence if available
    if 'mean_fan_share_observed' in comparison_data.columns and 'mean_judge_pct_observed' in comparison_data.columns:
        controversy_metrics['Fan-Judge divergence'] = (comparison_data['mean_fan_share_observed'] - 
                                           comparison_data['mean_judge_pct_observed']).abs()
    
    # Standardize data for heatmap
    controversy_normalized = controversy_metrics.set_index('Season')
    controversy_normalized = (controversy_normalized - controversy_normalized.mean()) / controversy_normalized.std()
    
    # Handle any NaN values
    controversy_normalized = controversy_normalized.fillna(0)
    
    # Create heatmap
    sns.heatmap(
        controversy_normalized.T,
        ax=ax,
        cmap='RdBu_r',
        center=0,
        annot=False,
        fmt='.1f',
        cbar_kws={'label': 'Standardized score'},
    )
    ax.set_title('Controversial Seasons Heatmap', fontweight='bold')
    ax.set_xlabel('Season')
    ax.set_ylabel('Metrics')
    
    # Get season list for indexing
    season_list = list(controversy_metrics['Season'])
    
    # Highlight special seasons (like S27 Bobby Bones if present)
    if 27 in season_list:
        s27_idx = season_list.index(27)
        ax.axvline(x=s27_idx+0.5, color='gold', linewidth=3, alpha=0.8)
        ax.text(s27_idx+0.5, -0.5, 'S27\nBobby Bones', ha='center', 
               bbox=dict(boxstyle='round', facecolor='gold', alpha=0.7))
    
    # Identify and annotate top controversial seasons
    if len(controversy_normalized) > 0:
        season_score = controversy_normalized.abs().mean(axis=1)
        top_controversial = season_score.nlargest(min(3, len(season_score)))
        
        for season, score in top_controversial.items():
            if season != 27 and season in season_list:  # Don't double-annotate S27
                season_idx = season_list.index(season)
                ax.axvline(x=season_idx+0.5, color='orange', linewidth=2, alpha=0.6)
                ax.text(season_idx+0.5, len(controversy_normalized.columns), f'S{season}', 
                       ha='center', va='bottom', fontsize=10,
                       bbox=dict(boxstyle='round', facecolor='orange', alpha=0.7))
    
    plt.tight_layout()
    
    save_figure_with_config(fig, 'q2_controversial_seasons_heatmap', output_dirs, config)


def create_q2_fan_judge_divergence(
    comparison_data: pd.DataFrame,
    output_dirs: Dict[str, Path],
    config: VisualizationConfig
) -> None:
    """
    Create fan-judge divergence scatter plot.
    
    Args:
        comparison_data: DataFrame with mechanism comparison results
        output_dirs: Dictionary with 'tiff' and 'eps' paths
        config: Configuration object
    """
    fig, ax = plt.subplots(figsize=config.get_figure_size('single_column'))
    
    if 'mean_fan_share_observed' not in comparison_data.columns or 'mean_judge_pct_observed' not in comparison_data.columns:
        raise ValueError('Required columns missing: mean_fan_share_observed, mean_judge_pct_observed')

    fan_judge_divergence = (comparison_data['mean_fan_share_observed'] - comparison_data['mean_judge_pct_observed']).abs()
    
    # Create scatter plot
    scatter = ax.scatter(fan_judge_divergence, comparison_data['diff_weeks_percent_vs_rank'],
                        c=comparison_data['season'], cmap='viridis', s=80, alpha=0.7)
    
    # Add trend line
    if len(fan_judge_divergence) > 1:
        z = np.polyfit(fan_judge_divergence, comparison_data['diff_weeks_percent_vs_rank'], 1)
        p = np.poly1d(z)
        ax.plot(fan_judge_divergence, p(fan_judge_divergence), "r--", alpha=0.8, linewidth=2)
        
        # Calculate correlation
        correlation = np.corrcoef(fan_judge_divergence, comparison_data['diff_weeks_percent_vs_rank'])[0,1]
        ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}', transform=ax.transAxes,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.set_xlabel('Fan-Judge divergence |mean fan share - mean judge pct|')
    ax.set_ylabel('Divergent weeks (Percent vs Rank)')
    ax.set_title('Fan-Judge Divergence vs Divergent Weeks', fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Season')
    ax.grid(True, alpha=0.3)
    
    # Annotate extreme cases
    extreme_seasons = comparison_data[comparison_data['diff_weeks_percent_vs_rank'] >= 3]['season']
    for season in extreme_seasons:
        season_data = comparison_data[comparison_data['season'] == season].iloc[0]
        season_divergence = fan_judge_divergence[comparison_data['season'] == season].iloc[0]
        ax.annotate(f'S{season}', 
                   (season_divergence, season_data['diff_weeks_percent_vs_rank']),
                   xytext=(5, 5), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    
    save_figure_with_config(fig, 'q2_fan_judge_divergence', output_dirs, config)


def create_q2_bobby_bones_case_study(
    comparison_data: pd.DataFrame,
    output_dirs: Dict[str, Path],
    config: VisualizationConfig
) -> None:
    """
    Create Bobby Bones case study analysis (Season 27 if available).
    
    Args:
        comparison_data: DataFrame with mechanism comparison results
        output_dirs: Dictionary with 'tiff' and 'eps' paths
        config: Configuration object
    """
    raise NotImplementedError('Bobby Bones case study requires additional per-week data not produced by the mainline pipeline.')


def create_q2_ml_mechanism_prediction(
    prediction_summary: pd.DataFrame,
    feature_importance: pd.DataFrame,
    output_dirs: Dict[str, Path],
    config: VisualizationConfig
) -> None:
    """
    Create ML mechanism prediction results (showcase).
    
    Args:
        prediction_summary: DataFrame with prediction results
        feature_importance: DataFrame with feature importance
        output_dirs: Dictionary with 'tiff' and 'eps' paths
        config: Configuration object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=config.get_figure_size('double_column'))
    
    perf = prediction_summary.dropna(subset=['cv_accuracy_mean']).copy()
    models = perf['model_name'].astype(str)
    accuracies = perf['cv_accuracy_mean'].astype(float)

    bars = ax1.bar(models, accuracies, color='steelblue', alpha=0.8, edgecolor='black')
    ax1.set_ylabel('CV accuracy (mean)')
    ax1.set_title('Showcase: Mechanism Prediction (Classifier)', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.0, 1.0)
    
    # Add value labels
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    ax1.tick_params(axis='x', rotation=45)
    
    fi = feature_importance.copy()
    if 'task_type' in fi.columns:
        fi = fi[(fi['task_type'] == 'classification') & (fi['target'] == 'is_best_mechanism')]
    fi = fi.sort_values('importance', ascending=False).head(10)

    bars2 = ax2.barh(fi['feature'].astype(str), fi['importance'].astype(float), color='darkred', alpha=0.8)
    ax2.set_xlabel('Feature importance')
    ax2.set_title('Showcase: Top Features', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    for bar in bars2:
        width = bar.get_width()
        ax2.text(width + 0.01, bar.get_y() + bar.get_height()/2.,
                f'{width:.2f}', ha='left', va='center', fontweight='bold')
    
    plt.tight_layout()
    
    save_figure_with_config(fig, 'q2_ml_mechanism_prediction', output_dirs, config)


def generate_all_q2_visualizations(
    data_dir: Path,
    output_dirs: Dict[str, Path],
    config: VisualizationConfig,
    showcase: bool = False
) -> None:
    """
    Generate all Q2 visualizations.
    
    Args:
        data_dir: Directory containing input data files
        output_dirs: Dictionary with 'tiff' and 'eps' paths
        config: Configuration object
    """
    print("🎨 Generating Q2 visualizations...")
    
    # Load data
    try:
        comparison_data = pd.read_csv(data_dir / 'outputs' / 'tables' / 'mcm2026c_q2_mechanism_comparison.csv')
        
        print(f"✅ Loaded data: {len(comparison_data)} comparison records")

        config.apply_matplotlib_style()
        
        # Generate visualizations (updating function calls to use new signature)
        create_q2_mechanism_difference_distribution(comparison_data, output_dirs, config)
        print("✅ Created mechanism difference distribution")

        create_q2_judge_save_impact(comparison_data, output_dirs, config)
        print("✅ Created Judge Save impact")

        create_q2_controversial_seasons_heatmap(comparison_data, output_dirs, config)
        print("✅ Created controversial seasons heatmap")

        create_q2_fan_judge_divergence(comparison_data, output_dirs, config)
        print("✅ Created fan-judge divergence")

        if showcase:
            pred = pd.read_csv(
                data_dir
                / 'outputs'
                / 'tables'
                / 'showcase'
                / 'mcm2026c_q2_ml_mechanism_prediction.csv'
            )
            fi = pd.read_csv(
                data_dir
                / 'outputs'
                / 'tables'
                / 'showcase'
                / 'mcm2026c_q2_ml_feature_importance.csv'
            )
            create_q2_ml_mechanism_prediction(pred, fi, output_dirs, config)
            print("✅ Created showcase ML mechanism prediction")
        
        print(f"🎉 Q2 visualizations completed! Saved to {output_dirs['tiff']} and {output_dirs['eps']}")
        
    except Exception as e:
        print(f"❌ Error generating Q2 visualizations: {e}")
        raise

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Generate Q2 figures (TIFF + EPS).')
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
    output_structure = create_output_directories(args.data_dir / 'outputs' / 'figures', ['Q2'])

    generate_all_q2_visualizations(args.data_dir, output_structure['Q2'], config, showcase=args.showcase)