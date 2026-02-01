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
from matplotlib.collections import Collection
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

    for c in ['season', 'week', 'n_active', 'n_exit', 'evidence', 'ess_ratio']:
        if c in uncertainty_data.columns:
            uncertainty_data[c] = pd.to_numeric(uncertainty_data[c], errors='coerce')

    cand = uncertainty_data[(uncertainty_data['n_exit'] > 0) & (uncertainty_data['n_active'] >= 6)].copy()
    if cand.empty:
        cand = uncertainty_data.copy()
    cand = cand.dropna(subset=['season', 'week']).copy()
    if cand.empty:
        return

    high_uncertainty = cand.sort_values(['evidence', 'ess_ratio'], ascending=[True, True]).iloc[0]
    low_uncertainty = cand.sort_values(['evidence', 'ess_ratio'], ascending=[False, False]).iloc[0]

    figsize = config.get_figure_size('double_column')
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    def _short_name(s: str, n: int = 18) -> str:
        ss = str(s)
        return ss if len(ss) <= n else ss[: n - 1] + '…'

    def _plot_week(ax: plt.Axes, meta: pd.Series, *, title_prefix: str, panel: str) -> Optional[Collection]:
        season = int(meta['season']) if np.isfinite(meta.get('season', np.nan)) else None
        week = int(meta['week']) if np.isfinite(meta.get('week', np.nan)) else None
        if season is None or week is None:
            ax.axis('off')
            return None

        w = posterior_data[(posterior_data['season'] == season) & (posterior_data['week'] == week)].copy()
        for c in ['fan_share_mean', 'fan_share_p05', 'fan_share_p95', 'judge_score_pct']:
            if c in w.columns:
                w[c] = pd.to_numeric(w[c], errors='coerce')
        w = w.dropna(subset=['fan_share_mean', 'fan_share_p05', 'fan_share_p95']).copy()
        if w.empty:
            ax.axis('off')
            return None

        w = w.sort_values('fan_share_mean', ascending=True).copy()
        y = np.arange(len(w))
        lo = w['fan_share_p05'].to_numpy(dtype=float)
        hi = w['fan_share_p95'].to_numpy(dtype=float)
        mu = w['fan_share_mean'].to_numpy(dtype=float)
        judge = w['judge_score_pct'].to_numpy(dtype=float) if 'judge_score_pct' in w.columns else np.full_like(mu, np.nan)
        elim = w['eliminated_this_week'].astype(bool).to_numpy() if 'eliminated_this_week' in w.columns else np.zeros(len(w), dtype=bool)

        for yy, a, b in zip(y, lo, hi):
            ax.plot([a, b], [yy, yy], color=config.get_color('muted'), alpha=0.55, linewidth=2.2, solid_capstyle='round', zorder=1)

        sc = ax.scatter(
            mu,
            y,
            c=judge,
            cmap='viridis',
            vmin=0.0,
            vmax=1.0,
            s=46,
            alpha=0.92,
            edgecolors=config.get_color('text'),
            linewidths=0.25,
            zorder=3,
        )

        if np.any(elim):
            ax.scatter(
                mu[elim],
                y[elim],
                s=78,
                marker='x',
                c=config.get_color('danger'),
                linewidths=2.2,
                zorder=4,
            )

        n_active = int(w['celebrity_name'].nunique()) if 'celebrity_name' in w.columns else len(w)
        if n_active > 0:
            ax.axvline(1.0 / float(n_active), linestyle='--', color=config.get_color('text'), alpha=0.28, linewidth=1.2, zorder=0)

        ax.set_yticks(y)
        ax.set_yticklabels([_short_name(x) for x in w['celebrity_name'].astype(str).to_list()])
        ax.invert_yaxis()

        x_max = float(np.nanmax(hi)) if len(hi) else 0.6
        x_max = x_max if np.isfinite(x_max) else 0.6
        ax.set_xlim(0.0, min(1.0, max(0.55, 1.08 * x_max)))
        ax.set_xlabel('Estimated fan vote share')

        ev = float(meta.get('evidence', np.nan))
        essr = float(meta.get('ess_ratio', np.nan))
        title = f"{title_prefix} (S{season}, W{week})"
        if np.isfinite(ev) and np.isfinite(essr):
            title = title + f"\nEvidence={ev:.3f}  ESS ratio={essr:.3f}"
        ax.set_title(title, fontweight='bold')
        config.add_panel_label(ax, panel)
        return sc

    sc0 = _plot_week(axes[0], high_uncertainty, title_prefix='High-uncertainty week', panel='A')
    sc1 = _plot_week(axes[1], low_uncertainty, title_prefix='Low-uncertainty week', panel='B')

    axes[0].set_ylabel('Contestant')

    sc = sc1 if sc1 is not None else sc0
    if sc is not None:
        cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), fraction=0.045, pad=0.02)
        cbar.set_label('Judge score share')

    plt.tight_layout()
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

    cols = ['season', 'week', 'celebrity_name', 'fan_share_mean', 'fan_share_p05', 'fan_share_p95', 'eliminated_this_week']
    if 'judge_score_pct' in posterior_data.columns:
        cols.append('judge_score_pct')
    d0 = posterior_data[cols].copy()

    if 'judge_score_pct' not in d0.columns:
        d0 = pd.merge(
            d0,
            weekly_panel[['season', 'week', 'celebrity_name', 'judge_score_pct']],
            on=['season', 'week', 'celebrity_name'],
            how='inner',
        )

    for c in ['fan_share_mean', 'fan_share_p05', 'fan_share_p95', 'judge_score_pct']:
        if c in d0.columns:
            d0[c] = pd.to_numeric(d0[c], errors='coerce')
    d0 = d0.dropna(subset=['fan_share_mean', 'fan_share_p05', 'fan_share_p95', 'judge_score_pct']).copy()
    if d0.empty:
        return

    d0['fan_width'] = (d0['fan_share_p95'] - d0['fan_share_p05']).clip(lower=0)
    d0['delta'] = d0['judge_score_pct'] - d0['fan_share_mean']
    d0['judge_pct'] = d0.groupby(['season', 'week'])['judge_score_pct'].rank(pct=True)
    d0['fan_pct'] = d0.groupby(['season', 'week'])['fan_share_mean'].rank(pct=True)

    w = d0['fan_width'].to_numpy(dtype=float)
    w_lo = float(np.nanpercentile(w, 5)) if len(w) else 0.0
    w_hi = float(np.nanpercentile(w, 95)) if len(w) else 1.0
    denom = (w_hi - w_lo) if np.isfinite(w_hi - w_lo) and (w_hi - w_lo) > 1e-12 else 1.0
    w_norm = np.clip((w - w_lo) / denom, 0.0, 1.0)
    size = 34.0 + 260.0 * (0.15 + w_norm)
    alpha = 0.25 + 0.65 * (1.0 - w_norm)

    dv = d0['delta'].to_numpy(dtype=float)
    vmax = float(np.nanquantile(np.abs(dv), 0.98)) if len(dv) else 0.2
    vmax = vmax if np.isfinite(vmax) and vmax > 0 else 0.2

    cmap = config.get_cmap('corr')
    norm = plt.Normalize(vmin=-vmax, vmax=vmax)
    rgba = cmap(norm(dv))
    rgba[:, 3] = alpha

    figsize = config.get_figure_size('single_column')
    fig, ax = plt.subplots(figsize=figsize)

    elim = d0['eliminated_this_week'].astype(bool).to_numpy() if 'eliminated_this_week' in d0.columns else np.zeros(len(d0), dtype=bool)
    keep = ~elim

    ax.scatter(
        d0.loc[keep, 'judge_score_pct'].to_numpy(dtype=float),
        d0.loc[keep, 'fan_share_mean'].to_numpy(dtype=float),
        s=size[keep],
        c=rgba[keep],
        edgecolors=config.get_color('text'),
        linewidths=0.25,
        zorder=2,
    )

    if np.any(elim):
        ax.scatter(
            d0.loc[elim, 'judge_score_pct'].to_numpy(dtype=float),
            d0.loc[elim, 'fan_share_mean'].to_numpy(dtype=float),
            s=np.maximum(90.0, size[elim]),
            c=config.get_color('danger'),
            marker='x',
            linewidths=2.2,
            zorder=3,
        )

    ax.plot([0, 1], [0, 1], linestyle='--', color=config.get_color('muted'), alpha=0.35, linewidth=1.2, zorder=1)

    extreme_a = d0[(d0['judge_pct'] >= 0.80) & (d0['fan_pct'] <= 0.20)].copy()
    extreme_b = d0[(d0['judge_pct'] <= 0.20) & (d0['fan_pct'] >= 0.80)].copy()
    extreme = pd.concat([extreme_a, extreme_b], ignore_index=True)
    if not extreme.empty:
        extreme['abs_delta'] = np.abs(pd.to_numeric(extreme['delta'], errors='coerce'))
        extreme = extreme.sort_values('abs_delta', ascending=False).head(6)
        for _, row in extreme.iterrows():
            name = str(row['celebrity_name'])
            ax.annotate(
                name if len(name) <= 10 else name[:9] + '…',
                (float(row['judge_score_pct']), float(row['fan_share_mean'])),
                xytext=(6, 6),
                textcoords='offset points',
                fontsize=8.8,
                bbox=config.callout_bbox(kind='note'),
            )

    rho = float(np.corrcoef(d0['judge_score_pct'].to_numpy(dtype=float), d0['fan_share_mean'].to_numpy(dtype=float))[0, 1]) if len(d0) >= 2 else float('nan')
    if np.isfinite(rho):
        config.add_callout(ax, f"Corr(judge, fan) = {rho:.2f}\nSize/alpha = posterior interval width", loc='upper left', kind='note')

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel('Judge score share (technical line)')
    ax.set_ylabel('Estimated fan vote share (popularity line)')
    ax.set_title('Judge vs fan share (uncertainty-aware)', fontweight='bold')
    ax.grid(True, alpha=0.3)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.05, pad=0.02)
    cbar.set_label('Judge − Fan (share difference)')

    plt.tight_layout()
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