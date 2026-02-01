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
import matplotlib.patheffects as pe
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


def create_q4_mechanism_tradeoff_scatter(
    metrics_data: pd.DataFrame,
    showcase_pareto: pd.DataFrame | None,
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
    df = metrics_data.copy()
    if df.empty:
        return

    for c in [
        'tpi_season_avg',
        'fan_vs_uniform_contrast',
        'robust_fail_rate',
        'outlier_mult',
        'alpha',
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    df = df.dropna(subset=['mechanism', 'outlier_mult', 'tpi_season_avg', 'fan_vs_uniform_contrast']).copy()
    if df.empty:
        return

    if 'alpha' in df.columns:
        alpha_vals = df['alpha'].dropna().unique()
        if len(alpha_vals) > 1:
            alpha0 = float(np.nanmedian(df['alpha']))
            df = df[np.isclose(df['alpha'], alpha0)].copy()

    outlier_levels = sorted(df['outlier_mult'].dropna().unique())
    if not outlier_levels:
        return

    mechanisms = sorted(df['mechanism'].dropna().unique())
    if not mechanisms:
        return

    colors = {m: config.get_color(m) for m in mechanisms}
    stroke_fc = str(config.callout_bbox(kind='note').get('facecolor', '#ffffff'))

    n_out = len(outlier_levels)
    fig = plt.figure(figsize=(6.1 * n_out + 4.8, 8.0))
    gs = fig.add_gridspec(
        2,
        n_out + 1,
        width_ratios=[1.0] * n_out + [0.72],
        height_ratios=[1.15, 0.85],
        wspace=0.25,
        hspace=0.25,
    )
    axes_top = [fig.add_subplot(gs[0, i]) for i in range(n_out)]
    ax_rank = fig.add_subplot(gs[1, :n_out])
    ax_chips = fig.add_subplot(gs[:, -1])
    ax_chips.axis('off')

    def _pareto_front(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if len(x) == 0:
            return np.array([]), np.array([])
        order = np.argsort(x)
        xs = x[order]
        ys = y[order]
        keep_x: list[float] = []
        keep_y: list[float] = []
        best = -np.inf
        for xi, yi in zip(xs[::-1], ys[::-1]):
            if np.isfinite(yi) and yi >= best:
                keep_x.append(float(xi))
                keep_y.append(float(yi))
                best = float(yi)
        if not keep_x:
            return np.array([]), np.array([])
        return np.array(keep_x[::-1]), np.array(keep_y[::-1])

    recs: list[dict] = []
    rank_rows: list[dict] = []

    for i, outlier_mult in enumerate(outlier_levels):
        ax = axes_top[i]
        sub = df[df['outlier_mult'] == outlier_mult].copy()
        if sub.empty:
            ax.axis('off')
            continue

        try:
            if len(sub) >= 30:
                sns.kdeplot(
                    data=sub,
                    x='fan_vs_uniform_contrast',
                    y='tpi_season_avg',
                    fill=True,
                    thresh=0.03,
                    levels=7,
                    cmap='Greys',
                    alpha=0.16,
                    bw_adjust=0.85,
                    ax=ax,
                )
                sns.kdeplot(
                    data=sub,
                    x='fan_vs_uniform_contrast',
                    y='tpi_season_avg',
                    fill=False,
                    thresh=0.03,
                    levels=7,
                    color='#111827',
                    alpha=0.10,
                    linewidths=0.8,
                    bw_adjust=0.85,
                    ax=ax,
                )
        except Exception:
            pass

        x_front, y_front = _pareto_front(
            sub['fan_vs_uniform_contrast'].to_numpy(dtype=float),
            sub['tpi_season_avg'].to_numpy(dtype=float),
        )
        if len(x_front) and len(y_front):
            ax.plot(
                x_front,
                y_front,
                color=config.get_color('text'),
                alpha=0.35,
                linewidth=1.6,
                zorder=3,
            )

        has_fan_se = 'fan_vs_uniform_contrast_se' in sub.columns
        has_tpi_boot = 'tpi_boot_p025' in sub.columns and 'tpi_boot_p975' in sub.columns
        has_tpi_std = 'tpi_std' in sub.columns and 'tpi_n' in sub.columns

        fan_xerr: dict[str, float] = {}
        tpi_yerr: dict[str, float] = {}
        if has_fan_se:
            for mech in mechanisms:
                s = sub.loc[sub['mechanism'] == mech, 'fan_vs_uniform_contrast_se']
                s = s[np.isfinite(s)]
                fan_xerr[mech] = float(1.96 * np.sqrt(np.mean(np.square(s)))) if len(s) else 0.0

        if has_tpi_boot:
            for mech in mechanisms:
                lo = sub.loc[sub['mechanism'] == mech, 'tpi_boot_p025']
                hi = sub.loc[sub['mechanism'] == mech, 'tpi_boot_p975']
                lo = lo[np.isfinite(lo)]
                hi = hi[np.isfinite(hi)]
                if len(lo) and len(hi):
                    se = np.mean((hi.to_numpy() - lo.to_numpy()) / (2.0 * 1.96))
                    tpi_yerr[mech] = float(1.96 * se)
                else:
                    tpi_yerr[mech] = 0.0
        elif has_tpi_std:
            for mech in mechanisms:
                std = sub.loc[sub['mechanism'] == mech, 'tpi_std']
                n = sub.loc[sub['mechanism'] == mech, 'tpi_n']
                std = std[np.isfinite(std)]
                n = n[np.isfinite(n)]
                if len(std) and len(n):
                    se = np.mean(std.to_numpy() / np.sqrt(np.maximum(n.to_numpy(), 1.0)))
                    tpi_yerr[mech] = float(1.96 * se)
                else:
                    tpi_yerr[mech] = 0.0

        mechanism_avg = (
            sub.groupby('mechanism')
            .agg(
                tpi_season_avg=('tpi_season_avg', 'mean'),
                fan_vs_uniform_contrast=('fan_vs_uniform_contrast', 'mean'),
                robust_fail_rate=('robust_fail_rate', 'mean'),
            )
            .reset_index()
        )

        for _, row in mechanism_avg.iterrows():
            mech = str(row['mechanism'])
            base_c = colors.get(mech, config.get_color('muted'))
            rf = float(row.get('robust_fail_rate', np.nan))
            rf = rf if np.isfinite(rf) else float(np.nanmean(sub['robust_fail_rate']))
            size = (1.0 - rf) * 300.0 + 60.0

            ax.scatter(
                float(row['fan_vs_uniform_contrast']),
                float(row['tpi_season_avg']),
                s=float(size) * 1.8,
                c=base_c,
                alpha=0.14,
                linewidths=0.0,
                zorder=4,
            )
            ax.scatter(
                float(row['fan_vs_uniform_contrast']),
                float(row['tpi_season_avg']),
                s=float(size),
                c=base_c,
                alpha=0.82,
                linewidths=0.35,
                edgecolors=config.get_color('text'),
                zorder=6,
            )

            xerr = fan_xerr.get(mech, 0.0)
            yerr = tpi_yerr.get(mech, 0.0)
            if (xerr and np.isfinite(xerr)) or (yerr and np.isfinite(yerr)):
                ax.errorbar(
                    float(row['fan_vs_uniform_contrast']),
                    float(row['tpi_season_avg']),
                    xerr=xerr if xerr and np.isfinite(xerr) else None,
                    yerr=yerr if yerr and np.isfinite(yerr) else None,
                    fmt='none',
                    ecolor=base_c,
                    elinewidth=1,
                    alpha=0.35,
                    capsize=2,
                    zorder=5,
                )

            t = ax.annotate(
                mech.replace('_', '\n'),
                (float(row['fan_vs_uniform_contrast']), float(row['tpi_season_avg'])),
                xytext=(6, 6),
                textcoords='offset points',
                fontsize=9,
                zorder=8,
            )
            t.set_path_effects([pe.withStroke(linewidth=2.6, foreground=stroke_fc)])

        if showcase_pareto is not None and not showcase_pareto.empty and i == 0:
            try:
                dfp = showcase_pareto.copy()
                for c in ['tpi_season_avg', 'fan_vs_uniform_contrast']:
                    dfp[c] = pd.to_numeric(dfp.get(c, np.nan), errors='coerce')
                dfp = dfp.dropna(subset=['tpi_season_avg', 'fan_vs_uniform_contrast']).copy()
                if not dfp.empty:
                    ax.scatter(
                        dfp['fan_vs_uniform_contrast'].to_numpy(dtype=float),
                        dfp['tpi_season_avg'].to_numpy(dtype=float),
                        s=34,
                        facecolors='none',
                        edgecolors=config.get_color('danger'),
                        linewidths=1.0,
                        alpha=0.70,
                        zorder=2,
                    )
            except Exception:
                pass

        ideal_rect = plt.Rectangle(
            (0.6, 0.7),
            0.3,
            0.2,
            fill=True,
            facecolor=config.get_color('warning'),
            alpha=0.07,
            edgecolor=config.get_color('warning'),
            linewidth=1.8,
            linestyle='--',
            zorder=1,
        )
        ax.add_patch(ideal_rect)
        ax.text(
            0.75,
            0.8,
            'Ideal region',
            ha='center',
            va='center',
            bbox=config.callout_bbox(kind='warn'),
            zorder=9,
        )

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Fan expression strength (fan vs uniform contrast)')
        ax.set_ylabel('Technical Protection Index (TPI)' if i == 0 else '')
        ax.set_title(f'Stress: outlier_mult={outlier_mult}', fontweight='bold')
        ax.grid(True, alpha=0.22)

        def _minmax(s: pd.Series) -> pd.Series:
            v = pd.to_numeric(s, errors='coerce')
            lo = float(np.nanmin(v.to_numpy(dtype=float)))
            hi = float(np.nanmax(v.to_numpy(dtype=float)))
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                return v * 0.0
            return (v - lo) / (hi - lo)

        mech_stats = mechanism_avg.copy()
        mech_stats['robust_ok'] = 1.0 - pd.to_numeric(mech_stats.get('robust_fail_rate', np.nan), errors='coerce')
        mech_stats['x_norm'] = _minmax(mech_stats['fan_vs_uniform_contrast'])
        mech_stats['y_norm'] = _minmax(mech_stats['tpi_season_avg'])
        mech_stats['r_norm'] = _minmax(mech_stats['robust_ok'])
        mech_stats['decision_score'] = 0.45 * mech_stats['y_norm'] + 0.35 * mech_stats['x_norm'] + 0.20 * mech_stats['r_norm']

        mech_stats = mech_stats.dropna(subset=['decision_score']).copy()
        if not mech_stats.empty:
            best_idx = int(mech_stats['decision_score'].astype(float).idxmax())
            best = mech_stats.loc[best_idx]
            recs.append(
                {
                    'outlier_mult': float(outlier_mult),
                    'mechanism': str(best['mechanism']),
                    'tpi_season_avg': float(best['tpi_season_avg']),
                    'fan_vs_uniform_contrast': float(best['fan_vs_uniform_contrast']),
                    'robust_fail_rate': float(best.get('robust_fail_rate', np.nan)),
                    'decision_score': float(best['decision_score']),
                }
            )

            tmp = mech_stats[['mechanism', 'decision_score']].copy()
            tmp['outlier_mult'] = float(outlier_mult)
            tmp['rank'] = tmp['decision_score'].rank(ascending=False, method='min')
            for _, rr in tmp.iterrows():
                rank_rows.append(
                    {
                        'outlier_mult': float(outlier_mult),
                        'mechanism': str(rr['mechanism']),
                        'rank': float(rr['rank']),
                    }
                )

    fig.text(0.01, 0.98, 'A', fontweight='bold', fontsize=12, va='top')
    fig.text(0.01, 0.46, 'B', fontweight='bold', fontsize=12, va='top')
    fig.text(0.86, 0.98, 'C', fontweight='bold', fontsize=12, va='top')

    if rank_rows:
        r = pd.DataFrame(rank_rows)
        r['outlier_mult'] = pd.to_numeric(r['outlier_mult'], errors='coerce')
        r['rank'] = pd.to_numeric(r['rank'], errors='coerce')
        r = r.dropna(subset=['outlier_mult', 'rank']).copy()

        xticks = [float(x) for x in outlier_levels]
        ax_rank.set_xticks(range(len(xticks)))
        ax_rank.set_xticklabels([str(x) for x in xticks])
        ax_rank.set_xlim(-0.4, len(xticks) - 0.6)

        y_max = float(np.nanmax(r['rank'].to_numpy(dtype=float))) if not r.empty else float(len(mechanisms))
        if not np.isfinite(y_max) or y_max <= 0:
            y_max = float(len(mechanisms))

        for mech in mechanisms:
            rr = r[r['mechanism'].astype(str) == str(mech)].copy()
            if rr.empty:
                continue
            rr = rr.sort_values('outlier_mult')
            xs = [xticks.index(float(x)) for x in rr['outlier_mult'].to_list() if float(x) in xticks]
            ys = rr.loc[rr['outlier_mult'].isin(xticks), 'rank'].to_numpy(dtype=float)
            if len(xs) != len(ys) or len(xs) == 0:
                continue

            base_c = colors.get(mech, config.get_color('muted'))
            ax_rank.plot(xs, ys, color=base_c, alpha=0.22, linewidth=4.4, zorder=1)
            ax_rank.plot(xs, ys, color=base_c, alpha=0.86, linewidth=2.0, zorder=3)
            ax_rank.scatter(xs, ys, s=48, c=base_c, alpha=0.92, edgecolors=config.get_color('text'), linewidths=0.25, zorder=4)

            t = ax_rank.annotate(
                mech,
                (xs[-1], float(ys[-1])),
                xytext=(8, 0),
                textcoords='offset points',
                va='center',
                fontsize=9,
                zorder=5,
            )
            t.set_path_effects([pe.withStroke(linewidth=2.6, foreground=stroke_fc)])

        ax_rank.set_title('Rank shift under stress (by decision score)', fontweight='bold')
        ax_rank.set_xlabel('Stress level (outlier_mult)')
        ax_rank.set_ylabel('Rank (1 = best)')
        ax_rank.set_yticks(range(1, int(y_max) + 1))
        ax_rank.set_ylim(float(y_max) + 0.6, 0.4)
        ax_rank.grid(True, alpha=0.18)

    if recs:
        ax_chips.set_title('Recommendation', fontweight='bold', fontsize=11)
        y0 = 0.92
        dy = 0.27 if len(recs) <= 3 else 0.20
        for rec in sorted(recs, key=lambda d: float(d.get('outlier_mult', 0.0))):
            mech = str(rec.get('mechanism', ''))
            base_c = colors.get(mech, config.get_color('muted'))
            rf = rec.get('robust_fail_rate', np.nan)
            rf_txt = f"{float(rf):.2f}" if rf is not None and np.isfinite(float(rf)) else 'NA'
            label = (
                f"Stress={float(rec.get('outlier_mult', np.nan))}\n"
                f"Pick: {mech}\n"
                f"TPI={float(rec.get('tpi_season_avg', np.nan)):.2f}  "
                f"Fan={float(rec.get('fan_vs_uniform_contrast', np.nan)):.2f}\n"
                f"Fail={rf_txt}"
            )
            ax_chips.text(
                0.02,
                y0,
                label,
                transform=ax_chips.transAxes,
                ha='left',
                va='top',
                fontsize=9.3,
                bbox={
                    'boxstyle': 'round,pad=0.38,rounding_size=0.18',
                    'facecolor': base_c,
                    'edgecolor': config.get_color('text'),
                    'linewidth': 0.6,
                    'alpha': 0.12,
                },
            )
            ax_chips.text(
                0.02,
                y0,
                label,
                transform=ax_chips.transAxes,
                ha='left',
                va='top',
                fontsize=9.3,
                color=config.get_color('text'),
            )
            y0 -= dy

    save_figure_with_config(fig, 'q4_mechanism_tradeoff_scatter', output_dirs, config)


def create_q4_tradeoff_pareto_frontier_2d(
    metrics_data: pd.DataFrame,
    output_dirs: Dict[str, Path],
    config: VisualizationConfig,
) -> None:
    outlier_levels = sorted(metrics_data['outlier_mult'].unique())
    if len(outlier_levels) == 0:
        return

    ncols = len(outlier_levels)
    fig, axes = plt.subplots(1, ncols, figsize=(4.2 * ncols, 4.0), sharey=True)
    if ncols == 1:
        axes = [axes]

    mechanisms = sorted(metrics_data['mechanism'].unique())
    colors = {m: config.get_color(m) for m in mechanisms}

    for i, outlier_mult in enumerate(outlier_levels):
        ax = axes[i]
        data_subset = metrics_data[metrics_data['outlier_mult'] == outlier_mult]

        mechanism_avg = data_subset.groupby('mechanism').agg({
            'tpi_season_avg': 'mean',
            'fan_vs_uniform_contrast': 'mean',
            'robust_fail_rate': 'mean',
        }).reset_index()
        if mechanism_avg.empty:
            ax.axis('off')
            continue

        mechanism_avg['robustness'] = 1.0 - pd.to_numeric(mechanism_avg['robust_fail_rate'], errors='coerce')

        size = 40.0 + 220.0 * mechanism_avg['robustness'].clip(0, 1).to_numpy(dtype=float)
        ax.scatter(
            mechanism_avg['fan_vs_uniform_contrast'].to_numpy(dtype=float),
            mechanism_avg['tpi_season_avg'].to_numpy(dtype=float),
            s=size,
            c=[colors.get(m, config.get_color('muted')) for m in mechanism_avg['mechanism'].astype(str)],
            alpha=0.78,
            linewidths=0.6,
            edgecolors=config.get_color('text'),
        )

        for _, r in mechanism_avg.iterrows():
            ax.annotate(
                str(r['mechanism']).replace('_', '\n'),
                (float(r['fan_vs_uniform_contrast']), float(r['tpi_season_avg'])),
                xytext=(4, 3),
                textcoords='offset points',
                fontsize=8,
            )

        df2 = mechanism_avg[['fan_vs_uniform_contrast', 'tpi_season_avg']].copy()
        df2 = df2.apply(pd.to_numeric, errors='coerce').dropna().sort_values('fan_vs_uniform_contrast')
        xs = df2['fan_vs_uniform_contrast'].to_numpy(dtype=float)
        ys = df2['tpi_season_avg'].to_numpy(dtype=float)
        pareto_x: list[float] = []
        pareto_y: list[float] = []
        y_max = -1e9
        for x, y in zip(xs, ys):
            if y > y_max:
                pareto_x.append(float(x))
                pareto_y.append(float(y))
                y_max = float(y)
        if len(pareto_x) >= 2:
            ax.plot(pareto_x, pareto_y, color=config.get_color('text'), linewidth=1.8, alpha=0.85, label='Pareto envelope')

        ax.set_xlabel('Fan expression (fan vs uniform contrast)')
        ax.set_title(f'outlier_mult={outlier_mult}', fontweight='bold')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        if i == 0:
            ax.set_ylabel('Technical Protection Index (TPI)')

        if len(pareto_x) >= 2:
            ax.legend(loc='lower right')

    plt.tight_layout()
    save_figure_with_config(fig, 'q4_tradeoff_pareto_frontier_2d', output_dirs, config)


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
    colors = {m: config.get_color(m) for m in mechanisms}

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

        c = colors.get(mech, config.get_color('muted'))

        ax1.plot(
            outlier_values,
            fail_rates,
            '-',
            linewidth=4.2,
            alpha=0.20,
            color=c,
            zorder=2,
        )
        ax1.plot(
            outlier_values,
            fail_rates,
            'o-',
            label=mech,
            linewidth=2.2,
            markersize=7,
            color=c,
            zorder=3,
        )
        ax1.fill_between(outlier_values, 
                        np.array(fail_rates) - np.array(fail_rate_band),
                        np.array(fail_rates) + np.array(fail_rate_band),
                        alpha=0.15, color=c, zorder=1)

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
            c = colors.get(mech, config.get_color('muted'))
            ax2.plot(outlier_values, ranks, '-', linewidth=4.2, alpha=0.20, color=c, zorder=2)
            ax2.plot(outlier_values, ranks, 'o-', label=mech, linewidth=2.2, markersize=7, color=c, zorder=3)

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
    df = metrics_data.copy()
    if df.empty:
        return

    for c in [
        'champion_entropy',
        'champion_mode_prob',
        'tpi_season_avg',
        'robust_fail_rate',
        'outlier_mult',
        'alpha',
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    if 'alpha' in df.columns:
        alpha_vals = df['alpha'].dropna().unique()
        if len(alpha_vals) > 1:
            alpha0 = float(np.nanmedian(df['alpha']))
            df = df[np.isclose(df['alpha'], alpha0)].copy()

    base_outlier = None
    if 'outlier_mult' in df.columns:
        outliers = df['outlier_mult'].dropna().unique()
        if len(outliers):
            base_outlier = 2.0 if np.any(np.isclose(outliers, 2.0)) else float(np.nanmin(outliers))

    df_base = df.copy()
    if base_outlier is not None and 'outlier_mult' in df_base.columns:
        df_base = df_base[np.isclose(df_base['outlier_mult'], float(base_outlier))].copy()

    if 'mechanism' not in df_base.columns or df_base.empty:
        return

    if 'champion_entropy' in df_base.columns:
        mech_ord = (
            df_base
            .dropna(subset=['mechanism', 'champion_entropy'])
            .groupby('mechanism')['champion_entropy']
            .median()
            .sort_values()
        )
        mechanism_order = [str(m) for m in mech_ord.index.to_list()]
    else:
        mechanism_order = sorted(df_base['mechanism'].dropna().astype(str).unique())

    colors = {m: config.get_color(m) for m in mechanism_order}

    stroke_fc = str(config.callout_bbox(kind='note').get('facecolor', '#ffffff'))

    fig = plt.figure(figsize=config.get_figure_size('large_figure'))
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.55, 1.0],
        height_ratios=[1.0, 1.0],
        wspace=0.28,
        hspace=0.34,
    )
    ax1 = fig.add_subplot(gs[:, 0])
    ax3 = fig.add_subplot(gs[0, 1])
    ax4 = fig.add_subplot(gs[1, 1])

    def _pretty_mech(m: str) -> str:
        return str(m).replace('_', ' ')

    # Subplot 1: Champion entropy distribution
    ent = df_base[['mechanism', 'champion_entropy']].copy() if 'champion_entropy' in df_base.columns else pd.DataFrame()
    ent = ent.dropna(subset=['mechanism', 'champion_entropy']).copy() if not ent.empty else ent

    entropy_xlim = None
    if not ent.empty:
        xv = ent['champion_entropy'].to_numpy(dtype=float)
        xv = xv[np.isfinite(xv)]
        if xv.size:
            x0 = float(np.nanquantile(xv, 0.02))
            x1 = float(np.nanquantile(xv, 0.98))
            pad = 0.06 * max(x1 - x0, 1e-6)
            entropy_xlim = (max(0.0, x0 - pad), x1 + pad)

    if not ent.empty:
        ent['mechanism'] = ent['mechanism'].astype(str)
        pal = {m: colors.get(m, config.get_color('muted')) for m in mechanism_order}
        sns.violinplot(
            data=ent,
            y='mechanism',
            x='champion_entropy',
            order=mechanism_order,
            orient='h',
            cut=0,
            inner=None,
            linewidth=0.8,
            scale='width',
            palette=pal,
            ax=ax1,
        )
        for coll in ax1.collections:
            try:
                coll.set_alpha(0.35)
            except Exception:
                pass
        sns.stripplot(
            data=ent,
            y='mechanism',
            x='champion_entropy',
            order=mechanism_order,
            orient='h',
            jitter=0.18,
            size=2.4,
            alpha=0.22,
            color=config.get_color('text'),
            ax=ax1,
        )

        for yi, m in enumerate(mechanism_order):
            s = ent.loc[ent['mechanism'] == m, 'champion_entropy']
            if len(s) == 0:
                continue
            med = float(np.nanmedian(s.to_numpy(dtype=float)))
            ax1.scatter(
                [med],
                [yi],
                s=80,
                c=colors.get(m, config.get_color('muted')),
                alpha=0.16,
                linewidths=0.0,
                zorder=3,
            )
            ax1.scatter(
                [med],
                [yi],
                s=46,
                c=colors.get(m, config.get_color('muted')),
                alpha=0.90,
                edgecolors=config.get_color('text'),
                linewidths=0.35,
                zorder=4,
            )

    ax1.set_xlabel('Champion entropy (baseline stress)')
    ax1.set_ylabel('')
    ax1.set_title('Outcome randomness (champion entropy)', fontweight='bold')
    ax1.set_yticklabels([_pretty_mech(m) for m in mechanism_order])
    if entropy_xlim is not None:
        ax1.set_xlim(entropy_xlim)
    ax1.grid(True, axis='x', alpha=0.25)
    config.add_panel_label(ax1, 'A')

    # Subplot 2: Champion mode probability distribution
    ax2 = ax1.inset_axes([0.56, 0.06, 0.41, 0.34])
    mp = df_base[['mechanism', 'champion_mode_prob']].copy() if 'champion_mode_prob' in df_base.columns else pd.DataFrame()
    mp = mp.dropna(subset=['mechanism', 'champion_mode_prob']).copy() if not mp.empty else mp
    if not mp.empty:
        mp['mechanism'] = mp['mechanism'].astype(str)
        pal = {m: colors.get(m, config.get_color('muted')) for m in mechanism_order}
        sns.violinplot(
            data=mp,
            y='mechanism',
            x='champion_mode_prob',
            order=mechanism_order,
            orient='h',
            cut=0,
            inner=None,
            linewidth=0.6,
            scale='width',
            palette=pal,
            ax=ax2,
        )
        for coll in ax2.collections:
            try:
                coll.set_alpha(0.30)
            except Exception:
                pass
        ax2.set_title('Mode probability', fontsize=9.0, fontweight='bold')
        ax2.set_xlabel('')
        ax2.set_ylabel('')
        ax2.set_yticklabels([])
        ax2.grid(True, axis='x', alpha=0.20)

    # Subplot 3: Uncertainty vs Technical Protection scatter
    df_traj = df.copy()
    df_traj = df_traj.dropna(subset=['mechanism', 'champion_entropy', 'tpi_season_avg', 'outlier_mult']).copy()
    if not df_traj.empty:
        g = (
            df_traj
            .groupby(['mechanism', 'outlier_mult'], as_index=False)
            .agg({'champion_entropy': 'mean', 'tpi_season_avg': 'mean'})
            .sort_values(['mechanism', 'outlier_mult'])
        )
        for mech in mechanism_order:
            sub = g[g['mechanism'].astype(str) == mech].copy()
            if len(sub) < 2:
                continue
            xs = sub['champion_entropy'].to_numpy(dtype=float)
            ys = sub['tpi_season_avg'].to_numpy(dtype=float)
            oms = sub['outlier_mult'].to_numpy(dtype=float)
            c = colors.get(str(mech), config.get_color('muted'))

            ax3.plot(xs, ys, '-', linewidth=4.0, alpha=0.18, color=c, zorder=1)
            if np.isfinite(oms).any():
                o0 = float(np.nanmin(oms))
                o1 = float(np.nanmax(oms))
                denom = max(o1 - o0, 1e-9)
                sizes = 26.0 + 60.0 * (oms - o0) / denom
            else:
                sizes = np.full_like(xs, 46.0)

            ax3.scatter(xs, ys, s=sizes * 1.35, c=c, alpha=0.14, linewidths=0.0, zorder=2)
            ax3.scatter(xs, ys, s=sizes, c=c, alpha=0.88, edgecolors=config.get_color('text'), linewidths=0.30, zorder=3)

            if base_outlier is not None and np.isfinite(oms).any():
                idx0 = int(np.argmin(np.abs(oms - float(base_outlier))))
                ax3.scatter(
                    [float(xs[idx0])],
                    [float(ys[idx0])],
                    s=float(sizes[idx0]) + 18.0,
                    facecolors='none',
                    edgecolors=c,
                    linewidths=1.2,
                    alpha=0.90,
                    zorder=4,
                )

            if np.isfinite(xs[-1]) and np.isfinite(ys[-1]):
                t = ax3.annotate(
                    _pretty_mech(str(mech)),
                    (float(xs[-1]), float(ys[-1])),
                    xytext=(5, 3),
                    textcoords='offset points',
                    fontsize=8.0,
                    color=config.get_color('text'),
                    zorder=5,
                )
                t.set_path_effects([pe.withStroke(linewidth=2.2, foreground=stroke_fc)])

    ax3.set_xlabel('Champion entropy')
    ax3.set_ylabel('Technical Protection Index (TPI)')
    ax3.set_title('Randomness vs technical protection', fontweight='bold')
    if entropy_xlim is not None:
        ax3.set_xlim(entropy_xlim)
    else:
        ax3.set_xlim(0.0, 2.0)
    ax3.set_ylim(0.0, 1.0)
    ax3.grid(True, alpha=0.25)
    config.add_panel_label(ax3, 'B')
    config.add_callout(ax3, 'Trajectories show how stress shifts systems', loc='upper right', kind='note')

    # Subplot 4: Ideal region analysis
    ideal_entropy_range = (0.5, 0.8)  # Moderate randomness
    ideal_tpi_threshold = 0.7  # High technical protection

    summary_rows: list[dict[str, float | str]] = []
    for mech in mechanism_order:
        sub = df_base[df_base['mechanism'].astype(str) == str(mech)].copy()
        sub = sub.dropna(subset=['champion_entropy', 'tpi_season_avg']).copy()
        if sub.empty:
            continue

        entv = sub['champion_entropy'].to_numpy(dtype=float)
        tpiv = sub['tpi_season_avg'].to_numpy(dtype=float)
        mpv = sub['champion_mode_prob'].to_numpy(dtype=float) if 'champion_mode_prob' in sub.columns else np.array([])
        rf = sub['robust_fail_rate'].to_numpy(dtype=float) if 'robust_fail_rate' in sub.columns else np.array([])

        ent_med = float(np.nanmedian(entv))
        tpi_med = float(np.nanmedian(tpiv))
        mp_med = float(np.nanmedian(mpv)) if mpv.size else float('nan')
        rf_med = float(np.nanmedian(rf)) if rf.size else float('nan')

        mask = (
            (entv >= ideal_entropy_range[0])
            & (entv <= ideal_entropy_range[1])
            & (tpiv >= ideal_tpi_threshold)
        )
        ideal_rate = float(np.mean(mask)) if mask.size else 0.0

        summary_rows.append(
            {
                'mechanism': str(mech),
                'entropy_med': ent_med,
                'tpi_med': tpi_med,
                'mode_prob_med': mp_med,
                'robust_fail_med': rf_med,
                'ideal_rate': ideal_rate,
                'entropy_q25': float(np.nanquantile(entv, 0.25)),
                'entropy_q75': float(np.nanquantile(entv, 0.75)),
                'tpi_q25': float(np.nanquantile(tpiv, 0.25)),
                'tpi_q75': float(np.nanquantile(tpiv, 0.75)),
            }
        )

    summ = pd.DataFrame(summary_rows)
    summ = summ.dropna(subset=['entropy_med', 'tpi_med']).copy() if not summ.empty else summ

    ideal_rect = plt.Rectangle(
        (ideal_entropy_range[0], ideal_tpi_threshold),
        ideal_entropy_range[1] - ideal_entropy_range[0],
        1.0 - ideal_tpi_threshold,
        fill=True,
        facecolor=config.get_color('warning'),
        alpha=0.10,
        edgecolor=config.get_color('warning'),
        linewidth=1.8,
        linestyle='--',
        zorder=0,
    )
    ax4.add_patch(ideal_rect)

    if not summ.empty:
        summ = summ.sort_values(['ideal_rate', 'tpi_med'], ascending=[False, False]).copy()
        for _, r in summ.iterrows():
            mech = str(r['mechanism'])
            c = colors.get(mech, config.get_color('muted'))

            x = float(r['entropy_med'])
            y = float(r['tpi_med'])
            xerr = [[max(0.0, x - float(r['entropy_q25']))], [max(0.0, float(r['entropy_q75']) - x)]]
            yerr = [[max(0.0, y - float(r['tpi_q25']))], [max(0.0, float(r['tpi_q75']) - y)]]

            ax4.scatter([x], [y], s=140, c=c, alpha=0.14, linewidths=0.0, zorder=2)
            ax4.scatter([x], [y], s=70, c=c, alpha=0.90, edgecolors=config.get_color('text'), linewidths=0.35, zorder=3)
            ax4.errorbar([x], [y], xerr=xerr, yerr=yerr, fmt='none', ecolor=c, elinewidth=1.0, alpha=0.30, capsize=2, zorder=2)

        top = summ.head(3)
        for i, r in enumerate(top.itertuples(index=False)):
            mech = str(getattr(r, 'mechanism'))
            chip_lines = [
                _pretty_mech(mech),
                f"Ideal share: {float(getattr(r, 'ideal_rate')):.0%}",
                f"Median TPI: {float(getattr(r, 'tpi_med')):.2f}",
            ]
            mpv = float(getattr(r, 'mode_prob_med'))
            if np.isfinite(mpv):
                chip_lines.append(f"Median mode prob: {mpv:.2f}")
            rf = float(getattr(r, 'robust_fail_med'))
            if np.isfinite(rf):
                chip_lines.append(f"Median fail rate: {rf:.2f}")

            ax4.text(
                0.02,
                0.98 - 0.22 * i,
                "\n".join(chip_lines),
                transform=ax4.transAxes,
                ha='left',
                va='top',
                fontsize=8.3,
                bbox=config.callout_bbox(kind='note'),
                zorder=10,
            )

    ax4.set_xlabel('Champion entropy (baseline)')
    ax4.set_ylabel('Technical Protection Index (TPI)')
    ax4.set_title('Share in ideal region (moderate randomness + high TPI)', fontweight='bold')
    if entropy_xlim is not None:
        ax4.set_xlim(entropy_xlim)
    else:
        ax4.set_xlim(0.0, 2.0)
    ax4.set_ylim(0.0, 1.0)
    ax4.grid(True, alpha=0.25)
    config.add_panel_label(ax4, 'C')
    config.add_callout(ax4, 'Choose systems inside the highlighted band', loc='upper left', kind='warn')

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
    im1 = ax1.imshow(tpi_cv_matrix.T, cmap=config.get_cmap('heatmap'), aspect='auto')

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

    colors_metrics = [config.get_color('primary'), config.get_color('rank'), config.get_color('percent_judge_save')]

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

    base_fc = config.get_color('muted')
    pareto_fc = config.get_color('danger')

    ax.scatter(
        df['tpi_season_avg'].astype(float),
        df['fan_vs_uniform_contrast'].astype(float),
        df['robustness_score'].astype(float),
        c=base_fc,
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
            c=pareto_fc,
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
    df = metrics_data.copy()
    if df.empty:
        return

    for c in [
        'tpi_season_avg',
        'fan_vs_uniform_contrast',
        'robust_fail_rate',
        'outlier_mult',
        'alpha',
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    df = df.dropna(subset=['mechanism', 'outlier_mult', 'tpi_season_avg', 'fan_vs_uniform_contrast']).copy()
    if df.empty:
        return

    if 'alpha' in df.columns:
        a = df['alpha'].dropna().unique()
        if len(a) > 1:
            a0 = float(np.nanmedian(df['alpha']))
            df = df[np.isclose(df['alpha'], a0)].copy()

    outlier_levels = sorted(df['outlier_mult'].dropna().unique())
    if not outlier_levels:
        return

    mechanisms = sorted(df['mechanism'].dropna().unique())
    if not mechanisms:
        return

    colors = {m: config.get_color(m) for m in mechanisms}
    stroke_fc = str(config.callout_bbox(kind='note').get('facecolor', '#ffffff'))

    fig = plt.figure(figsize=config.get_figure_size('large_figure'))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.12, 0.88], height_ratios=[0.62, 0.38], wspace=0.22, hspace=0.25)
    ax_map = fig.add_subplot(gs[0, 0])
    ax_cards = fig.add_subplot(gs[0, 1])
    ax_rank = fig.add_subplot(gs[1, 0])
    ax_notes = fig.add_subplot(gs[1, 1])
    ax_notes.axis('off')

    from matplotlib.patches import FancyBboxPatch

    def _agg_for_outlier(sub: pd.DataFrame) -> pd.DataFrame:
        g = (
            sub.groupby('mechanism')
            .agg(
                tpi=('tpi_season_avg', 'mean'),
                fan=('fan_vs_uniform_contrast', 'mean'),
                fail=('robust_fail_rate', 'mean'),
            )
            .reset_index()
        )
        g['robust'] = 1.0 - pd.to_numeric(g['fail'], errors='coerce')

        if 'tpi_boot_p025' in sub.columns and 'tpi_boot_p975' in sub.columns:
            lo = sub.groupby('mechanism')['tpi_boot_p025'].mean()
            hi = sub.groupby('mechanism')['tpi_boot_p975'].mean()
            g = g.merge(lo.rename('tpi_lo'), left_on='mechanism', right_index=True, how='left')
            g = g.merge(hi.rename('tpi_hi'), left_on='mechanism', right_index=True, how='left')
        else:
            g['tpi_lo'] = np.nan
            g['tpi_hi'] = np.nan

        if 'fan_vs_uniform_contrast_se' in sub.columns:
            se = sub.groupby('mechanism')['fan_vs_uniform_contrast_se'].apply(
                lambda x: float(np.sqrt(np.mean(np.square(pd.to_numeric(x, errors='coerce').dropna().to_numpy(dtype=float)))))
                if len(pd.to_numeric(x, errors='coerce').dropna())
                else 0.0
            )
            g = g.merge(se.rename('fan_se'), left_on='mechanism', right_index=True, how='left')
            g['fan_lo'] = g['fan'] - 1.96 * g['fan_se']
            g['fan_hi'] = g['fan'] + 1.96 * g['fan_se']
        else:
            g['fan_lo'] = np.nan
            g['fan_hi'] = np.nan

        if 'robust_fail_rate_se' in sub.columns:
            se = sub.groupby('mechanism')['robust_fail_rate_se'].apply(
                lambda x: float(np.sqrt(np.mean(np.square(pd.to_numeric(x, errors='coerce').dropna().to_numpy(dtype=float)))))
                if len(pd.to_numeric(x, errors='coerce').dropna())
                else 0.0
            )
            g = g.merge(se.rename('fail_se'), left_on='mechanism', right_index=True, how='left')
            g['fail_lo'] = (g['fail'] - 1.96 * g['fail_se']).clip(0, 1)
            g['fail_hi'] = (g['fail'] + 1.96 * g['fail_se']).clip(0, 1)
        else:
            g['fail_lo'] = np.nan
            g['fail_hi'] = np.nan

        for c in ['tpi', 'fan', 'robust', 'tpi_lo', 'tpi_hi', 'fan_lo', 'fan_hi', 'fail', 'fail_lo', 'fail_hi']:
            if c in g.columns:
                g[c] = pd.to_numeric(g[c], errors='coerce')

        return g

    def _pick_priority(g: pd.DataFrame, priority: str) -> str | None:
        if g.empty:
            return None
        gg = g.copy()
        if priority == 'fairness':
            gg = gg.sort_values(['tpi', 'robust', 'fan'], ascending=[False, False, False])
        elif priority == 'entertainment':
            gg = gg.sort_values(['fan', 'robust', 'tpi'], ascending=[False, False, False])
        else:
            def _minmax(v: np.ndarray) -> np.ndarray:
                lo = float(np.nanmin(v))
                hi = float(np.nanmax(v))
                if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                    return np.zeros_like(v)
                return (v - lo) / (hi - lo)

            x = _minmax(gg['fan'].to_numpy(dtype=float))
            y = _minmax(gg['tpi'].to_numpy(dtype=float))
            r = _minmax(gg['robust'].to_numpy(dtype=float))
            gg = gg.assign(score=0.45 * y + 0.35 * x + 0.20 * r)
            gg = gg.sort_values(['score', 'tpi', 'fan'], ascending=[False, False, False])
        mech = str(gg['mechanism'].iloc[0])
        return mech

    base_outlier = float(outlier_levels[0])
    base = _agg_for_outlier(df[df['outlier_mult'] == base_outlier])
    ax_map.set_title('Mechanism map (baseline stress) + ideal region', fontweight='bold')

    ideal_rect = plt.Rectangle(
        (0.6, 0.7),
        0.4,
        0.3,
        fill=True,
        facecolor=config.get_color('warning'),
        alpha=0.08,
        edgecolor=config.get_color('warning'),
        linewidth=1.6,
        linestyle='--',
        zorder=0,
    )
    ax_map.add_patch(ideal_rect)
    ax_map.text(0.98, 0.98, 'Ideal region', ha='right', va='top', transform=ax_map.transAxes, bbox=config.callout_bbox(kind='warn'))

    if not base.empty:
        for _, r in base.iterrows():
            mech = str(r['mechanism'])
            c = colors.get(mech, config.get_color('muted'))
            ax_map.scatter(float(r['fan']), float(r['tpi']), s=120, c=c, alpha=0.14, linewidths=0.0, zorder=2)
            ax_map.scatter(float(r['fan']), float(r['tpi']), s=60, c=c, alpha=0.90, edgecolors=config.get_color('text'), linewidths=0.35, zorder=3)
            t = ax_map.annotate(
                mech.replace('_', '\n'),
                (float(r['fan']), float(r['tpi'])),
                xytext=(6, 6),
                textcoords='offset points',
                fontsize=8.8,
                zorder=5,
            )
            t.set_path_effects([pe.withStroke(linewidth=2.4, foreground=stroke_fc)])

        picks = {
            'fairness': _pick_priority(base, 'fairness'),
            'balanced': _pick_priority(base, 'balanced'),
            'entertainment': _pick_priority(base, 'entertainment'),
        }
        markers = {'fairness': 's', 'balanced': 'o', 'entertainment': '^'}
        for k, mech in picks.items():
            if mech is None:
                continue
            rr = base[base['mechanism'].astype(str) == str(mech)]
            if rr.empty:
                continue
            x = float(rr['fan'].iloc[0])
            y = float(rr['tpi'].iloc[0])
            ax_map.scatter(
                [x],
                [y],
                s=160,
                facecolors='none',
                edgecolors=config.get_color('text'),
                linewidths=2.0,
                marker=markers.get(k, 'o'),
                zorder=6,
            )

    ax_map.set_xlim(0, 1)
    ax_map.set_ylim(0, 1)
    ax_map.set_xlabel('Fan expression (fan vs uniform contrast)')
    ax_map.set_ylabel('Technical Protection Index (TPI)')
    ax_map.grid(True, alpha=0.22)
    config.add_panel_label(ax_map, 'A')

    ax_cards.set_title('Decision cards (priority × stress)', fontweight='bold')
    ax_cards.axis('off')

    priorities = [
        ('Fairness-first', 'fairness', config.get_color('rank')),
        ('Balanced', 'balanced', config.get_color('percent_judge_save')),
        ('Entertainment-first', 'entertainment', config.get_color('percent')),
    ]

    card_y = [0.94, 0.63, 0.32]
    card_h = 0.26
    card_x = 0.04
    card_w = 0.92

    for (title, key, c0), y0 in zip(priorities, card_y):
        card = FancyBboxPatch(
            (card_x, y0 - card_h),
            card_w,
            card_h,
            boxstyle="round,pad=0.015",
            transform=ax_cards.transAxes,
            facecolor=c0,
            edgecolor=config.get_color('text'),
            linewidth=0.8,
            alpha=0.10,
        )
        ax_cards.add_patch(card)
        ax_cards.text(card_x + 0.02, y0 - 0.03, title, transform=ax_cards.transAxes, ha='left', va='top', fontsize=10.8, fontweight='bold')

        lines: list[str] = []
        for om in outlier_levels[:3]:
            g = _agg_for_outlier(df[df['outlier_mult'] == float(om)])
            mech = _pick_priority(g, key)
            if mech is None:
                continue
            rr = g[g['mechanism'].astype(str) == str(mech)]
            if rr.empty:
                continue
            r0 = rr.iloc[0]
            tpi = float(r0.get('tpi', np.nan))
            fan = float(r0.get('fan', np.nan))
            fail = float(r0.get('fail', np.nan))
            lines.append(f"stress={float(om)}  pick={mech}  TPI={tpi:.2f}  Fan={fan:.2f}  Fail={fail:.2f}")

        ax_cards.text(
            card_x + 0.02,
            y0 - 0.10,
            "\n".join(lines) if lines else 'No data',
            transform=ax_cards.transAxes,
            ha='left',
            va='top',
            fontsize=9.1,
            color=config.get_color('text'),
        )

    config.add_panel_label(ax_cards, 'B')

    rank_rows: list[dict] = []
    for om in outlier_levels:
        g = _agg_for_outlier(df[df['outlier_mult'] == float(om)])
        if g.empty:
            continue

        def _minmax(v: np.ndarray) -> np.ndarray:
            lo = float(np.nanmin(v))
            hi = float(np.nanmax(v))
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                return np.zeros_like(v)
            return (v - lo) / (hi - lo)

        x = _minmax(g['fan'].to_numpy(dtype=float))
        y = _minmax(g['tpi'].to_numpy(dtype=float))
        r = _minmax(g['robust'].to_numpy(dtype=float))
        score = 0.45 * y + 0.35 * x + 0.20 * r
        g = g.assign(score=score)
        g['rank'] = g['score'].rank(ascending=False, method='min')
        for _, rr in g.iterrows():
            rank_rows.append({'outlier_mult': float(om), 'mechanism': str(rr['mechanism']), 'rank': float(rr['rank'])})

    if rank_rows:
        r = pd.DataFrame(rank_rows)
        r = r.dropna(subset=['outlier_mult', 'rank']).copy()
        xticks = [float(x) for x in outlier_levels]
        ax_rank.set_xticks(range(len(xticks)))
        ax_rank.set_xticklabels([str(x) for x in xticks])
        ax_rank.set_xlim(-0.4, len(xticks) - 0.6)

        y_max = float(np.nanmax(r['rank'].to_numpy(dtype=float))) if not r.empty else float(len(mechanisms))
        y_max = y_max if np.isfinite(y_max) and y_max > 0 else float(len(mechanisms))

        focus = ['rank', 'percent_judge_save', 'percent', 'dynamic_weight']
        focus = [m for m in focus if m in mechanisms]
        focus = focus if focus else mechanisms[: min(6, len(mechanisms))]

        for mech in focus:
            rr = r[r['mechanism'].astype(str) == str(mech)].copy()
            if rr.empty:
                continue
            rr = rr.sort_values('outlier_mult')
            xs = [xticks.index(float(x)) for x in rr['outlier_mult'].to_list() if float(x) in xticks]
            ys = rr.loc[rr['outlier_mult'].isin(xticks), 'rank'].to_numpy(dtype=float)
            if len(xs) != len(ys) or len(xs) == 0:
                continue
            c = colors.get(mech, config.get_color('muted'))
            ax_rank.plot(xs, ys, color=c, alpha=0.20, linewidth=4.2, zorder=1)
            ax_rank.plot(xs, ys, color=c, alpha=0.90, linewidth=2.0, zorder=3)
            ax_rank.scatter(xs, ys, s=44, c=c, alpha=0.92, edgecolors=config.get_color('text'), linewidths=0.25, zorder=4)
            t = ax_rank.annotate(mech, (xs[-1], float(ys[-1])), xytext=(8, 0), textcoords='offset points', va='center', fontsize=9, zorder=5)
            t.set_path_effects([pe.withStroke(linewidth=2.4, foreground=stroke_fc)])

        ax_rank.set_title('Rank shift under stress (balanced score)', fontweight='bold')
        ax_rank.set_xlabel('Stress level (outlier_mult)')
        ax_rank.set_ylabel('Rank (1 = best)')
        ax_rank.set_yticks(range(1, int(y_max) + 1))
        ax_rank.set_ylim(float(y_max) + 0.6, 0.4)
        ax_rank.grid(True, alpha=0.18)

    config.add_panel_label(ax_rank, 'C')

    ax_notes.text(
        0.02,
        0.98,
        "How to use:\n"
        "1) Pick producer priority\n"
        "2) Check stress level\n"
        "3) Choose mechanism card\n\n"
        "Balanced score = 0.45·TPI + 0.35·Fan + 0.20·Robust\n"
        "(all normalized within each stress)",
        ha='left',
        va='top',
        fontsize=9.2,
        bbox=config.callout_bbox(kind='note'),
        transform=ax_notes.transAxes,
    )

    fig.suptitle('DWTS mechanism selection guide (data-informed)', fontsize=15, fontweight='bold')
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
    ax.barh(df['feature'].astype(str), df['importance'].astype(float), color=config.get_color('muted'), alpha=0.85)

    ax.set_xlabel('Importance')
    ax.set_title('Showcase: Feature importance (Q4 meta-model)', fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_figure_with_config(fig, 'q4_ml_feature_importance', output_dirs, config)


def generate_all_q4_visualizations(
    data_dir: Path,
    output_dirs: Dict[str, Path],
    config: VisualizationConfig,
    showcase: bool = False,
    mode: str = 'paper'
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

        mode = str(mode).strip().lower()

        pareto_df = None
        try:
            fp = data_dir / 'outputs' / 'tables' / 'showcase' / 'mcm2026c_q4_ml_pareto_frontier.csv'
            if fp.exists():
                pareto_df = pd.read_csv(fp)
        except Exception:
            pareto_df = None

        # Generate visualizations (paper mode = 4 core figures)
        create_q4_mechanism_tradeoff_scatter(metrics_data, pareto_df, output_dirs, config)
        print("✅ Created mechanism trade-off scatter plot")

        create_q4_robustness_curves(metrics_data, output_dirs, config)
        print("✅ Created robustness curves")

        create_q4_champion_uncertainty_analysis(metrics_data, output_dirs, config)
        print("✅ Created champion uncertainty analysis")

        create_q4_mechanism_recommendation(metrics_data, output_dirs, config)
        print("✅ Created mechanism recommendation decision tree")

        if mode != 'paper':
            create_q4_tradeoff_pareto_frontier_2d(metrics_data, output_dirs, config)
            print("✅ Created 2D trade-off Pareto frontier")

            create_q4_seasonal_variation_analysis(metrics_data, output_dirs, config)
            print("✅ Created seasonal variation analysis")

        if showcase and mode != 'paper':
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
    parser.add_argument('--mode', type=str, default='paper', help='paper (4 core figs) or full')
    args = parser.parse_args()

    config = VisualizationConfig.from_ini(args.ini) if args.ini is not None else VisualizationConfig()
    output_structure = create_output_directories(args.data_dir / 'outputs' / 'figures', ['Q4'])

    generate_all_q4_visualizations(
        args.data_dir,
        output_structure['Q4'],
        config,
        showcase=args.showcase,
        mode=str(args.mode),
    )