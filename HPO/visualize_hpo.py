# Trajecto: Real-time 3D Trajectory Reconstruction System (Software)
# Copyright 2025-2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# NOTICE: This software implements the "Hybrid ESKF-Stateful TCN" logic 
# protected under ROK Patent Application No. 10-2025-YYYYYYY.
# Commercial use requires a separate license from the author.

"""
HPO Results Visualization Script for Trajecto

Generates various plots to analyze hyperparameter optimization results:
- Parameter importance (correlation with loss)
- Loss distribution by rung
- Parameter scatter plots
- 2D contour/surface plots
- Parallel coordinates plot
- Best trials summary
"""

import ast
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def parse_params(params_str: str) -> dict:
    """Parse the params string into a dictionary."""
    # Replace np.int64/np.float64 with regular values
    cleaned = re.sub(r'np\.int64\((\d+)\)', r'\1', params_str)
    cleaned = re.sub(r'np\.float64\(([\d.e\-+]+)\)', r'\1', cleaned)
    try:
        return ast.literal_eval(cleaned)
    except:
        return {}


def load_hpo_results(csv_path: str) -> pd.DataFrame:
    """Load and parse HPO results CSV."""
    df = pd.read_csv(csv_path)

    # Parse params column
    params_list = df['params'].apply(parse_params)
    params_df = pd.DataFrame(params_list.tolist())

    # Combine with original df
    result = pd.concat([df[['timestamp', 'loss', 'rung', 'trial_id']], params_df], axis=1)

    # Filter out extreme losses (likely failed runs)
    result = result[result['loss'] < result['loss'].quantile(0.95)]

    return result


def plot_loss_by_rung(df: pd.DataFrame, save_dir: Path):
    """Plot loss distribution by ASHA rung."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Box plot
    ax1 = axes[0]
    df.boxplot(column='loss', by='rung', ax=ax1)
    ax1.set_xlabel('Rung')
    ax1.set_ylabel('Validation Loss')
    ax1.set_title('Loss Distribution by ASHA Rung')
    plt.suptitle('')  # Remove automatic title

    # Violin plot
    ax2 = axes[1]
    rung_order = sorted(df['rung'].unique())
    sns.violinplot(data=df, x='rung', y='loss', order=rung_order, ax=ax2)
    ax2.set_xlabel('Rung')
    ax2.set_ylabel('Validation Loss')
    ax2.set_title('Loss Violin Plot by Rung')

    plt.tight_layout()
    plt.savefig(save_dir / 'loss_by_rung.png', dpi=150)
    plt.close()
    print(f"Saved: loss_by_rung.png")


def plot_parameter_importance(df: pd.DataFrame, save_dir: Path):
    """Plot parameter importance based on correlation with loss."""
    # Select numerical hyperparameters
    param_cols = ['lr', 'dropout', 'reg_weight', 'mahalanobis_threshold',
                  'kernel_size', 'tcn_channel_size', 'num_tcn_layers']
    param_cols = [c for c in param_cols if c in df.columns]

    # Calculate correlation with loss
    correlations = {}
    for col in param_cols:
        if df[col].dtype in ['float64', 'int64']:
            # Use log scale for lr and reg_weight
            if col in ['lr', 'reg_weight']:
                corr = df['loss'].corr(np.log10(df[col].replace(0, 1e-10)))
            else:
                corr = df['loss'].corr(df[col])
            correlations[col] = corr

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['green' if c < 0 else 'red' for c in correlations.values()]
    bars = ax.barh(list(correlations.keys()), list(correlations.values()), color=colors, alpha=0.7)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Correlation with Loss (negative = better)')
    ax.set_title('Hyperparameter Importance (Correlation with Loss)')

    # Add correlation values
    for bar, val in zip(bars, correlations.values()):
        ax.text(val + 0.01 if val >= 0 else val - 0.05, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_dir / 'parameter_importance.png', dpi=150)
    plt.close()
    print(f"Saved: parameter_importance.png")


def plot_lr_vs_loss(df: pd.DataFrame, save_dir: Path):
    """Plot learning rate vs loss with rung coloring."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scatter plot
    ax1 = axes[0]
    scatter = ax1.scatter(df['lr'], df['loss'], c=df['rung'], cmap='viridis',
                          alpha=0.6, s=50, edgecolors='white', linewidth=0.5)
    ax1.set_xscale('log')
    ax1.set_xlabel('Learning Rate (log scale)')
    ax1.set_ylabel('Validation Loss')
    ax1.set_title('Learning Rate vs Loss')
    plt.colorbar(scatter, ax=ax1, label='Rung')

    # Hexbin density plot
    ax2 = axes[1]
    hb = ax2.hexbin(np.log10(df['lr']), df['loss'], gridsize=20, cmap='YlOrRd', mincnt=1)
    ax2.set_xlabel('log10(Learning Rate)')
    ax2.set_ylabel('Validation Loss')
    ax2.set_title('Learning Rate vs Loss (Density)')
    plt.colorbar(hb, ax=ax2, label='Count')

    plt.tight_layout()
    plt.savefig(save_dir / 'lr_vs_loss.png', dpi=150)
    plt.close()
    print(f"Saved: lr_vs_loss.png")


def plot_2d_contours(df: pd.DataFrame, save_dir: Path):
    """Plot 2D contour plots for key parameter pairs."""
    param_pairs = [
        ('lr', 'dropout'),
        ('lr', 'reg_weight'),
        ('dropout', 'num_tcn_layers'),
        ('kernel_size', 'tcn_channel_size'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for idx, (p1, p2) in enumerate(param_pairs):
        if p1 not in df.columns or p2 not in df.columns:
            continue

        ax = axes[idx]

        # Prepare data
        x = np.log10(df[p1]) if p1 in ['lr', 'reg_weight'] else df[p1]
        y = np.log10(df[p2]) if p2 in ['lr', 'reg_weight'] else df[p2]
        z = df['loss']

        # Create scatter with loss coloring
        scatter = ax.scatter(x, y, c=z, cmap='RdYlGn_r', alpha=0.7, s=40,
                            vmin=z.quantile(0.05), vmax=z.quantile(0.95))

        xlabel = f'log10({p1})' if p1 in ['lr', 'reg_weight'] else p1
        ylabel = f'log10({p2})' if p2 in ['lr', 'reg_weight'] else p2
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f'{p1} vs {p2}')
        plt.colorbar(scatter, ax=ax, label='Loss')

    plt.tight_layout()
    plt.savefig(save_dir / 'parameter_contours.png', dpi=150)
    plt.close()
    print(f"Saved: parameter_contours.png")


def plot_categorical_analysis(df: pd.DataFrame, save_dir: Path):
    """Plot analysis for categorical parameters."""
    cat_params = ['kernel_size', 'tcn_channel_size', 'num_tcn_layers']
    cat_params = [c for c in cat_params if c in df.columns]

    if not cat_params:
        return

    fig, axes = plt.subplots(1, len(cat_params), figsize=(5*len(cat_params), 5))
    if len(cat_params) == 1:
        axes = [axes]

    for ax, param in zip(axes, cat_params):
        # Group by parameter and get statistics
        grouped = df.groupby(param)['loss'].agg(['mean', 'std', 'count', 'min'])

        x = range(len(grouped))
        ax.bar(x, grouped['mean'], yerr=grouped['std'], capsize=5, alpha=0.7,
               color='steelblue', edgecolor='black')
        ax.scatter(x, grouped['min'], color='red', s=100, zorder=5,
                   label='Best', marker='*')

        ax.set_xticks(x)
        ax.set_xticklabels(grouped.index)
        ax.set_xlabel(param)
        ax.set_ylabel('Validation Loss')
        ax.set_title(f'Loss by {param}')
        ax.legend()

        # Add count labels
        for i, (_, row) in enumerate(grouped.iterrows()):
            ax.text(i, row['mean'] + row['std'] + 0.1, f'n={int(row["count"])}',
                    ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_dir / 'categorical_params.png', dpi=150)
    plt.close()
    print(f"Saved: categorical_params.png")


def plot_optimization_progress(df: pd.DataFrame, save_dir: Path):
    """Plot optimization progress over time."""
    df_sorted = df.sort_values('timestamp')

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Cumulative best loss
    ax1 = axes[0, 0]
    cumulative_best = df_sorted['loss'].cummin()
    ax1.plot(range(len(cumulative_best)), cumulative_best, 'b-', linewidth=2)
    ax1.scatter(range(len(df_sorted)), df_sorted['loss'], alpha=0.3, s=20, c='gray')
    ax1.set_xlabel('Trial Number')
    ax1.set_ylabel('Validation Loss')
    ax1.set_title('Optimization Progress (Cumulative Best)')
    ax1.legend(['Cumulative Best', 'All Trials'])

    # Loss by trial with rung coloring
    ax2 = axes[0, 1]
    for rung in sorted(df_sorted['rung'].unique()):
        mask = df_sorted['rung'] == rung
        ax2.scatter(np.where(mask)[0], df_sorted.loc[mask, 'loss'],
                    label=f'Rung {rung}', alpha=0.6, s=30)
    ax2.set_xlabel('Trial Number')
    ax2.set_ylabel('Validation Loss')
    ax2.set_title('All Trials by Rung')
    ax2.legend()

    # Rolling average
    ax3 = axes[1, 0]
    window = min(20, len(df_sorted) // 5)
    if window > 1:
        rolling_mean = df_sorted['loss'].rolling(window=window, min_periods=1).mean()
        ax3.plot(range(len(rolling_mean)), rolling_mean, 'r-', linewidth=2, label=f'Rolling Mean (w={window})')
        ax3.fill_between(range(len(rolling_mean)),
                         df_sorted['loss'].rolling(window=window, min_periods=1).min(),
                         df_sorted['loss'].rolling(window=window, min_periods=1).max(),
                         alpha=0.2, color='red')
    ax3.set_xlabel('Trial Number')
    ax3.set_ylabel('Validation Loss')
    ax3.set_title('Rolling Statistics')
    ax3.legend()

    # Trials per rung
    ax4 = axes[1, 1]
    rung_counts = df['rung'].value_counts().sort_index()
    ax4.bar(rung_counts.index, rung_counts.values, color='steelblue', edgecolor='black')
    ax4.set_xlabel('Rung')
    ax4.set_ylabel('Number of Trials')
    ax4.set_title('Trial Distribution by Rung')

    plt.tight_layout()
    plt.savefig(save_dir / 'optimization_progress.png', dpi=150)
    plt.close()
    print(f"Saved: optimization_progress.png")


def plot_parallel_coordinates(df: pd.DataFrame, save_dir: Path):
    """Plot parallel coordinates for top trials."""
    # Select top N trials
    n_top = min(50, len(df))
    top_df = df.nsmallest(n_top, 'loss').copy()

    # Normalize parameters for visualization
    param_cols = ['lr', 'dropout', 'reg_weight', 'mahalanobis_threshold',
                  'kernel_size', 'tcn_channel_size', 'num_tcn_layers']
    param_cols = [c for c in param_cols if c in top_df.columns]

    # Create normalized dataframe
    norm_df = top_df.copy()
    for col in param_cols:
        if col in ['lr', 'reg_weight']:
            vals = np.log10(norm_df[col].replace(0, 1e-10))
        else:
            vals = norm_df[col]
        norm_df[col + '_norm'] = (vals - vals.min()) / (vals.max() - vals.min() + 1e-10)

    norm_cols = [c + '_norm' for c in param_cols]

    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot each trial as a line
    colors = plt.cm.RdYlGn_r(np.linspace(0, 1, n_top))
    for idx, (_, row) in enumerate(top_df.iterrows()):
        values = [norm_df.loc[row.name, c] for c in norm_cols]
        ax.plot(range(len(param_cols)), values, c=colors[idx], alpha=0.5, linewidth=1)

    # Highlight best trial
    best_row = top_df.iloc[0]
    best_values = [norm_df.loc[best_row.name, c] for c in norm_cols]
    ax.plot(range(len(param_cols)), best_values, 'b-', linewidth=3, label=f'Best (loss={best_row["loss"]:.4f})')

    ax.set_xticks(range(len(param_cols)))
    ax.set_xticklabels(param_cols, rotation=45, ha='right')
    ax.set_ylabel('Normalized Value')
    ax.set_title(f'Parallel Coordinates (Top {n_top} Trials)')
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_dir / 'parallel_coordinates.png', dpi=150)
    plt.close()
    print(f"Saved: parallel_coordinates.png")


def print_best_trials(df: pd.DataFrame, n: int = 10):
    """Print the best N trials."""
    print("\n" + "="*80)
    print(f"TOP {n} BEST TRIALS")
    print("="*80)

    top_df = df.nsmallest(n, 'loss')

    for idx, (_, row) in enumerate(top_df.iterrows(), 1):
        print(f"\n#{idx} - Loss: {row['loss']:.6f} (Rung {row['rung']})")
        print(f"  trial_id: {row['trial_id']}")
        print(f"  lr: {row.get('lr', 'N/A'):.6e}")
        print(f"  dropout: {row.get('dropout', 'N/A'):.4f}")
        print(f"  reg_weight: {row.get('reg_weight', 'N/A'):.6e}")
        print(f"  mahalanobis_threshold: {row.get('mahalanobis_threshold', 'N/A'):.2f}")
        print(f"  kernel_size: {row.get('kernel_size', 'N/A')}")
        print(f"  tcn_channel_size: {row.get('tcn_channel_size', 'N/A')}")
        print(f"  num_tcn_layers: {row.get('num_tcn_layers', 'N/A')}")


def create_summary_plot(df: pd.DataFrame, save_dir: Path):
    """Create a summary dashboard."""
    fig = plt.figure(figsize=(16, 12))

    # Best trials table
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.axis('off')
    top5 = df.nsmallest(5, 'loss')[['trial_id', 'loss', 'rung', 'lr', 'dropout']].round(6)
    table = ax1.table(cellText=top5.values, colLabels=top5.columns,
                      loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    ax1.set_title('Top 5 Trials', fontsize=12, fontweight='bold')

    # Loss distribution
    ax2 = fig.add_subplot(2, 3, 2)
    df['loss'].hist(bins=30, ax=ax2, color='steelblue', edgecolor='black', alpha=0.7)
    ax2.axvline(df['loss'].min(), color='red', linestyle='--', label=f'Best: {df["loss"].min():.4f}')
    ax2.set_xlabel('Loss')
    ax2.set_ylabel('Count')
    ax2.set_title('Loss Distribution')
    ax2.legend()

    # LR vs Loss
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.scatter(df['lr'], df['loss'], c=df['rung'], cmap='viridis', alpha=0.5, s=30)
    ax3.set_xscale('log')
    ax3.set_xlabel('Learning Rate')
    ax3.set_ylabel('Loss')
    ax3.set_title('LR vs Loss')

    # Categorical params
    ax4 = fig.add_subplot(2, 3, 4)
    if 'kernel_size' in df.columns:
        df.groupby('kernel_size')['loss'].mean().plot(kind='bar', ax=ax4, color='steelblue')
        ax4.set_xlabel('Kernel Size')
        ax4.set_ylabel('Mean Loss')
        ax4.set_title('Mean Loss by Kernel Size')

    # Trials by rung
    ax5 = fig.add_subplot(2, 3, 5)
    df['rung'].value_counts().sort_index().plot(kind='bar', ax=ax5, color='coral')
    ax5.set_xlabel('Rung')
    ax5.set_ylabel('Count')
    ax5.set_title('Trials per Rung')

    # Optimization progress
    ax6 = fig.add_subplot(2, 3, 6)
    df_sorted = df.sort_values('timestamp')
    ax6.plot(range(len(df_sorted)), df_sorted['loss'].cummin(), 'b-', linewidth=2)
    ax6.set_xlabel('Trial Number')
    ax6.set_ylabel('Best Loss')
    ax6.set_title('Optimization Progress')

    plt.suptitle('HPO Results Summary', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_dir / 'summary_dashboard.png', dpi=150)
    plt.close()
    print(f"Saved: summary_dashboard.png")


def main():
    # Setup
    csv_path = Path(__file__).parent / 'hpo_results.csv'
    save_dir = Path(__file__).parent / 'plots'
    save_dir.mkdir(exist_ok=True)

    print(f"Loading HPO results from: {csv_path}")
    df = load_hpo_results(str(csv_path))
    print(f"Loaded {len(df)} trials")

    # Generate all plots
    print("\nGenerating plots...")
    plot_loss_by_rung(df, save_dir)
    plot_parameter_importance(df, save_dir)
    plot_lr_vs_loss(df, save_dir)
    plot_2d_contours(df, save_dir)
    plot_categorical_analysis(df, save_dir)
    plot_optimization_progress(df, save_dir)
    plot_parallel_coordinates(df, save_dir)
    create_summary_plot(df, save_dir)

    # Print best trials
    print_best_trials(df, n=10)

    print(f"\n{'='*80}")
    print(f"All plots saved to: {save_dir}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
