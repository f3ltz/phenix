import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def generate_results_chart():
    # Set publication style
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2), dpi=200)

    models = ['Random Forest', 'XGBoost', 'Ridge Reg.', 'PhyloGNN']
    colors = ['#10b981', '#0ea5e9', '#f59e0b', '#8b5cf6']

    # 1. In-Domain Random CV R2 (Augmented features)
    r2_means = [0.8554, 0.8597, 0.7893, -0.7726]
    r2_stds = [0.0143, 0.0194, 0.0295, 0.7381]

    # Bar chart for in-domain accuracy
    y_pos = np.arange(len(models))
    bars1 = ax1.bar(models, [max(0, x) for x in r2_means], 
                    yerr=[r2_stds[i] if r2_means[i] > 0 else 0 for i in range(4)], 
                    color=colors, width=0.52, edgecolor='#1e293b', linewidth=1.2, capsize=4)
    ax1.set_ylim(0, 1.0)
    ax1.set_ylabel('In-Domain Random CV $R^2$', fontsize=10.5, fontweight='bold', color='#1e293b')
    ax1.set_title('(a) In-Domain Imputation Accuracy ($R^2$)', fontsize=11.5, fontweight='bold', pad=10, color='#0f172a')
    ax1.axhline(0.85, color='#dc2626', linestyle='--', linewidth=1.1, alpha=0.75, label=r'High Precision ($R^2 \geq 0.85$)')

    for bar, mean in zip(bars1, r2_means):
        if mean > 0:
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.03, f'{mean:.3f}',
                     ha='center', va='bottom', fontsize=9.2, fontweight='bold', color='#0f172a')
        else:
            ax1.text(bar.get_x() + bar.get_width()/2., 0.05, f'Fail ({mean:.2f})',
                     ha='center', va='bottom', fontsize=8.8, fontweight='bold', color='#dc2626')

    ax1.legend(loc='upper right', fontsize=8.5, framealpha=0.9)
    ax1.tick_params(axis='x', rotation=12, labelsize=9.5)
    ax1.grid(True, linestyle='--', alpha=0.5, color='#cbd5e1')

    # 2. Out-of-Clade Extrapolation RMSE (140 Ma Buffer)
    extrap_rmse = [1.4254, 1.3137, 4.0708, 1.6606]
    bars2 = ax2.bar(models, extrap_rmse, color=colors, width=0.52, edgecolor='#1e293b', linewidth=1.2)
    ax2.set_ylim(0, 4.8)
    ax2.set_ylabel('Out-of-Clade Extrapolation RMSE', fontsize=10.5, fontweight='bold', color='#1e293b')
    ax2.set_title('(b) Out-of-Clade Extrapolation Error (140 Ma Buffer)', fontsize=11.5, fontweight='bold', pad=10, color='#0f172a')

    for bar, rmse, m in zip(bars2, extrap_rmse, models):
        badge = ' (Best)' if m == 'XGBoost' else ''
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1, f'{rmse:.3f}{badge}',
                 ha='center', va='bottom', fontsize=9.0, fontweight='bold', 
                 color='#0369a1' if m == 'XGBoost' else '#0f172a')

    ax2.tick_params(axis='x', rotation=12, labelsize=9.5)
    ax2.grid(True, linestyle='--', alpha=0.5, color='#cbd5e1')

    plt.tight_layout(pad=1.2)
    out_path = Path('report_assets/fig_results_chart.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote results chart -> {out_path}')

if __name__ == '__main__':
    generate_results_chart()
