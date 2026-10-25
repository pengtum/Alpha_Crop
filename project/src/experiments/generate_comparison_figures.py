"""
生成 with-coords vs without-coords 对比图表。
字体: Arial (TrueType 42)，Illustrator 可编辑。

Usage:
    python -m src.experiments.generate_comparison_figures
"""
import os
import sys
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap

# ─── Illustrator-compatible settings ─────────────────────────────────
plt.rcParams.update({
    'pdf.fonttype': 42,       # TrueType — editable in Illustrator
    'ps.fonttype': 42,
    'font.family': 'Arial',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

# 配色方案
C_AE = '#2196F3'       # 蓝色
C_S2 = '#FF9800'       # 橙色
C_FUSION = '#4CAF50'   # 绿色
C_DELTA_POS = '#2E7D32' # 正向delta (无坐标更好) 绿
C_DELTA_NEG = '#C62828'  # 负向delta (有坐标更好) 红

RESULTS_DIR = os.path.join('results', 'experiments')
OUT_DIR = os.path.join(RESULTS_DIR, 'comparison')


def load_csv(exp, filename):
    return pd.read_csv(os.path.join(RESULTS_DIR, exp, filename))


def load_json(exp, filename='tier_b.json'):
    with open(os.path.join(RESULTS_DIR, exp, filename)) as f:
        return json.load(f)


def savefig(fig, name):
    """保存到 results/experiments/comparison/"""
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path)
    # 同时保存 PNG 用于 HTML
    png_path = path.replace('.pdf', '.png')
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# =====================================================================
# Figure 1: Exp 1 ID Baseline — With vs Without Aux Features
# =====================================================================
def plot_fig1():
    print("\n=== Fig 1: Exp 1 ID Baseline Comparison ===")
    df_c = load_csv('exp1_id_baseline', 'summary_tier_b.csv')
    df_n = load_csv('exp1_id_baseline', 'summary_tier_b_nocoords.csv')

    metrics = ['BA_mean', 'F1_mean', 'Kappa_mean']
    metric_labels = ['Balanced\nAccuracy', 'Macro F1', "Cohen's\nKappa"]

    fts = ['ae', 's2', 'fusion']
    ft_labels = ['AE', 'S2', 'Fusion']
    ft_colors = [C_AE, C_S2, C_FUSION]

    n_metrics = len(metrics)
    n_bars = len(fts) * 2  # coords + nocoords 各一组
    x = np.arange(n_metrics)
    w = 0.12  # 每个 bar 的宽度
    total_w = n_bars * w
    offsets = np.arange(n_bars) * w - total_w / 2 + w / 2

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, (ft, color, ft_label) in enumerate(zip(fts, ft_colors, ft_labels)):
        row_c = df_c[df_c.feature_type == ft].iloc[0]
        row_n = df_n[df_n.feature_type == ft].iloc[0]

        vals_c = [row_c[m] for m in metrics]
        vals_n = [row_n[m] for m in metrics]
        errs_c = [row_c[m.replace('mean', 'std')] for m in metrics]
        errs_n = [row_n[m.replace('mean', 'std')] for m in metrics]

        # coords = solid
        ax.bar(x + offsets[i * 2], vals_c, w, yerr=errs_c,
               label=f'{ft_label} +coords', color=color,
               edgecolor='white', linewidth=0.5, capsize=2, alpha=0.9)
        # nocoords = hatched
        ax.bar(x + offsets[i * 2 + 1], vals_n, w, yerr=errs_n,
               label=f'{ft_label} -coords', color=color,
               edgecolor='gray', linewidth=0.5, capsize=2, alpha=0.5,
               hatch='//')

        # Delta 标注
        for j in range(n_metrics):
            delta = vals_n[j] - vals_c[j]
            y_pos = max(vals_c[j], vals_n[j]) + max(errs_c[j], errs_n[j]) + 0.005
            color_d = C_DELTA_POS if delta >= 0 else C_DELTA_NEG
            mid_x = (x[j] + offsets[i * 2] + x[j] + offsets[i * 2 + 1]) / 2
            ax.text(mid_x, y_pos, f'{delta:+.3f}', ha='center', va='bottom',
                    fontsize=7, color=color_d, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel('Score')
    ax.set_ylim(0.55, 0.82)
    ax.set_title('Exp 1: ID Baseline — With vs Without Auxiliary Features')
    ax.legend(frameon=True, fancybox=False, edgecolor='gray',
              ncol=3, fontsize=8, loc='upper left')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    fig.tight_layout()
    savefig(fig, 'fig1_exp1_id_comparison.pdf')


# =====================================================================
# Figure 2: Exp 0 Per-Country BA Delta Heatmap
# =====================================================================
def plot_fig2():
    print("\n=== Fig 2: Exp 0 Per-Country BA Delta Heatmap ===")
    df_c = load_csv('exp0_per_country_cv', 'summary_tier_b.csv')
    df_n = load_csv('exp0_per_country_cv', 'summary_tier_b_nocoords.csv')

    fts = ['ae', 's2', 'fusion']
    ft_labels = ['AE', 'S2', 'Fusion']
    countries = sorted(df_c['country'].unique())

    # 构建 delta 矩阵: country × feature_type
    delta_matrix = np.full((len(countries), len(fts)), np.nan)
    for j, ft in enumerate(fts):
        for i, c in enumerate(countries):
            row_c = df_c[(df_c.country == c) & (df_c.feature_type == ft)]
            row_n = df_n[(df_n.country == c) & (df_n.feature_type == ft)]
            if len(row_c) > 0 and len(row_n) > 0:
                delta_matrix[i, j] = row_n.iloc[0]['BA_mean'] - row_c.iloc[0]['BA_mean']

    # 按平均 delta 排序国家
    mean_delta = np.nanmean(delta_matrix, axis=1)
    sort_idx = np.argsort(mean_delta)
    delta_matrix = delta_matrix[sort_idx]
    countries_sorted = [countries[i] for i in sort_idx]

    fig, ax = plt.subplots(figsize=(6, 8))
    cmap_div = LinearSegmentedColormap.from_list('div',
        [C_DELTA_NEG, '#FFCDD2', '#FFFFFF', '#C8E6C9', C_DELTA_POS], N=256)
    vmax = max(abs(np.nanmin(delta_matrix)), abs(np.nanmax(delta_matrix)), 0.01)
    im = ax.imshow(delta_matrix, cmap=cmap_div, aspect='auto',
                   vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(ft_labels)))
    ax.set_xticklabels(ft_labels, fontsize=10)
    ax.set_yticks(range(len(countries_sorted)))
    ax.set_yticklabels(countries_sorted, fontsize=9)
    ax.set_xlabel('Feature Set')
    ax.set_ylabel('Country')
    ax.set_title('Exp 0: Per-Country BA Change After Removing Auxiliary Features')

    for i in range(len(countries_sorted)):
        for j in range(len(fts)):
            val = delta_matrix[i, j]
            if not np.isnan(val):
                txt_color = 'white' if abs(val) > vmax * 0.6 else 'black'
                ax.text(j, i, f'{val:+.3f}', ha='center', va='center',
                        fontsize=8, color=txt_color, fontweight='bold')

    plt.colorbar(im, ax=ax, shrink=0.7, label=r'$\Delta$ BA (no-coords $-$ coords)')
    fig.tight_layout()
    savefig(fig, 'fig2_exp0_country_ba_delta.pdf')


# =====================================================================
# Figure 3: Exp 0 Mean Per-Country BA
# =====================================================================
def plot_fig3():
    print("\n=== Fig 3: Exp 0 Mean Per-Country BA Comparison ===")
    df_c = load_csv('exp0_per_country_cv', 'summary_tier_b.csv')
    df_n = load_csv('exp0_per_country_cv', 'summary_tier_b_nocoords.csv')

    fts = ['ae', 's2', 'fusion']
    ft_labels = ['AlphaEarth', 'Sentinel-2', 'Fusion']
    ft_colors = [C_AE, C_S2, C_FUSION]

    # 计算每个 feature type 的平均 BA
    mean_c = [df_c[df_c.feature_type == ft]['BA_mean'].mean() for ft in fts]
    std_c = [df_c[df_c.feature_type == ft]['BA_mean'].std() for ft in fts]
    mean_n = [df_n[df_n.feature_type == ft]['BA_mean'].mean() for ft in fts]
    std_n = [df_n[df_n.feature_type == ft]['BA_mean'].std() for ft in fts]

    x = np.arange(len(fts))
    w = 0.3

    fig, ax = plt.subplots(figsize=(6, 4.5))

    bars_c = ax.bar(x - w / 2, mean_c, w, yerr=std_c,
                    label='+coords', color=ft_colors,
                    edgecolor='white', linewidth=0.5, capsize=4, alpha=0.9)
    bars_n = ax.bar(x + w / 2, mean_n, w, yerr=std_n,
                    label='-coords', color=ft_colors,
                    edgecolor='gray', linewidth=0.5, capsize=4, alpha=0.5,
                    hatch='//')

    # Delta 标注
    for i in range(len(fts)):
        delta = mean_n[i] - mean_c[i]
        y_pos = max(mean_c[i], mean_n[i]) + max(std_c[i], std_n[i]) + 0.005
        color_d = C_DELTA_POS if delta >= 0 else C_DELTA_NEG
        ax.text(i, y_pos, f'{delta:+.3f}', ha='center', va='bottom',
                fontsize=9, color=color_d, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(ft_labels)
    ax.set_ylabel('Mean Balanced Accuracy')
    ax.set_title('Exp 0: Mean Per-Country BA — With vs Without Auxiliary Features')
    ax.legend(frameon=True, fancybox=False, edgecolor='gray')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    fig.tight_layout()
    savefig(fig, 'fig3_exp0_country_comparison.pdf')


# =====================================================================
# Figure 4: Exp 2 LOCO OOD Mean BA Comparison
# =====================================================================
def plot_fig4():
    print("\n=== Fig 4: Exp 2 LOCO OOD Mean BA Comparison ===")
    d_c = load_json('exp2_loco', 'tier_b.json')
    d_n = load_json('exp2_loco', 'tier_b_nocoords.json')

    fts = ['ae', 's2', 'fusion']
    ft_labels = ['AlphaEarth', 'Sentinel-2', 'Fusion']
    ft_colors = [C_AE, C_S2, C_FUSION]

    # 计算 mean BA 和 std
    mean_c, std_c, mean_n, std_n = [], [], [], []
    for ft in fts:
        countries = list(d_c[ft]['per_country'].keys())
        ba_c = [d_c[ft]['per_country'][c]['balanced_accuracy'] for c in countries]
        ba_n = [d_n[ft]['per_country'][c]['balanced_accuracy'] for c in countries]
        mean_c.append(np.mean(ba_c))
        std_c.append(np.std(ba_c))
        mean_n.append(np.mean(ba_n))
        std_n.append(np.std(ba_n))

    x = np.arange(len(fts))
    w = 0.3

    fig, ax = plt.subplots(figsize=(6, 4.5))

    ax.bar(x - w / 2, mean_c, w, yerr=std_c,
           label='+coords', color=ft_colors,
           edgecolor='white', linewidth=0.5, capsize=4, alpha=0.9)
    ax.bar(x + w / 2, mean_n, w, yerr=std_n,
           label='-coords', color=ft_colors,
           edgecolor='gray', linewidth=0.5, capsize=4, alpha=0.5,
           hatch='//')

    # Delta 标注
    for i in range(len(fts)):
        delta = mean_n[i] - mean_c[i]
        y_pos = max(mean_c[i], mean_n[i]) + max(std_c[i], std_n[i]) + 0.005
        color_d = C_DELTA_POS if delta >= 0 else C_DELTA_NEG
        ax.text(i, y_pos, f'{delta:+.3f}', ha='center', va='bottom',
                fontsize=9, color=color_d, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(ft_labels)
    ax.set_ylabel('Mean Balanced Accuracy')
    ax.set_ylim(0, 0.7)
    ax.set_title('Exp 2: LOCO OOD Mean BA — With vs Without Auxiliary Features')
    ax.legend(frameon=True, fancybox=False, edgecolor='gray')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    fig.tight_layout()
    savefig(fig, 'fig4_exp2_loco_comparison.pdf')


# =====================================================================
# Figure 5: Exp 2 LOCO BA Delta Per Country (Fusion)
# =====================================================================
def plot_fig5():
    print("\n=== Fig 5: Exp 2 LOCO BA Delta Per Country (Fusion) ===")
    d_c = load_json('exp2_loco', 'tier_b.json')
    d_n = load_json('exp2_loco', 'tier_b_nocoords.json')

    ft = 'fusion'
    countries = sorted(d_c[ft]['per_country'].keys())
    deltas = []
    for c in countries:
        ba_c = d_c[ft]['per_country'][c]['balanced_accuracy']
        ba_n = d_n[ft]['per_country'][c]['balanced_accuracy']
        deltas.append(ba_n - ba_c)

    # 按 delta 排序
    sorted_pairs = sorted(zip(countries, deltas), key=lambda x: x[1])
    countries_sorted = [p[0] for p in sorted_pairs]
    deltas_sorted = [p[1] for p in sorted_pairs]

    fig, ax = plt.subplots(figsize=(7, 6))
    colors = [C_DELTA_POS if d >= 0 else C_DELTA_NEG for d in deltas_sorted]
    y = np.arange(len(countries_sorted))
    ax.barh(y, deltas_sorted, color=colors, edgecolor='white', linewidth=0.5,
            height=0.7)

    # 标注数值
    for i, (c, d) in enumerate(zip(countries_sorted, deltas_sorted)):
        x_pos = d + 0.002 if d >= 0 else d - 0.002
        ha = 'left' if d >= 0 else 'right'
        ax.text(x_pos, i, f'{d:+.3f}', ha=ha, va='center', fontsize=8,
                fontweight='bold')

    ax.set_yticks(y)
    ax.set_yticklabels(countries_sorted, fontsize=9)
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.set_xlabel(r'$\Delta$ BA (no-coords $-$ coords)')
    ax.set_title('Exp 2: LOCO OOD BA Change (Fusion) — Per Country')
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    fig.tight_layout()
    savefig(fig, 'fig5_exp2_loco_delta_per_country.pdf')


# =====================================================================
# Figure 6: ID vs OOD Gap — Impact of Auxiliary Features
# =====================================================================
def plot_fig6():
    print("\n=== Fig 6: ID vs OOD Gap ===")
    # ID data from exp1
    id_c = load_csv('exp1_id_baseline', 'summary_tier_b.csv')
    id_n = load_csv('exp1_id_baseline', 'summary_tier_b_nocoords.csv')
    # OOD data from exp2
    ood_c = load_json('exp2_loco', 'tier_b.json')
    ood_n = load_json('exp2_loco', 'tier_b_nocoords.json')

    fts = ['ae', 's2', 'fusion']
    ft_labels = ['AlphaEarth', 'Sentinel-2', 'Fusion']
    ft_colors = [C_AE, C_S2, C_FUSION]

    x = np.arange(len(fts))
    w = 0.18

    fig, ax = plt.subplots(figsize=(8, 5))

    id_c_vals = [id_c[id_c.feature_type == ft]['BA_mean'].values[0] for ft in fts]
    ood_c_vals = [ood_c[ft]['mean_ba'] for ft in fts]
    id_n_vals = [id_n[id_n.feature_type == ft]['BA_mean'].values[0] for ft in fts]
    ood_n_vals = [ood_n[ft]['mean_ba'] for ft in fts]

    # 4 组 bar: ID-coords, OOD-coords, ID-nocoords, OOD-nocoords
    for i, (ft, color) in enumerate(zip(fts, ft_colors)):
        ax.bar(i - 1.5 * w, id_c_vals[i], w, color=color, alpha=0.9,
               edgecolor='white', linewidth=0.5,
               label='ID +coords' if i == 0 else '')
        ax.bar(i - 0.5 * w, ood_c_vals[i], w, color=color, alpha=0.5,
               edgecolor='gray', linewidth=0.5,
               label='OOD +coords' if i == 0 else '')
        ax.bar(i + 0.5 * w, id_n_vals[i], w, color=color, alpha=0.9,
               edgecolor='white', linewidth=0.5, hatch='//',
               label='ID -coords' if i == 0 else '')
        ax.bar(i + 1.5 * w, ood_n_vals[i], w, color=color, alpha=0.5,
               edgecolor='gray', linewidth=0.5, hatch='//',
               label='OOD -coords' if i == 0 else '')

        # 标注 gap (coords)
        gap_c = id_c_vals[i] - ood_c_vals[i]
        mid_y_c = (id_c_vals[i] + ood_c_vals[i]) / 2
        ax.annotate(f'gap={gap_c:.2f}', xy=(i - 0.5 * w, ood_c_vals[i]),
                    xytext=(i - 2.5 * w, mid_y_c - 0.02),
                    fontsize=7, ha='center', color='red',
                    arrowprops=dict(arrowstyle='->', color='red', lw=0.6))

        # 标注 gap (nocoords)
        gap_n = id_n_vals[i] - ood_n_vals[i]
        mid_y_n = (id_n_vals[i] + ood_n_vals[i]) / 2
        ax.annotate(f'gap={gap_n:.2f}', xy=(i + 1.5 * w, ood_n_vals[i]),
                    xytext=(i + 2.5 * w, mid_y_n - 0.02),
                    fontsize=7, ha='center', color='blue',
                    arrowprops=dict(arrowstyle='->', color='blue', lw=0.6))

    ax.set_xticks(x)
    ax.set_xticklabels(ft_labels)
    ax.set_ylabel('Balanced Accuracy')
    ax.set_ylim(0, 0.85)
    ax.set_title('ID vs OOD Gap — Impact of Auxiliary Features')
    ax.legend(frameon=True, fancybox=False, edgecolor='gray',
              fontsize=8, ncol=2, loc='upper left')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    fig.tight_layout()
    savefig(fig, 'fig6_exp2_id_ood_gap.pdf')


# =====================================================================
# Figure 7: Exp 3 LOZO Comparison
# =====================================================================
def plot_fig7():
    print("\n=== Fig 7: Exp 3 LOZO Comparison ===")
    d_c = load_json('exp3_climate_zone', 'tier_b.json')
    d_n = load_json('exp3_climate_zone', 'tier_b_nocoords.json')

    zones = sorted(d_c['ae']['zone_results'].keys())
    zone_labels = [z.replace('_', '\n') for z in zones]

    fts = ['ae', 's2']
    ft_labels = ['AE', 'S2']
    ft_colors = [C_AE, C_S2]

    n_zones = len(zones)
    x = np.arange(n_zones)
    w = 0.18
    n_bars = len(fts) * 2
    total_w = n_bars * w
    offsets = np.arange(n_bars) * w - total_w / 2 + w / 2

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, (ft, color, ft_label) in enumerate(zip(fts, ft_colors, ft_labels)):
        vals_c = [d_c[ft]['zone_results'][z]['balanced_accuracy'] for z in zones]
        vals_n = [d_n[ft]['zone_results'][z]['balanced_accuracy'] for z in zones]

        ax.bar(x + offsets[i * 2], vals_c, w,
               label=f'{ft_label} +coords', color=color,
               edgecolor='white', linewidth=0.5, alpha=0.9)
        ax.bar(x + offsets[i * 2 + 1], vals_n, w,
               label=f'{ft_label} -coords', color=color,
               edgecolor='gray', linewidth=0.5, alpha=0.5, hatch='//')

    ax.set_xticks(x)
    ax.set_xticklabels(zone_labels, fontsize=9)
    ax.set_ylabel('Balanced Accuracy')
    ax.set_ylim(0, 0.75)
    ax.set_title('Exp 3: LOZO BA — With vs Without Auxiliary Features')
    ax.legend(frameon=True, fancybox=False, edgecolor='gray',
              ncol=2, fontsize=8)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    fig.tight_layout()
    savefig(fig, 'fig7_exp3_lozo_comparison.pdf')


# =====================================================================
# Figure 8: Exp 3 Mean Cross-Zone Transfer BA
# =====================================================================
def plot_fig8():
    print("\n=== Fig 8: Exp 3 Mean Cross-Zone Transfer BA ===")
    d_c = load_json('exp3_climate_zone', 'tier_b.json')
    d_n = load_json('exp3_climate_zone', 'tier_b_nocoords.json')

    fts = ['ae', 's2', 'fusion']
    ft_labels = ['AlphaEarth', 'Sentinel-2', 'Fusion']
    ft_colors = [C_AE, C_S2, C_FUSION]

    mean_c, mean_n = [], []
    std_c, std_n = [], []

    for ft in fts:
        # 从 transfer_matrix 提取所有 BA
        tm_c = d_c[ft]['transfer_matrix']
        tm_n = d_n[ft]['transfer_matrix']
        ba_c_list, ba_n_list = [], []
        for src in tm_c:
            for tgt in tm_c[src]:
                ba_c_list.append(tm_c[src][tgt]['balanced_accuracy'])
        for src in tm_n:
            for tgt in tm_n[src]:
                ba_n_list.append(tm_n[src][tgt]['balanced_accuracy'])
        mean_c.append(np.mean(ba_c_list))
        std_c.append(np.std(ba_c_list))
        mean_n.append(np.mean(ba_n_list))
        std_n.append(np.std(ba_n_list))

    x = np.arange(len(fts))
    w = 0.3

    fig, ax = plt.subplots(figsize=(6, 4.5))

    ax.bar(x - w / 2, mean_c, w, yerr=std_c,
           label='+coords', color=ft_colors,
           edgecolor='white', linewidth=0.5, capsize=4, alpha=0.9)
    ax.bar(x + w / 2, mean_n, w, yerr=std_n,
           label='-coords', color=ft_colors,
           edgecolor='gray', linewidth=0.5, capsize=4, alpha=0.5,
           hatch='//')

    # Delta 标注
    for i in range(len(fts)):
        delta = mean_n[i] - mean_c[i]
        y_pos = max(mean_c[i], mean_n[i]) + max(std_c[i], std_n[i]) + 0.005
        color_d = C_DELTA_POS if delta >= 0 else C_DELTA_NEG
        ax.text(i, y_pos, f'{delta:+.3f}', ha='center', va='bottom',
                fontsize=9, color=color_d, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(ft_labels)
    ax.set_ylabel('Mean Balanced Accuracy')
    ax.set_title('Exp 3: Mean Cross-Zone Transfer BA')
    ax.legend(frameon=True, fancybox=False, edgecolor='gray')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    fig.tight_layout()
    savefig(fig, 'fig8_exp3_transfer_comparison.pdf')


# =====================================================================
# Figure 9: Exp 4 Learning Curve Comparison
# =====================================================================
def plot_fig9():
    print("\n=== Fig 9: Exp 4 Learning Curve Comparison ===")
    d_c = load_json('exp4_label_efficiency', 'tier_b.json')
    d_n = load_json('exp4_label_efficiency', 'tier_b_nocoords.json')

    fig, ax = plt.subplots(figsize=(7, 4.5))

    curve_configs = [
        ('ae_curve', C_AE, 'AE +coords', '-', 'o'),
        ('s2_curve', C_S2, 'S2 +coords', '-', 's'),
    ]

    for key, color, label, ls, marker in curve_configs:
        ss = d_c[key]['sample_sizes']
        # 排除 '0' (全数据), 按 n_train_actual 排序
        sizes = sorted([k for k in ss if k != '0'], key=lambda k: ss[k]['n_train_actual'])
        x_vals = [ss[k]['n_train_actual'] for k in sizes]
        y_vals = [ss[k]['ba_mean'] for k in sizes]
        y_errs = [ss[k]['ba_std'] for k in sizes]
        ax.errorbar(x_vals, y_vals, yerr=y_errs, marker=marker,
                    label=label, color=color, linewidth=2, linestyle=ls,
                    capsize=3, markersize=5)

    # nocoords
    curve_configs_n = [
        ('ae_curve', C_AE, 'AE -coords', '--', 'o'),
        ('s2_curve', C_S2, 'S2 -coords', '--', 's'),
    ]

    for key, color, label, ls, marker in curve_configs_n:
        ss = d_n[key]['sample_sizes']
        sizes = sorted([k for k in ss if k != '0'], key=lambda k: ss[k]['n_train_actual'])
        x_vals = [ss[k]['n_train_actual'] for k in sizes]
        y_vals = [ss[k]['ba_mean'] for k in sizes]
        y_errs = [ss[k]['ba_std'] for k in sizes]
        ax.errorbar(x_vals, y_vals, yerr=y_errs, marker=marker,
                    label=label, color=color, linewidth=2, linestyle=ls,
                    capsize=3, markersize=5, markerfacecolor='white',
                    markeredgecolor=color, markeredgewidth=1.5)

    ax.set_xscale('log')
    ax.set_xlabel('Number of Training Samples')
    ax.set_ylabel('Balanced Accuracy (OOD)')
    ax.set_title('Exp 4: OOD Learning Curve — With vs Without Auxiliary Features')
    ax.legend(frameon=True, fancybox=False, edgecolor='gray', fontsize=8)
    ax.grid(alpha=0.3, linestyle='--')
    fig.tight_layout()
    savefig(fig, 'fig9_exp4_learning_curve_comparison.pdf')


# =====================================================================
# Figure 10: Exp 5 CKA Comparison (KEY FIGURE)
# =====================================================================
def plot_fig10():
    print("\n=== Fig 10: Exp 5 CKA Comparison ===")
    d_c = load_json('exp5_feature_analysis', 'tier_b.json')
    d_n = load_json('exp5_feature_analysis', 'tier_b_nocoords.json')

    groups = ['AE-crop', 'AE-country', 'S2-crop', 'S2-country']
    vals_c = [
        d_c['ae']['cka']['cka_crop'],
        d_c['ae']['cka']['cka_country'],
        d_c['s2']['cka']['cka_crop'],
        d_c['s2']['cka']['cka_country'],
    ]
    vals_n = [
        d_n['ae']['cka']['cka_crop'],
        d_n['ae']['cka']['cka_country'],
        d_n['s2']['cka']['cka_crop'],
        d_n['s2']['cka']['cka_country'],
    ]

    x = np.arange(len(groups))
    w = 0.3

    fig, ax = plt.subplots(figsize=(8, 5))

    bars_c = ax.bar(x - w / 2, vals_c, w, label='+coords',
                    color=[C_AE, C_AE, C_S2, C_S2],
                    edgecolor='white', linewidth=0.5, alpha=0.9)
    bars_n = ax.bar(x + w / 2, vals_n, w, label='-coords',
                    color=[C_AE, C_AE, C_S2, C_S2],
                    edgecolor='gray', linewidth=0.5, alpha=0.5, hatch='//')

    # 标注数值
    for i in range(len(groups)):
        ax.text(x[i] - w / 2, vals_c[i] + 0.005, f'{vals_c[i]:.3f}',
                ha='center', va='bottom', fontsize=8)
        ax.text(x[i] + w / 2, vals_n[i] + 0.005, f'{vals_n[i]:.3f}',
                ha='center', va='bottom', fontsize=8)

    # 关键标注: AE-crop 从 0.024 -> 0.207 (8.7x)
    ratio = vals_n[0] / vals_c[0] if vals_c[0] > 0 else float('inf')
    ax.annotate(f'{ratio:.1f}x',
                xy=(x[0] + w / 2, vals_n[0]),
                xytext=(x[0] + 0.8, vals_n[0] + 0.05),
                fontsize=12, fontweight='bold', color=C_DELTA_POS,
                arrowprops=dict(arrowstyle='->', color=C_DELTA_POS, lw=2))

    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=10)
    ax.set_ylabel('Linear CKA')
    ax.set_title('Exp 5: CKA Feature-Label Alignment — Impact of Auxiliary Features')
    ax.legend(frameon=True, fancybox=False, edgecolor='gray')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    fig.tight_layout()
    savefig(fig, 'fig10_exp5_cka_comparison.pdf')


# =====================================================================
# Figure 11: Exp 5 Country Predictability
# =====================================================================
def plot_fig11():
    print("\n=== Fig 11: Exp 5 Country Predictability ===")
    d_c = load_json('exp5_feature_analysis', 'tier_b.json')
    d_n = load_json('exp5_feature_analysis', 'tier_b_nocoords.json')

    groups = ['AE', 'S2']
    vals_c = [
        d_c['ae']['country_prediction']['balanced_accuracy'],
        d_c['s2']['country_prediction']['balanced_accuracy'],
    ]
    vals_n = [
        d_n['ae']['country_prediction']['balanced_accuracy'],
        d_n['s2']['country_prediction']['balanced_accuracy'],
    ]

    x = np.arange(len(groups))
    w = 0.3

    fig, ax = plt.subplots(figsize=(5, 4.5))

    ax.bar(x - w / 2, vals_c, w, label='+coords',
           color=[C_AE, C_S2], edgecolor='white', linewidth=0.5, alpha=0.9)
    ax.bar(x + w / 2, vals_n, w, label='-coords',
           color=[C_AE, C_S2], edgecolor='gray', linewidth=0.5, alpha=0.5,
           hatch='//')

    # 标注数值
    for i in range(len(groups)):
        ax.text(x[i] - w / 2, vals_c[i] + 0.005, f'{vals_c[i]:.3f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.text(x[i] + w / 2, vals_n[i] + 0.005, f'{vals_n[i]:.3f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Delta 标注
    for i in range(len(groups)):
        delta = vals_n[i] - vals_c[i]
        y_pos = max(vals_c[i], vals_n[i]) + 0.03
        ax.text(i, y_pos, f'{delta:+.3f}', ha='center', va='bottom',
                fontsize=10, color=C_DELTA_NEG, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=11)
    ax.set_ylabel('Balanced Accuracy')
    ax.set_ylim(0.5, 1.08)
    ax.set_title('Exp 5: Country Predictability — With vs Without Auxiliary Features')
    ax.legend(frameon=True, fancybox=False, edgecolor='gray')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    fig.tight_layout()
    savefig(fig, 'fig11_exp5_country_pred.pdf')


# =====================================================================
# Figure 12: Summary — All Experiments Key Metric
# =====================================================================
def plot_fig12():
    print("\n=== Fig 12: Summary All Experiments ===")

    # 收集各实验的 Fusion (或最佳) BA for coords vs nocoords
    # Exp 0: Mean Per-Country CV BA (Fusion)
    df0_c = load_csv('exp0_per_country_cv', 'summary_tier_b.csv')
    df0_n = load_csv('exp0_per_country_cv', 'summary_tier_b_nocoords.csv')
    exp0_c = df0_c[df0_c.feature_type == 'fusion']['BA_mean'].mean()
    exp0_n = df0_n[df0_n.feature_type == 'fusion']['BA_mean'].mean()

    # Exp 1: ID Baseline BA (Fusion)
    df1_c = load_csv('exp1_id_baseline', 'summary_tier_b.csv')
    df1_n = load_csv('exp1_id_baseline', 'summary_tier_b_nocoords.csv')
    exp1_c = df1_c[df1_c.feature_type == 'fusion']['BA_mean'].values[0]
    exp1_n = df1_n[df1_n.feature_type == 'fusion']['BA_mean'].values[0]

    # Exp 2: LOCO Mean BA (Fusion)
    d2_c = load_json('exp2_loco', 'tier_b.json')
    d2_n = load_json('exp2_loco', 'tier_b_nocoords.json')
    exp2_c = d2_c['fusion']['mean_ba']
    exp2_n = d2_n['fusion']['mean_ba']

    # Exp 3: LOZO Mean BA (Fusion)
    d3_c = load_json('exp3_climate_zone', 'tier_b.json')
    d3_n = load_json('exp3_climate_zone', 'tier_b_nocoords.json')
    zones = list(d3_c['fusion']['zone_results'].keys())
    exp3_c = np.mean([d3_c['fusion']['zone_results'][z]['balanced_accuracy'] for z in zones])
    exp3_n = np.mean([d3_n['fusion']['zone_results'][z]['balanced_accuracy'] for z in zones])

    # Exp 3 Transfer: Mean cross-zone transfer BA (Fusion)
    tm_c = d3_c['fusion']['transfer_matrix']
    tm_n = d3_n['fusion']['transfer_matrix']
    exp3t_c = np.mean([tm_c[s][t]['balanced_accuracy'] for s in tm_c for t in tm_c[s]])
    exp3t_n = np.mean([tm_n[s][t]['balanced_accuracy'] for s in tm_n for t in tm_n[s]])

    # Exp 5: CKA crop/country ratio (AE) — 用比值而非 BA
    d5_c = load_json('exp5_feature_analysis', 'tier_b.json')
    d5_n = load_json('exp5_feature_analysis', 'tier_b_nocoords.json')
    exp5_c = d5_c['ae']['cka']['ratio']
    exp5_n = d5_n['ae']['cka']['ratio']

    experiments = [
        'Exp 0: CV (Fusion)',
        'Exp 1: ID (Fusion)',
        'Exp 2: LOCO (Fusion)',
        'Exp 3: LOZO (Fusion)',
        'Exp 3: Transfer (Fusion)',
        'Exp 5: CKA ratio (AE)',
    ]
    c_vals = [exp0_c, exp1_c, exp2_c, exp3_c, exp3t_c, exp5_c]
    n_vals = [exp0_n, exp1_n, exp2_n, exp3_n, exp3t_n, exp5_n]

    y = np.arange(len(experiments))
    h = 0.3

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.barh(y + h / 2, c_vals, h, label='+coords', color=C_FUSION,
            edgecolor='white', linewidth=0.5, alpha=0.9)
    ax.barh(y - h / 2, n_vals, h, label='-coords', color=C_FUSION,
            edgecolor='gray', linewidth=0.5, alpha=0.5, hatch='//')

    # Delta 标注
    for i in range(len(experiments)):
        delta = n_vals[i] - c_vals[i]
        x_pos = max(c_vals[i], n_vals[i]) + 0.01
        color_d = C_DELTA_POS if delta >= 0 else C_DELTA_NEG
        ax.text(x_pos, y[i], f'{delta:+.3f}', ha='left', va='center',
                fontsize=9, color=color_d, fontweight='bold')

    ax.set_yticks(y)
    ax.set_yticklabels(experiments, fontsize=9)
    ax.set_xlabel('Score')
    ax.set_title('Summary: Impact of Auxiliary Features Across All Experiments')
    ax.legend(frameon=True, fancybox=False, edgecolor='gray', loc='lower right')
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.invert_yaxis()
    fig.tight_layout()
    savefig(fig, 'fig12_summary_all_experiments.pdf')


# =====================================================================
# Main
# =====================================================================
if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))

    print("=" * 60)
    print("生成 with-coords vs without-coords 对比图表")
    print(f"输出目录: {os.path.abspath(OUT_DIR)}")
    print("=" * 60)

    plot_fig1()
    plot_fig2()
    plot_fig3()
    plot_fig4()
    plot_fig5()
    plot_fig6()
    plot_fig7()
    plot_fig8()
    plot_fig9()
    plot_fig10()
    plot_fig11()
    plot_fig12()

    print("\n" + "=" * 60)
    print("All 12 comparison figures generated!")
    print(f"输出目录: {os.path.abspath(OUT_DIR)}")
    print("=" * 60)
