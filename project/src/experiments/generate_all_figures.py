"""
生成所有实验的 PDF 图表。
字体: Arial (TrueType 42)，Illustrator 可编辑。

Usage:
    python -m src.experiments.generate_all_figures
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
C_ID = '#1976D2'       # 深蓝
C_OOD = '#E53935'      # 红色

RESULTS_DIR = os.path.join('results', 'experiments')


def load_csv(exp, filename):
    return pd.read_csv(os.path.join(RESULTS_DIR, exp, filename))


def load_json(exp, filename='tier_b.json'):
    with open(os.path.join(RESULTS_DIR, exp, filename)) as f:
        return json.load(f)


def savefig(fig, exp, name):
    out_dir = os.path.join(RESULTS_DIR, exp)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.savefig(path)
    # 同时保存 PNG 用于 HTML
    png_path = path.replace('.pdf', '.png')
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# =====================================================================
# Exp 0: Per-Country CV — Heatmap (per-country × per-class F1)
# =====================================================================
def plot_exp0():
    print("\n=== Exp 0: Per-Country CV ===")

    # CSV 格式: country, class, feature_type, f1, precision, recall, support
    for ft in ['ae', 's2', 'fusion']:
        df = load_csv('exp0_per_country_cv', f'per_class_{ft}_tier_b.csv')
        # Pivot to matrix: country × class
        pivot = df.pivot_table(index='country', columns='class',
                               values='f1', aggfunc='mean')
        countries = pivot.index.tolist()
        class_names = pivot.columns.tolist()
        matrix = pivot.values

        fig, ax = plt.subplots(figsize=(max(8, len(class_names)*0.9),
                                        max(5, len(countries)*0.35)))
        cmap = LinearSegmentedColormap.from_list('custom',
            ['#FFFFFF', '#FFF9C4', '#FFE082', '#FF9800', '#E65100'], N=256)
        im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=0, vmax=1)
        ax.set_xticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.set_yticks(range(len(countries)))
        ax.set_yticklabels(countries)
        ax.set_xlabel('Crop Class')
        ax.set_ylabel('Country')
        ax.set_title(f'Per-Class F1 Score — {ft.upper()} (Tier B, 5-Fold CV)')

        for i in range(len(countries)):
            for j in range(len(class_names)):
                val = matrix[i, j]
                if not np.isnan(val):
                    txt_color = 'white' if val > 0.65 else 'black'
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                            fontsize=7, color=txt_color)

        plt.colorbar(im, ax=ax, shrink=0.8, label='F1 Score')
        fig.tight_layout()
        savefig(fig, 'exp0_per_country_cv', f'heatmap_{ft}_tier_b.pdf')

    # AE vs S2 差异热力图
    ae_df = load_csv('exp0_per_country_cv', 'per_class_ae_tier_b.csv')
    s2_df = load_csv('exp0_per_country_cv', 'per_class_s2_tier_b.csv')
    ae_pivot = ae_df.pivot_table(index='country', columns='class',
                                 values='f1', aggfunc='mean')
    s2_pivot = s2_df.pivot_table(index='country', columns='class',
                                 values='f1', aggfunc='mean')
    common_classes = sorted(set(ae_pivot.columns) & set(s2_pivot.columns))
    common_countries = sorted(set(ae_pivot.index) & set(s2_pivot.index))
    ae_m = ae_pivot.loc[common_countries, common_classes].values
    s2_m = s2_pivot.loc[common_countries, common_classes].values
    diff = ae_m - s2_m

    fig, ax = plt.subplots(figsize=(max(8, len(common_classes)*0.9),
                                    max(5, len(common_countries)*0.35)))
    cmap_div = LinearSegmentedColormap.from_list('div',
        ['#E53935', '#FFCDD2', '#FFFFFF', '#C8E6C9', '#2E7D32'], N=256)
    vmax = max(abs(np.nanmin(diff)), abs(np.nanmax(diff)), 0.01)
    im = ax.imshow(diff, cmap=cmap_div, aspect='auto', vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(common_classes)))
    ax.set_xticklabels(common_classes, rotation=45, ha='right')
    ax.set_yticks(range(len(common_countries)))
    ax.set_yticklabels(common_countries)
    ax.set_xlabel('Crop Class')
    ax.set_ylabel('Country')
    ax.set_title('AE − S2 F1 Difference (Tier B, 5-Fold CV)')

    for i in range(len(common_countries)):
        for j in range(len(common_classes)):
            val = diff[i, j]
            if not np.isnan(val):
                ax.text(j, i, f'{val:+.2f}', ha='center', va='center',
                        fontsize=7, color='black')

    plt.colorbar(im, ax=ax, shrink=0.8, label='ΔF1 (AE − S2)')
    fig.tight_layout()
    savefig(fig, 'exp0_per_country_cv', 'heatmap_ae_vs_s2_diff.pdf')


# =====================================================================
# Exp 1: ID Baseline — Bar chart
# =====================================================================
def plot_exp1():
    print("\n=== Exp 1: ID Baseline ===")
    df = load_csv('exp1_id_baseline', 'summary_tier_b.csv')

    metrics = ['BA_mean', 'F1_mean', 'Kappa_mean']
    labels = ['Balanced\nAccuracy', 'Macro F1', 'Cohen\'s\nKappa']
    x = np.arange(len(metrics))
    w = 0.25

    fig, ax = plt.subplots(figsize=(6, 4))
    for i, (ft, color, label) in enumerate([
        ('ae', C_AE, 'AlphaEarth'), ('s2', C_S2, 'Sentinel-2'),
        ('fusion', C_FUSION, 'Fusion')
    ]):
        row = df[df.feature_type == ft].iloc[0]
        vals = [row[m] for m in metrics]
        errs = [row[m.replace('mean', 'std')] for m in metrics]
        ax.bar(x + i*w, vals, w, yerr=errs, label=label, color=color,
               edgecolor='white', linewidth=0.5, capsize=3)

    ax.set_xticks(x + w)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Score')
    ax.set_ylim(0.6, 0.82)
    ax.set_title('Exp 1: ID Baseline (Pooled 5-Fold CV, Tier B)')
    ax.legend(frameon=True, fancybox=False, edgecolor='gray')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    fig.tight_layout()
    savefig(fig, 'exp1_id_baseline', 'id_baseline_tier_b.pdf')


# =====================================================================
# Exp 2: LOCO — grouped bar chart per country + summary
# =====================================================================
def plot_exp2():
    print("\n=== Exp 2: LOCO ===")

    ae = load_csv('exp2_loco', 'loco_ae_tier_b.csv')
    s2 = load_csv('exp2_loco', 'loco_s2_tier_b.csv')
    fu = load_csv('exp2_loco', 'loco_fusion_tier_b.csv')

    countries = ae['test_country'].values
    n = len(countries)
    x = np.arange(n)
    w = 0.27

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x - w, ae['balanced_accuracy'], w, label='AlphaEarth',
           color=C_AE, edgecolor='white', linewidth=0.3)
    ax.bar(x, s2['balanced_accuracy'], w, label='Sentinel-2',
           color=C_S2, edgecolor='white', linewidth=0.3)
    ax.bar(x + w, fu['balanced_accuracy'], w, label='Fusion',
           color=C_FUSION, edgecolor='white', linewidth=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels(countries, fontsize=9)
    ax.set_ylabel('Balanced Accuracy')
    ax.set_title('Exp 2: Leave-One-Country-Out (LOCO) — OOD Performance')
    ax.legend(frameon=True, fancybox=False, edgecolor='gray')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 0.75)
    fig.tight_layout()
    savefig(fig, 'exp2_loco', 'loco_per_country_tier_b.pdf')

    # ID vs OOD gap summary
    id_df = load_csv('exp1_id_baseline', 'summary_tier_b.csv')
    fig, ax = plt.subplots(figsize=(6, 4))
    fts = ['ae', 's2', 'fusion']
    labels_ft = ['AlphaEarth', 'Sentinel-2', 'Fusion']
    colors_ft = [C_AE, C_S2, C_FUSION]
    x = np.arange(len(fts))
    w = 0.35

    id_vals = [id_df[id_df.feature_type == ft]['BA_mean'].values[0] for ft in fts]
    ood_vals = [ae['balanced_accuracy'].mean(),
                s2['balanced_accuracy'].mean(),
                fu['balanced_accuracy'].mean()]

    ax.bar(x - w/2, id_vals, w, label='ID (5-Fold CV)', color=[C_AE, C_S2, C_FUSION],
           alpha=0.9, edgecolor='white')
    ax.bar(x + w/2, ood_vals, w, label='OOD (LOCO Mean)', color=[C_AE, C_S2, C_FUSION],
           alpha=0.4, edgecolor='gray', linewidth=0.5, hatch='//')

    # Annotate gap
    for i in range(len(fts)):
        gap = id_vals[i] - ood_vals[i]
        mid_y = (id_vals[i] + ood_vals[i]) / 2
        ax.annotate(f'Δ={gap:.2f}', xy=(i + w/2, ood_vals[i]),
                    xytext=(i + 0.55, mid_y),
                    fontsize=8, ha='left', color='red',
                    arrowprops=dict(arrowstyle='->', color='red', lw=0.8))

    ax.set_xticks(x)
    ax.set_xticklabels(labels_ft)
    ax.set_ylabel('Balanced Accuracy')
    ax.set_title('ID vs OOD Performance Gap')
    ax.legend(frameon=True, fancybox=False, edgecolor='gray')
    ax.set_ylim(0, 0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    fig.tight_layout()
    savefig(fig, 'exp2_loco', 'id_vs_ood_gap_tier_b.pdf')


# =====================================================================
# Exp 3: Climate Zone Transfer — LOZO bar + transfer matrix heatmap
# =====================================================================
def plot_exp3():
    print("\n=== Exp 3: Climate Zone Transfer ===")

    ae = load_csv('exp3_climate_zone', 'lozo_ae_tier_b.csv')
    s2 = load_csv('exp3_climate_zone', 'lozo_s2_tier_b.csv')
    fu = load_csv('exp3_climate_zone', 'lozo_fusion_tier_b.csv')

    zones = ae['test_zone'].values
    n = len(zones)
    x = np.arange(n)
    w = 0.27

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - w, ae['balanced_accuracy'], w, label='AlphaEarth',
           color=C_AE, edgecolor='white', linewidth=0.3)
    ax.bar(x, s2['balanced_accuracy'], w, label='Sentinel-2',
           color=C_S2, edgecolor='white', linewidth=0.3)
    ax.bar(x + w, fu['balanced_accuracy'], w, label='Fusion',
           color=C_FUSION, edgecolor='white', linewidth=0.3)

    zone_labels = [z.replace('_', '\n') for z in zones]
    ax.set_xticks(x)
    ax.set_xticklabels(zone_labels, fontsize=9)
    ax.set_ylabel('Balanced Accuracy')
    ax.set_title('Exp 3: Leave-One-Zone-Out (LOZO) — OOD Performance')
    ax.legend(frameon=True, fancybox=False, edgecolor='gray')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 0.75)
    fig.tight_layout()
    savefig(fig, 'exp3_climate_zone', 'lozo_per_zone_tier_b.pdf')

    # Transfer matrix heatmaps
    for ft, ft_label in [('ae', 'AlphaEarth'), ('s2', 'Sentinel-2'),
                          ('fusion', 'Fusion')]:
        tm = load_csv('exp3_climate_zone', f'transfer_matrix_{ft}_tier_b.csv')
        zones_src = sorted(tm['train_zone'].unique())
        zones_tgt = sorted(tm['test_zone'].unique())
        all_zones = sorted(set(zones_src) | set(zones_tgt))

        matrix = np.full((len(all_zones), len(all_zones)), np.nan)
        for _, row in tm.iterrows():
            i = all_zones.index(row['train_zone'])
            j = all_zones.index(row['test_zone'])
            matrix[i, j] = row['balanced_accuracy']

        fig, ax = plt.subplots(figsize=(6, 5))
        cmap = LinearSegmentedColormap.from_list('custom',
            ['#FFFFFF', '#E3F2FD', '#64B5F6', '#1976D2', '#0D47A1'], N=256)
        im = ax.imshow(matrix, cmap=cmap, aspect='equal', vmin=0, vmax=0.7)
        zone_short = [z.replace('nordic_baltic', 'nordic\nbalt.').replace('mediterranean', 'mediter.')
                      for z in all_zones]
        ax.set_xticks(range(len(all_zones)))
        ax.set_xticklabels(zone_short, rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(len(all_zones)))
        ax.set_yticklabels(zone_short, fontsize=8)
        ax.set_xlabel('Test Zone')
        ax.set_ylabel('Train Zone')
        ax.set_title(f'Cross-Zone Transfer Matrix — {ft_label}')

        for i in range(len(all_zones)):
            for j in range(len(all_zones)):
                val = matrix[i, j]
                if not np.isnan(val):
                    txt_color = 'white' if val > 0.4 else 'black'
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                            fontsize=9, fontweight='bold', color=txt_color)
                elif i == j:
                    ax.text(j, i, '—', ha='center', va='center',
                            fontsize=10, color='gray')

        plt.colorbar(im, ax=ax, shrink=0.8, label='Balanced Accuracy')
        fig.tight_layout()
        savefig(fig, 'exp3_climate_zone', f'transfer_matrix_{ft}_tier_b.pdf')


# =====================================================================
# Exp 4: Label Efficiency — Learning curves
# =====================================================================
def plot_exp4():
    print("\n=== Exp 4: Label Efficiency ===")
    df = load_csv('exp4_label_efficiency', 'label_efficiency_tier_b.csv')

    # Part 1: Learning curve
    curve_df = df[df['n_per_class'].notna() & (df['target_country'].isna())]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for ft, color, label in [('ae', C_AE, 'AlphaEarth'), ('s2', C_S2, 'Sentinel-2')]:
        sub = curve_df[curve_df.feature_type == ft].copy()
        sub['n_pc'] = sub['n_per_class'].apply(lambda x: int(float(x)) if x != 'all' else 99999)
        sub = sub.sort_values('n_pc')
        x_vals = sub['n_train'].values
        ax.errorbar(x_vals, sub['BA_mean'].values, yerr=sub['BA_std'].values,
                    marker='o', label=label, color=color, linewidth=2,
                    capsize=3, markersize=5)

    ax.set_xscale('log')
    ax.set_xlabel('Number of Training Samples')
    ax.set_ylabel('Balanced Accuracy (OOD)')
    ax.set_title('Exp 4: Label Efficiency — OOD Learning Curve')
    ax.legend(frameon=True, fancybox=False, edgecolor='gray')
    ax.grid(alpha=0.3, linestyle='--')
    fig.tight_layout()
    savefig(fig, 'exp4_label_efficiency', 'learning_curve_tier_b.pdf')

    # Part 2: Mixed scenario
    mixed_df = df[df['target_country'].notna()]
    if len(mixed_df) > 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        for ft, color, label in [('ae', C_AE, 'AlphaEarth'), ('s2', C_S2, 'Sentinel-2')]:
            sub = mixed_df[mixed_df.feature_type == ft].sort_values('n_target_samples')
            ax.plot(sub['n_target_samples'], sub['BA'], marker='s', label=label,
                    color=color, linewidth=2, markersize=6)

        ax.set_xlabel('Number of Target Domain Samples')
        ax.set_ylabel('Balanced Accuracy')
        ax.set_title('Exp 4b: Mixed Scenario (Source + Target, ES)')
        ax.legend(frameon=True, fancybox=False, edgecolor='gray')
        ax.grid(alpha=0.3, linestyle='--')
        fig.tight_layout()
        savefig(fig, 'exp4_label_efficiency', 'mixed_scenario_tier_b.pdf')


# =====================================================================
# Exp 5: Feature Analysis — CKA, SHAP, Country Prediction
# =====================================================================
def plot_exp5():
    print("\n=== Exp 5: Feature Analysis ===")
    data = load_json('exp5_feature_analysis')

    # --- CKA comparison ---
    fig, ax = plt.subplots(figsize=(5, 4))
    x = np.arange(2)
    w = 0.35
    cka_crop = [data['ae']['cka']['cka_crop'], data['s2']['cka']['cka_crop']]
    cka_country = [data['ae']['cka']['cka_country'], data['s2']['cka']['cka_country']]
    ax.bar(x - w/2, cka_crop, w, label='CKA(Embedding, Crop)', color='#66BB6A')
    ax.bar(x + w/2, cka_country, w, label='CKA(Embedding, Country)', color='#EF5350')
    ax.set_xticks(x)
    ax.set_xticklabels(['AlphaEarth', 'Sentinel-2'])
    ax.set_ylabel('Linear CKA')
    ax.set_title('Exp 5: Feature–Label Alignment (CKA)')
    ax.legend(frameon=True, fancybox=False, edgecolor='gray')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    fig.tight_layout()
    savefig(fig, 'exp5_feature_analysis', 'cka_comparison_tier_b.pdf')

    # --- SHAP Top 10 (AE, ID vs OOD) ---
    for ft, ft_label in [('ae', 'AlphaEarth'), ('s2', 'Sentinel-2')]:
        shap_data = data[ft]['shap']
        id_imp = sorted(shap_data['id_importance'].items(),
                        key=lambda x: x[1], reverse=True)[:10]
        ood_imp_dict = shap_data['ood_importance']

        features = [f[0] for f in id_imp]
        id_vals = [f[1] for f in id_imp]
        ood_vals = [ood_imp_dict.get(f, 0) for f in features]

        fig, ax = plt.subplots(figsize=(7, 4.5))
        y = np.arange(len(features))
        h = 0.35
        ax.barh(y + h/2, id_vals, h, label='ID', color=C_ID, alpha=0.8)
        ax.barh(y - h/2, ood_vals, h, label='OOD', color=C_OOD, alpha=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(features, fontsize=8)
        ax.set_xlabel('Mean |SHAP|')
        ax.set_title(f'SHAP Feature Importance — {ft_label} (ID vs OOD)')
        ax.legend(frameon=True, fancybox=False, edgecolor='gray')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        fig.tight_layout()
        savefig(fig, 'exp5_feature_analysis', f'shap_{ft}_tier_b.pdf')

    # --- Country Prediction ---
    fig, ax = plt.subplots(figsize=(4, 3.5))
    x = np.arange(2)
    ba_vals = [data['ae']['country_prediction']['balanced_accuracy'],
               data['s2']['country_prediction']['balanced_accuracy']]
    ax.bar(x, ba_vals, 0.5, color=[C_AE, C_S2], edgecolor='white')
    for i, v in enumerate(ba_vals):
        ax.text(i, v + 0.002, f'{v:.4f}', ha='center', fontsize=9,
                fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['AlphaEarth', 'Sentinel-2'])
    ax.set_ylabel('Balanced Accuracy')
    ax.set_title('Country Predictability')
    ax.set_ylim(0.95, 1.005)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    fig.tight_layout()
    savefig(fig, 'exp5_feature_analysis', 'country_prediction_tier_b.pdf')


# =====================================================================
# Exp 6: Ablation — ID vs OOD bar chart
# =====================================================================
def plot_exp6():
    print("\n=== Exp 6: Ablation ===")
    df = load_csv('exp6_ablation', 'ablation_fusion_tier_b.csv')

    groups = df['group'].values
    labels = [g.replace('_', '\n') for g in groups]
    n = len(groups)
    x = np.arange(n)
    w = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars_id = ax.bar(x - w/2, df['id_BA'], w, label='ID', color=C_ID, alpha=0.85,
                     edgecolor='white', linewidth=0.5)
    bars_ood = ax.bar(x + w/2, df['ood_BA'], w, label='OOD', color=C_OOD, alpha=0.85,
                      edgecolor='white', linewidth=0.5)

    # Annotate feature count
    for i, row in df.iterrows():
        ax.text(i, max(row['id_BA'], row['ood_BA']) + 0.02,
                f'{int(row["n_features"])}d', ha='center', fontsize=8,
                color='gray')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('Balanced Accuracy')
    ax.set_title('Exp 6: Ablation Study — ID vs OOD (Tier B)')
    ax.legend(frameon=True, fancybox=False, edgecolor='gray')
    ax.set_ylim(0, 0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    fig.tight_layout()
    savefig(fig, 'exp6_ablation', 'ablation_fusion_tier_b.pdf')

    # Delta chart
    fig, ax = plt.subplots(figsize=(10, 4))
    colors_delta = ['#E53935' if d > 0.31 else '#FF9800' if d > 0.29 else '#66BB6A'
                    for d in df['delta_BA']]
    ax.bar(x, df['delta_BA'], 0.6, color=colors_delta, edgecolor='white', linewidth=0.5)
    for i, row in df.iterrows():
        ax.text(i, row['delta_BA'] + 0.003, f'{row["delta_BA"]:.3f}',
                ha='center', fontsize=8, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('ID − OOD Gap (Δ BA)')
    ax.set_title('Generalization Gap by Feature Configuration')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    fig.tight_layout()
    savefig(fig, 'exp6_ablation', 'ablation_delta_tier_b.pdf')


# =====================================================================
# Main
# =====================================================================
if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))

    plot_exp0()
    plot_exp1()
    plot_exp2()
    plot_exp3()
    plot_exp4()
    plot_exp5()
    plot_exp6()

    print("\nAll figures generated!")
