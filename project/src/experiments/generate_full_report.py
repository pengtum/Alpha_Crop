#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成完整实验 HTML 报告 — 涵盖所有已完成实验 (Exp 0-5, 2b)。

Usage:
    python -m src.experiments.generate_full_report
"""
import os
import sys
import io
import json
import base64
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap

# --- Matplotlib 全局设置 ---
plt.rcParams.update({
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'font.family': 'Arial',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

# --- 常量 ---
C_AE = '#2196F3'
C_S2 = '#FF9800'
C_FUSION = '#4CAF50'
C_ID = '#1976D2'
C_OOD = '#E53935'

RESULTS_DIR = os.path.join('results', 'experiments')

COUNTRIES = [
    "AT", "BE", "CZ", "DE", "DK", "EE", "ES", "FR",
    "HR", "IE", "LT", "LV", "NL", "PT", "RO", "SE", "SI", "SK",
]

COUNTRY_NAMES = {
    "AT": "奥地利", "BE": "比利时", "CZ": "捷克", "DE": "德国",
    "DK": "丹麦", "EE": "爱沙尼亚", "ES": "西班牙", "FR": "法国",
    "HR": "克罗地亚", "IE": "爱尔兰", "LT": "立陶宛", "LV": "拉脱维亚",
    "NL": "荷兰", "PT": "葡萄牙", "RO": "罗马尼亚", "SE": "瑞典",
    "SI": "斯洛文尼亚", "SK": "斯洛伐克",
}

ZONES = {
    "atlantic":      {"label": "大西洋", "countries": ["BE", "DE", "DK", "FR", "IE", "NL"]},
    "continental":   {"label": "大陆性", "countries": ["AT", "CZ", "SK"]},
    "mediterranean": {"label": "地中海", "countries": ["ES", "HR", "PT"]},
    "nordic_baltic": {"label": "北欧-波罗的海", "countries": ["EE", "LT", "LV", "SE"]},
    "southeast":     {"label": "东南欧", "countries": ["RO"]},
}

FEATURE_SETS = ["ae", "s2", "fusion"]
FS_LABELS = {"ae": "AlphaEarth", "s2": "Sentinel-2", "fusion": "Fusion"}
FS_COLORS = {"ae": C_AE, "s2": C_S2, "fusion": C_FUSION}
FS_DIMS = {"ae": 128, "s2": 168, "fusion": 296}
FS_DIMS_COORDS = {"ae": 131, "s2": 171, "fusion": 299}

CURVE_LABELS = {
    "10": "100", "50": "500", "100": "1,000",
    "500": "5,000", "1000": "10,000", "5000": "~46k", "0": "all (~795k)",
}
CURVE_ORDER = ["10", "50", "100", "500", "1000", "5000", "0"]

TIER_B_CROPS = [
    "grassland", "maize", "orchards", "other_crops", "potatoes",
    "root_crops", "strawberries", "sunflower", "vineyards", "winter_wheat",
]


# --- 数据加载 ---
def load_csv(exp, filename):
    return pd.read_csv(os.path.join(RESULTS_DIR, exp, filename))


def load_json(exp, filename):
    with open(os.path.join(RESULTS_DIR, exp, filename), encoding='utf-8') as f:
        return json.load(f)


def load_png_as_base64(filepath):
    """读取 PNG 文件并转为 base64。"""
    with open(filepath, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


# --- 图表工具 ---
def fig_to_base64(fig):
    """将 matplotlib figure 转为 base64 PNG 字符串。"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def img_tag(b64, alt="figure", width="100%"):
    return f'<img src="data:image/png;base64,{b64}" alt="{alt}" style="max-width:{width};display:block;margin:12px auto;">'


# --- HTML 工具 ---
def pct(v):
    return f"{v * 100:.2f}"

def pct1(v):
    return f"{v * 100:.1f}"

def delta_str(d):
    sign = "+" if d >= 0 else ""
    return f"{sign}{d * 100:.2f}"


# ===================================================================
# 图表生成函数
# ===================================================================

def plot_exp1_bars():
    """Exp 1: ID Baseline 柱状图。"""
    df = load_csv('exp1_id_baseline', 'summary_tier_b_nocoords.csv')
    metrics = ['BA_mean', 'F1_mean', 'Kappa_mean']
    labels = ['Balanced\nAccuracy', 'Macro F1', "Cohen's\nKappa"]
    x = np.arange(len(metrics))
    w = 0.25

    fig, ax = plt.subplots(figsize=(6, 4))
    for i, (ft, color, label) in enumerate([
        ('ae', C_AE, 'AlphaEarth (128d)'),
        ('s2', C_S2, 'Sentinel-2 (168d)'),
        ('fusion', C_FUSION, 'Fusion (296d)')
    ]):
        row = df[df.feature_type == ft].iloc[0]
        vals = [row[m] for m in metrics]
        errs = [row[m.replace('mean', 'std')] for m in metrics]
        bars = ax.bar(x + i * w, vals, w, yerr=errs, label=label, color=color,
                      edgecolor='white', linewidth=0.5, capsize=3)
        for j, v in enumerate(vals):
            ax.text(x[j] + i * w, v + errs[j] + 0.003, f'{v:.3f}',
                    ha='center', va='bottom', fontsize=7, fontweight='bold')

    ax.set_xticks(x + w)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Score')
    ax.set_ylim(0.55, 0.80)
    ax.set_title('Exp 1: ID Baseline (Tier B, 5-Fold CV, No Coords)')
    ax.legend(frameon=True, fancybox=False, edgecolor='gray')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    fig.tight_layout()
    return fig_to_base64(fig)


def plot_exp0_country_bars():
    """Exp 0: 逐国 CV BA 柱状图。"""
    df = load_csv('exp0_per_country_cv', 'summary_tier_b_nocoords.csv')

    fig, ax = plt.subplots(figsize=(14, 5))
    countries = sorted(df['country'].unique())
    n = len(countries)
    x = np.arange(n)
    w = 0.27

    for i, (ft, color, label) in enumerate([
        ('ae', C_AE, 'AlphaEarth'), ('s2', C_S2, 'Sentinel-2'),
        ('fusion', C_FUSION, 'Fusion')
    ]):
        sub = df[df.feature_type == ft].set_index('country').loc[countries]
        ax.bar(x + (i - 1) * w, sub['BA_mean'].values, w, yerr=sub['BA_std'].values,
               label=label, color=color, edgecolor='white', linewidth=0.3, capsize=2)

    ax.set_xticks(x)
    ax.set_xticklabels(countries, fontsize=9)
    ax.set_ylabel('Balanced Accuracy')
    ax.set_title('Exp 0: Per-Country 5-Fold CV (No Coords)')
    ax.legend(frameon=True, fancybox=False, edgecolor='gray')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 0.85)
    fig.tight_layout()
    return fig_to_base64(fig)


def plot_exp2_loco():
    """Exp 2: LOCO 逐国 OOD BA 柱状图。"""
    ae = load_csv('exp2_loco', 'loco_ae_nocoords_tier_b.csv')
    s2 = load_csv('exp2_loco', 'loco_s2_nocoords_tier_b.csv')
    fu = load_csv('exp2_loco', 'loco_fusion_nocoords_tier_b.csv')

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
    ax.set_title('Exp 2: Leave-One-Country-Out OOD (No Coords)')
    ax.legend(frameon=True, fancybox=False, edgecolor='gray')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 0.75)
    fig.tight_layout()
    return fig_to_base64(fig)


def plot_exp2_id_vs_ood():
    """Exp 2: ID vs OOD gap 对比。"""
    id_df = load_csv('exp1_id_baseline', 'summary_tier_b_nocoords.csv')
    ae = load_csv('exp2_loco', 'loco_ae_nocoords_tier_b.csv')
    s2 = load_csv('exp2_loco', 'loco_s2_nocoords_tier_b.csv')
    fu = load_csv('exp2_loco', 'loco_fusion_nocoords_tier_b.csv')

    fig, ax = plt.subplots(figsize=(6, 4))
    fts = ['ae', 's2', 'fusion']
    labels_ft = ['AlphaEarth\n(128d)', 'Sentinel-2\n(168d)', 'Fusion\n(296d)']
    x = np.arange(len(fts))
    w = 0.35

    id_vals = [id_df[id_df.feature_type == ft]['BA_mean'].values[0] for ft in fts]
    ood_vals = [ae['balanced_accuracy'].mean(),
                s2['balanced_accuracy'].mean(),
                fu['balanced_accuracy'].mean()]

    ax.bar(x - w / 2, id_vals, w, label='ID (5-Fold CV)',
           color=[C_AE, C_S2, C_FUSION], alpha=0.9, edgecolor='white')
    ax.bar(x + w / 2, ood_vals, w, label='OOD (LOCO Mean)',
           color=[C_AE, C_S2, C_FUSION], alpha=0.4, edgecolor='gray',
           linewidth=0.5, hatch='//')

    for i in range(len(fts)):
        gap = id_vals[i] - ood_vals[i]
        mid_y = (id_vals[i] + ood_vals[i]) / 2
        ax.annotate(f'Gap={gap:.3f}', xy=(i + w / 2, ood_vals[i]),
                    xytext=(i + 0.55, mid_y), fontsize=8, ha='left', color='red',
                    arrowprops=dict(arrowstyle='->', color='red', lw=0.8))

    ax.set_xticks(x)
    ax.set_xticklabels(labels_ft)
    ax.set_ylabel('Balanced Accuracy')
    ax.set_title('ID vs OOD Performance Gap (No Coords)')
    ax.legend(frameon=True, fancybox=False, edgecolor='gray')
    ax.set_ylim(0, 0.85)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    fig.tight_layout()
    return fig_to_base64(fig)


def plot_exp3_lozo():
    """Exp 3: LOZO 各气候带 BA。"""
    ae = load_csv('exp3_climate_zone', 'lozo_ae_nocoords_tier_b.csv')
    s2 = load_csv('exp3_climate_zone', 'lozo_s2_nocoords_tier_b.csv')
    fu = load_csv('exp3_climate_zone', 'lozo_fusion_nocoords_tier_b.csv')

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
    ax.set_title('Exp 3: Leave-One-Zone-Out (LOZO) (No Coords)')
    ax.legend(frameon=True, fancybox=False, edgecolor='gray')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 0.7)
    fig.tight_layout()
    return fig_to_base64(fig)


def plot_exp3_transfer_matrices():
    """Exp 3: 跨气候带迁移矩阵热力图。"""
    figs_b64 = {}
    for ft, ft_label in [('ae', 'AlphaEarth'), ('s2', 'Sentinel-2'), ('fusion', 'Fusion')]:
        tm = load_csv('exp3_climate_zone', f'transfer_matrix_{ft}_nocoords_tier_b.csv')
        zones_src = sorted(tm['train_zone'].unique())
        zones_tgt = sorted(tm['test_zone'].unique())
        all_zones = sorted(set(zones_src) | set(zones_tgt))

        matrix = np.full((len(all_zones), len(all_zones)), np.nan)
        for _, row in tm.iterrows():
            i = all_zones.index(row['train_zone'])
            j = all_zones.index(row['test_zone'])
            matrix[i, j] = row['balanced_accuracy']

        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        cmap = LinearSegmentedColormap.from_list('custom',
            ['#FFFFFF', '#E3F2FD', '#64B5F6', '#1976D2', '#0D47A1'], N=256)
        im = ax.imshow(matrix, cmap=cmap, aspect='equal', vmin=0, vmax=0.7)
        zone_short = [z.replace('nordic_baltic', 'nordic\nbalt.')
                       .replace('mediterranean', 'mediter.') for z in all_zones]
        ax.set_xticks(range(len(all_zones)))
        ax.set_xticklabels(zone_short, rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(len(all_zones)))
        ax.set_yticklabels(zone_short, fontsize=8)
        ax.set_xlabel('Test Zone')
        ax.set_ylabel('Train Zone')
        ax.set_title(f'Zone Transfer — {ft_label} (No Coords)')

        for i in range(len(all_zones)):
            for j in range(len(all_zones)):
                val = matrix[i, j]
                if not np.isnan(val):
                    txt_color = 'white' if val > 0.4 else 'black'
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                            fontsize=9, fontweight='bold', color=txt_color)
                elif i == j:
                    ax.text(j, i, '---', ha='center', va='center',
                            fontsize=10, color='gray')

        plt.colorbar(im, ax=ax, shrink=0.8, label='Balanced Accuracy')
        fig.tight_layout()
        figs_b64[ft] = fig_to_base64(fig)

    return figs_b64


def plot_exp4_learning_curve():
    """Exp 4: 学习曲线 (AE vs S2)。"""
    data = load_json('exp4_label_efficiency', 'tier_b_nocoords.json')

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for ft_key, color, label in [
        ('ae_curve', C_AE, 'AlphaEarth'),
        ('s2_curve', C_S2, 'Sentinel-2')
    ]:
        d = data[ft_key]['sample_sizes']
        sizes = []
        ba_means = []
        ba_stds = []
        for sk in CURVE_ORDER:
            if sk in d:
                sizes.append(d[sk]['n_train_actual'])
                ba_means.append(d[sk]['ba_mean'])
                ba_stds.append(d[sk]['ba_std'])

        ax.errorbar(sizes, ba_means, yerr=ba_stds, marker='o', label=label,
                    color=color, linewidth=2, capsize=3, markersize=5)

    ax.set_xscale('log')
    ax.set_xlabel('Number of Training Samples')
    ax.set_ylabel('Balanced Accuracy (OOD)')
    ax.set_title('Exp 4: Label Efficiency — OOD Learning Curve (No Coords)')
    ax.legend(frameon=True, fancybox=False, edgecolor='gray')
    ax.grid(alpha=0.3, linestyle='--')
    fig.tight_layout()
    return fig_to_base64(fig)


def plot_exp4_mixed():
    """Exp 4: Mixed scenario (大域训练 + 少量目标域标注)。"""
    data = load_json('exp4_label_efficiency', 'tier_b_nocoords.json')

    fig, ax = plt.subplots(figsize=(6, 4))

    for ft_key, color, label in [
        ('ae_mixed', C_AE, 'AlphaEarth'),
        ('s2_mixed', C_S2, 'Sentinel-2')
    ]:
        d = data[ft_key]
        keys = sorted([k for k in d.keys() if k not in ('feature_type', 'tier', 'test_countries')],
                       key=lambda x: int(x))
        x_vals = [int(k) for k in keys]
        ba_vals = [d[k]['balanced_accuracy'] for k in keys]
        ax.plot(x_vals, ba_vals, marker='s', label=label, color=color,
                linewidth=2, markersize=6)

    ax.set_xlabel('Target Domain Mix-in Ratio (%)')
    ax.set_ylabel('Balanced Accuracy')
    ax.set_title('Exp 4b: Mixed Scenario (No Coords)')
    ax.legend(frameon=True, fancybox=False, edgecolor='gray')
    ax.grid(alpha=0.3, linestyle='--')
    fig.tight_layout()
    return fig_to_base64(fig)


def plot_exp5_cka():
    """Exp 5: CKA 对比 — coords vs nocoords 四组柱状图。"""
    nc = load_json('exp5_feature_analysis', 'tier_b_nocoords.json')
    # coords CKA from CSV
    cka_c = load_csv('exp5_feature_analysis', 'cka_comparison_tier_b.csv')
    cka_nc = load_csv('exp5_feature_analysis', 'cka_comparison_tier_b_nocoords.csv')

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Panel 1: CKA_crop
    ax = axes[0]
    x = np.arange(2)
    w = 0.35
    ae_c_crop = cka_c[cka_c.feature_type == 'ae']['cka_crop'].values[0]
    s2_c_crop = cka_c[cka_c.feature_type == 's2']['cka_crop'].values[0]
    ae_nc_crop = nc['ae']['cka']['cka_crop']
    s2_nc_crop = nc['s2']['cka']['cka_crop']

    b1 = ax.bar(x - w/2, [ae_c_crop, s2_c_crop], w, label='With Coords', color='#90CAF9')
    b2 = ax.bar(x + w/2, [ae_nc_crop, s2_nc_crop], w, label='No Coords', color='#66BB6A')
    for bg in [b1, b2]:
        for bar in bg:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.002, f'{h:.4f}',
                    ha='center', va='bottom', fontsize=7, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['AlphaEarth', 'Sentinel-2'])
    ax.set_ylabel('Linear CKA')
    ax.set_title('CKA(Embedding, Crop)')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Panel 2: CKA_country
    ax = axes[1]
    ae_c_ctry = cka_c[cka_c.feature_type == 'ae']['cka_country'].values[0]
    s2_c_ctry = cka_c[cka_c.feature_type == 's2']['cka_country'].values[0]
    ae_nc_ctry = nc['ae']['cka']['cka_country']
    s2_nc_ctry = nc['s2']['cka']['cka_country']

    b1 = ax.bar(x - w/2, [ae_c_ctry, s2_c_ctry], w, label='With Coords', color='#EF9A9A')
    b2 = ax.bar(x + w/2, [ae_nc_ctry, s2_nc_ctry], w, label='No Coords', color='#EF5350')
    for bg in [b1, b2]:
        for bar in bg:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.002, f'{h:.4f}',
                    ha='center', va='bottom', fontsize=7, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['AlphaEarth', 'Sentinel-2'])
    ax.set_ylabel('Linear CKA')
    ax.set_title('CKA(Embedding, Country)')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    fig.suptitle('Exp 5: Feature-Label Alignment (CKA) — Coords Ablation', fontsize=13)
    fig.tight_layout()
    return fig_to_base64(fig)


def plot_exp5_shap(ft='ae'):
    """Exp 5: SHAP Top 15 (ID vs OOD)。"""
    data = load_json('exp5_feature_analysis', 'tier_b_nocoords.json')
    shap_data = data[ft]['shap']
    ft_label = FS_LABELS[ft]

    id_imp = sorted(shap_data['id_importance'].items(), key=lambda x: x[1], reverse=True)[:15]
    ood_imp_dict = shap_data['ood_importance']

    features = [f[0] for f in id_imp]
    id_vals = [f[1] for f in id_imp]
    ood_vals = [ood_imp_dict.get(f, 0) for f in features]

    fig, ax = plt.subplots(figsize=(7, 5))
    y = np.arange(len(features))
    h = 0.35
    ax.barh(y + h / 2, id_vals, h, label='ID', color=C_ID, alpha=0.8)
    ax.barh(y - h / 2, ood_vals, h, label='OOD', color=C_OOD, alpha=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(features, fontsize=8)
    ax.set_xlabel('Mean |SHAP|')
    ax.set_title(f'SHAP Feature Importance — {ft_label} (ID vs OOD, No Coords)')
    ax.legend(frameon=True, fancybox=False, edgecolor='gray')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    fig.tight_layout()
    return fig_to_base64(fig)


def plot_exp5_country_pred():
    """Exp 5: Country Predictability。"""
    data = load_json('exp5_feature_analysis', 'tier_b_nocoords.json')

    fig, ax = plt.subplots(figsize=(4, 3.5))
    x = np.arange(2)
    ba_vals = [data['ae']['country_prediction']['balanced_accuracy'],
               data['s2']['country_prediction']['balanced_accuracy']]
    bars = ax.bar(x, ba_vals, 0.5, color=[C_AE, C_S2], edgecolor='white')
    for i, v in enumerate(ba_vals):
        ax.text(i, v + 0.005, f'{v:.3f}', ha='center', fontsize=9, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(['AlphaEarth', 'Sentinel-2'])
    ax.set_ylabel('Balanced Accuracy')
    ax.set_title('Country Predictability (No Coords)')
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    fig.tight_layout()
    return fig_to_base64(fig)


def plot_summary_overview():
    """综合概览：ID + LOCO + LOZO 三组对比。"""
    id_df = load_csv('exp1_id_baseline', 'summary_tier_b_nocoords.csv')
    ae_loco = load_csv('exp2_loco', 'loco_ae_nocoords_tier_b.csv')
    s2_loco = load_csv('exp2_loco', 'loco_s2_nocoords_tier_b.csv')
    fu_loco = load_csv('exp2_loco', 'loco_fusion_nocoords_tier_b.csv')
    ae_lozo = load_csv('exp3_climate_zone', 'lozo_ae_nocoords_tier_b.csv')
    s2_lozo = load_csv('exp3_climate_zone', 'lozo_s2_nocoords_tier_b.csv')
    fu_lozo = load_csv('exp3_climate_zone', 'lozo_fusion_nocoords_tier_b.csv')

    fig, ax = plt.subplots(figsize=(8, 5))

    scenarios = ['ID\n(5-Fold CV)', 'OOD\n(LOCO Mean)', 'OOD\n(LOZO Mean)']
    x = np.arange(len(scenarios))
    w = 0.22

    for i, (ft, color, label) in enumerate([
        ('ae', C_AE, 'AlphaEarth'),
        ('s2', C_S2, 'Sentinel-2'),
        ('fusion', C_FUSION, 'Fusion')
    ]):
        id_val = id_df[id_df.feature_type == ft]['BA_mean'].values[0]
        loco_df = [ae_loco, s2_loco, fu_loco][i]
        lozo_df = [ae_lozo, s2_lozo, fu_lozo][i]
        ood_loco = loco_df['balanced_accuracy'].mean()
        ood_lozo = lozo_df['balanced_accuracy'].mean()
        vals = [id_val, ood_loco, ood_lozo]
        bars = ax.bar(x + (i - 1) * w, vals, w, label=label, color=color,
                      edgecolor='white', linewidth=0.5)
        for j, v in enumerate(vals):
            ax.text(x[j] + (i - 1) * w, v + 0.005, f'{v:.3f}',
                    ha='center', va='bottom', fontsize=7, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.set_ylabel('Balanced Accuracy')
    ax.set_title('Performance Summary (No Coords, Tier B)')
    ax.legend(frameon=True, fancybox=False, edgecolor='gray')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 0.82)
    fig.tight_layout()
    return fig_to_base64(fig)


# ===================================================================
# HTML 生成
# ===================================================================

def build_report():
    """构建完整 HTML 报告。"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    # --- 加载所有数据 ---
    print("[DATA] Loading experiment results...")

    # Exp 1
    exp1_nc = load_csv('exp1_id_baseline', 'summary_tier_b_nocoords.csv')
    exp1_c = load_csv('exp1_id_baseline', 'summary_tier_b.csv')

    # Exp 0
    exp0_nc = load_csv('exp0_per_country_cv', 'summary_tier_b_nocoords.csv')

    # Exp 2
    exp2_json = load_json('exp2_loco', 'tier_b_nocoords.json')
    ae_loco = load_csv('exp2_loco', 'loco_ae_nocoords_tier_b.csv')
    s2_loco = load_csv('exp2_loco', 'loco_s2_nocoords_tier_b.csv')
    fu_loco = load_csv('exp2_loco', 'loco_fusion_nocoords_tier_b.csv')

    # Exp 2b transfer matrices (LGBM)
    exp2b_lgbm = {}
    for ft in FEATURE_SETS:
        exp2b_lgbm[ft] = load_csv('exp2b_country_transfer', f'transfer_{ft}_nocoords_tier_b.csv')
    # Exp 2b XGB
    exp2b_xgb = {}
    for ft in FEATURE_SETS:
        exp2b_xgb[ft] = load_csv('exp2b_country_transfer', f'transfer_{ft}_nocoords_xgb_tier_b.csv')

    # Exp 3
    exp3_json = load_json('exp3_climate_zone', 'tier_b_nocoords.json')

    # Exp 4
    exp4_json = load_json('exp4_label_efficiency', 'tier_b_nocoords.json')

    # Exp 5
    exp5_nc = load_json('exp5_feature_analysis', 'tier_b_nocoords.json')

    # CKA comparison (coords vs nocoords)
    cka_c = load_csv('exp5_feature_analysis', 'cka_comparison_tier_b.csv')
    cka_nc = load_csv('exp5_feature_analysis', 'cka_comparison_tier_b_nocoords.csv')

    # Coords comparison data
    all_data = load_json('.', 'all_results_combined.json')

    print("[DATA] All data loaded.")

    # --- 生成图表 ---
    print("[PLOT] Generating charts...")
    fig_exp1 = plot_exp1_bars()
    print("  [OK] Exp 1: ID Baseline")
    fig_exp0 = plot_exp0_country_bars()
    print("  [OK] Exp 0: Country bars")
    fig_exp2_loco = plot_exp2_loco()
    print("  [OK] Exp 2: LOCO per-country")
    fig_exp2_gap = plot_exp2_id_vs_ood()
    print("  [OK] Exp 2: ID vs OOD gap")
    fig_exp3_lozo = plot_exp3_lozo()
    print("  [OK] Exp 3: LOZO")
    fig_exp3_tm = plot_exp3_transfer_matrices()
    print("  [OK] Exp 3: Transfer matrices")
    fig_exp4_lc = plot_exp4_learning_curve()
    print("  [OK] Exp 4: Learning curve")
    fig_exp4_mix = plot_exp4_mixed()
    print("  [OK] Exp 4: Mixed scenario")
    fig_exp5_cka = plot_exp5_cka()
    print("  [OK] Exp 5: CKA")
    fig_exp5_shap_ae = plot_exp5_shap('ae')
    print("  [OK] Exp 5: SHAP AE")
    fig_exp5_shap_s2 = plot_exp5_shap('s2')
    print("  [OK] Exp 5: SHAP S2")
    fig_exp5_cpred = plot_exp5_country_pred()
    print("  [OK] Exp 5: Country prediction")
    fig_summary = plot_summary_overview()
    print("  [OK] Summary overview")

    # Load Exp 2b PNGs
    print("[PLOT] Loading Exp 2b existing PNGs...")
    exp2b_dir = os.path.join(RESULTS_DIR, 'exp2b_country_transfer')
    exp2b_pngs = {}
    png_files = [
        'transfer_matrix_ae_nocoords.png',
        'transfer_matrix_s2_nocoords.png',
        'transfer_matrix_fusion_nocoords.png',
        'transfer_symmetry.png',
        'train_generalization_lgbm_vs_xgb.png',
        'lgbm_vs_xgb_scatter.png',
        'winner_map_xgb_vs_lgbm.png',
    ]
    for fn in png_files:
        fp = os.path.join(exp2b_dir, fn)
        if os.path.exists(fp):
            exp2b_pngs[fn] = load_png_as_base64(fp)
            print(f"  [OK] {fn}")
        else:
            print(f"  [--] {fn} not found, skipping")

    # --- 计算关键统计量 ---
    # Exp 1 nocoords
    ae_id_ba = exp1_nc[exp1_nc.feature_type == 'ae']['BA_mean'].values[0]
    s2_id_ba = exp1_nc[exp1_nc.feature_type == 's2']['BA_mean'].values[0]
    fu_id_ba = exp1_nc[exp1_nc.feature_type == 'fusion']['BA_mean'].values[0]
    total_samples = int(exp1_nc[exp1_nc.feature_type == 'ae']['n_samples'].values[0])

    # Exp 2 LOCO means
    ae_ood_mean = ae_loco['balanced_accuracy'].mean()
    s2_ood_mean = s2_loco['balanced_accuracy'].mean()
    fu_ood_mean = fu_loco['balanced_accuracy'].mean()

    # Exp 2b: mean BA per feature set (LGBM)
    exp2b_means_lgbm = {}
    for ft in FEATURE_SETS:
        df = exp2b_lgbm[ft]
        # exclude self-transfer (train==test is not in the data anyway)
        exp2b_means_lgbm[ft] = df['balanced_accuracy'].mean()

    # Exp 2b: mean BA per feature set (XGB)
    exp2b_means_xgb = {}
    for ft in FEATURE_SETS:
        df = exp2b_xgb[ft]
        exp2b_means_xgb[ft] = df['balanced_accuracy'].mean()

    # Exp 2b: best/worst train countries per feature set
    exp2b_train_stats = {}
    for ft in FEATURE_SETS:
        df = exp2b_lgbm[ft]
        train_means = df.groupby('train_country')['balanced_accuracy'].mean()
        exp2b_train_stats[ft] = {
            'best_train': train_means.idxmax(),
            'best_train_ba': train_means.max(),
            'worst_train': train_means.idxmin(),
            'worst_train_ba': train_means.min(),
        }
        # best/worst test country
        test_means = df.groupby('test_country')['balanced_accuracy'].mean()
        exp2b_train_stats[ft]['best_test'] = test_means.idxmax()
        exp2b_train_stats[ft]['best_test_ba'] = test_means.max()
        exp2b_train_stats[ft]['worst_test'] = test_means.idxmin()
        exp2b_train_stats[ft]['worst_test_ba'] = test_means.min()

    # Exp 5 CKA
    ae_nc_crop = exp5_nc['ae']['cka']['cka_crop']
    s2_nc_crop = exp5_nc['s2']['cka']['cka_crop']
    ae_nc_ctry = exp5_nc['ae']['cka']['cka_country']
    s2_nc_ctry = exp5_nc['s2']['cka']['cka_country']
    ae_c_crop = cka_c[cka_c.feature_type == 'ae']['cka_crop'].values[0]
    s2_c_crop = cka_c[cka_c.feature_type == 's2']['cka_crop'].values[0]

    ae_cpred = exp5_nc['ae']['country_prediction']['balanced_accuracy']
    s2_cpred = exp5_nc['s2']['country_prediction']['balanced_accuracy']

    # --- HTML 构建 ---
    print("[HTML] Building report...")
    parts = []

    # === HEAD ===
    parts.append(f"""\
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AlphaEarth 泛欧作物分类 -- 完整实验报告</title>
<style>
:root {{
    --ae: #2196F3; --s2: #FF9800; --fusion: #4CAF50;
    --bg: #FAFAFA; --card: #FFFFFF; --border: #E0E0E0;
    --text: #212121; --text2: #616161; --accent: #1565C0;
    --red: #C62828; --green: #2E7D32;
}}
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    font-family: -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif;
    background: var(--bg); color: var(--text);
    line-height: 1.7; padding: 0 20px;
}}
.container {{ max-width: 1200px; margin: 0 auto; }}

/* Header */
header {{
    background: linear-gradient(135deg, #1565C0, #0D47A1);
    color: white; padding: 40px 30px;
    margin: 20px -20px 30px; border-radius: 0 0 12px 12px;
}}
header h1 {{ font-size: 26px; margin-bottom: 8px; }}
header .subtitle {{ opacity: 0.92; font-size: 15px; margin-bottom: 12px; }}
header p.meta {{ opacity: 0.7; font-size: 12px; }}
.tag {{
    display: inline-block; background: rgba(255,255,255,0.2);
    padding: 2px 10px; border-radius: 12px; font-size: 12px;
    margin-right: 8px; margin-top: 6px;
}}

/* Section */
.section {{
    background: var(--card); border-radius: 10px;
    border: 1px solid var(--border); margin-bottom: 24px; overflow: hidden;
}}
.section-header {{
    background: #F5F5F5; padding: 16px 24px;
    border-bottom: 1px solid var(--border);
    display: flex; justify-content: space-between; align-items: center;
}}
.section-header h2 {{ font-size: 18px; color: var(--accent); }}
.section-header .exp-tag {{
    background: var(--accent); color: white;
    padding: 2px 10px; border-radius: 4px; font-size: 12px; font-weight: 600;
}}
.section-body {{ padding: 24px; }}
.section-body p {{ margin-bottom: 12px; font-size: 14px; }}

/* Tables */
table {{
    width: 100%; border-collapse: collapse; margin: 16px 0; font-size: 13px;
}}
th, td {{ padding: 7px 10px; text-align: center; border: 1px solid #E8E8E8; }}
th {{ background: #F5F5F5; font-weight: 600; font-size: 12px; letter-spacing: 0.3px; }}
tr:hover {{ background: #FAFAFA; }}
.mean-row {{ font-weight: 700; background: #F5F5F5 !important; }}
.mean-row:hover {{ background: #EEEEEE !important; }}
td.best {{ color: var(--green); background: #E8F5E9; font-weight: 600; }}
td.worst {{ color: var(--red); background: #FFEBEE; }}

/* Feature set badges */
.fs-badge {{
    display: inline-block; padding: 1px 8px; border-radius: 4px;
    color: white; font-size: 11px; font-weight: 600;
}}
.fs-ae {{ background: var(--ae); }}
.fs-s2 {{ background: var(--s2); }}
.fs-fusion {{ background: var(--fusion); }}

/* Overview grid */
.overview-grid {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 14px; margin: 20px 0;
}}
.stat-card {{
    background: var(--card); border-radius: 10px;
    padding: 18px 14px; border: 1px solid var(--border); text-align: center;
}}
.stat-card .number {{ font-size: 28px; font-weight: 700; }}
.stat-card .label {{ font-size: 12px; color: var(--text2); margin-top: 4px; }}

/* Findings */
.finding {{
    border-left: 4px solid var(--accent); background: #E3F2FD;
    padding: 12px 16px; margin: 16px 0; border-radius: 0 6px 6px 0; font-size: 14px;
}}
.finding strong {{ color: var(--accent); }}
.finding.warning {{ background: #FFF3E0; border-left-color: #E65100; }}
.finding.warning strong {{ color: #E65100; }}
.finding.success {{ background: #E8F5E9; border-left-color: var(--green); }}
.finding.success strong {{ color: var(--green); }}
.finding.critical {{ background: #FCE4EC; border-left-color: var(--red); }}
.finding.critical strong {{ color: var(--red); }}

/* Sub-title */
.sub-title {{
    font-size: 15px; font-weight: 600; color: var(--text); margin: 20px 0 8px;
}}
th.col-ae {{ background: #E3F2FD; }}
th.col-s2 {{ background: #FFF3E0; }}
th.col-fusion {{ background: #E8F5E9; }}
.compact-table td, .compact-table th {{ padding: 5px 8px; font-size: 12px; }}

/* Figure container */
.fig-container {{ margin: 20px 0; text-align: center; }}
.fig-container img {{
    max-width: 100%; border-radius: 6px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}}
.fig-caption {{
    font-size: 12px; color: var(--text2); margin-top: 6px; font-style: italic;
}}

/* Two-column layout for figures */
.fig-row {{
    display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin: 16px 0;
}}
.fig-row.three {{ grid-template-columns: 1fr 1fr 1fr; }}
@media (max-width: 900px) {{
    .fig-row {{ grid-template-columns: 1fr; }}
    .fig-row.three {{ grid-template-columns: 1fr; }}
}}

/* TOC */
.toc {{
    background: #FAFAFA; border: 1px solid var(--border); border-radius: 8px;
    padding: 16px 24px; margin: 20px 0;
}}
.toc h3 {{ margin-bottom: 10px; color: var(--accent); }}
.toc ol {{ padding-left: 20px; }}
.toc li {{ padding: 4px 0; }}
.toc a {{ color: var(--accent); text-decoration: none; }}
.toc a:hover {{ text-decoration: underline; }}

footer {{
    text-align: center; padding: 30px 0 20px; color: var(--text2); font-size: 12px;
}}
</style>
</head>
<body>
<div class="container">
""")

    # === HEADER ===
    parts.append(f"""\
<header>
    <h1>AlphaEarth 泛欧作物分类 -- 完整实验报告</h1>
    <div class="subtitle">AlphaEarth vs Sentinel-2 跨国迁移能力探索</div>
    <div>
        <span class="tag">18 个欧洲国家</span>
        <span class="tag">10 类作物 (Tier B)</span>
        <span class="tag">5 个气候带</span>
        <span class="tag">3 种特征集</span>
        <span class="tag">7 组实验</span>
    </div>
    <p class="meta">生成时间: {now} | LightGBM + XGBoost | Balanced Accuracy 为主要指标 | 纯光谱特征 (nocoords)</p>
</header>
""")

    # === TOC ===
    parts.append("""\
<div class="toc">
    <h3>目录</h3>
    <ol>
        <li><a href="#sec0">研究问题与实验框架</a></li>
        <li><a href="#sec1">数据概览</a></li>
        <li><a href="#sec2">Exp 1: 域内 (ID) 基线</a></li>
        <li><a href="#sec3">Exp 0: 逐国交叉验证</a></li>
        <li><a href="#sec4">Exp 2: Leave-One-Country-Out (LOCO)</a></li>
        <li><a href="#sec5">Exp 2b: 国家对国家迁移矩阵 [KEY]</a></li>
        <li><a href="#sec6">Exp 3: 跨气候带迁移</a></li>
        <li><a href="#sec7">Exp 4: 标签效率</a></li>
        <li><a href="#sec8">Exp 5: 特征空间分析</a></li>
        <li><a href="#sec9">辅助信息消融 (coords vs no-coords)</a></li>
        <li><a href="#sec10">综合结论</a></li>
    </ol>
</div>
""")

    # ========================================
    # Section 0: 研究问题与实验框架
    # ========================================
    parts.append("""\
<div class="section" id="sec0">
    <div class="section-header">
        <h2>Section 0: 研究问题与实验框架</h2>
        <span class="exp-tag">DESIGN</span>
    </div>
    <div class="section-body">
        <div class="sub-title">科学问题</div>
        <p>Google AlphaEarth 基础模型嵌入是否能实现跨国作物分类迁移？与传统 Sentinel-2 光谱特征相比，在泛欧尺度下的分类能力和跨域泛化如何？</p>

        <div class="sub-title">实验框架</div>
        <table>
        <tr>
            <th>实验</th>
            <th>设计</th>
            <th>测试的问题</th>
        </tr>
        <tr><td><strong>Exp 0</strong></td><td>逐国 5-fold CV</td><td>各国独立分类能力（国内泛化）</td></tr>
        <tr><td><strong>Exp 1</strong></td><td>全数据 5-fold CV</td><td>混合训练的域内 (ID) 基线</td></tr>
        <tr><td><strong>Exp 2</strong></td><td>Leave-One-Country-Out (17->1)</td><td>跨国泛化 (OOD) 能力</td></tr>
        <tr><td><strong>Exp 2b</strong></td><td>18x18 国家对国家迁移矩阵 (1->1)</td><td>单国训练 -> 单国测试的迁移格局</td></tr>
        <tr><td><strong>Exp 3</strong></td><td>Leave-One-Zone-Out + 气候带迁移矩阵</td><td>跨气候带迁移能力</td></tr>
        <tr><td><strong>Exp 4</strong></td><td>学习曲线 + 混合比例</td><td>标签效率与少样本泛化</td></tr>
        <tr><td><strong>Exp 5</strong></td><td>CKA / SHAP / 国家可预测性</td><td>特征空间内在结构分析</td></tr>
        </table>

        <div class="sub-title">特征集定义</div>
        <table>
        <tr>
            <th>特征集</th>
            <th>维度 (纯光谱)</th>
            <th>说明</th>
        </tr>
        <tr>
            <td><span class="fs-badge fs-ae">AE</span> AlphaEarth</td>
            <td>128d</td>
            <td>64 原始波段 (A00-A63) 的均值 + 标准差</td>
        </tr>
        <tr>
            <td><span class="fs-badge fs-s2">S2</span> Sentinel-2</td>
            <td>168d</td>
            <td>10 光学波段 + 4 光谱指数, 季度中位数/标准差/均值等统计量</td>
        </tr>
        <tr>
            <td><span class="fs-badge fs-fusion">FUS</span> Fusion</td>
            <td>296d</td>
            <td>AE 128d + S2 168d 拼接</td>
        </tr>
        </table>

        <div class="finding">
            <strong>辅助信息消融：</strong>
            每种特征集均有「含坐标」(+longitude, latitude, log_area) 与「纯光谱」两个版本。
            本报告以纯光谱 (nocoords) 为主线，Section 9 专门讨论消融对比。
        </div>
    </div>
</div>
""")

    # ========================================
    # Section 1: 数据概览
    # ========================================
    # Build country sample table from exp0
    country_samples = {}
    for _, row in exp0_nc[exp0_nc.feature_type == 'ae'].iterrows():
        country_samples[row['country']] = int(row['n_samples'])

    country_rows = []
    for zone_key, zone_info in ZONES.items():
        for cc in zone_info['countries']:
            ns = country_samples.get(cc, 0)
            country_rows.append(f"<tr><td><strong>{cc}</strong></td>"
                                f"<td>{COUNTRY_NAMES[cc]}</td>"
                                f"<td>{zone_info['label']}</td>"
                                f"<td>{ns:,}</td></tr>")

    parts.append(f"""\
<div class="section" id="sec1">
    <div class="section-header">
        <h2>Section 1: 数据概览</h2>
        <span class="exp-tag">DATA</span>
    </div>
    <div class="section-body">
        <div class="overview-grid">
            <div class="stat-card">
                <div class="number" style="color:var(--accent)">18</div>
                <div class="label">欧洲国家</div>
            </div>
            <div class="stat-card">
                <div class="number" style="color:var(--fusion)">10</div>
                <div class="label">Tier B 作物类别</div>
            </div>
            <div class="stat-card">
                <div class="number" style="color:var(--s2)">5</div>
                <div class="label">气候带</div>
            </div>
            <div class="stat-card">
                <div class="number" style="color:var(--ae)">~{total_samples//1000}k</div>
                <div class="label">总样本数</div>
            </div>
        </div>

        <div class="sub-title">国家与气候带分布</div>
        <table class="compact-table">
        <tr><th>代码</th><th>国家</th><th>气候带</th><th>样本数</th></tr>
        {"".join(country_rows)}
        </table>

        <div class="sub-title">Tier B 作物类别 (10 类)</div>
        <p>grassland, maize, orchards, other_crops, potatoes, root_crops, strawberries, sunflower, vineyards, winter_wheat</p>
    </div>
</div>
""")

    # ========================================
    # Section 2: Exp 1 — ID 基线
    # ========================================
    exp1_rows = []
    for ft in FEATURE_SETS:
        row = exp1_nc[exp1_nc.feature_type == ft].iloc[0]
        ba = row['BA_mean']
        f1 = row['F1_mean']
        kp = row['Kappa_mean']
        ba_s = row['BA_std']
        f1_s = row['F1_std']
        kp_s = row['Kappa_std']
        best_ba = "best" if ft == 'fusion' else ""
        exp1_rows.append(
            f'<tr><td><span class="fs-badge fs-{ft}">{FS_LABELS[ft]}</span></td>'
            f'<td>{FS_DIMS[ft]}</td>'
            f'<td class="{best_ba}">{pct(ba)} +/-{pct(ba_s)}</td>'
            f'<td>{pct(f1)} +/-{pct(f1_s)}</td>'
            f'<td>{pct(kp)} +/-{pct(kp_s)}</td></tr>'
        )

    parts.append(f"""\
<div class="section" id="sec2">
    <div class="section-header">
        <h2>Section 2: Exp 1 -- 域内 (ID) 基线</h2>
        <span class="exp-tag">EXP 1</span>
    </div>
    <div class="section-body">
        <p>18 国合并数据集，5-fold CV，纯光谱特征。</p>

        <table>
        <tr>
            <th>特征集</th>
            <th>维度</th>
            <th>Balanced Accuracy (%)</th>
            <th>Macro F1 (%)</th>
            <th>Cohen's Kappa (%)</th>
        </tr>
        {"".join(exp1_rows)}
        </table>

        <div class="fig-container">
            {img_tag(fig_exp1, 'Exp 1 Bar Chart', '70%')}
            <div class="fig-caption">图 1: ID 基线 -- 三种特征集在域内 5-fold CV 的表现</div>
        </div>

        <div class="finding success">
            <strong>发现：</strong>
            排序: Fusion ({pct(fu_id_ba)}%) > AE ({pct(ae_id_ba)}%) > S2 ({pct(s2_id_ba)}%)。
            AE 单独即超越 S2 约 {pct1(ae_id_ba - s2_id_ba)} pp，Fusion 进一步提升。
        </div>
    </div>
</div>
""")

    # ========================================
    # Section 3: Exp 0 — 逐国 CV
    # ========================================
    exp0_rows = []
    means = {ft: [] for ft in FEATURE_SETS}
    for cc in COUNTRIES:
        cells = [f"<td><strong>{cc}</strong> {COUNTRY_NAMES[cc]}</td>"]
        row_vals = {}
        for ft in FEATURE_SETS:
            sub = exp0_nc[(exp0_nc.country == cc) & (exp0_nc.feature_type == ft)]
            if len(sub) > 0:
                ba = sub.iloc[0]['BA_mean']
            else:
                ba = 0
            row_vals[ft] = ba
            means[ft].append(ba)
        # find best
        best_ft = max(row_vals, key=row_vals.get)
        for ft in FEATURE_SETS:
            cls = "best" if ft == best_ft else ""
            cells.append(f'<td class="{cls}">{pct(row_vals[ft])}</td>')
        exp0_rows.append(f"<tr>{''.join(cells)}</tr>")

    # mean row
    mean_cells = ['<td class="mean-row"><strong>均值</strong></td>']
    for ft in FEATURE_SETS:
        m = np.mean(means[ft])
        mean_cells.append(f'<td class="mean-row">{pct(m)}</td>')
    exp0_rows.append(f'<tr class="mean-row">{"".join(mean_cells)}</tr>')

    parts.append(f"""\
<div class="section" id="sec3">
    <div class="section-header">
        <h2>Section 3: Exp 0 -- 逐国交叉验证</h2>
        <span class="exp-tag">EXP 0</span>
    </div>
    <div class="section-body">
        <p>每个国家独立 5-fold CV 的 Balanced Accuracy，绿色标记为该国最优特征集。</p>

        <table class="compact-table">
        <tr>
            <th>国家</th>
            <th class="col-ae">AE (%)</th>
            <th class="col-s2">S2 (%)</th>
            <th class="col-fusion">Fusion (%)</th>
        </tr>
        {"".join(exp0_rows)}
        </table>

        <div class="fig-container">
            {img_tag(fig_exp0, 'Exp 0 Country Bars', '100%')}
            <div class="fig-caption">图 2: 逐国 5-fold CV Balanced Accuracy 对比</div>
        </div>

        <div class="finding">
            <strong>发现：</strong>
            国家间难度差异显著；HR（克罗地亚）表现最佳，LV（拉脱维亚）最难。
            AE 在绝大多数国家优于 S2，Fusion 几乎总是最优。
        </div>
    </div>
</div>
""")

    # ========================================
    # Section 4: Exp 2 — LOCO (17->1)
    # ========================================
    loco_rows = []
    for cc in COUNTRIES:
        ae_ba = ae_loco[ae_loco.test_country == cc]['balanced_accuracy'].values[0] if cc in ae_loco.test_country.values else 0
        s2_ba = s2_loco[s2_loco.test_country == cc]['balanced_accuracy'].values[0] if cc in s2_loco.test_country.values else 0
        fu_ba = fu_loco[fu_loco.test_country == cc]['balanced_accuracy'].values[0] if cc in fu_loco.test_country.values else 0
        vals = {'ae': ae_ba, 's2': s2_ba, 'fusion': fu_ba}
        best_ft = max(vals, key=vals.get)
        cells = [f"<td><strong>{cc}</strong> {COUNTRY_NAMES[cc]}</td>"]
        for ft in FEATURE_SETS:
            cls = "best" if ft == best_ft else ""
            cells.append(f'<td class="{cls}">{pct(vals[ft])}</td>')
        loco_rows.append(f"<tr>{''.join(cells)}</tr>")

    # mean row
    mean_cells = [f'<td class="mean-row"><strong>均值</strong></td>']
    for ft, m in [('ae', ae_ood_mean), ('s2', s2_ood_mean), ('fusion', fu_ood_mean)]:
        mean_cells.append(f'<td class="mean-row">{pct(m)}</td>')
    loco_rows.append(f'<tr class="mean-row">{"".join(mean_cells)}</tr>')

    # ID-OOD gap
    ae_gap = ae_id_ba - ae_ood_mean
    s2_gap = s2_id_ba - s2_ood_mean
    fu_gap = fu_id_ba - fu_ood_mean

    parts.append(f"""\
<div class="section" id="sec4">
    <div class="section-header">
        <h2>Section 4: Exp 2 -- Leave-One-Country-Out (LOCO)</h2>
        <span class="exp-tag">EXP 2</span>
    </div>
    <div class="section-body">
        <p>在其他 17 国训练，目标国测试。评估跨国泛化 (OOD) 能力。</p>

        <table class="compact-table">
        <tr>
            <th>测试国</th>
            <th class="col-ae">AE (%)</th>
            <th class="col-s2">S2 (%)</th>
            <th class="col-fusion">Fusion (%)</th>
        </tr>
        {"".join(loco_rows)}
        </table>

        <div class="fig-row">
            <div class="fig-container">
                {img_tag(fig_exp2_loco, 'Exp 2 LOCO', '100%')}
                <div class="fig-caption">图 3: LOCO 逐国 OOD BA</div>
            </div>
            <div class="fig-container">
                {img_tag(fig_exp2_gap, 'Exp 2 ID vs OOD', '100%')}
                <div class="fig-caption">图 4: ID vs OOD 性能差距</div>
            </div>
        </div>

        <div class="finding warning">
            <strong>发现：</strong>
            ID -> OOD 性能下降约 ~22 pp (AE: {pct1(ae_gap)} pp, S2: {pct1(s2_gap)} pp, Fusion: {pct1(fu_gap)} pp)。
            Fusion > AE > S2 的排序在 OOD 中保持一致。
        </div>
    </div>
</div>
""")

    # ========================================
    # Section 5: Exp 2b — 国家对国家迁移矩阵 [KEY]
    # ========================================
    # Summary table of mean BA
    exp2b_summary_rows = []
    for ft in FEATURE_SETS:
        lgbm_m = exp2b_means_lgbm[ft]
        xgb_m = exp2b_means_xgb[ft]
        stats = exp2b_train_stats[ft]
        exp2b_summary_rows.append(
            f'<tr><td><span class="fs-badge fs-{ft}">{FS_LABELS[ft]}</span></td>'
            f'<td>{pct(lgbm_m)}</td>'
            f'<td>{pct(xgb_m)}</td>'
            f'<td>{stats["best_train"]} ({pct(stats["best_train_ba"])}%)</td>'
            f'<td>{stats["worst_test"]} ({pct(stats["worst_test_ba"])}%)</td></tr>'
        )

    # Embed existing PNGs
    def maybe_img(key, alt, width="100%"):
        if key in exp2b_pngs:
            return img_tag(exp2b_pngs[key], alt, width)
        return f'<p style="color:gray;text-align:center;">[{alt} -- 文件未找到]</p>'

    parts.append(f"""\
<div class="section" id="sec5">
    <div class="section-header">
        <h2>Section 5: Exp 2b -- 国家对国家迁移矩阵</h2>
        <span class="exp-tag">EXP 2b [KEY]</span>
    </div>
    <div class="section-body">
        <p><strong>实验设计：</strong>18x18 的完整迁移矩阵，每个国家分别作为训练集，预测其余 17 个国家。
        共 18x17 = 306 组迁移实验，分别用 LightGBM 和 XGBoost 运行。</p>

        <div class="sub-title">摘要统计</div>
        <table>
        <tr>
            <th>特征集</th>
            <th>LGBM Mean BA (%)</th>
            <th>XGB Mean BA (%)</th>
            <th>最佳训练国</th>
            <th>最难预测国</th>
        </tr>
        {"".join(exp2b_summary_rows)}
        </table>

        <div class="sub-title">迁移矩阵热力图 (LightGBM)</div>
        <div class="fig-row three">
            <div class="fig-container">
                {maybe_img('transfer_matrix_ae_nocoords.png', 'AE Transfer Matrix')}
                <div class="fig-caption">AlphaEarth 迁移矩阵</div>
            </div>
            <div class="fig-container">
                {maybe_img('transfer_matrix_s2_nocoords.png', 'S2 Transfer Matrix')}
                <div class="fig-caption">Sentinel-2 迁移矩阵</div>
            </div>
            <div class="fig-container">
                {maybe_img('transfer_matrix_fusion_nocoords.png', 'Fusion Transfer Matrix')}
                <div class="fig-caption">Fusion 迁移矩阵</div>
            </div>
        </div>

        <div class="sub-title">迁移对称性分析</div>
        <div class="fig-row">
            <div class="fig-container">
                {maybe_img('transfer_symmetry.png', 'Transfer Symmetry', '100%')}
                <div class="fig-caption">图: A->B vs B->A 迁移对称性散点图</div>
            </div>
        </div>

        <div class="sub-title">LightGBM vs XGBoost 对比</div>
        <div class="fig-row three">
            <div class="fig-container">
                {maybe_img('train_generalization_lgbm_vs_xgb.png', 'LGBM vs XGB bars', '100%')}
                <div class="fig-caption">训练国泛化能力对比</div>
            </div>
            <div class="fig-container">
                {maybe_img('lgbm_vs_xgb_scatter.png', 'LGBM vs XGB scatter', '100%')}
                <div class="fig-caption">306 组迁移结果散点对比</div>
            </div>
            <div class="fig-container">
                {maybe_img('winner_map_xgb_vs_lgbm.png', 'Winner Map', '100%')}
                <div class="fig-caption">XGB vs LGBM 胜负图</div>
            </div>
        </div>

        <div class="finding success">
            <strong>核心发现：</strong>
            <ul style="margin-top:8px;padding-left:20px;">
                <li><strong>DE（德国）</strong>是所有三种特征集中泛化能力最强的训练国（"hub" 国家）</li>
                <li><strong>PT（葡萄牙）</strong>是最难被其他国家预测的目标国</li>
                <li>同气候带内的迁移效果普遍优于跨气候带迁移</li>
                <li>AE 具有更高的迁移对称性 (r=0.468 vs S2 r=0.205)，说明 AE 嵌入编码了更对等的作物语义</li>
            </ul>
        </div>

        <div class="finding">
            <strong>LightGBM vs XGBoost：</strong>
            XGBoost 在所有特征集上一致优于 LightGBM 约 ~1 pp，但排序结论完全一致。
            迁移格局（最优训练国、最难测试国）在两种分类器间高度稳定。
        </div>
    </div>
</div>
""")

    # ========================================
    # Section 6: Exp 3 — 跨气候带迁移
    # ========================================
    # LOZO table
    ae_lozo = load_csv('exp3_climate_zone', 'lozo_ae_nocoords_tier_b.csv')
    s2_lozo = load_csv('exp3_climate_zone', 'lozo_s2_nocoords_tier_b.csv')
    fu_lozo = load_csv('exp3_climate_zone', 'lozo_fusion_nocoords_tier_b.csv')

    lozo_rows = []
    for _, ae_row in ae_lozo.iterrows():
        zone = ae_row['test_zone']
        ae_ba = ae_row['balanced_accuracy']
        s2_ba = s2_lozo[s2_lozo.test_zone == zone]['balanced_accuracy'].values[0]
        fu_ba = fu_lozo[fu_lozo.test_zone == zone]['balanced_accuracy'].values[0]
        vals = {'ae': ae_ba, 's2': s2_ba, 'fusion': fu_ba}
        best_ft = max(vals, key=vals.get)
        zone_cn = ZONES.get(zone, {}).get('label', zone)
        cells = [f"<td><strong>{zone}</strong> ({zone_cn})</td>"]
        for ft in FEATURE_SETS:
            cls = "best" if ft == best_ft else ""
            cells.append(f'<td class="{cls}">{pct(vals[ft])}</td>')
        lozo_rows.append(f"<tr>{''.join(cells)}</tr>")

    # mean row
    mean_cells = [f'<td class="mean-row"><strong>均值</strong></td>']
    for lozo_df in [ae_lozo, s2_lozo, fu_lozo]:
        mean_cells.append(f'<td class="mean-row">{pct(lozo_df["balanced_accuracy"].mean())}</td>')
    lozo_rows.append(f'<tr class="mean-row">{"".join(mean_cells)}</tr>')

    parts.append(f"""\
<div class="section" id="sec6">
    <div class="section-header">
        <h2>Section 6: Exp 3 -- 跨气候带迁移</h2>
        <span class="exp-tag">EXP 3</span>
    </div>
    <div class="section-body">
        <p>Leave-One-Zone-Out (LOZO): 留出一个气候带测试，用其余 4 个气候带训练。</p>

        <table>
        <tr>
            <th>测试气候带</th>
            <th class="col-ae">AE (%)</th>
            <th class="col-s2">S2 (%)</th>
            <th class="col-fusion">Fusion (%)</th>
        </tr>
        {"".join(lozo_rows)}
        </table>

        <div class="fig-container">
            {img_tag(fig_exp3_lozo, 'Exp 3 LOZO', '80%')}
            <div class="fig-caption">图: LOZO 各气候带 OOD BA</div>
        </div>

        <div class="sub-title">气候带间迁移矩阵</div>
        <div class="fig-row three">
            <div class="fig-container">
                {img_tag(fig_exp3_tm['ae'], 'Exp 3 TM AE')}
                <div class="fig-caption">AlphaEarth</div>
            </div>
            <div class="fig-container">
                {img_tag(fig_exp3_tm['s2'], 'Exp 3 TM S2')}
                <div class="fig-caption">Sentinel-2</div>
            </div>
            <div class="fig-container">
                {img_tag(fig_exp3_tm['fusion'], 'Exp 3 TM Fusion')}
                <div class="fig-caption">Fusion</div>
            </div>
        </div>

        <div class="finding">
            <strong>发现：</strong>
            大西洋气候带 (atlantic) 泛化性最强；
            在纯光谱特征下，北欧-波罗的海区域改善最明显（坐标不再泄漏地理位置信息）。
        </div>
    </div>
</div>
""")

    # ========================================
    # Section 7: Exp 4 — 标签效率
    # ========================================
    # Learning curve table
    lc_rows = []
    ae_curve = exp4_json['ae_curve']['sample_sizes']
    s2_curve = exp4_json['s2_curve']['sample_sizes']
    for sk in CURVE_ORDER:
        if sk in ae_curve and sk in s2_curve:
            ae_ba = ae_curve[sk]['ba_mean']
            s2_ba = s2_curve[sk]['ba_mean']
            n_train = ae_curve[sk]['n_train_actual']
            label = CURVE_LABELS.get(sk, sk)
            delta = ae_ba - s2_ba
            cls = "best" if delta > 0 else ""
            lc_rows.append(
                f'<tr><td>{label}</td><td>{n_train:,}</td>'
                f'<td>{pct(ae_ba)}</td><td>{pct(s2_ba)}</td>'
                f'<td class="{cls}">{delta_str(delta)}</td></tr>'
            )

    # Mixed scenario table
    mix_rows = []
    ae_mix = exp4_json['ae_mixed']
    s2_mix = exp4_json['s2_mixed']
    for k in sorted([k for k in ae_mix.keys() if k not in ('feature_type', 'tier', 'test_countries')],
                     key=lambda x: int(x)):
        ae_ba = ae_mix[k]['balanced_accuracy']
        s2_ba = s2_mix[k]['balanced_accuracy']
        delta = ae_ba - s2_ba
        cls = "best" if delta > 0 else ""
        mix_rows.append(
            f'<tr><td>{k}%</td>'
            f'<td>{pct(ae_ba)}</td><td>{pct(s2_ba)}</td>'
            f'<td class="{cls}">{delta_str(delta)}</td></tr>'
        )

    parts.append(f"""\
<div class="section" id="sec7">
    <div class="section-header">
        <h2>Section 7: Exp 4 -- 标签效率</h2>
        <span class="exp-tag">EXP 4</span>
    </div>
    <div class="section-body">
        <p>在 OOD 设置下，不同训练样本量的学习曲线（AE vs S2）。</p>

        <div class="sub-title">学习曲线</div>
        <table>
        <tr>
            <th>标签</th><th>训练样本数</th>
            <th class="col-ae">AE BA (%)</th>
            <th class="col-s2">S2 BA (%)</th>
            <th>AE-S2 (pp)</th>
        </tr>
        {"".join(lc_rows)}
        </table>

        <div class="fig-container">
            {img_tag(fig_exp4_lc, 'Exp 4 Learning Curve', '70%')}
            <div class="fig-caption">图: OOD 学习曲线</div>
        </div>

        <div class="sub-title">混合场景 (源域 + 目标域标注)</div>
        <table>
        <tr>
            <th>目标域混入比例</th>
            <th class="col-ae">AE BA (%)</th>
            <th class="col-s2">S2 BA (%)</th>
            <th>AE-S2 (pp)</th>
        </tr>
        {"".join(mix_rows)}
        </table>

        <div class="fig-container">
            {img_tag(fig_exp4_mix, 'Exp 4 Mixed Scenario', '60%')}
            <div class="fig-caption">图: 混合场景 -- 随目标域标注增加的表现</div>
        </div>

        <div class="finding">
            <strong>发现：</strong>
            AE 在少样本设置下一致优于 S2，表明 AE 嵌入具有更高的标签效率。
        </div>
    </div>
</div>
""")

    # ========================================
    # Section 8: Exp 5 — 特征空间分析
    # ========================================
    # CKA table (nocoords)
    cka_jump_ae = ae_nc_crop / ae_c_crop if ae_c_crop > 0 else float('inf')
    cka_ratio_ae_nc = ae_nc_crop / s2_nc_crop if s2_nc_crop > 0 else float('inf')

    parts.append(f"""\
<div class="section" id="sec8">
    <div class="section-header">
        <h2>Section 8: Exp 5 -- 特征空间分析</h2>
        <span class="exp-tag">EXP 5</span>
    </div>
    <div class="section-body">
        <div class="sub-title">CKA 特征-标签对齐</div>
        <p>Linear CKA 衡量嵌入空间与标签的对齐程度。CKA_crop 越高代表特征越能区分作物类别。</p>

        <table>
        <tr>
            <th rowspan="2">特征集</th>
            <th colspan="2">含坐标 (coords)</th>
            <th colspan="2">纯光谱 (nocoords)</th>
            <th>CKA_crop 变化</th>
        </tr>
        <tr>
            <th>CKA_crop</th><th>CKA_country</th>
            <th>CKA_crop</th><th>CKA_country</th>
            <th>倍数</th>
        </tr>
        <tr>
            <td><span class="fs-badge fs-ae">AE</span></td>
            <td>{ae_c_crop:.4f}</td>
            <td>{cka_c[cka_c.feature_type == 'ae']['cka_country'].values[0]:.4f}</td>
            <td class="best">{ae_nc_crop:.4f}</td>
            <td>{ae_nc_ctry:.4f}</td>
            <td class="best">{cka_jump_ae:.1f}x</td>
        </tr>
        <tr>
            <td><span class="fs-badge fs-s2">S2</span></td>
            <td>{s2_c_crop:.4f}</td>
            <td>{cka_c[cka_c.feature_type == 's2']['cka_country'].values[0]:.4f}</td>
            <td>{s2_nc_crop:.4f}</td>
            <td>{s2_nc_ctry:.4f}</td>
            <td>{s2_nc_crop / s2_c_crop:.1f}x</td>
        </tr>
        </table>

        <div class="finding critical">
            <strong>关键发现 -- CKA_crop 跳跃：</strong>
            AE 去坐标后 CKA_crop 从 {ae_c_crop:.4f} 跃升至 {ae_nc_crop:.4f}（<strong>{cka_jump_ae:.1f}x</strong>），
            说明坐标严重掩盖了 AE 嵌入中的作物语义信息。
            纯光谱 AE 的 CKA_crop 是 S2 的 <strong>{cka_ratio_ae_nc:.1f}x</strong>。
        </div>

        <div class="fig-container">
            {img_tag(fig_exp5_cka, 'Exp 5 CKA', '80%')}
            <div class="fig-caption">图: CKA 对比 -- coords vs nocoords</div>
        </div>

        <div class="sub-title">SHAP 特征重要性 (ID vs OOD)</div>
        <div class="fig-row">
            <div class="fig-container">
                {img_tag(fig_exp5_shap_ae, 'SHAP AE', '100%')}
                <div class="fig-caption">AlphaEarth SHAP Top 15</div>
            </div>
            <div class="fig-container">
                {img_tag(fig_exp5_shap_s2, 'SHAP S2', '100%')}
                <div class="fig-caption">Sentinel-2 SHAP Top 15</div>
            </div>
        </div>

        <div class="sub-title">国家可预测性</div>
        <p>用纯光谱特征预测样本所在国家的 BA，衡量嵌入编码了多少地理/环境信息。</p>

        <div class="fig-container">
            {img_tag(fig_exp5_cpred, 'Country Prediction', '40%')}
            <div class="fig-caption">图: 国家可预测性 (纯光谱)</div>
        </div>

        <div class="finding">
            <strong>发现：</strong>
            AE 纯光谱的国家可预测性 ({pct1(ae_cpred)}%) 远高于 S2 ({pct1(s2_cpred)}%)。
            说明即使去除坐标，AE 嵌入仍编码了丰富的环境/地理信息 --
            这既是优势（更丰富的特征），也是潜在挑战（地理偏差的来源）。
        </div>
    </div>
</div>
""")

    # ========================================
    # Section 9: 辅助信息消融
    # ========================================
    # Build comparison table: coords vs nocoords for Exp 1, 2
    exp1_c_data = load_csv('exp1_id_baseline', 'summary_tier_b.csv')
    ablation_rows = []
    for ft in FEATURE_SETS:
        c_ba = exp1_c_data[exp1_c_data.feature_type == ft]['BA_mean'].values[0]
        nc_ba = exp1_nc[exp1_nc.feature_type == ft]['BA_mean'].values[0]
        d_id = nc_ba - c_ba

        # OOD from all_data
        ood_c_ba = all_data['exp2'][f'coords_{ft}']['mean_ba']
        ood_nc_ba = all_data['exp2'][f'nocoords_{ft}']['mean_ba']
        d_ood = ood_nc_ba - ood_c_ba

        ablation_rows.append(
            f'<tr><td><span class="fs-badge fs-{ft}">{FS_LABELS[ft]}</span></td>'
            f'<td>{pct(c_ba)}</td><td>{pct(nc_ba)}</td>'
            f'<td style="color:{"var(--red)" if d_id < 0 else "var(--green)"}">{delta_str(d_id)}</td>'
            f'<td>{pct(ood_c_ba)}</td><td>{pct(ood_nc_ba)}</td>'
            f'<td style="color:{"var(--green)" if d_ood > 0 else "var(--red)"};font-weight:600">{delta_str(d_ood)}</td>'
            f'</tr>'
        )

    parts.append(f"""\
<div class="section" id="sec9">
    <div class="section-header">
        <h2>Section 9: 辅助信息消融 (coords vs no-coords)</h2>
        <span class="exp-tag">ABLATION</span>
    </div>
    <div class="section-body">
        <p>对比含坐标（longitude, latitude, log_area）与纯光谱特征在 ID 和 OOD 设置下的表现差异。</p>

        <table>
        <tr>
            <th rowspan="2">特征集</th>
            <th colspan="3">ID (Exp 1, 5-Fold CV) BA %</th>
            <th colspan="3">OOD (Exp 2 LOCO, Mean) BA %</th>
        </tr>
        <tr>
            <th>Coords</th><th>NoCoords</th><th>Delta</th>
            <th>Coords</th><th>NoCoords</th><th>Delta</th>
        </tr>
        {"".join(ablation_rows)}
        </table>

        <div class="finding warning">
            <strong>核心发现：</strong>
            坐标在 ID 场景下帮助所有特征集（S2 受益最大），
            但在 OOD 场景下<strong>阻碍泛化</strong> -- 去坐标后 OOD 性能普遍提升。
            这是因为坐标让模型过拟合于训练集的地理分布，在面对新国家时反而产生误导。
        </div>
    </div>
</div>
""")

    # ========================================
    # Section 10: 综合结论
    # ========================================
    parts.append(f"""\
<div class="section" id="sec10">
    <div class="section-header">
        <h2>Section 10: 综合结论</h2>
        <span class="exp-tag">CONCLUSION</span>
    </div>
    <div class="section-body">
        <div class="fig-container">
            {img_tag(fig_summary, 'Summary', '80%')}
            <div class="fig-caption">图: 纯光谱特征在 ID/LOCO/LOZO 三种设置下的性能综合</div>
        </div>

        <div class="sub-title">核心结论</div>

        <div class="finding success">
            <strong>1. 性能排序：</strong>
            在所有实验设置（ID / LOCO / LOZO / 逐国 / 标签效率）中，
            <strong>Fusion > AE > S2</strong> 的排序一致成立。
        </div>

        <div class="finding success">
            <strong>2. AE 嵌入编码了更丰富的作物语义：</strong>
            纯光谱 AE 的 CKA_crop ({ae_nc_crop:.4f}) 是 S2 ({s2_nc_crop:.4f}) 的 {cka_ratio_ae_nc:.1f} 倍。
            AE 作为基础模型的预训练，确实抽象出了超越原始光谱的作物区分能力。
        </div>

        <div class="finding success">
            <strong>3. AE 具有更高的迁移对称性：</strong>
            在 Exp 2b 国家对国家迁移中，AE 的 A->B 与 B->A 对称性 (r=0.468)
            显著高于 S2 (r=0.205)，说明 AE 特征空间中国家间的距离更加「对等」。
        </div>

        <div class="finding warning">
            <strong>4. 坐标掩盖光谱信号：</strong>
            含坐标训练时 AE 的 CKA_crop 仅 {ae_c_crop:.4f}，去坐标后跃升至 {ae_nc_crop:.4f}（{cka_jump_ae:.1f}x）。
            坐标为 ID 分类提供「捷径」，但严重削弱 OOD 泛化。
        </div>

        <div class="finding">
            <strong>5. DE 是最佳「枢纽」训练国：</strong>
            在 Exp 2b 中，德国训练的模型对其他国家的平均预测能力最强，
            可能与其处于欧洲地理中心、农业类型多样有关。
        </div>

        <div class="finding">
            <strong>6. 实践意义：</strong>
            对于泛欧作物制图的实际应用，建议：
            (a) 优先使用 Fusion 特征集；
            (b) 训练时去除坐标以保证跨域泛化；
            (c) 以德国等大型多样化国家为核心训练集；
            (d) AE 嵌入在少样本场景下更有价值（标签效率更高）。
        </div>
    </div>
</div>
""")

    # === FOOTER ===
    parts.append(f"""\
<footer>
    <p>AlphaEarth Pan-European Crop Classification -- Full Experiment Report</p>
    <p>Generated: {now} | Python + LightGBM + XGBoost | Tier B (10 classes) | 18 EU Countries</p>
</footer>
</div>
</body>
</html>
""")

    html = "\n".join(parts)

    # --- 写入文件 ---
    outpath = os.path.join(RESULTS_DIR, 'full_report.html')
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, 'w', encoding='utf-8') as f:
        f.write(html)

    size_kb = os.path.getsize(outpath) / 1024
    print(f"[DONE] Report saved to: {outpath} ({size_kb:.0f} KB)")
    return outpath


if __name__ == '__main__':
    build_report()
