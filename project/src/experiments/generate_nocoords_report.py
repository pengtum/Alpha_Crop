#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成纯光谱特征（无辅助信息）的完整独立 HTML 报告。

包含：实验设计 → 数据概览 → Exp 0-5 全部结果 → 内嵌图表 → 分析讨论

Usage:
    python -m src.experiments.generate_nocoords_report
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

# ─── Matplotlib 全局设置 ───────────────────────────────────────────
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

# ─── 常量 ────────────────────────────────────────────────────────
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

CURVE_LABELS = {
    "10": "100", "50": "500", "100": "1,000",
    "500": "5,000", "1000": "10,000", "5000": "~46k", "0": "全部 (~795k)",
}
CURVE_ORDER = ["10", "50", "100", "500", "1000", "5000", "0"]


# ─── 数据加载 ────────────────────────────────────────────────────
def load_csv(exp, filename):
    return pd.read_csv(os.path.join(RESULTS_DIR, exp, filename))


def load_json(exp, filename):
    with open(os.path.join(RESULTS_DIR, exp, filename)) as f:
        return json.load(f)


# ─── 图表工具 ────────────────────────────────────────────────────
def fig_to_base64(fig):
    """将 matplotlib figure 转为 base64 PNG 字符串。"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def img_tag(b64, alt="figure", width="100%"):
    return f'<img src="data:image/png;base64,{b64}" alt="{alt}" style="max-width:{width};display:block;margin:12px auto;">'


# ─── HTML 工具 ────────────────────────────────────────────────────
def pct(v):
    return f"{v * 100:.2f}"

def pct1(v):
    return f"{v * 100:.1f}"

def delta_str(d):
    sign = "+" if d >= 0 else ""
    return f"{sign}{d * 100:.2f}"


# ═══════════════════════════════════════════════════════════════════
# 图表生成函数
# ═══════════════════════════════════════════════════════════════════

def plot_exp1_nocoords():
    """Exp 1: ID Baseline 柱状图 — 纯光谱。"""
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
    ax.set_title('Exp 1: ID Baseline — Pure Spectral Features (Tier B, 5-Fold CV)')
    ax.legend(frameon=True, fancybox=False, edgecolor='gray')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    fig.tight_layout()
    return fig_to_base64(fig)


def plot_exp0_country_bars():
    """Exp 0: 逐国 CV BA 柱状图 — 三种特征集对比。"""
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
    ax.set_title('Exp 0: Per-Country 5-Fold CV — Pure Spectral Features')
    ax.legend(frameon=True, fancybox=False, edgecolor='gray')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 0.85)
    fig.tight_layout()
    return fig_to_base64(fig)


def plot_exp0_heatmap_diff():
    """Exp 0: AE vs S2 F1 差异热力图。"""
    # 尝试加载逐类 CSV（如果存在的话）
    ae_csv = os.path.join(RESULTS_DIR, 'exp0_per_country_cv', 'per_class_ae_nocoords_tier_b.csv')
    s2_csv = os.path.join(RESULTS_DIR, 'exp0_per_country_cv', 'per_class_s2_nocoords_tier_b.csv')
    if not (os.path.exists(ae_csv) and os.path.exists(s2_csv)):
        return None  # 如果没有逐类数据则跳过

    ae_df = pd.read_csv(ae_csv)
    s2_df = pd.read_csv(s2_csv)
    ae_pivot = ae_df.pivot_table(index='country', columns='class', values='f1', aggfunc='mean')
    s2_pivot = s2_df.pivot_table(index='country', columns='class', values='f1', aggfunc='mean')
    common_classes = sorted(set(ae_pivot.columns) & set(s2_pivot.columns))
    common_countries = sorted(set(ae_pivot.index) & set(s2_pivot.index))
    diff = (ae_pivot.loc[common_countries, common_classes].values -
            s2_pivot.loc[common_countries, common_classes].values)

    fig, ax = plt.subplots(figsize=(max(8, len(common_classes) * 0.9),
                                    max(5, len(common_countries) * 0.35)))
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
    ax.set_title('AE − S2 F1 Difference (Pure Spectral, Tier B)')
    for i in range(len(common_countries)):
        for j in range(len(common_classes)):
            val = diff[i, j]
            if not np.isnan(val):
                ax.text(j, i, f'{val:+.2f}', ha='center', va='center',
                        fontsize=7, color='black')
    plt.colorbar(im, ax=ax, shrink=0.8, label='ΔF1 (AE − S2)')
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
    ax.set_title('Exp 2: Leave-One-Country-Out — OOD Performance (Pure Spectral)')
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
        ax.annotate(f'Δ={gap:.3f}', xy=(i + w / 2, ood_vals[i]),
                    xytext=(i + 0.55, mid_y), fontsize=8, ha='left', color='red',
                    arrowprops=dict(arrowstyle='->', color='red', lw=0.8))

    ax.set_xticks(x)
    ax.set_xticklabels(labels_ft)
    ax.set_ylabel('Balanced Accuracy')
    ax.set_title('ID vs OOD Performance Gap (Pure Spectral)')
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
    ax.set_title('Exp 3: Leave-One-Zone-Out (LOZO) — Pure Spectral')
    ax.legend(frameon=True, fancybox=False, edgecolor='gray')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 0.7)
    fig.tight_layout()
    return fig_to_base64(fig)


def plot_exp3_transfer_matrices():
    """Exp 3: 跨气候带迁移矩阵热力图 (AE + S2 + Fusion 并列)。"""
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
        ax.set_title(f'Transfer Matrix — {ft_label} (Pure Spectral)')

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
    ax.set_title('Exp 4: Label Efficiency — OOD Learning Curve (Pure Spectral)')
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
    ax.set_title('Exp 4b: Mixed Scenario — Source + Target (Pure Spectral)')
    ax.legend(frameon=True, fancybox=False, edgecolor='gray')
    ax.grid(alpha=0.3, linestyle='--')
    fig.tight_layout()
    return fig_to_base64(fig)


def plot_exp5_cka():
    """Exp 5: CKA 对比 (crop vs country)。"""
    data = load_json('exp5_feature_analysis', 'tier_b_nocoords.json')

    fig, ax = plt.subplots(figsize=(5, 4))
    x = np.arange(2)
    w = 0.35
    cka_crop = [data['ae']['cka']['cka_crop'], data['s2']['cka']['cka_crop']]
    cka_country = [data['ae']['cka']['cka_country'], data['s2']['cka']['cka_country']]

    b1 = ax.bar(x - w / 2, cka_crop, w, label='CKA(Embedding, Crop)', color='#66BB6A')
    b2 = ax.bar(x + w / 2, cka_country, w, label='CKA(Embedding, Country)', color='#EF5350')

    for bar_group in [b1, b2]:
        for bar in bar_group:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005, f'{h:.3f}',
                    ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(['AlphaEarth', 'Sentinel-2'])
    ax.set_ylabel('Linear CKA')
    ax.set_title('Exp 5: Feature–Label Alignment (CKA) — Pure Spectral')
    ax.legend(frameon=True, fancybox=False, edgecolor='gray')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
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
    ax.set_title(f'SHAP Feature Importance — {ft_label} (ID vs OOD, Pure Spectral)')
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
    ax.set_title('Country Predictability (Pure Spectral)')
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    fig.tight_layout()
    return fig_to_base64(fig)


def plot_summary_overview():
    """综合概览：ID (Exp 1) + OOD (Exp 2 Mean) + LOZO Mean 三组对比。"""
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
    ax.set_title('Performance Summary — Pure Spectral Features (Tier B)')
    ax.legend(frameon=True, fancybox=False, edgecolor='gray')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 0.82)
    fig.tight_layout()
    return fig_to_base64(fig)


# ═══════════════════════════════════════════════════════════════════
# HTML 生成
# ═══════════════════════════════════════════════════════════════════

def build_report():
    """构建完整 HTML 报告。"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    # ─── 加载所有数据 ─────────────────────────────────────
    print("加载数据...")

    # Exp 1
    exp1_df = load_csv('exp1_id_baseline', 'summary_tier_b_nocoords.csv')

    # Exp 0
    exp0_df = load_csv('exp0_per_country_cv', 'summary_tier_b_nocoords.csv')

    # Exp 2
    exp2_json = load_json('exp2_loco', 'tier_b_nocoords.json')
    ae_loco = load_csv('exp2_loco', 'loco_ae_nocoords_tier_b.csv')
    s2_loco = load_csv('exp2_loco', 'loco_s2_nocoords_tier_b.csv')
    fu_loco = load_csv('exp2_loco', 'loco_fusion_nocoords_tier_b.csv')

    # Exp 3
    exp3_json = load_json('exp3_climate_zone', 'tier_b_nocoords.json')

    # Exp 4
    exp4_json = load_json('exp4_label_efficiency', 'tier_b_nocoords.json')

    # Exp 5
    exp5_json = load_json('exp5_feature_analysis', 'tier_b_nocoords.json')

    # ─── 生成图表 ─────────────────────────────────────────
    print("生成图表...")
    fig_exp1 = plot_exp1_nocoords()
    print("  [OK] Exp 1: ID Baseline")
    fig_exp0_bars = plot_exp0_country_bars()
    print("  [OK] Exp 0: Country bars")
    fig_exp0_diff = plot_exp0_heatmap_diff()
    print("  [OK] Exp 0: AE vs S2 diff" if fig_exp0_diff else "  [--] Exp 0: no per-class CSV, skipped")
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

    # ─── HTML 构建 ─────────────────────────────────────────
    print("构建 HTML...")
    parts = []

    # ═══ HEAD ═══
    parts.append(f"""\
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AlphaEarth 泛欧作物分类 — 纯光谱特征完整实验报告</title>
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
.fig-container {{
    margin: 20px 0; text-align: center;
}}
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

    # ═══ HEADER ═══
    total_samples = int(exp1_df[exp1_df.feature_type == 'ae']['n_samples'].values[0])
    parts.append(f"""\
<header>
    <h1>🌾 AlphaEarth 泛欧作物分类 — 纯光谱特征完整实验报告</h1>
    <div class="subtitle">Pure Spectral Features Only — 无坐标、无面积辅助信息</div>
    <div>
        <span class="tag">18 个欧洲国家</span>
        <span class="tag">10 类作物 (Tier B)</span>
        <span class="tag">5 个气候带</span>
        <span class="tag">~{total_samples//1000}k 样本</span>
        <span class="tag">3 种特征集</span>
        <span class="tag">6 组实验</span>
    </div>
    <p class="meta">生成时间: {now} | LightGBM 分类器 | Balanced Accuracy 为主要指标</p>
</header>
""")

    # ═══ TOC ═══
    parts.append("""\
<div class="toc">
    <h3>目录</h3>
    <ol>
        <li><a href="#sec0">实验设计总览</a></li>
        <li><a href="#sec1">数据概览与特征集定义</a></li>
        <li><a href="#sec2">Exp 1: 域内 (ID) 基线</a></li>
        <li><a href="#sec3">Exp 0: 逐国交叉验证</a></li>
        <li><a href="#sec4">Exp 2: Leave-One-Country-Out (LOCO)</a></li>
        <li><a href="#sec5">Exp 3: 跨气候带迁移</a></li>
        <li><a href="#sec6">Exp 4: 标签效率</a></li>
        <li><a href="#sec7">Exp 5: 特征空间分析 (CKA / SHAP / 国家可预测性)</a></li>
        <li><a href="#sec8">综合结论与讨论</a></li>
    </ol>
</div>
""")

    # ════════════════════════════════════════════════
    # Section 0: 实验设计总览
    # ════════════════════════════════════════════════
    parts.append("""\
<div class="section" id="sec0">
    <div class="section-header">
        <h2>Section 0: 实验设计总览</h2>
        <span class="exp-tag">DESIGN</span>
    </div>
    <div class="section-body">
        <div class="sub-title">科学问题</div>
        <p>Google 的 AlphaEarth V1 年度嵌入能否抽象出跨物候、气候和农业实践差异的<strong>通用作物语义</strong>？
        与传统 Sentinel-2 光谱特征相比，其在跨国/跨区域应用中有哪些独特特性（优势或劣势）？</p>

        <div class="sub-title">实验框架</div>
        <table>
        <tr>
            <th>实验</th>
            <th>设计</th>
            <th>测试的问题</th>
        </tr>
        <tr>
            <td><strong>Exp 0</strong></td>
            <td>逐国 5-fold CV</td>
            <td>各国独立分类能力（国内泛化）</td>
        </tr>
        <tr>
            <td><strong>Exp 1</strong></td>
            <td>全数据 5-fold CV</td>
            <td>混合训练的域内 (ID) 基线</td>
        </tr>
        <tr>
            <td><strong>Exp 2</strong></td>
            <td>Leave-One-Country-Out</td>
            <td>跨国泛化 (OOD) 能力</td>
        </tr>
        <tr>
            <td><strong>Exp 3</strong></td>
            <td>Leave-One-Zone-Out + Transfer Matrix</td>
            <td>跨气候带迁移能力</td>
        </tr>
        <tr>
            <td><strong>Exp 4</strong></td>
            <td>学习曲线 + 混合比例</td>
            <td>标签效率 / 少样本学习</td>
        </tr>
        <tr>
            <td><strong>Exp 5</strong></td>
            <td>CKA / SHAP / 国家可预测性</td>
            <td>嵌入空间结构分析</td>
        </tr>
        </table>

        <div class="sub-title">关键设计选择</div>
        <div class="finding">
            <strong>纯光谱特征：</strong>本报告中所有实验均使用<strong>纯光谱特征</strong>，
            即移除了 longitude, latitude, log_area 三个辅助特征。
            这确保了对 AlphaEarth 嵌入与 S2 光谱特征本身质量的公平评估，
            避免了地理坐标作为「捷径」干扰结论。
        </div>
    </div>
</div>
""")

    # ════════════════════════════════════════════════
    # Section 1: 数据概览与特征集定义
    # ════════════════════════════════════════════════
    n_ae = int(exp1_df[exp1_df.feature_type == 'ae']['n_samples'].values[0])
    n_s2 = int(exp1_df[exp1_df.feature_type == 's2']['n_samples'].values[0])
    n_fu = int(exp1_df[exp1_df.feature_type == 'fusion']['n_samples'].values[0])

    parts.append(f"""\
<div class="section" id="sec1">
    <div class="section-header">
        <h2>Section 1: 数据概览与特征集定义</h2>
        <span class="exp-tag">DATA</span>
    </div>
    <div class="section-body">
        <div class="overview-grid">
            <div class="stat-card">
                <div class="number">18</div>
                <div class="label">欧洲国家/地区</div>
            </div>
            <div class="stat-card">
                <div class="number">10</div>
                <div class="label">Tier B 作物类别</div>
            </div>
            <div class="stat-card">
                <div class="number">~{n_ae // 1000}k</div>
                <div class="label">AE 样本量</div>
            </div>
            <div class="stat-card">
                <div class="number">5</div>
                <div class="label">气候带</div>
            </div>
        </div>

        <div class="sub-title">三种特征集</div>
        <table>
        <tr>
            <th>特征集</th>
            <th>维度</th>
            <th>来源</th>
            <th>可用样本</th>
            <th>描述</th>
        </tr>
        <tr>
            <td><span class="fs-badge fs-ae">AE</span></td>
            <td>128</td>
            <td>AlphaEarth V1</td>
            <td>{n_ae:,}</td>
            <td>64 维年度嵌入 × 2 统计量 (mean + stdDev)</td>
        </tr>
        <tr>
            <td><span class="fs-badge fs-s2">S2</span></td>
            <td>168</td>
            <td>Sentinel-2 SR</td>
            <td>{n_s2:,}</td>
            <td>10 波段光谱统计 + 6 植被指数统计 + 季节性特征</td>
        </tr>
        <tr>
            <td><span class="fs-badge fs-fusion">FUS</span></td>
            <td>296</td>
            <td>AE + S2</td>
            <td>{n_fu:,}</td>
            <td>两者拼接（取交集样本）</td>
        </tr>
        </table>

        <div class="sub-title">Tier B 10 类作物</div>
        <p>grassland, maize, orchards, other_crops, potatoes, root_crops, strawberries,
        tree_wood_forest, vegetables, winter_cereals</p>

        <div class="sub-title">气候带划分</div>
        <table>
        <tr><th>气候带</th><th>包含国家</th></tr>
        <tr><td>大西洋 (atlantic)</td><td>BE, DE, DK, FR, IE, NL</td></tr>
        <tr><td>大陆性 (continental)</td><td>AT, CZ, SK</td></tr>
        <tr><td>地中海 (mediterranean)</td><td>ES, HR, PT</td></tr>
        <tr><td>北欧-波罗的海 (nordic_baltic)</td><td>EE, LT, LV, SE</td></tr>
        <tr><td>东南欧 (southeast)</td><td>RO</td></tr>
        </table>
    </div>
</div>
""")

    # ════════════════════════════════════════════════
    # Section 2: Exp 1 — ID Baseline
    # ════════════════════════════════════════════════
    rows1 = []
    for ft in FEATURE_SETS:
        row = exp1_df[exp1_df.feature_type == ft].iloc[0]
        ba = row['BA_mean']
        ba_std = row['BA_std']
        f1 = row['F1_mean']
        f1_std = row['F1_std']
        kappa = row['Kappa_mean']
        kappa_std = row['Kappa_std']
        n_feat = int(row['n_features'])
        n_samp = int(row['n_samples'])
        rows1.append(f"""\
        <tr>
            <td><span class="fs-badge fs-{ft}">{FS_LABELS[ft]}</span></td>
            <td>{n_feat}</td>
            <td>{n_samp:,}</td>
            <td><strong>{pct(ba)}%</strong> ±{pct(ba_std)}</td>
            <td>{pct(f1)}% ±{pct(f1_std)}</td>
            <td>{pct(kappa)}% ±{pct(kappa_std)}</td>
        </tr>""")

    best_ft = exp1_df.loc[exp1_df['BA_mean'].idxmax(), 'feature_type']
    best_ba = exp1_df['BA_mean'].max()

    parts.append(f"""\
<div class="section" id="sec2">
    <div class="section-header">
        <h2>Section 2: Exp 1 — 域内 (ID) 基线</h2>
        <span class="exp-tag">EXP 1</span>
    </div>
    <div class="section-body">
        <p><strong>设计：</strong>18 国数据合并，Stratified 5-Fold CV，使用纯光谱特征。</p>

        <table>
        <tr>
            <th>特征集</th>
            <th>维度</th>
            <th>样本量</th>
            <th>Balanced Accuracy</th>
            <th>Macro F1</th>
            <th>Cohen's Kappa</th>
        </tr>
        {"".join(rows1)}
        </table>

        <div class="fig-container">
            {img_tag(fig_exp1, 'Exp 1 ID Baseline', '650px')}
            <div class="fig-caption">图 1: 域内基线 — 三种纯光谱特征集的分类性能对比</div>
        </div>

        <div class="finding success">
            <strong>关键发现：</strong>
            Fusion ({pct(best_ba)}%) > AlphaEarth ({pct(exp1_df[exp1_df.feature_type=='ae']['BA_mean'].values[0])}%) > Sentinel-2 ({pct(exp1_df[exp1_df.feature_type=='s2']['BA_mean'].values[0])}%)。
            AlphaEarth 128 维嵌入在 ID 场景下优于 168 维 S2 特征，
            说明基础模型嵌入在域内分类任务上确实比传统光谱统计特征更有效。
            Fusion 特征集表现最佳，二者互补。
        </div>
    </div>
</div>
""")

    # ════════════════════════════════════════════════
    # Section 3: Exp 0 — 逐国 CV
    # ════════════════════════════════════════════════
    rows0 = []
    for cc in COUNTRIES:
        cells = [f"<td><strong>{cc}</strong> {COUNTRY_NAMES[cc]}</td>"]
        ba_vals = {}
        for ft in FEATURE_SETS:
            sub = exp0_df[(exp0_df.country == cc) & (exp0_df.feature_type == ft)]
            if len(sub) > 0:
                ba = sub.iloc[0]['BA_mean']
                ba_vals[ft] = ba
                cells.append(f"<td>{pct(ba)}</td>")
            else:
                ba_vals[ft] = None
                cells.append("<td>—</td>")
        # 标记最佳特征集
        rows0.append(f"<tr>{''.join(cells)}</tr>")

    # 均值行
    mean_vals = {}
    mean_cells = ['<td class="mean-row"><strong>均值</strong></td>']
    for ft in FEATURE_SETS:
        sub = exp0_df[exp0_df.feature_type == ft]
        mv = sub['BA_mean'].mean()
        mean_vals[ft] = mv
        mean_cells.append(f'<td class="mean-row"><strong>{pct(mv)}</strong></td>')
    rows0.append(f'<tr class="mean-row">{"".join(mean_cells)}</tr>')

    # 找出 AE > S2 和 S2 > AE 的国家数量
    ae_wins = 0
    s2_wins = 0
    for cc in COUNTRIES:
        ae_sub = exp0_df[(exp0_df.country == cc) & (exp0_df.feature_type == 'ae')]
        s2_sub = exp0_df[(exp0_df.country == cc) & (exp0_df.feature_type == 's2')]
        if len(ae_sub) > 0 and len(s2_sub) > 0:
            if ae_sub.iloc[0]['BA_mean'] > s2_sub.iloc[0]['BA_mean']:
                ae_wins += 1
            else:
                s2_wins += 1

    parts.append(f"""\
<div class="section" id="sec3">
    <div class="section-header">
        <h2>Section 3: Exp 0 — 逐国交叉验证</h2>
        <span class="exp-tag">EXP 0</span>
    </div>
    <div class="section-body">
        <p><strong>设计：</strong>每个国家独立 5-Fold CV，评估各国的分类难度和特征集表现差异。</p>

        <table class="compact-table">
        <tr>
            <th>国家</th>
            <th class="col-ae">AE BA (%)</th>
            <th class="col-s2">S2 BA (%)</th>
            <th class="col-fusion">Fusion BA (%)</th>
        </tr>
        {"".join(rows0)}
        </table>

        <div class="fig-container">
            {img_tag(fig_exp0_bars, 'Exp 0 Country Bars', '100%')}
            <div class="fig-caption">图 2: 逐国 5-Fold CV Balanced Accuracy — 纯光谱特征</div>
        </div>
""")
    if fig_exp0_diff:
        parts.append(f"""\
        <div class="fig-container">
            {img_tag(fig_exp0_diff, 'AE vs S2 Diff Heatmap', '100%')}
            <div class="fig-caption">图 2b: AE − S2 逐国逐类 F1 差异热力图（蓝=AE 优, 红=S2 优）</div>
        </div>
""")

    parts.append(f"""\
        <div class="finding">
            <strong>关键发现：</strong>
            在 18 个国家中，AE 在 {ae_wins} 个国家优于 S2，S2 在 {s2_wins} 个国家优于 AE。
            国家间分类难度差异显著：HR（克罗地亚，仅 3 类）BA 最高（~77%），
            PT（葡萄牙，9 类）BA 最低（~39%）。
            Fusion 在绝大多数国家表现最佳。
        </div>
    </div>
</div>
""")

    # ════════════════════════════════════════════════
    # Section 4: Exp 2 — LOCO
    # ════════════════════════════════════════════════
    rows2 = []
    for ft_key, ft_label in [('ae', 'AlphaEarth'), ('s2', 'Sentinel-2'), ('fusion', 'Fusion')]:
        per_country = exp2_json[ft_key]['per_country']
        for cc in COUNTRIES:
            if cc in per_country:
                ba = per_country[cc]['balanced_accuracy']
                f1 = per_country[cc]['macro_f1']
                n_test = per_country[cc]['n_test']
                if ft_key == 'ae':
                    rows2.append(f"<tr><td><strong>{cc}</strong> {COUNTRY_NAMES[cc]}</td>")
                rows2[-1] += f"<td>{pct(ba)}</td>"

    # 构建 LOCO 表格 — 更简洁的方式
    loco_rows = []
    for cc in COUNTRIES:
        cells = [f"<td><strong>{cc}</strong> {COUNTRY_NAMES[cc]}</td>"]
        for ft_key in FEATURE_SETS:
            per_country = exp2_json[ft_key]['per_country']
            if cc in per_country:
                ba = per_country[cc]['balanced_accuracy']
                cells.append(f"<td>{pct(ba)}</td>")
            else:
                cells.append("<td>—</td>")
        loco_rows.append(f"<tr>{''.join(cells)}</tr>")

    # 均值
    loco_means = {}
    loco_mean_cells = ['<td class="mean-row"><strong>均值</strong></td>']
    for ft_key in FEATURE_SETS:
        per_country = exp2_json[ft_key]['per_country']
        bas = [per_country[cc]['balanced_accuracy'] for cc in COUNTRIES if cc in per_country]
        m = np.mean(bas)
        loco_means[ft_key] = m
        loco_mean_cells.append(f'<td class="mean-row"><strong>{pct(m)}</strong></td>')
    loco_rows.append(f'<tr class="mean-row">{"".join(loco_mean_cells)}</tr>')

    # 找出每个特征集中 OOD 最好和最差的国家
    best_ood = {}
    worst_ood = {}
    for ft_key in FEATURE_SETS:
        per_country = exp2_json[ft_key]['per_country']
        sorted_cc = sorted(per_country.keys(), key=lambda c: per_country[c]['balanced_accuracy'])
        best_ood[ft_key] = sorted_cc[-1]
        worst_ood[ft_key] = sorted_cc[0]

    # ID-OOD gap
    gaps = {}
    for ft in FEATURE_SETS:
        id_ba = exp1_df[exp1_df.feature_type == ft]['BA_mean'].values[0]
        gaps[ft] = id_ba - loco_means[ft]

    parts.append(f"""\
<div class="section" id="sec4">
    <div class="section-header">
        <h2>Section 4: Exp 2 — Leave-One-Country-Out (LOCO)</h2>
        <span class="exp-tag">EXP 2</span>
    </div>
    <div class="section-body">
        <p><strong>设计：</strong>依次留出每个国家作为测试集（OOD），其余 17 国用于训练。
        评估纯光谱特征的跨国迁移能力。</p>

        <table class="compact-table">
        <tr>
            <th>国家</th>
            <th class="col-ae">AE OOD BA (%)</th>
            <th class="col-s2">S2 OOD BA (%)</th>
            <th class="col-fusion">Fusion OOD BA (%)</th>
        </tr>
        {"".join(loco_rows)}
        </table>

        <div class="fig-container">
            {img_tag(fig_exp2_loco, 'Exp 2 LOCO per country', '100%')}
            <div class="fig-caption">图 3: LOCO 逐国 OOD Balanced Accuracy — 纯光谱特征</div>
        </div>

        <div class="fig-container">
            {img_tag(fig_exp2_gap, 'ID vs OOD gap', '550px')}
            <div class="fig-caption">图 4: ID vs OOD 性能差距 — 域内泛化 vs 跨国迁移</div>
        </div>

        <div class="finding warning">
            <strong>ID → OOD 性能衰减：</strong>
            AE: {pct(exp1_df[exp1_df.feature_type=='ae']['BA_mean'].values[0])}% → {pct(loco_means['ae'])}% (Δ = -{pct1(gaps['ae'])}pp)；
            S2: {pct(exp1_df[exp1_df.feature_type=='s2']['BA_mean'].values[0])}% → {pct(loco_means['s2'])}% (Δ = -{pct1(gaps['s2'])}pp)；
            Fusion: {pct(exp1_df[exp1_df.feature_type=='fusion']['BA_mean'].values[0])}% → {pct(loco_means['fusion'])}% (Δ = -{pct1(gaps['fusion'])}pp)。
            三者的 ID→OOD gap 幅度相近（~22-23pp），说明跨国迁移对所有特征集都是显著挑战。
        </div>

        <div class="finding">
            <strong>OOD 排名：</strong>
            Fusion ({pct(loco_means['fusion'])}%) > AE ({pct(loco_means['ae'])}%) > S2 ({pct(loco_means['s2'])}%)。
            AlphaEarth 在 OOD 场景下仍优于 Sentinel-2，二者的融合进一步提升泛化性能。
        </div>
    </div>
</div>
""")

    # ════════════════════════════════════════════════
    # Section 5: Exp 3 — 跨气候带迁移
    # ════════════════════════════════════════════════
    zone_keys = ["atlantic", "continental", "mediterranean", "nordic_baltic", "southeast"]

    lozo_rows = []
    for zk in zone_keys:
        zl = ZONES[zk]["label"]
        cells = [f"<td><strong>{zl}</strong></td>"]
        for ft_key in FEATURE_SETS:
            zr = exp3_json[ft_key]['zone_results']
            if zk in zr:
                ba = zr[zk]['balanced_accuracy']
                cells.append(f"<td>{pct(ba)}</td>")
            else:
                cells.append("<td>—</td>")
        lozo_rows.append(f"<tr>{''.join(cells)}</tr>")

    # 均值行
    lozo_mean_cells = ['<td class="mean-row"><strong>均值</strong></td>']
    lozo_means = {}
    for ft_key in FEATURE_SETS:
        zr = exp3_json[ft_key]['zone_results']
        bas = [zr[zk]['balanced_accuracy'] for zk in zone_keys if zk in zr]
        m = np.mean(bas)
        lozo_means[ft_key] = m
        lozo_mean_cells.append(f'<td class="mean-row"><strong>{pct(m)}</strong></td>')
    lozo_rows.append(f'<tr class="mean-row">{"".join(lozo_mean_cells)}</tr>')

    parts.append(f"""\
<div class="section" id="sec5">
    <div class="section-header">
        <h2>Section 5: Exp 3 — 跨气候带迁移</h2>
        <span class="exp-tag">EXP 3</span>
    </div>
    <div class="section-body">
        <p><strong>设计：</strong>Leave-One-Zone-Out (LOZO)：在其他 4 个气候带训练，目标气候带测试。
        另外构建 zone-to-zone 迁移矩阵。</p>

        <div class="sub-title">A. LOZO — 各气候带 OOD BA</div>
        <table>
        <tr>
            <th>气候带</th>
            <th class="col-ae">AE BA (%)</th>
            <th class="col-s2">S2 BA (%)</th>
            <th class="col-fusion">Fusion BA (%)</th>
        </tr>
        {"".join(lozo_rows)}
        </table>

        <div class="fig-container">
            {img_tag(fig_exp3_lozo, 'LOZO per zone', '700px')}
            <div class="fig-caption">图 5: LOZO 各气候带 OOD Balanced Accuracy</div>
        </div>

        <div class="sub-title">B. 跨气候带迁移矩阵</div>
        <div class="fig-row three">
            <div class="fig-container">
                {img_tag(fig_exp3_tm['ae'], 'Transfer AE')}
                <div class="fig-caption">AlphaEarth</div>
            </div>
            <div class="fig-container">
                {img_tag(fig_exp3_tm['s2'], 'Transfer S2')}
                <div class="fig-caption">Sentinel-2</div>
            </div>
            <div class="fig-container">
                {img_tag(fig_exp3_tm['fusion'], 'Transfer Fusion')}
                <div class="fig-caption">Fusion</div>
            </div>
        </div>

        <div class="finding">
            <strong>关键发现：</strong>
            LOZO 均值：Fusion ({pct(lozo_means['fusion'])}%) > AE ({pct(lozo_means['ae'])}%) > S2 ({pct(lozo_means['s2'])}%)。
            大西洋气候带（BE, DE, DK, FR, IE, NL）在纯光谱特征下的 LOZO BA 最高，
            因其训练集包含数据量最大、作物覆盖最丰富的国家组合。
            东南欧（仅 RO）由于训练集中缺少类似区域，迁移相对困难。
        </div>
    </div>
</div>
""")

    # ════════════════════════════════════════════════
    # Section 6: Exp 4 — 标签效率
    # ════════════════════════════════════════════════
    # 学习曲线表格
    curve_rows = []
    for sk in CURVE_ORDER:
        cells = [f"<td>{CURVE_LABELS[sk]}</td>"]
        for ft_key in ['ae_curve', 's2_curve']:
            d = exp4_json[ft_key]['sample_sizes']
            if sk in d:
                ba = d[sk]['ba_mean']
                ba_std = d[sk]['ba_std']
                n_actual = d[sk]['n_train_actual']
                cells.append(f"<td>{pct(ba)}% ±{pct(ba_std)}</td>")
            else:
                cells.append("<td>—</td>")
        curve_rows.append(f"<tr>{''.join(cells)}</tr>")

    # Mixed scenario 表格
    mixed_rows = []
    mixed_keys = sorted([k for k in exp4_json['ae_mixed'].keys()
                         if k not in ('feature_type', 'tier', 'test_countries')],
                        key=lambda x: int(x))
    for mk in mixed_keys:
        cells = [f"<td>{mk}%</td>"]
        for ft_key in ['ae_mixed', 's2_mixed']:
            d = exp4_json[ft_key]
            if mk in d:
                ba = d[mk]['balanced_accuracy']
                cells.append(f"<td>{pct(ba)}</td>")
            else:
                cells.append("<td>—</td>")
        mixed_rows.append(f"<tr>{''.join(cells)}</tr>")

    parts.append(f"""\
<div class="section" id="sec6">
    <div class="section-header">
        <h2>Section 6: Exp 4 — 标签效率</h2>
        <span class="exp-tag">EXP 4</span>
    </div>
    <div class="section-body">
        <p><strong>设计 A (学习曲线)：</strong>固定 OOD 测试集 (ES, SE, PT)，
        逐步增加训练样本量 (每类 10~5000 / 全部)，对比 AE vs S2 的 OOD BA。</p>
        <p><strong>设计 B (混合比例)：</strong>在源域训练集中混入目标域少量标注样本 (0%~500%)，
        评估域适应效果。</p>

        <div class="sub-title">A. 学习曲线</div>
        <table>
        <tr>
            <th>训练总样本</th>
            <th class="col-ae">AE OOD BA (%)</th>
            <th class="col-s2">S2 OOD BA (%)</th>
        </tr>
        {"".join(curve_rows)}
        </table>

        <div class="fig-container">
            {img_tag(fig_exp4_lc, 'Learning curve', '600px')}
            <div class="fig-caption">图 6: OOD 学习曲线 — AE vs S2 (纯光谱)</div>
        </div>

        <div class="sub-title">B. 混合比例实验</div>
        <table>
        <tr>
            <th>目标域混入 (%)</th>
            <th class="col-ae">AE BA (%)</th>
            <th class="col-s2">S2 BA (%)</th>
        </tr>
        {"".join(mixed_rows)}
        </table>

        <div class="fig-container">
            {img_tag(fig_exp4_mix, 'Mixed scenario', '500px')}
            <div class="fig-caption">图 7: 混合比例实验 — 目标域标注混入效果</div>
        </div>

        <div class="finding">
            <strong>关键发现：</strong>
            在纯光谱特征下，AlphaEarth 在所有样本量级别上均优于 S2。
            特别是在低样本（≤1000 总样本）时，AE 已展现更强的学习效率，
            表明基础模型嵌入包含了更易被分类器利用的结构化信息。
            混合实验中，增加目标域标注对两者都有提升，但 AE 的起点和增幅均更高。
        </div>
    </div>
</div>
""")

    # ════════════════════════════════════════════════
    # Section 7: Exp 5 — 特征空间分析
    # ════════════════════════════════════════════════
    ae_cka = exp5_json['ae']['cka']
    s2_cka = exp5_json['s2']['cka']
    ae_cpred = exp5_json['ae']['country_prediction']
    s2_cpred = exp5_json['s2']['country_prediction']

    parts.append(f"""\
<div class="section" id="sec7">
    <div class="section-header">
        <h2>Section 7: Exp 5 — 特征空间分析</h2>
        <span class="exp-tag">EXP 5</span>
    </div>
    <div class="section-body">

        <div class="sub-title">A. CKA (Centered Kernel Alignment)</div>
        <p>CKA 衡量特征矩阵与标签矩阵的线性对齐程度。
        CKA<sub>crop</sub> 越高 → 特征越能区分作物类别；
        CKA<sub>country</sub> 越高 → 特征越能区分国家（地理信息）。</p>

        <table>
        <tr>
            <th>特征集</th>
            <th>CKA<sub>crop</sub></th>
            <th>CKA<sub>country</sub></th>
            <th>Crop/Country 比值</th>
        </tr>
        <tr>
            <td><span class="fs-badge fs-ae">AlphaEarth</span></td>
            <td><strong>{ae_cka['cka_crop']:.4f}</strong></td>
            <td>{ae_cka['cka_country']:.4f}</td>
            <td>{ae_cka['ratio']:.3f}</td>
        </tr>
        <tr>
            <td><span class="fs-badge fs-s2">Sentinel-2</span></td>
            <td>{s2_cka['cka_crop']:.4f}</td>
            <td>{s2_cka['cka_country']:.4f}</td>
            <td>{s2_cka['ratio']:.3f}</td>
        </tr>
        </table>

        <div class="fig-container">
            {img_tag(fig_exp5_cka, 'CKA comparison', '450px')}
            <div class="fig-caption">图 8: CKA — 特征与作物/国家标签的线性对齐程度</div>
        </div>

        <div class="finding">
            <strong>分析：</strong>
            在纯光谱特征下，AE 的 CKA<sub>crop</sub> ({ae_cka['cka_crop']:.4f}) 显著高于 S2 ({s2_cka['cka_crop']:.4f})，
            证明 AlphaEarth 嵌入确实编码了更强的作物语义信息。
            同时 AE 的 CKA<sub>country</sub> 也更高 ({ae_cka['cka_country']:.4f} vs {s2_cka['cka_country']:.4f})，
            说明 AE 嵌入在捕获作物信息的同时也保留了更多的地理区分信息。
        </div>

        <div class="sub-title">B. SHAP 特征重要性 (ID vs OOD)</div>
        <p>使用 TreeSHAP 分析 LightGBM 在 ID 和 OOD 场景下的特征使用模式差异。</p>

        <div class="fig-row">
            <div class="fig-container">
                {img_tag(fig_exp5_shap_ae, 'SHAP AE')}
                <div class="fig-caption">图 9a: AlphaEarth SHAP Top 15</div>
            </div>
            <div class="fig-container">
                {img_tag(fig_exp5_shap_s2, 'SHAP S2')}
                <div class="fig-caption">图 9b: Sentinel-2 SHAP Top 15</div>
            </div>
        </div>

        <div class="finding">
            <strong>SHAP 分析发现：</strong>
            AE 嵌入中，mean 统计量（如 A59_mean, A19_mean）占据主导地位，
            表明年度均值编码了最关键的作物信息。
            ID vs OOD 的特征重要性分布存在明显差异，说明模型在不同域中依赖的特征维度不完全相同。
        </div>

        <div class="sub-title">C. 国家可预测性 (Country Predictability)</div>
        <p>用纯光谱特征训练国家分类器。预测准确率越高 → 嵌入中残留地理信息越多。</p>

        <table>
        <tr>
            <th>特征集</th>
            <th>国家预测 BA</th>
            <th>分析</th>
        </tr>
        <tr>
            <td><span class="fs-badge fs-ae">AlphaEarth</span></td>
            <td><strong>{pct(ae_cpred['balanced_accuracy'])}%</strong></td>
            <td>128 维嵌入仍可较好地区分 18 国，嵌入编码了一定地理信息</td>
        </tr>
        <tr>
            <td><span class="fs-badge fs-s2">Sentinel-2</span></td>
            <td><strong>{pct(s2_cpred['balanced_accuracy'])}%</strong></td>
            <td>168 维 S2 特征地理区分度较低，更加「地理无关」</td>
        </tr>
        </table>

        <div class="fig-container">
            {img_tag(fig_exp5_cpred, 'Country prediction', '350px')}
            <div class="fig-caption">图 10: 纯光谱特征的国家可预测性</div>
        </div>

        <div class="finding warning">
            <strong>重要发现：</strong>
            即使去除坐标，AE 嵌入仍能以 {pct(ae_cpred['balanced_accuracy'])}% BA 预测国家身份，
            说明 AlphaEarth 嵌入内在地编码了环境/气候信息。这既是优势（丰富的环境上下文）
            也是潜在限制（可能阻碍极端 OOD 迁移）。
            S2 的国家可预测性更低 ({pct(s2_cpred['balanced_accuracy'])}%)，
            意味着 S2 光谱特征更加「地理无关」。
        </div>
    </div>
</div>
""")

    # ════════════════════════════════════════════════
    # Section 8: 综合结论
    # ════════════════════════════════════════════════
    parts.append(f"""\
<div class="section" id="sec8">
    <div class="section-header">
        <h2>Section 8: 综合结论与讨论</h2>
        <span class="exp-tag">CONCLUSION</span>
    </div>
    <div class="section-body">

        <div class="fig-container">
            {img_tag(fig_summary, 'Summary overview', '700px')}
            <div class="fig-caption">图 11: 综合性能概览 — ID / OOD (LOCO) / OOD (LOZO)</div>
        </div>

        <div class="sub-title">1. 纯光谱特征下的核心排名</div>
        <table>
        <tr>
            <th>场景</th>
            <th>排名</th>
            <th>最佳 BA</th>
        </tr>
        <tr>
            <td>ID (5-Fold CV)</td>
            <td>Fusion > AE > S2</td>
            <td>{pct(exp1_df[exp1_df.feature_type=='fusion']['BA_mean'].values[0])}%</td>
        </tr>
        <tr>
            <td>OOD (LOCO Mean)</td>
            <td>Fusion > AE > S2</td>
            <td>{pct(loco_means['fusion'])}%</td>
        </tr>
        <tr>
            <td>OOD (LOZO Mean)</td>
            <td>Fusion > AE > S2</td>
            <td>{pct(lozo_means['fusion'])}%</td>
        </tr>
        </table>

        <div class="finding success">
            <strong>核心结论 1：AlphaEarth 嵌入在纯光谱条件下一致优于 Sentinel-2 传统特征。</strong>
            无论 ID 还是 OOD 场景，128 维 AE 嵌入均优于 168 维 S2 特征。
            这证明了基础模型嵌入确实编码了超越传统光谱统计的高阶作物语义信息。
        </div>

        <div class="finding success">
            <strong>核心结论 2：Fusion 策略一致最优。</strong>
            AE 与 S2 的特征融合在所有场景下均取得最佳表现，
            说明二者编码了互补的信息：AE 提供高层语义，S2 提供精细光谱细节。
        </div>

        <div class="sub-title">2. 跨国迁移的挑战</div>
        <p>所有特征集在 ID→OOD 转换中都经历约 22-23 个百分点的性能衰减。
        这主要源于欧洲各国的农业实践、作物品种、物候差异和类别分布不平衡。
        纯光谱特征在 OOD 场景下被迫依赖光谱模式而非地理位置进行分类，
        虽然整体性能低于含坐标版本的 ID 分数，但泛化机制更为健康。</p>

        <div class="sub-title">3. 嵌入空间的双重特性</div>
        <div class="finding warning">
            <strong>AE 嵌入既编码作物语义又编码环境信息。</strong>
            CKA 分析表明 AE 在作物对齐 ({ae_cka['cka_crop']:.4f}) 和国家对齐 ({ae_cka['cka_country']:.4f}) 上
            都高于 S2。国家预测 BA 为 {pct(ae_cpred['balanced_accuracy'])}%（vs S2 的 {pct(s2_cpred['balanced_accuracy'])}%）。
            这意味着 AE 嵌入是一个「环境感知」的表征——它同时捕获了作物类型和生长环境的信息。
            这在同域应用中是优势（ID BA 更高），但可能在极端跨域迁移中成为阻力。
        </div>

        <div class="sub-title">4. 论文写作建议</div>
        <table>
        <tr><th>论文章节</th><th>核心材料</th></tr>
        <tr><td>Introduction</td><td>AE vs S2 ID 基线结果 (Exp 1)</td></tr>
        <tr><td>Cross-country Transfer</td><td>LOCO 逐国结果 + ID→OOD gap (Exp 2)</td></tr>
        <tr><td>Climate Zone Transfer</td><td>LOZO + 迁移矩阵 (Exp 3)</td></tr>
        <tr><td>Label Efficiency</td><td>学习曲线 + 混合实验 (Exp 4)</td></tr>
        <tr><td>Feature Analysis</td><td>CKA + SHAP + Country Pred (Exp 5)</td></tr>
        <tr><td>Ablation Study</td><td>有/无坐标对比（另一份 comparison 报告）</td></tr>
        </table>

        <div class="sub-title">5. 局限性</div>
        <ul style="padding-left: 20px; font-size: 14px; line-height: 2;">
            <li><strong>大国采样</strong>：FR, ES 等大国采样至 ~50k-100k 地块，未使用全部数据。</li>
            <li><strong>分类器</strong>：仅使用 LightGBM 作为分类器，未测试深度学习方案。</li>
            <li><strong>年份匹配</strong>：各国 LPIS 数据年份不完全一致。</li>
            <li><strong>Tier B 10 类</strong>：粗粒度分类体系可能掩盖了细粒度差异。</li>
        </ul>
    </div>
</div>
""")

    # ═══ FOOTER ═══
    parts.append(f"""\
<footer>
    <p>Alpha_Crops — AlphaEarth 泛欧作物分类实验报告（纯光谱特征版）</p>
    <p>生成于 {now} | LightGBM | Tier B (10 类) | 18 EU 国家 | 5 气候带</p>
</footer>
</div>
</body>
</html>
""")

    return "".join(parts)


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))

    html = build_report()

    out_path = os.path.join(RESULTS_DIR, 'nocoords_report.html')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html)

    size_kb = os.path.getsize(out_path) / 1024
    print(f"\n✅ 报告已生成: {out_path} ({size_kb:.0f} KB)")
