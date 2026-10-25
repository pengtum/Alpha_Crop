"""
生成论文所需的空间地图：
1. 研究区地图（11 国按气候带着色）
2. 逐国分类性能地图（Exp 0 per-country BA）
3. Hub 国家辐射地图（DE -> 各国 transfer BA）
4. 迁移性能空间可视化
"""
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable

plt.rcParams.update({
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'font.family': 'Arial',
    'font.size': 10, 'figure.dpi': 300, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.1,
})

OUT_DIR = 'Manuscript/Paper_Alpha_Earth/0001_draft_v1/figures'
os.makedirs(OUT_DIR, exist_ok=True)

C2021 = ['AT', 'BE', 'DE', 'EE', 'LT', 'LV', 'PT', 'RO', 'SE', 'SI', 'SK']

COUNTRY_NAMES = {
    'AT': 'Austria', 'BE': 'Belgium', 'DE': 'Germany',
    'EE': 'Estonia', 'LT': 'Lithuania', 'LV': 'Latvia',
    'PT': 'Portugal', 'RO': 'Romania', 'SE': 'Sweden',
    'SI': 'Slovenia', 'SK': 'Slovakia',
}

ZONE_MAP = {
    'AT': 'Continental', 'BE': 'Atlantic', 'DE': 'Atlantic',
    'EE': 'Nordic-Baltic', 'LT': 'Nordic-Baltic', 'LV': 'Nordic-Baltic',
    'PT': 'Mediterranean', 'RO': 'Southeast', 'SE': 'Nordic-Baltic',
    'SI': 'Mediterranean', 'SK': 'Continental',
}

ZONE_COLORS = {
    'Atlantic': '#4CAF50',
    'Continental': '#2196F3',
    'Mediterranean': '#FF9800',
    'Nordic-Baltic': '#9C27B0',
    'Southeast': '#F44336',
}

# 国家中心坐标（用于标注）
LABEL_COORDS = {
    'AT': (14.5, 47.5), 'BE': (4.5, 50.8), 'DE': (10.5, 51.2),
    'EE': (25.0, 58.6), 'LT': (24.0, 55.5), 'LV': (24.6, 57.0),
    'PT': (-8.0, 39.5), 'RO': (25.0, 45.9), 'SE': (16.0, 62.0),
    'SI': (14.5, 46.1), 'SK': (19.7, 48.7),
}

# ISO_A2 -> 我们的 code 映射
ISO_MAP = {
    'AT': 'AT', 'BE': 'BE', 'DE': 'DE', 'EE': 'EE',
    'LT': 'LT', 'LV': 'LV', 'PT': 'PT', 'RO': 'RO',
    'SE': 'SE', 'SI': 'SI', 'SK': 'SK',
}


def load_europe():
    """加载欧洲国家边界。"""
    world = gpd.read_file(
        'https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip')
    europe = world[world['CONTINENT'] == 'Europe'].copy()
    return europe


def savefig(fig, name):
    for ext in ['png', 'pdf']:
        fig.savefig(os.path.join(OUT_DIR, f'{name}.{ext}'))
    plt.close(fig)
    print(f'  [OK] {name}')


# ═══════════════════════════════════════════════════
# Map 1: Study area — 11 countries by climate zone
# ═══════════════════════════════════════════════════
def map_study_area():
    print('\n1. Study area map...')
    europe = load_europe()

    fig, ax = plt.subplots(figsize=(10, 10))

    # 背景：全欧洲灰色
    europe.plot(ax=ax, color='#F5F5F5', edgecolor='#CCCCCC', linewidth=0.5)

    # 11 个研究国家按气候带着色
    for iso in C2021:
        country = europe[europe['ISO_A2'] == iso]
        if len(country) == 0:
            # 尝试 ISO_A2_EH
            country = europe[europe['ISO_A2_EH'] == iso]
        if len(country) == 0:
            print(f'    WARNING: {iso} not found')
            continue
        zone = ZONE_MAP[iso]
        color = ZONE_COLORS[zone]
        country.plot(ax=ax, color=color, edgecolor='white', linewidth=1.2, alpha=0.85)

    # 标注国家代码
    for iso in C2021:
        x, y = LABEL_COORDS[iso]
        ax.annotate(iso, xy=(x, y), fontsize=9, fontweight='bold',
                    ha='center', va='center', color='white',
                    path_effects=[pe.withStroke(linewidth=2.5, foreground='black')])

    # 图例
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=v, edgecolor='white', label=k)
                       for k, v in ZONE_COLORS.items()]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=9,
              title='Climate Zone', title_fontsize=10,
              frameon=True, fancybox=False, edgecolor='gray')

    ax.set_xlim(-12, 32)
    ax.set_ylim(35, 72)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Study Area: 11 European Countries (2021 LPIS Data)', fontsize=13)
    ax.grid(alpha=0.2, linestyle='--')

    savefig(fig, 'study_area_map')


# ═══════════════════════════════════════════════════
# Map 2: Per-country BA (Exp 0) — choropleth
# ═══════════════════════════════════════════════════
def map_per_country_ba():
    print('\n2. Per-country BA map...')
    europe = load_europe()

    # 加载 Exp 0 结果
    df = pd.read_csv('results/experiments/exp0_per_country_cv/summary_tier_b_nocoords_2021.csv')

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    cmap = LinearSegmentedColormap.from_list('ba',
        ['#FFEBEE', '#EF9A9A', '#EF5350', '#C62828', '#4A148C'], N=256)

    for idx, ft in enumerate(['ae', 's2', 'fusion']):
        ax = axes[idx]
        europe.plot(ax=ax, color='#F5F5F5', edgecolor='#CCCCCC', linewidth=0.5)

        sub = df[df.feature_type == ft]
        ba_dict = dict(zip(sub.country, sub.BA_mean))

        for iso in C2021:
            country = europe[europe['ISO_A2'] == iso]
            if len(country) == 0:
                country = europe[europe['ISO_A2_EH'] == iso]
            if len(country) == 0:
                continue
            ba = ba_dict.get(iso, None)
            if ba is not None:
                # Normalize BA to color
                norm = Normalize(vmin=0.3, vmax=0.75)
                color = cmap(norm(ba))
                country.plot(ax=ax, color=color, edgecolor='white', linewidth=1)
                x, y = LABEL_COORDS[iso]
                ax.annotate(f'{ba:.2f}', xy=(x, y), fontsize=7, fontweight='bold',
                            ha='center', va='center', color='white',
                            path_effects=[pe.withStroke(linewidth=2, foreground='black')])

        ft_label = {'ae': 'AlphaEarth', 's2': 'Sentinel-2', 'fusion': 'Fusion'}[ft]
        ax.set_title(ft_label, fontsize=12)
        ax.set_xlim(-12, 32)
        ax.set_ylim(35, 72)
        ax.set_xticks([])
        ax.set_yticks([])

    # Colorbar
    sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=0.3, vmax=0.75))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.6, aspect=30, pad=0.02)
    cbar.set_label('Balanced Accuracy', fontsize=11)

    fig.suptitle('Per-Country Classification Performance (5-Fold CV, Tier B)',
                 fontsize=14, y=0.98)
    fig.tight_layout()
    savefig(fig, 'map_per_country_ba')


# ═══════════════════════════════════════════════════
# Map 3: Hub country — DE transfer to all others
# ═══════════════════════════════════════════════════
def map_hub_transfer():
    print('\n3. Hub country transfer map...')
    europe = load_europe()

    # 加载 Exp 2b Fusion transfer
    t = pd.read_csv('results/experiments/exp2b_country_transfer/transfer_fusion_nocoords_2021_tier_b.csv')
    de_transfers = t[t.train_country == 'DE']
    ba_dict = dict(zip(de_transfers.test_country, de_transfers.balanced_accuracy))

    fig, ax = plt.subplots(figsize=(10, 10))
    europe.plot(ax=ax, color='#F5F5F5', edgecolor='#CCCCCC', linewidth=0.5)

    cmap = LinearSegmentedColormap.from_list('hub',
        ['#FFEBEE', '#FFCDD2', '#EF9A9A', '#66BB6A', '#2E7D32'], N=256)
    norm = Normalize(vmin=0.1, vmax=0.55)

    # 着色目标国
    for iso in C2021:
        country = europe[europe['ISO_A2'] == iso]
        if len(country) == 0:
            country = europe[europe['ISO_A2_EH'] == iso]
        if len(country) == 0:
            continue

        if iso == 'DE':
            country.plot(ax=ax, color='#1565C0', edgecolor='white', linewidth=1.5)
            x, y = LABEL_COORDS[iso]
            ax.annotate('DE\n(Source)', xy=(x, y), fontsize=10, fontweight='bold',
                        ha='center', va='center', color='white',
                        path_effects=[pe.withStroke(linewidth=2.5, foreground='black')])
        else:
            ba = ba_dict.get(iso, 0)
            color = cmap(norm(ba))
            country.plot(ax=ax, color=color, edgecolor='white', linewidth=1)
            x, y = LABEL_COORDS[iso]
            ax.annotate(f'{iso}\n{ba:.2f}', xy=(x, y), fontsize=8, fontweight='bold',
                        ha='center', va='center', color='black',
                        path_effects=[pe.withStroke(linewidth=2, foreground='white')])

    # Colorbar
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.5, aspect=25)
    cbar.set_label('Transfer BA (DE -> Target)', fontsize=11)

    ax.set_xlim(-12, 32)
    ax.set_ylim(35, 72)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Germany as Hub Country: Transfer to All Targets (Fusion)',
                 fontsize=13)
    ax.grid(alpha=0.2, linestyle='--')

    savefig(fig, 'map_hub_germany')


# ═══════════════════════════════════════════════════
# Map 4: Mean outgoing transfer per country
# ═══════════════════════════════════════════════════
def map_mean_outgoing():
    print('\n4. Mean outgoing transfer map...')
    europe = load_europe()

    t = pd.read_csv('results/experiments/exp2b_country_transfer/transfer_fusion_nocoords_2021_tier_b.csv')
    mean_out = t.groupby('train_country')['balanced_accuracy'].mean()

    fig, ax = plt.subplots(figsize=(10, 10))
    europe.plot(ax=ax, color='#F5F5F5', edgecolor='#CCCCCC', linewidth=0.5)

    cmap = LinearSegmentedColormap.from_list('out',
        ['#FFCDD2', '#FF8A65', '#FFA726', '#66BB6A', '#2E7D32'], N=256)
    norm = Normalize(vmin=0.15, vmax=0.40)

    for iso in C2021:
        country = europe[europe['ISO_A2'] == iso]
        if len(country) == 0:
            country = europe[europe['ISO_A2_EH'] == iso]
        if len(country) == 0:
            continue
        ba = mean_out.get(iso, 0)
        color = cmap(norm(ba))
        country.plot(ax=ax, color=color, edgecolor='white', linewidth=1.2)
        x, y = LABEL_COORDS[iso]
        ax.annotate(f'{iso}\n{ba:.2f}', xy=(x, y), fontsize=8, fontweight='bold',
                    ha='center', va='center', color='black',
                    path_effects=[pe.withStroke(linewidth=2, foreground='white')])

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.5, aspect=25)
    cbar.set_label('Mean Outgoing Transfer BA', fontsize=11)

    ax.set_xlim(-12, 32)
    ax.set_ylim(35, 72)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Generalization Power: Mean OOD BA When Training on Each Country (Fusion)',
                 fontsize=12)
    ax.grid(alpha=0.2, linestyle='--')

    savefig(fig, 'map_generalization_power')


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    map_study_area()
    map_per_country_ba()
    map_hub_transfer()
    map_mean_outgoing()
    print('\nAll maps done!')
