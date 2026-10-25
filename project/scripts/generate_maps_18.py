"""
生成附录 18 国版本的空间地图：
1. 研究区地图（18 国按气候带着色）
2. 逐国分类性能地图（Exp 0）
3. Hub 国家辐射地图（DE -> 各国）
4. 泛化能力地图（每国作为训练源）
5. 18x18 迁移矩阵热力图（已有，但重新生成更清晰的版本）
"""
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Patch

plt.rcParams.update({
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'font.family': 'Arial',
    'font.size': 10, 'figure.dpi': 300, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.1,
})

OUT_DIR = 'Manuscript/Paper_Alpha_Earth/0001_draft_v1/figures'
os.makedirs(OUT_DIR, exist_ok=True)

C18 = ['AT','BE','CZ','DE','DK','EE','ES','FR','HR','IE','LT','LV','NL','PT','RO','SE','SI','SK']

ZONE_MAP = {
    'AT':'Continental','BE':'Atlantic','CZ':'Continental','DE':'Atlantic',
    'DK':'Atlantic','EE':'Nordic-Baltic','ES':'Mediterranean','FR':'Atlantic',
    'HR':'Mediterranean','IE':'Atlantic','LT':'Nordic-Baltic','LV':'Nordic-Baltic',
    'NL':'Atlantic','PT':'Mediterranean','RO':'Southeast','SE':'Nordic-Baltic',
    'SI':'Mediterranean','SK':'Continental',
}

ZONE_COLORS = {
    'Atlantic': '#4CAF50', 'Continental': '#2196F3',
    'Mediterranean': '#FF9800', 'Nordic-Baltic': '#9C27B0', 'Southeast': '#F44336',
}

LABEL_COORDS = {
    'AT': (14.5, 47.5), 'BE': (4.5, 50.8), 'CZ': (15.5, 49.8),
    'DE': (10.5, 51.2), 'DK': (9.5, 56.0), 'EE': (25.0, 58.6),
    'ES': (-3.7, 40.4), 'FR': (2.2, 46.6), 'HR': (16.0, 45.1),
    'IE': (-8.0, 53.4), 'LT': (24.0, 55.5), 'LV': (24.6, 57.0),
    'NL': (5.3, 52.1), 'PT': (-8.0, 39.5), 'RO': (25.0, 45.9),
    'SE': (16.0, 62.0), 'SI': (14.5, 46.1), 'SK': (19.7, 48.7),
}

DATA_YEARS = {
    'AT':'2021','BE':'2021','CZ':'2023','DE':'2021-23','DK':'2019',
    'EE':'2021','ES':'2023','FR':'2018','HR':'2020','IE':'2023',
    'LT':'2021','LV':'2021','NL':'2020','PT':'2021','RO':'2021',
    'SE':'2021','SI':'2021','SK':'2021',
}


def load_europe():
    world = gpd.read_file(
        'https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip')
    return world[world['CONTINENT'] == 'Europe'].copy()


def get_country_geom(europe, iso):
    country = europe[europe['ISO_A2'] == iso]
    if len(country) == 0:
        country = europe[europe['ISO_A2_EH'] == iso]
    return country


def savefig(fig, name):
    for ext in ['png', 'pdf']:
        fig.savefig(os.path.join(OUT_DIR, f'{name}.{ext}'))
    plt.close(fig)
    print(f'  [OK] {name}')


def map_study_area_18():
    print('\n1. Study area map (18 countries)...')
    europe = load_europe()

    fig, ax = plt.subplots(figsize=(10, 10))
    europe.plot(ax=ax, color='#F5F5F5', edgecolor='#CCCCCC', linewidth=0.5)

    for iso in C18:
        country = get_country_geom(europe, iso)
        if len(country) == 0:
            print(f'    WARNING: {iso} not found')
            continue
        zone = ZONE_MAP[iso]
        color = ZONE_COLORS[zone]
        country.plot(ax=ax, color=color, edgecolor='white', linewidth=1.0, alpha=0.85)
        x, y = LABEL_COORDS[iso]
        ax.annotate(iso, xy=(x, y), fontsize=8, fontweight='bold',
                    ha='center', va='center', color='white',
                    path_effects=[pe.withStroke(linewidth=2.5, foreground='black')])

    legend_elements = [Patch(facecolor=v, edgecolor='white', label=k)
                       for k, v in ZONE_COLORS.items()]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=9,
              title='Climate Zone', title_fontsize=10,
              frameon=True, fancybox=False, edgecolor='gray')

    ax.set_xlim(-12, 32)
    ax.set_ylim(35, 72)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Extended Study Area: 18 European Countries (2018-2023 LPIS Data)',
                 fontsize=12)
    ax.grid(alpha=0.2, linestyle='--')
    savefig(fig, 'study_area_map_18')


def map_per_country_ba_18():
    print('\n2. Per-country BA map (18 countries)...')
    europe = load_europe()
    df = pd.read_csv('results/experiments/exp0_per_country_cv/summary_tier_b_nocoords.csv')

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    cmap = LinearSegmentedColormap.from_list('ba',
        ['#FFEBEE', '#EF9A9A', '#EF5350', '#C62828', '#4A148C'], N=256)
    norm = Normalize(vmin=0.3, vmax=0.80)

    for idx, ft in enumerate(['ae', 's2', 'fusion']):
        ax = axes[idx]
        europe.plot(ax=ax, color='#F5F5F5', edgecolor='#CCCCCC', linewidth=0.5)
        sub = df[df.feature_type == ft]
        ba_dict = dict(zip(sub.country, sub.BA_mean))

        for iso in C18:
            country = get_country_geom(europe, iso)
            if len(country) == 0:
                continue
            ba = ba_dict.get(iso, None)
            if ba is not None:
                color = cmap(norm(ba))
                country.plot(ax=ax, color=color, edgecolor='white', linewidth=0.8)
                x, y = LABEL_COORDS[iso]
                ax.annotate(f'{ba:.2f}', xy=(x, y), fontsize=6, fontweight='bold',
                            ha='center', va='center', color='white',
                            path_effects=[pe.withStroke(linewidth=1.8, foreground='black')])

        ft_label = {'ae': 'AlphaEarth', 's2': 'Sentinel-2', 'fusion': 'Fusion'}[ft]
        ax.set_title(ft_label, fontsize=12)
        ax.set_xlim(-12, 32); ax.set_ylim(35, 72)
        ax.set_xticks([]); ax.set_yticks([])

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.6, aspect=30, pad=0.02)
    cbar.set_label('Balanced Accuracy', fontsize=11)

    fig.suptitle('Per-Country Classification Performance — 18 Countries (5-Fold CV)',
                 fontsize=14, y=0.98)
    fig.tight_layout()
    savefig(fig, 'map_per_country_ba_18')


def map_hub_germany_18():
    print('\n3. Hub Germany map (18 countries)...')
    europe = load_europe()
    t = pd.read_csv('results/experiments/exp2b_country_transfer/transfer_fusion_nocoords_tier_b.csv')
    de_transfers = t[t.train_country == 'DE']
    ba_dict = dict(zip(de_transfers.test_country, de_transfers.balanced_accuracy))

    fig, ax = plt.subplots(figsize=(10, 10))
    europe.plot(ax=ax, color='#F5F5F5', edgecolor='#CCCCCC', linewidth=0.5)

    cmap = LinearSegmentedColormap.from_list('hub',
        ['#FFEBEE', '#FFCDD2', '#EF9A9A', '#66BB6A', '#2E7D32'], N=256)
    norm = Normalize(vmin=0.1, vmax=0.60)

    for iso in C18:
        country = get_country_geom(europe, iso)
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
            country.plot(ax=ax, color=color, edgecolor='white', linewidth=0.8)
            x, y = LABEL_COORDS[iso]
            ax.annotate(f'{iso}\n{ba:.2f}', xy=(x, y), fontsize=7, fontweight='bold',
                        ha='center', va='center', color='black',
                        path_effects=[pe.withStroke(linewidth=1.8, foreground='white')])

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.5, aspect=25)
    cbar.set_label('Transfer BA (DE -> Target)', fontsize=11)

    ax.set_xlim(-12, 32); ax.set_ylim(35, 72)
    ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
    ax.set_title('Germany as Hub: Transfer to 17 Targets — 18 Countries (Fusion)',
                 fontsize=12)
    ax.grid(alpha=0.2, linestyle='--')
    savefig(fig, 'map_hub_germany_18')


def map_generalization_18():
    print('\n4. Generalization power map (18 countries)...')
    europe = load_europe()
    t = pd.read_csv('results/experiments/exp2b_country_transfer/transfer_fusion_nocoords_tier_b.csv')
    mean_out = t.groupby('train_country')['balanced_accuracy'].mean()

    fig, ax = plt.subplots(figsize=(10, 10))
    europe.plot(ax=ax, color='#F5F5F5', edgecolor='#CCCCCC', linewidth=0.5)

    cmap = LinearSegmentedColormap.from_list('out',
        ['#FFCDD2', '#FF8A65', '#FFA726', '#66BB6A', '#2E7D32'], N=256)
    norm = Normalize(vmin=0.15, vmax=0.42)

    for iso in C18:
        country = get_country_geom(europe, iso)
        if len(country) == 0:
            continue
        ba = float(mean_out.get(iso, 0))
        color = cmap(norm(ba))
        country.plot(ax=ax, color=color, edgecolor='white', linewidth=0.8)
        x, y = LABEL_COORDS[iso]
        ax.annotate(f'{iso}\n{ba:.2f}', xy=(x, y), fontsize=7, fontweight='bold',
                    ha='center', va='center', color='black',
                    path_effects=[pe.withStroke(linewidth=1.8, foreground='white')])

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.5, aspect=25)
    cbar.set_label('Mean Outgoing Transfer BA', fontsize=11)

    ax.set_xlim(-12, 32); ax.set_ylim(35, 72)
    ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
    ax.set_title('Generalization Power — 18 Countries (Fusion)', fontsize=12)
    ax.grid(alpha=0.2, linestyle='--')
    savefig(fig, 'map_generalization_power_18')


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    map_study_area_18()
    map_per_country_ba_18()
    map_hub_germany_18()
    map_generalization_18()
    print('\nAll 18-country maps done!')
