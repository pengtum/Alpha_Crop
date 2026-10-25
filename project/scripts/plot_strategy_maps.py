"""
可视化 6 种 borrowing strategy 的空间分区逻辑（以 DE 为 source 为例）。
每个 strategy 画一张小地图，显示采样来源国家的"权重"。
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
    'font.size': 9, 'figure.dpi': 150, 'savefig.dpi': 150,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.1,
})

OUT_DIR = 'Manuscript/Paper_Alpha_Earth/0001_draft_v1/figures'
os.makedirs(OUT_DIR, exist_ok=True)

SOURCES = ['AT', 'BE', 'DE', 'EE', 'LT', 'LV', 'PT', 'SE', 'SI', 'SK']
COUNTRY_ZONE = {
    'AT': 'continental', 'BE': 'atlantic', 'DE': 'atlantic',
    'EE': 'nordic_baltic', 'LT': 'nordic_baltic', 'LV': 'nordic_baltic',
    'PT': 'mediterranean', 'SE': 'nordic_baltic',
    'SI': 'mediterranean', 'SK': 'continental',
}
ZONE_COLORS = {
    'atlantic': '#4CAF50', 'continental': '#2196F3',
    'mediterranean': '#FF9800', 'nordic_baltic': '#9C27B0',
}
LABEL_COORDS = {
    'AT': (14.5, 47.5), 'BE': (4.5, 50.8), 'DE': (10.5, 51.2),
    'EE': (25.0, 58.6), 'LT': (24.0, 55.5), 'LV': (24.6, 57.0),
    'PT': (-8.0, 39.5), 'SE': (16.0, 62.0),
    'SI': (14.5, 46.1), 'SK': (19.7, 48.7),
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


def get_budget_allocation(strategy, source, budget=10000):
    """返回给定策略下，每个 target 国获得的样本数（相对权重）。"""
    others = [c for c in SOURCES if c != source]
    allocation = {c: 0 for c in others}

    if strategy == 'random':
        # 按各国 pool 大小比例。近似：BE 大，其他国差不多
        pool_sizes = {c: 0.8 * (95000 if c == 'BE' else 50000) for c in others}
        total = sum(pool_sizes.values())
        for c in others:
            allocation[c] = budget * pool_sizes[c] / total

    elif strategy == 'balanced':
        # 每国均分
        per = budget // len(others)
        for c in others:
            allocation[c] = per

    elif strategy == 'climate_similar':
        # 只从同气候带
        same = [c for c in others if COUNTRY_ZONE[c] == COUNTRY_ZONE[source]]
        if same:
            per = budget // len(same)
            for c in same:
                allocation[c] = per

    elif strategy == 'climate_diverse':
        # 只从不同气候带
        diff = [c for c in others if COUNTRY_ZONE[c] != COUNTRY_ZONE[source]]
        if diff:
            # 简化：按不同 zone 的国家均分
            per = budget // len(diff)
            for c in diff:
                allocation[c] = per

    elif strategy == 'representativeness':
        # 按气候带分组，每组均分 budget/n_zones
        zones = {}
        for c in others:
            zones.setdefault(COUNTRY_ZONE[c], []).append(c)
        per_zone = budget // len(zones)
        for zone, countries in zones.items():
            per_c = per_zone // len(countries)
            for c in countries:
                allocation[c] = per_c

    elif strategy == 'transfer_strong':
        # 按 Exp 2b 迁移排名，从最高开始填
        csv = f'results/experiments/exp2b_country_transfer/transfer_fusion_nocoords_2021_tier_b.csv'
        try:
            df = pd.read_csv(csv)
            ranking = df[df.train_country == source].sort_values(
                'balanced_accuracy', ascending=False)
            ranking = ranking[ranking.test_country.isin(others)]
            remaining = budget
            for _, r in ranking.iterrows():
                c = r.test_country
                if remaining <= 0:
                    break
                take = min(40000, remaining)
                allocation[c] = take
                remaining -= take
        except Exception:
            pass

    elif strategy == 'transfer_weak':
        # 从迁移最弱的开始
        csv = f'results/experiments/exp2b_country_transfer/transfer_fusion_nocoords_2021_tier_b.csv'
        try:
            df = pd.read_csv(csv)
            ranking = df[df.train_country == source].sort_values(
                'balanced_accuracy', ascending=True)
            ranking = ranking[ranking.test_country.isin(others)]
            remaining = budget
            for _, r in ranking.iterrows():
                c = r.test_country
                if remaining <= 0:
                    break
                take = min(40000, remaining)
                allocation[c] = take
                remaining -= take
        except Exception:
            pass

    return allocation


def plot_strategy_maps(source='DE', budget=10000):
    """画 6 种策略的小地图（source=DE）。"""
    print(f'Plotting strategy maps for source = {source}, budget = {budget}...')
    europe = load_europe()

    strategies = ['random', 'balanced', 'climate_similar',
                  'climate_diverse', 'representativeness',
                  'transfer_strong', 'transfer_weak']
    strat_labels = {
        'random': 'Random\n(proportional to pool size)',
        'balanced': 'Balanced\n(N/9 per country)',
        'climate_similar': 'Climate-similar\n(same zone only)',
        'climate_diverse': 'Climate-diverse\n(different zones only)',
        'representativeness': 'Representativeness\n(N/5 per climate zone)',
        'transfer_strong': 'Transfer-strong\n(exploit top-BA targets)',
        'transfer_weak': 'Transfer-weak\n(explore bottom-BA targets)',
    }

    n = len(strategies)
    fig, axes = plt.subplots(2, 4, figsize=(18, 10))
    axes = axes.flatten()

    cmap = LinearSegmentedColormap.from_list(
        'alloc', ['#F5F5F5', '#FFE082', '#FFA726', '#E65100', '#4A148C'], N=256)

    for idx, strat in enumerate(strategies):
        ax = axes[idx]
        alloc = get_budget_allocation(strat, source, budget)

        # 每个子图独立 color scale（从 0 到该策略的最大分配量）
        panel_max = max(alloc.values()) if alloc.values() else 1
        norm = Normalize(vmin=0, vmax=panel_max)

        # 背景
        europe.plot(ax=ax, color='#F5F5F5', edgecolor='#CCCCCC', linewidth=0.3)

        for iso in SOURCES:
            country = get_country_geom(europe, iso)
            if len(country) == 0:
                continue
            if iso == source:
                # Source 国
                country.plot(ax=ax, color='#1565C0', edgecolor='white',
                             linewidth=1.0)
                x, y = LABEL_COORDS[iso]
                ax.annotate(f'{iso}\n(Src)', xy=(x, y), fontsize=7,
                            fontweight='bold', ha='center', va='center',
                            color='white',
                            path_effects=[pe.withStroke(linewidth=2,
                                                         foreground='black')])
            else:
                n_samples = alloc.get(iso, 0)
                color = cmap(norm(n_samples))
                country.plot(ax=ax, color=color, edgecolor='white',
                             linewidth=0.8)
                x, y = LABEL_COORDS[iso]
                if n_samples > 0:
                    label = f'{iso}\n{int(n_samples/1000)}k' if n_samples >= 1000 else f'{iso}\n0'
                else:
                    label = f'{iso}\n0'
                ax.annotate(label, xy=(x, y), fontsize=6,
                            fontweight='bold', ha='center', va='center',
                            color='black',
                            path_effects=[pe.withStroke(linewidth=1.5,
                                                         foreground='white')])

        ax.set_xlim(-12, 32)
        ax.set_ylim(35, 72)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f'{strat_labels[strat]}\n(max={int(panel_max):,})',
                      fontsize=9, fontweight='bold')

    # 最后一个 subplot：source_only（只有 source）
    ax = axes[-1]
    europe.plot(ax=ax, color='#F5F5F5', edgecolor='#CCCCCC', linewidth=0.3)
    for iso in SOURCES:
        country = get_country_geom(europe, iso)
        if len(country) == 0:
            continue
        if iso == source:
            country.plot(ax=ax, color='#1565C0', edgecolor='white', linewidth=1.0)
            x, y = LABEL_COORDS[iso]
            ax.annotate(f'{iso}\n(Src)', xy=(x, y), fontsize=7, fontweight='bold',
                        ha='center', va='center', color='white',
                        path_effects=[pe.withStroke(linewidth=2, foreground='black')])
        else:
            country.plot(ax=ax, color='#F5F5F5', edgecolor='white', linewidth=0.8)
            x, y = LABEL_COORDS[iso]
            ax.annotate(f'{iso}\n0', xy=(x, y), fontsize=6,
                        ha='center', va='center', color='gray')
    ax.set_xlim(-12, 32); ax.set_ylim(35, 72)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title('Source only\n(N=0, baseline)', fontsize=10, fontweight='bold')

    fig.suptitle(f'Spatial Allocation of Borrowing Strategies\n'
                 f'Source = {source} (blue), Budget N = {budget:,} '
                 f'(per-panel color scale; max value shown in title)',
                 fontsize=12, y=1.00)
    fig.tight_layout()

    for ext in ['png', 'pdf']:
        fig.savefig(os.path.join(OUT_DIR, f'exp6_strategy_maps_{source}.{ext}'))
    plt.close(fig)
    print(f'  [OK] exp6_strategy_maps_{source}')


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    # 给 DE、PT、EE 三个典型 source 各画一张
    for src in ['DE', 'PT', 'EE']:
        plot_strategy_maps(source=src, budget=10000)
    print('\nAll maps done!')
