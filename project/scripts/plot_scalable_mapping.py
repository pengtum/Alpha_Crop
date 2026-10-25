"""
生成 Exp 6 scalable mapping 的核心图表。
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.patches import Patch

plt.rcParams.update({
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'font.family': 'Arial',
    'font.size': 10, 'figure.dpi': 150, 'savefig.dpi': 150,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.1,
})

OUT_DIR = 'results/experiments/exp6_scalable_mapping'
FIG_DIR = 'Manuscript/Paper_Alpha_Earth/0001_draft_v1/figures'
os.makedirs(FIG_DIR, exist_ok=True)

SOURCES = ['AT', 'BE', 'DE', 'EE', 'LT', 'LV', 'PT', 'SE', 'SI', 'SK']
STRATEGIES = ['source_only', 'random', 'balanced', 'climate_diverse',
              'representativeness', 'climate_similar',
              'transfer_weak', 'transfer_strong']
STRAT_LABELS = {
    'source_only': 'Source only', 'random': 'Random',
    'climate_similar': 'Climate-similar', 'climate_diverse': 'Climate-diverse',
    'representativeness': 'Representativeness', 'balanced': 'Balanced',
    'transfer_strong': 'Transfer-strong', 'transfer_weak': 'Transfer-weak',
}
FEATURES = ['ae', 's2', 'fusion']
FT_LABELS = {'ae': 'AlphaEarth', 's2': 'Sentinel-2', 'fusion': 'Fusion'}
FT_COLORS = {'ae': '#2196F3', 's2': '#FF9800', 'fusion': '#4CAF50'}

STRAT_COLORS = {
    'source_only': '#9E9E9E',
    'random': '#1976D2',
    'balanced': '#2E7D32',
    'climate_diverse': '#43A047',
    'representativeness': '#7CB342',
    'climate_similar': '#D84315',
    'transfer_weak': '#FDD835',
    'transfer_strong': '#E53935',
}


def load_all_data():
    all_data = []
    for s in SOURCES:
        for ft in FEATURES:
            f = os.path.join(OUT_DIR, f'scalable_{ft}_{s}_tier_b.csv')
            if not os.path.exists(f):
                continue
            df = pd.read_csv(f)
            df['source'] = s
            df['feature_type_norm'] = ft
            all_data.append(df)
    return pd.concat(all_data, ignore_index=True)


def savefig(fig, name):
    for ext in ['png', 'pdf']:
        fig.savefig(os.path.join(FIG_DIR, f'{name}.{ext}'))
    plt.close(fig)
    print(f'  [OK] {name}')


# ========================================================================
# Fig 1: Strategy × Source heatmap (BA at max budget), 3 subplots for features
# ========================================================================
def plot_strategy_heatmap(df):
    print('\n1. Strategy × Source heatmap...')
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    cmap = LinearSegmentedColormap.from_list(
        'ba', ['#FFEBEE', '#FFAB91', '#FFA726', '#66BB6A', '#2E7D32'], N=256)
    vmin, vmax = 0.20, 0.53

    # 对每个 feature 构建 source × strategy 矩阵
    for idx, ft in enumerate(FEATURES):
        ax = axes[idx]
        sub = df[(df.feature_type_norm == ft)
                 & ((df.budget == 50000) | (df.strategy == 'source_only'))]
        mat = sub.groupby(['source', 'strategy'])['mean_ood_ba_equal'].mean()
        pivot = mat.unstack()
        pivot = pivot[STRATEGIES].loc[SOURCES]

        im = ax.imshow(pivot.values, cmap=cmap, aspect='auto',
                       vmin=vmin, vmax=vmax)

        ax.set_xticks(range(len(STRATEGIES)))
        ax.set_xticklabels([STRAT_LABELS[s] for s in STRATEGIES],
                           rotation=45, ha='right', fontsize=9)
        ax.set_yticks(range(len(SOURCES)))
        ax.set_yticklabels(SOURCES, fontsize=9)
        ax.set_xlabel('Borrowing strategy')
        if idx == 0:
            ax.set_ylabel('Source country')
        ax.set_title(FT_LABELS[ft])

        # 数值标注
        for i in range(len(SOURCES)):
            for j in range(len(STRATEGIES)):
                v = pivot.iloc[i, j]
                if not pd.isna(v):
                    color = 'white' if v > 0.42 else 'black'
                    ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                            fontsize=7, color=color)

    cbar = fig.colorbar(im, ax=axes, shrink=0.65, label='Mean OOD BA',
                         pad=0.02)
    fig.suptitle('Strategy Performance by Source Country (budget=50k)',
                 fontsize=12, y=1.02)
    savefig(fig, 'exp6_strategy_heatmap')


# ========================================================================
# Fig 2: Budget scaling curves, one subplot per source (Fusion only)
# ========================================================================
def plot_budget_curves(df):
    print('\n2. Budget scaling curves (Fusion)...')
    ft = 'fusion'
    sub = df[df.feature_type_norm == ft]

    fig, axes = plt.subplots(2, 5, figsize=(20, 7), sharey=True)
    axes = axes.flatten()

    for idx, src in enumerate(SOURCES):
        ax = axes[idx]
        src_data = sub[sub.source == src]

        # source_only baseline
        base = src_data[src_data.strategy == 'source_only']['mean_ood_ba_equal'].mean()
        ax.axhline(y=base, color='gray', linestyle=':', linewidth=1,
                   label='Source only', alpha=0.7)

        for strategy in STRATEGIES:
            if strategy == 'source_only':
                continue
            strat_data = src_data[src_data.strategy == strategy]
            if len(strat_data) == 0:
                continue
            means = strat_data.groupby('budget')['mean_ood_ba_equal'].mean()
            ax.plot(means.index, means.values, marker='o',
                    color=STRAT_COLORS.get(strategy, 'gray'), linewidth=1.5,
                    markersize=4, label=STRAT_LABELS[strategy])

        ax.set_xscale('log')
        ax.set_title(f'Source: {src}', fontsize=10)
        ax.grid(alpha=0.3, linestyle='--')
        ax.set_ylim(0.15, 0.55)
        if idx == 0:
            ax.set_ylabel('Mean OOD BA')
        if idx == 5:
            ax.set_ylabel('Mean OOD BA')
        if idx >= 5:
            ax.set_xlabel('Budget (log scale)')

    # Shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right',
               bbox_to_anchor=(1.08, 0.5), fontsize=9, frameon=True)

    fig.suptitle('Budget Scaling of Borrowing Strategies (Fusion)',
                 fontsize=13, y=1.00)
    fig.tight_layout()
    savefig(fig, 'exp6_budget_curves_fusion')


# ========================================================================
# Fig 3: Strategy popularity bar chart (votes per rank)
# ========================================================================
def plot_strategy_votes(df):
    print('\n3. Strategy votes per rank...')
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    max_b = 50000
    aug_strats = [s for s in STRATEGIES if s != 'source_only']

    for idx, ft in enumerate(FEATURES):
        ax = axes[idx]
        sub = df[(df.feature_type_norm == ft)
                 & (df.budget == max_b)
                 & (df.strategy != 'source_only')]
        src_strat_ba = sub.groupby(['source', 'strategy'])[
            'mean_ood_ba_equal'].mean().reset_index()

        # 对每个 source 排名
        ranks = {strat: [0, 0, 0, 0, 0, 0, 0] for strat in aug_strats}
        for src in SOURCES:
            src_sub = src_strat_ba[src_strat_ba.source == src].sort_values(
                'mean_ood_ba_equal', ascending=False).reset_index(drop=True)
            for rank, (_, r) in enumerate(src_sub.iterrows()):
                if r.strategy in ranks:
                    ranks[r.strategy][rank] += 1

        # 画堆叠柱状图：x=strategy, y=count, hue=rank
        x = np.arange(len(aug_strats))
        bottom = np.zeros(len(aug_strats))
        rank_colors = ['#2E7D32', '#66BB6A', '#FDD835', '#FFB74D',
                       '#FF8A65', '#EF5350', '#C62828']
        for r_idx in range(7):
            heights = [ranks[strat][r_idx] for strat in aug_strats]
            ax.bar(x, heights, bottom=bottom,
                   color=rank_colors[r_idx], edgecolor='white', linewidth=0.5,
                   label=f'Rank {r_idx+1}' if idx == 2 else None)
            bottom += np.array(heights)

        ax.set_xticks(x)
        ax.set_xticklabels([STRAT_LABELS[s] for s in aug_strats],
                           rotation=45, ha='right', fontsize=9)
        ax.set_title(FT_LABELS[ft])
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        if idx == 0:
            ax.set_ylabel('Count (across 10 sources)')

    axes[2].legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
                    fontsize=8, frameon=True)
    fig.suptitle('Strategy Rank Distribution (budget=50k)', fontsize=12, y=1.02)
    fig.tight_layout()
    savefig(fig, 'exp6_strategy_votes')


# ========================================================================
# Fig 4: AE vs S2 comparison — winners per source
# ========================================================================
def plot_ae_vs_s2(df):
    print('\n4. AE vs S2 best strategy per source...')
    fig, ax = plt.subplots(figsize=(12, 5))
    max_b = 50000

    ae_bests, s2_bests = [], []
    ae_bas, s2_bas = [], []

    for src in SOURCES:
        for ft, lst_b, lst_s in [('ae', ae_bas, ae_bests), ('s2', s2_bas, s2_bests)]:
            sub = df[(df.source == src)
                     & (df.feature_type_norm == ft)
                     & (df.budget == max_b)
                     & (df.strategy != 'source_only')]
            if len(sub) == 0:
                lst_b.append(np.nan)
                lst_s.append('—')
                continue
            grp = sub.groupby('strategy')['mean_ood_ba_equal'].mean()
            best_strat = grp.idxmax()
            lst_b.append(grp.max())
            lst_s.append(best_strat)

    x = np.arange(len(SOURCES))
    w = 0.35

    bars_ae = ax.bar(x - w/2, ae_bas, w, label='AlphaEarth',
                      color=FT_COLORS['ae'], edgecolor='white', linewidth=0.5)
    bars_s2 = ax.bar(x + w/2, s2_bas, w, label='Sentinel-2',
                      color=FT_COLORS['s2'], edgecolor='white', linewidth=0.5)

    # 标注最优策略（缩写）
    abbr = {'random': 'R', 'balanced': 'B', 'climate_diverse': 'Cd',
            'climate_similar': 'Cs', 'representativeness': 'Rp',
            'transfer_strong': 'Ts', 'transfer_weak': 'Tw'}
    for i in range(len(SOURCES)):
        if not np.isnan(ae_bas[i]):
            ax.text(i - w/2, ae_bas[i] + 0.005, abbr.get(ae_bests[i], '?'),
                    ha='center', fontsize=8, fontweight='bold',
                    color=FT_COLORS['ae'])
        if not np.isnan(s2_bas[i]):
            ax.text(i + w/2, s2_bas[i] + 0.005, abbr.get(s2_bests[i], '?'),
                    ha='center', fontsize=8, fontweight='bold',
                    color=FT_COLORS['s2'])

    ax.set_xticks(x)
    ax.set_xticklabels(SOURCES)
    ax.set_xlabel('Source country')
    ax.set_ylabel('Best OOD BA at budget=50k')
    ax.set_title('Optimal Borrowing Strategy Differs Between AlphaEarth and Sentinel-2\n'
                 '(R=Random, B=Balanced, Cd=Climate-diverse, Cs=Similar, '
                 'Rp=Representativeness, Ts=Transfer-strong, Tw=Transfer-weak)',
                 fontsize=10)
    ax.legend(frameon=True, fancybox=False, edgecolor='gray')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0.30, 0.55)

    fig.tight_layout()
    savefig(fig, 'exp6_ae_vs_s2_best')


# ========================================================================
# Fig 5: Gain from borrowing (boxplot) per source
# ========================================================================
def plot_gain_from_borrowing(df):
    print('\n5. Gain from borrowing boxplot...')
    max_b = 50000

    rows = []
    for ft in FEATURES:
        for src in SOURCES:
            sub = df[(df.source == src) & (df.feature_type_norm == ft)]
            baseline = sub[sub.strategy == 'source_only']['mean_ood_ba_equal'].mean()
            aug = sub[(sub.budget == max_b) & (sub.strategy != 'source_only')]
            for strat, bas in aug.groupby('strategy')['mean_ood_ba_equal']:
                for b in bas.values:
                    rows.append({
                        'feature_type': ft, 'source': src,
                        'strategy': strat,
                        'gain': b - baseline,
                    })

    gain_df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for idx, ft in enumerate(FEATURES):
        ax = axes[idx]
        sub = gain_df[gain_df.feature_type == ft]
        aug_strats = [s for s in STRATEGIES if s != 'source_only']

        # 箱线图
        data = [sub[sub.strategy == s]['gain'].values for s in aug_strats]
        bp = ax.boxplot(data, labels=[STRAT_LABELS[s] for s in aug_strats],
                        patch_artist=True, showmeans=True)
        for patch, strat in zip(bp['boxes'], aug_strats):
            patch.set_facecolor(STRAT_COLORS.get(strat, 'gray'))
            patch.set_alpha(0.6)

        ax.set_title(FT_LABELS[ft])
        ax.set_ylabel('BA gain over source-only' if idx == 0 else '')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.axhline(y=0, color='red', linestyle='--', linewidth=1)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)

    fig.suptitle('Gain from Borrowing (relative to Source-only baseline)',
                 fontsize=12, y=1.02)
    fig.tight_layout()
    savefig(fig, 'exp6_gain_boxplot')


# ========================================================================
# Main
# ========================================================================
if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    df = load_all_data()
    print(f'Loaded {len(df)} rows from {len(SOURCES)} sources')

    plot_strategy_heatmap(df)
    plot_budget_curves(df)
    plot_strategy_votes(df)
    plot_ae_vs_s2(df)
    plot_gain_from_borrowing(df)

    print('\nAll figures saved to:', FIG_DIR)
