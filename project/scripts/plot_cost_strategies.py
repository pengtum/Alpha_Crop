"""绘制 cost_only vs cost_aware_* 的 Pareto 对比图。

两张图：
  1. Pareto frontier（3 feature 子图，每图 4 条线，budget 连线）
  2. Per-source bar chart @ budget=10k (Fusion)
"""
import os, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'font.family': 'Arial',
    'font.size': 10, 'figure.dpi': 150, 'savefig.dpi': 150,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.1,
})

OUT_DIR = 'results/experiments/exp6_scalable_mapping'
FIG_DIR = 'Manuscript/Paper_Alpha_Earth/0001_draft_v1/figures'
COST = {'AT':1.05,'BE':1.00,'DE':1.00,'EE':0.36,'LT':0.40,'LV':0.31,
        'PT':0.48,'SE':1.36,'SI':0.63,'SK':0.38}
SOURCES = ['AT','BE','DE','EE','LT','LV','PT','SE','SI','SK']
FEATURES = ['ae','s2','fusion']
FT_LABELS = {'ae':'AlphaEarth','s2':'Sentinel-2','fusion':'Fusion'}
BUDGETS = [1000, 5000, 10000, 20000, 50000]

STRAT_STYLE = {
    'balanced':          {'c':'#2E7D32','m':'s','ls':'--','lbl':'Balanced (no prior)'},
    'cost_only':         {'c':'#BF360C','m':'*','ls':'-', 'lbl':'Cost-only (no transfer prior)'},
    'cost_aware_strong': {'c':'#AD1457','m':'^','ls':'-', 'lbl':'Cost-aware-strong (cost + BA)'},
    'cost_aware_weak':   {'c':'#006064','m':'v','ls':'-', 'lbl':'Cost-aware-weak (cost + 1-BA)'},
}


def parse_info(s):
    if pd.isna(s) or s == '{}': return {}
    out = {}
    for c, n in re.findall(r"['\"]([A-Z]{2})['\"]?\s*:\s*(?:np\.int64\()?(\d+)\)?", str(s)):
        out[c] = int(n)
    return out


def compute_cost(s):
    return sum(n * COST.get(c, 1.0) for c, n in parse_info(s).items())


def load():
    dfs = []
    for s in SOURCES:
        for ft in FEATURES:
            f = os.path.join(OUT_DIR, f'scalable_{ft}_{s}_tier_b.csv')
            if not os.path.exists(f): continue
            df = pd.read_csv(f)
            df['source'], df['feature'] = s, ft
            df['aug_cost'] = df['sampling_info'].apply(compute_cost)
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def plot_pareto(df):
    """Fig A: 3 子图（3 feature），每图 4 策略，点=budget。"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), sharey=False)
    for idx, ft in enumerate(FEATURES):
        ax = axes[idx]
        for strat, st in STRAT_STYLE.items():
            grp = df[(df.feature==ft)&(df.strategy==strat)&(df.budget>0)].groupby('budget').agg(
                cost=('aug_cost','mean'),
                ba=('mean_ood_ba_equal','mean'),
                ba_se=('mean_ood_ba_equal', lambda x: x.std()/np.sqrt(len(x))),
            ).reset_index()
            ax.errorbar(grp['cost'], grp['ba'], yerr=grp['ba_se'],
                        marker=st['m'], markersize=9, color=st['c'],
                        linewidth=1.8, linestyle=st['ls'], capsize=3,
                        label=st['lbl'], markeredgecolor='white', markeredgewidth=0.5)
            # 标注 budget（仅 cost_only 标，避免拥挤）
            if strat == 'cost_only':
                for _, row in grp.iterrows():
                    b_lbl = f'{int(row.budget/1000)}k' if row.budget >= 1000 else str(int(row.budget))
                    ax.annotate(b_lbl, (row['cost'], row['ba']),
                                xytext=(5, -10), textcoords='offset points',
                                fontsize=7.5, color='#6a1b00')
        ax.set_xscale('log')
        ax.set_xlabel('Wage-weighted labeling cost (log scale)', fontsize=10)
        if idx == 0:
            ax.set_ylabel('Mean OOD Balanced Accuracy', fontsize=10)
        ax.set_title(FT_LABELS[ft], fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        if idx == 2:
            ax.legend(loc='lower right', fontsize=8.5, framealpha=0.9)
    fig.suptitle(
        'Pareto Frontier: Adding transfer prior to cost-aware sampling barely helps\n'
        '(averaged over 10 source countries × 3 seeds; budgets = 1k, 5k, 10k, 20k, 50k)',
        fontsize=11.5, y=1.02)
    fig.tight_layout()
    for ext in ['png','pdf']:
        fig.savefig(os.path.join(FIG_DIR, f'exp6_v2_cost_strategies_pareto.{ext}'))
    plt.close(fig)
    print('[OK] exp6_v2_cost_strategies_pareto')


def plot_per_source_bar(df, budget=10000, feature='fusion'):
    """Fig B: 每 source 下 4 策略的 BA + Cost（双轴 bar）。"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    strats = list(STRAT_STYLE.keys())
    x = np.arange(len(SOURCES))
    w = 0.2

    for i, strat in enumerate(strats):
        ba_vals, cost_vals = [], []
        for src in SOURCES:
            sub = df[(df.source==src)&(df.feature==feature)&
                     (df.strategy==strat)&(df.budget==budget)]
            ba_vals.append(sub.mean_ood_ba_equal.mean() if len(sub)>0 else np.nan)
            cost_vals.append(sub.aug_cost.mean() if len(sub)>0 else np.nan)
        st = STRAT_STYLE[strat]
        ax1.bar(x + (i-1.5)*w, ba_vals, w, color=st['c'],
                label=st['lbl'], edgecolor='white', linewidth=0.5)
        ax2.bar(x + (i-1.5)*w, cost_vals, w, color=st['c'],
                edgecolor='white', linewidth=0.5)

    ax1.set_ylabel('Mean OOD BA', fontsize=11)
    ax1.set_title(f'Per-source comparison ({FT_LABELS[feature]}, budget={budget:,})',
                  fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.legend(ncol=4, fontsize=8.5, loc='upper center',
               bbox_to_anchor=(0.5, 1.18), framealpha=0.9)
    ax1.set_ylim(0.30, max(0.50, ax1.get_ylim()[1]))

    ax2.set_ylabel('Wage-weighted Cost', fontsize=11)
    ax2.set_xlabel('Source country', fontsize=11)
    ax2.set_xticks(x); ax2.set_xticklabels(SOURCES)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    fig.tight_layout()
    for ext in ['png','pdf']:
        fig.savefig(os.path.join(FIG_DIR, f'exp6_v2_cost_strategies_by_source.{ext}'))
    plt.close(fig)
    print('[OK] exp6_v2_cost_strategies_by_source')


def plot_info_ladder(df, budget=10000):
    """Fig C: "Information Ladder" — balanced → cost_only → cost_aware_*
    用 3 个 feature 聚合展示增量"""
    fig, ax = plt.subplots(figsize=(9, 5))
    strats_ord = ['balanced','cost_only','cost_aware_weak','cost_aware_strong']
    labels = ['Balanced\n(no info)', 'Cost-only\n(+cost)',
              'Cost-aware-weak\n(+cost+1-BA)', 'Cost-aware-strong\n(+cost+BA)']
    ft_c = {'ae':'#2196F3','s2':'#FF9800','fusion':'#4CAF50'}

    x = np.arange(len(strats_ord))
    w = 0.25
    for i, ft in enumerate(FEATURES):
        ba_vals, cost_vals = [], []
        for strat in strats_ord:
            sub = df[(df.feature==ft)&(df.strategy==strat)&(df.budget==budget)]
            ba_vals.append(sub.mean_ood_ba_equal.mean())
            cost_vals.append(sub.aug_cost.mean())
        # BA bars（主轴）
        bars = ax.bar(x + (i-1)*w, ba_vals, w, color=ft_c[ft],
                      label=FT_LABELS[ft], edgecolor='white', linewidth=0.5)
        # 文本标注 cost 相对 balanced 变化
        for j, (bar, c) in enumerate(zip(bars, cost_vals)):
            dc_pct = (c - cost_vals[0]) / cost_vals[0] * 100
            if j == 0:
                txt = 'ref'
            else:
                txt = f'{dc_pct:+.0f}%'
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
                    txt, ha='center', fontsize=7.5, color='#333')

    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Mean OOD BA', fontsize=11)
    ax.set_title(f'Information Ladder @ budget={budget:,} (bars=BA, text=ΔCost% vs Balanced)',
                 fontsize=11.5, fontweight='bold')
    ax.legend(title='Feature', fontsize=9, loc='lower right')
    ax.set_ylim(0.36, 0.46)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    fig.tight_layout()
    for ext in ['png','pdf']:
        fig.savefig(os.path.join(FIG_DIR, f'exp6_v2_information_ladder.{ext}'))
    plt.close(fig)
    print('[OK] exp6_v2_information_ladder')


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    df = load()
    print(f'Loaded {len(df)} rows')
    plot_pareto(df)
    plot_per_source_bar(df, 10000, 'fusion')
    plot_info_ladder(df, 10000)
    print('\nAll figures saved to', FIG_DIR)
