"""
Exp 6 完整分析 (v2): 10 sources × 13 strategies × 3 features × 5 budgets × 3 seeds

3 个分析维度：
1. Strategy 横向对比 (Pareto frontier)
2. AE vs S2 vs Fusion 对比
3. Prior Information Level (无 prior / cost-only / cost+transfer)
"""
import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

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
FT_COLORS = {'ae':'#2196F3','s2':'#FF9800','fusion':'#4CAF50'}

STRATEGIES_ALL = ['source_only','random','climate_similar','climate_diverse',
                  'representativeness','balanced','transfer_strong','transfer_weak',
                  'transfer_strong_weighted','transfer_weak_weighted',
                  'cost_aware_strong','cost_aware_weak','cost_only']

STRAT_GROUPS = {
    'Baselines': ['source_only','random','balanced'],
    'Geographic': ['climate_similar','climate_diverse','representativeness'],
    'Transfer-based': ['transfer_strong','transfer_weak',
                        'transfer_strong_weighted','transfer_weak_weighted'],
    'Cost-aware': ['cost_aware_strong','cost_aware_weak','cost_only'],
}

STRAT_COLORS = {
    'source_only':'#9E9E9E','random':'#1976D2','balanced':'#2E7D32',
    'climate_similar':'#D84315','climate_diverse':'#43A047',
    'representativeness':'#7CB342',
    'transfer_strong':'#E53935','transfer_weak':'#FDD835',
    'transfer_strong_weighted':'#8E24AA','transfer_weak_weighted':'#00838F',
    'cost_aware_strong':'#AD1457','cost_aware_weak':'#006064',
    'cost_only':'#BF360C',
}

STRAT_LABELS = {
    'source_only':'Source-only','random':'Random','balanced':'Balanced',
    'climate_similar':'Climate-similar','climate_diverse':'Climate-diverse',
    'representativeness':'Representativeness',
    'transfer_strong':'Transfer-strong','transfer_weak':'Transfer-weak',
    'transfer_strong_weighted':'Transfer-strong-wt',
    'transfer_weak_weighted':'Transfer-weak-wt',
    'cost_aware_strong':'Cost-aware-strong','cost_aware_weak':'Cost-aware-weak',
    'cost_only':'Cost-only',
}


def parse_sampling_info(info_str):
    if pd.isna(info_str) or info_str == '{}': return {}
    counts = {}
    try:
        matches = re.findall(
            r"['\"]([A-Z]{2})['\"]?\s*:\s*(?:np\.int64\()?(\d+)\)?",
            str(info_str))
        for c, n in matches:
            counts[c] = int(n)
    except: pass
    return counts


def compute_cost(info_str):
    counts = parse_sampling_info(info_str)
    return sum(n * COST.get(c, 1.0) for c, n in counts.items())


def load_all():
    dfs = []
    for s in SOURCES:
        for ft in FEATURES:
            f = os.path.join(OUT_DIR, f'scalable_{ft}_{s}_tier_b.csv')
            if not os.path.exists(f): continue
            df = pd.read_csv(f)
            df['source'] = s
            df['feature'] = ft
            df['aug_cost'] = df['sampling_info'].apply(compute_cost)
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


# =========================================================================
# Dimension 1: Strategy comparison
# =========================================================================
def plot_strategy_heatmap(df):
    """10 sources × 13 strategies × 3 features, at budget=10k."""
    print('\n[1] Strategy × Source Heatmap...')
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))

    for idx, ft in enumerate(FEATURES):
        ax = axes[idx]
        sub = df[(df.feature == ft) & (df.budget == 10000)]
        # 每 (source, strategy) 取 3 seed 均值
        pivot = sub.groupby(['source','strategy'])['mean_ood_ba_equal'].mean().reset_index()
        matrix = pivot.pivot(index='source', columns='strategy', values='mean_ood_ba_equal')
        matrix = matrix.reindex(index=SOURCES, columns=STRATEGIES_ALL)

        cmap = LinearSegmentedColormap.from_list('ba',
            ['#FFEBEE','#FFCDD2','#EF9A9A','#FF8A65','#FFA726','#66BB6A','#2E7D32'], N=256)
        im = ax.imshow(matrix.values, cmap=cmap, aspect='auto', vmin=0.25, vmax=0.55)
        ax.set_xticks(range(len(STRATEGIES_ALL)))
        ax.set_xticklabels([STRAT_LABELS[s] for s in STRATEGIES_ALL],
                           rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(len(SOURCES)))
        ax.set_yticklabels(SOURCES, fontsize=9)
        ax.set_xlabel('Strategy')
        if idx == 0: ax.set_ylabel('Source country')
        ax.set_title(FT_LABELS[ft], fontsize=12)

        # Annotate
        for i in range(len(SOURCES)):
            for j, s in enumerate(STRATEGIES_ALL):
                v = matrix.iloc[i, j]
                if pd.notna(v):
                    ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                            fontsize=7, color='white' if v > 0.42 else 'black')

    fig.suptitle('Strategy Performance by Source Country (budget=10k, mean of 3 seeds)',
                 fontsize=13, y=1.00)
    plt.colorbar(im, ax=axes, shrink=0.6, aspect=30, label='Mean OOD BA')
    fig.tight_layout()
    for ext in ['png','pdf']:
        fig.savefig(os.path.join(FIG_DIR, f'exp6_v2_strategy_heatmap.{ext}'))
    plt.close(fig)
    print('  [OK] exp6_v2_strategy_heatmap')


# =========================================================================
# Dimension 2: AE vs S2 vs Fusion
# =========================================================================
def plot_feature_comparison(df):
    """对每个策略，比较 AE/S2/Fusion 在不同 budget 下的表现。"""
    print('\n[2] Feature Comparison (AE vs S2 vs Fusion)...')

    # 选 6 个关键策略
    key_strats = ['balanced','random','cost_aware_weak','cost_only',
                   'climate_similar','transfer_strong']

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, strat in enumerate(key_strats):
        ax = axes[idx]
        for ft in FEATURES:
            sub = df[(df.strategy == strat) & (df.feature == ft)]
            grp = sub.groupby('budget')['mean_ood_ba_equal'].agg(['mean','std']).reset_index()
            ax.errorbar(grp['budget'].clip(lower=500), grp['mean'], yerr=grp['std'],
                        marker='o', markersize=5, color=FT_COLORS[ft],
                        linewidth=1.8, capsize=3, label=FT_LABELS[ft])
        ax.set_xscale('log')
        ax.set_xlabel('Budget (samples)')
        ax.set_ylabel('Mean OOD BA')
        ax.set_title(STRAT_LABELS[strat], fontsize=11)
        ax.grid(alpha=0.3, linestyle='--')
        ax.legend(fontsize=9, loc='lower right')

    fig.suptitle('Feature Comparison across Strategies (averaged over 10 sources)',
                 fontsize=13, y=1.00)
    fig.tight_layout()
    for ext in ['png','pdf']:
        fig.savefig(os.path.join(FIG_DIR, f'exp6_v2_feature_comparison.{ext}'))
    plt.close(fig)
    print('  [OK] exp6_v2_feature_comparison')


def analyze_fusion_dominance(df):
    """Fusion 在多少配置下胜过 max(AE, S2)."""
    print('\n[3] Fusion Dominance Analysis...')

    # 对每 (source, strategy, budget) 计算 fusion vs max(ae, s2)
    pivot = df.groupby(['source','strategy','budget','feature'])['mean_ood_ba_equal'].mean().reset_index()
    pivot = pivot.pivot(index=['source','strategy','budget'], columns='feature',
                        values='mean_ood_ba_equal').reset_index()
    pivot['max_single'] = pivot[['ae','s2']].max(axis=1)
    pivot['fusion_gap'] = pivot['fusion'] - pivot['max_single']
    pivot['fusion_wins'] = pivot['fusion_gap'] > 0

    total = len(pivot)
    wins = pivot['fusion_wins'].sum()
    mean_gap = pivot['fusion_gap'].mean()

    print(f'  Total configs: {total}')
    print(f'  Fusion wins: {wins} ({wins/total*100:.1f}%)')
    print(f'  Mean Fusion gap: +{mean_gap:.4f}')
    print(f'  Min/Max gap: {pivot.fusion_gap.min():+.4f} / {pivot.fusion_gap.max():+.4f}')

    # 按 strategy 分组看
    print('\n  Fusion dominance by strategy:')
    for s in STRATEGIES_ALL:
        sub = pivot[pivot.strategy == s]
        if len(sub) == 0: continue
        wins = sub['fusion_wins'].sum()
        total_s = len(sub)
        print(f'    {STRAT_LABELS[s]:25s}: {wins}/{total_s} ({wins/total_s*100:.0f}%), gap={sub.fusion_gap.mean():+.4f}')


def analyze_ae_vs_s2(df):
    """AE vs S2 谁更好？"""
    print('\n[4] AE vs S2 Analysis...')
    pivot = df.groupby(['source','strategy','budget','feature'])['mean_ood_ba_equal'].mean().reset_index()
    pivot = pivot.pivot(index=['source','strategy','budget'], columns='feature',
                        values='mean_ood_ba_equal').reset_index()
    pivot['ae_vs_s2'] = pivot['ae'] - pivot['s2']

    total = len(pivot)
    ae_wins = (pivot['ae_vs_s2'] > 0).sum()
    print(f'  Total configs: {total}')
    print(f'  AE wins vs S2: {ae_wins} ({ae_wins/total*100:.1f}%)')
    print(f'  Mean gap (AE-S2): {pivot.ae_vs_s2.mean():+.4f}')

    # 按 source 分组
    print('\n  AE vs S2 by source:')
    for s in SOURCES:
        sub = pivot[pivot.source == s]
        print(f'    {s}: AE wins {sub.ae_vs_s2.gt(0).sum()}/{len(sub)}, '
              f'mean gap={sub.ae_vs_s2.mean():+.4f}')


# =========================================================================
# Dimension 3: Prior Information Level (Analysis B)
# =========================================================================
def plot_pareto_frontier(df):
    """每个 feature 画 BA vs cost，展示 Pareto frontier."""
    print('\n[5] Pareto Frontier (BA vs Cost)...')

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    key_strats = ['balanced','random','representativeness',
                   'cost_only','cost_aware_strong','cost_aware_weak',
                   'climate_similar','transfer_strong']

    for idx, ft in enumerate(FEATURES):
        ax = axes[idx]
        for strat in key_strats:
            sub = df[(df.feature == ft) & (df.strategy == strat) & (df.budget > 0)]
            grp = sub.groupby('budget').agg(
                cost=('aug_cost','mean'),
                ba=('mean_ood_ba_equal','mean'),
                ba_std=('mean_ood_ba_equal','std'),
            ).reset_index()

            marker = 'o'
            if 'cost_aware' in strat or 'cost_only' in strat:
                marker = '*'
                lw = 2.5
                size = 10
            else:
                lw = 1.5
                size = 6

            ax.plot(grp['cost'], grp['ba'], marker=marker, markersize=size,
                    color=STRAT_COLORS.get(strat, 'gray'), linewidth=lw,
                    label=STRAT_LABELS[strat])

        ax.set_xscale('log')
        ax.set_xlabel('Wage-weighted Cost (log)', fontsize=11)
        ax.set_ylabel('Mean OOD BA')
        ax.set_title(FT_LABELS[ft], fontsize=12)
        ax.grid(alpha=0.3, linestyle='--')
        if idx == 2:
            ax.legend(fontsize=8, loc='lower right')

    fig.suptitle('Pareto Frontier: BA vs Cost (averaged over 10 sources, budgets 1k-50k)',
                 fontsize=13, y=1.01)
    fig.tight_layout()
    for ext in ['png','pdf']:
        fig.savefig(os.path.join(FIG_DIR, f'exp6_v2_pareto.{ext}'))
    plt.close(fig)
    print('  [OK] exp6_v2_pareto')


def analysis_b_prior_value(df):
    """Analysis B: value of adding transfer matrix prior."""
    print('\n[6] Analysis B: Value of Transfer Prior...')

    # Compare: balanced / cost_only / cost_aware_strong
    # These represent: no info / cost only / cost + transfer prior
    levels = {
        'No prior (Balanced)': 'balanced',
        'Cost only': 'cost_only',
        'Cost + Transfer prior': 'cost_aware_strong',
        'Cost + Inverse Transfer': 'cost_aware_weak',
    }

    for ft in FEATURES:
        print(f'\n  {FT_LABELS[ft]}:')
        for lbl, strat in levels.items():
            sub = df[(df.feature == ft) & (df.strategy == strat) & (df.budget == 10000)]
            mean_ba = sub['mean_ood_ba_equal'].mean()
            mean_cost = sub['aug_cost'].mean()
            print(f'    {lbl:28s} BA={mean_ba:.4f}  cost={mean_cost:>7.0f}')

    # Pareto dominance table: 每 feature 哪些策略 Pareto-dominate balanced
    print('\n  Pareto-dominance vs Balanced (across 10 sources, mean):')
    for ft in FEATURES:
        bal = df[(df.feature == ft) & (df.strategy == 'balanced') & (df.budget == 10000)]
        bal_ba = bal['mean_ood_ba_equal'].mean()
        bal_cost = bal['aug_cost'].mean()
        print(f'\n  {FT_LABELS[ft]} @ budget=10k:')
        print(f'    Balanced baseline: BA={bal_ba:.4f}, cost={bal_cost:.0f}')
        for strat in STRATEGIES_ALL:
            if strat in ['source_only','balanced']: continue
            sub = df[(df.feature == ft) & (df.strategy == strat) & (df.budget == 10000)]
            if len(sub) == 0: continue
            strat_ba = sub['mean_ood_ba_equal'].mean()
            strat_cost = sub['aug_cost'].mean()
            ba_diff = strat_ba - bal_ba
            cost_diff = (strat_cost - bal_cost) / bal_cost * 100
            # Pareto dominant: BA >= baseline AND cost <= baseline
            pareto = (ba_diff >= -0.002) and (strat_cost < bal_cost)
            flag = '✓ PARETO' if pareto else ''
            print(f'    {STRAT_LABELS[strat]:25s}: BA={strat_ba:.4f} ({ba_diff:+.4f}), '
                  f'cost={strat_cost:.0f} ({cost_diff:+.1f}%) {flag}')


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    df = load_all()
    print(f'Loaded {len(df)} rows from {df.source.nunique()} sources '
          f'× {df.feature.nunique()} features × {df.strategy.nunique()} strategies')

    plot_strategy_heatmap(df)
    plot_feature_comparison(df)
    analyze_fusion_dominance(df)
    analyze_ae_vs_s2(df)
    plot_pareto_frontier(df)
    analysis_b_prior_value(df)

    print('\nAll analyses complete!')
