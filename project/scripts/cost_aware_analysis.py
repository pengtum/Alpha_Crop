"""
Cost-aware analysis of borrowing strategies.
用欧盟 2021 最低工资作为 per-sample labeling cost proxy。
计算每种策略的总成本，对比 cost-quality tradeoff。
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

# EU 2021 minimum monthly wage (EUR), rough approximation
# Reference: Eurostat, countries without statutory minimum wage use average agricultural wage
COST_PER_SAMPLE = {
    'AT': 1.05,   # 1700 EUR
    'BE': 1.00,   # 1626 EUR
    'DE': 1.00,   # 1621 EUR (reference)
    'EE': 0.36,   # 584 EUR
    'LT': 0.40,   # 642 EUR
    'LV': 0.31,   # 500 EUR
    'PT': 0.48,   # 776 EUR
    'SE': 1.36,   # 2200 EUR (avg)
    'SI': 0.63,   # 1024 EUR
    'SK': 0.38,   # 623 EUR
}

SOURCES = list(COST_PER_SAMPLE.keys())
STRATEGIES = ['source_only', 'random', 'climate_similar', 'climate_diverse',
              'representativeness', 'balanced',
              'transfer_strong', 'transfer_weak',
              'transfer_strong_weighted', 'transfer_weak_weighted']
FEATURES = ['ae', 's2', 'fusion']
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
    'transfer_strong_weighted': '#8E24AA',
    'transfer_weak_weighted': '#00838F',
}

STRAT_LABELS = {
    'source_only': 'Source only', 'random': 'Random',
    'climate_similar': 'Climate-similar', 'climate_diverse': 'Climate-diverse',
    'representativeness': 'Representativeness', 'balanced': 'Balanced',
    'transfer_strong': 'Transfer-strong', 'transfer_weak': 'Transfer-weak',
    'transfer_strong_weighted': 'Transfer-strong-wt',
    'transfer_weak_weighted': 'Transfer-weak-wt',
}


def parse_sampling_info(info_str):
    """从 sampling_info 字符串解析每国样本数 dict。

    Format: "{'BE': np.int64(111), 'DE': np.int64(111), ...}"
    or: "{'BE': 111, ...}"
    """
    if pd.isna(info_str) or info_str == '{}':
        return {}
    counts = {}
    try:
        # 匹配 'XX': [np.int64(]N[)] 两种格式
        matches = re.findall(
            r"['\"]([A-Z]{2})['\"]?\s*:\s*(?:np\.int64\()?(\d+)\)?",
            str(info_str))
        for country, n in matches:
            counts[country] = int(n)
    except Exception:
        pass
    return counts


def compute_cost(sampling_info, source_country):
    """计算总标注成本（单位：CostPerSample-equivalent）。

    Source 国样本总是全部使用（假设已有标注，cost = sum(source pool) * source_cost）。
    Augmentation samples 按各国 cost 加权。
    """
    counts = parse_sampling_info(sampling_info)
    # 只计算 augmentation 成本（不含 source，因为 source 对所有策略一样）
    aug_cost = sum(n * COST_PER_SAMPLE.get(c, 1.0) for c, n in counts.items())
    return aug_cost


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
            # 计算 cost
            df['aug_cost'] = df.apply(
                lambda r: compute_cost(r.get('sampling_info', '{}'), s),
                axis=1)
            all_data.append(df)
    combined = pd.concat(all_data, ignore_index=True)
    return combined


def analysis_cost_ratios(df):
    """每策略的平均 cost/budget 比率。"""
    print('=== Strategy Cost Ratios (avg cost per sample) ===\n')
    sub = df[df.strategy != 'source_only'].copy()
    sub['cost_per_sample'] = sub['aug_cost'] / sub['n_aug'].clip(lower=1)

    # 按策略分组
    summary = sub.groupby('strategy')['cost_per_sample'].agg(['mean', 'std']).round(3)
    print('Strategy'.ljust(30) + f'{"Avg cost/sample":>18s} {"Std":>8s}')
    for strat in STRATEGIES:
        if strat in summary.index:
            r = summary.loc[strat]
            print(f'{STRAT_LABELS.get(strat, strat):28s} {r["mean"]:>10.3f}  {r["std"]:>8.3f}')
    print()


def plot_cost_quality_tradeoff(df):
    """图：BA vs augmentation cost (不是 budget)。"""
    print('Plotting cost-quality tradeoff...')

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    strategies_show = ['random', 'balanced', 'climate_diverse',
                       'representativeness', 'transfer_strong_weighted',
                       'transfer_weak_weighted']

    for idx, ft in enumerate(FEATURES):
        ax = axes[idx]
        sub = df[(df.feature_type_norm == ft) & (df.strategy != 'source_only')]

        for strat in strategies_show:
            strat_data = sub[sub.strategy == strat]
            if len(strat_data) == 0:
                continue
            # 按 budget 分组，取 cost 和 BA 的均值
            grp = strat_data.groupby('budget').agg(
                mean_cost=('aug_cost', 'mean'),
                mean_ba=('mean_ood_ba_equal', 'mean')
            ).reset_index()
            ax.plot(grp['mean_cost'], grp['mean_ba'],
                    marker='o', markersize=5,
                    color=STRAT_COLORS.get(strat, 'gray'), linewidth=1.5,
                    label=STRAT_LABELS[strat])

        ax.set_xlabel('Augmentation Cost (wage-weighted samples)')
        ax.set_ylabel('Mean OOD BA' if idx == 0 else '')
        ax.set_title({'ae': 'AlphaEarth', 's2': 'Sentinel-2',
                      'fusion': 'Fusion'}[ft])
        ax.grid(alpha=0.3, linestyle='--')
        if idx == 2:
            ax.legend(loc='lower right', fontsize=8, frameon=True)
        ax.set_xscale('log')

    fig.suptitle('Cost-Quality Tradeoff: BA vs Labeling Cost '
                 '(weighted by country wage)', fontsize=12, y=1.02)
    fig.tight_layout()

    for ext in ['png', 'pdf']:
        fig.savefig(os.path.join(FIG_DIR, f'exp6_cost_quality.{ext}'))
    plt.close(fig)
    print('  [OK] exp6_cost_quality')


def table_cost_to_target_ba(df, target_ba=0.45):
    """找到达到 target BA 的最小 cost 策略。"""
    print(f'\n=== Minimum cost to achieve BA >= {target_ba} ===\n')
    results = []
    for ft in FEATURES:
        for strat in STRATEGIES:
            if strat == 'source_only':
                continue
            sub = df[(df.feature_type_norm == ft)
                     & (df.strategy == strat)]
            if len(sub) == 0:
                continue
            grp = sub.groupby('budget').agg(
                mean_cost=('aug_cost', 'mean'),
                mean_ba=('mean_ood_ba_equal', 'mean')
            ).reset_index().sort_values('budget')
            # 找第一个达到 target 的 budget
            reached = grp[grp.mean_ba >= target_ba]
            if len(reached) == 0:
                min_cost = np.inf
                best_ba = grp.mean_ba.max()
            else:
                min_cost = reached.iloc[0].mean_cost
                best_ba = reached.iloc[0].mean_ba
            results.append({
                'feature': ft, 'strategy': strat,
                'min_cost_to_reach': min_cost,
                'best_ba': best_ba,
            })
    res_df = pd.DataFrame(results)
    res_df.to_csv(os.path.join(OUT_DIR, f'cost_to_reach_ba{target_ba}.csv'),
                  index=False)

    for ft in FEATURES:
        sub = res_df[res_df.feature == ft].sort_values('min_cost_to_reach')
        print(f'\n{ft.upper()} (target BA={target_ba}):')
        for _, r in sub.iterrows():
            if r.min_cost_to_reach == np.inf:
                print(f'  {STRAT_LABELS.get(r.strategy, r.strategy):28s} NOT REACHED (max BA = {r.best_ba:.3f})')
            else:
                print(f'  {STRAT_LABELS.get(r.strategy, r.strategy):28s} cost = {r.min_cost_to_reach:>8.0f} (BA = {r.best_ba:.3f})')
    print()
    return res_df


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    df = load_all_data()
    print(f'Loaded {len(df)} rows\n')

    analysis_cost_ratios(df)
    plot_cost_quality_tradeoff(df)
    table_cost_to_target_ba(df, target_ba=0.45)
    table_cost_to_target_ba(df, target_ba=0.48)
