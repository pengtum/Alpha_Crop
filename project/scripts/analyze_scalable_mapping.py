"""
聚合分析 Exp 6 scalable mapping：9 source × 8 strategy × 5 budget × 3 feature
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

plt.rcParams.update({
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'font.family': 'Arial',
    'font.size': 10, 'figure.dpi': 150, 'savefig.dpi': 150,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.1,
})

OUT_DIR = 'results/experiments/exp6_scalable_mapping'
SOURCES = ['AT', 'BE', 'DE', 'EE', 'LT', 'LV', 'PT', 'SE', 'SI', 'SK']
STRATEGIES = ['source_only', 'random', 'climate_similar', 'climate_diverse',
              'representativeness', 'balanced',
              'transfer_strong', 'transfer_weak']
STRAT_LABELS = {
    'source_only': 'Source only',
    'random': 'Random',
    'climate_similar': 'Climate-similar',
    'climate_diverse': 'Climate-diverse',
    'representativeness': 'Representativeness',
    'balanced': 'Balanced',
    'transfer_strong': 'Transfer-strong',
    'transfer_weak': 'Transfer-weak',
}
FEATURES = ['ae', 's2', 'fusion']
FT_LABELS = {'ae': 'AlphaEarth', 's2': 'Sentinel-2', 'fusion': 'Fusion'}


def load_all_data():
    """加载所有 source × feature 的结果。"""
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
    combined = pd.concat(all_data, ignore_index=True)
    print(f'Loaded {len(combined)} rows total')
    return combined


def analysis_1_strategy_ranking(df):
    """每个 source × feature 下的策略排名（用 max budget）。"""
    print('\n=== Analysis 1: Strategy ranking per source ===\n')
    max_budget = 50000
    sub = df[(df['budget'] == max_budget) | (df['strategy'] == 'source_only')]

    # 每 source × feature × strategy 的平均 BA
    summary = sub.groupby(['source', 'feature_type_norm', 'strategy'])[
        'mean_ood_ba_equal'].mean().reset_index()

    # 保存
    summary.to_csv(os.path.join(OUT_DIR, 'aggregate_strategy_ranking_max_budget.csv'),
                   index=False)
    print(f'Saved aggregate_strategy_ranking_max_budget.csv')

    # 对每个 feature，输出 source × strategy 的表
    for ft in FEATURES:
        print(f'\n--- {FT_LABELS[ft]} (budget={max_budget} for aug strategies) ---')
        pivot = summary[summary.feature_type_norm == ft].pivot(
            index='source', columns='strategy', values='mean_ood_ba_equal')
        # 重排列顺序
        cols_order = [c for c in STRATEGIES if c in pivot.columns]
        pivot = pivot[cols_order]
        print(pivot.round(3).to_string())

    return summary


def analysis_2_winner_per_source(summary):
    """每个 source 的最优策略是什么。"""
    print('\n=== Analysis 2: Best strategy per source ===\n')

    for ft in FEATURES:
        sub = summary[summary.feature_type_norm == ft]
        # 去掉 source_only 作为候选最优（它是 baseline）
        aug = sub[sub.strategy != 'source_only']
        best_per_source = aug.loc[aug.groupby('source')['mean_ood_ba_equal'].idxmax()]
        print(f'\n--- {FT_LABELS[ft]} ---')
        for _, r in best_per_source.iterrows():
            src_only = sub[(sub.source == r.source) & (sub.strategy == 'source_only')][
                'mean_ood_ba_equal'].values[0]
            gain = r.mean_ood_ba_equal - src_only
            print(f'  {r.source}: BEST = {r.strategy:22s} '
                  f'BA={r.mean_ood_ba_equal:.4f} (gain +{gain:.3f})')


def analysis_3_strategy_votes(summary):
    """统计每个策略在多少 source 下是 top-3 / best。"""
    print('\n=== Analysis 3: Strategy popularity (votes) ===\n')

    vote_data = []
    for ft in FEATURES:
        sub = summary[summary.feature_type_norm == ft]
        aug = sub[sub.strategy != 'source_only']
        for src in SOURCES:
            src_sub = aug[aug.source == src].sort_values(
                'mean_ood_ba_equal', ascending=False)
            if len(src_sub) == 0:
                continue
            for rank, (_, r) in enumerate(src_sub.iterrows()):
                vote_data.append({
                    'feature_type': ft,
                    'source': src,
                    'strategy': r.strategy,
                    'rank': rank + 1,
                    'ba': r.mean_ood_ba_equal,
                })

    vote_df = pd.DataFrame(vote_data)
    vote_df.to_csv(os.path.join(OUT_DIR, 'aggregate_strategy_votes.csv'),
                   index=False)

    # 每个策略在每个 feature 下，各 rank 的出现次数
    for ft in FEATURES:
        print(f'\n--- {FT_LABELS[ft]}: rank frequency per strategy ---')
        sub = vote_df[vote_df.feature_type == ft]
        cross = sub.pivot_table(
            index='strategy', columns='rank', values='source',
            aggfunc='count', fill_value=0)
        # 重排
        strat_order = [s for s in STRATEGIES if s != 'source_only']
        cross = cross.reindex(strat_order)
        cross.columns = [f'rank_{c}' for c in cross.columns]
        print(cross.to_string())


def analysis_4_per_target(df):
    """每个 target 国在各 source 下的最优策略。"""
    print('\n=== Analysis 4: Per-target best source+strategy ===\n')
    countries_except_ro = ['AT', 'BE', 'DE', 'EE', 'LT', 'LV', 'PT', 'SE', 'SI', 'SK']

    # 取 max budget，展开 per-country BA
    max_budget = 50000
    sub = df[(df['budget'] == max_budget) | (df['strategy'] == 'source_only')]

    # 对每个 feature，构建 target × source × strategy 的大表
    for ft in ['fusion']:  # 只展示 Fusion
        print(f'\n--- {FT_LABELS[ft]}: Best (source, strategy) per target country ---')
        data_ft = sub[sub.feature_type_norm == ft]

        for target in countries_except_ro:
            col = f'ba_{target}'
            if col not in data_ft.columns:
                continue
            valid = data_ft[data_ft.source != target].dropna(subset=[col])
            if len(valid) == 0:
                continue
            mean_ba = valid.groupby(['source', 'strategy'])[col].mean().reset_index()
            best = mean_ba.loc[mean_ba[col].idxmax()]
            worst = mean_ba.loc[mean_ba[col].idxmin()]
            print(f'  Target {target}: BEST = ({best.source}, {best.strategy:18s}) BA={best[col]:.3f} | '
                  f'WORST = ({worst.source}, {worst.strategy:18s}) BA={worst[col]:.3f}')


def analysis_5_ae_vs_s2(summary):
    """AE vs S2 最优策略对比。"""
    print('\n=== Analysis 5: AE vs S2 optimal strategy comparison ===\n')

    comparison = []
    for src in SOURCES:
        row = {'source': src}
        for ft in ['ae', 's2', 'fusion']:
            sub = summary[(summary.source == src)
                          & (summary.feature_type_norm == ft)
                          & (summary.strategy != 'source_only')]
            if len(sub) > 0:
                best = sub.loc[sub['mean_ood_ba_equal'].idxmax()]
                row[f'{ft}_best'] = best.strategy
                row[f'{ft}_ba'] = best.mean_ood_ba_equal
        comparison.append(row)

    comp_df = pd.DataFrame(comparison)
    comp_df.to_csv(os.path.join(OUT_DIR, 'aggregate_ae_vs_s2.csv'), index=False)
    print(comp_df.to_string())

    # 策略一致性
    ae_s2_same = (comp_df['ae_best'] == comp_df['s2_best']).sum()
    ae_fu_same = (comp_df['ae_best'] == comp_df['fusion_best']).sum()
    s2_fu_same = (comp_df['s2_best'] == comp_df['fusion_best']).sum()
    print(f'\n  AE best == S2 best: {ae_s2_same}/{len(comp_df)} sources')
    print(f'  AE best == Fusion best: {ae_fu_same}/{len(comp_df)} sources')
    print(f'  S2 best == Fusion best: {s2_fu_same}/{len(comp_df)} sources')


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    df = load_all_data()

    summary = analysis_1_strategy_ranking(df)
    analysis_2_winner_per_source(summary)
    analysis_3_strategy_votes(summary)
    analysis_4_per_target(df)
    analysis_5_ae_vs_s2(summary)

    print('\n\nAll analyses complete. CSV outputs saved to:', OUT_DIR)
