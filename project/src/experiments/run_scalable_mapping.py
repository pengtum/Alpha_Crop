"""
Exp 6: Scalable Mapping — Budget-Constrained Augmentation from a Single-Country Start

研究问题 (RQ4)：如果我只有一个国家 (DE) 的数据，希望用于全欧范围制图，
应该如何从其他国家借用样本来扩充训练集？不同采样策略在泛欧 OOD 上的表现差异如何？
AE 和 S2 的最优策略是否相同？

设计（受 Betti et al. AAAI 2026 "Mapping on a Budget" 启发）：
- Source: DE 全部训练样本
- Test: 其他 10 国各 10k 样本 (total 100k OOD test)
- Augmentation pool: 其他 10 国剩余样本
- 5 种采样策略（除 baseline）

8 种策略：
  S1: source_only         - 只用 DE（baseline）
  S2: random              - 从 10 国混合池随机抽 N
  S3: climate_similar     - 优先从同气候带（DE=Atlantic）国家抽
  S4: climate_diverse     - 优先从不同气候带国家抽
  S5: representativeness  - Betti-style: 每气候带均衡采样
  S6: balanced            - 每国均分 N/9
  S7: transfer_strong     - 基于 Exp 2b 迁移矩阵：从 DE 迁移性最强的国家抽（exploit）
  S8: transfer_weak       - 基于 Exp 2b 迁移矩阵：从 DE 迁移性最弱的国家抽（补短板）

Usage:
    python -m src.experiments.run_scalable_mapping
"""
import sys
import os
import argparse
import numpy as np
import pandas as pd
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from src.experiments.experiment_utils import (
    load_eu_features, prepare_xy, get_trainer, build_suffix,
    evaluate_predictions, filter_coords,
    save_experiment_results, save_results_csv, print_metrics,
    RANDOM_STATE, EXPERIMENTS_DIR, COL_COUNTRY, COUNTRIES_2021 as _C2021_FULL,
)

EXP_NAME = 'exp6_scalable_mapping'

# 排除 RO（仅 3 类，极端类别缺失使得迁移评估不公平）
COUNTRIES_2021 = [c for c in _C2021_FULL if c != 'RO']

# 气候带
COUNTRY_ZONE = {
    'AT': 'continental', 'BE': 'atlantic', 'DE': 'atlantic',
    'EE': 'nordic_baltic', 'LT': 'nordic_baltic', 'LV': 'nordic_baltic',
    'PT': 'mediterranean', 'SE': 'nordic_baltic',
    'SI': 'mediterranean', 'SK': 'continental',
}

# 各国标注成本（代理：EU 2021 最低工资，以 DE=1.0 归一化）
# 用于 cost-aware 采样策略
COUNTRY_COST = {
    'AT': 1.05,   # ~1700 EUR/月
    'BE': 1.00,   # ~1626 EUR/月
    'DE': 1.00,   # ~1621 EUR/月 (reference)
    'EE': 0.36,   # ~584 EUR/月
    'LT': 0.40,   # ~642 EUR/月
    'LV': 0.31,   # ~500 EUR/月 (最便宜)
    'PT': 0.48,   # ~776 EUR/月
    'SE': 1.36,   # ~2200 EUR/月 (最贵)
    'SI': 0.63,   # ~1024 EUR/月
    'SK': 0.38,   # ~623 EUR/月
}

# 每国测试集大小（固定）
TEST_SIZE_PER_COUNTRY = 10000


def get_transfer_ranking(source_country, feature_type, ascending=False):
    """从 Exp 2b 迁移矩阵获取 source 对其他国家的 transfer BA 排名。

    Args:
        source_country: 训练源国
        feature_type: 'ae' / 's2' / 'fusion'
        ascending: False=降序（最强在前），True=升序（最弱在前）

    Returns:
        list of (country, ba) 排序后的列表
    """
    csv_path = os.path.join(
        EXPERIMENTS_DIR, 'exp2b_country_transfer',
        f'transfer_{feature_type}_nocoords_2021_tier_b.csv')
    if not os.path.exists(csv_path):
        # 回退到 18 国结果
        full_path = os.path.join(
            EXPERIMENTS_DIR, 'exp2b_country_transfer',
            f'transfer_{feature_type}_nocoords_tier_b.csv')
        df = pd.read_csv(full_path)
    else:
        df = pd.read_csv(csv_path)

    sub = df[(df.train_country == source_country)
             & (df.test_country.isin(COUNTRIES_2021))]
    return [(row.test_country, row.balanced_accuracy)
            for _, row in sub.sort_values('balanced_accuracy',
                                           ascending=ascending).iterrows()]


def unlabeled_sample(df_src, n_samples, seed):
    """从 pool 完全随机抽样（不看标签，对应 Betti 的 active-labeling 假设）。

    在现实中 pool 代表一批未标注的 parcels，我们只能基于地理信息决策，
    不能用 class 来做分层。
    """
    if len(df_src) <= n_samples:
        return df_src
    return df_src.sample(n=n_samples, random_state=seed)


def split_target_test(df, source_country, tier='b', test_fraction=0.20):
    """为每个非 source 国按比例切出测试集 (proportional-to-LPIS-size)。

    每国抽 test_fraction 比例的样本作 test，剩余入 pool。
    这使得总 test 按国家样本量加权，对应 Betti 的 "proportional to population"。

    Returns:
        test_df, pool_df
    """
    from sklearn.model_selection import StratifiedShuffleSplit
    encoded_col = f'label_tier_{tier}_encoded'

    test_parts = []
    pool_parts = []

    for c in COUNTRIES_2021:
        if c == source_country:
            continue
        sub = df[df[COL_COUNTRY] == c].reset_index(drop=True)
        if len(sub) < 10:
            continue

        # 稀有类保护：只从 count>=2 的类中抽 test
        counts = Counter(sub[encoded_col].values)
        valid_idx = sub[encoded_col].isin(
            {cls for cls, cnt in counts.items() if cnt >= 2})
        main_df = sub[valid_idx].reset_index(drop=True)
        rare_df = sub[~valid_idx].reset_index(drop=True)

        try:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=test_fraction,
                                         random_state=RANDOM_STATE)
            pool_idx, test_idx = next(sss.split(main_df, main_df[encoded_col]))
            test_parts.append(main_df.iloc[test_idx].reset_index(drop=True))
            pool_parts.append(pd.concat([main_df.iloc[pool_idx], rare_df],
                                         ignore_index=True))
        except Exception:
            # Fallback: 随机 split
            rng = np.random.RandomState(RANDOM_STATE)
            n_test = max(1, int(len(main_df) * test_fraction))
            test_idx = rng.choice(len(main_df), n_test, replace=False)
            pool_idx = np.setdiff1d(np.arange(len(main_df)), test_idx)
            test_parts.append(main_df.iloc[test_idx].reset_index(drop=True))
            pool_parts.append(pd.concat([main_df.iloc[pool_idx], rare_df],
                                         ignore_index=True))

    test_df = pd.concat(test_parts, ignore_index=True)
    pool_df = pd.concat(pool_parts, ignore_index=True)
    return test_df, pool_df


def sample_strategy(pool_df, source_country, strategy, budget, seed,
                    tier='b', feature_type='ae'):
    """从 pool 中按策略采样 budget 个样本。

    Returns:
        aug_df: 增强 DataFrame
        info: 采样来源信息
    """
    if strategy == 'source_only':
        return pd.DataFrame(columns=pool_df.columns), {}

    other_countries = [c for c in COUNTRIES_2021 if c != source_country]

    if strategy == 'random':
        sampled = unlabeled_sample(pool_df, budget, seed)
        info = dict(sampled[COL_COUNTRY].value_counts())
        return sampled, info

    if strategy == 'climate_similar':
        # 从同气候带（除 source 外）的国家采
        source_zone = COUNTRY_ZONE[source_country]
        similar = [c for c in other_countries if COUNTRY_ZONE[c] == source_zone]
        if not similar:
            # fallback
            return sample_strategy(pool_df, source_country, 'random',
                                   budget, seed, tier, feature_type)
        sub_pool = pool_df[pool_df[COL_COUNTRY].isin(similar)]
        sampled = unlabeled_sample(sub_pool, budget, seed)
        info = dict(sampled[COL_COUNTRY].value_counts())
        return sampled, info

    if strategy == 'climate_diverse':
        # 从不同气候带的国家采
        source_zone = COUNTRY_ZONE[source_country]
        diverse = [c for c in other_countries if COUNTRY_ZONE[c] != source_zone]
        sub_pool = pool_df[pool_df[COL_COUNTRY].isin(diverse)]
        sampled = unlabeled_sample(sub_pool, budget, seed)
        info = dict(sampled[COL_COUNTRY].value_counts())
        return sampled, info

    if strategy == 'representativeness':
        # Betti-style: 按气候带分组，每组均衡采样
        zone_to_countries = {}
        for c in other_countries:
            zone = COUNTRY_ZONE[c]
            zone_to_countries.setdefault(zone, []).append(c)
        n_zones = len(zone_to_countries)
        per_zone_budget = budget // n_zones

        sampled_dfs = []
        for zone, countries in zone_to_countries.items():
            zone_pool = pool_df[pool_df[COL_COUNTRY].isin(countries)]
            sub = unlabeled_sample(zone_pool, per_zone_budget, seed)
            sampled_dfs.append(sub)
        result = pd.concat(sampled_dfs, ignore_index=True)
        info = dict(result[COL_COUNTRY].value_counts())
        return result, info

    if strategy == 'balanced':
        # 每国均匀
        per_country = budget // len(other_countries)
        sampled_dfs = []
        for c in other_countries:
            sub_pool = pool_df[pool_df[COL_COUNTRY] == c]
            sub = unlabeled_sample(sub_pool, per_country, seed)
            sampled_dfs.append(sub)
        result = pd.concat(sampled_dfs, ignore_index=True)
        info = dict(result[COL_COUNTRY].value_counts())
        return result, info

    if strategy == 'transfer_strong':
        # 按 Exp 2b 迁移矩阵：从迁移性最强的国家开始取，填满 budget
        ranking = get_transfer_ranking(source_country, feature_type,
                                        ascending=False)
        sampled_dfs = []
        total = 0
        for country, _ba in ranking:
            avail = pool_df[pool_df[COL_COUNTRY] == country]
            need = budget - total
            if need <= 0:
                break
            if len(avail) >= need:
                sub = unlabeled_sample(avail, need, seed)
                sampled_dfs.append(sub)
                total += need
                break
            else:
                sampled_dfs.append(avail)
                total += len(avail)
        result = pd.concat(sampled_dfs, ignore_index=True)
        info = dict(result[COL_COUNTRY].value_counts())
        return result, info

    if strategy == 'transfer_weak':
        # 按 Exp 2b 迁移矩阵：从迁移性最弱的国家开始取（补短板）
        ranking = get_transfer_ranking(source_country, feature_type,
                                        ascending=True)
        sampled_dfs = []
        total = 0
        for country, _ba in ranking:
            avail = pool_df[pool_df[COL_COUNTRY] == country]
            need = budget - total
            if need <= 0:
                break
            if len(avail) >= need:
                sub = unlabeled_sample(avail, need, seed)
                sampled_dfs.append(sub)
                total += need
                break
            else:
                sampled_dfs.append(avail)
                total += len(avail)
        result = pd.concat(sampled_dfs, ignore_index=True)
        info = dict(result[COL_COUNTRY].value_counts())
        return result, info

    if strategy == 'transfer_strong_weighted':
        # 加权版本：所有国家都贡献，权重 ∝ BA(S→T)
        ranking = get_transfer_ranking(source_country, feature_type,
                                        ascending=False)
        total_ba = sum(ba for _, ba in ranking)
        sampled_dfs = []
        for country, ba in ranking:
            avail = pool_df[pool_df[COL_COUNTRY] == country]
            # 分配该国 budget：按 BA 比例
            alloc = int(budget * ba / total_ba)
            if alloc <= 0 or len(avail) == 0:
                continue
            sub = unlabeled_sample(avail, min(alloc, len(avail)), seed)
            sampled_dfs.append(sub)
        if not sampled_dfs:
            return pd.DataFrame(columns=pool_df.columns), {}
        result = pd.concat(sampled_dfs, ignore_index=True)
        info = dict(result[COL_COUNTRY].value_counts())
        return result, info

    if strategy == 'transfer_weak_weighted':
        # 反向加权：所有国家贡献，权重 ∝ (1 - BA(S→T))，补短板
        ranking = get_transfer_ranking(source_country, feature_type,
                                        ascending=False)
        # 使用 (1 - BA) 作为权重
        weights = [(c, 1.0 - ba) for c, ba in ranking]
        total_w = sum(w for _, w in weights)
        sampled_dfs = []
        for country, w in weights:
            avail = pool_df[pool_df[COL_COUNTRY] == country]
            alloc = int(budget * w / total_w)
            if alloc <= 0 or len(avail) == 0:
                continue
            sub = unlabeled_sample(avail, min(alloc, len(avail)), seed)
            sampled_dfs.append(sub)
        if not sampled_dfs:
            return pd.DataFrame(columns=pool_df.columns), {}
        result = pd.concat(sampled_dfs, ignore_index=True)
        info = dict(result[COL_COUNTRY].value_counts())
        return result, info

    if strategy == 'cost_aware_strong':
        # Cost-aware: weight[T] ∝ BA(S→T) / cost(T)
        # "Bang-for-buck": 信息量高且成本低的国家优先
        ranking = get_transfer_ranking(source_country, feature_type,
                                        ascending=False)
        weights = [(c, ba / COUNTRY_COST.get(c, 1.0)) for c, ba in ranking]
        total_w = sum(w for _, w in weights)
        sampled_dfs = []
        for country, w in weights:
            avail = pool_df[pool_df[COL_COUNTRY] == country]
            alloc = int(budget * w / total_w)
            if alloc <= 0 or len(avail) == 0:
                continue
            sub = unlabeled_sample(avail, min(alloc, len(avail)), seed)
            sampled_dfs.append(sub)
        if not sampled_dfs:
            return pd.DataFrame(columns=pool_df.columns), {}
        result = pd.concat(sampled_dfs, ignore_index=True)
        info = dict(result[COL_COUNTRY].value_counts())
        return result, info

    if strategy == 'cost_aware_weak':
        # Cost-aware weak: weight[T] ∝ (1 - BA(S→T)) / cost(T)
        # 补短板 + 省钱
        ranking = get_transfer_ranking(source_country, feature_type,
                                        ascending=False)
        weights = [(c, (1.0 - ba) / COUNTRY_COST.get(c, 1.0))
                   for c, ba in ranking]
        total_w = sum(w for _, w in weights)
        sampled_dfs = []
        for country, w in weights:
            avail = pool_df[pool_df[COL_COUNTRY] == country]
            alloc = int(budget * w / total_w)
            if alloc <= 0 or len(avail) == 0:
                continue
            sub = unlabeled_sample(avail, min(alloc, len(avail)), seed)
            sampled_dfs.append(sub)
        if not sampled_dfs:
            return pd.DataFrame(columns=pool_df.columns), {}
        result = pd.concat(sampled_dfs, ignore_index=True)
        info = dict(result[COL_COUNTRY].value_counts())
        return result, info

    if strategy == 'cost_only':
        # 纯 cost-aware: weight[T] ∝ 1/cost(T) （ablation baseline）
        other_countries = [c for c in COUNTRIES_2021 if c != source_country]
        weights = [(c, 1.0 / COUNTRY_COST.get(c, 1.0)) for c in other_countries]
        total_w = sum(w for _, w in weights)
        sampled_dfs = []
        for country, w in weights:
            avail = pool_df[pool_df[COL_COUNTRY] == country]
            alloc = int(budget * w / total_w)
            if alloc <= 0 or len(avail) == 0:
                continue
            sub = unlabeled_sample(avail, min(alloc, len(avail)), seed)
            sampled_dfs.append(sub)
        if not sampled_dfs:
            return pd.DataFrame(columns=pool_df.columns), {}
        result = pd.concat(sampled_dfs, ignore_index=True)
        info = dict(result[COL_COUNTRY].value_counts())
        return result, info

    raise ValueError(f'Unknown strategy: {strategy}')


def evaluate_per_country(model, test_df, feature_cols, class_names, tier='b'):
    """在每个 target 国上分别评估。

    Returns:
        per_country: 逐国指标 dict
        mean_ba_equal: 等权平均（每国等权）
        mean_ba_weighted: 按 test 样本量加权平均（proportional-to-population）
    """
    encoded_col = f'label_tier_{tier}_encoded'
    per_country = {}

    for c in sorted(test_df[COL_COUNTRY].unique()):
        sub = test_df[test_df[COL_COUNTRY] == c]
        X_t = sub[feature_cols].fillna(0).values
        y_t = sub[encoded_col].values
        y_pred = model.predict(X_t)
        metrics = evaluate_predictions(y_t, y_pred, class_names)
        per_country[c] = {
            'balanced_accuracy': metrics['balanced_accuracy'],
            'macro_f1': metrics['macro_f1'],
            'n_test': len(sub),
        }

    # 等权均值
    mean_ba_equal = float(np.mean(
        [v['balanced_accuracy'] for v in per_country.values()]))
    # 按样本量加权均值（Betti-style proportional-to-population）
    total_n = sum(v['n_test'] for v in per_country.values())
    mean_ba_weighted = float(sum(
        v['balanced_accuracy'] * v['n_test'] for v in per_country.values()
    ) / total_n)

    return per_country, mean_ba_equal, mean_ba_weighted


def run_single(df, feature_cols, class_names, source_country,
               strategy, budget, seed, feature_type,
               tier='b', classifier_type='lgbm'):
    """单次实验：source + 策略增强训练，测 10 国 OOD mean BA。"""
    n_classes = len(class_names)
    encoded_col = f'label_tier_{tier}_encoded'

    # 切分：每国 10k test, 其余入 pool
    test_df, pool_df = split_target_test(df, source_country, tier=tier)

    # Source 训练数据（全部）
    source_df = df[df[COL_COUNTRY] == source_country].reset_index(drop=True)

    # 增强样本
    aug_df, info = sample_strategy(pool_df, source_country, strategy,
                                    budget, seed, tier=tier,
                                    feature_type=feature_type)

    # 合并
    if len(aug_df) > 0:
        train_df = pd.concat([source_df, aug_df], ignore_index=True)
    else:
        train_df = source_df.copy()

    X_train = train_df[feature_cols].fillna(0).values
    y_train = train_df[encoded_col].values

    # val: 从 train 中抽 10%
    from sklearn.model_selection import StratifiedShuffleSplit
    try:
        sss_v = StratifiedShuffleSplit(n_splits=1, test_size=0.1,
                                        random_state=seed)
        tr_idx, val_idx = next(sss_v.split(X_train, y_train))
        X_val, y_val = X_train[val_idx], y_train[val_idx]
        X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
    except Exception:
        rng_v = np.random.RandomState(seed)
        n_val = max(100, int(len(X_train) * 0.1))
        val_idx = rng_v.choice(len(X_train), n_val, replace=False)
        mask = np.ones(len(X_train), dtype=bool)
        mask[val_idx] = False
        X_val, y_val = X_train[val_idx], y_train[val_idx]
        X_tr, y_tr = X_train[mask], y_train[mask]

    # 训练
    trainer = get_trainer(classifier_type)
    model = trainer(X_tr, y_tr, X_val, y_val, n_classes)

    # 逐国评估
    per_country, mean_ba_equal, mean_ba_weighted = evaluate_per_country(
        model, test_df, feature_cols, class_names, tier=tier)

    return {
        'source_country': source_country,
        'strategy': strategy,
        'budget': budget,
        'seed': seed,
        'feature_type': feature_type,
        'n_source': len(source_df),
        'n_aug': len(aug_df),
        'n_train_total': len(train_df),
        'n_test_total': len(test_df),
        'mean_ood_ba_equal': mean_ba_equal,
        'mean_ood_ba_weighted': mean_ba_weighted,
        'per_country_ba': {c: v['balanced_accuracy'] for c, v in per_country.items()},
        'per_country_n': {c: v['n_test'] for c, v in per_country.items()},
        'sampling_info': info,
    }


def run_experiment(feature_type, source_country, strategies, budgets,
                   n_seeds, out_dir, tier='b', classifier_type='lgbm'):
    """对单个 feature_type 运行全部实验。"""
    print(f'\n{"=" * 70}')
    print(f'EXP 6: SCALABLE MAPPING — Source={source_country}, '
          f'Features={feature_type.upper()}')
    print(f'{"=" * 70}')

    # 加载数据
    df, feature_cols, class_names = load_eu_features(
        feature_type, tier, countries=COUNTRIES_2021)
    feature_cols = filter_coords(feature_cols)

    # 续跑支持：读取已有 CSV，跳过已完成 runs
    csv_path = os.path.join(out_dir,
                             f'scalable_{feature_type}_{source_country}_tier_{tier}.csv')
    all_rows = []
    completed = set()  # set of (strategy, budget, seed)
    if os.path.exists(csv_path):
        try:
            existing = pd.read_csv(csv_path)
            for _, r in existing.iterrows():
                completed.add((r['strategy'], int(r['budget']), int(r['seed'])))
                all_rows.append(r.to_dict())
            print(f'[Resume] Loaded {len(completed)} completed runs '
                  f'from {csv_path}')
        except Exception as e:
            print(f'[Resume] Failed to load existing CSV ({e}), start fresh')
            all_rows = []
            completed = set()

    # none 策略只跑一次（budget 无意义），其他策略每个 budget 都跑
    total_runs = n_seeds  # for source_only
    for s in strategies:
        if s != 'source_only':
            total_runs += len(budgets) * n_seeds
    count = 0

    for strategy in strategies:
        budgets_for_this = [0] if strategy == 'source_only' else budgets
        for budget in budgets_for_this:
            for seed in range(n_seeds):
                count += 1
                key = (strategy, budget, seed)
                if key in completed:
                    print(f'[{count}/{total_runs}] {strategy} | '
                          f'budget={budget} | seed={seed} — SKIP (done)')
                    continue
                print(f'\n[{count}/{total_runs}] {strategy} | '
                      f'budget={budget} | seed={seed}')

                try:
                    result = run_single(
                        df, feature_cols, class_names, source_country,
                        strategy, budget, seed, feature_type, tier,
                        classifier_type)

                    # 展平 per_country_ba 为列
                    row = {k: v for k, v in result.items()
                           if not isinstance(v, dict)}
                    for c, ba in result['per_country_ba'].items():
                        row[f'ba_{c}'] = ba
                    row['sampling_info'] = str(result['sampling_info'])

                    print(f'  equal-wt BA={result["mean_ood_ba_equal"]:.4f}, '
                          f'LPIS-wt BA={result["mean_ood_ba_weighted"]:.4f}, '
                          f'n_train={result["n_train_total"]}')
                    all_rows.append(row)

                    # 增量保存
                    pd.DataFrame(all_rows).to_csv(
                        os.path.join(out_dir,
                                     f'scalable_{feature_type}_{source_country}_tier_{tier}.csv'),
                        index=False)
                except Exception as e:
                    print(f'  ERROR: {e}')
                    import traceback
                    traceback.print_exc()
                    continue

    return all_rows


def main():
    parser = argparse.ArgumentParser(description='Exp 6: Scalable Mapping')
    parser.add_argument('--tier', choices=['a', 'b', 'c'], default='b')
    parser.add_argument('--source', default='DE',
                        help='Source country (default: DE)')
    parser.add_argument('--strategies', nargs='+',
                        default=['source_only', 'random',
                                 'climate_similar', 'climate_diverse',
                                 'representativeness', 'balanced',
                                 'transfer_strong', 'transfer_weak',
                                 'transfer_strong_weighted',
                                 'transfer_weak_weighted',
                                 'cost_aware_strong',
                                 'cost_aware_weak',
                                 'cost_only'])
    parser.add_argument('--budgets', nargs='+', type=int,
                        default=[1000, 5000, 10000, 20000, 50000])
    parser.add_argument('--n-seeds', type=int, default=3)
    parser.add_argument('--feature-types', nargs='+',
                        default=['ae', 's2', 'fusion'])
    parser.add_argument('--classifier', default='lgbm')
    args = parser.parse_args()

    out_dir = os.path.join(EXPERIMENTS_DIR, EXP_NAME)
    os.makedirs(out_dir, exist_ok=True)

    all_results = []
    for ft in args.feature_types:
        try:
            rows = run_experiment(
                ft, args.source, args.strategies, args.budgets,
                args.n_seeds, out_dir, args.tier, args.classifier)
            all_results.extend(rows)
        except Exception as e:
            print(f'Skipping {ft}: {e}')

    if all_results:
        combined_df = pd.DataFrame(all_results)
        combined_df.to_csv(
            os.path.join(out_dir,
                         f'scalable_all_{args.source}_tier_{args.tier}.csv'),
            index=False)
        print(f'\nTotal runs: {len(all_results)}')
        print(f'\nMean OOD BA by strategy & budget:')
        summary = combined_df.groupby(
            ['feature_type', 'strategy', 'budget']).agg(
            equal_mean=('mean_ood_ba_equal', 'mean'),
            weighted_mean=('mean_ood_ba_weighted', 'mean'),
        ).reset_index()
        for _, r in summary.iterrows():
            print(f'  {str(r["feature_type"]):7s} {str(r["strategy"]):22s} '
                  f'budget={int(r["budget"]):>6d}: '
                  f'equal={float(r["equal_mean"]):.4f}, '
                  f'weighted={float(r["weighted_mean"]):.4f}')

    print('\nExp 6 complete!')


if __name__ == '__main__':
    main()
