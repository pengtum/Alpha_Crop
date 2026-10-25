"""
Exp 6: Optimal Borrowing Strategies for Data-Scarce Target Countries

研究问题 (RQ4)：当 target 国标注稀缺时，从其他国家借用数据的最优策略是什么？
AE 和 S2 的最优策略是否不同？

6 种采样策略：
  S1: No augmentation (只用 target 自己)
  S2: Random (从其他 10 国随机)
  S3: Top-Hub (从 Exp 2b 里 mean OOD BA 最高的国家)
  S4: Climate-matched (只从同气候带国家)
  S5: Class-overlap-matched (从 Jaccard overlap 最高的国家)
  S6: Per-country-balanced (每国均匀贡献 N/10)

Usage:
    python -m src.experiments.run_borrowing_strategies --targets PT LT EE SI \\
        --budgets 5000 10000 20000 --feature-types ae s2 fusion
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
    RANDOM_STATE, EXPERIMENTS_DIR, COL_COUNTRY, COUNTRIES_2021,
)

EXP_NAME = 'exp6_borrowing_strategies'

# 气候带
COUNTRY_ZONE = {
    'AT': 'continental', 'BE': 'atlantic', 'DE': 'atlantic',
    'EE': 'nordic_baltic', 'LT': 'nordic_baltic', 'LV': 'nordic_baltic',
    'PT': 'mediterranean', 'RO': 'southeast', 'SE': 'nordic_baltic',
    'SI': 'mediterranean', 'SK': 'continental',
}


def get_hub_ranking(feature_type):
    """从 Exp 2b 结果获取国家作为训练源的 mean OOD BA 排名。"""
    csv_path = os.path.join(
        EXPERIMENTS_DIR, 'exp2b_country_transfer',
        f'transfer_{feature_type}_nocoords_2021_tier_b.csv')
    if not os.path.exists(csv_path):
        # fallback: 从 18 国结果筛选
        full_path = os.path.join(
            EXPERIMENTS_DIR, 'exp2b_country_transfer',
            f'transfer_{feature_type}_nocoords_tier_b.csv')
        full = pd.read_csv(full_path)
        sub = full[full.train_country.isin(COUNTRIES_2021)
                   & full.test_country.isin(COUNTRIES_2021)]
        mean_out = sub.groupby('train_country')['balanced_accuracy'].mean()
    else:
        df = pd.read_csv(csv_path)
        mean_out = df.groupby('train_country')['balanced_accuracy'].mean()
    return mean_out.sort_values(ascending=False)


def get_class_overlap(df, target, tier='b'):
    """返回 target 国与其他国家的 Jaccard class overlap。"""
    label_col = f'label_tier_{tier}'
    target_classes = set(df[df[COL_COUNTRY] == target][label_col].unique())
    overlaps = {}
    for c in COUNTRIES_2021:
        if c == target:
            continue
        other_classes = set(df[df[COL_COUNTRY] == c][label_col].unique())
        inter = len(target_classes & other_classes)
        union = len(target_classes | other_classes)
        overlaps[c] = inter / union if union > 0 else 0
    return pd.Series(overlaps).sort_values(ascending=False)


def stratified_sample(df_src, n_samples, seed, tier='b'):
    """分层抽样，保持类别比例。不足则取全部。"""
    if len(df_src) <= n_samples:
        return df_src

    label_col = f'label_tier_{tier}'
    rng = np.random.RandomState(seed)
    counts = df_src[label_col].value_counts()
    total = counts.sum()

    sampled_dfs = []
    for cls, cnt in counts.items():
        target_n = max(1, int(n_samples * cnt / total))
        cls_df = df_src[df_src[label_col] == cls]
        if len(cls_df) <= target_n:
            sampled_dfs.append(cls_df)
        else:
            sampled_dfs.append(cls_df.sample(n=target_n, random_state=seed))
    result = pd.concat(sampled_dfs, ignore_index=True)
    if len(result) > n_samples:
        result = result.sample(n=n_samples, random_state=seed)
    return result


def sample_strategy(df_all, target, strategy, budget, seed,
                    hub_ranking, overlaps, tier='b'):
    """根据策略从非 target 国中采样 budget 个样本。

    Returns:
        df_aug: 增强样本 DataFrame（不含 target 国）
        sampling_info: 采样来源国家和数量的 dict
    """
    other = df_all[df_all[COL_COUNTRY] != target]

    if strategy == 'none':
        return pd.DataFrame(columns=df_all.columns), {}

    if strategy == 'random':
        # 从其他 10 国总池随机抽 budget 个
        sampled = stratified_sample(other, budget, seed, tier=tier)
        info = dict(sampled[COL_COUNTRY].value_counts())
        return sampled, info

    if strategy == 'top_hub':
        # 按 hub_ranking 从最强训练源开始采，填满 budget
        sampled_dfs = []
        total = 0
        for country, _ba in hub_ranking.items():
            if country == target:
                continue
            avail = other[other[COL_COUNTRY] == country]
            need = budget - total
            if need <= 0:
                break
            if len(avail) >= need:
                sub = stratified_sample(avail, need, seed, tier=tier)
                sampled_dfs.append(sub)
                total += need
                break
            else:
                sampled_dfs.append(avail)
                total += len(avail)
        result = pd.concat(sampled_dfs, ignore_index=True)
        info = dict(result[COL_COUNTRY].value_counts())
        return result, info

    if strategy == 'climate_matched':
        # 从同气候带国家采样
        target_zone = COUNTRY_ZONE[target]
        same_zone = [c for c in COUNTRIES_2021
                     if COUNTRY_ZONE[c] == target_zone and c != target]
        if not same_zone:
            # fallback: 退回 random
            return sample_strategy(df_all, target, 'random', budget, seed,
                                   hub_ranking, overlaps, tier)
        pool = other[other[COL_COUNTRY].isin(same_zone)]
        sampled = stratified_sample(pool, budget, seed, tier=tier)
        info = dict(sampled[COL_COUNTRY].value_counts())
        return sampled, info

    if strategy == 'class_overlap':
        # 按 Jaccard overlap 从高到低采样
        sampled_dfs = []
        total = 0
        for country, _jac in overlaps.items():
            if country == target:
                continue
            avail = other[other[COL_COUNTRY] == country]
            need = budget - total
            if need <= 0:
                break
            if len(avail) >= need:
                sub = stratified_sample(avail, need, seed, tier=tier)
                sampled_dfs.append(sub)
                total += need
                break
            else:
                sampled_dfs.append(avail)
                total += len(avail)
        result = pd.concat(sampled_dfs, ignore_index=True)
        info = dict(result[COL_COUNTRY].value_counts())
        return result, info

    if strategy == 'balanced':
        # 每国均匀贡献 N/10
        other_countries = [c for c in COUNTRIES_2021 if c != target]
        per_country = budget // len(other_countries)
        sampled_dfs = []
        for country in other_countries:
            avail = other[other[COL_COUNTRY] == country]
            if len(avail) <= per_country:
                sampled_dfs.append(avail)
            else:
                sub = stratified_sample(avail, per_country, seed, tier=tier)
                sampled_dfs.append(sub)
        result = pd.concat(sampled_dfs, ignore_index=True)
        info = dict(result[COL_COUNTRY].value_counts())
        return result, info

    raise ValueError(f'Unknown strategy: {strategy}')


def run_single(df, feature_cols, class_names, target, strategy, budget,
               seed, feature_type, tier='b', classifier_type='lgbm',
               hub_ranking=None, overlaps=None):
    """单次实验：固定 target 测试集 + 策略采样增强训练集。"""
    n_classes = len(class_names)
    target_df = df[df[COL_COUNTRY] == target].reset_index(drop=True)

    # 固定分层 train/test split（80/20）
    rng = np.random.RandomState(RANDOM_STATE)  # 固定 split
    encoded_col = f'label_tier_{tier}_encoded'
    label_col = f'label_tier_{tier}'

    # 稀有类保护
    from collections import Counter as _Counter
    class_counts = _Counter(target_df[encoded_col].values)
    rare = {cls for cls, cnt in class_counts.items() if cnt < 5}

    from sklearn.model_selection import StratifiedShuffleSplit
    try:
        non_rare_mask = ~target_df[encoded_col].isin(rare)
        target_main = target_df[non_rare_mask].reset_index(drop=True)
        target_rare = target_df[~non_rare_mask].reset_index(drop=True)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2,
                                     random_state=RANDOM_STATE)
        train_idx, test_idx = next(sss.split(target_main, target_main[encoded_col]))
        target_train_own = pd.concat([target_main.iloc[train_idx], target_rare],
                                      ignore_index=True)
        target_test = target_main.iloc[test_idx].reset_index(drop=True)
    except Exception:
        rng2 = np.random.RandomState(RANDOM_STATE)
        n_test = int(len(target_df) * 0.2)
        test_idx = rng2.choice(len(target_df), n_test, replace=False)
        target_test = target_df.iloc[test_idx].reset_index(drop=True)
        target_train_own = target_df.drop(test_idx).reset_index(drop=True)

    # 增强样本（使用策略）
    aug_df, info = sample_strategy(df, target, strategy, budget, seed,
                                    hub_ranking, overlaps, tier)

    # 合并：target 自己的 train + 增强样本
    if len(aug_df) > 0:
        train_df = pd.concat([target_train_own, aug_df], ignore_index=True)
    else:
        train_df = target_train_own.copy()

    # 提取 X, y
    X_train = train_df[feature_cols].fillna(0).values
    y_train = train_df[encoded_col].values
    X_test = target_test[feature_cols].fillna(0).values
    y_test = target_test[encoded_col].values

    # val: 从 train 中分 10% stratified
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
    y_pred = model.predict(X_test)
    metrics = evaluate_predictions(y_test, y_pred, class_names)

    metrics['target_country'] = target
    metrics['strategy'] = strategy
    metrics['budget'] = budget
    metrics['seed'] = seed
    metrics['feature_type'] = feature_type
    metrics['n_train_own'] = len(target_train_own)
    metrics['n_train_aug'] = len(aug_df)
    metrics['n_train_total'] = len(train_df)
    metrics['n_test'] = len(target_test)
    metrics['sampling_info'] = str(info)

    return metrics


def run_experiment(feature_type, targets, strategies, budgets, n_seeds,
                   out_dir, tier='b', classifier_type='lgbm'):
    """对单个 feature_type 运行全部实验。"""
    print(f'\n{"=" * 70}')
    print(f'EXP 6: BORROWING STRATEGIES — {feature_type.upper()} Tier {tier.upper()}')
    print(f'{"=" * 70}')

    # 加载数据（11 countries, 2021 aligned）
    df, feature_cols, class_names = load_eu_features(
        feature_type, tier, countries=COUNTRIES_2021)
    feature_cols = filter_coords(feature_cols)

    # 预计算：hub ranking (基于 pre-existing Exp 2b 结果)
    hub_ranking = get_hub_ranking(feature_type)
    print(f'\nHub ranking ({feature_type}):')
    for c, ba in hub_ranking.items():
        print(f'  {c}: {ba:.4f}')

    all_rows = []
    total = len(targets) * len(strategies) * len(budgets) * n_seeds
    count = 0

    for target in targets:
        # 为每个 target 预计算 class overlap
        overlaps = get_class_overlap(df, target, tier=tier)
        print(f'\n--- Target: {target} ---')
        print(f'  Class overlap top 3: {overlaps.head(3).to_dict()}')

        for strategy in strategies:
            for budget in budgets:
                if strategy == 'none' and budget != budgets[0]:
                    # No-augmentation 只跑一次
                    continue

                for seed in range(n_seeds):
                    count += 1
                    print(f'\n[{count}/{total}] {target} | {strategy} | '
                          f'budget={budget} | seed={seed}')

                    try:
                        metrics = run_single(
                            df, feature_cols, class_names, target,
                            strategy, budget, seed, feature_type, tier,
                            classifier_type, hub_ranking, overlaps)
                        print(f'  BA={metrics["balanced_accuracy"]:.4f}, '
                              f'n_train={metrics["n_train_total"]}, '
                              f'n_test={metrics["n_test"]}')
                        all_rows.append(metrics)
                    except Exception as e:
                        print(f'  ERROR: {e}')
                        continue

                # 增量保存
                pd.DataFrame(all_rows).to_csv(
                    os.path.join(out_dir,
                                 f'borrowing_{feature_type}_tier_{tier}.csv'),
                    index=False)

    return all_rows


def main():
    parser = argparse.ArgumentParser(description='Exp 6: Borrowing Strategies')
    parser.add_argument('--tier', choices=['a', 'b', 'c'], default='b')
    parser.add_argument('--targets', nargs='+',
                        default=['PT', 'LT', 'EE', 'SI'])
    parser.add_argument('--strategies', nargs='+',
                        default=['none', 'random', 'top_hub',
                                 'climate_matched', 'class_overlap',
                                 'balanced'])
    parser.add_argument('--budgets', nargs='+', type=int,
                        default=[5000, 10000, 20000])
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
                ft, args.targets, args.strategies, args.budgets,
                args.n_seeds, out_dir, args.tier, args.classifier)
            all_results.extend(rows)
        except Exception as e:
            print(f'Skipping {ft}: {e}')

    # 最终汇总
    if all_results:
        combined_df = pd.DataFrame(all_results)
        combined_df.to_csv(
            os.path.join(out_dir, f'borrowing_all_tier_{args.tier}.csv'),
            index=False)
        print(f'\nTotal runs: {len(all_results)}')

    print('\nExp 6 complete!')


if __name__ == '__main__':
    main()
