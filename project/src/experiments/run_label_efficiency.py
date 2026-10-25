"""
Exp 4: Label Efficiency (Few-shot Learning Curves)

固定 OOD 测试集，递增训练样本量，比较 AE vs S2 的学习曲线。
额外：mixed 场景（大量源域 + 少量目标域标注）。

Usage:
    python -m src.experiments.run_label_efficiency [--tier b] [--classifier lgbm]
"""
import sys
import os
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from src.experiments.experiment_utils import (
    load_eu_features, prepare_xy, split_by_countries,
    get_trainer, build_suffix, subsample_train,
    evaluate_predictions,
    save_experiment_results, save_results_csv, print_metrics,
    filter_coords, get_all_country_isos, RANDOM_STATE,
    VALID_CLASSIFIERS, CLASSIFIER_TABPFN, TABPFN_MAX_TRAIN_SAMPLES, COUNTRIES_2021,
)
from config.config import COL_COUNTRY
from config.spatial_splits import EU_TEST_COUNTRIES, EU_VAL_COUNTRIES


SAMPLE_SIZES = [10, 50, 100, 500, 1000, 5000, 0]  # 0 = all


def subsample_per_class(X, y, n_per_class, seed=42):
    """Stratified subsampling: cap each class at n_per_class."""
    if n_per_class <= 0:
        return X, y

    indices = []
    rng = np.random.RandomState(seed)
    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        n = min(n_per_class, len(cls_idx))
        sampled = rng.choice(cls_idx, size=n, replace=False)
        indices.extend(sampled)

    indices = sorted(indices)
    return X[indices], y[indices]


def run_learning_curve(feature_type, tier, n_repeats=3, no_coords=False, countries=None,
                       classifier_type='lgbm', max_samples=0):
    """运行学习曲线实验。"""
    tag = f" [{classifier_type}]" + (" [no-coords]" if no_coords else "")
    print(f"\n{'=' * 60}")
    print(f"EXP 4: LABEL EFFICIENCY — {feature_type.upper()} Tier {tier.upper()}{tag}")
    print(f"{'=' * 60}")

    df, feature_cols, class_names = load_eu_features(feature_type, tier, countries=countries)
    if no_coords:
        feature_cols = filter_coords(feature_cols)
    all_countries = get_all_country_isos(df)
    n_classes = len(class_names)

    # OOD 划分：test = EU_TEST_COUNTRIES, val = EU_VAL_COUNTRIES
    test_countries = [c for c in EU_TEST_COUNTRIES if c in all_countries]
    val_countries = [c for c in EU_VAL_COUNTRIES if c in all_countries]
    train_countries = [c for c in all_countries
                       if c not in test_countries and c not in val_countries]

    X_train_full, y_train_full, X_val, y_val, X_test, y_test = \
        split_by_countries(df, feature_cols, tier,
                           train_countries=train_countries,
                           test_countries=test_countries,
                           val_countries=val_countries)

    print(f"Train: {len(X_train_full)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print(f"Test countries: {test_countries}")

    trainer = get_trainer(classifier_type)

    csv_rows = []
    results = {'sample_sizes': {}}

    for n_per_class in SAMPLE_SIZES:
        label = f"{n_per_class}/class" if n_per_class > 0 else "all"
        print(f"\n--- {label} ---")

        repeat_metrics = []
        for rep in range(n_repeats):
            # 先按类别子采样
            X_sub, y_sub = subsample_per_class(
                X_train_full, y_train_full, n_per_class,
                seed=RANDOM_STATE + rep)

            # 再按 max_samples 上限截断（例如 TabPFN 限制）
            if max_samples:
                X_sub, y_sub = subsample_train(X_sub, y_sub, max_samples,
                                               seed=RANDOM_STATE + rep)

            if len(X_sub) < n_classes:
                print(f"  Too few samples ({len(X_sub)}), skipping")
                break

            model = trainer(X_sub, y_sub, X_val, y_val, n_classes)
            y_pred = model.predict(X_test)
            metrics = evaluate_predictions(y_test, y_pred)
            repeat_metrics.append(metrics)

        if repeat_metrics:
            ba_vals = [m['balanced_accuracy'] for m in repeat_metrics]
            f1_vals = [m['macro_f1'] for m in repeat_metrics]

            result = {
                'ba_mean': float(np.mean(ba_vals)),
                'ba_std': float(np.std(ba_vals)),
                'f1_mean': float(np.mean(f1_vals)),
                'f1_std': float(np.std(f1_vals)),
                'n_train_actual': len(X_sub),
            }
            results['sample_sizes'][str(n_per_class)] = result

            print(f"  BA={result['ba_mean']:.4f}±{result['ba_std']:.4f}, "
                  f"F1={result['f1_mean']:.4f}±{result['f1_std']:.4f}, "
                  f"n_train={len(X_sub)}")

            csv_rows.append({
                'feature_type': feature_type,
                'classifier': classifier_type,
                'n_per_class': n_per_class if n_per_class > 0 else 'all',
                'n_train': len(X_sub),
                'BA_mean': result['ba_mean'],
                'BA_std': result['ba_std'],
                'F1_mean': result['f1_mean'],
                'F1_std': result['f1_std'],
            })

    results['feature_type'] = feature_type
    results['tier'] = tier
    results['test_countries'] = test_countries

    return results, csv_rows


def run_mixed_scenario(feature_type, tier, countries=None, target_country='ES',
                       n_target_samples=[0, 10, 50, 100, 500],
                       no_coords=False, classifier_type='lgbm',
                       max_samples=0):
    """混合场景：大量源域 + 少量目标域标注。"""
    tag = f" [{classifier_type}]" + (" [no-coords]" if no_coords else "")
    print(f"\n{'=' * 60}")
    print(f"EXP 4b: MIXED SCENARIO — {feature_type.upper()}, target={target_country}{tag}")
    print(f"{'=' * 60}")

    df, feature_cols, class_names = load_eu_features(feature_type, tier, countries=countries)
    if no_coords:
        feature_cols = filter_coords(feature_cols)
    all_countries = get_all_country_isos(df)
    n_classes = len(class_names)

    if target_country not in all_countries:
        print(f"Target country {target_country} not found, skipping")
        return None, []

    source_countries = [c for c in all_countries if c != target_country]

    # 源域数据
    source_mask = df[COL_COUNTRY].isin(source_countries)
    target_mask = df[COL_COUNTRY] == target_country

    X_source, y_source = prepare_xy(df[source_mask], feature_cols, tier)
    X_target, y_target = prepare_xy(df[target_mask], feature_cols, tier)

    # val 从源域取 10%
    rng = np.random.RandomState(RANDOM_STATE)
    n_val = max(1, int(len(X_source) * 0.1))
    val_idx = rng.choice(len(X_source), size=n_val, replace=False)
    train_idx = np.setdiff1d(np.arange(len(X_source)), val_idx)

    X_val, y_val = X_source[val_idx], y_source[val_idx]
    X_source_train, y_source_train = X_source[train_idx], y_source[train_idx]

    # test = 目标域后半
    n_target_test = len(X_target) // 2
    X_test = X_target[n_target_test:]
    y_test = y_target[n_target_test:]
    X_target_pool = X_target[:n_target_test]
    y_target_pool = y_target[:n_target_test]

    trainer = get_trainer(classifier_type)

    csv_rows = []
    results = {}

    for n_target in n_target_samples:
        if n_target == 0:
            X_train = X_source_train
            y_train = y_source_train
        else:
            n_use = min(n_target, len(X_target_pool))
            idx = rng.choice(len(X_target_pool), size=n_use, replace=False)
            X_train = np.vstack([X_source_train, X_target_pool[idx]])
            y_train = np.concatenate([y_source_train, y_target_pool[idx]])

        # 按 max_samples 上限截断
        if max_samples:
            X_train, y_train = subsample_train(X_train, y_train, max_samples,
                                               seed=RANDOM_STATE)

        model = trainer(X_train, y_train, X_val, y_val, n_classes)
        y_pred = model.predict(X_test)
        metrics = evaluate_predictions(y_test, y_pred)

        label = f"source+{n_target}target"
        print_metrics(metrics, label=label)

        results[str(n_target)] = metrics
        csv_rows.append({
            'feature_type': feature_type,
            'classifier': classifier_type,
            'target_country': target_country,
            'n_target_samples': n_target,
            'n_train_total': len(X_train),
            'BA': metrics['balanced_accuracy'],
            'F1': metrics['macro_f1'],
        })

    return results, csv_rows


def main():
    parser = argparse.ArgumentParser(description='Exp 4: Label Efficiency')
    parser.add_argument('--tier', choices=['a', 'b', 'c'], default='b')
    parser.add_argument('--feature-types', nargs='+', default=['ae', 's2'],
                        choices=['ae', 's2', 'fusion'])
    parser.add_argument('--n-repeats', type=int, default=3)
    parser.add_argument('--mixed', action='store_true',
                        help='Also run mixed source+target scenario')
    parser.add_argument('--no-coords', action='store_true',
                        help='移除地理坐标和面积辅助特征')
    parser.add_argument('--classifier', choices=VALID_CLASSIFIERS, default='lgbm',
                        help='分类器类型 (default: lgbm)')
    parser.add_argument('--max-samples', type=int, default=0,
                        help='训练集最大样本数 (0=不限制)')
    parser.add_argument('--year-aligned', action='store_true',
                        help='仅使用 2021 年数据对齐的 11 个国家')
    args = parser.parse_args()

    # TabPFN 自动设置 max_samples
    max_samples = args.max_samples
    if args.classifier == CLASSIFIER_TABPFN and not max_samples:
        max_samples = TABPFN_MAX_TRAIN_SAMPLES

    countries = COUNTRIES_2021 if args.year_aligned else None
    suffix = build_suffix(args.no_coords, max_samples, args.classifier,
                          year_aligned=args.year_aligned)

    all_results = {}
    all_csv_rows = []

    for ft in args.feature_types:
        try:
            results, csv_rows = run_learning_curve(
                ft, args.tier, args.n_repeats,
                no_coords=args.no_coords, countries=countries,
                classifier_type=args.classifier,
                max_samples=max_samples)
            all_results[f'{ft}_curve'] = results
            all_csv_rows.extend(csv_rows)

            if args.mixed:
                mixed_results, mixed_rows = run_mixed_scenario(
                    ft, args.tier, countries=countries,
                    no_coords=args.no_coords,
                    classifier_type=args.classifier,
                    max_samples=max_samples)
                if mixed_results:
                    all_results[f'{ft}_mixed'] = mixed_results
                    all_csv_rows.extend(mixed_rows)
        except FileNotFoundError as e:
            print(f"Skipping {ft}: {e}")

    exp_name = 'exp4_label_efficiency'
    save_experiment_results(exp_name, all_results,
                            sub_name=f'tier_{args.tier}{suffix}')
    if all_csv_rows:
        save_results_csv(exp_name, all_csv_rows,
                         filename=f'label_efficiency_tier_{args.tier}{suffix}.csv')

    print(f"\nExp 4 complete!")


if __name__ == '__main__':
    main()
