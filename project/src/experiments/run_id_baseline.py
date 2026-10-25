"""
Exp 1: In-Distribution Baseline

所有国家混合，随机 stratified split，比较 AE vs S2 vs Fusion。
5 次随机划分取均值±std。

Usage:
    python -m src.experiments.run_id_baseline [--tier b] [--n-repeats 5]
    python -m src.experiments.run_id_baseline --classifier xgb --no-coords
    python -m src.experiments.run_id_baseline --classifier tabpfn --no-coords --max-samples 10000
"""
import sys
import os
import argparse
import numpy as np
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from src.experiments.experiment_utils import (
    load_eu_features, prepare_xy, evaluate_predictions,
    save_experiment_results, save_results_csv, print_metrics,
    filter_coords, subsample_train, get_trainer, build_suffix,
    RANDOM_STATE, VALID_CLASSIFIERS, CLASSIFIER_TABPFN,
    TABPFN_MAX_TRAIN_SAMPLES, COUNTRIES_2021,
)


def run_single(X, y, class_names, fold_idx, n_classes,
               classifier_type='lgbm', max_samples=0):
    """单次训练+评估。"""
    skf = StratifiedKFold(n_splits=5, shuffle=True,
                          random_state=RANDOM_STATE + fold_idx)

    # 用第一个 fold 的 test 作为 test，第二个作为 val，其余作为 train
    splits = list(skf.split(X, y))
    test_idx = splits[0][1]
    val_idx = splits[1][1]
    train_idx = np.concatenate([splits[i][1] for i in range(2, 5)] +
                                [splits[0][0]])
    # 去掉 test 和 val 的索引
    train_idx = np.setdiff1d(train_idx, np.concatenate([test_idx, val_idx]))

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # 统一子采样（保证跨分类器一致性）
    if max_samples:
        X_train, y_train = subsample_train(X_train, y_train, max_samples,
                                           seed=RANDOM_STATE + fold_idx)
        X_val, y_val = subsample_train(X_val, y_val,
                                       max(1000, max_samples // 5),
                                       seed=RANDOM_STATE + fold_idx + 100)

    trainer = get_trainer(classifier_type)
    model = trainer(X_train, y_train, X_val, y_val, n_classes)
    y_pred = model.predict(X_test)
    metrics = evaluate_predictions(y_test, y_pred, class_names)

    return metrics


def run_experiment(feature_type, tier, n_repeats,
                   no_coords=False, classifier_type='lgbm', max_samples=0,
                   countries=None):
    """对单个 feature_type 运行 n_repeats 次。"""
    tags = []
    if no_coords:
        tags.append("no-coords")
    if max_samples:
        tags.append(f"sub{max_samples//1000}k")
    tags.append(classifier_type.upper())
    tag_str = f" [{', '.join(tags)}]" if tags else ""

    print(f"\n{'=' * 60}")
    print(f"EXP 1: ID BASELINE — {feature_type.upper()} Tier {tier.upper()}{tag_str}")
    print(f"{'=' * 60}")

    df, feature_cols, class_names = load_eu_features(feature_type, tier,
                                                      countries=countries)
    if no_coords:
        feature_cols = filter_coords(feature_cols)
    X, y = prepare_xy(df, feature_cols, tier)
    n_classes = len(class_names)

    all_metrics = []
    for i in range(n_repeats):
        print(f"\n--- Repeat {i + 1}/{n_repeats} ---")
        metrics = run_single(X, y, class_names, fold_idx=i,
                             n_classes=n_classes,
                             classifier_type=classifier_type,
                             max_samples=max_samples)
        print_metrics(metrics, label=f"Repeat {i+1}")
        all_metrics.append(metrics)

    # 汇总统计
    summary = {}
    for key in ['balanced_accuracy', 'overall_accuracy', 'cohen_kappa',
                'macro_f1', 'weighted_f1']:
        values = [m[key] for m in all_metrics]
        summary[key] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'values': values,
        }

    summary['feature_type'] = feature_type
    summary['tier'] = tier
    summary['n_repeats'] = n_repeats
    summary['n_samples'] = len(X)
    summary['n_features'] = len(feature_cols)
    summary['n_classes'] = n_classes
    summary['classifier'] = classifier_type
    summary['max_samples'] = max_samples

    print(f"\n--- Summary ({feature_type.upper()}) ---")
    for key in ['balanced_accuracy', 'macro_f1', 'cohen_kappa']:
        s = summary[key]
        print(f"  {key}: {s['mean']:.4f} ± {s['std']:.4f}")

    return summary


def main():
    parser = argparse.ArgumentParser(description='Exp 1: ID Baseline')
    parser.add_argument('--tier', choices=['a', 'b', 'c'], default='b')
    parser.add_argument('--n-repeats', type=int, default=5)
    parser.add_argument('--feature-types', nargs='+',
                        default=['ae', 's2', 'fusion'],
                        choices=['ae', 's2', 'fusion'])
    parser.add_argument('--no-coords', action='store_true',
                        help='移除地理坐标和面积辅助特征')
    parser.add_argument('--classifier', choices=VALID_CLASSIFIERS,
                        default='lgbm', help='分类器类型')
    parser.add_argument('--max-samples', type=int, default=0,
                        help='训练集最大样本数（0=不限制）')
    parser.add_argument('--year-aligned', action='store_true',
                        help='仅使用 2021 年数据对齐的 11 个国家')
    args = parser.parse_args()

    # TabPFN 强制子采样
    if args.classifier == CLASSIFIER_TABPFN and not args.max_samples:
        args.max_samples = TABPFN_MAX_TRAIN_SAMPLES
        print(f"[TabPFN] 自动设置 max_samples={args.max_samples}")

    countries = COUNTRIES_2021 if args.year_aligned else None
    suffix = build_suffix(args.no_coords, args.max_samples, args.classifier,
                          year_aligned=args.year_aligned)

    all_results = {}
    csv_rows = []

    for ft in args.feature_types:
        try:
            summary = run_experiment(ft, args.tier, args.n_repeats,
                                     no_coords=args.no_coords,
                                     classifier_type=args.classifier,
                                     max_samples=args.max_samples,
                                     countries=countries)
            all_results[ft] = summary
            csv_rows.append({
                'feature_type': ft,
                'tier': args.tier,
                'classifier': args.classifier,
                'BA_mean': summary['balanced_accuracy']['mean'],
                'BA_std': summary['balanced_accuracy']['std'],
                'F1_mean': summary['macro_f1']['mean'],
                'F1_std': summary['macro_f1']['std'],
                'Kappa_mean': summary['cohen_kappa']['mean'],
                'Kappa_std': summary['cohen_kappa']['std'],
                'n_samples': summary['n_samples'],
                'n_features': summary['n_features'],
            })
        except FileNotFoundError as e:
            print(f"Skipping {ft}: {e}")

    exp_name = 'exp1_id_baseline'
    save_experiment_results(exp_name, all_results,
                            sub_name=f'tier_{args.tier}{suffix}')
    if csv_rows:
        save_results_csv(exp_name, csv_rows,
                         filename=f'summary_tier_{args.tier}{suffix}.csv')

    print(f"\nExp 1 complete!")


if __name__ == '__main__':
    main()
