"""
Exp 0: Per-Country 5-Fold Cross-Validation Baseline

对每个国家单独进行 stratified 5-fold CV，建立每国分类性能基线。
为后续 LOCO (Exp 2) 的 OOD 分析提供参照：区分「国家本身难度」vs「跨域迁移损失」。

输出：
  - 每国 × 每 feature_type 的 BA/F1/Kappa
  - summary CSV 用于对比可视化
  - per-sample predictions CSV（含国家、真实标签、预测标签）

Usage:
    python -m src.experiments.run_per_country_cv [--tier b] [--n-folds 5] [--classifier lgbm]
"""
import sys
import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from src.experiments.experiment_utils import (
    load_eu_features, prepare_xy, get_trainer, build_suffix,
    subsample_train, evaluate_predictions,
    save_experiment_results, save_results_csv, print_metrics,
    filter_coords, RANDOM_STATE, COL_COUNTRY,
    VALID_CLASSIFIERS, CLASSIFIER_TABPFN, TABPFN_MAX_TRAIN_SAMPLES, COUNTRIES_2021,
)


def run_country_cv(X, y, class_names, n_folds, n_classes,
                   classifier_type='lgbm', max_samples=0):
    """对单个国家数据运行 k-fold CV，返回各 fold 指标均值及逐样本预测。"""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True,
                          random_state=RANDOM_STATE)

    fold_metrics = []
    all_y_test = []
    all_y_pred = []
    all_test_idx = []

    for fold_i, (train_val_idx, test_idx) in enumerate(skf.split(X, y)):
        # 从 train_val 中再分 val（取 15%）
        rng = np.random.RandomState(RANDOM_STATE + fold_i)
        n_val = max(1, int(len(train_val_idx) * 0.15))
        val_pos = rng.choice(len(train_val_idx), size=n_val, replace=False)
        val_idx = train_val_idx[val_pos]
        train_idx = np.delete(train_val_idx, val_pos)

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        # 统一子采样
        if max_samples:
            X_train, y_train = subsample_train(
                X_train, y_train, max_samples,
                seed=RANDOM_STATE + fold_i)
            X_val, y_val = subsample_train(
                X_val, y_val, max(500, max_samples // 5),
                seed=RANDOM_STATE + fold_i + 100)

        trainer = get_trainer(classifier_type)
        model = trainer(X_train, y_train, X_val, y_val, n_classes)
        y_pred = model.predict(X_test)
        metrics = evaluate_predictions(y_test, y_pred, class_names)
        fold_metrics.append(metrics)

        # 收集逐样本预测
        all_y_test.append(y_test)
        all_y_pred.append(y_pred)
        all_test_idx.append(test_idx)

    # 汇总各 fold
    summary = {}
    for key in ['balanced_accuracy', 'overall_accuracy', 'cohen_kappa',
                'macro_f1', 'weighted_f1']:
        values = [m[key] for m in fold_metrics]
        summary[key] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
        }

    # 合并所有 fold 的逐样本预测
    y_test_all = np.concatenate(all_y_test)
    y_pred_all = np.concatenate(all_y_pred)
    test_idx_all = np.concatenate(all_test_idx)

    return summary, y_test_all, y_pred_all, test_idx_all


def run_experiment(feature_type, tier, n_folds, no_coords=False, countries=None,
                   classifier_type='lgbm', max_samples=0):
    """对所有国家运行 per-country CV。"""
    tag = " [no-coords]" if no_coords else ""
    cls_tag = f" [{classifier_type}]" if classifier_type != 'lgbm' else ""
    print(f"\n{'=' * 60}")
    print(f"EXP 0: PER-COUNTRY CV — {feature_type.upper()} "
          f"Tier {tier.upper()}{tag}{cls_tag}")
    print(f"{'=' * 60}")

    df, feature_cols, class_names = load_eu_features(feature_type, tier, countries=countries)
    if no_coords:
        feature_cols = filter_coords(feature_cols)
    countries = sorted(df[COL_COUNTRY].unique())
    n_global_classes = len(class_names)

    country_results = {}
    csv_rows = []
    pred_dfs = []  # 逐样本预测 DataFrame 列表

    for country in countries:
        df_c = df[df[COL_COUNTRY] == country].reset_index(drop=True)
        n_samples = len(df_c)

        # 该国实际出现的类别
        label_col = f'label_tier_{tier}'
        unique_labels = df_c[label_col].unique()
        n_local_classes = len(unique_labels)

        # 检查是否有足够样本做 k-fold
        from collections import Counter
        label_counts = Counter(df_c[f'label_tier_{tier}_encoded'].values)
        min_count = min(label_counts.values())

        if min_count < n_folds:
            # 类别样本太少，过滤掉少于 n_folds 个样本的类别
            valid_classes = {cls for cls, cnt in label_counts.items()
                            if cnt >= n_folds}
            encoded_col = f'label_tier_{tier}_encoded'
            df_c = df_c[df_c[encoded_col].isin(valid_classes)].reset_index(
                drop=True)
            n_filtered = n_samples - len(df_c)
            if n_filtered > 0:
                print(f"  {country}: filtered {n_filtered} samples from "
                      f"rare classes (<{n_folds} samples)")
            n_samples = len(df_c)
            unique_labels = df_c[label_col].unique()
            n_local_classes = len(unique_labels)

        if n_samples < n_folds * 2:
            print(f"  {country}: skipped (only {n_samples} samples)")
            continue

        X_c, y_c = prepare_xy(df_c, feature_cols, tier)

        print(f"  {country}: {n_samples:,} samples, "
              f"{n_local_classes} classes ... ", end='', flush=True)

        try:
            summary, y_test_all, y_pred_all, test_idx_all = run_country_cv(
                X_c, y_c, unique_labels, n_folds, n_global_classes,
                classifier_type=classifier_type, max_samples=max_samples)
            ba = summary['balanced_accuracy']['mean']
            f1 = summary['macro_f1']['mean']
            kappa = summary['cohen_kappa']['mean']
            print(f"BA={ba:.4f}, F1={f1:.4f}, Kappa={kappa:.4f}")

            country_results[country] = {
                'n_samples': n_samples,
                'n_classes': n_local_classes,
                **summary,
            }
            csv_rows.append({
                'country': country,
                'feature_type': feature_type,
                'n_samples': n_samples,
                'n_classes': n_local_classes,
                'BA_mean': ba,
                'BA_std': summary['balanced_accuracy']['std'],
                'F1_mean': f1,
                'F1_std': summary['macro_f1']['std'],
                'WF1_mean': summary['weighted_f1']['mean'],
                'WF1_std': summary['weighted_f1']['std'],
                'Kappa_mean': kappa,
                'Kappa_std': summary['cohen_kappa']['std'],
            })

            # 构建逐样本预测 DataFrame
            pred_df = pd.DataFrame({
                'country': country,
                'feature_type': feature_type,
                'sample_idx': test_idx_all,
                'y_true': y_test_all,
                'y_pred': y_pred_all,
            })
            pred_dfs.append(pred_df)

        except Exception as e:
            print(f"ERROR: {e}")
            continue

    return country_results, csv_rows, pred_dfs


def main():
    parser = argparse.ArgumentParser(
        description='Exp 0: Per-Country 5-Fold CV Baseline')
    parser.add_argument('--tier', choices=['a', 'b', 'c'], default='b')
    parser.add_argument('--n-folds', type=int, default=5)
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
    all_csv_rows = []
    all_pred_dfs = []

    for ft in args.feature_types:
        try:
            results, csv_rows, pred_dfs = run_experiment(
                ft, args.tier, args.n_folds,
                no_coords=args.no_coords, countries=countries,
                classifier_type=args.classifier,
                max_samples=args.max_samples)
            all_results[ft] = results
            all_csv_rows.extend(csv_rows)
            all_pred_dfs.extend(pred_dfs)
        except FileNotFoundError as e:
            print(f"Skipping {ft}: {e}")

    exp_name = 'exp0_per_country_cv'
    save_experiment_results(exp_name, all_results,
                            sub_name=f'tier_{args.tier}{suffix}')
    if all_csv_rows:
        save_results_csv(exp_name, all_csv_rows,
                         filename=f'summary_tier_{args.tier}{suffix}.csv')

    # 保存逐样本预测
    if all_pred_dfs:
        from config.config import EXPERIMENTS_DIR
        pred_all = pd.concat(all_pred_dfs, ignore_index=True)
        pred_path = os.path.join(
            EXPERIMENTS_DIR, exp_name,
            f'predictions_tier_{args.tier}{suffix}.csv')
        os.makedirs(os.path.dirname(pred_path), exist_ok=True)
        pred_all.to_csv(pred_path, index=False)
        print(f"Per-sample predictions saved: {pred_path} "
              f"({len(pred_all):,} rows)")

    # 打印汇总排行
    if all_csv_rows:
        print(f"\n{'=' * 60}")
        print("SUMMARY: Per-Country BA (Tier B)")
        print(f"{'=' * 60}")
        summary_df = pd.DataFrame(all_csv_rows)
        pivot = summary_df.pivot(index='country', columns='feature_type',
                                 values='BA_mean')
        pivot = pivot.sort_values(
            by=pivot.columns[0] if len(pivot.columns) > 0 else 'country',
            ascending=False)
        print(pivot.to_string(float_format='%.4f'))

    print(f"\nExp 0 complete!")


if __name__ == '__main__':
    main()
