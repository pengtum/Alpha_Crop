"""
Exp 2: Leave-One-Country-Out (LOCO)

依次将每个国家/地区作为测试集，其余作训练+验证。
比较 AE vs S2 vs Fusion 在每个 OOD 国家上的表现。

每完成一个国家就保存：
  1. 累积的精度指标 (JSON + CSV)
  2. 每个测试样本的真值、预测值 (per-sample CSV)

Usage:
    python -m src.experiments.run_loco [--tier b] [--feature-types ae s2] \
        [--classifier lgbm] [--max-samples 0]
"""
import sys
import os
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from src.experiments.experiment_utils import (
    load_eu_features, split_by_countries, prepare_xy,
    get_trainer, build_suffix, subsample_train,
    evaluate_predictions,
    save_experiment_results, save_results_csv, print_metrics,
    filter_coords, get_all_country_isos, RANDOM_STATE, EXPERIMENTS_DIR,
    VALID_CLASSIFIERS, CLASSIFIER_TABPFN, TABPFN_MAX_TRAIN_SAMPLES, COUNTRIES_2021,
)
from config.config import COL_PARCEL_ID, COL_COUNTRY


def run_loco_single(df, feature_cols, class_names, tier,
                    test_country, all_countries,
                    classifier_type='lgbm', max_samples=0):
    """对单个留出国家执行 LOCO，返回指标 + per-sample 预测。"""
    train_countries = [c for c in all_countries if c != test_country]

    # 获取测试集 DataFrame 以保留 parcel_id 等元信息
    test_mask = df[COL_COUNTRY] == test_country
    df_test = df[test_mask]

    X_train, y_train, X_val, y_val, X_test, y_test = split_by_countries(
        df, feature_cols, tier,
        train_countries=train_countries,
        test_countries=[test_country])

    if len(X_test) < 10:
        print(f"  {test_country}: too few test samples ({len(X_test)}), skipping")
        return None, None

    # 训练集子采样（保证跨分类器一致性）
    if max_samples:
        X_train, y_train = subsample_train(X_train, y_train, max_samples)

    n_classes = len(class_names)
    trainer = get_trainer(classifier_type)
    model = trainer(X_train, y_train, X_val, y_val, n_classes)
    y_pred = model.predict(X_test)

    metrics = evaluate_predictions(y_test, y_pred, class_names)
    metrics['test_country'] = test_country
    metrics['n_train'] = len(X_train)
    metrics['n_test'] = len(X_test)

    # 构建 per-sample 预测 DataFrame
    encoded_col = f'label_tier_{tier}_encoded'
    label_col = f'label_tier_{tier}'

    # 真值类名和预测类名
    y_true_names = [class_names[i] if i < len(class_names) else str(i)
                    for i in y_test]
    y_pred_names = [class_names[i] if i < len(class_names) else str(i)
                    for i in y_pred]

    per_sample_df = pd.DataFrame({
        'parcel_id': df_test[COL_PARCEL_ID].values if COL_PARCEL_ID in df_test.columns else range(len(y_test)),
        'country': test_country,
        'y_true_encoded': y_test,
        'y_pred_encoded': y_pred,
        'y_true_label': y_true_names,
        'y_pred_label': y_pred_names,
        'correct': (y_test == y_pred).astype(int),
    })

    return metrics, per_sample_df


def run_experiment(feature_type, tier, out_dir, no_coords=False,
                   countries=None, classifier_type='lgbm', max_samples=0):
    """对单个 feature_type 运行完整 LOCO，每个国家完成后即保存。"""
    suffix = build_suffix(no_coords, max_samples, classifier_type)
    tag_parts = []
    if no_coords:
        tag_parts.append("no-coords")
    if classifier_type != 'lgbm':
        tag_parts.append(classifier_type.upper())
    if max_samples:
        tag_parts.append(f"max={max_samples}")
    tag = (" [" + ", ".join(tag_parts) + "]") if tag_parts else ""

    print(f"\n{'=' * 60}")
    print(f"EXP 2: LOCO — {feature_type.upper()} Tier {tier.upper()}{tag}")
    print(f"{'=' * 60}")

    df, feature_cols, class_names = load_eu_features(feature_type, tier, countries=countries)
    if no_coords:
        feature_cols = filter_coords(feature_cols)
    all_countries = get_all_country_isos(df)
    print(f"Countries: {all_countries}")

    n_classes = len(class_names)
    country_results = {}
    csv_rows = []
    all_per_sample = []

    for country in all_countries:
        print(f"\n--- LOCO: test={country} ---")
        metrics, per_sample_df = run_loco_single(
            df, feature_cols, class_names, tier, country, all_countries,
            classifier_type=classifier_type, max_samples=max_samples)

        if metrics is not None:
            print_metrics(metrics, label=country)
            country_results[country] = metrics
            csv_rows.append({
                'test_country': country,
                'feature_type': feature_type,
                'classifier': classifier_type,
                'balanced_accuracy': metrics['balanced_accuracy'],
                'macro_f1': metrics['macro_f1'],
                'weighted_f1': metrics['weighted_f1'],
                'cohen_kappa': metrics['cohen_kappa'],
                'overall_accuracy': metrics['overall_accuracy'],
                'n_train': metrics['n_train'],
                'n_test': metrics['n_test'],
            })
            all_per_sample.append(per_sample_df)

            # --- 每个国家完成后即保存 ---
            # 1. 精度指标 CSV
            pd.DataFrame(csv_rows).to_csv(
                os.path.join(out_dir, f'loco_{feature_type}{suffix}_tier_{tier}.csv'),
                index=False)

            # 2. Per-sample 预测 CSV
            pd.concat(all_per_sample, ignore_index=True).to_csv(
                os.path.join(out_dir,
                             f'loco_{feature_type}{suffix}_per_sample_tier_{tier}.csv'),
                index=False)

            print(f"  [Saved after {country}: {len(csv_rows)}/{len(all_countries)} countries]")

    # 汇总
    if country_results:
        ba_values = [m['balanced_accuracy'] for m in country_results.values()]
        f1_values = [m['macro_f1'] for m in country_results.values()]
        print(f"\n--- LOCO Summary ({feature_type.upper()}) ---")
        print(f"  Mean BA: {np.mean(ba_values):.4f} ± {np.std(ba_values):.4f}")
        print(f"  Mean F1: {np.mean(f1_values):.4f} ± {np.std(f1_values):.4f}")
        print(f"  Best: {max(country_results, key=lambda c: country_results[c]['balanced_accuracy'])}")
        print(f"  Worst: {min(country_results, key=lambda c: country_results[c]['balanced_accuracy'])}")

    return {
        'feature_type': feature_type,
        'tier': tier,
        'classifier': classifier_type,
        'per_country': country_results,
        'mean_ba': float(np.mean(ba_values)) if country_results else None,
        'std_ba': float(np.std(ba_values)) if country_results else None,
    }, csv_rows


def main():
    parser = argparse.ArgumentParser(description='Exp 2: LOCO')
    parser.add_argument('--tier', choices=['a', 'b', 'c'], default='b')
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

    exp_name = 'exp2_loco'
    out_dir = os.path.join(EXPERIMENTS_DIR, exp_name)
    os.makedirs(out_dir, exist_ok=True)

    all_results = {}
    all_csv_rows = []

    for ft in args.feature_types:
        try:
            results, csv_rows = run_experiment(ft, args.tier, out_dir,
                                               no_coords=args.no_coords,
                                               countries=countries,
                                               classifier_type=args.classifier,
                                               max_samples=args.max_samples)
            all_results[ft] = results
            all_csv_rows.extend(csv_rows)

            # 每完成一个 feature_type 也保存完整 JSON
            save_experiment_results(exp_name, all_results,
                                    sub_name=f'tier_{args.tier}{suffix}')
            print(f"  [Saved JSON after {ft}]")
        except FileNotFoundError as e:
            print(f"Skipping {ft}: {e}")

    # 最终汇总 CSV（所有 feature_type 合并）
    if all_csv_rows:
        save_results_csv(exp_name, all_csv_rows,
                         filename=f'loco_tier_{args.tier}{suffix}.csv')

    print(f"\nExp 2 complete!")


if __name__ == '__main__':
    main()
