"""
Exp 3: Cross-Climate-Zone Transfer

Leave-one-climate-zone-out (4-5 折)。
比较 AE vs S2 在不同气候带间的迁移能力。

每完成一步就保存结果 + per-sample 预测。

Usage:
    python -m src.experiments.run_climate_zone [--tier b] [--classifier lgbm]
"""
import sys
import os
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from src.experiments.experiment_utils import (
    load_eu_features, split_by_countries,
    get_trainer, build_suffix, subsample_train,
    evaluate_predictions,
    save_experiment_results, save_results_csv, print_metrics,
    filter_coords, get_all_country_isos, CLIMATE_ZONES, COUNTRY_TO_ZONE,
    EXPERIMENTS_DIR, RANDOM_STATE,
    VALID_CLASSIFIERS, CLASSIFIER_TABPFN, TABPFN_MAX_TRAIN_SAMPLES, COUNTRIES_2021,
)
from config.config import COL_PARCEL_ID, COL_COUNTRY


def build_per_sample_df(y_test, y_pred, df_test, class_names, extra_cols=None):
    """构建 per-sample 预测 DataFrame。"""
    y_true_names = [class_names[i] if i < len(class_names) else str(i)
                    for i in y_test]
    y_pred_names = [class_names[i] if i < len(class_names) else str(i)
                    for i in y_pred]
    result = {
        'parcel_id': df_test[COL_PARCEL_ID].values if COL_PARCEL_ID in df_test.columns else range(len(y_test)),
        'country': df_test[COL_COUNTRY].values if COL_COUNTRY in df_test.columns else '',
        'y_true_encoded': y_test,
        'y_pred_encoded': y_pred,
        'y_true_label': y_true_names,
        'y_pred_label': y_pred_names,
        'correct': (y_test == y_pred).astype(int),
    }
    if extra_cols:
        result.update(extra_cols)
    return pd.DataFrame(result)


def run_experiment(feature_type, tier, out_dir, no_coords=False, countries=None,
                   classifier_type='lgbm', max_samples=0):
    """运行跨气候带迁移实验。"""
    suffix = build_suffix(no_coords, max_samples, classifier_type)
    tag = f" [{classifier_type}]" + (" [no-coords]" if no_coords else "")
    print(f"\n{'=' * 60}")
    print(f"EXP 3: CLIMATE ZONE TRANSFER — {feature_type.upper()} Tier {tier.upper()}{tag}")
    print(f"{'=' * 60}")

    df, feature_cols, class_names = load_eu_features(feature_type, tier, countries=countries)
    if no_coords:
        feature_cols = filter_coords(feature_cols)
    all_countries = get_all_country_isos(df)
    n_classes = len(class_names)

    # 按气候带分组
    zone_countries = {}
    for zone, countries in CLIMATE_ZONES.items():
        present = [c for c in countries if c in all_countries]
        if present:
            zone_countries[zone] = present
            print(f"  {zone}: {present}")

    zones = sorted(zone_countries.keys())

    trainer = get_trainer(classifier_type)

    # ============================================
    # Part 1: Leave-one-zone-out
    # ============================================
    zone_results = {}
    csv_rows = []
    all_per_sample = []

    for test_zone in zones:
        test_countries = zone_countries[test_zone]
        train_countries = []
        for z in zones:
            if z != test_zone:
                train_countries.extend(zone_countries[z])

        print(f"\n--- Test zone: {test_zone} ({test_countries}) ---")

        test_mask = df[COL_COUNTRY].isin(test_countries)
        df_test = df[test_mask]

        X_train, y_train, X_val, y_val, X_test, y_test = split_by_countries(
            df, feature_cols, tier,
            train_countries=train_countries,
            test_countries=test_countries)

        if len(X_test) < 10:
            print(f"  Too few test samples, skipping")
            continue

        if max_samples:
            X_train, y_train = subsample_train(X_train, y_train, max_samples,
                                               seed=RANDOM_STATE)

        model = trainer(X_train, y_train, X_val, y_val, n_classes)
        y_pred = model.predict(X_test)
        metrics = evaluate_predictions(y_test, y_pred, class_names)

        print_metrics(metrics, label=test_zone)
        zone_results[test_zone] = metrics

        csv_rows.append({
            'test_zone': test_zone,
            'test_countries': ','.join(test_countries),
            'feature_type': feature_type,
            'classifier': classifier_type,
            'balanced_accuracy': metrics['balanced_accuracy'],
            'macro_f1': metrics['macro_f1'],
            'weighted_f1': metrics['weighted_f1'],
            'cohen_kappa': metrics['cohen_kappa'],
            'n_train': len(X_train),
            'n_test': len(X_test),
        })

        # Per-sample 预测
        ps_df = build_per_sample_df(
            y_test, y_pred, df_test, class_names,
            extra_cols={'test_zone': test_zone})
        all_per_sample.append(ps_df)

        # 增量保存
        pd.DataFrame(csv_rows).to_csv(
            os.path.join(out_dir, f'lozo_{feature_type}{suffix}_tier_{tier}.csv'),
            index=False)
        pd.concat(all_per_sample, ignore_index=True).to_csv(
            os.path.join(out_dir,
                         f'lozo_{feature_type}{suffix}_per_sample_tier_{tier}.csv'),
            index=False)
        print(f"  [Saved after {test_zone}]")

    # ============================================
    # Part 2: 跨气候带迁移矩阵（train 单区 -> test 单区）
    # ============================================
    print(f"\n--- Cross-zone transfer matrix ---")
    transfer_matrix = {}
    transfer_rows = []
    transfer_per_sample = []

    for train_zone in zones:
        transfer_matrix[train_zone] = {}
        for test_zone in zones:
            if train_zone == test_zone:
                continue

            train_countries = zone_countries[train_zone]
            test_countries = zone_countries[test_zone]

            test_mask = df[COL_COUNTRY].isin(test_countries)
            df_test = df[test_mask]

            X_train, y_train, X_val, y_val, X_test, y_test = split_by_countries(
                df, feature_cols, tier,
                train_countries=train_countries,
                test_countries=test_countries)

            if len(X_train) < 50 or len(X_test) < 10:
                continue

            if max_samples:
                X_train, y_train = subsample_train(X_train, y_train,
                                                   max_samples,
                                                   seed=RANDOM_STATE)

            model = trainer(X_train, y_train, X_val, y_val, n_classes)
            y_pred = model.predict(X_test)
            metrics = evaluate_predictions(y_test, y_pred, class_names)

            transfer_matrix[train_zone][test_zone] = metrics
            print(f"  {train_zone:15s} -> {test_zone:15s}: "
                  f"BA={metrics['balanced_accuracy']:.3f}")

            transfer_rows.append({
                'train_zone': train_zone,
                'test_zone': test_zone,
                'feature_type': feature_type,
                'classifier': classifier_type,
                'balanced_accuracy': metrics['balanced_accuracy'],
                'macro_f1': metrics['macro_f1'],
                'n_train': len(X_train),
                'n_test': len(X_test),
            })

            # Per-sample 预测
            ps_df = build_per_sample_df(
                y_test, y_pred, df_test, class_names,
                extra_cols={'train_zone': train_zone, 'test_zone': test_zone})
            transfer_per_sample.append(ps_df)

    # 保存迁移矩阵结果
    if transfer_rows:
        pd.DataFrame(transfer_rows).to_csv(
            os.path.join(out_dir,
                         f'transfer_matrix_{feature_type}{suffix}_tier_{tier}.csv'),
            index=False)
    if transfer_per_sample:
        pd.concat(transfer_per_sample, ignore_index=True).to_csv(
            os.path.join(out_dir,
                         f'transfer_{feature_type}{suffix}_per_sample_tier_{tier}.csv'),
            index=False)

    # 汇总
    if zone_results:
        ba_values = [m['balanced_accuracy'] for m in zone_results.values()]
        print(f"\n--- Zone Summary ({feature_type.upper()}) ---")
        print(f"  Mean BA: {np.mean(ba_values):.4f}")
        for z in sorted(zone_results.keys()):
            print(f"  {z:15s}: BA={zone_results[z]['balanced_accuracy']:.4f}")

    return {
        'feature_type': feature_type,
        'tier': tier,
        'zone_results': zone_results,
        'transfer_matrix': transfer_matrix,
    }, csv_rows + transfer_rows


def main():
    parser = argparse.ArgumentParser(description='Exp 3: Climate Zone Transfer')
    parser.add_argument('--tier', choices=['a', 'b', 'c'], default='b')
    parser.add_argument('--feature-types', nargs='+',
                        default=['ae', 's2', 'fusion'],
                        choices=['ae', 's2', 'fusion'])
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

    exp_name = 'exp3_climate_zone'
    out_dir = os.path.join(EXPERIMENTS_DIR, exp_name)
    os.makedirs(out_dir, exist_ok=True)

    all_results = {}
    all_csv_rows = []

    for ft in args.feature_types:
        try:
            results, csv_rows = run_experiment(
                ft, args.tier, out_dir,
                no_coords=args.no_coords, countries=countries,
                classifier_type=args.classifier,
                max_samples=max_samples)
            all_results[ft] = results
            all_csv_rows.extend(csv_rows)

            # 每完成一个 feature_type 保存 JSON
            save_experiment_results(exp_name, all_results,
                                    sub_name=f'tier_{args.tier}{suffix}')
            print(f"  [Saved JSON after {ft}]")
        except FileNotFoundError as e:
            print(f"Skipping {ft}: {e}")

    print(f"\nExp 3 complete!")


if __name__ == '__main__':
    main()
