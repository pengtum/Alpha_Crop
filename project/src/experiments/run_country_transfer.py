"""
Exp 2b: Country-to-Country Transfer Matrix

在单个国家训练，逐个预测其他所有国家，生成 18×18 迁移矩阵。
揭示国家对之间的 pairwise transferability。

每完成一个 train→test 对就增量保存：
  1. 迁移矩阵指标 CSV
  2. 每个测试样本的真值、预测值 (per-sample CSV)

Usage:
    python -m src.experiments.run_country_transfer [--tier b] \
        [--feature-types ae s2 fusion] [--classifier lgbm] [--no-coords]
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
    evaluate_predictions, save_experiment_results, print_metrics,
    filter_coords, get_all_country_isos,
    RANDOM_STATE, EXPERIMENTS_DIR,
    VALID_CLASSIFIERS, CLASSIFIER_TABPFN, TABPFN_MAX_TRAIN_SAMPLES, COUNTRIES_2021,
)
from config.config import COL_PARCEL_ID, COL_COUNTRY

EXP_NAME = 'exp2b_country_transfer'


def run_single_transfer(df, feature_cols, class_names, tier,
                        train_country, test_country,
                        classifier_type='lgbm', max_samples=0):
    """在 train_country 训练，在 test_country 测试。

    Returns:
        metrics: 评估指标字典（或 None）
        per_sample_df: 逐样本预测 DataFrame（或 None）
    """
    n_classes = len(class_names)

    X_train, y_train, X_val, y_val, X_test, y_test = split_by_countries(
        df, feature_cols, tier,
        train_countries=[train_country],
        test_countries=[test_country])

    if len(X_train) < 50 or len(X_test) < 10:
        return None, None

    # 统一子采样
    if max_samples:
        X_train, y_train = subsample_train(
            X_train, y_train, max_samples, seed=RANDOM_STATE)
        X_val, y_val = subsample_train(
            X_val, y_val, max(500, max_samples // 5),
            seed=RANDOM_STATE + 100)

    trainer = get_trainer(classifier_type)
    model = trainer(X_train, y_train, X_val, y_val, n_classes)
    y_pred = model.predict(X_test)

    metrics = evaluate_predictions(y_test, y_pred, class_names)
    metrics['train_country'] = train_country
    metrics['test_country'] = test_country
    metrics['n_train'] = len(X_train)
    metrics['n_test'] = len(X_test)

    # 构建 per-sample 预测 DataFrame
    test_mask = df[COL_COUNTRY] == test_country
    df_test = df[test_mask]

    y_true_names = [class_names[i] if i < len(class_names) else str(i)
                    for i in y_test]
    y_pred_names = [class_names[i] if i < len(class_names) else str(i)
                    for i in y_pred]

    per_sample_df = pd.DataFrame({
        'parcel_id': df_test[COL_PARCEL_ID].values[:len(y_test)]
                     if COL_PARCEL_ID in df_test.columns
                     else range(len(y_test)),
        'train_country': train_country,
        'test_country': test_country,
        'y_true_encoded': y_test,
        'y_pred_encoded': y_pred,
        'y_true_label': y_true_names,
        'y_pred_label': y_pred_names,
        'correct': (y_test == y_pred).astype(int),
    })

    return metrics, per_sample_df


def run_experiment(feature_type, tier, out_dir, suffix,
                   no_coords=False, countries=None, classifier_type='lgbm', max_samples=0):
    """对单个 feature_type 运行完整 country-to-country 迁移矩阵。

    优化：每个训练国只训练一次模型，然后 predict 所有目标国（17 倍加速）。
    """
    tags = []
    if no_coords:
        tags.append("no-coords")
    if max_samples:
        tags.append(f"sub{max_samples // 1000}k")
    tags.append(classifier_type.upper())
    tag_str = f" [{', '.join(tags)}]"

    print(f"\n{'=' * 60}")
    print(f"EXP 2b: COUNTRY TRANSFER — {feature_type.upper()} "
          f"Tier {tier.upper()}{tag_str}")
    print(f"{'=' * 60}")

    df, feature_cols, class_names = load_eu_features(feature_type, tier, countries=countries)
    if no_coords:
        feature_cols = filter_coords(feature_cols)
    all_countries = get_all_country_isos(df)
    n_countries = len(all_countries)
    n_classes = len(class_names)
    total_pairs = n_countries * (n_countries - 1)

    print(f"Countries: {all_countries} ({n_countries})")
    print(f"Total pairs: {total_pairs}")
    print(f"Optimized: train once per country, predict 17 targets")

    # 预先按国家切分数据（避免重复 filter）
    encoded_col = f'label_tier_{tier}_encoded'
    country_data = {}
    for c in all_countries:
        mask = df[COL_COUNTRY] == c
        df_c = df[mask]
        X_c = df_c[feature_cols].fillna(0).values
        y_c = df_c[encoded_col].values
        country_data[c] = {
            'X': X_c, 'y': y_c, 'df': df_c, 'n': len(X_c)
        }
        print(f"  {c}: {len(X_c):,} samples")

    transfer_results = {}
    csv_rows = []
    all_per_sample = []
    pair_count = 0

    for i_train, train_country in enumerate(all_countries):
        transfer_results[train_country] = {}
        cd = country_data[train_country]

        if cd['n'] < 50:
            print(f"\n  {train_country}: too few samples ({cd['n']}), skipping")
            continue

        # ─── 训练一次 ────────────────────────────────────
        print(f"\n{'─' * 50}")
        print(f"[{i_train+1}/{n_countries}] Training on {train_country} "
              f"({cd['n']:,} samples)...")

        X_all, y_all = cd['X'], cd['y']

        # 从训练国中分出 val（10%）
        # 先把极稀有类（<5 样本）的索引锁定在训练集，避免 val 中出现 unseen labels
        from collections import Counter
        class_counts = Counter(y_all)
        rare_idx = np.array([i for i, label in enumerate(y_all)
                             if class_counts[label] < 5], dtype=np.intp)
        rare_set = set(rare_idx)
        pool_idx = np.array([i for i in range(len(y_all))
                             if i not in rare_set], dtype=np.intp)

        rng = np.random.RandomState(RANDOM_STATE)
        n_val = max(1, int(len(pool_idx) * 0.1))
        val_pos = rng.choice(len(pool_idx), size=n_val, replace=False)
        val_idx = pool_idx[val_pos]
        train_idx = np.concatenate([
            np.setdiff1d(pool_idx, val_idx), rare_idx
        ])

        X_train, y_train = X_all[train_idx], y_all[train_idx]
        X_val, y_val = X_all[val_idx], y_all[val_idx]

        # 统一子采样
        if max_samples:
            X_train, y_train = subsample_train(
                X_train, y_train, max_samples, seed=RANDOM_STATE)
            X_val, y_val = subsample_train(
                X_val, y_val, max(500, max_samples // 5),
                seed=RANDOM_STATE + 100)

        trainer = get_trainer(classifier_type)
        model = trainer(X_train, y_train, X_val, y_val, n_classes)
        n_train_actual = len(X_train)

        # ─── 预测所有目标国 ────────────────────────────────
        train_bas = []
        for test_country in all_countries:
            if test_country == train_country:
                continue

            pair_count += 1
            td = country_data[test_country]
            X_test, y_test = td['X'], td['y']

            if len(X_test) < 10:
                print(f"  -> {test_country}: SKIPPED (too few)")
                continue

            y_pred = model.predict(X_test)
            metrics = evaluate_predictions(y_test, y_pred, class_names)
            ba = metrics['balanced_accuracy']
            train_bas.append(ba)

            transfer_results[train_country][test_country] = metrics

            csv_rows.append({
                'train_country': train_country,
                'test_country': test_country,
                'feature_type': feature_type,
                'classifier': classifier_type,
                'balanced_accuracy': ba,
                'macro_f1': metrics['macro_f1'],
                'weighted_f1': metrics['weighted_f1'],
                'cohen_kappa': metrics['cohen_kappa'],
                'overall_accuracy': metrics['overall_accuracy'],
                'n_train': n_train_actual,
                'n_test': len(X_test),
            })

            # per-sample 预测（跳过，文件太大 >800MB 导致 Google Drive 写入失败）
            # 如需保存，取消下方注释
            # df_test = td['df']
            # y_true_names = [class_names[j] if j < len(class_names) else str(j)
            #                 for j in y_test]
            # y_pred_names = [class_names[j] if j < len(class_names) else str(j)
            #                 for j in y_pred]
            # ps_df = pd.DataFrame({...})
            # all_per_sample.append(ps_df)

            print(f"  -> {test_country}: BA={ba:.4f} "
                  f"(n_test={len(X_test):,})")

        # 该训练国的汇总
        if train_bas:
            print(f"  [{train_country}] mean OOD BA: "
                  f"{np.mean(train_bas):.4f} ± {np.std(train_bas):.4f}")

        # 增量保存 CSV
        pd.DataFrame(csv_rows).to_csv(
            os.path.join(out_dir,
                         f'transfer_{feature_type}{suffix}_tier_{tier}.csv'),
            index=False)
        # per-sample CSV 跳过（文件太大，Google Drive 写入失败）
        print(f"  [Saved after train={train_country}: "
              f"{pair_count}/{total_pairs} pairs]")

    # 汇总
    if csv_rows:
        ba_values = [r['balanced_accuracy'] for r in csv_rows]
        print(f"\n--- Transfer Summary ({feature_type.upper()}) ---")
        print(f"  Pairs completed: {len(csv_rows)}/{total_pairs}")
        print(f"  Mean BA: {np.mean(ba_values):.4f} ± {np.std(ba_values):.4f}")
        print(f"  Min BA: {min(ba_values):.4f}")
        print(f"  Max BA: {max(ba_values):.4f}")

        best_row = max(csv_rows, key=lambda r: r['balanced_accuracy'])
        worst_row = min(csv_rows, key=lambda r: r['balanced_accuracy'])
        print(f"  Best:  {best_row['train_country']} -> "
              f"{best_row['test_country']} "
              f"(BA={best_row['balanced_accuracy']:.4f})")
        print(f"  Worst: {worst_row['train_country']} -> "
              f"{worst_row['test_country']} "
              f"(BA={worst_row['balanced_accuracy']:.4f})")

        print(f"\n  Per-train-country mean BA:")
        for tc in all_countries:
            tc_bas = [r['balanced_accuracy'] for r in csv_rows
                      if r['train_country'] == tc]
            if tc_bas:
                print(f"    {tc}: {np.mean(tc_bas):.4f} "
                      f"(n_targets={len(tc_bas)})")

    return {
        'feature_type': feature_type,
        'tier': tier,
        'classifier': classifier_type,
        'transfer_matrix': transfer_results,
        'n_pairs': len(csv_rows),
        'mean_ba': float(np.mean(ba_values)) if csv_rows else None,
    }, csv_rows


def main():
    parser = argparse.ArgumentParser(
        description='Exp 2b: Country-to-Country Transfer Matrix')
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

    out_dir = os.path.join(EXPERIMENTS_DIR, EXP_NAME)
    os.makedirs(out_dir, exist_ok=True)

    all_results = {}

    for ft in args.feature_types:
        try:
            results, csv_rows = run_experiment(
                ft, args.tier, out_dir, suffix,
                no_coords=args.no_coords, countries=countries,
                classifier_type=args.classifier,
                max_samples=args.max_samples)
            all_results[ft] = results

            # 每完成一个 feature_type 保存 JSON
            save_experiment_results(EXP_NAME, all_results,
                                    sub_name=f'tier_{args.tier}{suffix}')
            print(f"  [Saved JSON after {ft}]")
        except FileNotFoundError as e:
            print(f"Skipping {ft}: {e}")

    print(f"\nExp 2b complete!")


if __name__ == '__main__':
    main()
