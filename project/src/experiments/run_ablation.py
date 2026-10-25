"""
Exp 6: Ablation Study

量化各特征组件的边际贡献，在 ID 和 OOD 两种场景下评估。

消融组：
  (a) AE mean only (64d)
  (b) AE mean + std (128d)
  (c) AE 128d + coords (lon/lat)
  (d) AE 128d + area (log_area)
  (e) AE 128d + coords + area (full AE)
  (f) S2 only
  (g) S2 + coords + area
  (h) AE + S2 fusion (full)

Usage:
    python -m src.experiments.run_ablation [--tier b]
"""
import sys
import os
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from src.experiments.experiment_utils import (
    load_eu_features, prepare_xy, split_by_countries,
    train_lgbm, evaluate_predictions,
    save_experiment_results, save_results_csv, print_metrics,
    get_all_country_isos, RANDOM_STATE,
)
from config.config import AE_BANDS, COL_LON, COL_LAT
from config.spatial_splits import EU_TEST_COUNTRIES, EU_VAL_COUNTRIES


def define_ablation_groups(feature_cols):
    """根据可用特征列定义消融组。"""
    ae_mean = [c for c in feature_cols if any(
        c == f'{b}_mean' for b in AE_BANDS)]
    ae_std = [c for c in feature_cols if any(
        c in (f'{b}_std', f'{b}_stdDev') for b in AE_BANDS)]
    coords = [c for c in feature_cols if c in [COL_LON, COL_LAT,
                                                'longitude', 'latitude']]
    area = [c for c in feature_cols if c in ['log_area']]

    # S2 特征 = 非 AE、非辅助的特征列
    ae_all = set(ae_mean + ae_std)
    aux_all = set(coords + area)
    s2_cols = [c for c in feature_cols
               if c not in ae_all and c not in aux_all
               and not c.startswith('ctry_')]

    groups = {}

    if ae_mean:
        groups['ae_mean_only'] = ae_mean
    if ae_mean and ae_std:
        groups['ae_mean_std'] = ae_mean + ae_std
    if ae_mean and ae_std and coords:
        groups['ae_128d_coords'] = ae_mean + ae_std + coords
    if ae_mean and ae_std and area:
        groups['ae_128d_area'] = ae_mean + ae_std + area
    if ae_mean and ae_std:
        groups['ae_full'] = ae_mean + ae_std + coords + area
    if s2_cols:
        groups['s2_only'] = s2_cols
        groups['s2_full'] = s2_cols + coords + area
    if ae_mean and ae_std and s2_cols:
        groups['fusion_full'] = ae_mean + ae_std + s2_cols + coords + area

    return groups


def run_experiment(feature_type, tier):
    """运行消融实验。"""
    print(f"\n{'=' * 60}")
    print(f"EXP 6: ABLATION — {feature_type.upper()} Tier {tier.upper()}")
    print(f"{'=' * 60}")

    df, feature_cols, class_names = load_eu_features(feature_type, tier)
    all_countries = get_all_country_isos(df)
    n_classes = len(class_names)

    # 定义消融组
    groups = define_ablation_groups(feature_cols)
    print(f"\nAblation groups:")
    for name, cols in groups.items():
        print(f"  {name:20s}: {len(cols)} features")

    # 准备 ID 和 OOD 数据
    test_countries = [c for c in EU_TEST_COUNTRIES if c in all_countries]
    val_countries = [c for c in EU_VAL_COUNTRIES if c in all_countries]
    train_countries = [c for c in all_countries
                       if c not in test_countries and c not in val_countries]

    # OOD split
    ood_splits = split_by_countries(
        df, feature_cols, tier,
        train_countries=train_countries,
        test_countries=test_countries,
        val_countries=val_countries)

    # ID split (随机)
    X_all, y_all = prepare_xy(df, feature_cols, tier)
    rng = np.random.RandomState(RANDOM_STATE)
    indices = rng.permutation(len(X_all))
    n_test = len(X_all) // 10
    n_val = len(X_all) // 10
    id_test_idx = indices[:n_test]
    id_val_idx = indices[n_test:n_test + n_val]
    id_train_idx = indices[n_test + n_val:]

    csv_rows = []
    results = {}

    for group_name, group_cols in groups.items():
        print(f"\n--- {group_name} ({len(group_cols)}d) ---")

        # 找到 group_cols 在 feature_cols 中的索引
        col_indices = [feature_cols.index(c) for c in group_cols
                       if c in feature_cols]

        if not col_indices:
            print(f"  No valid columns, skipping")
            continue

        # ID 评估
        X_id_train = X_all[id_train_idx][:, col_indices]
        y_id_train = y_all[id_train_idx]
        X_id_val = X_all[id_val_idx][:, col_indices]
        y_id_val = y_all[id_val_idx]
        X_id_test = X_all[id_test_idx][:, col_indices]
        y_id_test = y_all[id_test_idx]

        model_id = train_lgbm(X_id_train, y_id_train, X_id_val, y_id_val,
                               n_classes)
        y_pred_id = model_id.predict(X_id_test)
        metrics_id = evaluate_predictions(y_id_test, y_pred_id)

        # OOD 评估
        X_ood_train, y_ood_train = ood_splits[0][:, col_indices], ood_splits[1]
        X_ood_val, y_ood_val = ood_splits[2][:, col_indices], ood_splits[3]
        X_ood_test, y_ood_test = ood_splits[4][:, col_indices], ood_splits[5]

        model_ood = train_lgbm(X_ood_train, y_ood_train,
                                X_ood_val, y_ood_val, n_classes)
        y_pred_ood = model_ood.predict(X_ood_test)
        metrics_ood = evaluate_predictions(y_ood_test, y_pred_ood)

        delta_ba = metrics_id['balanced_accuracy'] - metrics_ood['balanced_accuracy']

        print(f"  ID:  BA={metrics_id['balanced_accuracy']:.4f}, "
              f"F1={metrics_id['macro_f1']:.4f}")
        print(f"  OOD: BA={metrics_ood['balanced_accuracy']:.4f}, "
              f"F1={metrics_ood['macro_f1']:.4f}")
        print(f"  Δ:   {delta_ba:+.4f}")

        results[group_name] = {
            'n_features': len(col_indices),
            'id': metrics_id,
            'ood': metrics_ood,
            'delta_ba': float(delta_ba),
        }

        csv_rows.append({
            'group': group_name,
            'n_features': len(col_indices),
            'id_BA': metrics_id['balanced_accuracy'],
            'id_F1': metrics_id['macro_f1'],
            'ood_BA': metrics_ood['balanced_accuracy'],
            'ood_F1': metrics_ood['macro_f1'],
            'delta_BA': delta_ba,
        })

    return results, csv_rows


def main():
    parser = argparse.ArgumentParser(description='Exp 6: Ablation Study')
    parser.add_argument('--tier', choices=['a', 'b', 'c'], default='b')
    parser.add_argument('--feature-type', default='fusion',
                        choices=['ae', 's2', 'fusion'],
                        help='Base feature set for ablation')
    args = parser.parse_args()

    results, csv_rows = run_experiment(args.feature_type, args.tier)

    exp_name = 'exp6_ablation'
    save_experiment_results(exp_name, results,
                            sub_name=f'{args.feature_type}_tier_{args.tier}')
    if csv_rows:
        save_results_csv(exp_name, csv_rows,
                         filename=f'ablation_{args.feature_type}_tier_{args.tier}.csv')

    # 打印汇总表
    print(f"\n{'=' * 70}")
    print(f"{'Group':20s} {'#Feat':>5s} {'ID BA':>7s} {'OOD BA':>7s} {'Δ':>7s}")
    print(f"{'=' * 70}")
    for row in csv_rows:
        print(f"{row['group']:20s} {row['n_features']:5d} "
              f"{row['id_BA']:7.4f} {row['ood_BA']:7.4f} "
              f"{row['delta_BA']:+7.4f}")

    print(f"\nExp 6 complete!")


if __name__ == '__main__':
    main()
