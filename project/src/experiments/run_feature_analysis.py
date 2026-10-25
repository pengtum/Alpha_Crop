"""
Exp 5: Feature Space Analysis + SHAP

分析 1: t-SNE/UMAP 可视化 (按作物/按国家着色)
分析 2: CKA — 嵌入与作物标签 vs 嵌入与国家标签的相关性
分析 3: SHAP — 特征重要性、ID vs OOD SHAP 差异
分析 4: 国家可预测性 — 用嵌入预测国家的准确率（探索性）

Usage:
    python -m src.experiments.run_feature_analysis [--tier b] [--n-samples 5000] [--classifier lgbm]
"""
import sys
import os
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from src.experiments.experiment_utils import (
    load_eu_features, prepare_xy, split_by_countries,
    get_trainer, build_suffix, subsample_train,
    evaluate_predictions,
    save_experiment_results, save_results_csv,
    filter_coords, get_all_country_isos, EXPERIMENTS_DIR, RANDOM_STATE,
    VALID_CLASSIFIERS, CLASSIFIER_TABPFN, TABPFN_MAX_TRAIN_SAMPLES, COUNTRIES_2021,
)
from config.config import COL_COUNTRY, FIGURES_DIR
from config.spatial_splits import EU_TEST_COUNTRIES, EU_VAL_COUNTRIES


def analysis_tsne_umap(df, feature_cols, tier, feature_type, n_samples=5000):
    """t-SNE / UMAP 可视化。"""
    print(f"\n--- Analysis 1: t-SNE/UMAP ({feature_type.upper()}) ---")

    rng = np.random.RandomState(RANDOM_STATE)
    if len(df) > n_samples:
        idx = rng.choice(len(df), size=n_samples, replace=False)
        df_sub = df.iloc[idx].reset_index(drop=True)
    else:
        df_sub = df

    X = df_sub[feature_cols].fillna(0).values
    labels_crop = df_sub[f'label_tier_{tier}'].values
    labels_country = df_sub[COL_COUNTRY].values

    results = {
        'n_samples': len(df_sub),
        'n_features': len(feature_cols),
    }

    # t-SNE
    try:
        from sklearn.manifold import TSNE
        print("  Computing t-SNE...")
        tsne = TSNE(n_components=2, random_state=RANDOM_STATE, perplexity=30)
        X_tsne = tsne.fit_transform(X)
        results['tsne'] = {
            'coords': X_tsne.tolist(),
            'crop_labels': labels_crop.tolist(),
            'country_labels': labels_country.tolist(),
        }
        print(f"  t-SNE done: shape={X_tsne.shape}")
    except Exception as e:
        print(f"  t-SNE failed: {e}")

    # UMAP
    try:
        import umap
        print("  Computing UMAP...")
        reducer = umap.UMAP(n_components=2, random_state=RANDOM_STATE)
        X_umap = reducer.fit_transform(X)
        results['umap'] = {
            'coords': X_umap.tolist(),
            'crop_labels': labels_crop.tolist(),
            'country_labels': labels_country.tolist(),
        }
        print(f"  UMAP done: shape={X_umap.shape}")
    except ImportError:
        print("  UMAP not installed (pip install umap-learn)")
    except Exception as e:
        print(f"  UMAP failed: {e}")

    return results


def analysis_cka(df, feature_cols, tier, feature_type, n_samples=5000):
    """Centered Kernel Alignment (CKA) 分析。

    比较嵌入与作物标签 vs 嵌入与国家标签的相关性。
    """
    print(f"\n--- Analysis 2: CKA ({feature_type.upper()}) ---")

    rng = np.random.RandomState(RANDOM_STATE)
    if len(df) > n_samples:
        idx = rng.choice(len(df), size=n_samples, replace=False)
        df_sub = df.iloc[idx].reset_index(drop=True)
    else:
        df_sub = df

    X = df_sub[feature_cols].fillna(0).values

    # 作物标签矩阵 (one-hot)
    from sklearn.preprocessing import LabelEncoder
    le_crop = LabelEncoder()
    y_crop = le_crop.fit_transform(df_sub[f'label_tier_{tier}'])
    Y_crop = np.eye(len(le_crop.classes_))[y_crop]

    # 国家标签矩阵 (one-hot)
    le_country = LabelEncoder()
    y_country = le_country.fit_transform(df_sub[COL_COUNTRY])
    Y_country = np.eye(len(le_country.classes_))[y_country]

    def linear_cka(X, Y):
        """Compute Linear CKA between two matrices (different dims OK)."""
        X = X - X.mean(axis=0)
        Y = Y - Y.mean(axis=0)
        XtY = X.T @ Y   # (d_x, d_y)
        XtX = X.T @ X   # (d_x, d_x)
        YtY = Y.T @ Y   # (d_y, d_y)
        hsic_xy = np.linalg.norm(XtY, 'fro') ** 2 / (len(X) - 1) ** 2
        hsic_xx = np.linalg.norm(XtX, 'fro') ** 2 / (len(X) - 1) ** 2
        hsic_yy = np.linalg.norm(YtY, 'fro') ** 2 / (len(X) - 1) ** 2
        return hsic_xy / np.sqrt(hsic_xx * hsic_yy + 1e-10)

    cka_crop = linear_cka(X, Y_crop)
    cka_country = linear_cka(X, Y_country)

    print(f"  CKA(embedding, crop): {cka_crop:.4f}")
    print(f"  CKA(embedding, country): {cka_country:.4f}")
    print(f"  Ratio (crop/country): {cka_crop / (cka_country + 1e-10):.2f}")

    return {
        'cka_crop': float(cka_crop),
        'cka_country': float(cka_country),
        'ratio': float(cka_crop / (cka_country + 1e-10)),
        'n_samples': len(df_sub),
    }


def analysis_shap(df, feature_cols, class_names, tier, feature_type,
                  classifier_type='lgbm', max_samples=0):
    """SHAP 分析 — ID 和 OOD 场景对比。"""
    print(f"\n--- Analysis 3: SHAP ({feature_type.upper()}, {classifier_type}) ---")

    try:
        import shap
    except ImportError:
        print("  SHAP not installed (pip install shap)")
        return None

    all_countries = get_all_country_isos(df)
    test_countries = [c for c in EU_TEST_COUNTRIES if c in all_countries]
    val_countries = [c for c in EU_VAL_COUNTRIES if c in all_countries]
    train_countries = [c for c in all_countries
                       if c not in test_countries and c not in val_countries]

    X_train, y_train, X_val, y_val, X_test, y_test = split_by_countries(
        df, feature_cols, tier,
        train_countries=train_countries,
        test_countries=test_countries,
        val_countries=val_countries)

    n_classes = len(class_names)

    if max_samples:
        X_train, y_train = subsample_train(X_train, y_train, max_samples,
                                           seed=RANDOM_STATE)

    trainer = get_trainer(classifier_type)
    model = trainer(X_train, y_train, X_val, y_val, n_classes)

    # SHAP on ID (training subset) vs OOD (test)
    rng = np.random.RandomState(RANDOM_STATE)

    if classifier_type == CLASSIFIER_TABPFN:
        # TabPFN: 用 KernelExplainer（模型无关，较慢），样本量缩减
        n_shap = min(200, len(X_train))
        n_shap_ood = min(200, len(X_test))

        id_idx = rng.choice(len(X_train), size=n_shap, replace=False)
        X_id_sample = X_train[id_idx]
        ood_idx = rng.choice(len(X_test), size=n_shap_ood, replace=False)
        X_ood_sample = X_test[ood_idx]

        print(f"  Computing SHAP via KernelExplainer (ID: {n_shap}, OOD: {n_shap_ood})...")
        background = shap.sample(X_train, min(100, len(X_train)))
        explainer = shap.KernelExplainer(model.predict_proba, background)
    else:
        # Tree-based: 用 TreeExplainer（快速精确）
        n_shap = min(1000, len(X_train))
        n_shap_ood = min(1000, len(X_test))

        id_idx = rng.choice(len(X_train), size=n_shap, replace=False)
        X_id_sample = X_train[id_idx]
        ood_idx = rng.choice(len(X_test), size=n_shap_ood, replace=False)
        X_ood_sample = X_test[ood_idx]

        print(f"  Computing SHAP via TreeExplainer (ID: {n_shap}, OOD: {n_shap_ood})...")
        explainer = shap.TreeExplainer(model)

    shap_id = explainer.shap_values(X_id_sample)
    shap_ood = explainer.shap_values(X_ood_sample)

    # 计算平均绝对 SHAP 值 (跨类别)
    if isinstance(shap_id, list):
        # 多分类 old SHAP: list of (n_samples, n_features)
        mean_abs_shap_id = np.mean([np.abs(s).mean(axis=0) for s in shap_id],
                                    axis=0)
        mean_abs_shap_ood = np.mean([np.abs(s).mean(axis=0) for s in shap_ood],
                                     axis=0)
    elif hasattr(shap_id, 'ndim') and shap_id.ndim == 3:
        # 多分类 SHAP >=0.50: (n_samples, n_features, n_classes)
        mean_abs_shap_id = np.abs(shap_id).mean(axis=(0, 2))
        mean_abs_shap_ood = np.abs(shap_ood).mean(axis=(0, 2))
    else:
        mean_abs_shap_id = np.abs(shap_id).mean(axis=0)
        mean_abs_shap_ood = np.abs(shap_ood).mean(axis=0)

    # 排序
    id_importance = sorted(zip(feature_cols, mean_abs_shap_id),
                            key=lambda x: x[1], reverse=True)
    ood_importance = sorted(zip(feature_cols, mean_abs_shap_ood),
                             key=lambda x: x[1], reverse=True)

    print(f"\n  Top 10 features (ID):")
    for name, val in id_importance[:10]:
        print(f"    {name:30s}: {val:.6f}")

    print(f"\n  Top 10 features (OOD):")
    for name, val in ood_importance[:10]:
        print(f"    {name:30s}: {val:.6f}")

    # 计算 ID vs OOD SHAP 差异
    shap_diff = mean_abs_shap_ood - mean_abs_shap_id
    diff_sorted = sorted(zip(feature_cols, shap_diff),
                          key=lambda x: abs(x[1]), reverse=True)

    print(f"\n  Largest SHAP shifts (OOD - ID):")
    for name, val in diff_sorted[:10]:
        direction = "+" if val > 0 else "-"
        print(f"    {name:30s}: {direction} {abs(val):.6f}")

    return {
        'id_importance': {name: float(val) for name, val in id_importance},
        'ood_importance': {name: float(val) for name, val in ood_importance},
        'shap_diff': {name: float(val) for name, val in diff_sorted[:30]},
        'n_id_samples': n_shap,
        'n_ood_samples': n_shap_ood,
        'classifier': classifier_type,
    }


def analysis_country_prediction(df, feature_cols, feature_type, n_samples=10000,
                                classifier_type='lgbm', max_samples=0):
    """国家可预测性 — 用嵌入预测国家的准确率。"""
    print(f"\n--- Analysis 4: Country Predictability ({feature_type.upper()}, {classifier_type}) ---")

    from sklearn.preprocessing import LabelEncoder

    rng = np.random.RandomState(RANDOM_STATE)
    if len(df) > n_samples:
        idx = rng.choice(len(df), size=n_samples, replace=False)
        df_sub = df.iloc[idx].reset_index(drop=True)
    else:
        df_sub = df

    X = df_sub[feature_cols].fillna(0).values
    le = LabelEncoder()
    y = le.fit_transform(df_sub[COL_COUNTRY])
    n_classes = len(le.classes_)

    # 随机 80/10/10 split
    from sklearn.model_selection import train_test_split
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.1, random_state=RANDOM_STATE, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.11, random_state=RANDOM_STATE,
        stratify=y_trainval)

    if max_samples:
        X_train, y_train = subsample_train(X_train, y_train, max_samples,
                                           seed=RANDOM_STATE)

    trainer = get_trainer(classifier_type)
    model = trainer(X_train, y_train, X_val, y_val, n_classes)
    y_pred = model.predict(X_test)

    from sklearn.metrics import balanced_accuracy_score, accuracy_score
    ba = balanced_accuracy_score(y_test, y_pred)
    oa = accuracy_score(y_test, y_pred)

    print(f"  Country prediction: BA={ba:.4f}, OA={oa:.4f}")
    print(f"  ({n_classes} countries, {len(X_test)} test samples)")

    # Per-country accuracy
    per_country = {}
    for i, country in enumerate(le.classes_):
        mask = y_test == i
        if mask.sum() > 0:
            acc = (y_pred[mask] == y_test[mask]).mean()
            per_country[country] = float(acc)

    return {
        'balanced_accuracy': float(ba),
        'overall_accuracy': float(oa),
        'n_countries': n_classes,
        'per_country_accuracy': per_country,
        'classifier': classifier_type,
    }


def main():
    parser = argparse.ArgumentParser(description='Exp 5: Feature Analysis')
    parser.add_argument('--tier', choices=['a', 'b', 'c'], default='b')
    parser.add_argument('--n-samples', type=int, default=5000,
                        help='Max samples for visualization')
    parser.add_argument('--feature-types', nargs='+', default=['ae', 's2'],
                        choices=['ae', 's2', 'fusion'])
    parser.add_argument('--skip-tsne', action='store_true')
    parser.add_argument('--skip-shap', action='store_true')
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
    tag = f" [{args.classifier}]" + (" [no-coords]" if args.no_coords else "")

    all_results = {}

    for ft in args.feature_types:
        try:
            df, feature_cols, class_names = load_eu_features(ft, args.tier, countries=countries)
        except FileNotFoundError as e:
            print(f"Skipping {ft}: {e}")
            continue

        if args.no_coords:
            feature_cols = filter_coords(feature_cols)

        ft_results = {}

        if not args.skip_tsne:
            ft_results['tsne_umap'] = analysis_tsne_umap(
                df, feature_cols, args.tier, ft, args.n_samples)

        ft_results['cka'] = analysis_cka(
            df, feature_cols, args.tier, ft, args.n_samples)

        if not args.skip_shap:
            shap_result = analysis_shap(
                df, feature_cols, class_names, args.tier, ft,
                classifier_type=args.classifier,
                max_samples=max_samples)
            if shap_result:
                ft_results['shap'] = shap_result

        ft_results['country_prediction'] = analysis_country_prediction(
            df, feature_cols, ft, args.n_samples,
            classifier_type=args.classifier,
            max_samples=max_samples)

        all_results[ft] = ft_results

    # CKA 对比表
    if len(all_results) > 1:
        print(f"\n--- CKA Comparison{tag} ---")
        csv_rows = []
        for ft, res in all_results.items():
            cka = res.get('cka', {})
            print(f"  {ft:8s}: crop={cka.get('cka_crop', 0):.4f}, "
                  f"country={cka.get('cka_country', 0):.4f}, "
                  f"ratio={cka.get('ratio', 0):.2f}")
            csv_rows.append({
                'feature_type': ft,
                'cka_crop': cka.get('cka_crop'),
                'cka_country': cka.get('cka_country'),
                'ratio': cka.get('ratio'),
            })

        save_results_csv('exp5_feature_analysis', csv_rows,
                         filename=f'cka_comparison_tier_{args.tier}{suffix}.csv')

    exp_name = 'exp5_feature_analysis'
    save_experiment_results(exp_name, all_results,
                            sub_name=f'tier_{args.tier}{suffix}')

    print(f"\nExp 5 complete!")


if __name__ == '__main__':
    main()
