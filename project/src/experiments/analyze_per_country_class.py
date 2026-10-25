"""
Exp 0 补充分析：按国家 × 作物类型分解 F1 结果，生成热力图。

对每个国家 × 每种 feature_type，做单次 stratified split，
记录 per-class F1 并可视化。

Usage:
    python -m src.experiments.analyze_per_country_class
"""
import sys
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from src.experiments.experiment_utils import (
    load_eu_features, prepare_xy, train_lgbm, RANDOM_STATE, COL_COUNTRY,
)
from sklearn.metrics import f1_score, classification_report


def get_per_class_f1(X_train, y_train, X_val, y_val, X_test, y_test,
                     n_classes, class_names):
    """训练模型并返回 per-class F1。"""
    model = train_lgbm(X_train, y_train, X_val, y_val, n_classes)
    y_pred = model.predict(X_test)

    report = classification_report(
        y_test, y_pred, output_dict=True, zero_division=0)

    per_class = {}
    for cls_idx in range(n_classes):
        cls_name = class_names[cls_idx] if cls_idx < len(class_names) else str(cls_idx)
        key = str(cls_idx)
        if key in report:
            per_class[cls_name] = {
                'f1': report[key]['f1-score'],
                'precision': report[key]['precision'],
                'recall': report[key]['recall'],
                'support': report[key]['support'],
            }
    return per_class


def run_analysis(feature_type='ae', tier='b'):
    """对所有国家运行 per-class 分析。"""
    print(f"\n{'=' * 60}")
    print(f"PER-CLASS ANALYSIS — {feature_type.upper()} Tier {tier.upper()}")
    print(f"{'=' * 60}")

    df, feature_cols, class_names = load_eu_features(feature_type, tier)
    countries = sorted(df[COL_COUNTRY].unique())
    label_col = f'label_tier_{tier}'
    encoded_col = f'{label_col}_encoded'

    all_results = {}  # country -> {class_name: f1}

    for country in countries:
        df_c = df[df[COL_COUNTRY] == country].reset_index(drop=True)
        n_samples = len(df_c)

        # 过滤稀有类（<10 样本）
        from collections import Counter
        label_counts = Counter(df_c[encoded_col].values)
        valid_classes = {cls for cls, cnt in label_counts.items() if cnt >= 10}
        df_c = df_c[df_c[encoded_col].isin(valid_classes)].reset_index(drop=True)

        if len(df_c) < 50:
            print(f"  {country}: skipped (too few samples)")
            continue

        # 本国出现的类别名
        local_labels = sorted(df_c[encoded_col].unique())
        local_class_names = [class_names[i] for i in local_labels
                             if i < len(class_names)]

        X_c = df_c[feature_cols].fillna(0).values
        y_c = df_c[encoded_col].values

        # 重新编码为连续 0-based（LightGBM 需要）
        label_map = {old: new for new, old in enumerate(sorted(set(y_c)))}
        y_c_mapped = np.array([label_map[v] for v in y_c])
        n_local = len(label_map)

        # Stratified split: 70% train, 15% val, 15% test
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3,
                                     random_state=RANDOM_STATE)
        train_idx, temp_idx = next(sss.split(X_c, y_c_mapped))

        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5,
                                      random_state=RANDOM_STATE)
        val_sub, test_sub = next(sss2.split(X_c[temp_idx], y_c_mapped[temp_idx]))
        val_idx = temp_idx[val_sub]
        test_idx = temp_idx[test_sub]

        X_train, y_train = X_c[train_idx], y_c_mapped[train_idx]
        X_val, y_val = X_c[val_idx], y_c_mapped[val_idx]
        X_test, y_test = X_c[test_idx], y_c_mapped[test_idx]

        per_class = get_per_class_f1(
            X_train, y_train, X_val, y_val, X_test, y_test,
            n_local, local_class_names)

        all_results[country] = per_class

        top3 = sorted(per_class.items(), key=lambda x: x[1]['f1'], reverse=True)[:3]
        bot3 = sorted(per_class.items(), key=lambda x: x[1]['f1'])[:3]
        mean_f1 = np.mean([v['f1'] for v in per_class.values()])
        print(f"  {country}: {n_local} classes, mean F1={mean_f1:.3f}")

    return all_results, class_names


def build_heatmap_data(all_results, class_names):
    """构建 国家 × 类别 的 F1 矩阵。"""
    countries = sorted(all_results.keys())
    # 收集所有出现过的类别
    all_classes = set()
    for cr in all_results.values():
        all_classes.update(cr.keys())
    # 按 class_names 顺序排列
    ordered_classes = [c for c in class_names if c in all_classes]
    # 加上 class_names 里没有的
    for c in sorted(all_classes):
        if c not in ordered_classes:
            ordered_classes.append(c)

    matrix = np.full((len(countries), len(ordered_classes)), np.nan)
    support_matrix = np.full((len(countries), len(ordered_classes)), 0)

    for i, country in enumerate(countries):
        for j, cls in enumerate(ordered_classes):
            if cls in all_results[country]:
                matrix[i, j] = all_results[country][cls]['f1']
                support_matrix[i, j] = all_results[country][cls]['support']

    return matrix, support_matrix, countries, ordered_classes


def plot_heatmap(matrix, support_matrix, countries, classes, feature_type,
                 out_path):
    """绘制 F1 热力图。"""
    fig, ax = plt.subplots(figsize=(max(14, len(classes) * 0.9),
                                     max(8, len(countries) * 0.5)))

    # 自定义颜色映射：NaN=灰色，0=红，0.5=黄，1=深绿
    cmap = plt.cm.RdYlGn.copy()
    cmap.set_bad(color='lightgray')

    masked = np.ma.masked_invalid(matrix)
    im = ax.imshow(masked, cmap=cmap, aspect='auto', vmin=0, vmax=1)

    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(len(countries)))
    ax.set_yticklabels(countries, fontsize=10)

    # 在每个格子中标注 F1 值
    for i in range(len(countries)):
        for j in range(len(classes)):
            if not np.isnan(matrix[i, j]):
                f1_val = matrix[i, j]
                sup = support_matrix[i, j]
                color = 'white' if f1_val < 0.3 or f1_val > 0.8 else 'black'
                ax.text(j, i, f'{f1_val:.2f}', ha='center', va='center',
                        fontsize=7, color=color, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('F1 Score', fontsize=11)

    ax.set_title(f'Per-Country Per-Class F1 Score — {feature_type.upper()} (Tier B)',
                 fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Crop Type (Tier B)', fontsize=11)
    ax.set_ylabel('Country', fontsize=11)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nHeatmap saved to {out_path}")
    plt.close()


def plot_comparison_heatmap(results_dict, out_path):
    """绘制 AE vs S2 的 F1 差值热力图。"""
    ae_res = results_dict.get('ae', {})
    s2_res = results_dict.get('s2', {})
    if not ae_res or not s2_res:
        return

    countries = sorted(set(ae_res.keys()) & set(s2_res.keys()))
    all_classes = set()
    for cr in list(ae_res.values()) + list(s2_res.values()):
        all_classes.update(cr.keys())
    ordered_classes = sorted(all_classes)

    diff_matrix = np.full((len(countries), len(ordered_classes)), np.nan)

    for i, country in enumerate(countries):
        for j, cls in enumerate(ordered_classes):
            ae_f1 = ae_res[country].get(cls, {}).get('f1', np.nan)
            s2_f1 = s2_res[country].get(cls, {}).get('f1', np.nan)
            if not np.isnan(ae_f1) and not np.isnan(s2_f1):
                diff_matrix[i, j] = ae_f1 - s2_f1

    fig, ax = plt.subplots(figsize=(max(14, len(ordered_classes) * 0.9),
                                     max(8, len(countries) * 0.5)))

    cmap = plt.cm.RdBu.copy()
    cmap.set_bad(color='lightgray')

    masked = np.ma.masked_invalid(diff_matrix)
    vmax = max(0.3, np.nanmax(np.abs(diff_matrix)))
    im = ax.imshow(masked, cmap=cmap, aspect='auto', vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(ordered_classes)))
    ax.set_xticklabels(ordered_classes, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(len(countries)))
    ax.set_yticklabels(countries, fontsize=10)

    for i in range(len(countries)):
        for j in range(len(ordered_classes)):
            if not np.isnan(diff_matrix[i, j]):
                val = diff_matrix[i, j]
                color = 'white' if abs(val) > vmax * 0.6 else 'black'
                ax.text(j, i, f'{val:+.2f}', ha='center', va='center',
                        fontsize=7, color=color, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('AE F1 - S2 F1 (blue=AE better, red=S2 better)', fontsize=10)

    ax.set_title('AE vs S2: Per-Class F1 Difference (Tier B)',
                 fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Crop Type (Tier B)', fontsize=11)
    ax.set_ylabel('Country', fontsize=11)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nDifference heatmap saved to {out_path}")
    plt.close()


def main():
    from config.config import EXPERIMENTS_DIR

    out_dir = os.path.join(EXPERIMENTS_DIR, 'exp0_per_country_cv')
    os.makedirs(out_dir, exist_ok=True)

    results_dict = {}

    for ft in ['ae', 's2', 'fusion']:
        try:
            results, class_names = run_analysis(ft, 'b')
            results_dict[ft] = results

            matrix, support, countries, classes = build_heatmap_data(
                results, class_names)
            plot_heatmap(matrix, support, countries, classes, ft,
                         os.path.join(out_dir, f'heatmap_{ft}_tier_b.png'))

            # 保存 per-class CSV
            rows = []
            for country in countries:
                for cls in classes:
                    if cls in results[country]:
                        r = results[country][cls]
                        rows.append({
                            'country': country,
                            'class': cls,
                            'feature_type': ft,
                            'f1': r['f1'],
                            'precision': r['precision'],
                            'recall': r['recall'],
                            'support': r['support'],
                        })
            pd.DataFrame(rows).to_csv(
                os.path.join(out_dir, f'per_class_{ft}_tier_b.csv'),
                index=False)

        except Exception as e:
            print(f"Error for {ft}: {e}")
            import traceback
            traceback.print_exc()

    # AE vs S2 差值热力图
    if 'ae' in results_dict and 's2' in results_dict:
        plot_comparison_heatmap(
            results_dict,
            os.path.join(out_dir, 'heatmap_ae_vs_s2_diff.png'))

    print("\nPer-class analysis complete!")


if __name__ == '__main__':
    main()
