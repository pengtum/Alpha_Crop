"""
分析 1: 在 shared classes 上重算 pairwise transfer BA
分析 2: Transfer determinants pair-level regression

输出到 results/experiments/exp2b_country_transfer/
"""
import os, sys, json
import numpy as np
import pandas as pd
from collections import Counter
from scipy.spatial.distance import cdist

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams.update({
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'font.family': 'Arial',
    'font.size': 10, 'figure.dpi': 150, 'savefig.dpi': 150,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.1,
})

OUT_DIR = 'results/experiments/exp2b_country_transfer'
C2021 = ['AT','BE','DE','EE','LT','LV','PT','RO','SE','SI','SK']

# 气候带
COUNTRY_ZONE = {
    'AT':'continental','BE':'atlantic','DE':'atlantic',
    'EE':'nordic_baltic','LT':'nordic_baltic','LV':'nordic_baltic',
    'PT':'mediterranean','RO':'southeast','SE':'nordic_baltic',
    'SI':'mediterranean','SK':'continental',
}

# 近似国家中心坐标 (lon, lat)
COUNTRY_COORDS = {
    'AT': (13.3, 47.5), 'BE': (4.5, 50.8), 'DE': (10.0, 51.2),
    'EE': (25.0, 58.6), 'LT': (24.0, 55.2), 'LV': (24.6, 56.9),
    'PT': (-8.2, 39.4), 'RO': (25.0, 45.9), 'SE': (15.4, 62.2),
    'SI': (14.5, 46.2), 'SK': (19.7, 48.7),
}


def load_data():
    """加载数据集，获取每国类别信息和样本量。"""
    feat_dir = os.path.join('data', 'eu_processed', 'features', 'ae')
    dfs = [pd.read_parquet(os.path.join(feat_dir, f'features_{s}.parquet'),
           columns=['country', 'parcel_id', 'label_tier_b', 'label_tier_b_encoded'])
           for s in ['train', 'val', 'test']]
    df = pd.concat(dfs, ignore_index=True)
    df = df[df.country.isin(C2021)]
    df = df[~df.parcel_id.str.startswith('DE_BB_')]
    return df


def compute_class_overlap(df):
    """计算每对国家的类别重叠。"""
    presence = {}
    sample_counts = {}
    for c in C2021:
        sub = df[df.country == c]
        presence[c] = set(sub.label_tier_b.unique())
        sample_counts[c] = len(sub)

    overlap_data = []
    for c1 in C2021:
        for c2 in C2021:
            if c1 == c2:
                continue
            shared = presence[c1] & presence[c2]
            only_c1 = presence[c1] - presence[c2]
            jaccard = len(shared) / len(presence[c1] | presence[c2])
            overlap_data.append({
                'train_country': c1,
                'test_country': c2,
                'n_shared_classes': len(shared),
                'n_train_classes': len(presence[c1]),
                'n_test_classes': len(presence[c2]),
                'jaccard_overlap': jaccard,
                'shared_classes': ','.join(sorted(shared)),
                'train_only_classes': ','.join(sorted(only_c1)),
            })

    return pd.DataFrame(overlap_data), presence, sample_counts


def recompute_transfer_shared_classes(transfer_csv, per_sample_csv, presence):
    """在 shared classes 上重算 transfer BA。

    如果 per_sample CSV 不存在，从 transfer CSV 的原始 BA 出发，
    无法精确重算，返回 None。
    """
    # per_sample CSV 在之前被跳过（太大），所以我们用另一种方法：
    # 从 Exp 2b 的原始 JSON 中提取 per-class metrics，
    # 只计算 shared classes 的 per-class recall 均值
    return None


def analysis_1_shared_class_transfer():
    """分析 1: 基于类别重叠的 transfer 分析。"""
    print('=' * 60)
    print('Analysis 1: Class Overlap and Transfer Performance')
    print('=' * 60)

    df = load_data()
    overlap_df, presence, sample_counts = compute_class_overlap(df)

    # 合并 transfer results
    for ft in ['ae', 's2', 'fusion']:
        csv_path = os.path.join(OUT_DIR, f'transfer_{ft}_nocoords_2021_tier_b.csv')
        if not os.path.exists(csv_path):
            full_csv = os.path.join(OUT_DIR, f'transfer_{ft}_nocoords_tier_b.csv')
            if os.path.exists(full_csv):
                full = pd.read_csv(full_csv)
                sub = full[full.train_country.isin(C2021) & full.test_country.isin(C2021)]
                sub.to_csv(csv_path, index=False)
                print(f'  Filtered {ft}: {len(full)} -> {len(sub)} pairs')

        t = pd.read_csv(csv_path)
        ba_col = t[['train_country', 'test_country', 'balanced_accuracy']].copy()
        ba_col.rename(columns={'balanced_accuracy': f'ba_{ft}'}, inplace=True)
        overlap_df = overlap_df.merge(ba_col, on=['train_country', 'test_country'],
                                       how='left')

    # 保存
    overlap_df.to_csv(os.path.join(OUT_DIR, 'class_overlap_analysis_2021.csv'),
                      index=False)
    print(f'\nSaved class_overlap_analysis_2021.csv ({len(overlap_df)} pairs)')

    # 统计：按 class overlap 分组的平均 BA
    print('\n--- Mean BA by class overlap (Jaccard) ---')
    for ft in ['ae', 's2', 'fusion']:
        col = f'ba_{ft}'
        if col not in overlap_df.columns:
            continue
        bins = [0, 0.3, 0.5, 0.7, 0.9, 1.01]
        labels = ['<0.3', '0.3-0.5', '0.5-0.7', '0.7-0.9', '0.9-1.0']
        overlap_df['jaccard_bin'] = pd.cut(overlap_df['jaccard_overlap'],
                                           bins=bins, labels=labels)
        print(f'\n  {ft.upper()}:')
        for b_label in labels:
            mask = overlap_df['jaccard_bin'] == b_label
            vals = overlap_df.loc[mask, col].dropna()
            if len(vals) > 0:
                print(f'    Jaccard {b_label}: BA={float(vals.mean()):.4f} +/- {float(vals.std()):.4f} '
                      f'(n={len(vals)})')

    # 图: BA vs Jaccard scatter
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    colors = {'ae': '#2196F3', 's2': '#FF9800', 'fusion': '#4CAF50'}
    labels = {'ae': 'AlphaEarth', 's2': 'Sentinel-2', 'fusion': 'Fusion'}

    for idx, ft in enumerate(['ae', 's2', 'fusion']):
        ax = axes[idx]
        col = f'ba_{ft}'
        if col not in overlap_df.columns:
            continue
        ax.scatter(overlap_df['jaccard_overlap'], overlap_df[col],
                   alpha=0.6, s=30, color=colors[ft])
        # Correlation
        valid = overlap_df[[col, 'jaccard_overlap']].dropna()
        r = np.corrcoef(valid['jaccard_overlap'], valid[col])[0, 1]
        ax.text(0.05, 0.95, f'r = {r:.3f}', transform=ax.transAxes,
                fontsize=11, va='top', fontweight='bold')
        ax.set_xlabel('Class Overlap (Jaccard)')
        ax.set_ylabel('Balanced Accuracy')
        ax.set_title(labels[ft])
        ax.grid(alpha=0.3, linestyle='--')

    fig.suptitle('Transfer BA vs. Class-Vocabulary Overlap', fontsize=13, y=1.02)
    fig.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(os.path.join(OUT_DIR, f'ba_vs_class_overlap.{ext}'))
    plt.close(fig)
    print('\nSaved ba_vs_class_overlap.png/pdf')

    return overlap_df


def analysis_2_transfer_determinants(overlap_df):
    """分析 2: Transfer determinants pair-level regression。"""
    print('\n' + '=' * 60)
    print('Analysis 2: Transfer Determinants Regression')
    print('=' * 60)

    df_data = load_data()
    _, presence, sample_counts = compute_class_overlap(df_data)

    # 构建特征矩阵
    rows = []
    for _, row in overlap_df.iterrows():
        c1, c2 = row['train_country'], row['test_country']

        # Geographic distance (Haversine approx in degrees)
        lon1, lat1 = COUNTRY_COORDS[c1]
        lon2, lat2 = COUNTRY_COORDS[c2]
        geo_dist = np.sqrt((lon1 - lon2)**2 + (lat1 - lat2)**2)

        # Climate zone match
        same_zone = int(COUNTRY_ZONE[c1] == COUNTRY_ZONE[c2])

        # Sample sizes
        n_train = sample_counts.get(c1, 0)
        n_test = sample_counts.get(c2, 0)

        rows.append({
            'train_country': c1,
            'test_country': c2,
            'geo_distance': geo_dist,
            'same_climate_zone': same_zone,
            'jaccard_overlap': row['jaccard_overlap'],
            'n_shared_classes': row['n_shared_classes'],
            'n_train_samples': n_train,
            'n_test_samples': n_test,
            'log_n_train': np.log10(n_train) if n_train > 0 else 0,
            'ba_ae': row.get('ba_ae', np.nan),
            'ba_s2': row.get('ba_s2', np.nan),
            'ba_fusion': row.get('ba_fusion', np.nan),
        })

    reg_df = pd.DataFrame(rows)
    reg_df.to_csv(os.path.join(OUT_DIR, 'transfer_determinants_2021.csv'),
                  index=False)
    print(f'Saved transfer_determinants_2021.csv ({len(reg_df)} pairs)')

    # OLS regression for each feature set
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler

    features = ['geo_distance', 'same_climate_zone', 'jaccard_overlap',
                'log_n_train', 'n_shared_classes']

    print('\n--- OLS Regression: BA ~ determinants ---')
    for ft in ['ae', 's2', 'fusion']:
        col = f'ba_{ft}'
        if col not in reg_df.columns:
            continue
        valid = reg_df[features + [col]].dropna()
        X = valid[features].values
        y = valid[col].values

        # Standardize for coefficient comparison
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = LinearRegression()
        model.fit(X_scaled, y)
        r2 = model.score(X_scaled, y)

        print(f'\n  {ft.upper()} (R2 = {r2:.4f}):')
        print(f'    {"Feature":25s} {"Coeff (std)":>12s} {"Direction":>10s}')
        for fname, coef in zip(features, model.coef_):
            direction = '+' if coef > 0 else '-'
            print(f'    {fname:25s} {coef:>12.4f} {direction:>10s}')
        print(f'    {"Intercept":25s} {model.intercept_:>12.4f}')

    # 图: coefficient comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    feat_labels = ['Geographic\nDistance', 'Same Climate\nZone',
                   'Class Overlap\n(Jaccard)', 'Training Set\nSize (log)',
                   'Shared\nClasses']
    x = np.arange(len(features))
    w = 0.25
    colors_ft = {'ae': '#2196F3', 's2': '#FF9800', 'fusion': '#4CAF50'}

    for i, ft in enumerate(['ae', 's2', 'fusion']):
        col = f'ba_{ft}'
        valid = reg_df[features + [col]].dropna()
        X = valid[features].values
        y = valid[col].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = LinearRegression()
        model.fit(X_scaled, y)
        ax.bar(x + (i - 1) * w, model.coef_, w,
               label={'ae': 'AlphaEarth', 's2': 'Sentinel-2', 'fusion': 'Fusion'}[ft],
               color=colors_ft[ft], edgecolor='white', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(feat_labels, fontsize=9)
    ax.set_ylabel('Standardized Coefficient')
    ax.set_title('Transfer Determinants: Standardized Regression Coefficients')
    ax.legend(frameon=True, fancybox=False, edgecolor='gray')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.axhline(y=0, color='black', linewidth=0.5)
    fig.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(os.path.join(OUT_DIR, f'transfer_determinants_coefficients.{ext}'))
    plt.close(fig)
    print('\nSaved transfer_determinants_coefficients.png/pdf')

    # 图: partial dependence scatter (top 2 predictors)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    for col_idx, ft in enumerate(['ae', 's2', 'fusion']):
        col = f'ba_{ft}'
        if col not in reg_df.columns:
            continue

        # Row 0: BA vs geographic distance, colored by same_zone
        ax = axes[0, col_idx]
        mask_same = reg_df['same_climate_zone'] == 1
        ax.scatter(reg_df.loc[mask_same, 'geo_distance'],
                   reg_df.loc[mask_same, col],
                   alpha=0.6, s=25, color='#4CAF50', label='Same zone')
        ax.scatter(reg_df.loc[~mask_same, 'geo_distance'],
                   reg_df.loc[~mask_same, col],
                   alpha=0.6, s=25, color='#E53935', label='Diff zone')
        ax.set_xlabel('Geographic Distance')
        ax.set_ylabel('Balanced Accuracy')
        ft_label = {'ae': 'AlphaEarth', 's2': 'Sentinel-2', 'fusion': 'Fusion'}[ft]
        ax.set_title(f'{ft_label}')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, linestyle='--')

        # Row 1: BA vs jaccard overlap, colored by n_train
        ax = axes[1, col_idx]
        sc = ax.scatter(reg_df['jaccard_overlap'], reg_df[col],
                        c=reg_df['log_n_train'], cmap='viridis',
                        alpha=0.7, s=30, edgecolors='white', linewidth=0.3)
        ax.set_xlabel('Class Overlap (Jaccard)')
        ax.set_ylabel('Balanced Accuracy')
        ax.set_title(f'{ft_label}')
        ax.grid(alpha=0.3, linestyle='--')
        plt.colorbar(sc, ax=ax, label='log10(n_train)')

    fig.suptitle('Transfer BA vs. Geographic Distance and Class Overlap',
                 fontsize=13, y=1.02)
    fig.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(os.path.join(OUT_DIR, f'transfer_determinants_scatter.{ext}'))
    plt.close(fig)
    print('Saved transfer_determinants_scatter.png/pdf')

    return reg_df


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    overlap_df = analysis_1_shared_class_transfer()
    reg_df = analysis_2_transfer_determinants(overlap_df)
    print('\n' + '=' * 60)
    print('All analyses complete!')
    print('=' * 60)
