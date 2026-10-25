"""
构建修复后的 11 国数据集 dataset_v2.parquet（本地盘）。

修复内容：
1. Tier-B 映射 v2（common.TIER_B_MAPPING_V2），并排除休耕地；
2. BE_WAL 的 AE 最后一批 id 列名为 parcel_idx（值格式与 parcel_id 相同），合并时补回；
3. LV 的 S2 第 0008 批为另一版本导出代码生成（二次汇总的嵌套列名），无法与其它批次对齐，
   这些地块标记为 has_s2=False，不做近似填补。

输入（先拷贝到 DATA_DIR）：parcels_eu.parquet、ae/embeddings_{REGION}.parquet、s2/s2_{REGION}.parquet
"""
import os

import pandas as pd

from common import DATA_DIR, DATASET_V2, COUNTRIES, REGIONS, EXCLUDE_V2, tier_b_v2


def load_region(path, prefix, cols):
    d = pd.read_parquet(path)
    if 'parcel_idx' in d.columns:
        d['parcel_id'] = d['parcel_id'].fillna(d['parcel_idx'])
    d = d[['parcel_id'] + cols].dropna(subset=['parcel_id'])
    complete = d[cols].notna().all(axis=1)
    n_bad = int((~complete).sum())
    d = d[complete].drop_duplicates('parcel_id')
    return d.rename(columns={c: prefix + c for c in cols}), n_bad


def main():
    p = pd.read_parquet(os.path.join(DATA_DIR, 'parcels_eu.parquet'))
    p = p[p['country'].isin(COUNTRIES) & ~p['parcel_id'].str.startswith('DE_BB_')]
    n0 = len(p)
    # 排除非作物类别，以及没有任何作物信息的地块（SK 有 825 个，EC_hcat_n 为空）
    p = p[~p['EC_hcat_n'].isin(EXCLUDE_V2) & p['EC_hcat_n'].notna()].copy()
    print(f'parcels: {n0:,} → {len(p):,} after excluding {sorted(EXCLUDE_V2)} and missing EC_hcat_n')
    p['label_v1'] = p['label_tier_b']
    p['label'] = p['EC_hcat_n'].map(tier_b_v2)
    moved = p[p['label'] != p['label_v1']]
    print(f'relabelled: {len(moved):,}')
    print(moved.groupby(['country', 'label_v1', 'label']).size().loc[lambda s: s >= 100].to_string())

    # 标准列清单：AE 为 A00–A63 × mean/stdDev；S2 以格式正常的 AT 文件为准
    ae_cols = [f'A{i:02d}_{s}' for i in range(64) for s in ('mean', 'stdDev')]
    s2_cols = [c for c in pd.read_parquet(os.path.join(DATA_DIR, 's2', 's2_AT.parquet')).columns
               if c != 'parcel_id']
    ae, s2 = [], []
    for r in REGIONS:
        a, bad_a = load_region(os.path.join(DATA_DIR, 'ae', f'embeddings_{r}.parquet'), 'ae_', ae_cols)
        s, bad_s = load_region(os.path.join(DATA_DIR, 's2', f's2_{r}.parquet'), 's2_', s2_cols)
        print(f'{r:7s} AE {len(a):6,} (incomplete {bad_a})  S2 {len(s):6,} (incomplete {bad_s})')
        ae.append(a)
        s2.append(s)
    ae = pd.concat(ae, ignore_index=True)
    s2 = pd.concat(s2, ignore_index=True)
    assert ae.shape[1] - 1 == 128, ae.shape
    assert s2.shape[1] - 1 == 168, s2.shape

    meta = ['parcel_id', 'country', 'longitude', 'latitude', 'area_ha', 'log_area',
            'EC_hcat_n', 'label_v1', 'label']
    df = p[meta].merge(ae, on='parcel_id', how='left').merge(s2, on='parcel_id', how='left')
    df['has_ae'] = df['ae_A00_mean'].notna()
    df['has_s2'] = df['s2_B11_Q1_median_mean'].notna()
    print(pd.crosstab(df['country'], [df['has_ae'], df['has_s2']]).to_string())
    both = df[df['has_ae'] & df['has_s2']]
    print(f'total {len(df):,}; AE {df.has_ae.sum():,}; S2 {df.has_s2.sum():,}; both {len(both):,}')
    print(pd.crosstab(both['country'], both['label']).to_string())
    df.to_parquet(DATASET_V2, index=False)
    print('saved', DATASET_V2)


if __name__ == '__main__':
    main()
