"""
Combine AlphaEarth embeddings with auxiliary parcel features,
apply preprocessing, and save train/val/test feature matrices.

支持两种模式：
  - Belgian: 原始比利时流水线（centroid/polygon）
  - EU:      泛欧洲流水线

Usage:
    # 比利时模式
    python -m src.features.build_features [--mode centroid|polygon]

    # 泛欧洲模式
    python -m src.features.build_features --mode eu [--add-country]
"""
import sys
import os
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from config.config import (
    PARCELS_FILTERED_PATH, EMBEDDINGS_CENTROID_PATH,
    EMBEDDINGS_POLYGON_PATH, FEATURES_TRAIN_PATH,
    FEATURES_VAL_PATH, FEATURES_TEST_PATH,
    MODELS_DIR, AE_BANDS,
    EU_PARCELS_PATH, EU_EMBEDDINGS_DIR, EU_FEATURES_DIR,
    EU_S2_EMBEDDINGS_DIR,
    EU_FEATURES_TRAIN_PATH, EU_FEATURES_VAL_PATH, EU_FEATURES_TEST_PATH,
    COL_PARCEL_ID, COL_COUNTRY, COL_AREA, COL_LON, COL_LAT,
    COL_HCAT3_NAME,
)


# ============================================================
# 泛欧洲模式
# ============================================================

def load_and_join_eu():
    """Load EU parcels and embeddings, join them by parcel_id."""
    print("=" * 60)
    print("LOADING AND JOINING EU DATA")
    print("=" * 60)

    parcels = pd.read_parquet(EU_PARCELS_PATH)
    print(f"Parcels: {len(parcels)}")

    # 加载合并的嵌入
    emb_path = os.path.join(EU_EMBEDDINGS_DIR, 'embeddings_eu.parquet')
    if not os.path.exists(emb_path):
        print(f"ERROR: {emb_path} not found. Run merge_embeddings --mode eu --merge-all first.")
        sys.exit(1)

    emb = pd.read_parquet(emb_path)
    print(f"Embeddings: {len(emb)}")

    # Join by parcel_id
    df = parcels.merge(emb, on=COL_PARCEL_ID, how='inner',
                       suffixes=('', '_emb'))
    print(f"Joined: {len(df)} (join rate: {len(df)/len(parcels)*100:.1f}%)")

    # 识别嵌入列（兼容 _std 和 _stdDev 两种命名）
    mean_cols = [f'{b}_mean' for b in AE_BANDS]
    std_cols = [f'{b}_std' for b in AE_BANDS]
    std_cols_alt = [f'{b}_stdDev' for b in AE_BANDS]
    available_mean = [c for c in mean_cols if c in df.columns]
    available_std = [c for c in std_cols if c in df.columns]
    if not available_std:
        available_std = [c for c in std_cols_alt if c in df.columns]
        if available_std:
            print("  (using _stdDev naming convention)")

    print(f"Embedding mean columns: {len(available_mean)}")
    print(f"Embedding std columns: {len(available_std)}")

    # 删除全空嵌入的行
    if available_mean:
        null_mask = df[available_mean].isnull().all(axis=1)
        n_null = null_mask.sum()
        if n_null > 0:
            print(f"Dropping {n_null} rows with null embeddings "
                  f"({n_null/len(df)*100:.2f}%)")
            df = df[~null_mask].reset_index(drop=True)

    print(f"Final dataset: {len(df)} rows")
    return df, available_mean, available_std


def build_feature_matrix_eu(df, emb_mean_cols, emb_std_cols,
                             add_country=False):
    """Build EU feature matrix."""
    print("\n" + "=" * 60)
    print("BUILDING EU FEATURE MATRIX")
    print("=" * 60)

    # Embedding features
    feature_cols = list(emb_mean_cols) + list(emb_std_cols)

    # Auxiliary numeric features
    aux_numeric = ['log_area', COL_LON, COL_LAT]
    feature_cols.extend(aux_numeric)

    # Pixel count (if available)
    if 'pixel_count' in df.columns:
        feature_cols.append('pixel_count')

    # 可选：国家独热编码
    if add_country:
        country_dummies = pd.get_dummies(df[COL_COUNTRY], prefix='ctry')
        for col in country_dummies.columns:
            df[col] = country_dummies[col].values
            feature_cols.append(col)
        print(f"  Country one-hot: {len(country_dummies.columns)} columns")

    print(f"Total features: {len(feature_cols)}")
    print(f"  Embedding mean: {len(emb_mean_cols)}")
    print(f"  Embedding std: {len(emb_std_cols)}")
    print(f"  Auxiliary numeric: {len(aux_numeric)}")

    return feature_cols


def split_and_save_eu(df, feature_cols):
    """Split EU data and save."""
    print("\n" + "=" * 60)
    print("SPLITTING AND SAVING EU DATA")
    print("=" * 60)

    # Encode labels
    label_encoders = {}
    for tier in ['label_tier_b', 'label_tier_a', 'label_tier_c']:
        le = LabelEncoder()
        df[f'{tier}_encoded'] = le.fit_transform(df[tier])
        label_encoders[tier] = le
        print(f"{tier}: {len(le.classes_)} classes")

    # Meta columns to keep
    meta_cols = [
        COL_PARCEL_ID, COL_COUNTRY, COL_AREA,
        COL_HCAT3_NAME, 'split', 'size_class',
        'label_tier_b', 'label_tier_a', 'label_tier_c',
        'label_tier_b_encoded', 'label_tier_a_encoded',
        'label_tier_c_encoded',
    ]
    save_cols = feature_cols + [c for c in meta_cols if c in df.columns]

    # Save splits
    os.makedirs(EU_FEATURES_DIR, exist_ok=True)
    for split_name, out_path in [('train', EU_FEATURES_TRAIN_PATH),
                                  ('val', EU_FEATURES_VAL_PATH),
                                  ('test', EU_FEATURES_TEST_PATH)]:
        split_df = df[df['split'] == split_name][save_cols].reset_index(
            drop=True)
        split_df.to_parquet(out_path, index=False)
        n_countries = split_df[COL_COUNTRY].nunique() if COL_COUNTRY in split_df.columns else 0
        print(f"\n{split_name}: {len(split_df)} parcels, "
              f"{n_countries} countries -> {out_path}")
        print(f"  File size: {os.path.getsize(out_path) / 1e6:.1f} MB")

    # Fit scaler on training data
    train_df = df[df['split'] == 'train']
    scaler = StandardScaler()
    numeric_feature_cols = [c for c in feature_cols
                            if not c.startswith('ctry_')]
    scaler.fit(train_df[numeric_feature_cols].fillna(0))

    # Save artifacts
    eu_models_dir = os.path.join(MODELS_DIR, 'eu')
    os.makedirs(eu_models_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(eu_models_dir, 'scaler.joblib'))
    joblib.dump(label_encoders, os.path.join(eu_models_dir,
                                              'label_encoders.joblib'))
    joblib.dump(feature_cols, os.path.join(eu_models_dir,
                                            'feature_cols.joblib'))
    print(f"\nSaved scaler and label encoders to {eu_models_dir}")


# ============================================================
# S2 / Fusion 模式
# ============================================================

def load_and_join_s2_eu():
    """Load EU parcels and S2 features, join by parcel_id."""
    print("=" * 60)
    print("LOADING AND JOINING EU S2 DATA")
    print("=" * 60)

    parcels = pd.read_parquet(EU_PARCELS_PATH)
    print(f"Parcels: {len(parcels)}")

    s2_path = os.path.join(EU_S2_EMBEDDINGS_DIR, 's2_embeddings_eu.parquet')
    if not os.path.exists(s2_path):
        print(f"ERROR: {s2_path} not found. "
              "Run merge_embeddings --mode eu --source s2 --merge-all first.")
        sys.exit(1)

    s2 = pd.read_parquet(s2_path)
    print(f"S2 features: {len(s2)}")

    df = parcels.merge(s2, on=COL_PARCEL_ID, how='inner',
                       suffixes=('', '_s2'))
    print(f"Joined: {len(df)} (join rate: {len(df)/len(parcels)*100:.1f}%)")

    # 识别 S2 特征列 (排除 parcel_id 和其他元数据列)
    meta_cols_set = {COL_PARCEL_ID, COL_COUNTRY, COL_AREA, COL_LON, COL_LAT,
                     COL_HCAT3_NAME, 'split', 'size_class', 'log_area',
                     'label_tier_b', 'label_tier_a', 'label_tier_c',
                     'label_tier_b_encoded', 'label_tier_a_encoded',
                     'label_tier_c_encoded', 'EC_hcat_c', 'EC_trans_n'}
    s2_feature_cols = [c for c in s2.columns if c not in meta_cols_set]

    # 过滤掉高空值列（GEE reduceNeighborhood 产生的嵌套命名工件）
    null_rates = df[s2_feature_cols].isnull().mean()
    valid_s2_cols = [c for c in s2_feature_cols if null_rates[c] < 0.5]
    dropped_cols = len(s2_feature_cols) - len(valid_s2_cols)
    if dropped_cols > 0:
        print(f"Dropped {dropped_cols} S2 columns with >50% null values "
              f"(keeping {len(valid_s2_cols)})")
    s2_feature_cols = valid_s2_cols

    # 删除全空 S2 特征的行
    if s2_feature_cols:
        null_mask = df[s2_feature_cols].isnull().all(axis=1)
        n_null = null_mask.sum()
        if n_null > 0:
            print(f"Dropping {n_null} rows with null S2 features")
            df = df[~null_mask].reset_index(drop=True)

    print(f"S2 feature columns: {len(s2_feature_cols)}")
    print(f"Final dataset: {len(df)} rows")
    return df, s2_feature_cols


def load_and_join_fusion_eu():
    """Load EU parcels, AE embeddings and S2 features, join all."""
    print("=" * 60)
    print("LOADING AND JOINING EU FUSION DATA")
    print("=" * 60)

    parcels = pd.read_parquet(EU_PARCELS_PATH)
    print(f"Parcels: {len(parcels)}")

    # AE 嵌入
    ae_path = os.path.join(EU_EMBEDDINGS_DIR, 'embeddings_eu.parquet')
    if not os.path.exists(ae_path):
        print(f"ERROR: {ae_path} not found.")
        sys.exit(1)
    ae = pd.read_parquet(ae_path)
    print(f"AE embeddings: {len(ae)}")

    # S2 特征
    s2_path = os.path.join(EU_S2_EMBEDDINGS_DIR, 's2_embeddings_eu.parquet')
    if not os.path.exists(s2_path):
        print(f"ERROR: {s2_path} not found.")
        sys.exit(1)
    s2 = pd.read_parquet(s2_path)
    print(f"S2 features: {len(s2)}")

    # 三方 join
    df = parcels.merge(ae, on=COL_PARCEL_ID, how='inner',
                       suffixes=('', '_ae'))
    df = df.merge(s2, on=COL_PARCEL_ID, how='inner',
                  suffixes=('', '_s2'))
    print(f"Joined: {len(df)} (join rate: {len(df)/len(parcels)*100:.1f}%)")

    # 识别各特征列（兼容 _std 和 _stdDev 两种命名）
    ae_mean_cols = [f'{b}_mean' for b in AE_BANDS if f'{b}_mean' in df.columns]
    ae_std_cols = [f'{b}_std' for b in AE_BANDS if f'{b}_std' in df.columns]
    if not ae_std_cols:
        ae_std_cols = [f'{b}_stdDev' for b in AE_BANDS
                       if f'{b}_stdDev' in df.columns]
        if ae_std_cols:
            print("  (AE using _stdDev naming convention)")

    meta_cols_set = {COL_PARCEL_ID, COL_COUNTRY, COL_AREA, COL_LON, COL_LAT,
                     COL_HCAT3_NAME, 'split', 'size_class', 'log_area',
                     'label_tier_b', 'label_tier_a', 'label_tier_c',
                     'label_tier_b_encoded', 'label_tier_a_encoded',
                     'label_tier_c_encoded', 'EC_hcat_c', 'EC_trans_n'}
    ae_cols_set = set(ae_mean_cols + ae_std_cols)
    s2_feature_cols = [c for c in s2.columns
                       if c not in meta_cols_set and c not in ae_cols_set]

    # 过滤掉高空值 S2 列（GEE 嵌套命名工件）
    null_rates = df[s2_feature_cols].isnull().mean()
    valid_s2 = [c for c in s2_feature_cols if null_rates[c] < 0.5]
    dropped = len(s2_feature_cols) - len(valid_s2)
    if dropped > 0:
        print(f"Dropped {dropped} S2 columns with >50% null (keeping {len(valid_s2)})")
    s2_feature_cols = valid_s2

    # 删除全空行
    all_feat = ae_mean_cols + s2_feature_cols
    if all_feat:
        null_mask = df[all_feat].isnull().all(axis=1)
        n_null = null_mask.sum()
        if n_null > 0:
            print(f"Dropping {n_null} rows with all-null features")
            df = df[~null_mask].reset_index(drop=True)

    print(f"AE features: {len(ae_mean_cols)} mean + {len(ae_std_cols)} std")
    print(f"S2 features: {len(s2_feature_cols)}")
    print(f"Final dataset: {len(df)} rows")
    return df, ae_mean_cols, ae_std_cols, s2_feature_cols


def build_feature_matrix_s2_eu(df, s2_feature_cols, add_country=False):
    """Build EU S2-only feature matrix."""
    print("\n" + "=" * 60)
    print("BUILDING EU S2 FEATURE MATRIX")
    print("=" * 60)

    feature_cols = list(s2_feature_cols)
    aux_numeric = ['log_area', COL_LON, COL_LAT]
    feature_cols.extend(aux_numeric)

    if add_country:
        country_dummies = pd.get_dummies(df[COL_COUNTRY], prefix='ctry')
        for col in country_dummies.columns:
            df[col] = country_dummies[col].values
            feature_cols.append(col)

    print(f"Total features: {len(feature_cols)}")
    print(f"  S2 spectral: {len(s2_feature_cols)}")
    print(f"  Auxiliary numeric: {len(aux_numeric)}")
    return feature_cols


def build_feature_matrix_fusion_eu(df, ae_mean_cols, ae_std_cols,
                                    s2_feature_cols, add_country=False):
    """Build EU Fusion (AE + S2) feature matrix."""
    print("\n" + "=" * 60)
    print("BUILDING EU FUSION FEATURE MATRIX")
    print("=" * 60)

    feature_cols = list(ae_mean_cols) + list(ae_std_cols) + list(s2_feature_cols)
    aux_numeric = ['log_area', COL_LON, COL_LAT]
    feature_cols.extend(aux_numeric)

    if add_country:
        country_dummies = pd.get_dummies(df[COL_COUNTRY], prefix='ctry')
        for col in country_dummies.columns:
            df[col] = country_dummies[col].values
            feature_cols.append(col)

    print(f"Total features: {len(feature_cols)}")
    print(f"  AE mean: {len(ae_mean_cols)}")
    print(f"  AE std: {len(ae_std_cols)}")
    print(f"  S2 spectral: {len(s2_feature_cols)}")
    print(f"  Auxiliary numeric: {len(aux_numeric)}")
    return feature_cols


def split_and_save_eu_typed(df, feature_cols, feature_type='ae'):
    """Split EU data and save to feature-type-specific subdirectory."""
    print("\n" + "=" * 60)
    print(f"SPLITTING AND SAVING EU DATA ({feature_type.upper()})")
    print("=" * 60)

    label_encoders = {}
    for tier in ['label_tier_b', 'label_tier_a', 'label_tier_c']:
        le = LabelEncoder()
        df[f'{tier}_encoded'] = le.fit_transform(df[tier])
        label_encoders[tier] = le
        print(f"{tier}: {len(le.classes_)} classes")

    meta_cols = [
        COL_PARCEL_ID, COL_COUNTRY, COL_AREA,
        COL_HCAT3_NAME, 'split', 'size_class',
        'label_tier_b', 'label_tier_a', 'label_tier_c',
        'label_tier_b_encoded', 'label_tier_a_encoded',
        'label_tier_c_encoded',
    ]
    save_cols = feature_cols + [c for c in meta_cols if c in df.columns]

    # 输出到子目录
    out_dir = os.path.join(EU_FEATURES_DIR, feature_type)
    os.makedirs(out_dir, exist_ok=True)

    for split_name in ['train', 'val', 'test']:
        out_path = os.path.join(out_dir, f'features_{split_name}.parquet')
        split_df = df[df['split'] == split_name][save_cols].reset_index(
            drop=True)
        split_df.to_parquet(out_path, index=False)
        n_countries = split_df[COL_COUNTRY].nunique() if COL_COUNTRY in split_df.columns else 0
        print(f"\n{split_name}: {len(split_df)} parcels, "
              f"{n_countries} countries -> {out_path}")
        print(f"  File size: {os.path.getsize(out_path) / 1e6:.1f} MB")

    # Fit scaler and save artifacts
    train_df = df[df['split'] == 'train']
    scaler = StandardScaler()
    numeric_feature_cols = [c for c in feature_cols
                            if not c.startswith('ctry_')]
    scaler.fit(train_df[numeric_feature_cols].fillna(0))

    models_dir = os.path.join(MODELS_DIR, 'eu', feature_type)
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(models_dir, 'scaler.joblib'))
    joblib.dump(label_encoders, os.path.join(models_dir,
                                              'label_encoders.joblib'))
    joblib.dump(feature_cols, os.path.join(models_dir, 'feature_cols.joblib'))
    print(f"\nSaved artifacts to {models_dir}")


# ============================================================
# 比利时模式（保留兼容）
# ============================================================

def load_and_join(mode='centroid'):
    """Load parcels and embeddings, join them."""
    print("=" * 60)
    print("LOADING AND JOINING DATA")
    print("=" * 60)

    parcels = pd.read_parquet(PARCELS_FILTERED_PATH)
    print(f"Parcels: {len(parcels)}")

    if mode == 'centroid':
        emb_path = EMBEDDINGS_CENTROID_PATH
    else:
        emb_path = EMBEDDINGS_POLYGON_PATH

    if not os.path.exists(emb_path):
        print(f"ERROR: Embeddings not found at {emb_path}")
        sys.exit(1)

    emb = pd.read_parquet(emb_path)
    print(f"Embeddings: {len(emb)}")

    if mode == 'centroid' and 'parcel_idx' in emb.columns:
        df = parcels.copy()
        emb_aligned = emb.set_index('parcel_idx').reindex(range(len(parcels)))
        emb_cols = [c for c in emb_aligned.columns if c != 'parcel_idx']
        for col in emb_cols:
            df[col] = emb_aligned[col].values
    elif 'REF_ID' in emb.columns:
        df = parcels.merge(emb, on='REF_ID', how='left', suffixes=('', '_emb'))
    else:
        print("WARNING: No join key found, assuming row alignment")
        df = parcels.copy()
        emb_cols = [c for c in emb.columns]
        for col in emb_cols:
            df[col] = emb[col].values

    mean_cols = [f'{b}_mean' for b in AE_BANDS]
    std_cols = [f'{b}_std' for b in AE_BANDS]
    available_mean = [c for c in mean_cols if c in df.columns]
    available_std = [c for c in std_cols if c in df.columns]

    if not available_mean:
        available_mean = [c for c in AE_BANDS if c in df.columns]
        if available_mean:
            print("Using raw band names (no _mean suffix)")

    print(f"Embedding mean columns: {len(available_mean)}")
    print(f"Embedding std columns: {len(available_std)}")

    if available_mean:
        null_mask = df[available_mean].isnull().all(axis=1)
        n_null = null_mask.sum()
        if n_null > 0:
            print(f"Dropping {n_null} rows with null embeddings "
                  f"({n_null/len(df)*100:.2f}%)")
            df = df[~null_mask].reset_index(drop=True)

    print(f"Final dataset: {len(df)} rows")
    return df, available_mean, available_std


def build_feature_matrix(df, emb_mean_cols, emb_std_cols, mode='centroid'):
    """Build the feature matrix and labels."""
    print("\n" + "=" * 60)
    print("BUILDING FEATURE MATRIX")
    print("=" * 60)

    feature_cols = list(emb_mean_cols) + list(emb_std_cols)
    aux_numeric = ['log_area', 'longitude', 'latitude']
    feature_cols.extend(aux_numeric)

    if 'pixel_count' in df.columns:
        feature_cols.append('pixel_count')

    soil_dummies = pd.get_dummies(df['LANDBSTR'], prefix='soil')
    for col in soil_dummies.columns:
        df[col] = soil_dummies[col].values
        feature_cols.append(col)

    print(f"Total features: {len(feature_cols)}")
    print(f"  Embedding mean: {len(emb_mean_cols)}")
    print(f"  Embedding std: {len(emb_std_cols)}")
    print(f"  Auxiliary numeric: {len(aux_numeric)}")
    print(f"  Soil type one-hot: {len(soil_dummies.columns)}")

    return feature_cols


def split_and_save(df, feature_cols):
    """Split by pre-assigned split column and save."""
    print("\n" + "=" * 60)
    print("SPLITTING AND SAVING")
    print("=" * 60)

    label_encoders = {}
    for tier in ['label_tier_b', 'label_tier_a', 'label_tier_c']:
        le = LabelEncoder()
        df[f'{tier}_encoded'] = le.fit_transform(df[tier])
        label_encoders[tier] = le
        print(f"{tier}: {len(le.classes_)} classes")

    meta_cols = ['fid', 'REF_ID', 'GRAF_OPP', 'EC_hcat_n', 'EC_trans_n',
                 'LANDBSTR', 'EC_NUTS3', 'split', 'size_class',
                 'label_tier_b', 'label_tier_a', 'label_tier_c',
                 'label_tier_b_encoded', 'label_tier_a_encoded',
                 'label_tier_c_encoded']

    save_cols = feature_cols + [c for c in meta_cols if c in df.columns]

    for split_name, out_path in [('train', FEATURES_TRAIN_PATH),
                                  ('val', FEATURES_VAL_PATH),
                                  ('test', FEATURES_TEST_PATH)]:
        split_df = df[df['split'] == split_name][save_cols].reset_index(
            drop=True)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        split_df.to_parquet(out_path, index=False)
        print(f"\n{split_name}: {len(split_df)} parcels -> {out_path}")
        print(f"  File size: {os.path.getsize(out_path) / 1e6:.1f} MB")

    train_df = df[df['split'] == 'train']
    scaler = StandardScaler()
    numeric_feature_cols = [c for c in feature_cols
                            if not c.startswith('soil_')]
    scaler.fit(train_df[numeric_feature_cols].fillna(0))

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.joblib'))
    joblib.dump(label_encoders, os.path.join(MODELS_DIR,
                                              'label_encoders.joblib'))
    joblib.dump(feature_cols, os.path.join(MODELS_DIR, 'feature_cols.joblib'))
    print(f"\nSaved scaler and label encoders to {MODELS_DIR}")


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Build feature matrices from embeddings and parcel data')
    parser.add_argument('--mode', choices=['centroid', 'polygon', 'eu'],
                        default='centroid',
                        help='Feature building mode')
    parser.add_argument('--feature-type', choices=['ae', 's2', 'fusion'],
                        default='ae',
                        help='EU mode: feature source (ae/s2/fusion)')
    parser.add_argument('--add-country', action='store_true',
                        help='EU mode: add country one-hot encoding')
    args = parser.parse_args()

    if args.mode == 'eu':
        if args.feature_type == 's2':
            df, s2_cols = load_and_join_s2_eu()
            feature_cols = build_feature_matrix_s2_eu(
                df, s2_cols, add_country=args.add_country)
            split_and_save_eu_typed(df, feature_cols, feature_type='s2')
        elif args.feature_type == 'fusion':
            df, ae_mean, ae_std, s2_cols = load_and_join_fusion_eu()
            feature_cols = build_feature_matrix_fusion_eu(
                df, ae_mean, ae_std, s2_cols,
                add_country=args.add_country)
            split_and_save_eu_typed(df, feature_cols, feature_type='fusion')
        else:
            # AE 模式
            df, emb_mean_cols, emb_std_cols = load_and_join_eu()
            feature_cols = build_feature_matrix_eu(
                df, emb_mean_cols, emb_std_cols,
                add_country=args.add_country)
            split_and_save_eu_typed(df, feature_cols, feature_type='ae')
    else:
        df, emb_mean_cols, emb_std_cols = load_and_join(mode=args.mode)
        feature_cols = build_feature_matrix(df, emb_mean_cols, emb_std_cols,
                                            mode=args.mode)
        split_and_save(df, feature_cols)

    print("\n Feature engineering complete!")


if __name__ == '__main__':
    main()
