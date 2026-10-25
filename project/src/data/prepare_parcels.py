"""
数据准备：从 EuroCrops shapefile 中加载、过滤、标准化地块数据。

支持两种模式：
  - Belgian (原始): 处理比利时佛兰德斯单一数据集
  - EU (泛欧洲):    处理 Data_Crop 中所有国家的 EuroCrops 数据

Usage:
    # 比利时模式（保留兼容）
    python -m src.data.prepare_parcels

    # 泛欧洲模式 — 所有国家
    python -m src.data.prepare_parcels --mode eu

    # 泛欧洲模式 — 指定国家
    python -m src.data.prepare_parcels --mode eu --country BE_VLG DE_BB FR
"""
import sys
import os
import glob
import tempfile
import zipfile
import shutil
import argparse
import numpy as np
import pandas as pd
import geopandas as gpd
from pyproj import Transformer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from config.config import (
    SHAPEFILE_PATH, PROCESSED_DIR, PARCELS_FILTERED_PATH,
    EXCLUDE_CLASSES, FLEMISH_NUTS3_PREFIX, MIN_AREA_HA,
    SOURCE_CRS, TARGET_CRS, EQUAL_AREA_CRS,
    DATA_CROP_DIR, EU_PROCESSED_DIR, EU_PARCELS_PATH,
    COL_PARCEL_ID, COL_COUNTRY, COL_AREA, COL_LON, COL_LAT,
    COL_HCAT3_NAME, COL_HCAT3_CODE, COL_TRANS_NAME,
)
from config.class_definitions import get_tier_a_label, get_tier_b_label, get_tier_c_label
from config.spatial_splits import assign_split, assign_split_eu
from config.country_registry import (
    COUNTRY_REGISTRY, get_country, get_all_country_codes
)


# ============================================================
# 泛欧洲模式
# ============================================================

def extract_and_read_shapefile(country_config):
    """解压 ZIP 并读取 shapefile。"""
    zip_path = os.path.join(DATA_CROP_DIR, country_config.zip_name)

    if not os.path.exists(zip_path):
        print(f"  WARNING: ZIP 不存在: {zip_path}")
        return None

    # 临时目录放在项目所在盘，避免 C 盘空间不足
    tmp_base = os.path.join(os.path.dirname(DATA_CROP_DIR), '_tmp')
    os.makedirs(tmp_base, exist_ok=True)
    tmp_dir = tempfile.mkdtemp(prefix=f'ec_{country_config.code}_', dir=tmp_base)
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(tmp_dir)

        # 查找 .shp 文件
        shp_files = glob.glob(os.path.join(tmp_dir, '**', '*.shp'),
                              recursive=True)
        if not shp_files:
            print(f"  WARNING: ZIP 中未找到 .shp 文件: {zip_path}")
            return None

        # 用第一个 .shp
        gdf = gpd.read_file(shp_files[0])
        print(f"  已读取 {len(gdf)} 个地块, CRS={gdf.crs}")
        return gdf
    except Exception as e:
        print(f"  ERROR: 处理 {country_config.code} 失败: {e}")
        return None
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def standardize_country(gdf, country_config):
    """将单个国家的 GeoDataFrame 标准化为统一格式。

    输出列：parcel_id, country, area_ha, longitude, latitude,
            EC_hcat_n, EC_hcat_c, EC_trans_n,
            label_tier_a, label_tier_b, label_tier_c,
            log_area, size_class, split
    """
    code = country_config.code
    n_start = len(gdf)

    # 检查必需列
    if COL_HCAT3_NAME not in gdf.columns:
        print(f"  WARNING: {code} 缺少 {COL_HCAT3_NAME} 列, 跳过")
        return None

    # --- 过滤非农业类别 ---
    mask_agri = ~gdf[COL_HCAT3_NAME].isin(EXCLUDE_CLASSES)
    gdf = gdf[mask_agri].copy()
    n_after_agri = len(gdf)

    # 过滤空几何
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()].copy()

    # --- 计算面积（投影到等面积 CRS）---
    try:
        gdf_area = gdf.to_crs(EQUAL_AREA_CRS)
        gdf[COL_AREA] = gdf_area.geometry.area / 10000.0  # m² -> ha
    except Exception as e:
        print(f"  WARNING: {code} CRS 转换失败 ({e}), 尝试从列获取面积")
        # 尝试从已有列获取面积
        area_candidates = ['GRAF_OPP', 'area_ha', 'AREA_HA', 'Shape_Area',
                           'shape_area', 'AREA', 'area']
        found = False
        for acol in area_candidates:
            if acol in gdf.columns:
                gdf[COL_AREA] = gdf[acol].astype(float)
                if acol == 'Shape_Area':
                    gdf[COL_AREA] = gdf[COL_AREA] / 10000.0
                found = True
                break
        if not found:
            gdf[COL_AREA] = 0.01  # 默认最小值

    # --- 面积过滤 ---
    gdf = gdf[gdf[COL_AREA] >= MIN_AREA_HA].copy()
    n_after_area = len(gdf)

    if len(gdf) == 0:
        print(f"  WARNING: {code} 过滤后无数据")
        return None

    # --- 转换到 WGS84 并计算质心 ---
    if gdf.crs is None:
        print(f"  WARNING: {code} 无 CRS 信息，假设 EPSG:4326")
        gdf = gdf.set_crs(TARGET_CRS)
    elif str(gdf.crs) != TARGET_CRS:
        gdf = gdf.to_crs(TARGET_CRS)

    centroids = gdf.geometry.centroid
    gdf[COL_LON] = centroids.x
    gdf[COL_LAT] = centroids.y

    # --- 构建标准化 DataFrame ---
    gdf[COL_PARCEL_ID] = [f'{code}_{i}' for i in range(len(gdf))]
    gdf[COL_COUNTRY] = country_config.country_iso

    # 确保 HCAT 列存在
    if COL_HCAT3_CODE not in gdf.columns:
        gdf[COL_HCAT3_CODE] = ''
    if COL_TRANS_NAME not in gdf.columns:
        gdf[COL_TRANS_NAME] = ''

    # --- 分配标签 ---
    gdf['label_tier_b'] = gdf[COL_HCAT3_NAME].apply(get_tier_b_label)
    gdf['label_tier_a'] = gdf[COL_HCAT3_NAME].apply(get_tier_a_label)
    gdf['label_tier_c'] = gdf[COL_HCAT3_NAME].apply(get_tier_c_label)

    # --- 派生特征 ---
    gdf['log_area'] = np.log1p(gdf[COL_AREA])
    bins = [0, 0.1, 0.5, 2.0, np.inf]
    labels = ['tiny', 'small', 'medium', 'large']
    gdf['size_class'] = pd.cut(gdf[COL_AREA], bins=bins, labels=labels)

    # --- 分配 split ---
    gdf['split'] = gdf[COL_COUNTRY].apply(assign_split_eu)

    # --- 选择输出列 ---
    keep_cols = [
        COL_PARCEL_ID, COL_COUNTRY, COL_AREA,
        COL_LON, COL_LAT,
        COL_HCAT3_NAME, COL_HCAT3_CODE, COL_TRANS_NAME,
        'label_tier_b', 'label_tier_a', 'label_tier_c',
        'log_area', 'size_class', 'split',
    ]
    df = gdf[keep_cols].reset_index(drop=True)

    print(f"  {code}: {n_start} -> {n_after_agri} (过滤非农) "
          f"-> {n_after_area} (面积过滤) -> {len(df)} 最终")

    return df


def subsample_country(df, max_per_country, seed=42):
    """对单个国家进行分层采样（按 label_tier_b 分层），保持类别比例。

    临时方案：为加速 GEE 提取，对大国进行采样。
    未来需要用全量数据重新运行。
    """
    if max_per_country <= 0 or len(df) <= max_per_country:
        return df

    n_orig = len(df)
    rng = np.random.RandomState(seed)

    # 按 tier_b 分层采样，保持类别比例
    sampled_parts = []
    for cls, group in df.groupby('label_tier_b'):
        ratio = len(group) / n_orig
        n_sample = max(1, int(max_per_country * ratio))
        n_sample = min(n_sample, len(group))
        sampled_parts.append(
            group.iloc[rng.choice(len(group), size=n_sample, replace=False)])

    df_sampled = pd.concat(sampled_parts, ignore_index=True)
    print(f"  采样: {n_orig:,} -> {len(df_sampled):,} "
          f"(max_per_country={max_per_country:,})")
    return df_sampled


def process_eu(country_codes=None, max_per_country=0):
    """处理全部或指定国家，合并为统一 parquet。

    Args:
        country_codes: 指定国家列表，None 表示全部
        max_per_country: 每国最大地块数，0 = 不限制
    """
    if country_codes is None:
        country_codes = get_all_country_codes()

    print("=" * 60)
    print(f"泛欧洲数据准备 — {len(country_codes)} 个国家/地区")
    if max_per_country > 0:
        print(f"[NOTE] 每国采样上限: {max_per_country:,} 地块（临时方案）")
    print("=" * 60)

    all_dfs = []
    for code in country_codes:
        config = get_country(code)
        print(f"\n[{code}] {config.zip_name} (year={config.year})")

        gdf = extract_and_read_shapefile(config)
        if gdf is None:
            continue

        df = standardize_country(gdf, config)
        if df is not None and len(df) > 0:
            if max_per_country > 0:
                df = subsample_country(df, max_per_country)
            all_dfs.append(df)

    if not all_dfs:
        print("\nERROR: 没有成功处理的国家数据")
        sys.exit(1)

    # 合并
    df_all = pd.concat(all_dfs, ignore_index=True)

    # 保存
    os.makedirs(EU_PROCESSED_DIR, exist_ok=True)
    df_all.to_parquet(EU_PARCELS_PATH, index=False)

    print(f"\n{'=' * 60}")
    print(f"泛欧洲数据准备完成")
    print(f"{'=' * 60}")
    print(f"总地块: {len(df_all):,}")
    print(f"总面积: {df_all[COL_AREA].sum():,.0f} ha")
    if max_per_country > 0:
        print(f"[NOTE] 采样模式: max_per_country={max_per_country:,}")
    print(f"保存至: {EU_PARCELS_PATH}")
    print(f"文件大小: {os.path.getsize(EU_PARCELS_PATH) / 1e6:.1f} MB")

    # 打印汇总
    print_eu_summary(df_all)

    return df_all


def print_eu_summary(df):
    """打印泛欧数据集汇总统计。"""
    print(f"\n--- 各国统计 ---")
    country_stats = df.groupby(COL_COUNTRY).agg(
        n_parcels=(COL_PARCEL_ID, 'count'),
        total_area=(COL_AREA, 'sum'),
        n_classes=(COL_HCAT3_NAME, 'nunique'),
        split=('split', 'first'),
    ).sort_values('n_parcels', ascending=False)

    for iso, row in country_stats.iterrows():
        print(f"  {iso:4s}: {row['n_parcels']:>9,} parcels, "
              f"{row['total_area']:>12,.0f} ha, "
              f"{row['n_classes']:>3} classes, split={row['split']}")

    print(f"\n--- Split 分布 ---")
    for split_name, group in df.groupby('split'):
        n = len(group)
        n_countries = group[COL_COUNTRY].nunique()
        pct = n / len(df) * 100
        print(f"  {split_name:5s}: {n:>10,} parcels ({pct:.1f}%), "
              f"{n_countries} 个国家")

    print(f"\n--- Tier B 类别分布 ---")
    for cls, count in df['label_tier_b'].value_counts().head(10).items():
        area = df[df['label_tier_b'] == cls][COL_AREA].sum()
        print(f"  {cls:25s}: {count:>10,} parcels, {area:>12,.0f} ha")


# ============================================================
# 比利时模式（保留兼容）
# ============================================================

def load_and_filter_parcels():
    """Load shapefile and apply all filters."""
    print("=" * 60)
    print("LOADING SHAPEFILE")
    print("=" * 60)
    gdf = gpd.read_file(SHAPEFILE_PATH)
    print(f"Loaded {len(gdf)} polygons")
    print(f"CRS: {gdf.crs}")

    n_start = len(gdf)

    mask_agri = ~gdf['EC_hcat_n'].isin(EXCLUDE_CLASSES)
    gdf = gdf[mask_agri].copy()
    print(f"\nAfter removing {EXCLUDE_CLASSES}: {len(gdf)} "
          f"(removed {n_start - len(gdf)})")

    mask_flemish = gdf['EC_NUTS3'].str.startswith(FLEMISH_NUTS3_PREFIX, na=False)
    n_before = len(gdf)
    gdf = gdf[mask_flemish].copy()
    print(f"After keeping Flemish regions: {len(gdf)} "
          f"(removed {n_before - len(gdf)} non-Flemish)")

    n_before = len(gdf)
    gdf = gdf[gdf['GRAF_OPP'] >= MIN_AREA_HA].copy()
    print(f"After min area filter ({MIN_AREA_HA} ha): {len(gdf)} "
          f"(removed {n_before - len(gdf)} tiny parcels)")

    return gdf


def convert_coordinates(gdf):
    """Convert X_REF/Y_REF from Belgian Lambert 72 to WGS84."""
    print("\n" + "=" * 60)
    print("CONVERTING COORDINATES")
    print("=" * 60)
    transformer = Transformer.from_crs(SOURCE_CRS, TARGET_CRS, always_xy=True)
    lons, lats = transformer.transform(gdf['X_REF'].values, gdf['Y_REF'].values)
    gdf['longitude'] = lons
    gdf['latitude'] = lats
    print(f"Longitude range: [{lons.min():.4f}, {lons.max():.4f}]")
    print(f"Latitude range:  [{lats.min():.4f}, {lats.max():.4f}]")
    return gdf


def assign_labels_and_splits(gdf):
    """Add tier labels and train/val/test split assignment."""
    print("\n" + "=" * 60)
    print("ASSIGNING CLASS LABELS AND SPLITS")
    print("=" * 60)
    gdf['label_tier_b'] = gdf['EC_hcat_n'].apply(get_tier_b_label)
    gdf['label_tier_a'] = gdf['EC_hcat_n'].apply(get_tier_a_label)
    gdf['label_tier_c'] = gdf['EC_hcat_n'].apply(get_tier_c_label)
    gdf['split'] = gdf['EC_NUTS3'].apply(assign_split)

    for tier in ['label_tier_b', 'label_tier_a', 'label_tier_c']:
        n_classes = gdf[tier].nunique()
        print(f"{tier}: {n_classes} classes")

    print(f"\nSplit distribution:")
    split_counts = gdf['split'].value_counts()
    for split_name, count in split_counts.items():
        pct = count / len(gdf) * 100
        print(f"  {split_name}: {count} ({pct:.1f}%)")
    return gdf


def add_derived_features(gdf):
    """Add derived features useful for modeling."""
    gdf['log_area'] = np.log1p(gdf['GRAF_OPP'])
    bins = [0, 0.1, 0.5, 2.0, np.inf]
    labels = ['tiny', 'small', 'medium', 'large']
    gdf['size_class'] = pd.cut(gdf['GRAF_OPP'], bins=bins, labels=labels)
    return gdf


def save_parcels(gdf):
    """Save processed parcels as parquet (drop geometry for speed)."""
    print("\n" + "=" * 60)
    print("SAVING PROCESSED DATA")
    print("=" * 60)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    keep_cols = [
        'fid', 'GRAF_OPP', 'REF_ID',
        'GWSCOD_H', 'GWSNAM_H',
        'EC_hcat_n', 'EC_hcat_c', 'EC_trans_n',
        'LANDBSTR', 'EC_NUTS3',
        'X_REF', 'Y_REF', 'longitude', 'latitude',
        'label_tier_b', 'label_tier_a', 'label_tier_c',
        'split', 'log_area', 'size_class',
    ]
    df = gdf[keep_cols].copy()
    df = df.reset_index(drop=True)
    df.to_parquet(PARCELS_FILTERED_PATH, index=False)
    print(f"Saved {len(df)} parcels to {PARCELS_FILTERED_PATH}")
    print(f"File size: {os.path.getsize(PARCELS_FILTERED_PATH) / 1e6:.1f} MB")
    return df


def run_belgian_pipeline():
    """运行原始比利时流水线。"""
    gdf = load_and_filter_parcels()
    gdf = convert_coordinates(gdf)
    gdf = assign_labels_and_splits(gdf)
    gdf = add_derived_features(gdf)
    df = save_parcels(gdf)

    print(f"\nTotal parcels: {len(df)}")
    print(f"Total area: {df['GRAF_OPP'].sum():,.0f} ha")
    print("\n Data preparation complete!")
    return df


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Prepare parcel data for crop classification')
    parser.add_argument('--mode', choices=['be', 'eu'], default='be',
                        help='Processing mode: be (Belgian only) or eu (pan-European)')
    parser.add_argument('--country', nargs='*', default=None,
                        help='EU mode: specific country codes (e.g. BE_VLG DE_BB FR). '
                             'Omit for all countries.')
    parser.add_argument('--max-per-country', type=int, default=0,
                        help='Max parcels per country (0=no limit). '
                             'Temporary: for faster GEE extraction during development.')
    args = parser.parse_args()

    if args.mode == 'eu':
        process_eu(country_codes=args.country,
                   max_per_country=args.max_per_country)
    else:
        run_belgian_pipeline()


if __name__ == '__main__':
    main()
