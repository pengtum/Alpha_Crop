"""
Merge exported CSV batches from Google Drive into parquet files.

支持两种模式：
  - Belgian: 合并单一数据集的 CSV（原始流水线）
  - EU:      逐国合并 CSV，最终汇总为统一 parquet

Usage:
    # 比利时模式
    python -m src.extraction.merge_embeddings --csv-dir /path/to/csvs [--mode centroid|polygon]

    # 泛欧洲模式 — 指定国家
    python -m src.extraction.merge_embeddings --mode eu --country BE_VLG --csv-dir /path/to/csvs

    # 泛欧洲模式 — 合并已有的各国 parquet
    python -m src.extraction.merge_embeddings --mode eu --merge-all
"""
import sys
import os
import glob
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from config.config import (
    PARCELS_FILTERED_PATH, EMBEDDINGS_CENTROID_PATH,
    EMBEDDINGS_POLYGON_PATH, AE_BANDS,
    EU_PARCELS_PATH, EU_EMBEDDINGS_DIR, EU_S2_EMBEDDINGS_DIR,
    COL_PARCEL_ID, GEE_EU_EXPORT_PREFIX,
)


def merge_csvs(csv_dir, file_pattern='*.csv'):
    """通用 CSV 合并函数。"""
    csv_files = sorted(glob.glob(os.path.join(csv_dir, file_pattern)))

    if not csv_files:
        # 回退：查找所有 CSV
        csv_files = sorted(glob.glob(os.path.join(csv_dir, '*.csv')))

    if not csv_files:
        print(f"ERROR: No CSV files found in {csv_dir}")
        return None

    print(f"Found {len(csv_files)} CSV files")

    dfs = []
    for f in csv_files:
        df = pd.read_csv(f)
        dfs.append(df)
        print(f"  {os.path.basename(f)}: {len(df)} rows")

    merged = pd.concat(dfs, ignore_index=True)
    print(f"\nMerged: {len(merged)} total rows")

    # Remove system columns (GEE export artifacts)
    drop_cols = [c for c in merged.columns if c.startswith('system:')]
    if drop_cols:
        merged = merged.drop(columns=drop_cols)

    # Remove .geo column if present
    if '.geo' in merged.columns:
        merged = merged.drop(columns=['.geo'])

    return merged


def merge_centroid_csvs(csv_dir):
    """Merge centroid extraction CSVs."""
    merged = merge_csvs(csv_dir, 'ae_centroid_batch_*.csv')
    if merged is not None and 'parcel_idx' in merged.columns:
        merged = merged.sort_values('parcel_idx').reset_index(drop=True)
    return merged


def merge_polygon_csvs(csv_dir):
    """Merge polygon extraction CSVs."""
    return merge_csvs(csv_dir, 'ae_polygon_*.csv')


def validate_embeddings(emb_df, parcels_path=None):
    """Validate merged embeddings."""
    print("\n" + "=" * 60)
    print("VALIDATION")
    print("=" * 60)

    emb_cols = ([f'{b}_mean' for b in AE_BANDS] +
                [f'{b}_std' for b in AE_BANDS])

    if 'pixel_count' in emb_df.columns:
        print(f"Pixel count range: [{emb_df['pixel_count'].min()}, "
              f"{emb_df['pixel_count'].max()}]")

    available = [c for c in emb_cols if c in emb_df.columns]
    missing = [c for c in emb_cols if c not in emb_df.columns]
    print(f"Expected embedding columns: {len(emb_cols)}")
    print(f"Available: {len(available)}")
    if missing:
        print(f"Missing: {missing[:5]}...")

    if available:
        null_counts = emb_df[available].isnull().sum(axis=1)
        n_complete = (null_counts == 0).sum()
        n_partial = ((null_counts > 0) & (null_counts < len(available))).sum()
        n_empty = (null_counts == len(available)).sum()
        print(f"\nRow completeness:")
        print(f"  Complete (no nulls): {n_complete} "
              f"({n_complete/len(emb_df)*100:.1f}%)")
        print(f"  Partial nulls: {n_partial}")
        print(f"  All null: {n_empty}")

        sample_col = available[0]
        vals = emb_df[sample_col].dropna()
        if len(vals) > 0:
            print(f"\nSample column '{sample_col}':")
            print(f"  Range: [{vals.min():.4f}, {vals.max():.4f}]")
            print(f"  Mean: {vals.mean():.4f}, Std: {vals.std():.4f}")

    if parcels_path and os.path.exists(parcels_path):
        parcels = pd.read_parquet(parcels_path)
        print(f"\nParcels in dataset: {len(parcels)}")
        print(f"Embeddings extracted: {len(emb_df)}")
        print(f"Join rate: {len(emb_df)/len(parcels)*100:.1f}%")


# ============================================================
# 泛欧洲模式
# ============================================================

def merge_eu_country(country_code, csv_dir):
    """合并单个国家的 CSV 导出。"""
    pattern = f'ae_{country_code}_batch_*.csv'
    merged = merge_csvs(csv_dir, pattern)

    if merged is None:
        return None

    # 按 parcel_id 排序
    if COL_PARCEL_ID in merged.columns:
        merged = merged.sort_values(COL_PARCEL_ID).reset_index(drop=True)

    # 保存
    os.makedirs(EU_EMBEDDINGS_DIR, exist_ok=True)
    out_path = os.path.join(EU_EMBEDDINGS_DIR,
                            f'embeddings_{country_code}.parquet')
    merged.to_parquet(out_path, index=False)
    print(f"Saved to {out_path}")

    return merged


def merge_all_eu_countries():
    """合并所有国家的嵌入 parquet 为统一文件。"""
    print("\n" + "=" * 60)
    print("MERGING ALL EU COUNTRY EMBEDDINGS")
    print("=" * 60)

    parquet_files = sorted(glob.glob(
        os.path.join(EU_EMBEDDINGS_DIR, 'embeddings_*.parquet')))

    if not parquet_files:
        print(f"ERROR: No embedding parquet files in {EU_EMBEDDINGS_DIR}")
        sys.exit(1)

    print(f"Found {len(parquet_files)} country embedding files")

    dfs = []
    for f in parquet_files:
        df = pd.read_parquet(f)
        country = os.path.basename(f).replace('embeddings_', '').replace(
            '.parquet', '')
        print(f"  {country}: {len(df)} rows")
        dfs.append(df)

    merged = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal: {len(merged)} rows")

    out_path = os.path.join(EU_EMBEDDINGS_DIR, 'embeddings_eu.parquet')
    merged.to_parquet(out_path, index=False)
    print(f"Saved to {out_path}")
    print(f"File size: {os.path.getsize(out_path) / 1e6:.1f} MB")

    validate_embeddings(merged, parcels_path=EU_PARCELS_PATH)
    return merged


# ============================================================
# S2 合并
# ============================================================

def merge_s2_country(country_code, csv_dir):
    """合并单个国家的 S2 CSV 导出。"""
    pattern = f's2_{country_code}_batch_*.csv'
    merged = merge_csvs(csv_dir, pattern)

    if merged is None:
        return None

    if COL_PARCEL_ID in merged.columns:
        merged = merged.sort_values(COL_PARCEL_ID).reset_index(drop=True)

    os.makedirs(EU_S2_EMBEDDINGS_DIR, exist_ok=True)
    out_path = os.path.join(EU_S2_EMBEDDINGS_DIR,
                            f's2_{country_code}.parquet')
    merged.to_parquet(out_path, index=False)
    print(f"Saved to {out_path}")

    return merged


def merge_all_s2_countries():
    """合并所有国家的 S2 parquet 为统一文件。"""
    print("\n" + "=" * 60)
    print("MERGING ALL EU S2 FEATURES")
    print("=" * 60)

    parquet_files = sorted(glob.glob(
        os.path.join(EU_S2_EMBEDDINGS_DIR, 's2_*.parquet')))

    # 排除汇总文件
    parquet_files = [f for f in parquet_files
                     if os.path.basename(f) != 's2_embeddings_eu.parquet']

    if not parquet_files:
        print(f"ERROR: No S2 parquet files in {EU_S2_EMBEDDINGS_DIR}")
        sys.exit(1)

    print(f"Found {len(parquet_files)} country S2 files")

    dfs = []
    for f in parquet_files:
        df = pd.read_parquet(f)
        country = os.path.basename(f).replace('s2_', '').replace(
            '.parquet', '')
        print(f"  {country}: {len(df)} rows, {len(df.columns)} cols")
        dfs.append(df)

    merged = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal: {len(merged)} rows")

    out_path = os.path.join(EU_S2_EMBEDDINGS_DIR, 's2_embeddings_eu.parquet')
    merged.to_parquet(out_path, index=False)
    print(f"Saved to {out_path}")
    print(f"File size: {os.path.getsize(out_path) / 1e6:.1f} MB")

    return merged


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Merge GEE export CSVs into parquet')
    parser.add_argument('--csv-dir', default=None,
                        help='Directory containing downloaded CSV files')
    parser.add_argument('--mode', choices=['centroid', 'polygon', 'eu'],
                        default='centroid',
                        help='Merge mode')
    parser.add_argument('--source', choices=['ae', 's2'], default='ae',
                        help='Feature source: ae (AlphaEarth) or s2 (Sentinel-2)')
    parser.add_argument('--country', type=str, default=None,
                        help='EU mode: country code to merge')
    parser.add_argument('--merge-all', action='store_true',
                        help='EU mode: merge all country parquets into one')
    args = parser.parse_args()

    if args.mode == 'eu':
        if args.source == 's2':
            # S2 模式
            if args.merge_all:
                merge_all_s2_countries()
            elif args.country and args.csv_dir:
                merge_s2_country(args.country, args.csv_dir)
            else:
                print("S2 EU mode requires --country + --csv-dir, or --merge-all")
                sys.exit(1)
        else:
            # AE 模式（原有逻辑）
            if args.merge_all:
                merge_all_eu_countries()
            elif args.country and args.csv_dir:
                merge_eu_country(args.country, args.csv_dir)
            else:
                print("EU mode requires --country + --csv-dir, or --merge-all")
                sys.exit(1)
    else:
        if not args.csv_dir:
            print("ERROR: --csv-dir is required for centroid/polygon mode")
            sys.exit(1)

        if args.mode == 'centroid':
            emb_df = merge_centroid_csvs(args.csv_dir)
            out_path = EMBEDDINGS_CENTROID_PATH
        else:
            emb_df = merge_polygon_csvs(args.csv_dir)
            out_path = EMBEDDINGS_POLYGON_PATH

        if emb_df is not None:
            validate_embeddings(emb_df, parcels_path=PARCELS_FILTERED_PATH)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            emb_df.to_parquet(out_path, index=False)
            print(f"\nSaved to {out_path}")
            print(f"File size: {os.path.getsize(out_path) / 1e6:.1f} MB")


if __name__ == '__main__':
    main()
