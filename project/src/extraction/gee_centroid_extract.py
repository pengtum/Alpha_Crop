"""
Extract AlphaEarth embeddings at parcel centroids using GEE.

Uses reduceNeighborhood with a circular kernel to capture local context
(mean + std within 30m radius), producing 128 features per point.

支持两种模式：
  - Belgian: 处理比利时单一数据集（原始流水线）
  - EU:      逐国处理泛欧洲数据，年份自动匹配

Usage:
    # 比利时模式
    python -m src.extraction.gee_centroid_extract [--interactive]

    # 泛欧洲模式 — 所有国家
    python -m src.extraction.gee_centroid_extract --mode eu

    # 泛欧洲模式 — 指定国家
    python -m src.extraction.gee_centroid_extract --mode eu --country BE_VLG DE_BB

    # 小批量测试
    python -m src.extraction.gee_centroid_extract --mode eu --country BE_VLG --limit 100 --interactive
"""
import sys
import os
import time
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from config.config import (
    PARCELS_FILTERED_PATH, AE_COLLECTION, AE_SCALE, AE_BANDS,
    AE_YEAR, AE_NEIGHBORHOOD_RADIUS, GEE_BATCH_SIZE, GEE_EXPORT_FOLDER,
    GEE_PROJECT, GEE_EU_EXPORT_PREFIX,
    EU_PARCELS_PATH, COL_PARCEL_ID, COL_COUNTRY, COL_LON, COL_LAT,
)


def init_gee():
    """Initialize Google Earth Engine."""
    import ee

    if GEE_PROJECT is None:
        print("ERROR: GEE_PROJECT is not set in config/config.py")
        sys.exit(1)

    try:
        ee.Initialize(project=GEE_PROJECT)
        print(f"GEE initialized with project: {GEE_PROJECT}")
    except Exception as e:
        print(f"GEE initialization failed: {e}")
        print("\nPlease run: earthengine authenticate")
        sys.exit(1)

    return ee


def load_parcels_be():
    """Load filtered Belgian parcels with WGS84 coordinates."""
    df = pd.read_parquet(PARCELS_FILTERED_PATH)
    print(f"Loaded {len(df)} parcels")
    return df


def load_parcels_eu(country_code=None):
    """Load EU parcels, optionally filtered by country code."""
    df = pd.read_parquet(EU_PARCELS_PATH)
    if country_code:
        # 按注册表 code 前缀过滤（如 DE_BB -> country_iso='DE' 但 parcel_id 以 DE_BB_ 开头）
        df = df[df[COL_PARCEL_ID].str.startswith(f'{country_code}_')].copy()
        df = df.reset_index(drop=True)
    print(f"Loaded {len(df)} parcels" +
          (f" for {country_code}" if country_code else ""))
    return df


def build_embedding_image(ee, year=None):
    """Load AlphaEarth image and apply neighborhood reduction.

    Args:
        ee: Earth Engine module
        year: 数据年份，默认使用 AE_YEAR
    """
    if year is None:
        year = AE_YEAR

    print(f"\nLoading AlphaEarth embeddings for year {year}...")

    emb_image = (ee.ImageCollection(AE_COLLECTION)
                 .filterDate(f'{year}-01-01', f'{year + 1}-01-01')
                 .mosaic()
                 .select(AE_BANDS))

    radius = AE_NEIGHBORHOOD_RADIUS
    if radius > 0:
        print(f"Applying neighborhood reduction (radius={radius} pixels, "
              f"{radius * AE_SCALE}m)")
        kernel = ee.Kernel.circle(radius=radius, units='pixels')

        mean_image = emb_image.reduceNeighborhood(
            reducer=ee.Reducer.mean(), kernel=kernel)
        std_image = emb_image.reduceNeighborhood(
            reducer=ee.Reducer.stdDev(), kernel=kernel)

        combined_image = mean_image.addBands(std_image)

        feature_names = ([f'{b}_mean' for b in AE_BANDS] +
                         [f'{b}_std' for b in AE_BANDS])
    else:
        print("No neighborhood reduction (point sampling)")
        combined_image = emb_image
        feature_names = list(AE_BANDS)

    print(f"Feature dimensions: {len(feature_names)}")
    return combined_image, feature_names


def extract_batch_interactive(ee, df_batch, combined_image, batch_idx,
                              n_batches, id_col='parcel_idx'):
    """Extract embeddings for a batch using getInfo() (interactive mode)."""
    features = []
    for idx, row in df_batch.iterrows():
        geom = ee.Geometry.Point([row[COL_LON], row[COL_LAT]])
        feat = ee.Feature(geom, {id_col: str(row.get(COL_PARCEL_ID, idx))})
        features.append(feat)

    fc = ee.FeatureCollection(features)

    sampled = combined_image.sampleRegions(
        collection=fc,
        properties=[id_col],
        scale=AE_SCALE,
        geometries=False
    )

    print(f"  Batch {batch_idx + 1}/{n_batches} ({len(df_batch)} points)...",
          end='', flush=True)

    try:
        result = sampled.getInfo()
        batch_features = [f['properties'] for f in result['features']]
        print(f" OK ({len(batch_features)} results)")
        return batch_features
    except Exception as e:
        print(f" FAILED: {e}")
        return None


def export_batch_to_drive(ee, df_batch, combined_image, batch_idx,
                          export_folder, task_prefix='ae_centroid',
                          id_col='parcel_idx'):
    """Export a batch as a CSV to Google Drive (batch mode)."""
    features = []
    for idx, row in df_batch.iterrows():
        geom = ee.Geometry.Point([row[COL_LON], row[COL_LAT]])
        feat = ee.Feature(geom, {id_col: str(row.get(COL_PARCEL_ID, idx))})
        features.append(feat)

    fc = ee.FeatureCollection(features)

    sampled = combined_image.sampleRegions(
        collection=fc,
        properties=[id_col],
        scale=AE_SCALE,
        geometries=False
    )

    task_name = f'{task_prefix}_batch_{batch_idx:04d}'
    task = ee.batch.Export.table.toDrive(
        collection=sampled,
        description=task_name,
        folder=export_folder,
        fileFormat='CSV'
    )
    task.start()
    return task, task_name


def extract_interactive(ee, df, combined_image, feature_names,
                        batch_size=500, id_col='parcel_idx'):
    """Interactive extraction using getInfo() - for small tests."""
    print("\n" + "=" * 60)
    print("INTERACTIVE EXTRACTION (getInfo)")
    print("=" * 60)

    all_features = []
    n_batches = (len(df) + batch_size - 1) // batch_size

    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end = min((batch_idx + 1) * batch_size, len(df))
        batch_df = df.iloc[start:end]

        result = extract_batch_interactive(
            ee, batch_df, combined_image, batch_idx, n_batches,
            id_col=id_col)

        if result is not None:
            all_features.extend(result)
        else:
            for _ in range(len(batch_df)):
                all_features.append(
                    {name: np.nan for name in feature_names + [id_col]})

    emb_df = pd.DataFrame(all_features)
    if id_col in emb_df.columns:
        emb_df = emb_df.sort_values(id_col).reset_index(drop=True)

    emb_cols = [c for c in emb_df.columns if c != id_col]
    n_valid = emb_df[emb_cols[0]].notna().sum() if emb_cols else 0
    print(f"\nValid embeddings: {n_valid}/{len(df)} "
          f"({n_valid / len(df) * 100:.1f}%)")

    return emb_df


def extract_batch_export(ee, df, combined_image, batch_size=5000,
                         export_folder=None, task_prefix='ae_centroid',
                         id_col='parcel_idx'):
    """Batch export to Google Drive - for full dataset."""
    if export_folder is None:
        export_folder = GEE_EXPORT_FOLDER

    print("\n" + "=" * 60)
    print("BATCH EXPORT TO GOOGLE DRIVE")
    print("=" * 60)
    print(f"Export folder: {export_folder}")
    print(f"Batch size: {batch_size}")

    n_batches = (len(df) + batch_size - 1) // batch_size
    print(f"Total batches: {n_batches}")

    tasks = []
    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end = min((batch_idx + 1) * batch_size, len(df))
        batch_df = df.iloc[start:end]

        task, task_name = export_batch_to_drive(
            ee, batch_df, combined_image, batch_idx, export_folder,
            task_prefix=task_prefix, id_col=id_col)
        tasks.append((task, task_name))
        print(f"  Started {task_name} ({len(batch_df)} points)")

        if (batch_idx + 1) % 20 == 0:
            print(f"  Pausing 5s after {batch_idx + 1} tasks...")
            time.sleep(5)

    print(f"\n{'=' * 60}")
    print(f"SUBMITTED {len(tasks)} EXPORT TASKS")
    print(f"{'=' * 60}")
    print(f"\nCheck status at: https://code.earthengine.google.com/tasks")
    print(f"CSVs will appear in Google Drive folder: '{export_folder}'")

    return tasks


def monitor_tasks(ee, tasks, check_interval=60):
    """Monitor GEE export tasks until completion."""
    print(f"\nMonitoring {len(tasks)} tasks (checking every {check_interval}s)...")

    while True:
        statuses = {}
        for task, name in tasks:
            status = task.status()
            state = status['state']
            statuses[state] = statuses.get(state, 0) + 1

        status_str = ', '.join(f"{k}: {v}" for k, v in sorted(statuses.items()))
        print(f"  [{time.strftime('%H:%M:%S')}] {status_str}")

        if all(task.status()['state'] in ('COMPLETED', 'FAILED', 'CANCELLED')
               for task, _ in tasks):
            break

        time.sleep(check_interval)

    completed = sum(1 for t, _ in tasks if t.status()['state'] == 'COMPLETED')
    failed = sum(1 for t, _ in tasks if t.status()['state'] == 'FAILED')
    print(f"\nDone! Completed: {completed}, Failed: {failed}")

    if failed > 0:
        print("Failed tasks:")
        for task, name in tasks:
            if task.status()['state'] == 'FAILED':
                print(f"  {name}: {task.status().get('error_message', 'unknown')}")


# ============================================================
# 泛欧洲模式入口
# ============================================================

def extract_eu_country(ee, country_code, interactive=False, limit=0,
                       batch_size=None, monitor=False):
    """提取单个国家的 AlphaEarth 嵌入。"""
    from config.country_registry import get_country

    config = get_country(country_code)
    print(f"\n{'=' * 60}")
    print(f"EXTRACTING: {country_code} (year={config.year})")
    print(f"{'=' * 60}")

    # 加载该国地块
    df = load_parcels_eu(country_code)
    if limit > 0:
        df = df.head(limit)
        print(f"Limited to {len(df)} parcels")

    if len(df) == 0:
        print(f"  No parcels for {country_code}, skipping")
        return []

    # 构建该年份的嵌入影像
    combined_image, feature_names = build_embedding_image(ee, year=config.year)

    export_folder = f'{GEE_EU_EXPORT_PREFIX}_{country_code}'
    task_prefix = f'ae_{country_code}'
    id_col = COL_PARCEL_ID

    if interactive:
        bs = batch_size or 500
        emb_df = extract_interactive(
            ee, df, combined_image, feature_names,
            batch_size=bs, id_col=id_col)

        # 直接保存
        from config.config import EU_EMBEDDINGS_DIR
        os.makedirs(EU_EMBEDDINGS_DIR, exist_ok=True)
        out_path = os.path.join(EU_EMBEDDINGS_DIR,
                                f'embeddings_{country_code}.parquet')
        emb_df.to_parquet(out_path, index=False)
        print(f"Saved to {out_path}")
        return []
    else:
        bs = batch_size or GEE_BATCH_SIZE
        tasks = extract_batch_export(
            ee, df, combined_image, batch_size=bs,
            export_folder=export_folder, task_prefix=task_prefix,
            id_col=id_col)

        if monitor:
            monitor_tasks(ee, tasks)

        return tasks


def extract_eu(ee, country_codes=None, interactive=False, limit=0,
               batch_size=None, monitor=False):
    """提取所有或指定国家的嵌入。"""
    from config.country_registry import get_all_country_codes

    if country_codes is None:
        country_codes = get_all_country_codes()

    print(f"\n泛欧洲嵌入提取 — {len(country_codes)} 个国家/地区")

    all_tasks = []
    for code in country_codes:
        tasks = extract_eu_country(
            ee, code, interactive=interactive, limit=limit,
            batch_size=batch_size, monitor=False)
        all_tasks.extend(tasks)

    if all_tasks and monitor:
        monitor_tasks(ee, all_tasks)

    print(f"\n总计提交 {len(all_tasks)} 个 GEE 任务")
    return all_tasks


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Extract AlphaEarth embeddings at parcel centroids')
    parser.add_argument('--mode', choices=['be', 'eu'], default='be',
                        help='Processing mode: be (Belgian) or eu (pan-European)')
    parser.add_argument('--country', nargs='*', default=None,
                        help='EU mode: specific country codes')
    parser.add_argument('--interactive', action='store_true',
                        help='Use interactive mode (getInfo) for small tests')
    parser.add_argument('--limit', type=int, default=0,
                        help='Limit number of parcels (0=all)')
    parser.add_argument('--monitor', action='store_true',
                        help='Monitor batch export tasks until completion')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Override batch size')
    args = parser.parse_args()

    ee = init_gee()

    if args.mode == 'eu':
        extract_eu(ee, country_codes=args.country,
                   interactive=args.interactive, limit=args.limit,
                   batch_size=args.batch_size, monitor=args.monitor)
    else:
        # 原始比利时模式
        df = load_parcels_be()
        if args.limit > 0:
            df = df.head(args.limit)
            print(f"Limited to {len(df)} parcels")

        combined_image, feature_names = build_embedding_image(ee)

        if args.interactive:
            batch_size = args.batch_size or 500
            emb_df = extract_interactive(ee, df, combined_image, feature_names,
                                         batch_size=batch_size)
            out_path = PARCELS_FILTERED_PATH.replace(
                '.parquet', '_embeddings_interactive.parquet')
            emb_df.to_parquet(out_path, index=False)
            print(f"\nSaved to {out_path}")
        else:
            batch_size = args.batch_size or GEE_BATCH_SIZE
            tasks = extract_batch_export(ee, df, combined_image,
                                         batch_size=batch_size)
            if args.monitor:
                monitor_tasks(ee, tasks)


if __name__ == '__main__':
    main()
