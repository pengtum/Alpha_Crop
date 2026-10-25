"""
Extract Sentinel-2 spectral features at parcel centroids using GEE.

作为 AlphaEarth 嵌入的传统光谱基线对照。使用相同的 centroid + 邻域核
提取方式和 batch export 工作流。

提取特征 (~100 维):
  1. 年度光谱统计: 10 波段 × (median, mean, std) = 30
  2. 植被指数年度统计: 6 指数 × (median, mean, std, max) = 24
  3. 季度 NDVI/EVI 统计: 4 季 × 2 指数 × (median, std) = 16
  4. 季度关键波段中位数: 4 季 × 3 波段 (B4/B8/B11) = 12
  5. 绿化指标: NDVI/EVI 的 Q3-Q1 差异 = 2
  总计: ~84 维 (邻域 mean+std 应用前)

Usage:
    # 泛欧洲模式 — 所有国家
    python -m src.extraction.gee_s2_extract --mode eu

    # 泛欧洲模式 — 指定国家
    python -m src.extraction.gee_s2_extract --mode eu --country BE_VLG DE_BB

    # 小批量测试
    python -m src.extraction.gee_s2_extract --mode eu --country BE_VLG --limit 100 --interactive
"""
import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from config.config import (
    S2_COLLECTION, S2_CLOUD_THRESHOLD, S2_BANDS, S2_SCALE,
    S2_NEIGHBORHOOD_RADIUS, GEE_BATCH_SIZE,
    GEE_S2_EU_EXPORT_PREFIX, GEE_PROJECT,
    EU_PARCELS_PATH, EU_S2_EMBEDDINGS_DIR,
    COL_PARCEL_ID, COL_COUNTRY, COL_LON, COL_LAT,
)
# 复用 AlphaEarth 提取模块的通用函数
from src.extraction.gee_centroid_extract import (
    init_gee, load_parcels_eu, load_parcels_be,
    extract_batch_interactive, export_batch_to_drive,
    extract_interactive, extract_batch_export, monitor_tasks,
)


# ============================================================
# S2 影像构建
# ============================================================

def cloud_mask_s2(ee, image):
    """使用 SCL 波段进行云掩膜。

    SCL 分类:
      4 = Vegetation, 5 = Bare soil, 6 = Water
      保留 4, 5, 6, 7(unclassified), 11(snow/ice 可选)
      去除 3(cloud shadow), 8(cloud medium), 9(cloud high), 10(cirrus)
    """
    scl = image.select('SCL')
    clear_mask = (scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6))
                  .Or(scl.eq(7)).Or(scl.eq(11)))
    return image.updateMask(clear_mask)


def add_vegetation_indices(ee, image):
    """计算植被/水体指数并添加为波段。"""
    b2 = image.select('B2')
    b3 = image.select('B3')
    b4 = image.select('B4')
    b8 = image.select('B8')
    b11 = image.select('B11')

    ndvi = b8.subtract(b4).divide(b8.add(b4)).rename('NDVI')
    evi = (b8.subtract(b4).multiply(2.5)
           .divide(b8.add(b4.multiply(6)).subtract(b2.multiply(7.5)).add(1))
           .rename('EVI'))
    savi = (b8.subtract(b4).multiply(1.5)
            .divide(b8.add(b4).add(0.5)).rename('SAVI'))
    ndwi = b3.subtract(b8).divide(b3.add(b8)).rename('NDWI')
    lswi = b8.subtract(b11).divide(b8.add(b11)).rename('LSWI')
    ndbi = b11.subtract(b8).divide(b11.add(b8)).rename('NDBI')

    return image.addBands([ndvi, evi, savi, ndwi, lswi, ndbi])


def build_s2_composite(ee, year):
    """构建 S2 年度合成影像，包含光谱统计、植被指数和季节性特征。

    Args:
        ee: Earth Engine module
        year: 数据年份

    Returns:
        (combined_image, feature_names): 合成影像和特征名列表
    """
    print(f"\nBuilding S2 composite for year {year}...")

    start_date = f'{year}-01-01'
    end_date = f'{year + 1}-01-01'

    # 加载 S2 SR，按云覆盖和日期过滤
    s2 = (ee.ImageCollection(S2_COLLECTION)
          .filterDate(start_date, end_date)
          .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', S2_CLOUD_THRESHOLD))
          .map(lambda img: cloud_mask_s2(ee, img)))

    # 添加植被指数
    s2_vi = s2.map(lambda img: add_vegetation_indices(ee, img))

    feature_bands = []
    feature_names = []

    # -------------------------------------------------------
    # 1. 年度光谱统计: median, mean, std (10 bands × 3 = 30)
    # -------------------------------------------------------
    bands_selected = s2_vi.select(S2_BANDS)

    annual_median = bands_selected.median()
    annual_mean = bands_selected.mean()
    annual_std = bands_selected.reduce(ee.Reducer.stdDev())

    for b in S2_BANDS:
        feature_bands.append(annual_median.select(b).rename(f'{b}_median'))
        feature_names.append(f'{b}_median')
    for b in S2_BANDS:
        feature_bands.append(annual_mean.select(b).rename(f'{b}_mean'))
        feature_names.append(f'{b}_mean')
    for b in S2_BANDS:
        feature_bands.append(annual_std.select(f'{b}_stdDev').rename(f'{b}_std'))
        feature_names.append(f'{b}_std')

    # -------------------------------------------------------
    # 2. 植被指数年度统计: median, mean, std, max (6 VI × 4 = 24)
    # -------------------------------------------------------
    vi_names = ['NDVI', 'EVI', 'SAVI', 'NDWI', 'LSWI', 'NDBI']
    vi_selected = s2_vi.select(vi_names)

    vi_median = vi_selected.median()
    vi_mean = vi_selected.mean()
    vi_std = vi_selected.reduce(ee.Reducer.stdDev())
    vi_max = vi_selected.max()

    for vi in vi_names:
        feature_bands.append(vi_median.select(vi).rename(f'{vi}_median'))
        feature_names.append(f'{vi}_median')
    for vi in vi_names:
        feature_bands.append(vi_mean.select(vi).rename(f'{vi}_mean'))
        feature_names.append(f'{vi}_mean')
    for vi in vi_names:
        feature_bands.append(vi_std.select(f'{vi}_stdDev').rename(f'{vi}_std'))
        feature_names.append(f'{vi}_std')
    for vi in vi_names:
        feature_bands.append(vi_max.select(vi).rename(f'{vi}_max'))
        feature_names.append(f'{vi}_max')

    # -------------------------------------------------------
    # 3. 季度 NDVI/EVI 统计 (4 季 × 2 指数 × 2 统计 = 16)
    # -------------------------------------------------------
    quarters = {
        'Q1': (f'{year}-01-01', f'{year}-04-01'),
        'Q2': (f'{year}-04-01', f'{year}-07-01'),
        'Q3': (f'{year}-07-01', f'{year}-10-01'),
        'Q4': (f'{year}-10-01', f'{year + 1}-01-01'),
    }

    q_median_bands = {}  # 存储季度中位数用于绿化指标

    for q_name, (q_start, q_end) in quarters.items():
        q_col = (s2_vi.filterDate(q_start, q_end)
                 .select(['NDVI', 'EVI']))

        q_med = q_col.median()
        q_std = q_col.reduce(ee.Reducer.stdDev())

        for vi in ['NDVI', 'EVI']:
            band_med = q_med.select(vi).rename(f'{vi}_{q_name}_median')
            feature_bands.append(band_med)
            feature_names.append(f'{vi}_{q_name}_median')

            band_std = q_std.select(f'{vi}_stdDev').rename(f'{vi}_{q_name}_std')
            feature_bands.append(band_std)
            feature_names.append(f'{vi}_{q_name}_std')

        q_median_bands[q_name] = q_med

    # -------------------------------------------------------
    # 4. 季度关键波段中位数 (4 季 × 3 波段 = 12)
    # -------------------------------------------------------
    key_bands = ['B4', 'B8', 'B11']
    for q_name, (q_start, q_end) in quarters.items():
        q_col = s2_vi.filterDate(q_start, q_end).select(key_bands)
        q_med = q_col.median()
        for b in key_bands:
            band = q_med.select(b).rename(f'{b}_{q_name}_median')
            feature_bands.append(band)
            feature_names.append(f'{b}_{q_name}_median')

    # -------------------------------------------------------
    # 5. 绿化指标: Q3 - Q1 差异 (NDVI, EVI)
    # -------------------------------------------------------
    for vi in ['NDVI', 'EVI']:
        q3_band = q_median_bands['Q3'].select(vi)
        q1_band = q_median_bands['Q1'].select(vi)
        diff = q3_band.subtract(q1_band).rename(f'{vi}_greenup')
        feature_bands.append(diff)
        feature_names.append(f'{vi}_greenup')

    # -------------------------------------------------------
    # 合并所有波段
    # -------------------------------------------------------
    combined = feature_bands[0]
    for band in feature_bands[1:]:
        combined = combined.addBands(band)

    print(f"S2 composite bands: {len(feature_names)}")

    # -------------------------------------------------------
    # 应用邻域 reduceNeighborhood (与 AlphaEarth 一致)
    # -------------------------------------------------------
    radius = S2_NEIGHBORHOOD_RADIUS
    if radius > 0:
        print(f"Applying neighborhood reduction (radius={radius} pixels, "
              f"{radius * S2_SCALE}m)")
        kernel = ee.Kernel.circle(radius=radius, units='pixels')

        mean_image = combined.reduceNeighborhood(
            reducer=ee.Reducer.mean(), kernel=kernel)
        std_image = combined.reduceNeighborhood(
            reducer=ee.Reducer.stdDev(), kernel=kernel)

        combined_image = mean_image.addBands(std_image)

        # GEE reduceNeighborhood 会添加 _mean / _stdDev 后缀
        final_names = ([f'{n}_mean' for n in feature_names] +
                       [f'{n}_std' for n in feature_names])
    else:
        combined_image = combined
        final_names = feature_names

    print(f"Final feature dimensions: {len(final_names)}")
    return combined_image, final_names


# ============================================================
# 泛欧洲模式
# ============================================================

def extract_s2_eu_country(ee, country_code, interactive=False, limit=0,
                          batch_size=None, monitor=False):
    """提取单个国家的 S2 光谱特征。"""
    from config.country_registry import get_country

    config = get_country(country_code)
    print(f"\n{'=' * 60}")
    print(f"S2 EXTRACTING: {country_code} (year={config.year})")
    print(f"{'=' * 60}")

    df = load_parcels_eu(country_code)
    if limit > 0:
        df = df.head(limit)
        print(f"Limited to {len(df)} parcels")

    if len(df) == 0:
        print(f"  No parcels for {country_code}, skipping")
        return []

    combined_image, feature_names = build_s2_composite(ee, year=config.year)

    export_folder = f'{GEE_S2_EU_EXPORT_PREFIX}_{country_code}'
    task_prefix = f's2_{country_code}'
    id_col = COL_PARCEL_ID

    if interactive:
        bs = batch_size or 500
        emb_df = extract_interactive(
            ee, df, combined_image, feature_names,
            batch_size=bs, id_col=id_col)

        os.makedirs(EU_S2_EMBEDDINGS_DIR, exist_ok=True)
        out_path = os.path.join(EU_S2_EMBEDDINGS_DIR,
                                f's2_{country_code}.parquet')
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


def extract_s2_eu(ee, country_codes=None, interactive=False, limit=0,
                  batch_size=None, monitor=False):
    """提取所有或指定国家的 S2 特征。"""
    from config.country_registry import get_all_country_codes

    if country_codes is None:
        country_codes = get_all_country_codes()

    print(f"\n泛欧洲 S2 特征提取 — {len(country_codes)} 个国家/地区")

    all_tasks = []
    for code in country_codes:
        tasks = extract_s2_eu_country(
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
        description='Extract Sentinel-2 spectral features at parcel centroids')
    parser.add_argument('--mode', choices=['be', 'eu'], default='eu',
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
        extract_s2_eu(ee, country_codes=args.country,
                      interactive=args.interactive, limit=args.limit,
                      batch_size=args.batch_size, monitor=args.monitor)
    else:
        # 比利时模式
        from config.config import AE_YEAR
        df = load_parcels_be()
        if args.limit > 0:
            df = df.head(args.limit)
            print(f"Limited to {len(df)} parcels")

        combined_image, feature_names = build_s2_composite(ee, year=AE_YEAR)

        if args.interactive:
            batch_size = args.batch_size or 500
            emb_df = extract_interactive(ee, df, combined_image, feature_names,
                                         batch_size=batch_size)
            out_path = os.path.join(EU_S2_EMBEDDINGS_DIR,
                                    's2_be_interactive.parquet')
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            emb_df.to_parquet(out_path, index=False)
            print(f"\nSaved to {out_path}")
        else:
            batch_size = args.batch_size or GEE_BATCH_SIZE
            tasks = extract_batch_export(ee, df, combined_image,
                                         batch_size=batch_size,
                                         export_folder='s2_be_embeddings',
                                         task_prefix='s2_be')
            if args.monitor:
                monitor_tasks(ee, tasks)


if __name__ == '__main__':
    main()
