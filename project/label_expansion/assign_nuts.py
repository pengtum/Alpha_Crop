"""
为 dataset_v2 的每个地块分配 NUTS 2021 行政区（NUTS3，以及由编码前缀得到的 NUTS2/NUTS1）。

边界：Eurostat GISCO NUTS_RG_01M_2021_4326_LEVL_3.geojson（1:1M，WGS84），放在 DATA_DIR/boundaries/。
方法：地块中心点落在哪个 NUTS3 多边形内；1:1M 边界对海岸线做了简化，少数点会落在所有多边形之外，
这些点以及落进邻国多边形的点，在本国的 NUTS3 中取最近的一个（按 ETRS89-LAEA 投影下的距离）。
输出：DATA_DIR/parcel_nuts.parquet（parcel_id, country, nuts3, nuts2, nuts1, join_method）
"""
import os

import geopandas as gpd
import pandas as pd

from common import DATA_DIR, COUNTRIES

NUTS_PATH = os.path.join(DATA_DIR, 'boundaries', 'NUTS_RG_01M_2021_4326_LEVL_3.geojson')
OUT = os.path.join(DATA_DIR, 'parcel_nuts.parquet')


def main():
    df = pd.read_parquet(os.path.join(DATA_DIR, 'dataset_v2.parquet'),
                         columns=['parcel_id', 'country', 'longitude', 'latitude'])
    nuts = gpd.read_file(NUTS_PATH)
    nuts = nuts[nuts['CNTR_CODE'].isin(COUNTRIES)][['NUTS_ID', 'CNTR_CODE', 'geometry']]
    pts = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['longitude'], df['latitude']), crs=4326)

    j = gpd.sjoin(pts, nuts, how='left', predicate='within')
    j = j[~j.index.duplicated(keep='first')]
    out = df[['parcel_id', 'country']].copy()
    out['nuts3'] = j['NUTS_ID'].to_numpy()
    out['nuts_cntr'] = j['CNTR_CODE'].to_numpy()
    out['join_method'] = 'within'

    # 落在所有多边形之外、或落进邻国多边形（国境线附近的边界简化）的点，都改取本国最近的 NUTS3
    miss = out['nuts3'].isna() | (out['nuts_cntr'] != out['country'])
    if miss.any():
        laea = 3035
        p_miss = pts[miss.to_numpy()].to_crs(laea)
        n_laea = nuts.to_crs(laea)
        for c, g in p_miss.groupby('country'):
            cand = n_laea[n_laea['CNTR_CODE'] == c]
            near = gpd.sjoin_nearest(g, cand, how='left')
            near = near[~near.index.duplicated(keep='first')]
            out.loc[near.index, 'nuts3'] = near['NUTS_ID'].to_numpy()
            out.loc[near.index, 'nuts_cntr'] = c
            out.loc[near.index, 'join_method'] = 'nearest'

    mismatch = out['nuts_cntr'] != out['country']
    print(f'parcels {len(out):,}; within {int((out["join_method"] == "within").sum()):,}; '
          f'nearest {int((out["join_method"] == "nearest").sum()):,}; country mismatch {int(mismatch.sum()):,}')
    if mismatch.any():
        print(out[mismatch].groupby(['country', 'nuts_cntr']).size().to_string())
    out['nuts2'] = out['nuts3'].str[:4]
    out['nuts1'] = out['nuts3'].str[:3]
    print(out.groupby('country').agg(nuts3=('nuts3', 'nunique'), nuts2=('nuts2', 'nunique'),
                                      nearest=('join_method', lambda s: int((s == 'nearest').sum()))).to_string())
    out.drop(columns='nuts_cntr').to_parquet(OUT, index=False)
    print('saved', OUT)


if __name__ == '__main__':
    main()
