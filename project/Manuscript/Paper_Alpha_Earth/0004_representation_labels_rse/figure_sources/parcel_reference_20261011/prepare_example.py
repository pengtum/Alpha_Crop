"""选取可追溯的真实地块示例；不修改原始数据或提取流水线。"""
from pathlib import Path
import sys
import json
import hashlib
import argparse
import numpy as np
import pandas as pd
import geopandas as gpd
from pyproj import Transformer
from shapely.geometry import box

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path.insert(0, str(ROOT))
from config.class_definitions import get_tier_b_label


def main():
    args = argparse.ArgumentParser()
    args.add_argument('--download', action='store_true')
    args.add_argument('--archive', type=Path)
    opt = args.parse_args()
    source = HERE / 'source_data'
    target_id = 'BE_VLG_456196'
    parcels = pd.read_parquet(ROOT / 'data/eu_processed/parcels_eu.parquet')
    selected = parcels.set_index('parcel_id').loc[target_id]
    forward = Transformer.from_crs(4326, 31370, always_xy=True)
    cx, cy = forward.transform(selected.longitude, selected.latitude)
    extent = [cx - 310, cy - 310, cx + 310, cy + 310]
    raw_path = ROOT / 'BE_VLG_2021/BE_VLG_2021_EC21.shp'
    if opt.archive:
        from zipfile import ZipFile
        import shutil
        cache = Path('/private/tmp/rse_be_vlg_source')
        cache.mkdir(exist_ok=True)
        assert shutil.disk_usage(cache).free > 3_000_000_000
        with ZipFile(opt.archive) as z:
            for ext in ['shp', 'shx', 'dbf', 'prj', 'cpg']:
                filename = f'BE_VLG_2021_EC21.{ext}'
                dst = cache / filename
                item = z.getinfo(f'BE_VLG_2021/{filename}')
                if not dst.exists() or dst.stat().st_size != item.file_size:
                    with z.open(item) as src, dst.open('wb') as out:
                        shutil.copyfileobj(src, out, 2**20)
        raw_path = cache / 'BE_VLG_2021_EC21.shp'
        print('Public source archive extracted to temporary cache.', flush=True)
    raw = gpd.read_file(raw_path, bbox=tuple(extent))
    raw['analysis_class'] = raw.EC_hcat_n.map(get_tier_b_label)
    raw['in_selected_record'] = False
    geo = raw.to_crs(4326)
    # 重现已使用的 WGS84 几何质心，仅核验，不更改流水线。
    centroids = geo.geometry.apply(lambda g: g.centroid)
    delta = np.hypot(centroids.x - selected.longitude,
                     centroids.y - selected.latitude)
    idx = delta.idxmin()
    assert delta.loc[idx] < 1e-8, delta.loc[idx]
    assert raw.loc[idx, 'EC_hcat_n'] == selected.EC_hcat_n
    assert raw.loc[idx, 'analysis_class'] == selected.label_tier_b
    raw.loc[idx, 'in_selected_record'] = True
    useful = ['fid', 'REF_ID', 'GWSNAM_H', 'GWSCOD_H', 'EC_trans_n',
              'EC_hcat_n', 'EC_hcat_c', 'analysis_class', 'in_selected_record',
              'geometry']
    raw[useful].to_file(source / 'parcels.geojson', driver='GeoJSON')
    raw.drop(columns='geometry').to_csv(source / 'parcel_records.csv', index=False)
    nparts = raw.geometry.apply(lambda g: box(*extent).covers(g)).sum()
    meta = {'parcel_id': target_id, 'reference_year': 2021,
            'longitude': float(selected.longitude), 'latitude': float(selected.latitude),
            'centroid_lambert72': [cx, cy], 'extent_lambert72': extent,
            'area_ha': float(selected.area_ha), 'source_record': raw.loc[idx, useful[:-1]].to_dict(),
            'displayed_parcels': len(raw), 'fully_visible_parcels': int(nparts),
            'selection': 'Geographic example in the Tielt area; no performance-based selection. All raw source polygons intersecting a 620 m square are retained.',
            'centroid_match_error_degrees': float(delta.loc[idx]),
            'basemap': {'service': 'https://geo.api.vlaanderen.be/OMW/wms',
                        'layer': 'OMWRGB21VL', 'year': 2021,
                        'agency': 'Digitaal Vlaanderen',
                        'use': 'Geographic context only; not the source of labels or classifier input',
                        'crs': 'EPSG:31370', 'native_resolution_m': 0.25,
                        'requested_image_size_px': [2048, 2048],
                        'requested_resolution_m': 620 / 2048,
                        'processing': 'WMS-rendered RGB; no brightness, contrast or local image edits'}}
    for modality, path in [('ae', 'embeddings/embeddings_BE_VLG.parquet'),
                            ('s2', 's2_embeddings/s2_BE_VLG.parquet')]:
        local = Path('/Users/perng_mini/AlphaCrops_local/data') / modality / Path(path).name
        data = pd.read_parquet(local if local.exists() else ROOT / 'data/eu_processed' / path)
        row = data[data.parcel_id == target_id]
        assert len(row) == 1
        cols = [c for c in row if c.endswith(('_mean', '_stdDev'))]
        assert len(cols) == (128 if modality == 'ae' else 168), len(cols)
        assert np.isfinite(row[cols].to_numpy(dtype=float)).all()
        row.to_csv(source / f'selected_{modality}_features.csv', index=False)
        meta[f'{modality}_features_verified'] = len(cols)
    (source / 'metadata.json').write_text(json.dumps(meta, indent=2, default=lambda v: v.item()) + '\n')
    print(json.dumps(meta, indent=2, default=lambda v: v.item()))
    if opt.download:
        import requests
        params = {'SERVICE': 'WMS', 'VERSION': '1.1.1', 'REQUEST': 'GetMap',
                  'LAYERS': 'OMWRGB21VL', 'STYLES': '', 'SRS': 'EPSG:31370',
                  'BBOX': ','.join(str(v) for v in extent), 'WIDTH': 2048,
                  'HEIGHT': 2048, 'FORMAT': 'image/png'}
        response = requests.get(meta['basemap']['service'], params=params, timeout=45)
        response.raise_for_status()
        assert response.headers.get('Content-Type', '').startswith('image/'), response.text[:200]
        (source / 'orthophoto_2021.png').write_bytes(response.content)
        meta['basemap']['request_url'] = response.url
        meta['basemap']['sha256'] = hashlib.sha256(response.content).hexdigest()
        (source / 'metadata.json').write_text(json.dumps(meta, indent=2, default=lambda v: v.item()) + '\n')
        print('Official orthophoto saved:', len(response.content), 'bytes')


if __name__ == '__main__':
    main()
