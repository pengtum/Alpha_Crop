"""
Track 2: Extract AlphaEarth embeddings using actual parcel polygon geometries.

Computes mean and std of all pixels within each polygon using reduceRegion,
producing 129 features per parcel (64 mean + 64 std + 1 pixel_count).

Requires the parcel shapefile to be uploaded as a GEE Asset first.

Usage:
    python -m src.extraction.gee_polygon_extract --asset-id users/YOUR_USER/parcels_filtered

Prerequisites:
    1. Upload the filtered shapefile as a GEE Asset via:
       earthengine upload table --asset_id users/YOUR_USER/parcels_filtered \
           path/to/parcels_filtered.shp
    2. Or use the GEE web interface: code.earthengine.google.com -> Assets -> Upload
"""
import sys
import os
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.config import (
    AE_COLLECTION, AE_SCALE, AE_BANDS, AE_YEAR,
    GEE_EXPORT_FOLDER, GEE_PROJECT
)
from config.spatial_splits import ALL_FLEMISH_REGIONS


def init_gee():
    """Initialize Google Earth Engine."""
    import ee

    if GEE_PROJECT is None:
        print("ERROR: Set GEE_PROJECT in config/config.py first.")
        sys.exit(1)

    try:
        ee.Initialize(project=GEE_PROJECT)
        print(f"GEE initialized with project: {GEE_PROJECT}")
    except Exception as e:
        print(f"GEE initialization failed: {e}")
        sys.exit(1)

    return ee


def extract_polygon_embeddings(ee, asset_id):
    """Extract embeddings using polygon geometries from a GEE Asset."""
    print("\n" + "=" * 60)
    print("POLYGON-BASED EMBEDDING EXTRACTION")
    print("=" * 60)

    # Load the AlphaEarth image
    print(f"Loading AlphaEarth embeddings for year {AE_YEAR}...")
    image = (ee.ImageCollection(AE_COLLECTION)
             .filterDate(f'{AE_YEAR}-01-01', f'{AE_YEAR + 1}-01-01')
             .mosaic()
             .select(AE_BANDS))

    # Load parcels from GEE Asset
    parcels = ee.FeatureCollection(asset_id)
    total_count = parcels.size().getInfo()
    print(f"Asset parcels: {total_count}")

    band_names = AE_BANDS
    mean_names = [f'{b}_mean' for b in band_names]
    std_names = [f'{b}_std' for b in band_names]
    select_props = ['REF_ID', 'fid', 'pixel_count'] + mean_names + std_names

    def reduce_parcel(feature):
        """Compute mean and std of embeddings within parcel polygon."""
        geom = feature.geometry()

        # Mean of all bands
        mean_stats = image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geom,
            scale=AE_SCALE,
            maxPixels=1e6
        )

        # Std of all bands
        std_stats = image.reduceRegion(
            reducer=ee.Reducer.stdDev(),
            geometry=geom,
            scale=AE_SCALE,
            maxPixels=1e6
        )

        # Pixel count (from first band)
        count_stats = image.select(band_names[0]).reduceRegion(
            reducer=ee.Reducer.count(),
            geometry=geom,
            scale=AE_SCALE,
            maxPixels=1e6
        )

        # Build output properties - rename to avoid collisions
        props = {'pixel_count': count_stats.get(band_names[0])}
        for b in band_names:
            props[f'{b}_mean'] = mean_stats.get(b)
            props[f'{b}_std'] = std_stats.get(b)

        return feature.set(props).setGeometry(None)

    # Export by NUTS3 region to manage batch sizes
    export_folder = GEE_EXPORT_FOLDER + '_polygon'
    tasks = []

    for region in ALL_FLEMISH_REGIONS:
        region_parcels = parcels.filter(ee.Filter.eq('EC_NUTS3', region))
        reduced = region_parcels.map(reduce_parcel)

        task_name = f'ae_polygon_{region}'
        task = ee.batch.Export.table.toDrive(
            collection=reduced,
            description=task_name,
            folder=export_folder,
            fileFormat='CSV',
            selectors=select_props
        )
        task.start()
        tasks.append((task, task_name))
        print(f"  Started {task_name}")

    print(f"\n{'=' * 60}")
    print(f"SUBMITTED {len(tasks)} EXPORT TASKS")
    print(f"{'=' * 60}")
    print(f"Export folder: '{export_folder}'")
    print(f"Check status: https://code.earthengine.google.com/tasks")
    print(f"\nAfter completion, run:")
    print(f"  python -m src.extraction.merge_embeddings --mode polygon")

    return tasks


def main():
    parser = argparse.ArgumentParser(
        description='Extract AlphaEarth embeddings using parcel polygons')
    parser.add_argument('--asset-id', required=True,
                        help='GEE Asset ID of the uploaded parcels')
    parser.add_argument('--monitor', action='store_true',
                        help='Monitor tasks until completion')
    args = parser.parse_args()

    ee = init_gee()
    tasks = extract_polygon_embeddings(ee, args.asset_id)

    if args.monitor:
        print(f"\nMonitoring {len(tasks)} tasks...")
        while True:
            states = {}
            for task, name in tasks:
                state = task.status()['state']
                states[state] = states.get(state, 0) + 1

            status_str = ', '.join(f"{k}: {v}" for k, v in sorted(states.items()))
            print(f"  [{time.strftime('%H:%M:%S')}] {status_str}")

            if all(task.status()['state'] in ('COMPLETED', 'FAILED', 'CANCELLED')
                   for task, _ in tasks):
                break
            time.sleep(60)

        completed = sum(1 for t, _ in tasks if t.status()['state'] == 'COMPLETED')
        failed = sum(1 for t, _ in tasks if t.status()['state'] == 'FAILED')
        print(f"\nDone! Completed: {completed}, Failed: {failed}")


if __name__ == '__main__':
    main()
