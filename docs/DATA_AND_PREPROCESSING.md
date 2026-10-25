# Data sources and preprocessing

## External sources

| Input | Source | Configuration |
|---|---|---|
| Crop parcels and harmonised crop labels | EuroCrops v11, https://doi.org/10.5281/zenodo.14094196 | `project/config/country_registry.py` records archive names, region codes and years |
| Annual AE embeddings | GEE `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL` | `project/config/config.py`; 64 bands, A00–A63 |
| Sentinel-2 surface reflectance | GEE `COPERNICUS/S2_SR_HARMONIZED` | Same configuration; cloud filtering, SCL masking and seasonal summaries in `gee_s2_extract.py` |
| Boundaries for diagnostics and maps | Sources referenced by `label_expansion/assign_nuts.py`, `storyline/rse_figs/prep_data.py`, and the parcel example scripts | Acquire locally before plotting |

EuroCrops labels originate from agricultural reference records. Their acquisition and verification entail ground-reference costs; they are not labels newly assigned by visually interpreting satellite images for this study. National coverage and source years differ. Use the region registry and manuscript data table rather than interpreting a legacy comment about universal 2021 alignment as a verified property of every record.

## Prepare parcel records

In the configured working copy, place the downloaded country archives in `Alpha_Crops/Data_Crop/`. From `Alpha_Crops/`:

```bash
python -m src.data.prepare_parcels --mode eu --country AT BE_VLG BE_WAL DE_LS DE_NRW EE LT LV PT RO SE SI SK --max-per-country 50000
```

The script standardises fields, removes invalid/non-agricultural records, computes area and geographic centroids, and saves `data/eu_processed/parcels_eu.parquet`. The command uses the archived launcher's 50,000-record cap, applied separately to each registered region with class-stratified sampling and seed 42. Despite its option name, BE and DE each contain two regional inputs. `--max-per-country 0` instead processes full inventories and defines a different extraction cohort. Preserve parcel IDs and row ordering. The final paired-feature dataset, rather than the initial parcel count, defines the analytical cohort; exact identity still requires matching the original inputs.

## Extract AE and S2 features

```bash
earthengine authenticate
python -m src.extraction.gee_centroid_extract --mode eu --country AT BE_VLG BE_WAL DE_LS DE_NRW EE LT LV PT RO SE SI SK
python -m src.extraction.gee_s2_extract --mode eu --country AT BE_VLG BE_WAL DE_LS DE_NRW EE LT LV PT RO SE SI SK
```

These are live GEE jobs. The scripts use each region's configured year and export parcel batches to Drive. Download the completed exports and use `merge_embeddings.py --help` to point to the export directory. Typical merging commands are:

```bash
for region in AT BE_VLG BE_WAL DE_LS DE_NRW EE LT LV PT RO SE SI SK; do
  python -m src.extraction.merge_embeddings --mode eu --source ae --country "$region" --csv-dir "/absolute/path/ae-exports/$region"
  python -m src.extraction.merge_embeddings --mode eu --source s2 --country "$region" --csv-dir "/absolute/path/s2-exports/$region"
done
# Optional combined tables for the earlier baseline workflow:
python -m src.extraction.merge_embeddings --mode eu --source ae --merge-all
python -m src.extraction.merge_embeddings --mode eu --source s2 --merge-all
```

AE uses neighbourhood mean and standard deviation for 64 bands (128 features). S2 uses 84 annual and seasonal summaries followed by the same two spatial summaries (168 features). The implementation contains its exact cloud mask, quarterly periods, bands, index expressions and spatial kernel. The S2 EVI/SAVI constants are applied to stored image values; this is the published implementation, not the standard reflectance-scaled formulation. Do not silently change it when reproducing the published comparison.

## Construct the paired analytical dataset

Set `ALPHA_LOCAL` to the configured `AlphaCrops_local` directory. Under its `data/`, supply:

```text
parcels_eu.parquet
ae/embeddings_{REGION}.parquet
s2/s2_{REGION}.parquet
```

The merge outputs originate in `Alpha_Crops/data/eu_processed/embeddings/` and `s2_embeddings/`; copy or link them into this layout. Then run from `Alpha_Crops/`:

```bash
python -B label_expansion/build_dataset_v2.py
```

The builder applies the revised Tier-B mapping, excludes specified non-crop/fallow records, reconciles the BE_WAL `parcel_idx` alias, and removes incomplete/incompatible feature rows. In the original extraction, one LV S2 batch used an incompatible column structure and was excluded. A fresh extraction may not reproduce that failure, so its cohort can differ. `common.load_dataset()` retains parcels with both AE and S2 and resets their index. The published paired cohort has 611,519 parcels; numerical identity requires the original cohort, class definitions, ordering, extraction products and split provenance.

No data are supplied by this repository. GEE availability and updated source archives can change regenerated inputs. Record checksums and software versions for any new run.
