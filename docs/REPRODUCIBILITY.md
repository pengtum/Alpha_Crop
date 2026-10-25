# Reproduction and provenance

## Archive and working copy

`project/` and `local_code/` retain research sources in their original layout. The source manifest provides SHA-256 checksums. The only source-content transformation is removing notebook outputs and embedded attachments. No scientific algorithms or experiment outputs were changed during packaging.

The scripts originally ran across a synced project directory and a separate local data/results directory. Some paths are relative; others are absolute historical paths. Setting `ALPHA_LOCAL` affects `common.py` but does not replace every path in the project.

`tools/configure_workspace.py` creates a new directory containing `Alpha_Crops/`, `AlphaCrops_local/` and `nature-skills-main/`. It rewrites those known path prefixes in copied Python/shell/notebook code and configures the GEE project ID. Its `configuration_report.json` lists transformations and residual machine-specific paths. The source archive stays unchanged.

## Frozen-run checks

The original confirmation pipeline checks analysis-plan hashes, dependency hashes, its own registration, a dataset path/size/mtime stamp, package versions, split hashes and training row hashes. These are intentional controls for the original run. A fresh path, modified file, changed dataset or package environment will fail one or more controls.

The configuration helper does not update the historical registration, change the source stamp, rewrite stored results or suppress an assertion. A new experiment needs its own registered environment, data identity and code hashes, followed by fresh splits/training/analysis. Do not relabel that new run as the original frozen experiment. Exact numerical reproduction also requires identical source records, feature exports, ordering and random seeds.

## Recorded software

`environment-confirm-recorded.txt` transcribes the versions enforced in `confirm_split.py`. These are historical runtime values, not a promise that all versions are available from every package index. The root `requirements.txt` describes the core dependency set for code inspection and setup. Optional historical experiments can need additional libraries, available from their imports.

## Scientific qualifications retained

- S2 EVI/SAVI formulas retain the stored-value implementation used for the paper. A corrected reflectance baseline is a separate experiment.
- Parcel IDs, order and feature column order determine indices in the saved splits. Do not sort or reset an intermediate table differently.
- A fresh LV extraction need not reproduce the incompatible batch excluded from the original cohort.
- Matched-OA curves use test labels for offline evaluation. They are not a deployable threshold-selection procedure without independent calibration.
- The eleven-country budget comparison is exploratory. Its uncertainty and all-country heterogeneity remain part of the result.
- Legacy comments and launcher defaults can describe earlier subsets. The manuscript, frozen confirmation plans and current experiment index define the current study.

## Release checks

Run `python3.12 tools/verify_release.py` before or after copying the folder. This checks source checksums, Python compilation, notebook output removal, forbidden binary/data extensions and common embedded credential patterns. It does not import training scripts, access GEE or validate a scientific result. `VALIDATION.json` records the packaging checks. No experimental data are required for them.
