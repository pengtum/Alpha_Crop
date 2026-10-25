# Earth embeddings and local-label budgets for cross-border crop mapping

Research code accompanying **Earth embeddings improve cross-border crop mapping under limited local-label budgets**, by Jingwei Zhu, Peng Luo and Jonathan Li. Release prepared on 25 October 2026 for manuscript v19.

This repository contains the code, analysis plans and plotting workflow. Obtain crop reference records from **EuroCrops v11** and satellite inputs from **Google Earth Engine**. No parcel records, extracted features, trained models, predictions or result tables are bundled.

The main study rotates each of 11 countries as the target, using the other 10 as sources. It compares AlphaEarth (AE) and Sentinel-2 (S2), establishes local training samples from zero, and expands existing surveys. The separate one-source/one-target experiment is retained as historical code and is not part of the manuscript's results.

## Start here

```bash
# Standard-library-only integrity check; does not run experiments.
python3.12 tools/verify_release.py

# Create a separate workspace and configure paths in copied scripts.
python3.12 tools/configure_workspace.py --dest /absolute/path/alphacrops-work --gee-project YOUR_GEE_PROJECT
```

Then read [data preparation](docs/DATA_AND_PREPROCESSING.md), [experiment order](docs/EXPERIMENTS.md), and [reproduction notes](docs/REPRODUCIBILITY.md). [中文说明](README_zh.md) summarises the same workflow.

The configuration command preserves this archive. It changes paths in a new working copy and records every changed file. The original confirmatory runners deliberately check code hashes, data stamps and software versions. A new workspace or newly downloaded data needs a new, reviewed run registration before these strict runners can execute; the helper does not bypass those checks.

## Code layout

| Location | Purpose |
|---|---|
| `project/config/` | Country registry, crop classes, feature and spatial settings |
| `project/src/data/`, `project/src/extraction/` | EuroCrops preparation; GEE AE and S2 extraction and export merging |
| `project/label_expansion/build_dataset_v2.py` | Corrected class mapping and paired feature dataset |
| `project/label_expansion/confirm/` | Main spatial splits, two survey scenarios, training, analysis and independent checks |
| `project/label_expansion/guidance_G.py` | Representation used for sampling guidance, separated from prediction representation |
| `project/label_expansion/multilearner_G.py` | Multi-learner checks for the guidance experiment |
| `project/results/representation_budget_20261025_v1/` | Exploratory representation-versus-label-budget analysis and numerical verification; code only |
| `project/Manuscript/Paper_Alpha_Earth/0019_eleven_country_release_20261025/figure_sources/` | Current manuscript plotting code |
| `project/Manuscript/Paper_Alpha_Earth/0004_representation_labels_rse/figure_sources/` | Earlier plotting modules imported by current figures |
| `project/label_expansion/storyline/rse_figs/` | Preparation of figure input tables and earlier plot implementations |
| `local_code/verify/` | Supporting metric implementations, independent checks and development analyses |
| `project/label_expansion/single_source_v1/` | Historical single-source experiment, excluded from current RSE results |
| `project/nc_main/`, `project/nc_pilot/`, `project/scripts/`, `project/src/experiments/` | Other development experiments and analyses; see the experiment index |
| `third_party/` | Figure audit utilities with their original Apache-2.0 license |
| `tools/` | Release verification and workspace configuration |

Paths inside `project/` preserve the original project layout because relative imports and hash checks depend on it. `source_manifest.json` records the origin and SHA-256 of every archived source file. Python and shell sources are copied unchanged; notebook outputs, execution counts and embedded attachments have been removed.

## Environment

Use Python 3.12. Install the core workflow dependencies in a virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The requirements file describes core dependencies, not an exact lock for every historical script. Optional development scripts also use packages such as SHAP, statsmodels, UMAP and plotting helpers. The recorded versions enforced by the original confirmation run are in `environment-confirm-recorded.txt` and `project/label_expansion/confirm/confirm_split.py`.

GEE extraction requires an Earth Engine account and a Cloud project enabled for Earth Engine. Configure your own project ID and authenticate using the Earth Engine client. No account credentials are included. Exported CSV files must be downloaded and merged before local model training.

## Scientific scope

The main country roster is AT, BE, DE, EE, LT, LV, PT, RO, SE, SI and SK. Each target receives 50,000 foreign training labels, 5,000 from each source country. The original confirmatory label-response analysis uses nine countries, excluding PT and RO following development evaluations. Representation comparisons and the new exploratory representation-budget headline use all 11 countries. The nine-country budget result remains a sensitivity analysis.

The budget contrast compares AE with zero local training labels against S2 with each observed local-label count on a shared OA support. It concerns country-mean balanced accuracy in an offline evaluation, not monetary expenditure or a saving in independent calibration/validation labels. The S2 implementation is retained exactly as used: EVI/SAVI constants operate on stored scaled values, as disclosed in the manuscript appendix. Correcting those expressions would define a new feature baseline and require a new experiment.

## Figures and validation

Figure input tables and geographic inputs are generated or acquired locally; they are not distributed here. [The figure index](docs/FIGURES.md) identifies the plotting scripts and their dependencies. Original font settings and style modules are retained. No training, GEE export or figure regeneration is triggered by release verification.

The release was checked for file integrity, Python syntax, empty notebook outputs and excluded data formats. The workspace configuration helper was tested in a temporary directory. This packaging step did not rerun the scientific experiments or establish end-to-end reproducibility from a fresh GEE download.

## Citation and contact

Jingwei Zhu, Peng Luo and Jonathan Li. *Earth embeddings improve cross-border crop mapping under limited local-label budgets*. Manuscript, 2026. No publication DOI has been assigned in this release.

Correspondence: peng-luo@uiowa.edu; jli@geoai.ecnu.edu.cn.

See [RIGHTS.md](RIGHTS.md) for code rights and third-party attribution. AI assistance in writing and analytical coding is described in [AI_USE.md](AI_USE.md).
