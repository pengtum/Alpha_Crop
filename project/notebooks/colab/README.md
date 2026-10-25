# Colab notebooks for Phase 2 experiments

These two notebooks complete the Phase 2 robustness/visualization tasks for the RSE paper without requiring a local LightGBM/UMAP setup.

## Prerequisites

1. **Add the `Alpha_Crops` shared folder to your Drive**
   - Right-click the shared folder → **Add shortcut to Drive** → place at `/MyDrive/Alpha_Crops`.
   - The notebooks auto-detect this path. If you place the shortcut elsewhere (e.g. `MyDrive/Research/Alpha_Crops`), edit `PROJECT_ROOT` in the second code cell.
2. **Open each notebook in Colab**
   - In Drive, double-click the `.ipynb` file → it opens in Colab.
   - Or upload from `Alpha_Crops/notebooks/colab/` to your local Colab.

The notebooks read from `data/eu_processed/features/{ae,s2}/features_{train,val,test}.parquet` and write outputs back into the manuscript folder.

---

## Notebook 1: `01_umap_visualization.ipynb`

**Generates** `fig11_umap_*.{pdf,png}` (7 files) → main paper figure is `fig11_umap_ae_vs_s2_grid.pdf`, used in Section 5.8.3.

**Configurable parameters (cell 4):**
- `SAMPLE_PER_COUNTRY = 5000` — UMAP scales poorly past ~50k samples; this gives 11 × 5,000 = 55,000 points per projection.
- `SEED = 42` — for reproducibility.

**Runtime:** ~10–15 min on free Colab CPU.

**What to verify after the run:**
- AE panel coloured by country shows tight, well-separated clusters.
- AE panel coloured by crop shows diffuse / overlapping clusters.
- S2 panel coloured by country is more diffuse than AE's; the crop colouring is relatively more visible.
- This visual contrast is the geometric counterpart of the CKA<sub>crop</sub> = 0.244 (AE) vs 0.056 (S2) finding.

---

## Notebook 2: `02_spatial_block_cv.ipynb`

**Generates** `tables/spatial_block_cv_per_country.csv` and `figures/main/fig12_spatial_vs_random_cv.{pdf,png}`.

**Configurable parameters (cell 4):**
- `COUNTRIES_TO_RUN` — defaults to a representative 4 (DE, PT, EE, SK), ~30 min CPU. Switch to `ALL_COUNTRIES` (11) for the full table, ~3 hr CPU.
- `FEATURES_TO_RUN` — defaults to `['ae', 's2']`. Add `'fusion'` for completeness if budget allows (+50% runtime).
- `GRID_RESOLUTION = 5` — 5×5 spatial grid. Increase to 7 for finer blocking on countries with many parcels (more conservative, but produces smaller blocks).

**Runtime estimates (Colab CPU, ~50k parcels per country):**

| Scope | Fits | Approx time |
|---|---|---|
| 4 countries × 2 features | 80 | 30 min |
| 11 countries × 2 features | 220 | 1.5 hr |
| 11 countries × 3 features (incl. fusion) | 330 | 3 hr |

**What the result tells you:**
- `random_cv_ba` is the original Exp 0 BA (within-country, random folds).
- `spatial_cv_ba` is the new BA under non-overlapping spatial blocks.
- `autocorrelation_inflation_pp = (random − spatial) × 100` quantifies how much within-country accuracy was inflated by spatial autocorrelation.

**Decision tree for paper integration** (after the run):
- **Mean inflation < 3 pp** → mention in Limitations (1 sentence): the original numbers are essentially correct.
- **Mean inflation 3–5 pp** → add a Section 5 robustness paragraph reporting spatial-CV alongside random-CV; update headline ID numbers if reviewers ask.
- **Mean inflation > 5 pp** → re-report Exp 0 with spatial-block CV as the headline; the paper's *cross-border* (LOCO, pairwise) experiments are unaffected because they already use disjoint countries.

---

## Tips for working in Colab

- **Don't close the tab during a long fit.** Use the keep-alive trick or run on Colab Pro if you need >2 hours uninterrupted.
- **If the parquet reads are slow** (Drive-mounted reads can be slow for large files), copy the relevant features to local Colab disk first:
  ```python
  !cp -r "{FEATURES_DIR}" /content/local_features
  FEATURES_DIR = Path('/content/local_features')
  ```
  This trades startup time for faster repeated reads.
- **Output is written back to Drive** (the manuscript folder). No need to download anything — re-compile the LaTeX paper in Overleaf and the new figures appear.

---

## After running both notebooks

1. The LaTeX paper already references `fig11_umap_ae_vs_s2_grid.pdf` (Section 5.8.3). Once the file exists, re-compile.
2. For `fig12_spatial_vs_random_cv.pdf`, the figure is *not* yet inserted into the LaTeX — decide based on inflation magnitude (see decision tree above) whether to add it as a Section 5 robustness subsection or just cite the result in Limitations §5.
3. Both CSV outputs (`spatial_block_cv_per_country.csv`, the existing `strategy_wilcoxon_*.csv`) belong in the `exp6_borrowing_strategies/tables/` directory and are useful as supplementary materials for submission.
