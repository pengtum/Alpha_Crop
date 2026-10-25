# Figure source index

The current plotting directory is `project/Manuscript/Paper_Alpha_Earth/0019_eleven_country_release_20261025/figure_sources/`.

| Figure content | Current script |
|---|---|
| Dataset, parcel reference example and imbalance | `plot_dataset.py` |
| Transfer rotation, two acquisition scenarios, representation/label concept | `plot_concept.py` |
| OA/BA and matched-OA evaluation | `fig2_evaluation.py` |
| Random local-label response | `fig3_labels_zero.py` |
| Existing-survey expansion | `plot_existing_survey.py` |
| Label equivalents and cost scenarios | `fig5_exchange_cost.py` |
| AE/S2 representation comparison | `fig6_representation.py` |
| Spatial and prediction diagnostics | `plot_diagnostics.py` |
| Sampling strategy supplement | `figS4_strategies.py` |
| Style and layout audits | `nstyle.py`, `layout_readability.py` |

Script names retain earlier figure numbers; the final manuscript determines numbering. `figure_contract.json` records layout expectations.

Generate analysis tables using the confirmatory/guidance analyses first. `label_expansion/storyline/rse_figs/prep_data.py` prepares figure tables and geographic inputs; `revision_20261007/prep_diagnostics.py` prepares additional diagnostics. Current plots also reference earlier `0004_representation_labels_rse/figure_sources/` modules and their generated `source_data/`. These code dependencies are bundled; their generated data are not. Preserve those relative directories or configure a working copy.

The parcel example is built by `0004_representation_labels_rse/figure_sources/parcel_reference_20261011/prepare_example.py` and `plot_figures.py`. Map and example inputs must be acquired/generated before `plot_dataset.py` or `plot_concept.py` can run. Some intermediate preparation steps were manual in the original workflow; the source archive does not claim an automated single-command rebuild of every figure from a new download.

Figure styling remains in the original Python modules, including fonts, colours and panel sizes. The required audit helpers from nature-skills are included under `third_party/` with their original license. No generative-image service or API credential is needed for the current data figures.
