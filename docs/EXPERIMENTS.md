# Experiment index and execution order

Paths below are relative to the configured `Alpha_Crops/` working directory. Commands identify entry points; data and the frozen-run requirements in [REPRODUCIBILITY.md](REPRODUCIBILITY.md) must be satisfied first.

| Stage | Scripts | Role in current paper |
|---|---|---|
| Data preparation | `src/data/prepare_parcels.py`, `src/extraction/gee_centroid_extract.py`, `gee_s2_extract.py`, `merge_embeddings.py`, `label_expansion/build_dataset_v2.py` | Shared inputs |
| Development spatial sets | `label_expansion/stageA.py stage0`, `label_expansion/guidance_G.py stage0` | Earlier test sets checked for overlap by the confirmation split builder |
| Confirmatory spatial design | `label_expansion/confirm/confirm_split.py` | 22 evaluation units; geographic buffering, foreign pool, target candidate pool and survey selections |
| Main prediction and label experiments | `label_expansion/confirm/confirm_run.py` | AE/S2, zero local labels, random additions, existing-survey expansion, target-only reference |
| Main analysis | `analyze_confirm.py`, `indep_confirm.py`, `crosscheck_confirm.py` in the same directory | Matched-OA metrics, hypotheses, label curves, placement, label equivalents and cost scenarios |
| Sampling guidance | `label_expansion/guidance_G.py`, `analyze_G.py` | Separate predictive and acquisition representations |
| Learner sensitivity | `label_expansion/multilearner_G.py`, `analyze_ML.py`, `analyze_ML_rev3.py` | Additional classifiers and guidance comparisons |
| Representation versus budget | `results/representation_budget_20261025_v1/analyze.py`, `verify.py` | Exploratory analysis of existing confirmation probabilities; no training |
| Figure inputs and plotting | `label_expansion/storyline/rse_figs/`, current manuscript figure sources | Dataset imbalance, diagrams, confusion and accuracy, survey and cost plots |

## Main confirmation experiment

```bash
# Earlier development stage0 outputs must exist for overlap checks.
python -B label_expansion/stageA.py stage0
python -B label_expansion/guidance_G.py stage0

# Original formal run interface (requires matching registration and inputs).
python -B label_expansion/confirm/confirm_split.py --root "$ALPHA_LOCAL/results/confirm_v1"
python -B label_expansion/confirm/confirm_run.py --root "$ALPHA_LOCAL/results/confirm_v1" --learner lgb --jobs 5 --threads 2
python -B label_expansion/confirm/confirm_run.py --root "$ALPHA_LOCAL/results/confirm_v1" --learner rf --jobs 4 --threads 2
python -B label_expansion/confirm/analyze_confirm.py --root "$ALPHA_LOCAL/results/confirm_v1" --structure-check
python -B label_expansion/confirm/analyze_confirm.py --root "$ALPHA_LOCAL/results/confirm_v1"
```

Run independent recomputation and cross-checks using `indep_confirm.py --help` and `crosscheck_confirm.py --help`. Formal analysis refuses to overwrite existing results. Do not point new runs at an existing study archive.

Each target is paired with all other ten countries as the foreign training pool. No-local-label random training counts are 0, 50, 100, 200 and 400. The acquisition-strategy comparison from zero uses additions of 50 or 200. Existing surveys start with 100 clustered labels and add 100 or 300, yielding total inventories of 200 or 400. New-label counts and inventory totals are distinct quantities.

`CONFIRM_PLAN_v5.md` and its implementation addendum record the protocol. The original H1 label-response test uses nine countries; H2 zero-label representation comparison uses all eleven. PT/RO exclusion for H1 followed development results with low zero-label OA anchors, before the confirmation experiment. All-country estimates remain available.

## Guidance and learner checks

```bash
python -B label_expansion/guidance_G.py run --tier 1 --jobs 5
python -B label_expansion/analyze_G.py
python -B label_expansion/multilearner_G.py --help
```

`analyze_G.py` performs analysis directly and does not implement a `--help` switch; inspect its input/output paths before execution. Read `ANALYSIS_PLAN_guidance.md` and revisions before selecting tiers or learner runs. Oracle and diagnostic conditions in later tiers are not operational label acquisition policies. `--unfrozen` options in historical test scripts are for controlled development, not substitutes for the formal registration.

## Exploratory budget analysis

After the confirmation outputs exist, `results/representation_budget_20261025_v1/analyze.py` reads the saved probabilities and builds a shared OA support across AE/S2, every budget and draw. `verify.py` independently reconstructs the contrasts. Neither script trains models. The main summary uses eleven countries; the original nine-country set is retained for sensitivity. The 200-label and 400-label comparisons must both be retained, including their uncertainty and negative country effects.

## Historical work

`label_expansion/single_source_v1/` implements one-source/one-target transfer. Its debug attempt is archived under `results/single_source_20261009/debug/`. It is not included in the current paper's evidence. Other early baselines and cross-country experiments are in `src/experiments/`, `scripts/`, `nc_main/` and `nc_pilot/`. `local_code/verify/` also includes development, scratch and independent review code. They are retained for provenance, not presented as additional confirmatory tests.

The current analysis imports `local_code/verify/step0/partB/step0B.py`, `partB2/step0B2.py` and supporting modules. Preserve their directory structure; do not discard the entire verification tree as unused scratch code.
