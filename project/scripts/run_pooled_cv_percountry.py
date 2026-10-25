"""Pooled cross-validation, broken down by country.

exp1 stores pooled CV as a single number per feature set, which makes the
broad-training row of the protocol 2x2 impossible to test per country: the
"border cost" for a held-out country becomes (constant - LOCO_c), which is
algebraically the LOCO comparison shifted, not an independent measurement.

This script re-runs the SAME pooled CV — identical splits, identical trainer,
identical seed — and additionally records balanced accuracy on each country's
share of the held-out test fold. Overall numbers must reproduce exp1
(AE 68.5 / S2 67.7 / Fusion 71.5) or the run is not comparable.

Writes results/experiments/exp1_id_baseline/percountry_tier_b_nocoords_2021.json
incrementally, so the run is resumable.
"""
import sys, os, json, time
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, recall_score
from src.experiments.experiment_utils import (
    load_eu_features, prepare_xy, filter_coords, get_trainer,
    RANDOM_STATE, COUNTRIES_2021)
from config.config import COL_COUNTRY

OUT = 'results/experiments/exp1_id_baseline/percountry_tier_b_nocoords_2021.json'
PRED_DIR = 'results/experiments/exp1_id_baseline/pooled_cv_predictions'
os.makedirs(PRED_DIR, exist_ok=True)
N_REPEATS = 5
FEATS = ['ae', 's2', 'fusion']
res = json.load(open(OUT)) if os.path.exists(OUT) else {}
t_start = time.time()


def split(X, y, fold_idx):
    """Exactly the split used by src/experiments/run_id_baseline.py."""
    skf = StratifiedKFold(n_splits=5, shuffle=True,
                          random_state=RANDOM_STATE + fold_idx)
    s = list(skf.split(X, y))
    test_idx, val_idx = s[0][1], s[1][1]
    train_idx = np.concatenate([s[i][1] for i in range(2, 5)] + [s[0][0]])
    train_idx = np.setdiff1d(train_idx, np.concatenate([test_idx, val_idx]))
    return train_idx, val_idx, test_idx


for ft in FEATS:
    if all(f'{ft}|{r}' in res for r in range(N_REPEATS)):
        print(f'[{ft}] all {N_REPEATS} repeats already done, skipping', flush=True)
        continue
    t0 = time.time()
    df, fcols, cnames = load_eu_features(ft, 'b', countries=COUNTRIES_2021)
    fcols = filter_coords(fcols)
    X, y = prepare_xy(df, fcols, 'b')
    ctry = df[COL_COUNTRY].values
    del df
    print(f'[{ft}] loaded {X.shape[0]:,} x {X.shape[1]} in {time.time()-t0:.0f}s',
          flush=True)
    trainer = get_trainer('lgbm')
    for r in range(N_REPEATS):
        key = f'{ft}|{r}'
        if key in res:
            print(f'[{ft}] repeat {r+1}/{N_REPEATS} cached', flush=True)
            continue
        t1 = time.time()
        tr, va, te = split(X, y, r)
        model = trainer(X[tr], y[tr], X[va], y[va], len(cnames))
        yp = model.predict(X[te])
        overall = balanced_accuracy_score(y[te], yp) * 100
        # per-sample test predictions, so ANY support threshold can be applied later
        # without re-fitting. This is what the first attempt at this script lacked.
        pd.DataFrame({'country': ctry[te], 'y_true': y[te], 'y_pred': yp}).to_parquet(
            f'{PRED_DIR}/preds_{ft}_r{r}.parquet', index=False)
        per_country = {}
        for c in COUNTRIES_2021:
            m = ctry[te] == c
            if m.sum() >= 10:
                yt, yq = y[te][m], yp[m]
                cls = np.unique(yt)
                rec = recall_score(yt, yq, labels=cls, average=None, zero_division=0)
                per_country[c] = dict(
                    balanced_accuracy=float(balanced_accuracy_score(yt, yq) * 100),
                    n_test=int(m.sum()),
                    n_classes_present=int(len(cls)),
                    per_class={cnames[k]: dict(recall=float(rr * 100),
                                               support=int((yt == k).sum()))
                               for k, rr in zip(cls, rec)})
        res[key] = dict(feature_type=ft, repeat=r, n_train=int(len(tr)),
                        n_test=int(len(te)), overall_balanced_accuracy=float(overall),
                        per_country=per_country)
        json.dump(res, open(OUT, 'w'), indent=1)
        el = time.time() - t1
        print(f'[{ft}] repeat {r+1}/{N_REPEATS}  overall BA {overall:5.2f}%  '
              f'{el/60:.1f} min  (elapsed {(time.time()-t_start)/60:.0f} min)', flush=True)
    del X, y, ctry

print('\n=== SANITY CHECK vs exp1 ===', flush=True)
e1 = json.load(open('results/experiments/exp1_id_baseline/tier_b_nocoords_2021.json'))
for ft in FEATS:
    v = [res[f'{ft}|{r}']['overall_balanced_accuracy'] for r in range(N_REPEATS)
         if f'{ft}|{r}' in res]
    if v:
        ref = e1[ft]['balanced_accuracy']['mean'] * 100
        print(f'  {ft.upper():<7} this run {np.mean(v):5.2f} +- {np.std(v):.2f}   '
              f'exp1 {ref:5.2f}   diff {np.mean(v)-ref:+.2f} pp', flush=True)
print(f'\nDONE in {(time.time()-t_start)/60:.0f} min -> {OUT}', flush=True)
