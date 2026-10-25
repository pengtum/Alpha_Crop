"""Linear probe: same features, same splits, same seed -- only the head changes.

Purpose. LightGBM splits on individual features, which suits Sentinel-2's 168
engineered bands and indices (each dimension is meaningful) and may systematically
under-use AlphaEarth's 128-d dense embedding (no dimension is meaningful on its
own; information lives in linear combinations). A linear probe has the OPPOSITE
inductive bias, so it is the natural test of whether "AlphaEarth is
indistinguishable from Sentinel-2" is an artefact of the classifier head.

Absolute accuracy will be lower than LightGBM's for BOTH feature sets. That is
expected. The quantity of interest is the AE - S2 difference, not the level.

Standardisation is fitted on the training split only; fitting it on the full
matrix would leak test information into the scaler.

Writes results/experiments/exp10_linear_probe/ incrementally (resumable), and
saves per-sample test predictions so any support threshold or metric can be
recomputed later without re-fitting.
"""
import sys, os, json, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, recall_score
from src.experiments.experiment_utils import (
    load_eu_features, prepare_xy, filter_coords, RANDOM_STATE, COUNTRIES_2021)
from config.config import COL_COUNTRY

OUT_DIR = 'results/experiments/exp10_linear_probe'
PRED_DIR = f'{OUT_DIR}/predictions'
os.makedirs(PRED_DIR, exist_ok=True)
OUT = f'{OUT_DIR}/linear_probe_tier_b_nocoords_2021.json'
MAX_ITER = int(os.environ.get('PROBE_MAX_ITER', 200))
SOLVER = os.environ.get('PROBE_SOLVER', 'lbfgs')
FEATS = ['ae', 's2', 'fusion']
res = json.load(open(OUT)) if os.path.exists(OUT) else {}
t0 = time.time()


def fit_eval(Xtr, ytr, Xte, yte, ctry_te, cnames):
    sc = StandardScaler().fit(Xtr)                    # train-only, no leakage
    # The L2-regularised multinomial objective is strictly convex, so every
    # converged solver returns the SAME optimum; only convergence matters, not a
    # common solver. lbfgs converges on AlphaEarth in ~280 iterations but exceeds
    # 1000 on Sentinel-2, whose 168 engineered features are highly collinear
    # (the indices are functions of the same bands). newton-cholesky is built for
    # n >> d and converged in 17 iterations on the same data to within 0.005 pp.
    m = LogisticRegression(solver=SOLVER, max_iter=MAX_ITER, class_weight='balanced')
    m.fit(sc.transform(Xtr), ytr)
    yp = m.predict(sc.transform(Xte))
    per_country = {}
    for c in COUNTRIES_2021:
        k = ctry_te == c
        if k.sum() >= 10:
            yt, yq = yte[k], yp[k]
            cls = np.unique(yt)
            rec = recall_score(yt, yq, labels=cls, average=None, zero_division=0)
            per_country[c] = dict(
                balanced_accuracy=float(balanced_accuracy_score(yt, yq) * 100),
                n_test=int(k.sum()),
                per_class={cnames[i]: dict(recall=float(r * 100),
                                           support=int((yt == i).sum()))
                           for i, r in zip(cls, rec)})
    return (float(balanced_accuracy_score(yte, yp) * 100), per_country,
            int(np.max(m.n_iter_)), yp)


def pooled_split(X, y, fold):
    """identical to src/experiments/run_id_baseline.py"""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE + fold)
    s = list(skf.split(X, y))
    te, va = s[0][1], s[1][1]
    tr = np.concatenate([s[i][1] for i in range(2, 5)] + [s[0][0]])
    return np.setdiff1d(tr, np.concatenate([te, va])), te


def loco_split(ctry, target):
    """identical to src/experiments/run_loco.py: 10% of the training countries held for val"""
    tr_all = np.where(ctry != target)[0]
    rng = np.random.RandomState(RANDOM_STATE)
    n_val = max(1, int(len(tr_all) * 0.1))
    val = set(rng.choice(tr_all, size=n_val, replace=False).tolist())
    tr = np.array([i for i in tr_all if i not in val])
    return tr, np.where(ctry == target)[0]


for ft in FEATS:
    keys = [f'{ft}|pooled|{r}' for r in range(5)] + [f'{ft}|loco|{c}' for c in COUNTRIES_2021]
    if all(k in res for k in keys):
        print(f'[{ft}] complete, skipping', flush=True); continue
    t = time.time()
    df, fcols, cnames = load_eu_features(ft, 'b', countries=COUNTRIES_2021)
    fcols = filter_coords(fcols)
    X, y = prepare_xy(df, fcols, 'b')
    ctry = df[COL_COUNTRY].values
    del df
    print(f'[{ft}] loaded {X.shape[0]:,} x {X.shape[1]} in {time.time()-t:.0f}s', flush=True)

    for r in range(5):
        k = f'{ft}|pooled|{r}'
        if k in res: continue
        t1 = time.time(); tr, te = pooled_split(X, y, r)
        ba, pc, ni, yp = fit_eval(X[tr], y[tr], X[te], y[te], ctry[te], cnames)
        pd.DataFrame({'country': ctry[te], 'y_true': y[te], 'y_pred': yp}).to_parquet(
            f'{PRED_DIR}/pooled_{ft}_r{r}.parquet', index=False)
        res[k] = dict(protocol='pooled_cv', feature_type=ft, repeat=r, n_train=int(len(tr)),
                      overall_balanced_accuracy=ba, n_iter=ni, converged=ni < MAX_ITER,
                      per_country=pc)
        json.dump(res, open(OUT, 'w'), indent=1)
        print(f'[{ft}] pooled {r+1}/5  BA {ba:5.2f}  iter {ni}{"" if ni<MAX_ITER else " NOT CONVERGED"}'
              f'  {(time.time()-t1)/60:.1f} min  (elapsed {(time.time()-t0)/60:.0f} min)', flush=True)

    for c in COUNTRIES_2021:
        k = f'{ft}|loco|{c}'
        if k in res: continue
        t1 = time.time(); tr, te = loco_split(ctry, c)
        ba, pc, ni, yp = fit_eval(X[tr], y[tr], X[te], y[te], ctry[te], cnames)
        pd.DataFrame({'country': ctry[te], 'y_true': y[te], 'y_pred': yp}).to_parquet(
            f'{PRED_DIR}/loco_{ft}_{c}.parquet', index=False)
        res[k] = dict(protocol='loco', feature_type=ft, held_out=c, n_train=int(len(tr)),
                      overall_balanced_accuracy=ba, n_iter=ni, converged=ni < MAX_ITER,
                      per_country=pc)
        json.dump(res, open(OUT, 'w'), indent=1)
        print(f'[{ft}] loco {c}  BA {ba:5.2f}  iter {ni}{"" if ni<MAX_ITER else " NOT CONVERGED"}'
              f'  {(time.time()-t1)/60:.1f} min  (elapsed {(time.time()-t0)/60:.0f} min)', flush=True)
    del X, y, ctry

print(f'\nDONE in {(time.time()-t0)/60:.0f} min -> {OUT}', flush=True)
