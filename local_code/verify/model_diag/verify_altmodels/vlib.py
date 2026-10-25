"""复核用公共函数（独立于 altmodels/altlib.py 重写；只读项目代码与 G 结果，输出只写本目录）。"""
import os, sys, time
THREADS = int(os.environ.get('VTHREADS', 2))
for v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[v] = str(THREADS)
PROJ = '/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops/label_expansion'
sys.path.insert(0, PROJ); os.chdir(PROJ)
import numpy as np, pandas as pd
import guidance_G as G
from stageA import class_weights_from, MIN_N
GOUT = '/Users/perng_mini/AlphaCrops_local/results/guidance_G'
OUT = '/Users/perng_mini/AlphaCrops_local/verify/model_diag/verify_altmodels'

def unit(D, tgt, split):
    S = np.load(os.path.join(GOUT, 'stage0', f'{tgt}_s{split}.npz'))
    fr, test = S['foreign'], S['test']
    yT = D.y[test]
    big = np.flatnonzero(np.bincount(yT, minlength=10) >= MIN_N)
    cw = class_weights_from(D.y[fr])
    return S, fr, test, yT, big, cw

def sel_of(tgt, split, rep, cond, method, k):
    return np.load(os.path.join(GOUT, 'sel', f'{tgt}_s{split}_r{rep}_{cond}_{method}_k{k}.npz'))['sel'].astype(np.int64)

def build(D, feat, fr, sel, cw, r=50.0):
    X = D.X[feat]
    Xtr = np.asarray(np.vstack([X[fr], X[sel]]), np.float32)
    ytr = np.r_[D.y[fr], D.y[sel]].astype(int)
    wtr = np.r_[cw[D.y[fr]], r * cw[D.y[sel]]]
    ist = np.r_[np.zeros(len(fr), bool), np.ones(len(sel), bool)]
    return Xtr, ytr, wtr, ist

def score(yT, ph, big):
    oa, ba, _, _ = G._scores(yT, ph, big)
    return oa, ba

def std(A, B):
    mu, sd = A.mean(0), A.std(0) + 1e-6
    return (A - mu) / sd, (B - mu) / sd

def lr_proba(A, ytr, wtr, B, C=1.0):
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression(C=C, max_iter=1000).fit(A, ytr, sample_weight=wtr / wtr.mean())
    P = np.zeros((len(B), 10)); P[:, m.classes_] = m.predict_proba(B)
    return P

TAUS = np.round(np.arange(-3, 3.001, 0.05), 2)
def curve(P, yT, big):
    L = np.log(np.clip(P.astype(np.float64), 1e-12, None)); pi = np.clip(P.mean(0), 1e-6, None)
    return np.array([(t, *score(yT, (L - t * np.log(pi)).argmax(1), big)) for t in TAUS])

def env(C, x):
    ok = C[:, 1] >= x - 1e-9
    return C[ok, 2].max() if ok.any() else np.nan

def at_tau(C, t):
    j = int(np.argmin(np.abs(C[:, 0] - t)));  return C[j, 1], C[j, 2]
