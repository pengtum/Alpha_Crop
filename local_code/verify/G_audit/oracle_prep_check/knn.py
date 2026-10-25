"""几何空间的类别信息量：候选池内 10 近邻交叉验证平衡精度（子样 8000）。ORIG=存档 ORACLE 几何；FIX=按方案 z 截断；
FIXDROP=z 截断并去掉折内常数（缺失类填充）列；S2/AE=指导几何前 32 维。"""
import numpy as np, os, sys, pandas as pd
sys.path.insert(0, '.')
import guidance_G as G
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold, KFold
from sklearn.metrics import balanced_accuracy_score
D = G.GData()
def zc(A):
    A = np.asarray(A, np.float64); return np.clip((A - A.mean(0)) / (A.std(0) + 1e-9), -5, 5)
rows = []
for tgt in D.countries:
    for split in (0, 1):
        S = dict(np.load(os.path.join(G.OUT, 'stage0', f'{tgt}_s{split}.npz')))
        o = np.load(G.oracle_path(tgt, split)); Gm = o['G'].astype(np.float64)
        pp = G.pool_positions(S, split, 0); y = D.y[S['allT'][pp]]
        folds = np.zeros(len(y), int)
        for f, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=0).split(Gm)): folds[te] = f
        keep = [c for c in range(10) if not any(np.ptp(Gm[folds == f, c]) == 0 for f in range(5))]
        sp = {'ORIG': Gm, 'FIX': zc(Gm), 'FIXDROP': zc(Gm[:, keep]), 'S2': S['G_s2'][pp][:, :32], 'AE': S['G_ae'][pp][:, :32]}
        rng = np.random.default_rng(0); sub = rng.choice(len(y), 8000, replace=False)
        r = dict(tgt=tgt, split=split)
        for nm, X in sp.items():
            ph = cross_val_predict(KNeighborsClassifier(10), X[sub], y[sub], cv=KFold(5, shuffle=True, random_state=0))
            r[nm] = 100 * balanced_accuracy_score(y[sub], ph)
        rows.append(r); print({k: (round(v, 1) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
df = pd.DataFrame(rows); df.to_csv('/Users/perng_mini/AlphaCrops_local/verify/G_audit/oracle_prep_check/knn.csv', index=False)
print(df.mean(numeric_only=True).round(1))
