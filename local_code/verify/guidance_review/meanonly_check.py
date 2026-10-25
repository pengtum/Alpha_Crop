"""
敏感性：只用 mean 组（去掉 stdDev 组，即去掉地块内异质性维度）时，AE 与 S2 几何的有效维数、重尾程度、kNN 类别信息。
变体：ae_all_z（现行）、ae_mean_z、ae_mean_raw（不标准化，保留 AE 原生尺度）、s2_all_z（现行）、s2_mean_z。
阶段 0 仿真同 pca_check.py（外国 10×5000 + 目标全部；3 万候选池）。
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pca_check import load, part_scale, pr, knn_bacc, COUNTRIES  # noqa: E402
from samplers.uncertainty import _diam_blocked  # noqa: E402


def main():
    ctry, y, X, ae_cols, s2_cols = load()
    ae_m = np.array([c.endswith('_mean') for c in ae_cols])
    s2_m = np.array([c.endswith('_mean') for c in s2_cols])
    V = {'ae_all_z': (X['ae'], True), 'ae_mean_z': (X['ae'][:, ae_m], True), 'ae_mean_raw': (X['ae'][:, ae_m], False),
         's2_all_z': (X['s2'], True), 's2_mean_z': (X['s2'][:, s2_m], True)}
    res = []
    for tgt in COUNTRIES:
        r = np.random.default_rng(1000 + COUNTRIES.index(tgt))
        foreign = np.concatenate([r.choice(np.flatnonzero(ctry == c), 5000, replace=False)
                                  for c in COUNTRIES if c != tgt])
        allT = np.flatnonzero(ctry == tgt)
        pool = r.choice(allT, min(30_000, len(allT)), replace=False)
        pool_pos = np.searchsorted(allT, np.sort(pool))
        nn_idx = r.choice(len(pool_pos), min(20_000, len(pool_pos)), replace=False)
        yt = y[allT][pool_pos][nn_idx]
        for v, (A, z) in V.items():
            Z = np.vstack([A[foreign], A[allT]]).astype(np.float64)
            mu = Z.mean(0)
            sd = Z.std(0) + 1e-9 if z else np.ones(Z.shape[1])
            p = PCA(32, random_state=0).fit((Z - mu) / sd)
            T = p.transform((A[allT].astype(np.float64) - mu) / sd)[pool_pos]
            tot = (((A[allT].astype(np.float64) - mu) / sd)[pool_pos]).var(0).sum()
            row = dict(target=tgt, var=v)
            for d in (16, 32):
                vv = T[:, :d].var(0)
                row[f'poolkeep{d}'] = vv.sum() / tot
                row[f'PR{d}'] = pr(vv)
            row['top1share16'] = T[:, 0].var() / T[:, :16].var(0).sum()
            S = part_scale(T, 32)
            i, j = r.integers(0, len(S), 20000), r.integers(0, len(S), 20000)
            med = np.median(np.linalg.norm(S[i] - S[j], axis=1))
            row['dmax_over_med'] = _diam_blocked(S, np.einsum('ij,ij->i', S, S), 1024)[0] / med
            row['knn16'] = knn_bacc(part_scale(T[nn_idx], 16), yt)
            row['knn32'] = knn_bacc(part_scale(T[nn_idx], 32), yt)
            res.append(row)
        print(tgt, 'done', flush=True)
    R = pd.DataFrame(res)
    R.to_csv('meanonly_check.csv', index=False)
    g = R.groupby('var').agg(['mean', 'min', 'max']).round(3)
    pd.set_option('display.width', 250)
    pd.set_option('display.max_columns', 40)
    print(g.to_string())
    W = R.pivot(index='target', columns='var')
    for a, b in (('ae_all_z', 's2_all_z'), ('ae_mean_z', 's2_mean_z'), ('ae_mean_raw', 's2_mean_z')):
        for col in ('knn16', 'knn32', 'PR16', 'dmax_over_med'):
            dd = W[(col, a)] - W[(col, b)]
            print(f'{a} − {b} {col}: Δ {dd.mean():+.2f} [{dd.min():+.2f},{dd.max():+.2f}]  AE>S2 国数 {int((dd > 0).sum())}/11')


if __name__ == '__main__':
    main()
