"""
G 实验的中介变量与外国模型质量（ANALYSIS_PLAN_guidance.md 第 7 节），分析阶段运行，不需训练。
只读选样文件与阶段 0；类别构成、10 近邻平衡精度与外国模型质量是事后用候选池标签计算的描述量。

用法：python mediators_G.py        输出：OUT/mediators_G.csv、OUT/guidance_quality_G.csv、OUT/report_mediators_G.txt
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

import guidance_G as G
from guidance_G import OUT, E1, PRED, HYB, _part, km_xy, pool_positions

N_REF = 3000  # MMD 与 kNN 用的候选池固定子样本


def nn_median(xy):
    from scipy.spatial import cKDTree
    d, _ = cKDTree(xy).query(xy, k=2)
    return float(np.median(d[:, 1]))


def mmd2(A, Bm, sigma):
    """RBF 核 MMD²（有偏估计）。"""
    def k(X, Y):
        d = (X ** 2).sum(1)[:, None] + (Y ** 2).sum(1)[None] - 2 * X @ Y.T
        return np.exp(-np.maximum(d, 0) / (2 * sigma ** 2))
    return float(k(A, A).mean() + k(Bm, Bm).mean() - 2 * k(A, Bm).mean())


def knn_ba(X, y, k=10):
    """留一 10 近邻多数投票的平衡精度（几何的类别信息量）。"""
    from sklearn.neighbors import NearestNeighbors
    idx = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X, return_distance=False)[:, 1:]
    pred = np.array([np.bincount(y[i], minlength=10).argmax() for i in idx])
    rec = [(pred[y == c] == c).mean() for c in np.unique(y) if (y == c).sum() >= 10]
    return 100 * float(np.mean(rec))


def margin_auroc(P, y):
    from sklearn.metrics import roc_auc_score
    s = np.sort(P, 1)
    unc = 1 - (s[:, -1] - s[:, -2])
    wrong = P.argmax(1) != y
    return float(roc_auc_score(wrong, unc)) if 0 < wrong.mean() < 1 else np.nan


def unit_mediators(D, S, tgt, split, rep, out=OUT):
    pp = pool_positions(S, split, rep)
    idx = S['allT'][pp]
    y = D.y[idx]
    lat0 = float(S['lat0'])
    xy_pool = km_xy(D.lon[idx], D.lat[idx], lat0)
    xy_test = km_xy(D.lon[S['test']], D.lat[S['test']], lat0)
    sp = {f: _part(S[f'G_{f}'][pp], 32) for f in ('ae', 's2')}
    common = np.hstack([sp['ae'], sp['s2']])
    far1 = {}
    for name, X in (('ae', sp['ae']), ('s2', sp['s2']), ('common', common)):
        r = np.linalg.norm(X - X.mean(0), axis=1)
        far1[name] = r >= np.quantile(r, 0.99)
    ref = np.sort(np.random.default_rng(0).choice(len(pp), min(N_REF, len(pp)), replace=False))
    sig = {f: float(np.median(np.linalg.norm(sp[f][ref][:500, None] - sp[f][ref][None, :500], axis=-1)))
           for f in sp}
    pi_pool = np.bincount(y, minlength=10) / len(y)
    rows = []
    from scipy.spatial import cKDTree
    for m in E1 + (PRED + HYB if rep == 0 else []):
        for k in (50, 200):
            sel = {}
            for cond in ('AE', 'S2'):
                p = os.path.join(out, 'sel', f'{tgt}_s{split}_r{rep}_{cond}_{m}_k{k}.npz')
                if os.path.exists(p):
                    sel[cond] = np.load(p)['sel_pos']
            if len(sel) < 2:
                continue
            jac = len(np.intersect1d(sel['AE'], sel['S2'])) / len(np.union1d(sel['AE'], sel['S2']))
            for cond, s in sel.items():
                own = 'ae' if cond == 'AE' else 's2'
                d5, _ = cKDTree(xy_pool[s]).query(xy_test)
                rows.append(dict(target=tgt, split=split, rep=rep, method=m, k=k, cond=cond, jaccard=jac,
                                 nn_km=nn_median(xy_pool[s]), test_within5km=float((d5 <= 5).mean()),
                                 far1_own=float(far1[own][s].mean()), far1_common=float(far1['common'][s].mean()),
                                 comp_L1=float(np.abs(np.bincount(y[s], minlength=10) / k - pi_pool).sum()),
                                 n_classes=len(np.unique(y[s])),
                                 mmd_ae=mmd2(sp['ae'][s], sp['ae'][ref], sig['ae']),
                                 mmd_s2=mmd2(sp['s2'][s], sp['s2'][ref], sig['s2'])))
    return rows


def unit_quality(D, S, tgt, split):
    """外国模型质量与几何的类别信息量（每个国家、划分；重复 0 的候选池）。"""
    pp = pool_positions(S, split, 0)
    y = D.y[S['allT'][pp]]
    pi_pool = np.bincount(y, minlength=10) / len(y)
    ref = np.sort(np.random.default_rng(0).choice(len(pp), min(N_REF, len(pp)), replace=False))
    out = []
    for f in ('ae', 's2'):
        P = S[f'P_T_{f}'][pp].astype(np.float64)
        out.append(dict(target=tgt, split=split, feat=f, temp=float(S[f'temp_{f}']), ece_oof=float(S[f'ece_{f}']),
                        nll_oof=float(S[f'nll_{f}']),
                        nll_pool=float(-np.log(np.clip(P[np.arange(len(y)), y], 1e-300, None)).mean()),
                        auroc_margin=margin_auroc(P, y),
                        prior_L1_argmax=float(np.abs(np.bincount(P.argmax(1), minlength=10) / len(y) - pi_pool).sum()),
                        knn10_ba=knn_ba(_part(S[f'G_{f}'][pp], 32)[ref], y[ref])))
    return out


def main(out=OUT):
    D = G.GData()
    med, qual = [], []
    for p in sorted(glob.glob(os.path.join(out, 'stage0', '*_s*.npz'))):
        tgt, split = os.path.basename(p)[:-4].split('_s')
        split = int(split)
        S = dict(np.load(p))
        qual += unit_quality(D, S, tgt, split)
        for rep in G.REPS:
            med += unit_mediators(D, S, tgt, split, rep, out)
        print(f'{tgt} s{split} 完成', flush=True)
    M, Q = pd.DataFrame(med), pd.DataFrame(qual)
    M.to_csv(os.path.join(out, 'mediators_G.csv'), index=False)
    Q.to_csv(os.path.join(out, 'guidance_quality_G.csv'), index=False)
    with open(os.path.join(out, 'report_mediators_G.txt'), 'w') as fh:
        for f in (sys.stdout, fh):
            if len(M):
                cols = ['jaccard', 'nn_km', 'test_within5km', 'far1_own', 'far1_common', 'comp_L1', 'n_classes',
                        'mmd_ae', 'mmd_s2']
                t = M.groupby(['method', 'k', 'cond'])[cols].mean().round(3)
                print('中介变量（各国单元平均）：\n' + t.to_string(), file=f)
            if len(Q):
                print('\n外国模型质量与几何的类别信息量（国家 × 划分平均）：\n' +
                      Q.groupby('feat').mean(numeric_only=True).round(3).to_string(), file=f)


if __name__ == '__main__':
    main()
