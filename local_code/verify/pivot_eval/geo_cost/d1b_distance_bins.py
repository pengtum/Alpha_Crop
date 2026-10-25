"""
D1b：同类最近标签距离的分箱效应（相对 ≥100 km），双向固定效应（选样×类、地块）同 D1；只用重复 0 的单元（每单元约 67 个选样）。
另算"空间神谕上限"：固定 SRS k=200 的逐类标签数，在类内按地理贪心最大覆盖（R=20 km）重新摆放，测试地块被同类标签覆盖的份额能增加多少。
"""
import os, glob
os.environ.setdefault('OMP_NUM_THREADS', '2')
import numpy as np, pandas as pd
from scipy.spatial import cKDTree

G = '/Users/perng_mini/AlphaCrops_local/results/guidance_G'
Cd = '/Users/perng_mini/AlphaCrops_local/data/cache_guidance_G'
y = np.load(f'{Cd}/y.npy'); lon = np.load(f'{Cd}/lon.npy'); lat = np.load(f'{Cd}/lat.npy')
EDGES = [0, 5, 10, 20, 50, 100]   # 最后一箱 [100, ∞) 为参照

def km_xy(lo, la, lat0):
    return np.c_[lo * 111.32 * np.cos(np.radians(lat0)), la * 110.57]

def pool_positions(S, split, rep):
    n_cand = len(S['cand'])
    rng = np.random.default_rng(10_000 * split + 100 * rep + 7)
    return np.sort(rng.choice(n_cand, min(30_000, n_cand), replace=False))

def dm(A):
    return A - A.mean(0, keepdims=True) - A.mean(1, keepdims=True) + A.mean()

out, cov = [], []
for f0 in sorted(glob.glob(f'{G}/stage0/*.npz')):
    tgt, sp = os.path.basename(f0)[:-4].split('_s')
    S = np.load(f0)
    test = S['test']; lat0 = float(S['lat0'])
    xyT = km_xy(lon[test], lat[test], lat0); yT = y[test]
    big = np.flatnonzero(np.bincount(yT, minlength=10) >= 30)
    unit = f'{tgt}_s{sp}_r0'
    rows = pd.concat([pd.read_csv(p) for p in glob.glob(f'{G}/rows_t*_{unit}.csv')])
    rows = rows[(rows.k > 0) & rows.clf.isin(['ae', 's2'])]
    for clf in ('ae', 's2'):
        rr = rows[rows.clf == clf].drop_duplicates('sel_hash')
        corr = [S[f'P0test_{clf}_i300'].astype(np.float32).argmax(1) == yT]; sels = [np.array([], int)]
        for _, r in rr.iterrows():
            pp = f'{G}/probs/{unit}_{clf}_k{int(r.k)}_{r.sel_hash}.npz'
            sf = f'{G}/sel/{unit}_{r.cond}_{r.method}_k{int(r.k)}.npz'
            if os.path.exists(pp) and os.path.exists(sf):
                corr.append(np.load(pp)['i300'].argmax(1) == yT); sels.append(np.load(sf)['sel'])
        Cm = np.array(corr, dtype=np.float64)
        XtX = np.zeros((len(EDGES), len(EDGES))); Xty = np.zeros(len(EDGES))
        for c in big:
            ic = np.flatnonzero(yT == c)
            D = np.full((len(sels), len(ic)), np.inf)
            for s, sel in enumerate(sels):
                m = y[sel] == c
                if m.any():
                    D[s] = cKDTree(km_xy(lon[sel[m]], lat[sel[m]], lat0)).query(xyT[ic], k=1)[0]
            Xs = [dm(((D >= EDGES[b]) & (D < (EDGES[b + 1] if b + 1 < len(EDGES) else 100))).astype(float))
                  for b in range(len(EDGES) - 1)]
            Yd = dm(Cm[:, ic]); w = 1.0 / len(ic)
            Xm = np.stack([x.ravel() for x in Xs], 1)
            XtX[:-1, :-1] += w * Xm.T @ Xm; Xty[:-1] += w * Xm.T @ Yd.ravel()
        beta = np.linalg.lstsq(XtX[:-1, :-1], Xty[:-1], rcond=None)[0] * 100
        out.append(dict(target=tgt, split=int(sp), clf=clf, n_sel=len(sels),
                        **{f'b_{EDGES[b]}_{EDGES[b + 1] if b + 1 < len(EDGES) else 100}': beta[b] for b in range(len(EDGES) - 1)}))
        print(unit, clf, len(sels), np.round(beta, 2), flush=True)
    # 空间神谕覆盖上限（与分类器无关）
    pp_ = pool_positions(S, int(sp), 0)
    idx = S['allT'][pp_]; yP = y[idx]; xyP = km_xy(lon[idx], lat[idx], lat0)
    for d in range(3):
        rng = np.random.default_rng(99 + d)
        sel = rng.choice(len(yP), 200, replace=False)
        for R in (10, 20):
            base = opt = 0.0
            for c in big:
                ic = np.flatnonzero(yT == c); n_c = int((yP[sel] == c).sum())
                if n_c == 0:
                    continue
                pc = np.flatnonzero(yP == c)
                dsel = cKDTree(xyP[sel[yP[sel] == c]]).query(xyT[ic], k=1)[0]
                base += (dsel <= R).mean() / len(big)
                # 贪心最大覆盖（候选：类 c 的池地块；最多取 2000 个候选以控制计算量）
                cand = pc if len(pc) <= 2000 else rng.choice(pc, 2000, replace=False)
                nbrs = cKDTree(xyT[ic]).query_ball_point(xyP[cand], r=R)
                covered = np.zeros(len(ic), bool)
                for _ in range(n_c):
                    gains = [(~covered[n]).sum() if len(n) else 0 for n in nbrs]
                    j = int(np.argmax(gains))
                    if gains[j] == 0:
                        break
                    covered[nbrs[j]] = True
                opt += covered.mean() / len(big)
            cov.append(dict(target=tgt, split=int(sp), draw=d, R=R, share_srs=base, share_oracle=opt))
    print(tgt, sp, 'coverage', pd.DataFrame(cov).tail(6)[['R', 'share_srs', 'share_oracle']].mean().round(3).to_dict(), flush=True)
pd.DataFrame(out).to_csv('/Users/perng_mini/AlphaCrops_local/verify/pivot_eval/geo_cost/d1b_distance_bins.csv', index=False)
pd.DataFrame(cov).to_csv('/Users/perng_mini/AlphaCrops_local/verify/pivot_eval/geo_cost/d1b_coverage_oracle.csv', index=False)
