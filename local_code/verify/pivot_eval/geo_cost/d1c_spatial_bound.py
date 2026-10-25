"""
D1c：把 D1b 的距离分箱效应（逐国平均）代入，估计"类别构成不变、只改变类内空间摆放"的平衡精度上限。
对照：SRS k=200（3 次抽样）；神谕摆放：每类标签数不变，类内按地理贪心最大覆盖（R=10 或 20 km，取较大者）。
增益 = Σ_c (1/|big|) · mean_{i∈c}[β(bin(d_oracle,i)) − β(bin(d_srs,i))]，单位 pp（与 ba_ge30 同口径的近似）。
"""
import os, glob
os.environ.setdefault('OMP_NUM_THREADS', '2')
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
from scipy import stats
H = '/Users/perng_mini/AlphaCrops_local/verify/pivot_eval/geo_cost'
G = '/Users/perng_mini/AlphaCrops_local/results/guidance_G'
Cd = '/Users/perng_mini/AlphaCrops_local/data/cache_guidance_G'
y = np.load(f'{Cd}/y.npy'); lon = np.load(f'{Cd}/lon.npy'); lat = np.load(f'{Cd}/lat.npy')
b = pd.read_csv(f'{H}/d1b_distance_bins.csv')
cols = [c for c in b.columns if c.startswith('b_')]
print('分箱效应（相对 ≥100 km；逐国先平均，再跨国）：')
cm = b.groupby('target')[cols].mean()
for c in cols:
    m, se = cm[c].mean(), cm[c].std(ddof=1) / np.sqrt(len(cm)); t = stats.t.ppf(.975, len(cm) - 1)
    print(f'  {c}: {m:+.2f} [{m - t * se:+.2f}, {m + t * se:+.2f}]  >0: {(cm[c] > 0).sum()}/11')
EDGES = [0, 5, 10, 20, 50, 100]

def beta_of(d, bt):
    out = np.zeros_like(d)
    for j in range(len(EDGES) - 1):
        hi = EDGES[j + 1] if j + 1 < len(EDGES) else 100
        out[(d >= EDGES[j]) & (d < hi)] = bt[j]
    return out  # ≥100 km 为 0

def km_xy(lo, la, lat0):
    return np.c_[lo * 111.32 * np.cos(np.radians(lat0)), la * 110.57]

def pool_positions(S, split, rep):
    n_cand = len(S['cand'])
    rng = np.random.default_rng(10_000 * split + 100 * rep + 7)
    return np.sort(rng.choice(n_cand, min(30_000, n_cand), replace=False))

res = []
for f0 in sorted(glob.glob(f'{G}/stage0/*.npz')):
    tgt, sp = os.path.basename(f0)[:-4].split('_s'); sp = int(sp)
    bt = cm.loc[tgt, cols].to_numpy() / 100
    S = np.load(f0); lat0 = float(S['lat0'])
    test = S['test']; yT = y[test]; xyT = km_xy(lon[test], lat[test], lat0)
    big = np.flatnonzero(np.bincount(yT, minlength=10) >= 30)
    pp = pool_positions(S, sp, 0); idx = S['allT'][pp]; yP = y[idx]; xyP = km_xy(lon[idx], lat[idx], lat0)
    for d in range(3):
        rng = np.random.default_rng(99 + d)
        sel = rng.choice(len(yP), 200, replace=False)
        best = {}
        for R in (10, 20):
            g = 0.0
            for c in big:
                ic = np.flatnonzero(yT == c); sc = sel[yP[sel] == c]; n_c = len(sc)
                if n_c == 0:
                    continue
                d_srs = cKDTree(xyP[sc]).query(xyT[ic])[0]
                pc = np.flatnonzero(yP == c)
                cand = pc if len(pc) <= 2000 else rng.choice(pc, 2000, replace=False)
                nbrs = cKDTree(xyT[ic]).query_ball_point(xyP[cand], r=R)
                covered = np.zeros(len(ic), bool); chosen = []
                for _ in range(n_c):
                    gains = np.array([(~covered[n]).sum() if len(n) else 0 for n in nbrs])
                    j = int(np.argmax(gains))
                    if gains[j] == 0:
                        break
                    covered[nbrs[j]] = True; chosen.append(cand[j])
                # 覆盖饱和后剩余名额按"离已选最远"补齐（仍是类内摆放）
                while len(chosen) < n_c:
                    dd = cKDTree(xyP[chosen]).query(xyP[cand])[0] if chosen else np.ones(len(cand))
                    chosen.append(cand[int(np.argmax(dd))])
                d_or = cKDTree(xyP[chosen]).query(xyT[ic])[0]
                g += (beta_of(d_or, bt) - beta_of(d_srs, bt)).mean() / len(big)
            best[R] = 100 * g
        res.append(dict(target=tgt, split=sp, draw=d, gain_R10=best[10], gain_R20=best[20], gain=max(best.values())))
    print(tgt, sp, pd.DataFrame(res).tail(3)[['gain_R10', 'gain_R20']].mean().round(3).to_dict(), flush=True)
r = pd.DataFrame(res)
r.to_csv(f'{H}/d1c_spatial_bound.csv', index=False)
c = r.groupby('target').gain.mean()
print('空间神谕摆放的 BA 增益近似（pp，k=200，相对 SRS 同构成）：均值 %.2f，范围 %.2f–%.2f' % (c.mean(), c.min(), c.max()))
print(c.round(2).to_string())
