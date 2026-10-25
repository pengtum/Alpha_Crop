"""
D4：Betti 式整群取标在"有站点固定成本"时是否划算（粗估，不训练）。
成本 = n_sites·c_s + n·c_p，ρ = c_s/c_p。预算固定为 SRS 200 个标签的成本 200(ρ+1)。
整群设计：随机取 n/M 个种子地块，各取候选池中地理最近的 M 个（含自身）。
(1) 同样 n 个标签时，整群相对 SRS 的"空间覆盖损失"：用 D1b 分箱效应换算成 BA（pp）；
(2) 构成：整群样本的类别多样性（≥5 个标签的大类数、blind 类标签数）；
(3) 标签数增益：用 G 的 SRS 学习曲线（k=50/100/200/400，11 国 × 2 分类器，rep0+rep1）按 log k 线性插值，
    超出 400 的不外推（记为 NaN）。
"""
import os, glob
os.environ.setdefault('OMP_NUM_THREADS', '2')
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
H = '/Users/perng_mini/AlphaCrops_local/verify/pivot_eval/geo_cost'
G = '/Users/perng_mini/AlphaCrops_local/results/guidance_G'
Cd = '/Users/perng_mini/AlphaCrops_local/data/cache_guidance_G'
y = np.load(f'{Cd}/y.npy'); lon = np.load(f'{Cd}/lon.npy'); lat = np.load(f'{Cd}/lat.npy')
b = pd.read_csv(f'{H}/d1b_distance_bins.csv'); cols = [c for c in b.columns if c.startswith('b_')]
cm = b.groupby('target')[cols].mean()
EDGES = [0, 5, 10, 20, 50, 100]

def beta_of(d, bt):
    out = np.zeros_like(d)
    for j in range(len(EDGES) - 1):
        out[(d >= EDGES[j]) & (d < EDGES[j + 1] if j + 1 < len(EDGES) else d < 100)] = bt[j]
    return out

def km_xy(lo, la, lat0):
    return np.c_[lo * 111.32 * np.cos(np.radians(lat0)), la * 110.57]

def pool_positions(S, split, rep):
    n_cand = len(S['cand'])
    rng = np.random.default_rng(10_000 * split + 100 * rep + 7)
    return np.sort(rng.choice(n_cand, min(30_000, n_cand), replace=False))

def spatial_score(sel, yP, xyP, yT, xyT, big, bt):
    g = 0.0
    for c in big:
        ic = np.flatnonzero(yT == c); sc = sel[yP[sel] == c]
        if len(sc):
            g += beta_of(cKDTree(xyP[sc]).query(xyT[ic])[0], bt).mean() / len(big)
    return 100 * g

rows = []
for f0 in sorted(glob.glob(f'{G}/stage0/*.npz')):
    tgt, sp = os.path.basename(f0)[:-4].split('_s'); sp = int(sp)
    bt = cm.loc[tgt, cols].to_numpy() / 100
    S = np.load(f0); lat0 = float(S['lat0'])
    test = S['test']; yT = y[test]; xyT = km_xy(lon[test], lat[test], lat0)
    big = np.flatnonzero(np.bincount(yT, minlength=10) >= 30)
    pp = pool_positions(S, sp, 0); idx = S['allT'][pp]; yP = y[idx]; xyP = km_xy(lon[idx], lat[idx], lat0)
    tree = cKDTree(xyP)
    for n in (200, 400):
        for d in range(3):
            rng = np.random.default_rng(5 + d + n)
            srs = rng.choice(len(yP), n, replace=False)
            base = spatial_score(srs, yP, xyP, yT, xyT, big, bt)
            rows.append(dict(target=tgt, split=sp, n=n, M=1, draw=d, spatial=base,
                             n_big_ge5=int(sum((yP[srs] == c).sum() >= 5 for c in big)),
                             n_classes=len(np.unique(yP[srs])), diam_km=0.0))
            for M in (5, 10, 20):
                seeds = rng.choice(len(yP), n // M, replace=False)
                sel, diam = [], []
                taken = np.zeros(len(yP), bool)
                for s in seeds:
                    dd, nn = tree.query(xyP[s], k=M * 3)
                    nn = [j for j in nn if not taken[j]][:M]
                    taken[nn] = True; sel += nn; diam.append(np.max(np.linalg.norm(xyP[nn] - xyP[s], axis=1)))
                sel = np.array(sel)
                rows.append(dict(target=tgt, split=sp, n=n, M=M, draw=d,
                                 spatial=spatial_score(sel, yP, xyP, yT, xyT, big, bt),
                                 n_big_ge5=int(sum((yP[sel] == c).sum() >= 5 for c in big)),
                                 n_classes=len(np.unique(yP[sel])), diam_km=float(np.median(diam))))
    print(tgt, sp, flush=True)
r = pd.DataFrame(rows); r.to_csv(f'{H}/d4_cluster_cost.csv', index=False)
base = r[r.M == 1].groupby(['target', 'split', 'n']).spatial.mean()
r['d_spatial'] = r.apply(lambda q: q.spatial - base.loc[(q.target, q.split, q.n)], axis=1)
print('\n整群相对 SRS（同 n）的空间覆盖损失（BA pp，D1b 换算）与构成：')
print(r.groupby(['n', 'M'])[['d_spatial', 'n_big_ge5', 'n_classes', 'diam_km']].mean().round(3).to_string())

# 学习曲线
g = pd.concat([pd.read_csv(f) for f in glob.glob(f'{G}/rows_t1_*.csv')])
g = g[g.method.isin(['srs', 'foreign_only'])]
lc = g.groupby(['target', 'clf', 'k']).ba_ge30.mean().unstack('k')
print('\nSRS 学习曲线（11 国平均，ba_ge30）：'); print(lc.groupby(level='clf').mean().round(2).to_string())
def interp(row, n):
    ks = np.array([50, 100, 200, 400]); v = row[ks].to_numpy()
    return np.nan if n > 400 else np.interp(np.log(n), np.log(ks), v)
out = []
loss = r[r.n == 200].groupby(['target', 'M']).d_spatial.mean()
for rho in (0, 1, 3, 10):
    for M in (5, 10, 20):
        n_cl = 200 * (rho + 1) / (rho / M + 1)
        for (t, clf), row in lc.iterrows():
            gain_n = interp(row, n_cl) - row[200]
            out.append(dict(rho=rho, M=M, n_equiv=round(n_cl), target=t, clf=clf, gain_labels=gain_n,
                            loss_spatial=loss.loc[(t, M)], net=gain_n + loss.loc[(t, M)]))
o = pd.DataFrame(out)
print('\n等成本下整群（M 个/站点）相对 SRS-200 的净效应粗估（pp；只含"标签数增益 + 空间覆盖损失"，未含簇内冗余与构成变化）：')
print(o.groupby(['rho', 'M'])[['n_equiv', 'gain_labels', 'loss_spatial', 'net']].mean().round(2).to_string())
o.to_csv(f'{H}/d4_cost_frontier.csv', index=False)
