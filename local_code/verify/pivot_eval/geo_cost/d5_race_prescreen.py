"""
D5：RACE（区域感知的自适应类别扩展）的免训练预筛。
RACE(λ)：第 1 批 predstrat_srs（k/2）→ Olofsson π̂ → sqrt 配额 → 欠额类从同类种子扩展：
  每个种子取 AE 最近的 m=50 个未标地块作候选；λ=0 取 AE 最近者；λ=+1 取"离已有同类标签地理最远"者（区域覆盖）；
  λ=−1 取"离种子地理最近"者（批量化，省站点）。新取到的同类地块成为新种子（自适应群抽样规则）。
指标：构成（同 D3）+ 用 D1b 分箱效应换算的"同类空间覆盖分"（BA pp 近似，与构成共同决定）。
"""
import os, glob, sys
os.environ.setdefault('OMP_NUM_THREADS', '2')
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
sys.path.insert(0, '/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops/label_expansion')
sys.path.insert(0, '/Users/perng_mini/AlphaCrops_local/verify/pivot_eval/geo_cost')
import samplers.base as B
import samplers.designs  # noqa
from samplers.designs import alloc, olofsson_prior
from d3_ace_composition import metrics, km_xy, pool_positions, y, lon, lat, G
H = '/Users/perng_mini/AlphaCrops_local/verify/pivot_eval/geo_cost'
b = pd.read_csv(f'{H}/d1b_distance_bins.csv'); bcols = [c for c in b.columns if c.startswith('b_')]
BM = b.groupby('target')[bcols].mean()
EDGES = [0, 5, 10, 20, 50, 100]

def beta_of(d, bt):
    out = np.zeros_like(d)
    for j in range(len(EDGES) - 1):
        out[(d >= EDGES[j]) & (d < (EDGES[j + 1] if j + 1 < len(EDGES) else 100))] = bt[j]
    return out

def spatial_score(sel, yP, xyP, yT, xyT, big, bt):
    g = 0.0
    for c in big:
        ic = np.flatnonzero(yT == c); sc = sel[yP[sel] == c]
        if len(sc):
            g += beta_of(cKDTree(xyP[sc]).query(xyT[ic])[0], bt).mean() / len(big)
    return 100 * g

def race(yP, P, Z, xy, tz, k, lam, seed, m=50, rho=0.5):
    N = len(yP); rng = np.random.default_rng(seed)
    k1 = int(round(rho * k))
    ctx = B.Ctx(N=N, C=10, T_ae=Z.astype(np.float32), T_s2=None, T_xy=None, T_nuts3=None, F_ae=None, F_s2=None,
                F_xy=None, F_y=np.zeros(1, int), F_country=None, P_T=P.astype(np.float64))
    s1 = B.REGISTRY['predstrat_srs'](ctx, k1, seed, geo_w=0.0)
    yh = P.argmax(1)
    pi_hat, _ = olofsson_prior(yP[s1], yh[s1], np.bincount(yh, minlength=10), C=10)
    lab = np.zeros(N, bool); lab[s1] = True
    cnt = np.bincount(yP[s1], minlength=10)
    cap = np.where(pi_hat > 0, np.maximum(np.ceil(pi_hat * N), cnt), 0).astype(int)
    deficit = np.maximum(alloc(cap, np.sqrt(pi_hat), k) - cnt, 0)
    rest = k - k1
    d_alloc = alloc(np.where(deficit > 0, rest, 0), deficit.astype(float), rest) if deficit.sum() > 0 else np.zeros(10, int)
    for c in np.argsort(-d_alloc):
        bgt = int(d_alloc[c])
        if bgt == 0:
            continue
        labc = [i for i in np.flatnonzero(lab) if yP[i] == c]
        if not labc:
            continue
        queue = list(labc); used = qi = 0
        while used < bgt and queue:
            s = queue[qi % len(queue)]
            cand = tz.query(Z[s], k=m + 64)[1]
            cand = np.array([j for j in cand if not lab[j]][:m])
            if len(cand) == 0:
                queue.remove(s); continue
            if lam == 0:
                j = cand[0]
            elif lam > 0:
                dmin = cKDTree(xy[labc]).query(xy[cand])[0]
                j = cand[int(np.argmax(dmin))]
            else:
                j = cand[int(np.argmin(np.linalg.norm(xy[cand] - xy[s], axis=1)))]
            lab[j] = True; used += 1
            if yP[j] == c:
                labc.append(j); queue.append(j)
            qi += 1
    left = k - lab.sum()
    if left > 0:
        lab[rng.choice(np.flatnonzero(~lab), left, replace=False)] = True
    return np.flatnonzero(lab)

rows = []
for f0 in sorted(glob.glob(f'{G}/stage0/*.npz')):
    tgt, sp = os.path.basename(f0)[:-4].split('_s'); sp = int(sp)
    bt = BM.loc[tgt, bcols].to_numpy() / 100
    S = np.load(f0); lat0 = float(S['lat0'])
    test = S['test']; yT = y[test]; xyT = km_xy(lon[test], lat[test], lat0)
    big = np.flatnonzero(np.bincount(yT, minlength=10) >= 30)
    pp = pool_positions(S, sp, 0); idx = S['allT'][pp]; yP = y[idx]; xy = km_xy(lon[idx], lat[idx], lat0)
    Z = S['G_ae'][pp][:, :16].astype(np.float64); Z /= np.sqrt(Z.var(0).sum() / 16)
    P = S['P_T_ae'][pp]; yh = P.argmax(1); tz = cKDTree(Z)
    pi = np.bincount(yP, minlength=10) / len(yP)
    rec = np.array([(yh[yP == c] == c).mean() if (yP == c).any() else np.nan for c in range(10)])
    blind = np.array([c for c in big if rec[c] < 0.2])
    for k in (200,):
        for d in range(3):
            seed = 4242 + 13 * d + 1000 * sp
            rng = np.random.default_rng(seed)
            ctx = B.Ctx(N=len(yP), C=10, T_ae=Z.astype(np.float32), T_s2=None, T_xy=None, T_nuts3=None, F_ae=None,
                        F_s2=None, F_xy=None, F_y=np.zeros(1, int), F_country=None, P_T=P.astype(np.float64))
            arms = {'srs': rng.choice(len(yP), k, replace=False),
                    'predstrat_srs': B.REGISTRY['predstrat_srs'](ctx, k, seed, geo_w=0.0)}
            Nc = np.bincount(yP, minlength=10); nq = alloc(Nc, np.sqrt(Nc), k)
            arms['oracle_sqrt'] = np.concatenate([rng.choice(np.flatnonzero(yP == c), nq[c], replace=False) for c in range(10) if nq[c]])
            for lam in (0, 1, -1):
                arms[f'race_l{lam:+d}'] = race(yP, P, Z, xy, tz, k, lam, seed)
            for nm, sel in arms.items():
                rows.append(dict(target=tgt, split=sp, k=k, draw=d, method=nm,
                                 spatial=spatial_score(sel, yP, xy, yT, xyT, big, bt), **metrics(sel, yP, pi, blind, big, xy)))
    print(tgt, sp, flush=True)
df = pd.DataFrame(rows); df.to_csv(f'{H}/d5_race_prescreen.csv', index=False)
pd.set_option('display.width', 250)
print(df.groupby('method')[['L1_sqrt', 'n_blind', 'min_big', 'n_sites2km', 'spatial']].mean().round(3).to_string())
ref = df[df.method == 'predstrat_srs'].set_index(['target', 'split', 'draw']).spatial
for m in ('race_l+0', 'race_l+1', 'race_l-1', 'oracle_sqrt', 'srs'):
    x = df[df.method == m].set_index(['target', 'split', 'draw']).spatial - ref
    c = x.groupby(level='target').mean()
    print(f'{m} − predstrat_srs 空间覆盖分：{c.mean():+.3f}（逐国 {c.min():+.2f}…{c.max():+.2f}，>0 {int((c > 0).sum())}/11）')
x = (df[df.method == 'race_l+1'].set_index(['target', 'split', 'draw']).spatial -
     df[df.method == 'race_l+0'].set_index(['target', 'split', 'draw']).spatial).groupby(level='target').mean()
print('race λ=+1 − λ=0 空间覆盖分：', x.round(2).to_dict())
