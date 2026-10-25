"""
D3b：第 1 批改用 predstrat_srs（现有最佳基线）后的两阶段种子扩展（PS-ACE），只看构成；与 D3 同口径。
π̂ 用 Olofsson 分层估计（层 = 外国 ŷ），配额 sqrt(π̂)。扩展邻域：'ae'、'ae_geo5'；种子：全部同类 / 仅外国误判的同类。
"""
import os, glob, sys
os.environ.setdefault('OMP_NUM_THREADS', '2')
import numpy as np, pandas as pd
sys.path.insert(0, '/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops/label_expansion')
sys.path.insert(0, '/Users/perng_mini/AlphaCrops_local/verify/pivot_eval/geo_cost')
import samplers.base as B
import samplers.designs  # noqa
from samplers.designs import alloc, olofsson_prior
from d3_ace_composition import Nbr, metrics, km_xy, pool_positions, y, lon, lat, G

def psace(yP, P, Z, nb, k, rho, kind, seed, seeds_rule='all', R=5.0):
    N = len(yP); rng = np.random.default_rng(seed)
    k1 = int(round(rho * k))
    ctx = B.Ctx(N=N, C=10, T_ae=Z.astype(np.float32), T_s2=None, T_xy=None, T_nuts3=None, F_ae=None, F_s2=None,
                F_xy=None, F_y=np.zeros(1, int), F_country=None, P_T=P.astype(np.float64))
    s1 = B.REGISTRY['predstrat_srs'](ctx, k1, seed, geo_w=0.0)
    yh = P.argmax(1)
    Nh = np.bincount(yh, minlength=10)
    pi_hat, _ = olofsson_prior(yP[s1], yh[s1], Nh, C=10)
    lab = np.zeros(N, bool); lab[s1] = True; order = list(s1)
    cnt = np.bincount(yP[s1], minlength=10)
    cap = np.where(pi_hat > 0, np.maximum(np.ceil(pi_hat * N), cnt), 0).astype(int)
    quota = alloc(cap, np.sqrt(pi_hat), k)
    deficit = np.maximum(quota - cnt, 0)
    rest = k - k1
    d_alloc = alloc(np.where(deficit > 0, rest, 0), deficit.astype(float), rest) if deficit.sum() > 0 else np.zeros(10, int)
    hits = spent = 0
    for c in np.argsort(-d_alloc):
        b = int(d_alloc[c])
        if b == 0:
            continue
        seeds = [i for i in order if yP[i] == c and (seeds_rule == 'all' or yh[i] != c)]
        if not seeds:
            seeds = [i for i in order if yP[i] == c]
        if not seeds:
            continue
        lists = {s: iter(nb.order(s, kind, R=R)) for s in seeds}
        queue = list(seeds); used = qi = 0
        while used < b and queue:
            s = queue[qi % len(queue)]
            nxt = next((j for j in lists[s] if not lab[j]), None)
            if nxt is None:
                queue.remove(s); continue
            lab[nxt] = True; order.append(nxt); used += 1; spent += 1
            if yP[nxt] == c:
                hits += 1; lists[nxt] = iter(nb.order(nxt, kind, R=R)); queue.append(nxt)
            qi += 1
    left = k - lab.sum()
    if left > 0:
        lab[rng.choice(np.flatnonzero(~lab), left, replace=False)] = True
    return np.flatnonzero(lab), hits / max(spent, 1)

rows = []
for f0 in sorted(glob.glob(f'{G}/stage0/*.npz')):
    tgt, sp = os.path.basename(f0)[:-4].split('_s'); sp = int(sp)
    S = np.load(f0)
    yT = y[S['test']]; big = np.flatnonzero(np.bincount(yT, minlength=10) >= 30)
    pp = pool_positions(S, sp, 0)
    idx = S['allT'][pp]; yP = y[idx]
    xy = km_xy(lon[idx], lat[idx], float(S['lat0']))
    Z = S['G_ae'][pp][:, :16].astype(np.float64); Z /= np.sqrt(Z.var(0).sum() / 16)
    P = S['P_T_ae'][pp]; yh = P.argmax(1)
    pi = np.bincount(yP, minlength=10) / len(yP)
    rec = np.array([(yh[yP == c] == c).mean() if (yP == c).any() else np.nan for c in range(10)])
    blind = np.array([c for c in big if rec[c] < 0.2])
    nb = Nbr(Z, xy)
    for k in (50, 200, 400):
        for d in range(5):
            seed = 777 + 31 * d + k + 1000 * sp
            ctx = B.Ctx(N=len(yP), C=10, T_ae=Z.astype(np.float32), T_s2=None, T_xy=None, T_nuts3=None, F_ae=None,
                        F_s2=None, F_xy=None, F_y=np.zeros(1, int), F_country=None, P_T=P.astype(np.float64))
            sel = B.REGISTRY['predstrat_srs'](ctx, k, seed, geo_w=0.0)
            rows.append(dict(target=tgt, split=sp, k=k, method='predstrat_srs', draw=d, hit=np.nan,
                             **metrics(sel, yP, pi, blind, big, xy)))
            for kind in ('ae', 'ae_geo5'):
                for sr in ('all', 'mis'):
                    sel, hit = psace(yP, P, Z, nb, k, 0.5, kind, seed, sr)
                    rows.append(dict(target=tgt, split=sp, k=k, method=f'psace_{kind}_{sr}', draw=d, hit=hit,
                                     **metrics(sel, yP, pi, blind, big, xy)))
    print(tgt, sp, flush=True)
df = pd.DataFrame(rows)
df.to_csv('/Users/perng_mini/AlphaCrops_local/verify/pivot_eval/geo_cost/d3b_psace.csv', index=False)
pd.set_option('display.width', 250)
for k in (50, 200, 400):
    print(f'\n=== k={k} ===')
    print(df[df.k == k].groupby('method')[['L1_sqrt', 'L1_prior', 'n_blind', 'n_big_ge5', 'min_big', 'n_sites2km', 'hit']].mean().round(3).to_string())
