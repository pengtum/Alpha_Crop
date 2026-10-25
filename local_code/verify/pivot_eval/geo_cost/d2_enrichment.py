"""
D2：目标国候选池内，真实类别在"地理邻域""特征邻域""地理+特征邻域"中的富集程度，以及外国模型对各类的可见性。
只读 G stage0 与数据缓存；不训练。
"""
import os, glob
os.environ.setdefault('OMP_NUM_THREADS', '2')
import numpy as np, pandas as pd
from scipy.spatial import cKDTree

G = '/Users/perng_mini/AlphaCrops_local/results/guidance_G'
Cd = '/Users/perng_mini/AlphaCrops_local/data/cache_guidance_G'
y = np.load(f'{Cd}/y.npy'); lon = np.load(f'{Cd}/lon.npy'); lat = np.load(f'{Cd}/lat.npy')
import json
classes = json.load(open(f'{Cd}/meta.json'))['classes']

def km_xy(lo, la, lat0):
    return np.c_[lo * 111.32 * np.cos(np.radians(lat0)), la * 110.57]

def pool_positions(S, split, rep):
    n_cand = len(S['cand'])
    rng = np.random.default_rng(10_000 * split + 100 * rep + 7)
    return np.sort(rng.choice(n_cand, min(30_000, n_cand), replace=False))

M = 10
out = []
for f0 in sorted(glob.glob(f'{G}/stage0/*_s0.npz')):
    tgt = os.path.basename(f0)[:2]
    S = np.load(f0)
    pp = pool_positions(S, 0, 0)
    idx = S['allT'][pp]; yP = y[idx]
    xy = km_xy(lon[idx], lat[idx], float(S['lat0']))
    Z = S['G_ae'][pp][:, :16].astype(np.float64); Z /= np.sqrt(Z.var(0).sum() / 16)
    Z2 = S['G_s2'][pp][:, :16].astype(np.float64); Z2 /= np.sqrt(Z2.var(0).sum() / 16)
    _, nn_geo = cKDTree(xy).query(xy, k=M + 1)
    _, nn_ae = cKDTree(Z).query(xy[:, :0] if False else Z, k=M + 1)
    _, nn_s2 = cKDTree(Z2).query(Z2, k=M + 1)
    # 地理+特征：地理最近 100 个中按 AE 距离取最近 M 个
    _, nn100 = cKDTree(xy).query(xy, k=101)
    nn100 = nn100[:, 1:]
    dz = np.linalg.norm(Z[nn100] - Z[:, None, :], axis=2)
    nn_gf = np.take_along_axis(nn100, np.argsort(dz, axis=1)[:, :M], axis=1)
    d_geo_med = np.median(cKDTree(xy).query(xy, k=M + 1)[0][:, -1])
    pi = np.bincount(yP, minlength=10) / len(yP)
    for clf in ('ae', 's2'):
        P = S[f'P_T_{clf}'][pp]
        yh = P.argmax(1)
        for c in range(10):
            ic = np.flatnonzero(yP == c)
            if len(ic) < 30:
                continue
            row = dict(target=tgt, clf=clf, cls=classes[c], n_pool=len(ic), pi=pi[c],
                       pred_share=(yh == c).mean(), foreign_recall=(yh[ic] == c).mean(),
                       foreign_prec=((yP[yh == c] == c).mean() if (yh == c).any() else np.nan),
                       d10_geo_km_median=d_geo_med)
            if clf == 'ae':
                for nm, nn in (('geo', nn_geo[:, 1:]), ('ae', nn_ae[:, 1:]), ('s2', nn_s2[:, 1:]), ('geo+ae', nn_gf)):
                    frac = (yP[nn[ic]] == c).mean()
                    row[f'hit_{nm}'] = frac
                    row[f'enrich_{nm}'] = frac / pi[c]
                # 外国模型误判的类 c 地块，其邻居中类 c 的比例（"外国看不见的"那一部分能否靠邻域找到）
                mis = ic[yh[ic] != c]
                for nm, nn in (('geo', nn_geo[:, 1:]), ('ae', nn_ae[:, 1:]), ('geo+ae', nn_gf)):
                    row[f'hit_mis_{nm}'] = (yP[nn[mis]] == c).mean() if len(mis) else np.nan
                    # 邻居中类 c 且同样被外国模型误判的比例
                    row[f'hit_mis2_{nm}'] = ((yP[nn[mis]] == c) & (yh[nn[mis]] != c)).mean() if len(mis) else np.nan
            out.append(row)
    print(tgt, 'done; d10 median km', round(d_geo_med, 2), flush=True)
df = pd.DataFrame(out)
df.to_csv('/Users/perng_mini/AlphaCrops_local/verify/pivot_eval/geo_cost/d2_enrichment.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 300)
a = df[df.clf == 'ae']
print(a[['target', 'cls', 'n_pool', 'pi', 'pred_share', 'foreign_recall', 'hit_geo', 'hit_ae', 'hit_s2', 'hit_geo+ae',
         'hit_mis_geo', 'hit_mis_ae', 'hit_mis_geo+ae', 'hit_mis2_geo', 'hit_mis2_ae', 'hit_mis2_geo+ae']].round(3).to_string())
