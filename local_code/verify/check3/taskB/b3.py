"""任务 B-3：空间聚集的 50 个标签 → 先验估计 → 恢复率。"""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from sklearn.cluster import KMeans
D = '/Users/perng_mini/AlphaCrops_local/results/check3'; C = 10
M = np.load('meta.npz', allow_pickle=True); lon, lat = M['lon'], M['lat']
T = ['AT','BE','DE','EE','LT','LV','PT','RO','SE','SI','SK']
def load(t,f,m,s):
    p = np.load(f'{D}/{t}_{f}_{m}_{s}.npy').astype(np.float64); return p/p.sum(1,keepdims=True)
def laplace(yl, a=1.0):
    n = np.bincount(yl, minlength=C).astype(float); return (n+a)/(n.sum()+a*C)
rows = []
NREP = 60
for t in T:
    sp = np.load(f'{D}/{t}_split.npz'); pool = sp['pool']; yp, yt = sp['y_pool'], sp['y_test']
    piT = np.bincount(yp, minlength=C)/len(yp)
    lat0 = lat[pool].mean()
    xy = np.c_[lon[pool]*111.32*np.cos(np.radians(lat0)), lat[pool]*110.57]
    for ncl in (20, 50):
        lab = KMeans(ncl, n_init=10, random_state=42).fit_predict(xy)
        csize = np.bincount(lab, minlength=ncl)
        L = {f: (load(t,f,'loco','test')) for f in ('ae','s2')}
        yh = {f: L[f].argmax(1) for f in L}
        po = {f: (L[f]*piT).argmax(1) for f in L}   # 训练类 = 全部 10 类（已核实），π_train 均匀
        for m in (1,2,3):
            for rep in range(NREP):
                rng = np.random.default_rng(10_000*ncl + 100*m + rep)
                cl = rng.choice(ncl, m, replace=False)
                cand = np.flatnonzero(np.isin(lab, cl))
                k = min(50, len(cand))
                sel = rng.choice(cand, k, replace=False)
                pk = laplace(yp[sel])
                pinf = np.bincount(yp[cand], minlength=C)/len(cand)   # 这些簇的全部地块（无限标签极限）
                # 簇的空间范围（km）
                ext = np.ptp(xy[cand],0).max()
                r = dict(target=t, ncl=ncl, m=m, rep=rep, k=k, n_cand=len(cand), extent_km=ext,
                         l1_pk=np.abs(pk-piT).sum(), l1_inf=np.abs(pinf-piT).sum(),
                         n_cls_seen=int((np.bincount(yp[sel],minlength=C)>0).sum()), n_cls_pool=int((piT>0).sum()))
                for f in ('ae','s2'):
                    r[f'{f}_L0'] = 100*(yh[f]==yt).mean(); r[f'{f}_PO'] = 100*(po[f]==yt).mean()
                    r[f'{f}_PK'] = 100*((L[f]*pk).argmax(1)==yt).mean()
                    r[f'{f}_Pinf'] = 100*((L[f]*np.where(pinf>0,pinf,0)+1e-300*L[f]).argmax(1)==yt).mean()
                rows.append(r)
    print(t, 'done', flush=True)
df = pd.DataFrame(rows); df.to_pickle('b3_rows.pkl')
