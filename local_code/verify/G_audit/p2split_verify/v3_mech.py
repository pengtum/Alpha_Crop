# 机制复核：指导模型 argmax 分层纯度、池上 argmax BA、predstrat_srs 期望/实现构成、margin 实现构成，及与 Δba 的相关
import sys, os, glob
sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd
from scipy import stats
import guidance_G as G
from samplers.designs import predstrat_weights, _alloc
OUT = "/Users/perng_mini/AlphaCrops_local/results/guidance_G"
V = "/Users/perng_mini/AlphaCrops_local/verify/G_audit/p2split_verify"
D = G.GData()
def tsum(v):
    v = np.asarray(v, float); n=len(v); m=v.mean(); se=v.std(ddof=1)/np.sqrt(n)
    q=stats.t.ppf(.975,n-1); p=2*stats.t.sf(abs(m/se),n-1)
    return f"{m:+.4f} [{m-q*se:+.4f}, {m+q*se:+.4f}] p={p:.3f} >0 {int((v>0).sum())}/{n}"
def ent(c): c=c[c>0]; return float(-(c*np.log(c)).sum())
rows=[]
for f in sorted(glob.glob(f"{OUT}/stage0/*_s*.npz")):
    tgt, sp = os.path.basename(f)[:-4].split('_s'); split=int(sp)
    S = np.load(f)
    pp = G.pool_positions(S, split, 0); idx = S['allT'][pp]; y = D.y[idx]
    yT = D.y[S['test']]; big = np.flatnonzero(np.bincount(yT, minlength=10) >= 30)
    pi_pool = np.bincount(y, minlength=10)/len(y)
    small_big = big[pi_pool[big] < np.median(pi_pool[big])]
    for cond, fe in (('AE','ae'),('S2','s2')):
        PT = S[f'P_T_{fe}'][pp].astype(float); yh = PT.argmax(1)
        Nh = np.bincount(yh, minlength=10)
        rec = np.array([(yh[y==c]==c).mean() if (y==c).any() else np.nan for c in range(10)])
        M = np.zeros((10,10))
        for h in range(10):
            if Nh[h]>0: M[h]=np.bincount(y[yh==h], minlength=10)/Nh[h]
        for k in (50,200):
            nh = _alloc(Nh, predstrat_weights(PT, yh, 10, 'sqrt'), k, 2)
            ec = (nh[:,None]*M).sum(0)/k
            base = dict(target=tgt, split=split, cond=cond, k=k,
                        pool_acc=(yh==y).mean(), pool_ba=np.nanmean(rec[big]),
                        purity_alloc=float((nh/k*M.max(1)).sum()),
                        purity_diag=float((nh/k*np.diag(M)).sum()))
            rows.append(dict(base, method='predstrat_exp', comp_L1=np.abs(ec-pi_pool).sum(), ent=ent(ec),
                             minority=ec[small_big].sum(), ncls=np.nan))
            for method in ('predstrat_srs','margin'):
                z = np.load(f"{OUT}/sel/{tgt}_s{split}_r0_{cond}_{method}_k{k}.npz")
                ys = D.y[z['sel']]; c = np.bincount(ys, minlength=10)/k
                rows.append(dict(base, method=method, comp_L1=np.abs(c-pi_pool).sum(), ent=ent(c),
                                 minority=c[small_big].sum(), ncls=len(np.unique(ys)),
                                 ncls_big=len(np.intersect1d(np.unique(ys), big))))
Q = pd.DataFrame(rows); Q.to_csv(f"{V}/mech_mine.csv", index=False)
cols = ['pool_acc','pool_ba','purity_alloc','purity_diag','comp_L1','ent','minority','ncls','ncls_big']
W = Q.pivot_table(index=['target','split','method','k'], columns='cond', values=cols)
Dd = pd.DataFrame({c: W[(c,'AE')]-W[(c,'S2')] for c in cols})
# 指导质量（与方法、k 无关的量只取一次）
g = Dd.xs(('predstrat_exp',50), level=('method','k'))
for c in ('pool_acc','pool_ba'):
    print(f"Δ{c} (country t):", tsum(g[c].groupby('target').mean()))
for k in (50,200):
    g = Dd.xs(('predstrat_exp',k), level=('method','k'))
    print(f"k={k} Δpurity_alloc:", tsum(g.purity_alloc.groupby('target').mean()), "| Δpurity_diag:", tsum(g.purity_diag.groupby('target').mean()))
g = Dd.xs('predstrat_exp', level='method')
print("Δpurity_alloc avg k:", tsum(g.purity_alloc.groupby('target').mean()))
print("\n期望/实现构成 AE−S2（国家 t；平均两档 k 与两划分）:")
for m in ('predstrat_exp','predstrat_srs','margin'):
    gm = Dd.xs(m, level='method')
    for c in ('comp_L1','ent','minority','ncls','ncls_big'):
        if gm[c].notna().any():
            print(f"  {m:14s} {c:9s}", tsum(gm[c].groupby('target').mean()))
    for k in (50,200):
        gk = gm.xs(k, level='k')
        print(f"    k={k}: ent {gk.ent.mean():+.4f}  minority {gk.minority.mean():+.4f}  comp_L1 {gk.comp_L1.mean():+.4f}  ncls {gk.ncls.mean():+.3f}")
# 相关：单元 Δba（两分类器平均，P0 与 P2）与实现 Δcomp_L1
R = pd.concat([pd.read_csv(f) for f in glob.glob(f"{OUT}/rows_t*_*.csv")], ignore_index=True)
KEY=['target','split','rep','method','clf','k']
out=[]
for met in ('ba_ge30','ba_P2'):
    a = R[(R.cond=='AE')&R.method.isin(G.PRED)].set_index(KEY)[met]; b = R[(R.cond=='S2')&R.method.isin(G.PRED)].set_index(KEY)[met]
    d = (a-b).dropna().rename('d').reset_index().groupby(['target','split','method','k']).d.mean()
    X = Dd.join(d, how='inner')
    for m in G.PRED:
        gx = X.xs(m, level='method')
        for c in ('comp_L1','ent','minority','ncls'):
            r,p = stats.spearmanr(gx[c], gx.d)
            print(f"  {met} {m:14s} Δba~Δ{c:8s} n={len(gx)} rho={r:+.2f} p={p:.3f}")
        # 国家层面（11 国）相关
        gc = gx.groupby('target')[['comp_L1','d']].mean(); r,p = stats.spearmanr(gc.comp_L1, gc.d)
        print(f"  {met} {m:14s} 国家层面 Δba~Δcomp_L1 n=11 rho={r:+.2f} p={p:.3f}")
