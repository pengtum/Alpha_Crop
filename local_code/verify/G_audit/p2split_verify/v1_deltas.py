# 独立复算：pred 通路（margin、predstrat_srs）与 E1 的 Δ(P0)、Δ(P2)、Δ(P0)−Δ(P2)
import glob, numpy as np, pandas as pd
from scipy import stats
OUT = "/Users/perng_mini/AlphaCrops_local/results/guidance_G"
R = pd.concat([pd.read_csv(f) for f in glob.glob(f"{OUT}/rows_t*_*.csv")], ignore_index=True)
print(len(R))
KEY = ['target','split','rep','method','clf','k']
def tsum(v):
    v = np.asarray(v, float); n=len(v); m=v.mean(); se=v.std(ddof=1)/np.sqrt(n)
    q=stats.t.ppf(.975,n-1); p=2*stats.t.sf(abs(m/se),n-1)
    return f"{m:+.3f} [{m-q*se:+.3f}, {m+q*se:+.3f}] p={p:.3f} >0 {int((v>0).sum())}/{n}"
def delta(methods, metric):
    a = R[(R.cond=='AE')&R.method.isin(methods)].set_index(KEY)[metric]
    b = R[(R.cond=='S2')&R.method.isin(methods)].set_index(KEY)[metric]
    return (a-b).dropna()
def cv(d):
    d = d.rename('d').reset_index()
    return d.groupby(['target','method','clf','k']).d.mean().groupby('target').mean()
for name, ms in (('E1',['lpm2_nogeo','kcenter','typiclust','kherding']),('pred',['margin','predstrat_srs']),
                 ('margin',['margin']),('predstrat_srs',['predstrat_srs'])):
    d0 = delta(ms,'ba_ge30'); d2 = delta(ms,'ba_P2'); d1 = delta(ms,'ba_P1')
    print(f"== {name}: n={len(d0)} reps={sorted(d0.reset_index().rep.unique())}")
    print("  Δ(P0)        ", tsum(cv(d0)))
    print("  Δ(P2)        ", tsum(cv(d2)))
    print("  Δ(P0)-Δ(P2)  ", tsum(cv(d0-d2)))
    print("  Δ(P1)        ", tsum(cv(d1)))
    # 各国 Δ(P0)-Δ(P2)
    print("  per-country Δ(P0)-Δ(P2):", cv(d0-d2).round(3).to_dict())
# P2 校正本身对 ba 的改变量（水平）：AE/S2 各自 ba_P2 - ba_ge30
for name, ms in (('E1',['lpm2_nogeo','kcenter','typiclust','kherding']),('pred',['margin','predstrat_srs'])):
    for c in ('AE','S2'):
        x = R[(R.cond==c)&R.method.isin(ms)]
        print(f"{name} {c}: mean(ba_P2-ba_ge30)={ (x.ba_P2-x.ba_ge30).mean():+.3f}  sd={(x.ba_P2-x.ba_ge30).std():.3f}")
