# 只读：同一特征内，MLP/LR 与 LightGBM 的零标签模型（K0mlp/K0lr vs Z0）在同 OA 窗口 [min(OA★)−4, min(OA★)] 的 BA 差。
# 目的：检验"MLP 的 BA 比 LGB 高约 5 pp"是否主要是工作点差。curves_B.csv，C 级，G s0/s1 复用。不训练。
import os
os.environ["OMP_NUM_THREADS"]="2"
import numpy as np, pandas as pd
from scipy import stats
import importlib.util
spec=importlib.util.spec_from_file_location("m","/private/tmp/ae_s2_rse_design_20260929/matched_oa_lr_mlp.py")
EPS=1e-9
C=['AT','BE','DE','EE','LT','LV','PT','RO','SE','SI','SK']; C9=[c for c in C if c not in ('RO','PT')]
def ba_at(oa,ba,grid):
    o0,o1,b0,b1=oa[:-1],oa[1:],ba[:-1],ba[1:]
    lo,hi=np.minimum(o0,o1)-EPS,np.maximum(o0,o1)+EPS; d=o1-o0; flat=np.abs(d)<EPS
    inside=(grid[:,None]>=lo[None])&(grid[:,None]<=hi[None])
    lm=np.clip((grid[:,None]-o0[None])/np.where(flat,1.0,d)[None],0,1)
    lm=np.where(flat[None],(b1>b0).astype(float)[None],lm)
    val=np.where(inside,b0[None]+lm*(b1-b0)[None],-np.inf); best=val.max(1); best[~np.isfinite(best)]=np.nan; return best
def tsum(v):
    v=pd.Series(v).dropna(); n=len(v); m=v.mean(); se=v.std(ddof=1)/np.sqrt(n); h=stats.t.ppf(.975,n-1)*se
    p=2*stats.t.sf(abs(m/se),n-1); return f"{m:+.2f} [{m-h:+.2f}, {m+h:+.2f}]，p={p:.3f}，>0 {int((v>0).sum())}/{n}"
cur=pd.read_csv('/Users/perng_mini/AlphaCrops_local/verify/step0/partB/curves_B.csv').sort_values(['unit','arm','clf','tau'])
out=[]
for alt in ['K0mlp','K0lr']:
    for f in ['ae','s2']:
        rows=[]
        for unit,g in cur[cur.clf==f].groupby('unit'):
            a=g[g.arm==alt]; b=g[g.arm=='Z0']
            t0a=a[np.isclose(a.tau,0)].iloc[0]; t0b=b[np.isclose(b.tau,0)].iloc[0]
            anc=min(t0a.oa,t0b.oa); grid=np.round(np.arange(anc-4,anc+1e-9,0.1),6)
            x=ba_at(a.oa.to_numpy(),a.ba_ge30.to_numpy(),grid); y=ba_at(b.oa.to_numpy(),b.ba_ge30.to_numpy(),grid)
            ok=np.isfinite(x)&np.isfinite(y); cov=ok.mean()
            rows.append(dict(target=g.target.iloc[0],d=float(np.mean(x[ok]-y[ok])) if cov>=.5 else np.nan,
                             d0ba=t0a.ba_ge30-t0b.ba_ge30,d0oa=t0a.oa-t0b.oa,miss=cov<.5))
        U=pd.DataFrame(rows); Cn=U.groupby('target')[['d','d0ba','d0oa']].mean().reindex(C)
        out.append(f"## {alt} − LightGBM Z0，特征 {f}（缺失单元 {int(U.miss.sum())}/44）")
        out.append(f"  τ=0 ΔBA 11 国 {tsum(Cn.d0ba)} | 9 国 {tsum(Cn.loc[C9,'d0ba'])}")
        out.append(f"  τ=0 ΔOA 11 国 {tsum(Cn.d0oa)} | 9 国 {tsum(Cn.loc[C9,'d0oa'])}")
        out.append(f"  同OA ΔBA 11 国 {tsum(Cn.d)} | 9 国 {tsum(Cn.loc[C9,'d'])}")
txt="\n".join(out); print(txt)
open('/private/tmp/ae_s2_rse_design_20260929/matched_oa_learner_contrast_out.txt','w').write(txt+"\n")
