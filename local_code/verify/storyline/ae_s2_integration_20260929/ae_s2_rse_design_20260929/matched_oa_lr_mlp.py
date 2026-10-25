# 只读：用第 0 步 B 已存的 τ 曲线（curves_B.csv；K0lr/K0mlp = guidance_G_ml 的 k=0，RT200lr/RT200mlp = SRS k=200；
# LightGBM 的 Z0/RT200 作对照），在同 OA 窗口 W4_min = [min(OA★_AE, OA★_S2) − 4, min] 内比较 AE 与 S2 的 BA。
# 与 /private/tmp/ae_s2_evidence/ae_s2_matched_oa.py 的 ba_at、窗口、覆盖 ≥50% 规则相同。不训练。C 级描述。
import os
os.environ["OMP_NUM_THREADS"]="2"
import numpy as np, pandas as pd
from scipy import stats
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
for arm in ['Z0','K0lr','K0mlp','RT200','RT200lr','RT200mlp']:
    rows=[]
    for unit,g in cur[cur.arm==arm].groupby('unit'):
        a=g[g.clf=='ae']; s=g[g.clf=='s2']
        t0a=a[np.isclose(a.tau,0)].iloc[0]; t0s=s[np.isclose(s.tau,0)].iloc[0]
        anc=min(t0a.oa,t0s.oa); grid=np.round(np.arange(anc-4,anc+1e-9,0.1),6)
        ba_a=ba_at(a.oa.to_numpy(),a.ba_ge30.to_numpy(),grid); ba_s=ba_at(s.oa.to_numpy(),s.ba_ge30.to_numpy(),grid)
        ok=np.isfinite(ba_a)&np.isfinite(ba_s); cov=ok.mean()
        rows.append(dict(unit=unit,target=g.target.iloc[0],cov=cov,
            d=float(np.mean(ba_a[ok]-ba_s[ok])) if cov>=.5 else np.nan,
            aedom=bool((ba_a[ok]>=ba_s[ok]).all()) if cov>=.5 else np.nan,
            s2dom=bool((ba_s[ok]>=ba_a[ok]).all()) if cov>=.5 else np.nan,
            d0ba=t0a.ba_ge30-t0s.ba_ge30, d0oa=t0a.oa-t0s.oa))
    U=pd.DataFrame(rows); Cn=U.groupby('target')[['d','d0ba','d0oa','cov']].mean().reindex(C)
    out.append(f"## {arm}（单元缺失 {int(U.d.isna().sum())}/44；AE 全窗口占优 {int(np.nansum(U.aedom.astype(float)))}，S2 全窗口占优 {int(np.nansum(U.s2dom.astype(float)))}）")
    out.append(f"  τ=0 ΔBA  11 国 {tsum(Cn.d0ba)} | 9 国 {tsum(Cn.loc[C9,'d0ba'])}")
    out.append(f"  τ=0 ΔOA  11 国 {tsum(Cn.d0oa)} | 9 国 {tsum(Cn.loc[C9,'d0oa'])}")
    out.append(f"  同OA ΔBA 11 国 {tsum(Cn.d)} | 9 国 {tsum(Cn.loc[C9,'d'])}")
    out.append("  逐国同OA ΔBA："+str(Cn.d.round(2).to_dict()))
txt="\n".join(out); print(txt)
open('/private/tmp/ae_s2_rse_design_20260929/matched_oa_lr_mlp_out.txt','w').write(
 "# 同 OA（W4_min）AE−S2 BA：LightGBM 对照与 LR、MLP（curves_B.csv，τ∈[−0.5,2]），C 级，G s0/s1 复用；LR/MLP 为非盲量的延伸\n"+txt+"\n")
