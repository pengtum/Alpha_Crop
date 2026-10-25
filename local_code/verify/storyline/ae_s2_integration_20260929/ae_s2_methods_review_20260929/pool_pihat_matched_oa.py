# 只读、不训练：检验"同 OA AE−S2（Z0，W4_min）"对 π̂ 定义的敏感性。
# C 级原估计用 π̂ = 测试集平均预测（转导）；FAB-11 确认运行拟用 π̂ = 无标签候选池平均预测。
# 这里用 G stage0 已存 P_T（候选 + 测试行），rep0 候选池（G.pool_positions），τ∈[-2,2] 步长 0.05，与 ae_s2_matched_oa.py 同窗口/插值/覆盖规则。
import os, sys
for v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'): os.environ[v]='2'
sys.dont_write_bytecode=True
import numpy as np, pandas as pd
from scipy import stats
L='/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops/label_expansion'
sys.path.insert(0,L); os.chdir(L)
import guidance_G as G
R_G='/Users/perng_mini/AlphaCrops_local/results/guidance_G'
C=['AT','BE','DE','EE','LT','LV','PT','RO','SE','SI','SK']; C9=[c for c in C if c not in ('RO','PT')]
TAUS=np.round(np.arange(-2,2+1e-9,0.05),2); EPS=1e-9
def ba_at(oa,ba,grid):
    o0,o1,b0,b1=oa[:-1],oa[1:],ba[:-1],ba[1:]
    lo,hi=np.minimum(o0,o1)-EPS,np.maximum(o0,o1)+EPS; d=o1-o0; flat=np.abs(d)<EPS
    inside=(grid[:,None]>=lo[None])&(grid[:,None]<=hi[None])
    lm=np.clip((grid[:,None]-o0[None])/np.where(flat,1.0,d)[None],0,1)
    lm=np.where(flat[None],(b1>b0).astype(float)[None],lm)
    val=np.where(inside,b0[None]+lm*(b1-b0)[None],-np.inf); best=val.max(1); best[~np.isfinite(best)]=np.nan; return best
def curve(P,pihat,yT,big):
    Z=np.log(np.clip(P,1e-300,None)); lp=np.log(np.clip(pihat,1e-12,None))
    oa=[];ba=[]
    for t in TAUS:
        ph=(Z-t*lp).argmax(1); c=ph==yT; oa.append(100*c.mean()); ba.append(100*np.mean([c[yT==k].mean() for k in big]))
    return np.array(oa),np.array(ba)
def tsum(v):
    v=pd.Series(v).dropna(); n=len(v); m=v.mean(); se=v.std(ddof=1)/np.sqrt(n); h=stats.t.ppf(.975,n-1)*se
    return f"{m:+.2f} [{m-h:+.2f}, {m+h:+.2f}] >0 {int((v>0).sum())}/{n}"
D=G.GData(); y=np.asarray(D.y)
rows=[]
for tgt in C:
    for split in (0,1):
        S=np.load(f'{R_G}/stage0/{tgt}_s{split}.npz')
        n_cand=len(S['cand']); allT,test=S['allT'],S['test']; yT=y[test]
        big=np.flatnonzero(np.bincount(yT,minlength=10)>=G.MIN_N)
        pool_pos=G.pool_positions(S,split,0)
        res={}
        for f in ('ae','s2'):
            Pa=S[f'P_T_{f}'].astype(np.float64); Pa=Pa/Pa.sum(1,keepdims=True)
            PT=Pa[n_cand:]; PP=Pa[pool_pos]
            for tag,pi in (('test',PT.mean(0)),('pool',PP.mean(0))):
                res[(f,tag)]=curve(PT,pi,yT,big)
        for tag in ('test','pool'):
            oa_a,ba_a=res[('ae',tag)]; oa_s,ba_s=res[('s2',tag)]; i0=int(np.flatnonzero(TAUS==0)[0]); i5=int(np.flatnonzero(TAUS==0.5)[0])
            anc=min(oa_a[i0],oa_s[i0]); grid=np.round(np.arange(anc-4,anc+1e-9,0.1),6)
            A=ba_at(oa_a,ba_a,grid); Sb=ba_at(oa_s,ba_s,grid); ok=np.isfinite(A)&np.isfinite(Sb)
            rows.append(dict(target=tgt,split=split,pihat=tag,cov=ok.mean(),d=np.mean(A[ok]-Sb[ok]) if ok.mean()>=.5 else np.nan,
                             d05ba=ba_a[i5]-ba_s[i5],d05oa=oa_a[i5]-oa_s[i5]))
        print(tgt,split,'done',flush=True)
R=pd.DataFrame(rows); R.to_csv('/private/tmp/ae_s2_methods_review_20260929/pool_pihat_units.csv',index=False)
out=[]
for tag in ('test','pool'):
    X=R[R.pihat==tag].groupby('target')[['d','d05ba','d05oa','cov']].mean().reindex(C)
    out.append(f"π̂={tag}: 同OA W4_min ΔBA 11国 {tsum(X.d)} | 9国 {tsum(X.loc[C9,'d'])}；τ=0.5 ΔBA 11国 {tsum(X.d05ba)}，ΔOA 11国 {tsum(X.d05oa)}；最小覆盖 {X['cov'].min():.2f}")
    out.append("   逐国同OA: "+str(X.d.round(2).to_dict()))
txt="\n".join(out); print(txt)
open('/private/tmp/ae_s2_methods_review_20260929/pool_pihat_matched_oa_out.txt','w').write(txt+"\n")
