# 评审补充对照：XV-Cal 的"两视图"增益是否只是"两个模型的集成"？
# 同一批 G 的 SRS 标签（rep0/rep1，k=50/200）：AE 重训、S2 重训、两视图重训集成（R2V）、XV-Cal、以及二者混合。
# 只读 results/guidance_G/{stage0,sel,probs} 与 data/cache_guidance_G/y.npy；输出只写本目录。
import os, sys, hashlib
for v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'): os.environ[v]='2'
sys.dont_write_bytecode = True
sys.path.insert(0, '/Users/perng_mini/AlphaCrops_local/verify/pivot_eval/multiview')
import numpy as np, pandas as pd
import stacker as st
R='/Users/perng_mini/AlphaCrops_local'; GO=f'{R}/results/guidance_G'
Y=np.load(f'{R}/data/cache_guidance_G/y.npy')
CS='AT,BE,DE,EE,LT,LV,PT,RO,SE,SI,SK'.split(',')
def pool_positions(nc, split, rep):
    rng=np.random.default_rng(10_000*split+100*rep+7); return np.sort(rng.choice(nc, min(30000,nc), replace=False))
def cl(L): return L - L.mean(1, keepdims=True)
def lg(P): return np.log(np.clip(np.asarray(P,np.float64),1e-6,None))
rows=[]
for t in CS:
  for s in (0,1):
    z=np.load(f'{GO}/stage0/{t}_s{s}.npz'); nc=len(z['cand'])
    Pa=z['P_T_ae'].astype(np.float64); Ps=z['P_T_s2'].astype(np.float64)
    yT=Y[z['test']]; big=np.flatnonzero(np.bincount(yT,minlength=10)>=30)
    LaT,LsT=lg(Pa[nc:]),lg(Ps[nc:])
    for r in (0,1):
      pp=pool_positions(nc,s,r); LaP,LsP=lg(Pa[pp]),lg(Ps[pp]); yP=Y[z['allT'][pp]]
      unit=f'{t}_s{s}_r{r}'
      for k in (50,200):
        f=f'{GO}/sel/{unit}_none_srs_k{k}.npz'
        if not os.path.exists(f): continue
        zz=np.load(f); sp, sel = zz['sel_pos'], zz['sel']
        ys=yP[sp]
        h=hashlib.md5(np.sort(np.asarray(sel,dtype=np.int64)).tobytes()).hexdigest()[:16]
        Lr={}
        for v in ('ae','s2'):
          pf=f'{GO}/probs/{unit}_{v}_k{k}_{h}.npz'
          Lr[v]=cl(lg(np.load(pf)['i300'])) if os.path.exists(pf) else None
        if Lr['ae'] is None or Lr['s2'] is None: print('miss',unit,k); continue
        R2V=0.5*Lr['ae']+0.5*Lr['s2']
        th=st.fit(LaP[sp],LsP[sp],ys,mode=0.5,gw=5.0,gb=1.0,views='both'); XV=cl(lg(st.predict(th,LaT,LsT)))
        thb=st.fit(LaP[sp],LsP[sp],ys,mode='bal',gw=5.0,gb=1.0,views='both'); XVb=cl(lg(st.predict(thb,LaT,LsT)))
        arms={'R_ae':Lr['ae'],'R_s2':Lr['s2'],'R2V':R2V,'XV_a0.5':XV}
        for b in (0.3,0.5,0.7): arms[f'mix{b}_XV_R2V']=b*XV+(1-b)*R2V
        for b in (0.3,0.5): arms[f'mix{b}_XV_Rae']=b*XV+(1-b)*Lr['ae']
        for b in (0.25,0.4): arms[f'mix{b}_XVbal_R2V']=b*XVb+(1-b)*R2V
        for b in (0.25,0.4): arms[f'mix{b}_XVbal_Rae']=b*XVb+(1-b)*Lr['ae']
        for n,L in arms.items():
          oa,ba=st.scores(yT, np.exp(L), big)
          rows.append(dict(target=t,split=s,rep=r,k=k,arm=n,oa=oa,ba=ba))
    print(t,s,flush=True)
df=pd.DataFrame(rows); df.to_csv('/Users/perng_mini/AlphaCrops_local/verify/pivot_eval/review_exec/ctrl_2view_rows.csv',index=False)
def tt(v):
    v=np.asarray(v); m=v.mean(); se=v.std(ddof=1)/np.sqrt(len(v)); return f"{m:+.2f} [{m-2.228*se:+.2f},{m+2.228*se:+.2f}] {(v>0).sum()}/{len(v)}"
out=[]
for k in (50,200):
  d=df[df.k==k]; base=d[d.arm=='R_ae'].set_index(['target','split','rep'])[['ba','oa']]
  b2=d[d.arm=='R2V'].set_index(['target','split','rep'])[['ba','oa']]
  for a in d.arm.unique():
    x=d[d.arm==a].set_index(['target','split','rep'])[['ba','oa']]
    j=x.join(base,rsuffix='_b'); j2=x.join(b2,rsuffix='_b')
    out.append(dict(k=k,arm=a,n=len(j),BA=round(x.ba.mean(),2),OA=round(x.oa.mean(),2),
      dBA_vs_Rae=tt((j.ba-j.ba_b).groupby(level=0).mean()),dOA_vs_Rae=tt((j.oa-j.oa_b).groupby(level=0).mean()),
      dBA_vs_R2V=tt((j2.ba-j2.ba_b).groupby(level=0).mean()),dOA_vs_R2V=tt((j2.oa-j2.oa_b).groupby(level=0).mean())))
pd.set_option('display.width',300)
rep=pd.DataFrame(out).to_string(index=False); print(rep)
open('/Users/perng_mini/AlphaCrops_local/verify/pivot_eval/review_exec/ctrl_2view_report.txt','w').write(rep)
