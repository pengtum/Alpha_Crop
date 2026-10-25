# 评审对照 R1：同一批 SRS 标签下，"两视图重训集成"（AE 重训 + S2 重训的对数概率平均）能否解释 XV-Cal 的两视图增益。
# 只读 results/guidance_G/{stage0,sel,probs} 与 multiview/d3_paired_rows.csv；输出只写本目录。
import os
for v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'): os.environ[v]='2'
import numpy as np, pandas as pd, hashlib
R='/Users/perng_mini/AlphaCrops_local'; G=f'{R}/results/guidance_G'
Y=np.load(f'{R}/data/cache_guidance_G/y.npy')
CS='AT,BE,DE,EE,LT,LV,PT,RO,SE,SI,SK'.split(',')
def h(sel): return hashlib.md5(np.sort(np.asarray(sel,dtype=np.int64)).tobytes()).hexdigest()[:16]
def lp(P): return np.log(np.clip(P.astype(np.float64),1e-9,None))
def sc(yT,ph,big):
    rec=[(ph[yT==c]==c).mean() for c in big]; return 100*(ph==yT).mean(),100*np.mean(rec)
rows=[]
for t in CS:
  for s in (0,1):
    z=np.load(f'{G}/stage0/{t}_s{s}.npz'); yT=Y[z['test']]; big=np.flatnonzero(np.bincount(yT,minlength=10)>=30)
    for r in (0,1):
      for k in (50,200):
        sp=f'{G}/sel/{t}_s{s}_r{r}_none_srs_k{k}.npz'
        if not os.path.exists(sp): continue
        hh=h(np.load(sp)['sel'])
        P={}
        for clf in ('ae','s2'):
          pf=f'{G}/probs/{t}_s{s}_r{r}_{clf}_k{k}_{hh}.npz'
          if not os.path.exists(pf): break
          P[clf]=np.load(pf)
        if len(P)<2: continue
        for it in ('i300','i200'):
          La,Ls=lp(P['ae'][it]),lp(P['s2'][it])
          for name,L in (('ret_ae',La),('ret_s2',Ls),('ret_2v',0.5*La+0.5*Ls)):
            oa,ba=sc(yT,L.argmax(1),big)
            rows.append(dict(target=t,split=s,rep=r,k=k,it=it,arm=name,oa=oa,ba=ba))
df=pd.DataFrame(rows); df.to_csv(os.path.join(os.path.dirname(__file__),'r1_rows.csv'),index=False)
# 与 XV-Cal（d3_paired_rows，SRS 标签，alpha=0.5）配对
d3=pd.read_csv(f'{R}/verify/pivot_eval/multiview/d3_paired_rows.csv')
d3=d3[(d3.cond=='none')&(d3.method=='srs')&(d3.alpha==0.5)]
U=['target','split','rep']
def fmt(v):
    v=np.asarray(v); m=v.mean(); se=v.std(ddof=1)/np.sqrt(len(v)); return f'{m:+.2f} [{m-2.228*se:+.2f},{m+2.228*se:+.2f}] {int((v>0).sum())}/{len(v)}'
def cmp(A,B,lab):
    j=A.set_index(U)[['ba','oa']].join(B.set_index(U)[['ba','oa']],rsuffix='_b',how='inner')
    dB=(j.ba-j.ba_b).groupby(level=0).mean(); dO=(j.oa-j.oa_b).groupby(level=0).mean()
    print(f'{lab:55s} n={len(j):3d}  dBA {fmt(dB)}  dOA {fmt(dO)}')
for k in (50,200):
  print(f'== k={k}（SRS 标签，11 国 × 2 划分 × 2 重复）==')
  for it in ('i300','i200'):
    a=df[(df.k==k)&(df.it==it)]
    cmp(a[a.arm=='ret_2v'],a[a.arm=='ret_ae'],f'两视图重训集成 − AE 重训 ({it})')
    cmp(a[a.arm=='ret_2v'],a[a.arm=='ret_s2'],f'两视图重训集成 − S2 重训 ({it})')
    cb=d3[(d3.k==k)&(d3.views=='both')]; ca=d3[(d3.k==k)&(d3.views=='ae')]
    cmp(cb,a[a.arm=='ret_2v'],f'XV-Cal(both) − 两视图重训集成 ({it})')
    cmp(ca,a[a.arm=='ret_ae'],f'单视图校准层(AE) − AE 重训 ({it})')
    cmp(cb,a[a.arm=='ret_ae'],f'XV-Cal(both) − AE 重训 ({it})')
