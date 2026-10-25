# 评审对照 R2：以"两视图重训集成"为更强基线，校准层（XV-Cal 两视图 / AE 单视图，类均衡程度 a）与之按 β 做对数概率混合后，
# 在 OA 基本不降时还能多拿多少 BA。只读缓存；每次拟合约 10 ms。
import os, sys
for v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'): os.environ[v]='2'
sys.dont_write_bytecode=True
sys.path.insert(0,'/Users/perng_mini/AlphaCrops_local/verify/pivot_eval/multiview')
import numpy as np, pandas as pd, hashlib, stacker as st
R='/Users/perng_mini/AlphaCrops_local'; G=f'{R}/results/guidance_G'
Y=np.load(f'{R}/data/cache_guidance_G/y.npy'); CS='AT,BE,DE,EE,LT,LV,PT,RO,SE,SI,SK'.split(',')
def hsh(sel): return hashlib.md5(np.sort(np.asarray(sel,dtype=np.int64)).tobytes()).hexdigest()[:16]
def lp(P): return np.log(np.clip(np.asarray(P,np.float64),1e-9,None))
def cen(L): return L-L.mean(1,keepdims=True)
def sc(yT,ph,big): return 100*(ph==yT).mean(),100*np.mean([(ph[yT==c]==c).mean() for c in big])
def ppos(nc,s,r):
    rng=np.random.default_rng(10_000*s+100*r+7); return np.sort(rng.choice(nc,min(30000,nc),replace=False))
rows=[]
for t in CS:
  for s in (0,1):
    z=np.load(f'{G}/stage0/{t}_s{s}.npz'); nc=len(z['cand']); yT=Y[z['test']]
    big=np.flatnonzero(np.bincount(yT,minlength=10)>=30)
    Pa,Ps=z['P_T_ae'].astype(np.float64),z['P_T_s2'].astype(np.float64)
    LaT,LsT=st._ll(Pa[nc:]),st._ll(Ps[nc:])
    for r in (0,1):
      pp=ppos(nc,s,r); yP=Y[z['allT'][pp]]; LaP,LsP=st._ll(Pa[pp]),st._ll(Ps[pp])
      for meth,cond in (('srs','none'),('predstrat_srs','AE')):
        for k in (50,200):
          sp=f'{G}/sel/{t}_s{s}_r{r}_{cond}_{meth}_k{k}.npz'
          if not os.path.exists(sp): continue
          zz=np.load(sp); sel_pos=zz['sel_pos']; hh=hsh(zz['sel'])
          P={}
          for clf in ('ae','s2'):
            pf=f'{G}/probs/{t}_s{s}_r{r}_{clf}_k{k}_{hh}.npz'
            if os.path.exists(pf): P[clf]=np.load(pf)
          if len(P)<2: continue
          ys=yP[sel_pos]
          Rae=cen(lp(P['ae']['i300'])); R2=cen(0.5*lp(P['ae']['i300'])+0.5*lp(P['s2']['i300']))
          base={'ret_ae':Rae,'ret_2v':R2}
          for bn,Lb in base.items():
            oa,ba=sc(yT,Lb.argmax(1),big); rows.append(dict(target=t,split=s,rep=r,meth=meth,k=k,arm=bn,beta=0,a=np.nan,oa=oa,ba=ba))
          for views in ('both','ae'):
            for a in (0.5,1.0):
              th=st.fit(LaP[sel_pos],LsP[sel_pos],ys,mode=a,gw=5.0,gb=1.0,views=views)
              H=cen(np.log(np.clip(st.predict(th,LaT,LsT),1e-9,None)))
              for bn,Lb in base.items():
                if (views=='ae')!=(bn=='ret_ae') and not (views=='both' and bn=='ret_2v'): continue
                for beta in (0.25,0.4,0.5,1.0):
                  L=(1-beta)*Lb+beta*H; oa,ba=sc(yT,L.argmax(1),big)
                  rows.append(dict(target=t,split=s,rep=r,meth=meth,k=k,arm=f'{bn}+head_{views}',beta=beta,a=a,oa=oa,ba=ba))
df=pd.DataFrame(rows); out=os.path.dirname(os.path.abspath(__file__)); df.to_csv(f'{out}/r2_rows.csv',index=False)
U=['target','split','rep']
def fmt(v):
    v=np.asarray(v); m=v.mean(); se=v.std(ddof=1)/np.sqrt(len(v)); return f'{m:+.2f} [{m-2.228*se:+.2f},{m+2.228*se:+.2f}] {int((v>0).sum())}/{len(v)}'
def cmp(A,B,lab):
    j=A.set_index(U)[['ba','oa']].join(B.set_index(U)[['ba','oa']],rsuffix='_b',how='inner')
    dB=(j.ba-j.ba_b).groupby(level=0).mean(); dO=(j.oa-j.oa_b).groupby(level=0).mean()
    print(f'  {lab:52s} n={len(j):3d} dBA {fmt(dB)}  dOA {fmt(dO)}')
for meth in ('srs','predstrat_srs'):
  for k in (50,200):
    d=df[(df.meth==meth)&(df.k==k)]
    print(f'== 标签={meth} k={k} ==')
    b2=d[d.arm=='ret_2v']; b1=d[d.arm=='ret_ae']
    for a in (0.5,1.0):
      for beta in (0.25,0.4,0.5,1.0):
        cmp(d[(d.arm=='ret_2v+head_both')&(d.a==a)&(d.beta==beta)],b2,f'2v重训+XV头 a={a} β={beta} − 2v重训')
      for beta in (0.25,0.4,0.5):
        cmp(d[(d.arm=='ret_ae+head_ae')&(d.a==a)&(d.beta==beta)],b1,f'AE重训+AE头 a={a} β={beta} − AE重训')
# 全流程对 SRS 两视图重训基线
print('== 全流程：predstrat 标签 + (2v重训+XV头) − SRS 标签 2v重训 ==')
for k in (50,200):
  for a in (0.5,1.0):
    for beta in (0.25,0.4):
      A=df[(df.meth=='predstrat_srs')&(df.k==k)&(df.arm=='ret_2v+head_both')&(df.a==a)&(df.beta==beta)]
      B=df[(df.meth=='srs')&(df.k==k)&(df.arm=='ret_2v')]
      cmp(A,B,f'k={k} a={a} β={beta}')
  A=df[(df.meth=='predstrat_srs')&(df.k==k)&(df.arm=='ret_2v')]; B=df[(df.meth=='srs')&(df.k==k)&(df.arm=='ret_2v')]
  cmp(A,B,f'k={k} 只换 predstrat 标签（2v重训）')
