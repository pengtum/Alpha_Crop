"""只读：对 G 的 SRS 重训模型（测试概率 i300）做零标签 τ 工作点搬移，
与同一批标签上拟合的决策头（D1 srs_G_check）在匹配 OA 下比较。π̂ = 测试集平均预测概率（无标签、转导）。"""
import os, sys, glob
for v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'): os.environ[v]='2'
sys.dont_write_bytecode=True
import numpy as np, pandas as pd
L='/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops/label_expansion'
sys.path.insert(0,L); os.chdir(L)
import guidance_G as G
R_G='/Users/perng_mini/AlphaCrops_local/results/guidance_G'
D=G.GData(); y=np.asarray(D.y); C=10
GR=pd.concat([pd.read_csv(f) for f in glob.glob(R_G+'/rows_t*.csv')],ignore_index=True)
GR=GR[(GR.cond=='none')&(GR.method=='srs')]
TAUS=np.round(np.arange(0,1.51,0.025),3)
rows=[]
for tgt in 'AT,BE,DE,EE,LT,LV,PT,RO,SE,SI,SK'.split(','):
  for split in (0,1):
    S=np.load(os.path.join(R_G,'stage0',f'{tgt}_s{split}.npz'))
    test=S['test']; yT=y[test]; big=np.flatnonzero(np.bincount(yT,minlength=C)>=G.MIN_N)
    for rep in (0,1):
      for f in ('ae','s2'):
        # foreign-only base with test-mean prior (to compare same π̂ construction)
        Q=S[f'P0test_{f}_i300'].astype(np.float64); Q/=Q.sum(1,keepdims=True)
        srcs=[('base',0,Q)]
        g=GR[(GR.target==tgt)&(GR.split==split)&(GR.rep==rep)&(GR.clf==f)]
        for _,r in g.iterrows():
          fn=os.path.join(R_G,'probs',f'{tgt}_s{split}_r{rep}_{f}_k{int(r.k)}_{r.sel_hash}.npz')
          P=np.load(fn)['i300'].astype(np.float64); P/=P.sum(1,keepdims=True)
          oa,ba,_,_=G._scores(yT,P.argmax(1),big)
          assert abs(oa-r.oa)<1e-6 and abs(ba-r.ba_ge30)<1e-6,(fn,oa,r.oa,ba,r.ba_ge30)
          srcs.append(('retrain',int(r.k),P))
        for nm,k,P in srcs:
          Z=np.log(np.clip(P,1e-12,None)); pi=P.mean(0)
          for tau in TAUS:
            oa,ba,_,_=G._scores(yT,(Z-tau*np.log(np.clip(pi,1e-12,None))).argmax(1),big)
            rows.append(dict(target=tgt,split=split,rep=rep,clf=f,src=nm,k=k,tau=tau,oa=oa,ba_ge30=ba))
    print(tgt,split,flush=True)
pd.DataFrame(rows).to_csv('/private/tmp/c3_skeptic/retrain_tau_rows.csv',index=False)
