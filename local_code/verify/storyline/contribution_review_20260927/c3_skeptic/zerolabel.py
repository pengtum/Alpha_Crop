"""只读：在 D1 的 P_T 底座上做零标签（不看任何目标标签）的工作点搬移，比较 D1 决策头。
零标签调整：logit' = log P_T − τ·log(π̂)，π̂ = 候选池（D1 的 3 万池，无标签）上平均预测概率；τ ∈ 网格。
另：迭代版（EM 式）让池上平均预测分布逼近均匀（只在外国模型会预测的类上）。
测试集标签只用于评估。"""
import os, sys
for v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'): os.environ[v]='2'
sys.dont_write_bytecode=True
import numpy as np, pandas as pd
L='/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops/label_expansion'
sys.path.insert(0,L); os.chdir(L)
import guidance_G as G
R_G='/Users/perng_mini/AlphaCrops_local/results/guidance_G'
D=G.GData(); y=np.asarray(D.y)
C=10
TAUS=np.round(np.arange(0,2.01,0.05),2)
rows=[]
for tgt in 'AT,BE,DE,EE,LT,LV,PT,RO,SE,SI,SK'.split(','):
  for split in (0,1):
    S=np.load(os.path.join(R_G,'stage0',f'{tgt}_s{split}.npz'))
    n_cand=len(S['cand']); allT,test=S['allT'],S['test']; yT=y[test]
    big=np.flatnonzero(np.bincount(yT,minlength=C)>=G.MIN_N)
    for rep in (0,1):
      pool_pos=G.pool_positions(S,split,rep)
      for f in ('ae','s2'):
        P=S[f'P_T_{f}'].astype(np.float64)
        PP=P[pool_pos]; PT=P[n_cand:]
        ZT=np.log(np.clip(PT,1e-12,None)); ZP=np.log(np.clip(PP,1e-12,None))
        pi=PP.mean(0)
        for tau in TAUS:
          b=-tau*np.log(np.clip(pi,1e-12,None))
          oa,ba,_,_=G._scores(yT,(ZT+b).argmax(1),big)
          rows.append(dict(target=tgt,split=split,rep=rep,clf=f,design='tau',tau=tau,oa=oa,ba_ge30=ba))
        # 迭代均衡：让池上 softmax 平均在"池上被预测过（均值>1e-3）"的类上均匀
        act=pi>1e-3
        b=np.zeros(C)
        for it in range(200):
          l=ZP+b; l-=l.max(1,keepdims=True); q=np.exp(l); q/=q.sum(1,keepdims=True)
          m=q.mean(0)
          b[act]-=0.5*np.log(m[act]*act.sum())
        oa,ba,_,_=G._scores(yT,(ZT+b).argmax(1),big)
        rows.append(dict(target=tgt,split=split,rep=rep,clf=f,design='eq_iter',tau=np.nan,oa=oa,ba_ge30=ba))
    print(tgt,split,flush=True)
pd.DataFrame(rows).to_csv('/private/tmp/c3_skeptic/zerolabel_rows.csv',index=False)
