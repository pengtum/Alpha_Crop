# 用当前（15:10:29 修改后的）step0A.py 的抽样函数重放两个单元的全部抽样，核对与 draws 逐位一致；另比对当前 summarize 输出与 summary_A.txt
import os, sys
for _v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'): os.environ[_v]='2'
sys.dont_write_bytecode=True
sys.path.insert(0,'/Users/perng_mini/AlphaCrops_local/verify/step0/partA')
import numpy as np, pandas as pd
import step0A as A
A.D_GLOBAL = A.G.GData()
bad=0; n=0
for t,s in (('SI',0),('LV',1)):
    F=A.build_frame(A.D_GLOBAL,t,s); Z=np.load(f'/Users/perng_mini/AlphaCrops_local/verify/step0/partA/draws/{t}_s{s}.npz')
    for sc,n0 in A.SCENARIOS:
        for r in range(20):
            sd=A.seed_of(t,s,sc,n0,r); pre=f'{sc}_{n0}_r{r:02d}'
            S0,info=A.draw_S0(F,sc,n0,np.random.default_rng(sd)); ok=np.array_equal(F['cand'][S0],Z[f'{pre}_S0'])
            for arm in ['SRS']+(['EXT-p'] if sc=='clus' else [])+A.BETTI_ARMS:
                res=A.arm_k(F,arm,S0,info,sd); ok&=all(np.array_equal(F['cand'][res[k][0]],Z[f'{pre}_{A.KEY_SAFE[arm]}_k{k}']) for k in A.KS)
            if sc=='clus':
                c=A.arms_cost(F,S0,info,sd)
                for arm in ('EXT','SPR','SPR='): ok&=np.array_equal(F['cand'][c[arm][0]],Z[f'{pre}_{A.KEY_SAFE[arm]}_B20'])
            n+=1; bad+= (not ok)
print('当前代码重放', n, '次抽样，不一致', bad)
rows=pd.read_csv('/Users/perng_mini/AlphaCrops_local/verify/step0/partA/rows_A.csv'); recon=pd.read_csv('/Users/perng_mini/AlphaCrops_local/verify/step0/partA/recon_A.csv'); alloc=pd.read_csv('/Users/perng_mini/AlphaCrops_local/verify/step0/partA/alloc_A.csv')
txt=A.summarize(rows,recon,20,alloc).split('\n'); old=open('/Users/perng_mini/AlphaCrops_local/verify/step0/partA/summary_A.txt').read().rstrip('\n').split('\n')
diff=[(i,a,b) for i,(a,b) in enumerate(zip(txt,old)) if a!=b]
print('summary 行数', len(txt), len(old), '不同行', len(diff), [d[0] for d in diff][:5])
