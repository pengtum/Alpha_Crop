# 从 diag_rows.csv 独立重算 diag_summary.txt 的关键数字与判定，并核对主实现报告中的若干说法
import os
for v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[v]='2'
import numpy as np, pandas as pd
from scipy import stats
D='/Users/perng_mini/AlphaCrops_local/verify/betti_transfer'
R=pd.read_csv(f'{D}/diag_rows.csv'); A=pd.read_csv(f'{D}/diag_alloc.csv')
txt=open(f'{D}/diag_summary.txt',encoding='utf-8').read()
key=['unit','space','K','thr','w','k']
A['a1']=(A.n_transfer-A.n_none).abs(); A['b1']=(A.n_transfer-A.n_prop).abs()
chk=A.groupby(key).agg(a1=('a1','sum'),b1=('b1','sum'),sm=('m_g','sum'),capsum=('capacity','sum'),ntsum=('n_transfer','sum'),nnsum=('n_none','sum')).reset_index()
chk['tva']=0.5*chk.a1/chk.k; chk['tvb']=0.5*chk.b1/chk.k
j=R.merge(chk,on=key)
print('alloc→rows 对账 行', len(j), 'TV_a 最大差', (j.TV_a-j.tva).abs().max(), 'TV_b 最大差', (j.TV_b-j.tvb).abs().max(), 'Σm 一致', bool((j.sum_m==j.sm).all()), 'Σcap==n_pool', bool((j.capsum==j.n_pool).all()), 'Σn==k', bool(((j.ntsum==j.k)&(j.nnsum==j.k)).all()))
print('n_pool 取值', R.groupby('country').n_pool.unique().to_dict())
# 容量约束 / r_g 单调（q80<=q95）
print('n_transfer<=capacity', bool((A.n_transfer<=A.capacity).all()), 'n0=w·m', bool(np.allclose(A.n0_g, A.m_g*A.w.map({'1/10':.1,'1/50':.02,'1/200':.005}))))
