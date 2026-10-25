# 从 diag_rows.csv 独立重算 diag_summary.txt 的关键数字与判定，并核对主实现报告中的若干说法
import os
for v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[v]='2'
import numpy as np, pandas as pd
from scipy import stats
D='/Users/perng_mini/AlphaCrops_local/verify/betti_transfer'
R=pd.read_csv(f'{D}/diag_rows.csv'); A=pd.read_csv(f'{D}/diag_alloc.csv')
txt=open(f'{D}/diag_summary.txt',encoding='utf-8').read()
def vd(x): return '退化' if x<0.10 else ('弱偏离' if x<0.20 else '明显偏离')
M=R[(R.K==8)&(R.thr=='q95')&(R.w=='1/50')&(R.k==200)]
for f in ['ae','s2']:
    x=M[M.space==f].TV_a
    print(f, 'n',len(x),'中位数',x.median(),'Q25',np.percentile(x,25),'Q75',np.percentile(x,75),'均值',round(x.mean(),4),'min',x.min(),'max',x.max(),
          '≥0.20',int((x>=0.2).sum()),'<0.20',int((x<0.2).sum()),'≥0.10',int((x>=0.1).sum()),'判定',vd(x.median()))
    line=[l for l in txt.splitlines() if l.startswith(f'{f} ') and '明显偏离' in l or l.startswith(f'{f} ') and '退化' in l][0]
    print('   summary 行:', line.strip())
# 逐国中位数
pc=M.groupby(['country','space']).TV_a.median().unstack()
print(pc.round(4).to_string())
# AE−S2
pa=M[M.space=='ae'].set_index('unit').TV_a; ps=M[M.space=='s2'].set_index('unit').TV_a
d=(pa-ps.reindex(pa.index)); dc=d.groupby(M[M.space=='ae'].set_index('unit').country).mean()
mu,sd=dc.mean(),dc.std(ddof=1); t=stats.t.ppf(.975,10)
print('AE−S2 国家层均值',round(mu,4),'CI',round(mu-t*sd/np.sqrt(11),4),round(mu+t*sd/np.sqrt(11),4),'单元层中位数',d.median(),'ae>s2',int((d>0).sum()),'国家全负',bool((dc<0).all()),'最小/最大',dc.idxmin(),round(dc.min(),4),dc.idxmax(),round(dc.max(),4))
# 54 设定判定计数
g=R.groupby(['space','K','thr','w','k']).TV_a.median().reset_index()
g['v']=g.TV_a.map(vd)
print(g.groupby(['space','v']).size().unstack().to_string())
# 5a
for nm,ch in [('K=16',dict(K=16)),('q80',dict(thr='q80')),('none',dict(thr='none')),('w=1/10',dict(w='1/10')),('w=1/200',dict(w='1/200')),('k=50',dict(k=50)),('k=400',dict(k=400))]:
    s={**dict(K=8,thr='q95',w='1/50',k=200),**ch}
    q=R.copy()
    for c,v in s.items(): q=q[q[c]==v]
    print(nm, {f: round(q[q.space==f].TV_a.median(),4) for f in ['ae','s2']}, '计入率', {f: round(q[q.space==f].foreign_counted_rate.median(),4) for f in ['ae','s2']})
# 其它报告说法
for f in ['ae','s2']:
    x=M[M.space==f]
    print(f,'Σn0 中位数',x.sum_n0.median(),'0名额组中位数',x.n_zero_groups.median(),'最大占比',x.max_share.median(),'无初始最大占比',x.max_share_none.median(),'TV(无初始,比例)',round(x.TV_none_prop.median(),4))
print('cov_cv 缺失行', int(R.cov_cv.isna().sum()), 'sum_m==0 行', int((R.sum_m==0).sum()))
# 行内一致性：rows 与 alloc 互相对账
key=['unit','space','K','thr','w','k']
Ag=A.groupby(key)
chk=Ag.apply(lambda d: pd.Series(dict(tva=0.5*np.abs(d.n_transfer-d.n_none).sum()/d.k.iloc[0], tvb=0.5*np.abs(d.n_transfer-d.n_prop).sum()/d.k.iloc[0],
                                         sm=d.m_g.sum(), capsum=d.capacity.sum(), ntsum=d.n_transfer.sum(), nnsum=d.n_none.sum(), gsum=d.gamma.sum())), include_groups=False).reset_index()
j=R.merge(chk,on=key)
print('alloc→rows 对账 行', len(j), 'TV_a 最大差', (j.TV_a-j.tva).abs().max(), 'TV_b 最大差', (j.TV_b-j.tvb).abs().max(), 'Σm 一致', bool((j.sum_m==j.sm).all()), 'Σcap==n_pool', bool((j.capsum==j.n_pool).all()), 'Σn==k', bool(((j.ntsum==j.k)&(j.nnsum==j.k)).all()))
print('n_pool 取值', R.groupby('country').n_pool.unique().to_dict())
# 容量约束 / r_g 单调（q80<=q95）
print('n_transfer<=capacity', bool((A.n_transfer<=A.capacity).all()), 'n0=w·m', bool(np.allclose(A.n0_g, A.m_g*A.w.map({'1/10':.1,'1/50':.02,'1/200':.005}))))
