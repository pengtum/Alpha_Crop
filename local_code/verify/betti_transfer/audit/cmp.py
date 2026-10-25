# 比对主实现 diag_rows.csv 与独立实现 indep_rows.csv 在共同设定上的 tv_a、tv_b、foreign_rate
import os
for v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[v]='2'
import numpy as np, pandas as pd
D='/Users/perng_mini/AlphaCrops_local/verify/betti_transfer'
A=pd.read_csv(f'{D}/diag_rows.csv')
B=pd.read_csv(f'{D}/indep/indep_rows.csv')
print('主实现行数', len(A), '独立行数', len(B))
print('主实现 列取值:', {c: sorted(A[c].unique().tolist()) for c in ['K','thr','w','k','space']})
print('单元数', A.unit.nunique(), '每单元×空间×设定重复?', A.duplicated(['unit','space','K','thr','w','k']).sum())
print('独立 列取值:', {c: sorted(B[c].unique().tolist()) for c in ['K','thr','w','k','space']})
A2=A[(A.K==8)&(A.w=='1/50')].copy()
B=B.rename(columns={'target':'country'})
m=A2.merge(B,on=['country','split','rep','space','K','thr','k'],how='inner',suffixes=('','_i'))
print('共同设定行数', len(m))
for a,b in [('TV_a','tv_a'),('TV_b','tv_b'),('foreign_counted_rate','foreign_rate'),('cov_cv','cov_cv_i'),('n_zero_groups','n_zero'),('max_share','max_share_i'),('TV_none_prop','tv_a_vs_b')]:
    if b not in m: b=b.replace('_i','')
    d=(m[a]-m[b]).abs()
    print(f'{a:22s} 不同(>1e-9) {int((d>1e-9).sum()):4d}/{len(m)}  最大差 {d.max():.6g}')
# 按 unit×space 聚合：哪些单元×空间有差异
m['dif']=(m.TV_a-m.tv_a).abs()>1e-9
m['diff_fr']=(m.foreign_counted_rate-m.foreign_rate).abs()>1e-9
g=m.groupby(['unit','space']).agg(dTV=('dif','any'),dFR=('diff_fr','any'))
print('单元×空间 数', len(g), '：TV_a 有差异', int(g.dTV.sum()), '；计入率有差异', int(g.dFR.sum()))
print('计入率有差异的 单元×空间 按空间:', g[g.dFR].reset_index().groupby('space').size().to_dict())
# 主设定
mm=m[(m.thr=='q95')&(m.k==200)]
for f in ['ae','s2']:
    x=mm[mm.space==f]
    d=(x.TV_a-x.tv_a)
    print(f'主设定 {f}: 不同单元 {int((d.abs()>1e-9).sum())}/44  最大|差| {d.abs().max():.4f}  中位数 主 {x.TV_a.median():.4f} 独立 {x.tv_a.median():.4f}  Q25/Q75 主 {x.TV_a.quantile(.25):.4f}/{x.TV_a.quantile(.75):.4f} 独立 {x.tv_a.quantile(.25):.4f}/{x.tv_a.quantile(.75):.4f}')
    print('   主 ≥0.20 单元', int((x.TV_a>=0.2).sum()), '独立', int((x.tv_a>=0.2).sum()), '; 主 ≥0.10', int((x.TV_a>=0.1).sum()), '独立', int((x.tv_a>=0.1).sum()))
    print('   计入率中位数 主', round(x.foreign_counted_rate.median(),4), '独立', round(x.foreign_rate.median(),4))
    dd=x[d.abs()>1e-9][['unit','TV_a','tv_a','foreign_counted_rate','foreign_rate']]
    print(dd.to_string(index=False))
    print('   排序后第 19–26 个值 主:', np.sort(x.TV_a.values)[18:26], ' 独立:', np.sort(x.tv_a.values)[18:26])
# 其他共同设定汇总
for (thr,k),x in m.groupby(['thr','k']):
    for f in ['ae','s2']:
        y=x[x.space==f]
        print(f'{thr} k={k} {f}: 中位数 主 {y.TV_a.median():.4f} 独立 {y.tv_a.median():.4f}; 不同单元 {int(((y.TV_a-y.tv_a).abs()>1e-9).sum())}; 最大|差| {(y.TV_a-y.tv_a).abs().max():.4f}; TV_b 最大|差| {(y.TV_b-y.tv_b).abs().max():.4f}')
# 分离比比对
S=pd.read_csv(f'{D}/indep/indep_sep.csv').rename(columns={'target':'country'})
sa=A.drop_duplicates(['unit','space'])[['country','split','rep','space','sep_ratio','med_d_foreign_pool','med_d_pool_pool','n_pool']]
s=sa.merge(S,on=['country','split','rep','space'],suffixes=('','_i'))
print('分离比 行', len(s), '最大|差|', (s.sep_ratio-s.sep_ratio_i).abs().max(), 'fp', (s.med_d_foreign_pool-s.med_foreign_to_pool).abs().max(), 'pp', (s.med_d_pool_pool-s.med_pool_nn).abs().max(), 'n_pool 一致', (s.n_pool==s.n_pool_i).all())
