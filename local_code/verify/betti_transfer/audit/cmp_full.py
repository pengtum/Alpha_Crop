# 全量重算（f64 / f32）分别对照主实现与独立实现
import numpy as np, pandas as pd
D='/Users/perng_mini/AlphaCrops_local/verify/betti_transfer'
F=pd.read_csv('full_recalc.csv')
A=pd.read_csv(f'{D}/diag_rows.csv'); A=A[(A.K==8)&(A.w=='1/50')]
B=pd.read_csv(f'{D}/indep/indep_rows.csv').rename(columns={'target':'country'})
k=['country','split','rep','space','thr','k']
for prec,ref,cols in [('f64',A,('TV_a','TV_b','foreign_counted_rate')),('f32',B,('tv_a','tv_b','foreign_rate')),('f64',B,('tv_a','tv_b','foreign_rate')),('f32',A,('TV_a','TV_b','foreign_counted_rate'))]:
    m=F[F.prec==prec].merge(ref,on=k,suffixes=('','_r'))
    out=[]
    for mine,theirs in zip(('tv_a','tv_b','fr'),cols):
        th = theirs if theirs not in ('tv_a','tv_b') else theirs+'_r'
        d=(m[mine]-m[th]).abs(); out.append(f'{mine}: 不同 {int((d>1e-9).sum())}/{len(m)} 最大差 {d.max():.3g}')
    print(f'我的 {prec} vs {"主实现" if ref is A else "独立实现"}: '+'; '.join(out))
# f32 与 f64 主设定中位数
for prec in ['f64','f32']:
    x=F[(F.prec==prec)&(F.thr=='q95')&(F.k==200)]
    print(prec, {f: x[x.space==f].tv_a.median() for f in ['ae','s2']}, '≥0.2', {f: int((x[x.space==f].tv_a>=0.2).sum()) for f in ['ae','s2']})
