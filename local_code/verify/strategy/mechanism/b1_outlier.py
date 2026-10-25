"""(b) 离群：far 比例与 Δ 的关系（G，AE/S2 指导）。"""
import sys; sys.path.insert(0,'.')
from load import ci, fmt
from wls import slope
import numpy as np, pandas as pd
from scipy import stats
pd.set_option('display.width',250)
A=pd.read_pickle('A.pkl'); A=A[A.cond.isin(['AE','S2'])].reset_index(drop=True)
L=pd.read_pickle('L.pkl')
# 构成预测（全部方法估计的合并 b；留一方法版本见 a3）
b=slope(L); A['cell']=np.arange(len(A))
A['pred_comp']=b*L.groupby('cell').dlog.mean().reindex(A.cell).to_numpy()
A['res']=A.d_ba_ge30-A.pred_comp
def within_corr(df,x,y,by='target'):
    d=df[[x,y,by]].dropna().copy()
    for v in (x,y): d[v]=d[v]-d.groupby(by)[v].transform('mean')
    r=np.corrcoef(d[x],d[y])[0,1]
    # 国家层面：逐国斜率
    bs=df.groupby(by).apply(lambda g: np.polyfit(g[x],g[y],1)[0] if g[x].std()>0 else np.nan).dropna()
    t=stats.ttest_1samp(bs,0)
    return r, bs.mean(), t.pvalue, int((bs<0).sum()), len(bs)
print('kcenter 单元层（国家内去均值）：far 比例（公共空间，相对 SRS）与各结果')
kc=A[A.method=='kcenter']
for y in ['d_ba_ge30','d_ba_P2','d_oa','d_oa_P2','res']:
    r,bm,p,nneg,n=within_corr(kc,'d_far_common',y)
    print(f'  {y:10s} r_within={r:+.2f}  逐国斜率均值 {bm:+.2f} pp/单位比例（p={p:.3f}，<0 的国家 {nneg}/{n}）')
print('\n全部方法（单元层，国家内去均值）：')
for y in ['d_ba_ge30','d_oa','res']:
    r,bm,p,nneg,n=within_corr(A,'d_far_common',y)
    print(f'  {y:10s} r_within={r:+.2f}  逐国斜率 {bm:+.2f}（p={p:.3f}，{nneg}/{n} 为负）')
# kcenter：AE vs S2 配对（同单元、分类器、k）
print('\nkcenter AE − S2 指导（同单元、分类器、k 配对；国家层面）：')
kc2=kc.pivot_table(index=['target','split','rep','clf','k'],columns='cond',values=['d_far_common','d_far_own','d_ba_ge30','d_oa','d_ba_P2','d_oa_P2','res','pred_comp'])
for v in ['d_far_own','d_far_common','d_ba_ge30','d_ba_P2','d_oa','d_oa_P2','pred_comp','res']:
    dd=(kc2[(v,'AE')]-kc2[(v,'S2')]).groupby(['clf','k','target']).mean().groupby('target').mean()
    print(f'  {v:12s} {fmt(ci(dd),3)}')
# 各国 kcenter 的 ΔOA
print('\nkcenter 逐国 ΔOA / ΔBA（两种指导平均）：')
print(kc.groupby('target')[['d_oa','d_ba_ge30','d_oa_P2','d_far_common']].mean().round(2).T.to_string())
# 按分类器
for clf in ('ae','s2'):
    v=kc[kc.clf==clf].groupby(['cond','k','target']).d_oa.mean().groupby('target').mean()
    print(clf,'kcenter ΔOA',fmt(ci(v)))
