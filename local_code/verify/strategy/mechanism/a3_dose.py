"""逐类剂量–反应：同一选样格内，比 SRS 多拿到某类标签的类，该类召回是否提高更多（格固定效应）。"""
import sys; sys.path.insert(0,'.')
from load import ci, fmt
import numpy as np, pandas as pd
from wls import slope, slope_ctry
pd.set_option('display.width',250)
A=pd.read_pickle('A.pkl'); X=pd.read_pickle('X.pkl')
A=A[A.cond.isin(['AE','S2'])].reset_index(drop=True)
A['cell']=np.arange(len(A))
fo=X[X.method=='foreign_only'].set_index(['target','split','rep','clf'])
ref=X[X.method=='srs'].set_index(['target','split','rep','clf','k'])
L=[]
for c in range(10):
    big=A[f'nT{c}']>=30
    a=A[big]
    kk=pd.MultiIndex.from_frame(a[['target','split','rep','clf']])
    kr=pd.MultiIndex.from_frame(a[['target','split','rep','clf','k']])
    L.append(pd.DataFrame(dict(cell=a.cell,target=a.target,method=a.method,cond=a.cond,clf=a.clf,k=a.k,cls=c,
        drec=100*a[f'd_rec{c}'], dlog=np.log1p(a[f'n{c}'])-np.log1p(ref[f'n{c}'].reindex(kr).to_numpy()),
        dn=a[f'd_n{c}'], n_srs=ref[f'n{c}'].reindex(kr).to_numpy(), rec_fo=100*fo[f'rec{c}'].reindex(kk).to_numpy(),
        rec_srs=100*ref[f'rec{c}'].reindex(kr).to_numpy(), pi=a[f'pi{c}'], nbig=a.nbig)))
L=pd.concat(L,ignore_index=True)
L['w']=1.0/L.nbig
# 格内去均值（格固定效应）
for v in ['drec','dlog']:
    L[v+'_w']=L[v]-L.groupby('cell')[v].transform('mean')
def fit(df,label):
    r=slope_ctry(df)
    print(f"{label:40s} 合并 b={r['pooled']:+.2f}；逐国均值 {r['mean']:+.2f} [{r['lo']:+.2f},{r['hi']:+.2f}] {r['npos']}/{r['n']}  n={len(df)}")
    return r['pooled']
b_all=fit(L,'全部（AE+S2, 两分类器, k50/200）')
for k in (50,200): fit(L[L.k==k],f'k={k}')
for clf in ('ae','s2'): fit(L[L.clf==clf],f'分类器 {clf}')
for cond in ('AE','S2'): fit(L[L.cond==cond],f'指导 {cond}')
# 按外国模型对该类的召回分组（该类在目标国上是否已被外国模型识别）
L['fo_bin']=pd.cut(L.rec_fo,[-1,20,50,80,101],labels=['<20','20-50','50-80','>=80'])
for g,d in L.groupby('fo_bin',observed=True): fit(d,f'外国召回 {g}')
L['n_bin']=pd.cut(L.n_srs,[-1,0,2,5,15,1e9],labels=['0','1-2','3-5','6-15','>15'])
for g,d in L.groupby('n_bin',observed=True): fit(d,f'SRS 该类标签数 {g}')
# 留一方法：用其余方法估计 b，预测该方法的构成贡献
print('\n留一方法估计 b 后，构成通道对 ΔBA 的预测（格内逐类平均 → 国家 → 方法×条件；pp）')
rows=[]
cellmean=L.groupby('cell').apply(lambda d: np.average(d.dlog,weights=None)).rename('mdlog')
for m in A.method.unique():
    b=slope(L[L.method!=m])
    for cond in ('AE','S2'):
        a=A[(A.method==m)&(A.cond==cond)].copy()
        a['pred']=b*cellmean.reindex(a.cell).to_numpy()
        a['resid']=a.d_ba_ge30-a.pred
        cv=lambda col: a.groupby(['clf','k','target'])[col].mean().groupby('target').mean()
        rows.append(dict(method=m,cond=cond,b_lomo=round(b,2),obs=fmt(ci(cv('d_ba_ge30'))),pred=round(cv('pred').mean(),2),
                         resid=fmt(ci(cv('resid'))),obs_P2=round(cv('d_ba_P2').mean(),2),d_oa=round(cv('d_oa').mean(),2)))
T=pd.DataFrame(rows); print(T.to_string(index=False)); T.to_csv('a3_dose_decomp.csv',index=False)
L.to_pickle('L.pkl')
# 各方法的 ΔBA 与构成预测的相关（方法×条件×k 32 点）
