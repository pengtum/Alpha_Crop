import sys; sys.path.insert(0,'.')
from load import ci, fmt
from wls import slope
import numpy as np, pandas as pd
pd.set_option('display.width',250)
A=pd.read_pickle('A.pkl'); L=pd.read_pickle('L.pkl')
b=slope(L); b9=slope(L[~L.target.isin(['RO','PT'])])
# 对 ORACLE / ORACLE2 以及同单元（rep0）的 S2 做构成分解（ORACLE 的逐类 dlog 需另算）
refX=pd.read_pickle('X.pkl'); ref=refX[refX.method=='srs'].set_index(['target','split','rep','clf','k'])
def mdlog(a):
    kr=pd.MultiIndex.from_frame(a[['target','split','rep','clf','k']]); out=[]
    vals=[]
    for c in range(10):
        big=(a[f'nT{c}']>=30).to_numpy()
        dl=np.log1p(a[f'n{c}'].to_numpy())-np.log1p(ref[f'n{c}'].reindex(kr).to_numpy())
        vals.append(np.where(big,dl,np.nan))
    return np.nanmean(np.vstack(vals),0)
A=A[A.rep==0].copy(); A['mdlog']=mdlog(A)
Q=A[A.method.isin(['margin','typiclust','lpm2_nogeo'])&A.cond.isin(['S2','AE','ORACLE','ORACLE2'])]
rows=[]
for (m,c),a in Q.groupby(['method','cond']):
    for g,mask in (('11国',np.ones(len(a),bool)),('9国',~a.target.isin(['RO','PT']).to_numpy())):
        aa=a[mask].copy(); bb=b if g=='11国' else b9
        aa['pred']=bb*aa.mdlog; aa['res']=aa.d_ba_ge30-aa.pred
        cv=lambda col: aa.groupby(['clf','k','target'])[col].mean().groupby('target').mean()
        rows.append(dict(method=m,cond=c,regime=g,dBA=fmt(ci(cv('d_ba_ge30'))),dOA=round(cv('d_oa').mean(),2),dOA_P2=round(cv('d_oa_P2').mean(),2),
                         dcomp=round(cv('d_comp_L1').mean(),3),dmaj=round(cv('d_maj_share').mean(),3),pred=round(cv('pred').mean(),2),res=fmt(ci(cv('res')))))
print(pd.DataFrame(rows).to_string(index=False))
# ORACLE margin 的逐类份额变化
CL=["grass","maize","orch","other","potat","root","straw","tree","veg","wcer"]
for c_ in ('S2','AE','ORACLE'):
    a=A[(A.method=='margin')&(A.cond==c_)]
    print(c_,'margin 逐类份额变化（pp of k）',{CL[c]:round(100*(a[f'd_n{c}']/a.k).mean(),1) for c in range(10)})
