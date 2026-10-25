import sys; sys.path.insert(0,'.')
from load import ci, fmt
from wls import slope, slope_ctry
import numpy as np, pandas as pd
pd.set_option('display.width',250)
CL=["grassland","maize","orchards","other_crops","potatoes","root_crops","strawberries","tree_wood_forest","vegetables","winter_cereals"]
L=pd.read_pickle('L.pkl'); A=pd.read_pickle('A.pkl'); A=A[A.cond.isin(['AE','S2'])].reset_index(drop=True); A['cell']=np.arange(len(A))
print('逐类斜率（格内去均值；全部方法）：')
for c in range(10):
    d=L[L.cls==c]
    if len(d)<100: continue
    r=slope_ctry(d)
    print(f"  {CL[c]:18s} 合并 {r['pooled']:+.2f}  逐国 {r['mean']:+.2f} [{r['lo']:+.2f},{r['hi']:+.2f}] {r['npos']}/{r['n']}  n={len(d)}")
# 留一方法 + 逐类斜率的构成预测
rows=[]
for m in A.method.unique():
    Lm=L[L.method!=m]
    bc={c:slope(Lm[Lm.cls==c]) for c in range(10) if (Lm.cls==c).sum()>100}
    b0=slope(Lm)
    d=L[L.method==m].copy()
    d['p']=d.dlog*d.cls.map(bc).fillna(b0)
    d['p0']=d.dlog*b0
    pc=d.groupby('cell')[['p','p0']].mean()
    for cond in ('AE','S2'):
        a=A[(A.method==m)&(A.cond==cond)].copy()
        a['pred_c']=pc.p.reindex(a.cell).to_numpy(); a['pred_0']=pc.p0.reindex(a.cell).to_numpy()
        a['res_c']=a.d_ba_ge30-a.pred_c
        cv=lambda col: a.groupby(['clf','k','target'])[col].mean().groupby('target').mean()
        rows.append(dict(method=m,cond=cond,obs=round(cv('d_ba_ge30').mean(),2),pred_common_b=round(cv('pred_0').mean(),2),
                         pred_class_b=round(cv('pred_c').mean(),2),resid_class_b=fmt(ci(cv('res_c')))))
T=pd.DataFrame(rows); print(T.to_string(index=False)); T.to_csv('a5_decomp_classb.csv',index=False)
