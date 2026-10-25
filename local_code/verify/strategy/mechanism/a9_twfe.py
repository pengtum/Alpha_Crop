"""剂量斜率的稳健性：双向固定效应（选样格 × [单元,分类器,k,类]），交替去均值；另做只用 E1 几何方法、只用预测/混合方法的版本。"""
import sys; sys.path.insert(0,'.')
from wls import slope, slope_ctry
import numpy as np, pandas as pd
L=pd.read_pickle('L.pkl'); A=pd.read_pickle('A.pkl'); A=A[A.cond.isin(['AE','S2'])].reset_index(drop=True); A['cell']=np.arange(len(A))
L=L.merge(A[['cell','split','rep']],on='cell')
L['grp']=L.target+'_'+L.split.astype(str)+'_'+L.rep.astype(str)+'_'+L.clf+'_'+L.k.astype(str)+'_'+L.cls.astype(str)
def twfe(df):
    d=df.copy()
    for v in ('drec','dlog'):
        x=d[v].copy()
        for _ in range(30):
            x=x-x.groupby(d.cell).transform('mean'); x=x-x.groupby(d.grp).transform('mean')
        d[v+'_w']=x
    return d
for name,mask in (('全部方法',np.ones(len(L),bool)),('只用 E1 几何 4 法',L.method.isin(['lpm2_nogeo','kcenter','typiclust','kherding'])),
                  ('只用预测/混合 4 法',L.method.isin(['margin','predstrat_srs','gist_margin','badge_c'])),('9 国',~L.target.isin(['RO','PT']))):
    d=twfe(L[mask]); r=slope_ctry(d)
    print(f"{name:18s} 双向 FE：合并 {r['pooled']:+.2f}；逐国 {r['mean']:+.2f} [{r['lo']:+.2f},{r['hi']:+.2f}] {r['npos']}/{r['n']}  n={len(d)}")
