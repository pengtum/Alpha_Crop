import sys; sys.path.insert(0,'.')
from load import ci, fmt
from wls import slope, slope_ctry
import numpy as np, pandas as pd
pd.set_option('display.width',260)
A=pd.read_pickle('A.pkl'); A=A[A.cond.isin(['AE','S2'])].reset_index(drop=True); A['cell']=np.arange(len(A))
L=pd.read_pickle('L.pkl')
L9=L[~L.target.isin(['RO','PT'])]
r=slope_ctry(L9); print(f"9 国剂量斜率：合并 {r['pooled']:+.2f}，逐国 {r['mean']:+.2f} [{r['lo']:+.2f},{r['hi']:+.2f}] {r['npos']}/{r['n']}")
for t in ['RO','PT']:
    print(t,'剂量斜率（合并）', round(slope(L[L.target==t]),2))
reg={'9国':~A.target.isin(['RO','PT']),'RO':A.target=='RO','PT':A.target=='PT'}
rows=[]
for m in ['predstrat_srs','margin','gist_margin','badge_c','kcenter','typiclust','kherding','lpm2_nogeo']:
    Lm=L9[L9.method!=m]; b=slope(Lm)
    pc=L[L.method==m].groupby('cell').dlog.mean()
    for g,mask in reg.items():
        a=A[(A.method==m)&mask].copy()
        a['pred']=b*pc.reindex(a.cell).to_numpy(); a['res']=a.d_ba_ge30-a.pred
        cv=lambda col: a.groupby(['cond','clf','k','target'])[col].mean().groupby('target').mean()
        rr=dict(method=m,regime=g)
        for col in ['d_ba_ge30','d_ba_P2','d_oa','d_oa_P2','pred','res','d_comp_L1','d_LD']:
            v=cv(col); rr[col]=fmt(ci(v)) if (g=='9国' and col in('d_ba_ge30','d_oa','res')) else round(v.mean(),2)
        rows.append(rr)
T=pd.DataFrame(rows); print(T.to_string(index=False)); T.to_csv('a6_regime.csv',index=False)
