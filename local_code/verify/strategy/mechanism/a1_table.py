import sys; sys.path.insert(0,'.')
from load import *
pd.set_option('display.width',250)
R=load()
A=vs_srs(R)
rows=[]
for (m,c),g in A.groupby(['method','cond']):
    r=dict(method=m,cond=c)
    for met in ['ba_ge30','ba_P2','oa','oa_P2','f1_ge30','ba_far','comp_L1','n_classes_sel']:
        v=cval(g,'d_'+met); d=ci(v); r[met]=fmt(d,2) if met not in('comp_L1',) else f"{d['mean']:+.3f}"
    r['comp_abs']=round(g.comp_L1.mean(),3); r['srs_comp']=round(g.srs_comp_L1.mean(),3)
    rows.append(r)
T=pd.DataFrame(rows); print(T.to_string(index=False))
T.to_csv('a1_method_vs_srs.csv',index=False)
# by k
print()
rows=[]
for (m,c,k),g in A.groupby(['method','cond','k']):
    r=dict(method=m,cond=c,k=k)
    for met in ['ba_ge30','ba_P2','oa','oa_P2']:
        r[met]=fmt(ci(cval(g,'d_'+met)),2)
    r['dcomp']=round(g.d_comp_L1.mean(),3); r['dncls']=round(g.d_n_classes_sel.mean(),2)
    rows.append(r)
T=pd.DataFrame(rows); print(T.to_string(index=False)); T.to_csv('a1_method_vs_srs_byk.csv',index=False)
