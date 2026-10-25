"""(e) 策略 × 标签用法 2×用法 表（ctrl_combo_rows.csv，C 级；AE_predstrat_srs vs SRS，同单元同 k）。"""
import sys; sys.path.insert(0,'.')
from load import ci, fmt
import numpy as np, pandas as pd
pd.set_option('display.width',250)
d=pd.read_csv('/Users/perng_mini/AlphaCrops_local/verify/pivot_eval/stat_review/ctrl_combo_rows.csv')
d=d.rename(columns={'t':'target'})
key=['target','s','r','k']
# 用法：AE 重训（ba_ae）、R2V（ba_ens）、组合 bal γ=0.25、0.4、纯头 bal γ=1
U={}
base=d[(d['mode']=='bal')&(d.gamma==0.0)].drop_duplicates(key+['lab'])
U['AE 重训']=base.set_index(key+['lab'])[['ba_ae','oa_ae']].rename(columns={'ba_ae':'ba','oa_ae':'oa'})
U['R2V 重训集成']=base.set_index(key+['lab'])[['ba_ens','oa_ens']].rename(columns={'ba_ens':'ba','oa_ens':'oa'})
for g in (0.25,0.4,1.0):
    U[f'R2V+均衡头 γ={g}']=d[(d['mode']=='bal')&(d.gamma==g)].set_index(key+['lab'])[['ba','oa']]
rows=[]
for name,T in U.items():
    T=T.reset_index()
    p=T.pivot_table(index=key,columns='lab',values=['ba','oa'])
    for k in (50,200):
        q=p.xs(k,level='k')
        dba=(q[('ba','AE_predstrat_srs')]-q[('ba','none_srs')]).groupby('target').mean()
        doa=(q[('oa','AE_predstrat_srs')]-q[('oa','none_srs')]).groupby('target').mean()
        rows.append(dict(usage=name,k=k,strategy_dBA=fmt(ci(dba)),strategy_dOA=fmt(ci(doa)),dBA_9=round(dba.drop(['RO','PT']).mean(),2),
                         SRS_BA=round(q[('ba','none_srs')].groupby('target').mean().mean(),2)))
T=pd.DataFrame(rows); print(T.to_string(index=False)); T.to_csv('e2_strategy_by_usage.csv',index=False)
# 用法效应（同一批 SRS 标签）：相对 AE 重训
print()
b=U['AE 重训'].reset_index(); 
for name in list(U)[1:]:
    t=U[name].reset_index().merge(b,on=key+['lab'],suffixes=('','_b'))
    for lab in ('none_srs','AE_predstrat_srs'):
        for k in (50,200):
            q=t[(t.lab==lab)&(t.k==k)]
            v=(q.ba-q.ba_b).groupby(q.target).mean(); o=(q.oa-q.oa_b).groupby(q.target).mean()
            print(f'{name:18s} − AE 重训 | 标签 {lab:17s} k={k}: ΔBA {fmt(ci(v))}  ΔOA {fmt(ci(o))}')
