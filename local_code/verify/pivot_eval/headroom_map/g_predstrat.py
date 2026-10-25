import pandas as pd, numpy as np
D = pd.read_pickle('G_all.pkl')
B = D[(D.cond=='none')&(D.k.isin([50,200]))]
key=['target','split','rep','clf','k']
S=B.set_index(key)[['ba_ge30','oa','comp_L1']+[f'rec{i}' for i in range(10)]]
for cond in ['AE','S2']:
    P=D[(D.cond==cond)&(D.method=='predstrat_srs')].set_index(key)[['ba_ge30','oa','comp_L1']+[f'rec{i}' for i in range(10)]]
    j=P.join(S,rsuffix='_s',how='inner')
    j['dba']=j.ba_ge30-j.ba_ge30_s; j['doa']=j.oa-j.oa_s
    t=j.groupby(['k','target'])[['dba','doa']].mean().unstack('k').round(2)
    print(cond); print(t.to_string())
    for k in [50,200]:
        c=j.xs(k,level='k').groupby('target')[['dba','doa']].mean()
        print(k,'mean',c.dba.mean().round(2),'noRO_PT',c.drop(['RO','PT']).dba.mean().round(2),'noRO',c.drop(['RO']).dba.mean().round(2), 'median', c.dba.median().round(2))
    # per-class recall change pooled (class idx)
    for k in [50,200]:
        jj=j.xs(k,level='k')
        print(k,'逐类召回变化(pp, 各国均值; NaN 忽略):', {i: round(100*np.nanmean(jj[f'rec{i}']-jj[f'rec{i}_s']),2) for i in range(10)})
