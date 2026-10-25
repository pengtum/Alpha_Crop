import pandas as pd, numpy as np
pd.set_option('display.width',250)
C=pd.read_csv('/Users/perng_mini/AlphaCrops_local/results/check3_capacity_arms.csv')
A=pd.read_csv('/Users/perng_mini/AlphaCrops_local/results/check3_arms.csv')
print(sorted(A.arm.unique())); print(A.split.unique())
w=C[C.arm.isin(['L0','W0','Wnat'])].pivot_table(index=['feat','target'],columns=['cap','arm'],values=['ba_ge30','oa'])
D=pd.read_pickle('G_all.pkl')
B=D[D.cond=='none'].groupby(['clf','target','k'])[['ba_ge30','oa']].mean()
ps=D[(D.method=='predstrat_srs')&(D.cond=='AE')].groupby(['clf','target','k'])[['ba_ge30','oa']].mean()
rows=[]
for feat in ['ae','s2']:
    for t in sorted(D.target.unique()):
        g0=B.loc[(feat,t,0),'ba_ge30']; g200=B.loc[(feat,t,200),'ba_ge30']; g400=B.loc[(feat,t,400),'ba_ge30']
        p200=ps.loc[(feat,t,200),'ba_ge30']
        W0es=w.loc[(feat,t),('ba_ge30','es','W0')]; W0c=w.loc[(feat,t),('ba_ge30','c300','W0')]
        L0es=w.loc[(feat,t),('ba_ge30','es','L0')]; L0c=w.loc[(feat,t),('ba_ge30','c300','L0')]
        W0max=max(W0es,W0c)
        rows.append(dict(feat=feat,target=t,G_k0=g0,G_srs200=g200,G_srs400=g400,G_ps200=p200,chk3_L0_c300=L0c,chk3_W0_es=W0es,chk3_W0_c300=W0c,
                         gap=W0max-g0,frac_srs400=(g400-g0)/(W0max-g0),frac_ps200=(p200-g0)/(W0max-g0)))
R=pd.DataFrame(rows)
print(R.round(2).to_string(index=False))
print(R.groupby('feat')[['G_k0','G_srs400','G_ps200','chk3_W0_es','chk3_W0_c300','gap','frac_srs400','frac_ps200']].mean().round(3))
print(R[~R.target.isin(['RO','PT'])].groupby('feat')[['gap','frac_srs400','frac_ps200']].mean().round(3))
