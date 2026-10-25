import sys, os
sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd
from scipy import stats
import analyze_G as A
R = A.load_rows()
ms = ['lpm2_nogeo','typiclust','margin']
pd.set_option('display.width', 200)
for met in ['ba_ge30','oa','ba_P2']:
    V = pd.DataFrame({m: A.country_values(A.deltas(R,'ORACLE','S2',[m],met,reps=[0])) for m in ms})
    print(f'--- {met} ORACLE−S2 per country');
    print(V.round(3).T.to_string())
    c = V['typiclust'] - V['margin']
    print('typiclust−margin contrast:', A.fmt(A.t_summary(c)))
    c2 = V['lpm2_nogeo'] - V['margin']
    print('lpm2−margin contrast:', A.fmt(A.t_summary(c2)))
    # repeated-measures ANOVA method x country (country as block)
    X = V.to_numpy(); n, m = X.shape
    gm = X.mean(); ssm = n*((X.mean(0)-gm)**2).sum(); ssb = m*((X.mean(1)-gm)**2).sum()
    sst = ((X-gm)**2).sum(); sse = sst-ssm-ssb
    F = (ssm/(m-1))/(sse/((m-1)*(n-1))); p = stats.f.sf(F, m-1, (m-1)*(n-1))
    fr = stats.friedmanchisquare(*[V[c_] for c_ in ms])
    print(f'RM-ANOVA method effect F={F:.2f} p={p:.4f}; Friedman p={fr.pvalue:.4f}')
    print('mean of per-method country means =', V.mean(1).mean().round(4))
# ORACLE-SRS per country margin
for met in ['ba_ge30','oa']:
    d = A.vs_srs(R[R.rep==0], 'ORACLE', ['margin'], met); d = d[d.rep==0]
    print(met, 'ORACLE margin − SRS per country', A.country_values(d).round(2).to_dict())
    d = A.vs_srs(R[R.rep==0], 'ORACLE', ['typiclust'], met); d = d[d.rep==0]
    print(met, 'ORACLE typiclust − SRS per country', A.country_values(d).round(2).to_dict())
# per clf/k breakdown of ORACLE−S2
for met in ['ba_ge30','oa']:
    d = A.deltas(R,'ORACLE','S2',ms,met,reps=[0])
    print(met, d.groupby(['method','k','clf']).d.mean().round(3).unstack(['k','clf']).to_string())
