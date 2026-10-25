import pandas as pd, numpy as np
from scipy import stats
pd.set_option('display.width',250); pd.set_option('display.max_columns',50)
D = pd.read_pickle('G_all.pkl')
B = D[(D.cond=='none')].copy()
def ctry_ci(x):
    x=np.asarray(x); m=x.mean(); se=x.std(ddof=1)/np.sqrt(len(x)); t=stats.t.ppf(.975,len(x)-1)
    return f'{m:+.2f} [{m-t*se:+.2f},{m+t*se:+.2f}] {int((x>0).sum())}/{len(x)}'
print('##### (c) 先验修正：SRS 行上 P1/P2 − P0（11 国均值，国家层面 CI）')
for clf in ['s2','ae']:
    for k in [0,50,200,400]:
        b=B[(B.clf==clf)&(B.k==k)].groupby('target')[['ba_ge30','oa','ba_P1','oa_P1','ba_P2','oa_P2']].mean()
        print(clf,k,'dBA_P1',ctry_ci(b.ba_P1-b.ba_ge30),'| dOA_P1',ctry_ci(b.oa_P1-b.oa),'| dBA_P2',ctry_ci(b.ba_P2-b.ba_ge30),'| dOA_P2',ctry_ci(b.oa_P2-b.oa))
print('prior_src by k:', B.groupby('k').prior_src.agg(lambda s: s.value_counts().to_dict()).to_dict())
print('\n##### (c) 训练轮数：i200/i250 − i300（SRS 行）')
for clf in ['s2','ae']:
    for k in [0,50,200,400]:
        b=B[(B.clf==clf)&(B.k==k)].groupby('target')[['ba_i200','ba_i250','ba_i300','oa_i200','oa_i300']].mean()
        print(clf,k,'BA i200-i300',ctry_ci(b.ba_i200-b.ba_i300),'| OA i200-i300',ctry_ci(b.oa_i200-b.oa_i300))
print('\n##### (a/b) 各方法 − SRS（同单元配对；ba_ge30、OA、comp_L1；11 国均值 CI）')
M = D[(D.tier=='t1')&(D.cond.isin(['AE','S2']))]
key=['target','split','rep','clf','k']
S = B[B.k.isin([50,200])].set_index(key)[['ba_ge30','oa','comp_L1','n_classes_sel']]
out=[]
for (cond,meth,k),g in M.groupby(['cond','method','k']):
    g=g.groupby(key)[['ba_ge30','oa','comp_L1','n_classes_sel']].mean()  # average over cfg dupes if any
    j=g.join(S,rsuffix='_srs',how='inner')
    j['dba']=j.ba_ge30-j.ba_ge30_srs; j['doa']=j.oa-j.oa_srs; j['dcomp']=j.comp_L1-j.comp_L1_srs
    c=j.groupby('target')[['dba','doa','dcomp']].mean()
    out.append(dict(cond=cond,method=meth,k=k,dBA=ctry_ci(c.dba),dOA=ctry_ci(c.doa),dcompL1=round(c.dcomp.mean(),3),
                    best_ctry_dBA=round(c.dba.max(),2),worst=round(c.dba.min(),2)))
print(pd.DataFrame(out).to_string(index=False))
print('\n##### SRS 单元间 SD（同国同 k 同 clf，4 单元）与配对差 SD')
sd=B[B.k.isin([50,200])].groupby(['clf','k','target']).ba_ge30.std().groupby(['clf','k']).mean()
print('SRS 单元间 SD 均值', sd.round(2).to_dict())
