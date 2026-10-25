import sys, numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, '.')
import analyze_G as A
from guidance_G import E1
R = A.load_rows()
d = A.deltas(R, 'AE', 'S2', E1)
# ---- cell-level |d|
print('cells', len(d), 'P|d|>0.3 %.3f  P|d|>1 %.3f  sd %.3f'%((d.d.abs()>0.3).mean(), (d.d.abs()>1).mean(), d.d.std()))
# noise: deviations of cell from its (country,method,clf,k) mean across 4 units
g = d.groupby(['target','method','clf','k']).d
res = d.d - g.transform('mean')
s_noise = np.sqrt((res**2).sum()/(len(d)-g.ngroups))
print('within (c,m,clf,k) residual sd across units (pooled, df=%d): %.3f'%(len(d)-g.ngroups, s_noise))
for thr in (0.3,1.0):
    print(' null N(0,%.3f): P(|d|>%.1f)=%.3f'%(s_noise,thr,2*stats.norm.sf(thr/s_noise)))
# null by permutation: sign-flip each unit's cell deltas? simulate zero-mean with empirical residuals
rng=np.random.default_rng(0)
r=res.to_numpy()*np.sqrt(4/3)  # inflate to per-observation sd
sim=rng.choice(r,(2000,len(r)))*rng.choice([-1,1],(2000,len(r)))
print(' resampled-residual null: P(|d|>0.3)=%.3f  P(|d|>1)=%.3f'%((np.abs(sim)>0.3).mean(),(np.abs(sim)>1).mean()))
# ---- country x method at unit level (avg over clf,k)
cm = d.groupby(['target','method','split','rep']).d.mean().rename('y').reset_index()
cm['unit']=cm.target+'_'+cm.split.astype(str)+cm.rep.astype(str)
gm=cm.y.mean()
mC=cm.groupby('target').y.mean(); mM=cm.groupby('method').y.mean(); mCM=cm.groupby(['target','method']).y.mean()
mU=cm.groupby('unit').y.mean()
nC,nM,nU=11,4,4
SS_C = nM*nU*((mC-gm)**2).sum()
SS_M = nC*nU*((mM-gm)**2).sum()
SS_CM = nU*((mCM - mC.reindex(mCM.index.get_level_values(0)).to_numpy() - mM.reindex(mCM.index.get_level_values(1)).to_numpy() + gm)**2).sum()
cm['fit']= mCM.reindex(pd.MultiIndex.from_frame(cm[['target','method']])).to_numpy() + mU.reindex(cm.unit).to_numpy() - mC.reindex(cm.target).to_numpy()
SS_E = ((cm.y-cm.fit)**2).sum()   # M x U(C)
SS_U = nM*((mU - mC.reindex([u.split('_')[0] for u in mU.index]).to_numpy())**2).sum()
df_C,df_M,df_CM,df_U,df_E=10,3,30,33,99
print('check SS total', ((cm.y-gm)**2).sum(), SS_C+SS_M+SS_CM+SS_U+SS_E)
MS = dict(C=SS_C/df_C, M=SS_M/df_M, CM=SS_CM/df_CM, U=SS_U/df_U, E=SS_E/df_E)
print({k:round(v,4) for k,v in MS.items()})
F=MS['CM']/MS['E']; print('C x M vs M x U(C): F(30,99)=%.3f p=%.4g'%(F, stats.f.sf(F,30,99)))
F=MS['M']/MS['CM']; print('Method main vs CxM: F(3,30)=%.3f p=%.4f'%(F, stats.f.sf(F,3,30)))
F=MS['C']/MS['U']; print('Country vs U(C): F(10,33)=%.3f p=%.4f'%(F, stats.f.sf(F,10,33)))
# naive cell level interaction (as refute might have done): 704 cells, residual pooled
import itertools
# country x method means and unit-level t
T = cm.groupby(['target','method']).y.agg(['mean','std','count'])
T['t']=T['mean']/(T['std']/np.sqrt(T['count']))
T['p']=2*stats.t.sf(T.t.abs(),3)
T['npos_units']=cm.assign(pos=cm.y>0).groupby(['target','method']).pos.sum()
cc = d.assign(pos=d.d>0).groupby(['target','method']).pos.agg(['sum','count'])
T['cells_pos']=cc['sum'].astype(str)+'/'+cc['count'].astype(str)
print(T.sort_values('mean').round(3).to_string())
from scipy.stats import false_discovery_control
T['q']=false_discovery_control(T.p)
print('n p<0.05:', (T.p<0.05).sum(), 'of', len(T), ' n q<0.10:', (T.q<0.10).sum(), 'expected under null', 0.05*len(T))
print(T[T.p<0.05].round(4))
