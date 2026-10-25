import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
P = '/Users/perng_mini/AlphaCrops_local/results/preds'; D = '/Users/perng_mini/AlphaCrops_local/results/check3'; C=10
M = np.load('meta.npz', allow_pickle=True); y = M['y']
R = pd.read_pickle('b1_rows.pkl'); A = R[R.kind=='arms'].set_index(['feat','target'])
sel = pd.read_csv('/Users/perng_mini/AlphaCrops_local/results/pilot_selection.csv')
def bage(yt, yh):
    n = np.bincount(yt, minlength=C); return 100*np.mean([(yh[yt==c]==c).mean() for c in np.flatnonzero(n>=30)])
rows=[]
for t in ['PT','SE','SK','LV']:
    te2 = np.load(f'{P}/{t}_test_idx.npy'); sp = np.load(f'{D}/{t}_split.npz')
    same_test = (len(te2)==len(sp['test'])) and (te2==sp['test']).all()
    print(t, 'check2 test == check3 test:', same_test, len(te2), len(sp['test']), ' 集合相同:', set(te2)==set(sp['test']))
    yt = y[te2]
    for f in ('ae','s2'):
        for strat, ks in (('foreign_only',[0]),('random',[50,200,1000]),('target_full',[None]),('ceiling_w25',[None])):
            for k in ks:
                for s in range(3):
                    kk = k if k is not None else sel[(sel.target==t)&(sel.strategy==strat)].budget.iloc[0]
                    p = np.load(f'{P}/{t}_{strat}_{kk}_{s}_{f}.npy')
                    rows.append(dict(target=t, feat=f, strat=strat, k=kk if strat=='random' else 0, seed=s, oa=100*(p==yt).mean(), ba=bage(yt,p)))
df = pd.DataFrame(rows); df.to_pickle('b4_rows.pkl')
# 核对 ba_ge30 与 csv
m = df.merge(sel.rename(columns={'strategy':'strat','clf_feat':'feat'})[['target','strat','budget','seed','feat','ba_ge30']], left_on=['target','strat','seed','feat'], right_on=['target','strat','seed','feat'])
m = m[(m.strat!='random')|(m.k==m.budget)]
print('ba_ge30 与 pilot_selection.csv 最大差:', (m.ba - 100*m.ba_ge30).abs().max())
g = df.groupby(['feat','target','strat','k']).oa.agg(['mean','std']).round(2)
print(g.to_string())
