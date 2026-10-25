"""选样类别构成：原 ORACLE vs 按方案 z 截断 ORACLE vs S2 vs AE（重复 0，lpm2_nogeo/typiclust）。"""
import numpy as np, os, sys, pandas as pd
sys.path.insert(0, '.')
import guidance_G as G
from stageA import MIN_N
D = G.GData()
AUD = '/Users/perng_mini/AlphaCrops_local/verify/G_audit/oracle_prep_check'
rows = []
for tgt in D.countries:
    for split in (0, 1):
        S = dict(np.load(os.path.join(G.OUT, 'stage0', f'{tgt}_s{split}.npz')))
        pp = G.pool_positions(S, split, 0); idx = S['allT'][pp]; yP = D.y[idx]
        yT = D.y[S['test']]; big = np.flatnonzero(np.bincount(yT, minlength=10) >= MIN_N)
        piP = np.bincount(yP, minlength=10) / len(yP)
        for m in ('lpm2_nogeo', 'typiclust'):
            for k in (50, 200):
                u = f'{tgt}_s{split}_r0'
                src = {'ORIG': f'{G.OUT}/sel/{u}_ORACLE_{m}_k{k}.npz', 'FIX': f'{AUD}/sel_fixed/{u}_ORACLEfix_{m}_k{k}.npz',
                       'S2': f'{G.OUT}/sel/{u}_S2_{m}_k{k}.npz', 'AE': f'{G.OUT}/sel/{u}_AE_{m}_k{k}.npz'}
                for c, p in src.items():
                    ys = yP[np.load(p)['sel_pos']]
                    cnt = np.bincount(ys, minlength=10)
                    q = cnt / k
                    rows.append(dict(tgt=tgt, split=split, method=m, k=k, cond=c,
                                     big_cov=np.mean(cnt[big] > 0), ncls=int((cnt > 0).sum()),
                                     L1=np.abs(q - piP).sum(), minor_share=q[big][piP[big] < np.sort(piP[big])[-1]].sum()))
df = pd.DataFrame(rows); df.to_csv(AUD + '/comp.csv', index=False)
pv = df.groupby(['method', 'k', 'cond'])[['big_cov', 'ncls', 'L1', 'minor_share']].mean().round(3)
print(pv)
print(df[df.tgt.isin(['PT', 'LV', 'RO', 'LT'])].groupby(['tgt', 'method', 'cond'])[['big_cov', 'L1', 'minor_share']].mean().round(3))
