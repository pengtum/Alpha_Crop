import sys, os
sys.path.insert(0, "/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops/label_expansion")
os.chdir(sys.path[0])
import numpy as np, pandas as pd
from scipy import stats
import analyze_G as A
import guidance_G as G
from guidance_G import PRED, E1, HYB, pool_positions
OUT = "/Users/perng_mini/AlphaCrops_local/results/guidance_G"
V = "/Users/perng_mini/AlphaCrops_local/verify/G_audit/pred"
R = A.load_rows(OUT)
D = G.GData()
rows = []
for tgt in sorted(R.target.unique()):
  for split in (0,1):
    S = dict(np.load(f'{OUT}/stage0/{tgt}_s{split}.npz'))
    pp = pool_positions(S, split, 0)
    idx = S['allT'][pp]; y = D.y[idx]
    yT = D.y[S['test']]
    big = np.flatnonzero(np.bincount(yT, minlength=10) >= 30)
    pi_pool = np.bincount(y, minlength=10)/len(y)
    for f in ('ae','s2'):
        PT = S[f'P_T_{f}'][pp].astype(float)
        yh = PT.argmax(1)
        Nh = np.bincount(yh, minlength=10)
        acc = (yh==y).mean()
        # per-class recall of argmax on pool, BA on big classes
        rec = np.array([(yh[y==c]==c).mean() if (y==c).any() else np.nan for c in range(10)])
        # predstrat expected composition (sqrt allocation, m_min=2) using actual _alloc
        from samplers.designs import _alloc, predstrat_weights
        for k in (50,200):
            w = predstrat_weights(PT, yh, 10, 'sqrt')
            nh = _alloc(Nh, w, k, 2)
            # expected true-class composition: sum_h nh * P(y=c|h)
            M = np.zeros((10,10))
            for h in range(10):
                if Nh[h]>0: M[h] = np.bincount(y[yh==h], minlength=10)/Nh[h]
            ec = (nh[:,None]*M).sum(0)/k
            # margin selection composition (deterministic)
            s = np.sort(PT,1); unc = 1-(s[:,-1]-s[:,-2])
            sel_m = np.argsort(-unc, kind='stable')[:k]
            mc = np.bincount(y[sel_m], minlength=10)/k
            for meth, comp in (('predstrat_srs_expected', ec), ('margin', mc)):
                rows.append(dict(target=tgt, split=split, feat=f, k=k, method=meth,
                    comp_L1=np.abs(comp-pi_pool).sum(),
                    big_min_share=comp[big].min(), big_cov=(comp[big]*k>=1).sum()/len(big),
                    big_share=comp[big].sum(),
                    minority_share=comp[big][pi_pool[big] < np.median(pi_pool[big])].sum(),
                    ent=-(comp[comp>0]*np.log(comp[comp>0])).sum(),
                    pool_argmax_acc=acc, pool_argmax_ba=np.nanmean(rec[big]),
                    strata_purity=(nh/k * M.max(1)).sum()))
            # margin top-k precision: fraction of margin-selected that are misclassified by guidance model
            rows[-1]['margin_err_rate'] = (yh[sel_m]!=y[sel_m]).mean()
Q = pd.DataFrame(rows)
Q.to_csv(f'{V}/mech_comp.csv', index=False)
W = Q.pivot_table(index=['target','split','k','method'], columns='feat', values=['comp_L1','big_min_share','big_cov','big_share','minority_share','ent','pool_argmax_acc','pool_argmax_ba','strata_purity','margin_err_rate'])
Dl = pd.DataFrame({c: W[(c,'ae')]-W[(c,'s2')] for c in ['comp_L1','big_min_share','big_cov','big_share','minority_share','ent','pool_argmax_acc','pool_argmax_ba','strata_purity','margin_err_rate']})
print('Mean AE-S2 differences in selection composition (unit mean over 11x2):')
print(Dl.groupby(['method','k']).mean().round(4).T.to_string())
print('\nAE / S2 levels:')
print(W.groupby(['method','k']).mean().round(4).T.to_string())
# observed deltas at unit (target,split) level
d = A.deltas(R,'AE','S2',PRED,reps=[0])
du = d.groupby(['target','split','method','k']).d.mean().reset_index()
du['method'] = du.method.replace({'predstrat_srs':'predstrat_srs_expected'})
X = Dl.reset_index().merge(du, on=['target','split','method','k'])
print('\nSpearman/Pearson corr between unit Δba (avg clf) and Δmediator, by method (n=44: 22 units x 2 k):')
for m, g in X.groupby('method'):
    for c in ['comp_L1','big_min_share','big_share','minority_share','ent','pool_argmax_ba','strata_purity']:
        r, p = stats.spearmanr(g[c], g.d); rp, pp_ = stats.pearsonr(g[c], g.d)
        print(f'  {m:24s} {c:16s} rho={r:+.2f} p={p:.3f} | r={rp:+.2f} p={pp_:.3f}')
X.to_csv(f'{V}/mech_merge.csv', index=False)
