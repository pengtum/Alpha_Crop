import sys, os
sys.path.insert(0, "/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops/label_expansion")
os.chdir(sys.path[0])
import numpy as np, pandas as pd
from scipy import stats
import analyze_G as A
import guidance_G as G
from guidance_G import PRED, pool_positions
OUT = "/Users/perng_mini/AlphaCrops_local/results/guidance_G"
V = "/Users/perng_mini/AlphaCrops_local/verify/G_audit/pred"
R = A.load_rows(OUT)
# 1) method - SRS for pred methods under AE and S2 (rep0)
for cond in ('AE','S2','ORACLE'):
    for m in PRED:
        d = A.vs_srs(R[R.rep==0], cond, [m])
        if len(d)==0: continue
        print(f'{cond} {m} - SRS:', A.fmt(A.t_summary(A.country_values(d))))
# 2) ORACLE vs S2 / AE for margin
for a,b in (('ORACLE','S2'),('ORACLE','AE'),('AE','S2')):
    print(f'margin {a}-{b}:', A.fmt(A.t_summary(A.country_values(A.deltas(R,a,b,['margin'],reps=[0])))))
# 3) realized composition from mediators & rows, pred methods
M = pd.read_csv(f'{OUT}/mediators_G.csv')
Mp = M[M.method.isin(PRED)]
W = Mp.pivot_table(index=['target','split','method','k'], columns='cond', values=['comp_L1','n_classes','jaccard','nn_km'])
Dm = pd.DataFrame({c: W[(c,'AE')]-W[(c,'S2')] for c in ['comp_L1','n_classes','nn_km']})
Dm['jaccard'] = W[('jaccard','AE')]
print('\nrealized AE-S2 mediator diffs (pred), mean by method,k:'); print(Dm.groupby(['method','k']).mean().round(4).to_string())
for c in ['comp_L1','n_classes']:
    for m in PRED:
        v = Dm.xs(m, level='method')[c].groupby('target').mean()
        print(f'  {m} Δ{c} country t:', A.fmt(A.t_summary(v), ''))
d = A.deltas(R,'AE','S2',PRED,reps=[0]).groupby(['target','split','method','k']).d.mean()
X = Dm.join(d)
for m in PRED:
    g = X.xs(m, level='method')
    for c in ['comp_L1','n_classes']:
        r,p = stats.spearmanr(g[c], g.d); print(f'  {m} Δba~Δ{c} (n={len(g)}) rho={r:+.2f} p={p:.3f}')
g = X
for c in ['comp_L1','n_classes']:
    r,p = stats.spearmanr(g[c], g.d); print(f'  both Δba~Δ{c} (n={len(g)}) rho={r:+.2f} p={p:.3f}')
# rows comp_L1 consistency with mediators
rr = R[(R.method.isin(PRED))&R.cond.isin(['AE','S2'])&(R.clf=='s2')].set_index(['target','split','method','k','cond']).comp_L1
mm = Mp.set_index(['target','split','method','k','cond']).comp_L1
print('\nrows comp_L1 vs mediators comp_L1 max abs diff:', float((rr - mm.reindex(rr.index)).abs().max()))
# 4) verify saved margin selection == recomputed top-k from stage0 P_T
D = G.GData()
bad = 0; n=0
for tgt in sorted(R.target.unique()):
    for split in (0,1):
        S = np.load(f'{OUT}/stage0/{tgt}_s{split}.npz')
        pp = pool_positions(S, split, 0)
        for cond, f in (('AE','ae'),('S2','s2')):
            PT = S[f'P_T_{f}'][pp].astype(float); s = np.sort(PT,1); unc = 1-(s[:,-1]-s[:,-2])
            for k in (50,200):
                z = np.load(f'{OUT}/sel/{tgt}_s{split}_r0_{cond}_margin_k{k}.npz')
                mine = set(np.argsort(-unc, kind='stable')[:k]); n+=1
                if set(z['sel_pos'].tolist()) != mine: bad += 1
                # predstrat: strata allocation check
                zp = np.load(f'{OUT}/sel/{tgt}_s{split}_r0_{cond}_predstrat_srs_k{k}.npz')
                yh = PT.argmax(1)
                if k==50 and tgt=='RO' and split==0:
                    print(cond, 'predstrat strata counts in selection', np.bincount(yh[zp['sel_pos']], minlength=10), 'pool argmax', np.bincount(yh, minlength=10))
print(f'margin selections not matching recomputation: {bad}/{n}')
