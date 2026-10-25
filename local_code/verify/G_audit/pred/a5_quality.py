import sys, os
sys.path.insert(0, "/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops/label_expansion")
os.chdir(sys.path[0])
import numpy as np, pandas as pd
from scipy import stats
import analyze_G as A
from guidance_G import PRED
OUT = "/Users/perng_mini/AlphaCrops_local/results/guidance_G"
V = "/Users/perng_mini/AlphaCrops_local/verify/G_audit/pred"
R = A.load_rows(OUT)
Q = pd.read_csv(f'{OUT}/guidance_quality_G.csv')
print('quality means by feat:'); print(Q.groupby('feat').mean(numeric_only=True).round(4).to_string())
W = Q.pivot_table(index=['target','split'], columns='feat', values=['temp','ece_oof','nll_oof','nll_pool','auroc_margin','prior_L1_argmax','knn10_ba'])
Dq = pd.DataFrame({c: W[(c,'ae')]-W[(c,'s2')] for c in ['temp','ece_oof','nll_oof','nll_pool','auroc_margin','prior_L1_argmax','knn10_ba']})
# paired tests over countries (country mean over splits)
print('\nAE-S2 quality diffs, country-level t (n=11):')
for c in Dq:
    v = Dq[c].groupby('target').mean()
    s = A.t_summary(v); print(f'  {c:16s} mean {s["mean"]:+.4f} [{s["lo95"]:+.4f},{s["hi95"]:+.4f}] p={s["p"]:.3f} >0 {s["n_pos"]}/11')
d = A.deltas(R,'AE','S2',PRED,reps=[0])
du = d.groupby(['target','split']).d.mean()
dm = d.groupby(['target','split','method']).d.mean().unstack()
X = Dq.join(du.rename('d_pred')).join(dm)
print('\nunit-level (22 country x split) correlations Δquality vs Δba:')
for y in ['d_pred','margin','predstrat_srs']:
    for c in Dq.columns:
        r,p = stats.spearmanr(X[c], X[y]); rp,pp = stats.pearsonr(X[c], X[y])
        print(f'  {y:14s} ~ {c:16s} rho={r:+.2f} p={p:.3f} | r={rp:+.2f} p={pp:.3f}')
# country level (n=11)
Xc = X.groupby('target').mean()
print('\ncountry-level (n=11):')
for c in Dq.columns:
    r,p = stats.spearmanr(Xc[c], Xc['d_pred']); print(f'  d_pred ~ {c:16s} rho={r:+.2f} p={p:.3f}')
# add argmax pool BA diff (from mech_comp)
M = pd.read_csv(f'{V}/mech_comp.csv')
mm = M[(M.method=='margin')&(M.k==50)].pivot_table(index=['target','split'], columns='feat', values=['pool_argmax_ba','pool_argmax_acc','strata_purity'])
X2 = X.join(pd.DataFrame({c: mm[(c,'ae')]-mm[(c,'s2')] for c in ['pool_argmax_ba','pool_argmax_acc','strata_purity']}))
for c in ['pool_argmax_ba','pool_argmax_acc','strata_purity']:
    s = A.t_summary(X2[c].groupby('target').mean()); print(f'  quality {c}: AE-S2 {s["mean"]:+.4f} [{s["lo95"]:+.4f},{s["hi95"]:+.4f}] p={s["p"]:.3f} >0 {s["n_pos"]}/11')
    for y in ['d_pred','margin','predstrat_srs']:
        r,p = stats.spearmanr(X2[c], X2[y]); print(f'     {y} ~ {c} rho={r:+.2f} p={p:.3f}')
# simple multivariable OLS at unit level: d_pred ~ Δnll_pool + Δprior_L1 + Δpool_argmax_ba
import numpy.linalg as la
Z = X2[['nll_pool','prior_L1_argmax','pool_argmax_ba']].to_numpy(); Z = (Z-Z.mean(0))/Z.std(0)
Zc = np.c_[np.ones(len(Z)), Z]; yv = X2['d_pred'].to_numpy()
b, *_ = la.lstsq(Zc, yv, rcond=None); res = yv - Zc@b
print('\nOLS d_pred ~ z(Δnll_pool)+z(ΔpriorL1)+z(ΔargmaxBA): coef', b.round(3), 'R2', round(1-res.var()/yv.var(),3), 'intercept (Δ at zero quality diff approx)', round(b[0],3))
X2.to_csv(f'{V}/quality_merge.csv')
