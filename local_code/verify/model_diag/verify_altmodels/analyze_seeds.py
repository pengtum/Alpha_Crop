"""s1r0 k=200：MLP 3 种子平均（mlpF_ens_e20，预先指定；另报 e40）vs LightGBM 3 种子平均；以及各自单种子。国家层面（ae/s2 平均）。"""
import numpy as np, pandas as pd, glob, os
from scipy import stats
V = '/Users/perng_mini/AlphaCrops_local/verify/model_diag/verify_altmodels/'
TAUS = np.round(np.arange(-3, 3.001, 0.05), 2); TAU = np.round(np.arange(-3, 3.001, 0.1), 2)
def tci(v):
    v = np.asarray(v, float); v = v[~np.isnan(v)]; n = len(v)
    if n < 2: return f'n={n}'
    m = v.mean(); h = stats.t.ppf(.975, n-1) * v.std(ddof=1) / np.sqrt(n)
    return f'{m:+.2f} [{m-h:+.2f}, {m+h:+.2f}] {int((v>0).sum())}/{n}'
def env(C, x):
    ok = C[:, 1] >= x - 1e-9
    return C[ok, 2].max() if ok.any() else np.nan
def pt(C, t):
    j = int(np.argmin(np.abs(C[:, 0] - t))); return C[j, 1], C[j, 2]
rows = []
for f in sorted(glob.glob(V + 'lgb_seeds_curves/*.npz')):
    tgt, feat = os.path.basename(f)[:-4].split('_')
    L = np.load(f); M = np.load(V + f'mlp_fair_curves_s1r0/{tgt}_{feat}_k200.npz')
    for lname in ('1000', 'ens'):
        for mname in ('mlpF_ens_e20', 'mlpF_ens_e40', 'mlpF_s0_e20', 'mlpES', 'lr'):
            CL, CM = L[lname], M[mname]
            oaL, baL = pt(CL, 0); oaM, baM = pt(CM, 0); oaL5, baL5 = pt(CL, .5); oaM5, baM5 = pt(CM, .5)
            rows.append(dict(target=tgt, feat=feat, lgb=lname, m=mname, baL=baL, oaL=oaL, baM=baM, oaM=oaM, baL5=baL5, oaL5=oaL5,
                             baM5=baM5, oaM5=oaM5, lenv_at_mOA=env(CL, oaM), menv_at_lOA5=env(CM, oaL5), maxL=CL[:, 2].max(), maxM=CM[:, 2].max()))
R = pd.DataFrame(rows); R.to_csv(V + 'seeds_vs_mlp_rows.csv', index=False)
print('国家数', R.target.nunique(), '（国家×特征）', R.groupby(['target', 'feat']).ngroups)
for lname in ('1000', 'ens'):
    for mname in ('mlpF_ens_e20', 'mlpF_ens_e40', 'mlpF_s0_e20', 'mlpES', 'lr'):
        g = R[(R.lgb == lname) & (R.m == mname)].groupby('target').mean(numeric_only=True)
        print(f'LGB[{lname}] {g.baL.mean():.2f}/{g.oaL.mean():.2f} τ.5 {g.baL5.mean():.2f}/{g.oaL5.mean():.2f} vs {mname} {g.baM.mean():.2f}/{g.oaM.mean():.2f}')
        print(f'   m原生−LGB(τ.5): Δba {tci(g.baM-g.baL5)} ΔOA {tci(g.oaM-g.oaL5)} | 同τ.5: Δba {tci(g.baM5-g.baL5)} ΔOA {tci(g.oaM5-g.oaL5)}')
        print(f'   包络 m原生−LGB包络@m OA {tci(g.baM-g.lenv_at_mOA)} | m包络@LGB(τ.5)OA−LGB(τ.5) {tci(g.menv_at_lOA5-g.baL5)} | τ网格最大 {tci(g.maxM-g.maxL)}')
x = R[(R.m == 'lr')].pivot_table(index=['target', 'feat'], columns='lgb', values=['baL', 'oaL', 'baL5', 'oaL5']).groupby('target').mean()
print('LGB 集成 − 单种子(1000)：原生 Δba', tci(x[('baL', 'ens')] - x[('baL', '1000')]), 'ΔOA', tci(x[('oaL', 'ens')] - x[('oaL', '1000')]),
      '| τ.5 Δba', tci(x[('baL5', 'ens')] - x[('baL5', '1000')]), 'ΔOA', tci(x[('oaL5', 'ens')] - x[('oaL5', '1000')]))
S = pd.concat([pd.read_csv(f) for f in glob.glob(V + 'lgb_seeds_rows_*.csv')])
S = S[S.seed != 'ens']
print('LGB 单种子间 ba 标准差（国家×特征）中位', S.groupby(['target', 'feat']).ba0.std().median().round(2), '最大', S.groupby(['target', 'feat']).ba0.std().max().round(2))
