import sys, os, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
os.chdir('/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops/label_expansion')
sys.path.insert(0, os.getcwd())
import analyze_G as A
from sklearn.isotonic import IsotonicRegression
R = A.load_rows(); KS = np.array([0, 50, 100, 200, 400.])
s = A.t_summary(A.country_values(A.deltas(R, 'AE', 'S2', A.E1)))
g = R[R.method.isin(['srs', 'foreign_only'])]
P = g.pivot_table(index=['target', 'split', 'rep'], columns=['clf', 'k'], values='ba_ge30')
# Δ=0 discontinuity check
C = pd.read_csv('/Users/perng_mini/AlphaCrops_local/verify/G_audit/labeleq_verify/C.csv')
for _, r in C.iterrows():
    f = r[[f'pav{int(k)}' for k in KS]].to_numpy(float)
    z = [A.label_equiv(KS, f, k, 0.0) for k in (50, 200)]
    e = [A.label_equiv(KS, f, k, 1e-6) for k in (50, 200)]; m = [A.label_equiv(KS, f, k, -1e-6) for k in (50, 200)]
    if any(abs(x) > 1e-6 for x in z) or any(abs(x) > 5 for x in e + m):
        print(r.target, r.clf, 'Δ=0 ->', np.round(z, 1), ' Δ=+1e-6 ->', np.round(e, 1), ' Δ=-1e-6 ->', np.round(m, 1))
rng = np.random.default_rng(20260925)
B = 2000; res = []
cty = sorted(P.index.get_level_values(0).unique())
for b in range(B):
    v = {('+0.3', 200): [], ('L95', 50): [], ('+0.3', 50): [], ('−0.3', 50): []}
    for t in cty:
        M = P.loc[t]; idx = rng.integers(0, len(M), len(M))
        for clf in ('s2', 'ae'):
            raw = M[clf][KS].to_numpy()[idx].mean(0)
            f = IsotonicRegression().fit_transform(KS, raw)
            v[('+0.3', 200)].append(A.label_equiv(KS, f, 200, 0.3)); v[('+0.3', 50)].append(A.label_equiv(KS, f, 50, 0.3))
            v[('L95', 50)].append(A.label_equiv(KS, f, 50, s['lo95'])); v[('−0.3', 50)].append(A.label_equiv(KS, f, 50, -0.3))
    res.append({f'{a}@{k}_med': np.median(x) for (a, k), x in v.items()} | {f'{a}@{k}_oor': np.mean(~np.isfinite(x)) for (a, k), x in v.items()})
D = pd.DataFrame(res)
for c in ['+0.3@200', 'L95@50', '+0.3@50', '−0.3@50']:
    m = D[c + '_med']; o = D[c + '_oor']
    print(c, ' P(median infinite)=%.3f' % np.mean(~np.isfinite(m)), ' oor q10/50/90=', np.quantile(o, [.1, .5, .9]).round(3),
          ' finite-median q10/50/90=', np.quantile(m[np.isfinite(m)], [.1, .5, .9]).round(1) if np.isfinite(m).any() else '-')
