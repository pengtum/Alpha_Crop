import sys, os, numpy as np, pandas as pd
os.chdir('/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops/label_expansion')
sys.path.insert(0, os.getcwd())
import analyze_G as A
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 500)
R = A.load_rows()
KS = np.array([0, 50, 100, 200, 400.])

def pav(y):  # independent PAV (increasing, equal weights)
    blocks = [[v, 1] for v in y]
    out = []
    for v, w in blocks:
        out.append([v, w])
        while len(out) > 1 and out[-2][0] > out[-1][0]:
            v2, w2 = out.pop(); v1, w1 = out.pop()
            out.append([(v1*w1 + v2*w2)/(w1+w2), w1+w2])
    res = []
    for v, w in out: res += [v]*w
    return np.array(res)

def inv(f, k, d):
    fk = np.interp(k, KS, f); t = fk + d
    if t > f.max() + 1e-12: return np.inf
    if t < f.min() - 1e-12: return -np.inf
    # smallest x with f(x)=t on piecewise-linear f
    for j in range(1, len(KS)):
        if f[j] >= t - 1e-12:
            if f[j] == f[j-1]: return KS[j-1] - k if abs(f[j-1]-t) < 1e-12 else KS[j]-k
            return KS[j-1] + (t - f[j-1])*(KS[j]-KS[j-1])/(f[j]-f[j-1]) - k
    return KS[0] - k

vE1 = A.country_values(A.deltas(R, 'AE', 'S2', A.E1)); s = A.t_summary(vE1)
D = {'+0.3': 0.3, 'L95': s['lo95'], 'U95': s['hi95'], 'θ_E1': s['mean'], '−0.3': -0.3}
print({k: round(v, 4) for k, v in D.items()})
g = R[R.method.isin(['srs', 'foreign_only'])]
rows = []; curves = []
for t in sorted(R.target.unique()):
    for clf in ('s2', 'ae'):
        raw = g[(g.target == t) & (g.clf == clf)].groupby('k').ba_ge30.mean().reindex(KS).to_numpy()
        f = pav(raw); ks2, f2 = A.srs_curve(R, t, clf)
        assert np.allclose(f, f2) and np.allclose(ks2, KS)
        curves.append(dict(target=t, clf=clf, **{f'raw{int(k)}': r for k, r in zip(KS, raw)}, **{f'pav{int(k)}': r for k, r in zip(KS, f)},
                           gain0_50=f[1]-f[0], gain200_400=f[4]-f[3], plat0_50=f[1]==f[0], plat200_400=f[4]==f[3],
                           raw_gain0_50=raw[1]-raw[0], raw_gain200_400=raw[4]-raw[3]))
        for k in (50, 200):
            for n, dv in D.items():
                a = A.label_equiv(KS, f, k, dv); b = inv(f, k, dv)
                rows.append(dict(target=t, clf=clf, k=k, what=n, labels=a, mine=b))
L = pd.DataFrame(rows); C = pd.DataFrame(curves)
print('official vs mine mismatches:', (~np.isclose(L.labels, L.mine) & ~(L.labels == L.mine)).sum())
print(L.groupby(['what', 'k']).labels.median().unstack('k'))
def summ(v):
    v = np.sort(np.asarray(v, float)); fin = v[np.isfinite(v)]
    return pd.Series(dict(median=np.median(v), v11=v[10], v12=v[11], Q1=np.quantile(v, .25), Q3=np.quantile(v, .75),
                          pinf=int(np.isposinf(v).sum()), ninf=int(np.isneginf(v).sum()), fin_med=np.median(fin)))
print(L.groupby(['what', 'k']).labels.apply(summ).unstack().round(1).to_string())
print(C[['target','clf','raw0','raw50','raw200','raw400','pav0','pav50','pav200','pav400','gain0_50','gain200_400','plat0_50','plat200_400']].round(3).to_string())
print('plateau 0-50:', C.plat0_50.sum(), ' plateau 200-400:', C.plat200_400.sum())
print('gain200_400<0.3:', (C.gain200_400 < 0.3).sum(), ' gain0_50<|L95|:', (C.gain0_50 < -s['lo95']).sum())
print('raw gain200_400 median', C.raw_gain200_400.median(), ' raw gain 0_50 median', C.raw_gain0_50.median())
L.to_csv('/Users/perng_mini/AlphaCrops_local/verify/G_audit/labeleq_verify/L.csv', index=False); C.to_csv('/Users/perng_mini/AlphaCrops_local/verify/G_audit/labeleq_verify/C.csv', index=False)
