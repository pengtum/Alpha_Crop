"""复核 H2：θ_E1 在各指标上的 90% 上界（AE 优势上界），与 NUMBERS 的 H2.ae_adv_hi90_max 对照。只读 G rows。"""
import glob, os
os.environ['OMP_NUM_THREADS'] = '2'
import numpy as np, pandas as pd
from scipy import stats
R = pd.concat([pd.read_csv(f) for f in sorted(glob.glob('/Users/perng_mini/AlphaCrops_local/results/guidance_G/rows_t*_*.csv'))])
E1 = ['lpm2_nogeo', 'kcenter', 'typiclust', 'kherding']; UNIT = ['target', 'split', 'rep']
def theta(metric, ks=(50, 200)):
    sub = R[R.method.isin(E1) & R.k.isin(ks) & R.clf.isin(['ae', 's2'])]
    key = UNIT + ['method', 'clf', 'k']
    d = (sub[sub.cond == 'AE'].set_index(key)[metric] - sub[sub.cond == 'S2'].set_index(key)[metric]).dropna().rename('d').reset_index()
    v = d.groupby(['target', 'method', 'clf', 'k']).d.mean().groupby('target').mean()
    n = len(v); m = v.mean(); se = v.std(ddof=1) / np.sqrt(n)
    return m, m + stats.t.ppf(0.95, n - 1) * se, m - stats.t.ppf(0.95, n - 1) * se
for met in ['ba_ge30', 'oa', 'f1_ge30', 'ba_P1', 'oa_P1', 'ba_P2', 'oa_P2', 'ba_far', 'oa_far', 'ba_i200', 'ba_i250']:
    for ks in ((50, 200), (50,), (200,)):
        m, hi, lo = theta(met, ks)
        print(f'{met:8s} k={ks}: {m:+.4f} 90% [{lo:+.4f}, {hi:+.4f}]')
