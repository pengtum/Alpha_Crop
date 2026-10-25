"""独立核实：预测通路（margin、predstrat_srs，重复 0）AE−S2 在各指标上的表现，
以及 BA 与 OA 分歧本身是否显著。直接读 rows CSV，不 import analyze_G。"""
import glob, os
import numpy as np, pandas as pd
from scipy import stats

OUT = '/Users/perng_mini/AlphaCrops_local/results/guidance_G'
R = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(OUT, 'rows_t*_*.csv'))], ignore_index=True)
print('rows', len(R), R.cond.value_counts().to_dict())
KEY = ['target', 'split', 'rep', 'method', 'clf', 'k']
PRED = ['margin', 'predstrat_srs']
E1 = ['lpm2_nogeo', 'kcenter', 'typiclust', 'kherding']
HYB = ['gist_margin', 'badge_c']


def dl(methods, metric, reps=None):
    a = R[(R.cond == 'AE') & R.method.isin(methods)].set_index(KEY)[metric]
    b = R[(R.cond == 'S2') & R.method.isin(methods)].set_index(KEY)[metric]
    d = (a - b).dropna().rename('d').reset_index()
    if reps is not None:
        d = d[d.rep.isin(reps)]
    return d


def cv(d, col='d'):
    return d.groupby(['target', 'method', 'clf', 'k'])[col].mean().groupby('target').mean()


def ts(v):
    v = np.asarray(v, float); n = len(v); m = v.mean(); se = v.std(ddof=1) / np.sqrt(n)
    t = stats.t.ppf(.975, n - 1)
    return f'{m:+.3f} [{m - t * se:+.3f},{m + t * se:+.3f}] p={2 * stats.t.sf(abs(m / se), n - 1):.3f} pos={int((v > 0).sum())}/{n}'


mets = ['ba_ge30', 'oa', 'f1_ge30', 'ba_P1', 'oa_P1', 'ba_P2', 'oa_P2', 'ba_far', 'oa_far', 'ba_i200', 'ba_i250',
        'oa_i200', 'oa_i250']
for name, ms, reps in (('pred', PRED, [0]), ('geom(E1)', E1, None), ('hyb', HYB, [0])):
    print(f'\n===== 通路 {name} =====')
    d0 = dl(ms, 'ba_ge30', reps)
    print('n pairs', len(d0), 'reps', sorted(d0.rep.unique()), 'units', d0.groupby(['target', 'split', 'rep']).ngroups)
    for m in mets:
        print(f'  {m:10s}', ts(cv(dl(ms, m, reps))))
    # BA − OA 分歧本身（配对到国家）
    diff = cv(dl(ms, 'ba_ge30', reps)) - cv(dl(ms, 'oa', reps))
    print('  Δba − Δoa  ', ts(diff))
    diff2 = cv(dl(ms, 'ba_P2', reps)) - cv(dl(ms, 'oa_P2', reps))
    print('  Δba_P2 − Δoa_P2', ts(diff2))
    dd = dl(ms, 'ba_ge30', reps).set_index(KEY).d - dl(ms, 'ba_P2', reps).set_index(KEY).d
    print('  Δ(P0)−Δ(P2)', ts(cv(dd.rename('d').reset_index())))
    if name == 'pred':
        tab = pd.DataFrame({m: cv(dl(ms, m, reps)) for m in ['ba_ge30', 'oa', 'f1_ge30', 'ba_P2', 'oa_P2', 'ba_far']})
        print(tab.round(3).to_string())
        print('  country-level corr(Δba, Δoa) =', round(tab.ba_ge30.corr(tab.oa), 3),
              ' spearman', round(stats.spearmanr(tab.ba_ge30, tab.oa)[0], 3))
        print('  country SD: ', tab.std().round(3).to_dict())
