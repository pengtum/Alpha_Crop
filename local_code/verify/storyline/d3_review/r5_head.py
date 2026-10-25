"""D3 复核：H5 决策头相关行（frontier2_rows、calib_rows）与 H1附 缓冲格网决策层上界（oracle_G_buffered_rows）。只读已有行文件。"""
import os
os.environ['OMP_NUM_THREADS'] = '2'
import numpy as np
import pandas as pd
from scipy import stats

PS = '/Users/perng_mini/AlphaCrops_local/verify/pivot_eval/prior_shift'
DV = '/Users/perng_mini/AlphaCrops_local/verify/reposition/decision_vs_representation'


def t(v, lab):
    v = v.astype(float); n = len(v); m = v.mean(); h = stats.t.ppf(0.975, n - 1) * v.std(ddof=1) / np.sqrt(n)
    w = v.drop([c for c in ('RO', 'PT') if c in v.index])
    print(f'{lab}: {m:+.4f} [{m - h:+.4f}, {m + h:+.4f}] {int((v > 0).sum())}/{n} | 去RO/PT {w.mean():+.4f}')


f = pd.read_csv(os.path.join(PS, 'frontier2_rows.csv'))
key = ['unit', 'target', 'clf', 'k']
for k in (50, 200, 400):
    r = f[(f.design == 'srs_G') & (f.fam == 'retrain') & (f.k == k)].set_index(key)[['ba', 'oa']]
    x = f[(f.design == 'srs_G') & (f.fam == 'F3b') & (f.alpha == 0.4) & (f.k == k)].set_index(key)[['ba', 'oa']]
    d = (x - r).dropna().reset_index()
    c = d.groupby('target')[['ba', 'oa']].mean()
    t(c.ba, f'F3b α=0.4 − 重训 k={k} BA'); t(c.oa, f'F3b α=0.4 − 重训 k={k} OA')

c = pd.read_csv(os.path.join(PS, 'calib_rows.csv'))
k0 = c[(c.design == 'none') & (c['head'] == 'k0')]
print('k0 行: 每 (target,split,clf) rep 数', k0.groupby(['target', 'split', 'clf']).size().unique())
k0i = k0.drop_duplicates(['target', 'split', 'clf']).set_index(['target', 'split', 'clf'])[['ba', 'oa']]
for k in (50, 200, 400):
    h = c[(c.design == 'srs_G') & (c['head'] == 'bal_l0.1') & (c.k == k)].set_index(['target', 'split', 'rep', 'clf'])[['ba', 'oa']]
    rt = c[(c.design == 'srs_G') & (c['head'] == 'retrain') & (c.k == k)].set_index(['target', 'split', 'rep', 'clf'])[['ba', 'oa']]
    d = (h - rt).groupby('target').mean()
    t(d.ba, f'只用头 − 重训 k={k} BA'); t(d.oa, f'只用头 − 重训 k={k} OA')
    d0 = h.reset_index().merge(k0i.reset_index(), on=['target', 'split', 'clf'], suffixes=('', '_0'))
    d0 = d0.assign(ba=d0.ba - d0.ba_0, oa=d0.oa - d0.oa_0).groupby('target')[['ba', 'oa']].mean()
    t(d0.ba, f'只用头 − k0 k={k} BA')
full = c[(c.design == 'allpool') & (c['head'] == 'bal_full')].set_index(['target', 'split', 'clf'])[['ba', 'oa']]
d = (full - k0i).dropna().groupby('target').mean()
t(d.ba, '全池类均衡头 − k0 BA'); t(d.oa, '全池类均衡头 − k0 OA')
for clf in ('ae', 's2'):
    dd = (full - k0i).dropna().xs(clf, level='clf').groupby('target').mean()
    t(dd.ba, f'  全池类均衡头 − k0 BA [{clf}]')

o = pd.read_csv(os.path.join(DV, 'oracle_G_buffered_rows.csv'))
print('oracle_G_buffered 列:', o.columns.tolist()[:30], o.shape)
