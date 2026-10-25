"""D3 复核：独立重算检验 3 的合并份额（抽样行 H1.oa_comp.es.ae、H1.oa_comp.spatial.ae、H1.ba_resid.es.ae、H1.ba_resid.spatial.s2）。
只读 results/check3_arms.csv；bootstrap 按 analyze_check3.main 的 rng 调用顺序复现。"""
import os
os.environ['OMP_NUM_THREADS'] = '2'
import numpy as np
import pandas as pd

A = pd.read_csv('/Users/perng_mini/AlphaCrops_local/results/check3_arms.csv')
R = A[A.split == 'random'].pivot_table(index=['feat', 'target'], columns='arm', values=['oa', 'ba_ge30'])
S = A[A.split == 'spatial'].pivot_table(index=['feat', 'target'], columns='arm', values=['oa', 'ba_ge30'])
CH = {'oa': (['L0', 'PM', 'CMN', 'Wnat'], ['comp', 'size', 'resid']),
      'ba_ge30': (['L0', 'LS', 'CMB', 'W0'], ['set', 'size', 'resid'])}


def parts(tab, metric, feat):
    ch, nm = CH[metric]
    r = tab[metric].loc[feat]
    d = pd.DataFrame({'total': r[ch[-1]] - r[ch[0]]})
    for i, n in enumerate(nm):
        d[n] = r[ch[i + 1]] - r[ch[i]]
    return d


def boot(d, part, rng, B=5000):
    t = np.arange(len(d))
    bs = []
    for _ in range(B):
        s = rng.choice(len(t), len(t))
        den = d['total'].iloc[s].sum()
        bs.append(d[part].iloc[s].sum() / den if den != 0 else np.nan)
    return np.nanpercentile(bs, [2.5, 97.5])


rng = np.random.default_rng(42)
res = {}
for metric in ['oa', 'ba_ge30']:
    for feat in ['ae', 's2']:
        d = parts(R, metric, feat)
        for n in CH[metric][1]:
            v = d[n].sum() / d['total'].sum()
            lo, hi = boot(d, n, rng)
            res[(metric, feat, n)] = (100 * v, 100 * lo, 100 * hi)
        if metric == 'oa':
            boot(d, 'comp', rng)  # verdict 调用，消耗随机数
for k, v in res.items():
    print('random', k, ['%.4f' % x for x in v])

keep = lambda d: d.drop(index=[c for c in ('RO', 'PT') if c in d.index])
for metric, feat, n in [('oa', 'ae', 'comp'), ('ba_ge30', 'ae', 'resid'), ('ba_ge30', 's2', 'resid'), ('oa', 's2', 'comp')]:
    d = parts(R, metric, feat)
    dk = keep(d)
    print('noROPT random', metric, feat, n, '%.4f' % (100 * dk[n].sum() / dk['total'].sum()))
    ds = parts(S, metric, feat)
    dsk = keep(ds)
    print('spatial', metric, feat, n, '%.4f' % (100 * ds[n].sum() / ds['total'].sum()),
          'noROPT %.4f' % (100 * dsk[n].sum() / dsk['total'].sum()), 'n=', len(ds))

# 逐国份额（H1.oa_comp.es.ae 的 per_country）
d = parts(R, 'oa', 'ae')
print('per-country comp share AE:', (100 * d['comp'] / d['total']).round(3).to_dict())
