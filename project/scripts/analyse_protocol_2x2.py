"""Test the protocol 2x2 with the support threshold varied, not fixed.

The 10/11 "AlphaEarth pays a larger border cost" result held only when classes
with fewer than 10 parcels were included in the balanced-accuracy average. Any
claim from this analysis is therefore reported at three support regimes, and is
only asserted if it survives all three.

  regime 1  every class present (support >= 1)   -- what the stored BA fields use
  regime 2  support >= 10 in that evaluation
  regime 3  support >= 10 in BOTH feature sets   -- fixes the Latvia mismatch

Rows of the 2x2:
  broad   pooled CV (per country) - LOCO (per country)      both train on 10-11
  narrow  within-country CV       - pairwise (per source)    both train on 1
"""
import sys, os, json
import numpy as np, pandas as pd

class _Tee:
    def __init__(self, path):
        self.f = open(path, 'w'); self.o = sys.stdout
    def write(self, x): self.o.write(x); self.f.write(x)
    def flush(self): self.o.flush(); self.f.flush()

from scipy.stats import wilcoxon, binomtest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

C = ['AT','BE','DE','EE','LT','LV','PT','RO','SE','SI','SK']
R = 'results/experiments'
POOLED = f'{R}/exp1_id_baseline/percountry_tier_b_nocoords_2021.json'
loco = json.load(open(f'{R}/exp2_loco/tier_b_nocoords_2021.json'))
e2b  = json.load(open(f'{R}/exp2b_country_transfer/tier_b_nocoords_2021.json'))
cvcsv = {f: pd.read_csv(f'{R}/exp0_per_country_cv/per_class_{f}_tier_b.csv')
         for f in ['ae','s2','fusion']}
if not os.path.exists(POOLED):
    sys.exit('pooled per-country results not written yet')
pool = json.load(open(POOLED))
sys.stdout = _Tee('results/experiments/exp1_id_baseline/protocol_2x2_report.txt')
FE = [f for f in ['ae','s2','fusion'] if all(f'{f}|{r}' in pool for r in range(5))]
print(f'feature sets with all 5 repeats: {[f.upper() for f in FE]}\n')


def mean_recall(per_class, minsup, keep=None):
    k = [cl for cl, v in per_class.items() if int(v['support']) >= minsup
         and (keep is None or cl in keep)]
    return np.mean([per_class[cl]['recall'] for cl in k]) if k else np.nan


def common_classes(c, minsup, feats):
    """classes with >= minsup parcels in country c under EVERY feature set"""
    sets = []
    for f in feats:
        d = cvcsv[f].query('country == @c and support >= @minsup')
        sets.append(set(d['class']))
    return set.intersection(*sets)


def row_values(minsup, common):
    """returns {feat: {'pooled':[], 'loco':[], 'within':[], 'pair':[]}} per country"""
    keep = {c: common_classes(c, minsup, FE) for c in C} if common else {c: None for c in C}
    out = {}
    for f in FE:
        v = {k: [] for k in ('pooled','loco','within','pair')}
        for c in C:
            # pooled CV: average the 5 repeats' per-class recall
            # NOTE units: pooled per_class recall is stored already in PERCENT
            # (run_pooled_cv_percountry.py writes rr*100), whereas LOCO, the
            # per-country CV csv and the pairwise matrix all store FRACTIONS.
            # Do not scale the pooled values again.
            rs = [mean_recall(pool[f'{f}|{r}']['per_country'][c]['per_class'], minsup, keep[c])
                  for r in range(5) if c in pool[f'{f}|{r}']['per_country']]
            v['pooled'].append(np.nanmean(rs) if rs else np.nan)
            v['loco'].append(mean_recall(loco[f]['per_country'][c]['per_class'], minsup, keep[c]) * 100)
            d = cvcsv[f].query('country == @c and support >= @minsup')
            if keep[c] is not None: d = d[d['class'].isin(keep[c])]
            v['within'].append(d.recall.mean() * 100)
            per = [mean_recall(e['per_class'], minsup, keep[t])
                   for t, e in e2b[f]['transfer_matrix'][c].items()]
            v['pair'].append(np.nanmean(per) * 100)
        out[f] = {k: np.array(x) for k, x in v.items()}
    return out


def test(a, b, label):
    d = a - b
    ok = ~np.isnan(d); d = d[ok]
    k = int((d > 0).sum()); n = len(d)
    return (f'{label:<34} mean {d.mean():+6.2f}  median {np.median(d):+6.2f}  '
            f'{k}/{n}  sign p={binomtest(k, n, 0.5).pvalue:.4f}  '
            f'Wilcoxon p={wilcoxon(d).pvalue:.3f}')


# Regimes are fixed HERE, before any of them has been evaluated, and all four are
# reported whatever they show. Regime 3 was added after finding that support >= 10
# removes no class at all in Portugal, whose instability comes from classes with
# 27-86 parcels and near-zero recall -- i.e. the >= 10 threshold does not protect it.
REGIMES = [(1,   False, 'regime 1: every class (support >= 1)'),
           (10,  False, 'regime 2: support >= 10'),
           (100, False, 'regime 3: support >= 100'),
           (10,  True,  'regime 4: support >= 10 in ALL feature sets')]
for minsup, common, name in REGIMES:
    V = row_values(minsup, common)
    print('=' * 100); print(name); print('=' * 100)
    for f in FE:
        v = V[f]
        print(f'  {f.upper():<7} broad: pooled {np.nanmean(v["pooled"]):5.1f} - LOCO '
              f'{np.nanmean(v["loco"]):5.1f} = {np.nanmean(v["pooled"]-v["loco"]):5.1f} pp   |   '
              f'narrow: within {np.nanmean(v["within"]):5.1f} - pair '
              f'{np.nanmean(v["pair"]):5.1f} = {np.nanmean(v["within"]-v["pair"]):5.1f} pp')
    if 'ae' in V and 's2' in V:
        print()
        print('  ' + test(V['ae']['pooled']-V['ae']['loco'],
                          V['s2']['pooled']-V['s2']['loco'], 'BROAD  border cost, AE - S2'))
        print('  ' + test(V['ae']['within']-V['ae']['pair'],
                          V['s2']['within']-V['s2']['pair'], 'NARROW border cost, AE - S2'))
        print('  ' + test(V['ae']['loco'], V['s2']['loco'], 'LOCO level, AE - S2'))
        print('  ' + test(V['ae']['pooled'], V['s2']['pooled'], 'pooled level, AE - S2'))
        # interaction: is the border cost difference bigger in the narrow row?
        bn = (V['ae']['within']-V['ae']['pair']) - (V['s2']['within']-V['s2']['pair'])
        bb = (V['ae']['pooled']-V['ae']['loco'])  - (V['s2']['pooled']-V['s2']['loco'])
        print('  ' + test(bn, bb, 'INTERACTION narrow - broad'))
    print()
