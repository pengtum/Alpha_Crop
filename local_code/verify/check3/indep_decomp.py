"""独立复算检验 3 主分解（不参考 analyze_check3.py）。"""
import os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

D = '/Users/perng_mini/AlphaCrops_local/results/check3'
OUT = '/Users/perng_mini/AlphaCrops_local/verify/check3'
TS = ['AT', 'BE', 'DE', 'EE', 'LT', 'LV', 'PT', 'RO', 'SE', 'SI', 'SK']
FEATS = ['ae', 's2']


def oa(y, p):
    return float((y == p).mean())


def ba_ge30(y, p, minn=30):
    cls = [c for c in np.unique(y) if (y == c).sum() >= minn]
    return float(np.mean([(p[y == c] == c).mean() for c in cls])), cls


def pred(T, feat, model):
    return np.load(os.path.join(D, f'{T}_{feat}_{model}_test.npy'))


rows = []
for T in TS:
    sp = np.load(os.path.join(D, f'{T}_split.npz'))
    y = sp['y_test']
    ypool = sp['y_pool']
    C = 10
    present = np.bincount(ypool, minlength=C) > 0
    for feat in FEATS:
        P = {m: pred(T, feat, m) for m in ['loco', 'loco_pm', 'cmn', 'wnat', 'cmb', 'within']}
        for m, a in P.items():
            assert a.shape == (len(y), C), (T, feat, m, a.shape)
        pl = P['loco'].astype(np.float64).copy()
        pl[:, ~present] = 0
        pl = pl / pl.sum(1, keepdims=True)
        preds = {m: a.argmax(1) for m, a in P.items()}
        preds['ls'] = pl.argmax(1)
        r = dict(T=T, feat=feat, n_test=len(y), n_present=int(present.sum()))
        for m, p in preds.items():
            r[f'oa_{m}'] = oa(y, p)
            r[f'ba_{m}'], cls = ba_ge30(y, p)
        r['n_ba_cls'] = len(cls)
        # OA 链
        r['oa_total'] = r['oa_wnat'] - r['oa_loco']
        r['oa_comp'] = r['oa_loco_pm'] - r['oa_loco']
        r['oa_size'] = r['oa_cmn'] - r['oa_loco_pm']
        r['oa_dom'] = r['oa_wnat'] - r['oa_cmn']
        # ba 链
        r['ba_total'] = r['ba_within'] - r['ba_loco']
        r['ba_set'] = r['ba_ls'] - r['ba_loco']
        r['ba_size'] = r['ba_cmb'] - r['ba_ls']
        r['ba_dom'] = r['ba_within'] - r['ba_cmb']
        rows.append(r)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, 'indep_decomp.csv'), index=False)
pd.set_option('display.width', 250)
pd.set_option('display.max_columns', 40)
pp = df.copy()
num = [c for c in pp.columns if c.startswith(('oa_', 'ba_'))]
pp[num] = pp[num] * 100
print(pp[['T', 'feat', 'n_present', 'n_ba_cls', 'oa_loco', 'oa_loco_pm', 'oa_cmn', 'oa_wnat',
          'oa_total', 'oa_comp', 'oa_size', 'oa_dom']].round(2).to_string())
print(pp[['T', 'feat', 'ba_loco', 'ba_ls', 'ba_cmb', 'ba_within',
          'ba_total', 'ba_set', 'ba_size', 'ba_dom']].round(2).to_string())

print('\n== 合并比例 Σ部分/Σ总损失，以及均值/中位数 (pp) ==')
for feat in FEATS:
    s = df[df.feat == feat]
    for chain, parts in (('oa', ['comp', 'size', 'dom']), ('ba', ['set', 'size', 'dom'])):
        tot = s[f'{chain}_total'].sum()
        txt = ', '.join(f"{p}={s[f'{chain}_{p}'].sum() / tot * 100:.1f}%" for p in parts)
        means = ', '.join(f"{p}: mean {s[f'{chain}_{p}'].mean() * 100:.2f} med {s[f'{chain}_{p}'].median() * 100:.2f}"
                          for p in parts + ['total'])
        print(f'{feat} {chain}: {txt}  | {means}')

print('\n== 主要问题 2：AE−S2 域残差 ==')
for chain in ('oa', 'ba'):
    a = df[df.feat == 'ae'].set_index('T')[f'{chain}_dom']
    b = df[df.feat == 's2'].set_index('T')[f'{chain}_dom']
    d = (a - b).loc[TS] * 100
    w = wilcoxon(d.values)
    print(f'{chain}: AE dom mean {a.mean() * 100:.2f}, S2 dom mean {b.mean() * 100:.2f}, '
          f'diff mean {d.mean():.2f} pp, median {d.median():.2f}, '
          f'n(AE<S2)={int((d < 0).sum())}/11, Wilcoxon stat={w.statistic}, p={w.pvalue:.4g}')
    print('   per-T diff:', d.round(2).to_dict())
