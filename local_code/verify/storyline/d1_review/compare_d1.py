"""对比独立重算与 D1 输出；自行聚合国家值与 t 95% CI。"""
import glob
import os

import numpy as np
import pandas as pd
from scipy import stats

OUT = os.path.dirname(os.path.abspath(__file__))
D1 = '/Users/perng_mini/AlphaCrops_local/verify/storyline/decision_curve'
R = '/Users/perng_mini/AlphaCrops_local/results/guidance_G'
M = pd.read_csv(os.path.join(OUT, 'recheck_rows.csv'))
RT = pd.read_csv(os.path.join(OUT, 'recheck_retrain.csv'))
TT = pd.read_csv(os.path.join(OUT, 'recheck_tight.csv'))
D = pd.read_csv(os.path.join(D1, 'decision_curve_rows.csv'))
U = pd.read_csv(os.path.join(D1, 'decision_curve_units.csv'))
bc = [f'b{c}' for c in range(10)]

print('== 1. 逐行：头 ==')
key = ['unit', 'clf', 'design', 'k', 'r']
d = D[D.design.isin(['srs_D1', 'fullpool', 'base_PT'])].copy()
j = M.merge(d, on=key, suffixes=('', '_d1'), how='outer', indicator=True)
print(j._merge.value_counts().to_dict())
print('seed 一致:', bool((j[j.design == 'srs_D1'].seed == j[j.design == 'srs_D1'].seed_d1).all()))
for met, m2 in (('oa', 'oa'), ('ba', 'ba_ge30')):
    x = (j[met] - j[m2]).abs()
    print(f'{met}: max|Δ| {x.max():.2e}，>0.01 的行 {int((x > 0.01).sum())}')
hj = j[j.design != 'base_PT']
print('s max|Δ|', (hj.s - hj.s_d1).abs().max(), ' b max|Δ|', max((hj[c] - hj[c + '_d1']).abs().max() for c in bc))

print('== 2. Δretrain ==')
G = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(R, 'rows_t*_*.csv'))], ignore_index=True)
gs = G[(G.cond == 'none') & (G.method == 'srs')][['target', 'split', 'rep', 'clf', 'k', 'oa', 'ba_ge30', 'sel_hash']]
gf = G[(G.cond == 'none') & (G.method == 'foreign_only')][['target', 'split', 'rep', 'clf', 'oa', 'ba_ge30']].rename(columns={'oa': 'oa_Gfo', 'ba_ge30': 'ba_Gfo'})
print('G srs 行 k 分布', gs.k.value_counts().sort_index().to_dict(), '重复键', gs.duplicated(['target', 'split', 'rep', 'clf', 'k']).sum())
r = RT.merge(gs, on=['target', 'split', 'rep', 'clf', 'k'], how='left').merge(
    gf, on=['target', 'split', 'rep', 'clf'], how='left')
print('sel_hash 一致', bool((r.sel_hash_x == r.sel_hash_y).all()) if 'sel_hash_x' in r else None)
print('retrain BA max|Δ| vs G 行', (r.ba_ret - r.ba_ge30).abs().max(), ' OA', (r.oa_ret - r.oa).abs().max())
print('foreign BA max|Δ| vs G 行', (r.ba_fo - r.ba_Gfo).abs().max(), ' OA', (r.oa_fo - r.oa_Gfo).abs().max(), ' 缺失', r.ba_Gfo.isna().sum(), r.ba_ge30.isna().sum())
RT['dret_ba'] = RT.ba_ret - RT.ba_fo
RT['dret_oa'] = RT.oa_ret - RT.oa_fo
uu = U[U.k.isin([50, 100, 200, 400])].pivot_table(index=['unit', 'clf', 'k'], columns='metric', values='dret')
q = RT.set_index(['unit', 'clf', 'k'])[['dret_ba', 'dret_oa']].join(uu)
print('dret vs units.csv：BA max|Δ|', (q.dret_ba - q.ba_ge30).abs().max(), ' OA', (q.dret_oa - q.oa).abs().max())

print('== 3. 优化器容差敏感性（5 单元 × 2 特征，k 全部 r<2 + 全池）==')
TT['dba'] = TT.ba_t - TT.ba
TT['doa'] = TT.oa_t - TT.oa
TT['dfv'] = TT.fval - TT.fval_t
print(TT[['dba', 'doa', 'dfv', 'ds', 'db']].describe().T[['mean', 'max', 'min']])
print('|dBA|>0.01 的行', int((TT.dba.abs() > 0.01).sum()), '/', len(TT))
print(TT[TT.dba.abs() > 0.01][['unit', 'clf', 'k', 'r', 'ba', 'ba_t', 'dfv', 'ds', 'db']].to_string())

print('== 4. 自行聚合 ==')
base = M[M.design == 'base_PT'].set_index(['unit', 'clf'])[['oa', 'ba']]
H = M[M.design == 'srs_D1'].groupby(['target', 'unit', 'clf', 'k'])[['oa', 'ba']].mean().reset_index()
F = M[M.design == 'fullpool'][['target', 'unit', 'clf', 'oa', 'ba']]
for met in ('ba', 'oa'):
    H[f'dh_{met}'] = H[met].to_numpy() - base.loc[list(zip(H.unit, H.clf)), met].to_numpy()
    F[f'df_{met}'] = F[met].to_numpy() - base.loc[list(zip(F.unit, F.clf)), met].to_numpy()
H = H.merge(RT[['unit', 'clf', 'k', 'dret_ba', 'dret_oa']], on=['unit', 'clf', 'k'], how='left')


def t95(v):
    v = np.asarray(v, float)
    n, m, se = len(v), v.mean(), v.std(ddof=1) / np.sqrt(len(v))
    q = stats.t.ppf(0.975, n - 1)
    return f'{m:+.2f} [{m - q * se:+.2f}, {m + q * se:+.2f}] {int((v > 0).sum())}/{n}'


def cv(df, cols):
    a = df.groupby(['target', 'clf'])[cols].mean().reset_index()
    both = a.groupby('target')[cols].mean()
    return {f: a[a.clf == f].set_index('target')[cols] for f in ('ae', 's2')} | {'both': both}


for met in ('ba', 'oa'):
    fv = cv(F, [f'df_{met}'])
    for drop in (False, True):
        tag = '去RO/PT' if drop else '11国'
        for k in KS if (KS := [50, 100, 200, 400, 1000]) else []:
            hk = H[H.k == k]
            cols = [f'dh_{met}'] + ([f'dret_{met}'] if k <= 400 else [])
            hk = hk.assign(diff=hk[f'dh_{met}'] - hk[f'dret_{met}']) if k <= 400 else hk
            cc = cols + (['diff'] if k <= 400 else [])
            c = cv(hk, cc)
            for f in ('both', 'ae', 's2'):
                v = c[f]
                if drop:
                    v = v[~v.index.isin(['RO', 'PT'])]
                fp = fv[f][f'df_{met}']
                if drop:
                    fp = fp[~fp.index.isin(['RO', 'PT'])]
                s = f'{met} {tag} k={k:<4d} {f:4s} head {t95(v[f"dh_{met}"])}'
                if k <= 400:
                    s += f' | ret {t95(v[f"dret_{met}"])} | diff {t95(v["diff"])}'
                s += f' | ratio {v[f"dh_{met}"].mean() / fp.mean():.2f}'
                if k <= 400:
                    s += f' ret/full {v[f"dret_{met}"].mean() / fp.mean():.2f}'
                print(s)
        for f in ('both', 'ae', 's2'):
            fp = fv[f][f'df_{met}']
            if drop:
                fp = fp[~fp.index.isin(['RO', 'PT'])]
            print(f'{met} {tag} 全池 {f:4s} {t95(fp)}')

# 国家值（对照 summary 的国家表）
c50 = cv(H[H.k == 50], ['dh_ba'])
print('RO ae head k50', round(c50['ae'].loc['RO', 'dh_ba'], 2))
c = cv(H, ['dh_ba'])
for t in ('SE', 'SI', 'EE', 'LV', 'PT', 'RO'):
    vals = {k: round(H[(H.k == k) & (H.target == t) & (H.clf == 'ae')].dh_ba.mean(), 2) for k in [50, 100, 200, 400, 1000]}
    vs = {k: round(H[(H.k == k) & (H.target == t) & (H.clf == 's2')].dh_ba.mean(), 2) for k in [50, 100, 200, 400, 1000]}
    fa = round(F[(F.target == t) & (F.clf == 'ae')].df_ba.mean(), 2)
    fs = round(F[(F.target == t) & (F.clf == 's2')].df_ba.mean(), 2)
    ra = {k: round(RT[(RT.k == k) & (RT.target == t) & (RT.clf == 'ae')].dret_ba.mean(), 2) for k in [50, 100, 200, 400]}
    rs = {k: round(RT[(RT.k == k) & (RT.target == t) & (RT.clf == 's2')].dret_ba.mean(), 2) for k in [50, 100, 200, 400]}
    print(t, 'AE head', vals, 'full', fa, 'ret', ra)
    print(t, 'S2 head', vs, 'full', fs, 'ret', rs)
