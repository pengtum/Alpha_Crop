# 审计：主实现 fa_rows_B2 与独立实现 indep_fa_rows 的单元层、国家层比对；并从 fa_rows_B2 自算国家层估计量
import os, sys
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_v] = '2'
sys.dont_write_bytecode = True
import numpy as np, pandas as pd
from scipy import stats
V = '/Users/perng_mini/AlphaCrops_local/verify/step0'
M = pd.read_csv(f'{V}/partB2/fa_rows_B2.csv', float_precision='round_trip')
I = pd.read_csv(f'{V}/partB2_indep/indep_fa_rows.csv', float_precision='round_trip')
E = pd.read_csv(f'{V}/partB2_indep/indep_estimates.csv')
K = ['unit', 'clf', 'arm', 'grid']
J = M.merge(I, on=K, suffixes=('_m', '_i'), how='outer', indicator=True)
print('merge:', J._merge.value_counts().to_dict())
Jb = J[J._merge == 'both']
for g in ('new', 'old'):
    x = Jb[Jb.grid == g]
    d = (x.FA_m - x.FA_i).abs()
    nan_mis = int((x.FA_m.isna() != x.FA_i.isna()).sum())
    print(f'[{g}] 行 {len(x)}：FA 最大|Δ| {d.max():.2e}，NaN 不一致 {nan_mis}；reach 最大|Δ| {(x.reach_m - x.reach_i).abs().max():.2e}；'
          f'OA★ 最大|Δ| {(x.OA_star_m - x.OA_star_i).abs().max():.2e}；partial 不一致 {int((x.partial_m.astype(bool) != x.partial_i.astype(bool)).sum())}；'
          f'missing 不一致 {int((x.fa_missing.astype(bool) != x.missing.astype(bool)).sum())}')
    for arm in sorted(x.arm.unique()):
        xa = x[x.arm == arm]
        print(f'    {arm:6s} n={len(xa)} FA max|Δ| {(xa.FA_m - xa.FA_i).abs().max():.2e} 缺失 {int(xa.FA_m.isna().sum())}/{int(xa.FA_i.isna().sum())}')
print('仅主实现有的 arm：', sorted(J[J._merge == 'left_only'].arm.unique()), ' 仅独立实现：', sorted(J[J._merge == 'right_only'].arm.unique()))

# ---- 自写国家层汇总（只用 fa_rows_B2）
COUNTRIES = 'AT,BE,DE,EE,LT,LV,PT,RO,SE,SI,SK'.split(',')
DROP = ['RO', 'PT']
W = M[M.grid == 'new'].pivot_table(index=['target', 'unit', 'clf'], columns='arm', values='FA', aggfunc='first', dropna=False).reset_index()


def cvals(a, b):
    d = W[a] - W[b]
    T = pd.DataFrame({'target': W.target, 'clf': W.clf, 'd': d})
    ae = T[T.clf == 'ae'].groupby('target').d.mean()
    s2 = T[T.clf == 's2'].groupby('target').d.mean()
    both = pd.concat([ae, s2], axis=1).mean(axis=1, skipna=False)
    return {'ae': ae, 's2': s2, 'both': both}


def ts(v):
    v = v.dropna()
    n = len(v); m = v.mean(); sd = v.std(ddof=1); q = stats.t.ppf(.975, n - 1)
    return n, m, m - q * sd / np.sqrt(n), m + q * sd / np.sqrt(n), int((v > 0).sum())


mine = {}
defs = {'A_50': ('PK50', 'Z0'), 'A_200': ('PK200', 'Z0'), 'A_400': ('PK400', 'Z0'), 'A_o': ('PKo', 'Z0'),
        'R_50': ('RT50', 'PK50'), 'R_200': ('RT200', 'PK200'), 'R_400': ('RT400', 'PK400'),
        'H_50': ('HD50', 'PK50'), 'H_200': ('HD200', 'PK200'), 'H_400': ('HD400', 'PK400'),
        'dRT_200': ('RT200', 'Z0')}
print('\n国家层（自算，fa_rows_B2，新网格）  vs  独立实现 indep_estimates')
print(E.est.unique())
for name, (a, b) in defs.items():
    cv = cvals(a, b)
    for f in ('both', 'ae', 's2'):
        for s9 in (True, False):
            v = cv[f].drop(DROP) if s9 else cv[f]
            r = ts(v)
            mine[(name, f, s9)] = r
            e = E[(E.est == name) & (E.feat == f) & (E.set == ('9_new' if s9 else '11_new'))]
            etxt = (f'{e.iloc[0]["mean"]:+.4f} [{e.iloc[0].lo:+.4f},{e.iloc[0].hi:+.4f}] {int(e.iloc[0].npos)}/{int(e.iloc[0].n)}'
                    if len(e) else '—')
            if f == 'both' or name in ('A_200', 'R_200', 'A_o'):
                print(f'  {name:7s} {f:4s} {"9" if s9 else "11":2s}: {r[1]:+.4f} [{r[2]:+.4f},{r[3]:+.4f}] {r[4]}/{r[0]}   | indep {etxt}')
# 占比
for k in (50, 200, 400):
    a = cvals(f'PK{k}', 'Z0')['both'].drop(DROP).mean(); d = cvals(f'RT{k}', 'Z0')['both'].drop(DROP).mean()
    a11 = cvals(f'PK{k}', 'Z0')['both'].mean(); d11 = cvals(f'RT{k}', 'Z0')['both'].mean()
    print(f'  share k={k}: 9 国 {a / d:.4f}  11 国 {a11 / d11:.4f}')
for f in ('ae', 's2'):
    a = cvals('PK200', 'Z0')[f].drop(DROP).mean(); d = cvals('RT200', 'Z0')[f].drop(DROP).mean()
    print(f'  share k=200 {f} 9 国 {a / d:.4f}')
# 读法
n, m, lo, hi, npos = mine[('R_200', 'both', True)]
a = cvals('PK200', 'Z0')['both'].drop(DROP).mean(); d = cvals('RT200', 'Z0')['both'].drop(DROP).mean()
rd = 1 if lo > 0 else (2 if (lo <= 0 <= hi and a / d >= 0.7) else 3)
print(f'读法（自算）：R_200 CI [{lo:+.4f},{hi:+.4f}]，占比 {a / d:.4f} → 第 {rd} 条')
# 国家值逐国（both）A_200 / R_200 / dRT200
print('\n逐国 both：A200 R200 dRT200 Ao')
t = pd.DataFrame({k: cvals(*defs[k])['both'] for k in ('A_200', 'R_200', 'dRT_200', 'A_o')})
print(t.round(2).to_string())
# 单元层 A_200、R_200、A_o 与独立实现比
for name in ('A_200', 'R_200', 'A_o'):
    a, b = defs[name]
    Mi = I[I.grid == 'new'].pivot_table(index=['unit', 'clf'], columns='arm', values='FA', aggfunc='first', dropna=False)
    Mm = M[M.grid == 'new'].pivot_table(index=['unit', 'clf'], columns='arm', values='FA', aggfunc='first', dropna=False)
    dm, di = Mm[a] - Mm[b], Mi[a] - Mi[b]
    dd = (dm - di.reindex(dm.index)).abs()
    print(f'单元层 {name}：{len(dm)} 个单元×特征，最大|Δ| {dd.max():.2e}，NaN 不一致 {int((dm.isna() != di.reindex(dm.index).isna()).sum())}')
