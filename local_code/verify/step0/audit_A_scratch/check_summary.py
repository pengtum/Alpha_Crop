# -*- coding: utf-8 -*-
"""核对 summary_A.txt：按其各节顺序，用自写的聚合（直接从 rows_A.csv）重算每一个 "11国 m [lo, hi] k/11 | 9国 …" 元组、
恢复率、A-c6 行，并与 summary 中按顺序抽取的数字逐一比较（容差 = 显示位数的半个单位）。另核对 七、对账节。"""
import os
import re
import sys

sys.dont_write_bytecode = True
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_v] = '2'
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

A_DIR = '/Users/perng_mini/AlphaCrops_local/verify/step0/partA'
C11 = 'AT,BE,DE,EE,LT,LV,PT,RO,SE,SI,SK'.split(',')
C9 = [c for c in C11 if c not in ('RO', 'PT')]
N0 = (100, 500, 2000)
FE = ('ae', 's2', 'avg')
BETTI = ('Betti-Admin', 'Betti-Img-ae', 'Betti-Img-s2')
rows = pd.read_csv(os.path.join(A_DIR, 'rows_A.csv'))
txt = open(os.path.join(A_DIR, 'summary_A.txt')).read()
cv = rows.groupby(['feat', 's0_type', 'n0', 'arm', 'k_or_B', 'estimator', 'target'])[
    ['oa', 'ba_ge30', 'prior_L1_test', 'n_labels_S1', 's0_L1_frame', 'n_regions']].mean()


def V(m, f, s0, n0, arm, kb, est):
    if f == 'avg':
        return (V(m, 'ae', s0, n0, arm, kb, est) + V(m, 's2', s0, n0, arm, kb, est)) / 2
    return cv.loc[(f, s0, n0, arm, kb, est), m].reindex(C11)


def REF(m, f, est):
    return V(m, f, 'ref', 0, 'none', 0, est)


def st(v, cs):
    a = v.reindex(cs).to_numpy(float); n = len(a)
    h = stats.t.ppf(0.975, n - 1) * a.std(ddof=1) / np.sqrt(n)
    return a.mean(), a.mean() - h, a.mean() + h, int((a > 0).sum())


exp_t, exp_r = [], []   # 期望的 fmt 元组（带位数、是否带计数）与恢复率


def T(v, d=2, pos=True, lab=''):
    exp_t.append((lab, d, pos, st(v, C11), st(v, C9)))


def RC(num, den, lab=''):
    exp_r.append((lab, [(num.reindex(cs).mean() / den.reindex(cs).mean(), den.reindex(cs).mean() <= 0) for cs in (C11, C9)]))


# 〇
for f in FE:
    for e in ('E_none', 'E_oracle', 'E_EM'):
        T(REF('oa', f, e), pos=False, lab=f'ref {f} {e}')
    for e in ('E_oracle', 'E_EM'):
        T(REF('oa', f, e) - REF('oa', f, 'E_none'), lab=f'ref {f} {e}-none OA')
        T(REF('ba_ge30', f, e) - REF('ba_ge30', f, 'E_none'), lab=f'ref {f} {e}-none BA')
    RC(REF('oa', f, 'E_EM') - REF('oa', f, 'E_none'), REF('oa', f, 'E_oracle') - REF('oa', f, 'E_none'), f'EM {f}')
for e in ('E_none', 'E_EM'):
    T(REF('prior_L1_test', 'ae', e), 3, False, f'L1 {e}')
# S0 描述
for n0 in N0:
    for s0 in ('clus', 'rand'):
        T(V('s0_L1_frame', 'ae', s0, n0, 'none', 0, 'E_S0'), 3, False, f'bias {s0} {n0}')
    T(V('s0_L1_frame', 'ae', 'clus', n0, 'none', 0, 'E_S0') - V('s0_L1_frame', 'ae', 'rand', n0, 'none', 0, 'E_S0'), 3, lab=f'bias diff {n0}')
# A-c1
for n0 in N0:
    Lc = V('prior_L1_test', 'ae', 'clus', n0, 'none', 0, 'E_S0'); Lr = V('prior_L1_test', 'ae', 'rand', n0, 'none', 0, 'E_S0')
    T(Lc, 3, False, f'c1 L1 clus {n0}'); T(Lr, 3, False, f'c1 L1 rand {n0}'); T(Lc - Lr, 3, lab=f'c1 L1 diff {n0}')
    for f in FE:
        on, oo = REF('oa', f, 'E_none'), REF('oa', f, 'E_oracle')
        oc, orr = V('oa', f, 'clus', n0, 'none', 0, 'E_S0'), V('oa', f, 'rand', n0, 'none', 0, 'E_S0')
        RC(oc - on, oo - on, f'c1 rec clus {f} {n0}'); RC(orr - on, oo - on, f'c1 rec rand {f} {n0}')
        T(oc - on, lab=f'c1 {f} clus-none {n0}'); T(orr - on, lab=f'c1 {f} rand-none {n0}'); T(oc - orr, lab=f'c1 {f} clus-rand {n0}')
# A-c2
for s0 in ('clus', 'rand'):
    for n0 in N0:
        L1 = V('prior_L1_test', 'ae', s0, n0, 'SRS', 200, 'E_S1'); Lp = V('prior_L1_test', 'ae', s0, n0, 'SRS', 200, 'E_pool')
        L0 = V('prior_L1_test', 'ae', s0, n0, 'none', 0, 'E_S0')
        T(L1 - Lp, 3, lab=f'c2 L1 S1-pool {s0} {n0}'); T(L1 - L0, 3, lab=f'c2 L1 S1-S0 {s0} {n0}')
        for f in FE:
            s1 = V('oa', f, s0, n0, 'SRS', 200, 'E_S1'); pl = V('oa', f, s0, n0, 'SRS', 200, 'E_pool')
            e0 = V('oa', f, s0, n0, 'none', 0, 'E_S0')
            T(s1 - pl, lab=f'c2 {f} S1-pool {s0} {n0}'); T(s1 - e0, lab=f'c2 {f} S1-S0 {s0} {n0}')
for f in FE:
    on, oo = REF('oa', f, 'E_none'), REF('oa', f, 'E_oracle'); s1 = V('oa', f, 'none', 0, 'SRS', 200, 'E_S1')
    T(s1 - on, lab=f'c2 none {f}'); RC(s1 - on, oo - on, f'c2 none rec {f}')
# A-c3
lev = []
for n0 in N0:
    T(V('n_labels_S1', 'ae', 'clus', n0, 'SPR', 20, 'E_S1'), 1, False, f'c3 nlab {n0}')
    for est in ('E_pool', 'E_S1'):
        L = {a: V('prior_L1_test', 'ae', 'clus', n0, a, 20, est) for a in ('EXT', 'SPR', 'SPR=')}
        lev.append((f'c3 L1 lev {n0} {est}', [L[a].mean() for a in ('EXT', 'SPR', 'SPR=')], 3))
        T(L['SPR'] - L['EXT'], 3, lab=f'c3 L1 SPR-EXT {n0} {est}'); T(L['SPR='] - L['EXT'], 3, lab='')
        T(L['SPR='] - L['SPR'], 3, lab='')
        for f in FE:
            O = {a: V('oa', f, 'clus', n0, a, 20, est) for a in ('EXT', 'SPR', 'SPR=')}
            on, oo = REF('oa', f, 'E_none'), REF('oa', f, 'E_oracle')
            lev.append((f'c3 OA lev {n0} {est} {f}', [O[a].mean() for a in ('EXT', 'SPR', 'SPR=')], 2))
            for a in ('EXT', 'SPR', 'SPR='):
                RC(O[a] - on, oo - on, f'c3 rec {a} {f} {n0} {est}')
            T(O['SPR'] - O['EXT'], lab=f'c3 OA SPR-EXT {n0} {est} {f}'); T(O['SPR='] - O['EXT'], lab=f'c3 OA SPR=-EXT {n0} {est} {f}')
            T(O['SPR='] - O['SPR'], lab=f'c3 OA SPR=-SPR {n0} {est} {f}')
# A-c4
for k in (50, 200):
    for s0, n0 in [('none', 0)] + [('clus', n) for n in N0] + [('rand', n) for n in N0]:
        L = {a: V('prior_L1_test', 'ae', s0, n0, a, k, 'E_S1') for a in ('SRS',) + BETTI}
        for a in BETTI:
            T(L[a] - L['SRS'], 3, lab=f'c4 L1 {a} {k} {s0} {n0}')
        T(L['Betti-Img-ae'] - L['Betti-Img-s2'], 3, lab='')
        for f in FE:
            O = {a: V('oa', f, s0, n0, a, k, 'E_S1') for a in ('SRS',) + BETTI}
            for a in BETTI:
                T(O[a] - O['SRS'], lab=f'c4 OA {a} {k} {s0} {n0} {f}')
            T(O['Betti-Img-ae'] - O['Betti-Img-s2'], lab=f'c4 OA ae-s2 {k} {s0} {n0} {f}')
# A-c5
items = [('E_oracle', None), ('E_EM', None), ('n', ('none', 0, 'SRS', 50, 'E_S1')), ('n', ('none', 0, 'SRS', 200, 'E_S1'))]
for n0 in N0:
    for s0 in ('clus', 'rand'):
        items += [('x', (s0, n0, 'none', 0, 'E_S0')), ('x', (s0, n0, 'SRS', 200, 'E_pool')), ('x', (s0, n0, 'SRS', 200, 'E_S1'))]
    items.append(('x', ('clus', n0, 'SPR=', 20, 'E_S1')))
for f in FE:
    bn = REF('ba_ge30', f, 'E_none')
    for nm, sp in items:
        T((REF('ba_ge30', f, nm) if sp is None else V('ba_ge30', f, *sp)) - bn, lab=f'c5 {f} {nm} {sp}')

# ---- 抽取 summary 中的元组（按顺序）
num = r'([+-]\d+\.\d+)'
pat = re.compile(rf'11国 {num} \[{num}, {num}\](?: (\d+)/11)?  \|  9国 {num} \[{num}, {num}\](?: (\d+)/9)?')
got = pat.findall(txt)
bad = []
print(f'期望元组 {len(exp_t)}，summary 中抽到 {len(got)}')
for i, ((lab, d, pos, a, b), g) in enumerate(zip(exp_t, got)):
    tol = 0.5 * 10 ** -d + 1e-9
    ga = [float(g[0]), float(g[1]), float(g[2])]; gb = [float(g[4]), float(g[5]), float(g[6])]
    ok = all(abs(x - y) <= tol for x, y in zip(ga, a[:3])) and all(abs(x - y) <= tol for x, y in zip(gb, b[:3]))
    if pos:
        ok &= g[3] != '' and int(g[3]) == a[3] and g[7] != '' and int(g[7]) == b[3]
    else:
        ok &= g[3] == '' and g[7] == ''
    if not ok:
        bad.append((i, lab, g, np.round(a, 4), np.round(b, 4)))
print('元组不一致：', len(bad)); [print(x) for x in bad[:10]]
# 恢复率
rp = re.compile(r'11国 (-?\d+\.\d)%(†?)  \|  9国 (-?\d+\.\d)%(†?)')
gr = rp.findall(txt)
print(f'期望恢复率 {len(exp_r)}，抽到 {len(gr)}')
badr = []
for (lab, vals), g in zip(exp_r, gr):
    (r11, d11), (r9, d9) = vals
    ok = abs(float(g[0]) - 100 * r11) <= 0.05 + 1e-9 and abs(float(g[2]) - 100 * r9) <= 0.05 + 1e-9 \
        and (g[1] == '†') == bool(d11) and (g[3] == '†') == bool(d9)
    if not ok:
        badr.append((lab, g, r11, r9, d11, d9))
print('恢复率不一致：', len(badr)); [print(x) for x in badr[:10]]
# A-c3 水平
lp = re.compile(r'(?:先验 L1 水平|OA 水平)：EXT (\d+\.\d+)，SPR (\d+\.\d+)，SPR= (\d+\.\d+)')
gl = lp.findall(txt)
badl = [(lab, g, v) for (lab, v, d), g in zip(lev, gl) if any(abs(float(x) - y) > 0.5 * 10 ** -d + 1e-9 for x, y in zip(g, v))]
print(f'A-c3 水平：期望 {len(lev)}，抽到 {len(gl)}，不一致 {len(badl)}', badl[:3])
# A-c6
p6 = re.compile(rf'n0=(\d+): {num} \[{num}, {num}\] (\d+)/11; 9国 {num} \[{num}, {num}\] (\d+)/9')
g6 = p6.findall(txt)
exp6 = []
for s0 in ('clus', 'rand'):
    for arm, kb in (('SRS', 50), ('SRS', 200)) + ((('SPR=', 20), ('EXT', 20)) if s0 == 'clus' else ()):
        for f in FE:
            for n0 in N0:
                dd = V('oa', f, s0, n0, arm, kb, 'E_pool') - V('oa', f, s0, n0, arm, kb, 'E_S1')
                exp6.append((f'{s0} {arm} {kb} {f} {n0}', n0, st(dd, C11), st(dd, C9)))
bad6 = []
for (lab, n0, a, b), g in zip(exp6, g6):
    ok = int(g[0]) == n0 and all(abs(float(x) - y) <= 0.005 + 1e-9 for x, y in zip(g[1:4], a[:3])) and int(g[4]) == a[3] \
        and all(abs(float(x) - y) <= 0.005 + 1e-9 for x, y in zip(g[5:8], b[:3])) and int(g[8]) == b[3]
    if not ok:
        bad6.append((lab, g, a, b))
print(f'A-c6：期望 {len(exp6)}，抽到 {len(g6)}，不一致 {len(bad6)}', bad6[:3])
# 区数分布
for n0 in N0:
    sub = rows[(rows.s0_type == 'clus') & (rows.n0 == n0) & (rows.estimator == 'E_S0') & (rows.feat == 'ae')]
    print(n0, dict(sub.n_regions.value_counts().sort_index()))
