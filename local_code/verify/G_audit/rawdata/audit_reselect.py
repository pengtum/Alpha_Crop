"""选样复现：(a) 独立实现 margin（最小 p1−p2），核对 AE/S2/ORACLE 的 margin 选样确实来自各自的 P_T；
(b) 用项目的 select_one 在随机单元上重跑各 (cond, method)，与 sel 文件逐位比对（确认文件与条件名对应，无错位）。
在 label_expansion 目录下运行；只读。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
import guidance_G as G  # noqa: E402

RES = '/Users/perng_mini/AlphaCrops_local/results/guidance_G'
rng = np.random.default_rng(2026)
countries = ['AT', 'BE', 'DE', 'EE', 'LT', 'LV', 'PT', 'RO', 'SE', 'SI', 'SK']


def pool_pos_of(n_cand, split, rep):
    r = np.random.default_rng(10_000 * split + 100 * rep + 7)
    return np.sort(r.choice(n_cand, min(30_000, n_cand), replace=False))


def load_sel(unit, cond, method, k):
    return np.load(os.path.join(RES, 'sel', f'{unit}_{cond}_{method}_k{k}.npz'))['sel_pos']


print('== (a) 独立 margin')
n_ok = n_all = 0
for tgt in countries:
    for s in (0, 1):
        S = np.load(os.path.join(RES, 'stage0', f'{tgt}_s{s}.npz'))
        for r in (0,):
            pp = pool_pos_of(len(S['cand']), s, r)
            O = np.load(os.path.join(RES, 'oracle', f'{tgt}_s{s}.npz'))
            srcs = {'AE': S['P_T_ae'][pp], 'S2': S['P_T_s2'][pp], 'ORACLE': O['P_T']}
            for k in (50, 200):
                for cond, P in srcs.items():
                    Ps = np.sort(P.astype(np.float64), 1)
                    mg = Ps[:, -1] - Ps[:, -2]
                    mine = np.sort(np.argsort(mg, kind='stable')[:k])
                    f = load_sel(f'{tgt}_s{s}_r{r}', cond, 'margin', k)
                    ok = np.array_equal(mine, f)
                    # 与"另一种指导"的 margin 选样是否会一样（排除错位）
                    n_all += 1
                    n_ok += ok
                    if not ok:
                        print(tgt, s, r, cond, k, 'overlap', len(np.intersect1d(mine, f)))
print(f'margin 独立复现 {n_ok}/{n_all}')

print('== (b) select_one 复现')
D = G.GData()
combos = [(c, m, 'r01') for c in ('AE', 'S2') for m in G.E1] + \
         [(c, m, 'r0') for c in ('AE', 'S2') for m in G.PRED + G.HYB] + \
         [(c, m, 'r0') for c in ('XY', 'S2XY') for m in ('lpm2_nogeo', 'kcenter')] + \
         [('ORACLE', m, 'r0') for m in ('lpm2_nogeo', 'typiclust', 'margin')] + [('none', 'srs', 'r01')]
res = []
for cond, method, reps in combos:
    tgt = countries[rng.integers(11)]
    s = int(rng.integers(2))
    r = int(rng.integers(2)) if reps == 'r01' else 0
    k = int(rng.choice([50, 200]))
    S = dict(np.load(os.path.join(RES, 'stage0', f'{tgt}_s{s}.npz')))
    pp = G.pool_positions(S, s, r)
    oracle = dict(np.load(os.path.join(RES, 'oracle', f'{tgt}_s{s}.npz'))) if cond == 'ORACLE' else None
    seed = 1000 * s + r
    if method == 'srs':
        # srs 由 jobs_for 以 cond='none' 调度；select_one 对 none 条件同样可用
        pass
    sel_pos, meta = G.select_one(D, S, pp, cond, method, k, seed, oracle)
    f = load_sel(f'{tgt}_s{s}_r{r}', cond, method, k)
    same = np.array_equal(np.sort(sel_pos), f)
    other = {'AE': 'S2', 'S2': 'AE'}.get(cond)
    cross = None
    if other:
        fo = load_sel(f'{tgt}_s{s}_r{r}', other, method, k)
        cross = len(np.intersect1d(sel_pos, fo))
    res.append((cond, method, tgt, s, r, k, same, cross))
    print(cond, method, f'{tgt}_s{s}_r{r}', k, 'reproduced' if same else 'MISMATCH', 'overlap with other-cond file:', cross,
          flush=True)
print('复现', sum(x[6] for x in res), '/', len(res))
