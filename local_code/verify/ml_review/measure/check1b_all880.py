"""(1b) 全部 44 单元 × 20 个选样：sel 与 idx_pool[sel_pos] 一致、与测试/外国不相交、哈希与 G 行一致、G probs 存在且行数 = 测试集；
build_manifest 与 tier1_sel_index.csv 的对应子集一致；统计缓存命中（AE 与 S2 选样相同）后的实际拟合次数；测试集规模。"""
import os
import sys

L = sys.argv[1]
sys.path.insert(0, L)
os.chdir(L)
os.environ['ML_OUT'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scratch_out')
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import guidance_G as G  # noqa: E402
import multilearner_G as ML  # noqa: E402

GOUT = G.OUT
problems = []
rows = []
ntest = {}
for tgt, split, rep in G.unit_order():
    S = np.load(os.path.join(GOUT, 'stage0', f'{tgt}_s{split}.npz'))
    allT, test, fr, cand = S['allT'], S['test'], S['foreign'], S['cand']
    ntest[(tgt, split)] = len(test)
    pool_pos = G.pool_positions({'cand': cand}, split, rep)
    idx_pool = allT[pool_pos]
    unit = f'{tgt}_s{split}_r{rep}'
    GR = pd.read_csv(os.path.join(GOUT, f'rows_t1_{unit}.csv'), usecols=['cond', 'method', 'k', 'clf', 'sel_hash'],
                     dtype={'sel_hash': str})
    for cond, method, k in ML.unit_jobs(split, rep, None):
        if k == 0:
            continue
        z = np.load(ML.sel_path(tgt, split, rep, cond, method, k))
        sel = z['sel']
        if not np.array_equal(sel, idx_pool[z['sel_pos']]):
            problems.append((unit, cond, method, k, 'sel != idx_pool[sel_pos]'))
        if len(sel) != k or len(np.unique(sel)) != k:
            problems.append((unit, cond, method, k, 'len/unique', len(sel), len(np.unique(sel))))
        if np.intersect1d(sel, test).size or np.intersect1d(sel, fr).size:
            problems.append((unit, cond, method, k, 'overlap test/foreign'))
        h = G.sel_hash(sel)
        for clf in ('s2', 'ae'):
            g = GR[(GR.cond == cond) & (GR.method == method) & (GR.k == k) & (GR.clf == clf)]
            if len(g) != 1 or g.sel_hash.iloc[0] != h:
                problems.append((unit, cond, method, k, clf, 'G row hash', list(g.sel_hash), h))
            p = os.path.join(GOUT, 'probs', f'{unit}_{clf}_k{k}_{h}.npz')
            if not os.path.exists(p):
                problems.append((unit, cond, method, k, clf, 'G probs missing'))
        rows.append(dict(tgt=tgt, split=split, rep=rep, cond=cond, method=method, k=k, h=h, n=len(sel),
                         pi_sel_len=len(z['pi_sel'])))
M = pd.DataFrame(rows)
print('选样数', len(M), '；问题', len(problems))
for p in problems[:30]:
    print('  ', p)
# 实际拟合数：每单元 (clf, k, h) 去重 + k=0 每 (tgt, split, clf) 一次
u = M.drop_duplicates(['tgt', 'split', 'rep', 'k', 'h'])
fits = 2 * len(u) + 2 * 22
print('去重后 (单元, k, 哈希)：', len(u), '；每学习器实际拟合次数 =', fits, '（方案写 1,804）')
dup = M[M.duplicated(['tgt', 'split', 'rep', 'k', 'h'], keep=False)]
print('重复哈希的选样：', len(dup))
if len(dup):
    print(dup.to_string())
print('pi_sel 非空的方法：', M[M.pi_sel_len > 0].method.unique().tolist(), '；为空：', M[M.pi_sel_len == 0].method.unique().tolist())
# 与 tier1_sel_index.csv 对照
T = pd.read_csv('/Users/perng_mini/AlphaCrops_local/verify/multilearner/feasibility_compute/tier1_sel_index.csv',
                dtype={'h': str})
T = T[(T.method.isin(G.E1) & T.cond.isin(['AE', 'S2'])) | (T.method == 'srs')]
mm = M.merge(T, on=['tgt', 'split', 'rep', 'cond', 'method', 'k'], how='outer', indicator=True, suffixes=('', '_t'))
print('与 tier1_sel_index 子集对照：', mm._merge.value_counts().to_dict(), '；哈希不一致', int((mm.h != mm.h_t).sum()))
nt = pd.Series(ntest)
print('测试集规模：min', nt.min(), 'max', nt.max(), 'sum', nt.sum())
print(nt.sort_values().to_string())
nt.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ntest.csv'))
