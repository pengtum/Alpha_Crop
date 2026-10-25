"""静态审查辅助：只读选样文件、阶段 0 与 G 行的结构列（不打印任何精度列）。
在 label_expansion 目录下运行：python check_manifest.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
os.environ['ML_OUT'] = '/Users/perng_mini/AlphaCrops_local/verify/ml_review/static/ml_out_unused'
import guidance_G as G
import multilearner_G as ML

# 1) 清单：与 tier1_sel_index.csv 对照
M = ML.build_manifest()
I = pd.read_csv('/Users/perng_mini/AlphaCrops_local/verify/multilearner/feasibility_compute/tier1_sel_index.csv')
print('manifest 行数', len(M), '；E1', (M.method.isin(G.E1)).sum(), '；SRS', (M.method == 'srs').sum())
print('manifest 按 (cond, method, k) 计数:\n', M.groupby(['cond', 'method', 'k']).size().to_string())
key = ['tgt', 'split', 'rep', 'cond', 'method', 'k']
J = M.merge(I, on=key, how='left', suffixes=('', '_idx'))
print('在 tier1 索引中找到：', J.h_idx.notna().sum(), '/', len(J), '；哈希一致：', (J.h == J.h_idx).sum(),
      '；n 一致：', (J.n == J.n_idx).sum())
print('索引中未被使用的行（方法 × k）:\n', I.merge(M[key], on=key, how='left', indicator=True)
      .query('_merge == "left_only"').groupby(['method', 'k']).size().to_string())

# 2) 每个选样：sel == allT[pool_pos][sel_pos]；长度 = k；与测试集、外国池不相交；pi_sel 是否存在
bad = []
pi_info = {}
for (tgt, split), g in M.groupby(['tgt', 'split']):
    S = np.load(os.path.join(G.OUT, 'stage0', f'{tgt}_s{split}.npz'))
    allT, test, fr, cand = S['allT'], S['test'], S['foreign'], S['cand']
    for rep, gg in g.groupby('rep'):
        pp = G.pool_positions({'cand': cand}, split, rep)
        idx_pool = allT[pp]
        for r in gg.itertuples():
            z = np.load(ML.sel_path(tgt, split, rep, r.cond, r.method, r.k))
            sel, sp, pis = z['sel'], z['sel_pos'], z['pi_sel']
            ok = (np.array_equal(sel, idx_pool[sp]) and len(sel) == r.k and len(np.unique(sel)) == r.k
                  and not np.intersect1d(sel, test).size and not np.intersect1d(sel, fr).size
                  and np.isin(sel, cand).all())
            if not ok:
                bad.append((tgt, split, rep, r.cond, r.method, r.k))
            pi_info.setdefault((r.method, len(pis) > 0), 0)
            pi_info[(r.method, len(pis) > 0)] += 1
print('选样一致性不通过：', bad[:5], len(bad))
print('pi_sel 是否非空（方法, 有 pi）→ 个数：', pi_info)

# 3) 实际拟合次数（同一单元、同一 k、同一哈希只拟合一次；k=0 每 (国家, 划分) 一次）
u = M.drop_duplicates(['tgt', 'split', 'rep', 'k', 'h'])
print('每种特征的去重选样数', len(u), '；每个学习器的拟合次数 = 2 ×', len(u), '+ 2 × 22 =', 2 * len(u) + 44)
dup = M[M.duplicated(['tgt', 'split', 'rep', 'k', 'h'], keep=False)]
print('AE/S2 或方法间选样完全相同的组数：', dup.groupby(['tgt', 'split', 'rep', 'k', 'h']).ngroups)

# 4) analyze_ML 的 LightGBM 行筛选：来自哪些层级文件、每单元行数、是否有重复键
import glob
rows = []
for f in glob.glob(os.path.join(G.OUT, 'rows_t*_*.csv')):
    R = pd.read_csv(f, usecols=lambda c: c in ('cond', 'method', 'clf', 'k', 'target', 'split', 'rep'))
    R['file_tier'] = os.path.basename(f).split('_')[1]
    rows.append(R)
R = pd.concat(rows, ignore_index=True)
Gf = R[R.method.isin(G.E1 + ['srs', 'foreign_only']) & R.cond.isin(['AE', 'S2', 'none'])
       & R.clf.isin(['s2', 'ae']) & R.k.isin([0, 50, 100, 200, 400])]
print('全部 G 行', len(R), '；筛选后', len(Gf), '；按层级文件：', Gf.file_tier.value_counts().to_dict())
print('被筛掉的层级 3/4 行中与筛选条件只差一项的（检查混入风险）：')
t34 = R[R.file_tier.isin(['t3', 't4', 't2'])]
print('  t2/t3/t4 各 cond × method × clf × k:\n',
      t34.groupby(['file_tier', 'cond', 'method', 'clf', 'k']).size().to_string())
kk = ['target', 'split', 'rep', 'cond', 'method', 'clf', 'k']
print('筛选后重复键数：', Gf.duplicated(kk).sum(), '；每单元行数分布：',
      Gf.groupby(['target', 'split', 'rep']).size().value_counts().to_dict())
