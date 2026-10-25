"""C 选择只用外国池（不接触任何目标国数据）：在新网格上跑 ML.select_C，并记录每个 C 在各折上的 lbfgs 迭代数（收敛检查）。
输出写到本目录的 csel_out/（不写项目或正式结果目录）。用法：python csel_check.py SI 0
"""
import json
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.getcwd())
os.environ['ML_OUT'] = '/Users/perng_mini/AlphaCrops_local/verify/ml_review/static/csel_out'
import guidance_G as G
import multilearner_G as ML
from stageA import class_weights_from

tgt, split = sys.argv[1], int(sys.argv[2])
D = G.GData()
S = dict(np.load(os.path.join(G.OUT, 'stage0', f'{tgt}_s{split}.npz')))
for feat in ('ae', 's2'):
    best = ML.select_C(D, S, tgt, split, feat, 2, ML.OUT)
    loss = json.load(open(os.path.join(ML.OUT, 'lr_C', f'{tgt}_s{split}_{feat}.json')))['loss']
    print(f'{tgt} s{split} {feat}: C* = {best}；外国留国家组加权 log-loss：',
          {k: round(v, 4) for k, v in loss.items()}, flush=True)

# 收敛：一个折、两端的 C 与 C*
from sklearn.linear_model import LogisticRegression
fr = S['foreign']
y = D.y[fr]
cw = class_weights_from(y)
w = cw[y]
groups = G.fold_groups(D, tgt, split, fr)
print('fold_groups 组大小：', np.bincount(groups).tolist(), '；每组国家：',
      {g: sorted(set(D.ctry[fr][groups == g])) for g in np.unique(groups)})
for feat in ('ae', 's2'):
    X = np.asarray(D.X[feat][fr], dtype=np.float64)
    tr = groups != 0
    mu, sd = ML._std(X[tr])
    A = (X[tr] - mu) / sd
    for C in (1e-4, 1e-3, 1.0, 10.0):
        with warnings.catch_warnings(record=True) as cw_:
            warnings.simplefilter('always')
            m = LogisticRegression(C=C, **ML.PARAMS['lr']).fit(A, y[tr], sample_weight=w[tr] / w[tr].mean())
        print(f'  {feat} C={C:g}: n_iter {int(m.n_iter_.max())}，警告 {len(cw_)}', flush=True)
