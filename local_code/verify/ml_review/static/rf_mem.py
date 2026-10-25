"""RF 内存：只在外国 5 万（stage0 的 foreign）上拟合 500 棵树，不做任何测试集预测，只记节点数与峰值 RSS。
用法（在 label_expansion 目录下）：python rf_mem.py SI 0 s2
"""
import os
import resource
import sys
import time

import numpy as np

sys.path.insert(0, os.getcwd())
import guidance_G as G
from stageA import class_weights_from

tgt, split, feat = sys.argv[1], int(sys.argv[2]), sys.argv[3]
D = G.GData()
S = np.load(os.path.join(G.OUT, 'stage0', f'{tgt}_s{split}.npz'))
fr = S['foreign']
y = D.y[fr]
X = np.asarray(D.X[feat][fr], dtype=np.float64)
cw = class_weights_from(y)
rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30
from sklearn.ensemble import RandomForestClassifier
t = time.time()
m = RandomForestClassifier(n_estimators=500, max_features='sqrt', min_samples_leaf=1, bootstrap=True, class_weight=None,
                           random_state=0, n_jobs=4).fit(X, y, sample_weight=cw[y])
t_fit = time.time() - t
nodes = np.array([e.tree_.node_count for e in m.estimators_])
depth = np.array([e.tree_.max_depth for e in m.estimators_])
rss1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30
per_node = 8 * 8 + 10 * 8  # Node 结构体 64 B + value 数组（1 输出 × 10 类 × float64）
print(f'{tgt} s{split} {feat}: n_train {len(y)}, fit {t_fit:.1f}s, 节点/树 均值 {nodes.mean():.0f} 最大 {nodes.max()}，'
      f'深度均值 {depth.mean():.1f}；估计树内存 {nodes.sum() * per_node / 2 ** 30:.2f} GB；'
      f'峰值 RSS 拟合前 {rss0:.2f} GB → 拟合后 {rss1:.2f} GB', flush=True)
