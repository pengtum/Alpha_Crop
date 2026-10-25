"""analyze_ML 的流程与判定逻辑检查：LR/MLP/RF 的行全部是合成的（用 G 行的键 + 人为设定的效应），不读、不算任何真实的新学习器精度。
LightGBM 行是 G 的真实行（结果已公开：report_G.txt θ_E1 = −0.054）。图输出被关掉（不写项目目录）。
用法（在 label_expansion 目录下）：python harness_analyze.py {noise|planted} OUTDIR
"""
import os
import sys

scen, outdir = sys.argv[1], sys.argv[2]
os.makedirs(outdir, exist_ok=True)
os.environ['ML_OUT'] = outdir
sys.argv = ['analyze_ML.py'] + (['--noise'] if scen == 'noise' else [])
sys.path.insert(0, os.getcwd())

import numpy as np
import pandas as pd
import analyze_G as A
import analyze_ML as AM

AM.plot = lambda *a, **k: None  # 不写 figures/

# 合成新学习器的行：键取自 G 层级 1 的筛选行
G = A.load_rows()
G = G[G.method.isin(AM.E1 + ['srs', 'foreign_only']) & G.cond.isin(['AE', 'S2', 'none'])
      & G.clf.isin(['s2', 'ae']) & G.k.isin([0, 50, 100, 200, 400])]
keys = G[['cond', 'method', 'clf', 'k', 'target', 'split', 'rep']].reset_index(drop=True)
rng = np.random.default_rng(123)
countries = sorted(keys.target.unique())
# planted：LR 的 AE 指导效应 +0.15（国家间 SD 0.05）→ 95% CI 不含 0，90% CI 在 ±0.3 内；MLP、RF 效应 0
theta = {'lr': 0.15, 'mlp': 0.0, 'rf': 0.0}
for L in AM.LEARNERS:
    eff = {c: theta[L] + rng.normal(0, 0.05) for c in countries}
    R = keys.copy()
    R['learner'] = L
    base = 60 + rng.normal(0, 0.02, len(R))
    add = np.where(R.cond == 'AE', R.target.map(eff), 0.0)
    for m in AM.METRICS:
        R[m] = base + add
    if L == 'lr':
        R['info_converged'] = True
    for (t, s, r), g in R.groupby(['target', 'split', 'rep']):
        g.to_csv(os.path.join(outdir, f'rows_{L}_{t}_s{s}_r{r}.csv'), index=False)
AM.main()
