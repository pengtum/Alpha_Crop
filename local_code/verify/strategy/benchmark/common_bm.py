"""策略基准视角：公共加载与工具函数（只读 G 结果，不训练、不选样）。"""
import os
import sys

os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')

import numpy as np
import pandas as pd
from scipy import stats

LE = '/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops/label_expansion'
sys.path.insert(0, LE)
import analyze_G as A  # noqa: E402
from guidance_G import E1, FAMILIES  # noqa: E402

OUTD = '/Users/perng_mini/AlphaCrops_local/verify/strategy/benchmark'
UNIT = ['target', 'split', 'rep']
PRED = FAMILIES['pred']
HYB = FAMILIES['hyb']
ALL8 = E1 + PRED + HYB
COUNTRIES = ['AT', 'BE', 'DE', 'EE', 'LT', 'LV', 'PT', 'RO', 'SE', 'SI', 'SK']


def load():
    R = A.load_rows()
    return R


def vs_srs(R, cond, methods, metric='ba_ge30', ks=(50, 200), clfs=('s2', 'ae'), rep0_only=False):
    """方法 − SRS（同一单元、分类器、k）。E1 默认 44 单元；预测/混合法只有重复 0。"""
    RR = R[R.rep == 0] if rep0_only else R
    d = A.vs_srs(RR, cond, methods, metric, ks, clfs)
    d = d[['target', 'split', 'rep', 'method', 'clf', 'k', 'd']].copy()
    d['cond'] = cond
    return d


def all_vs_srs(R, metric='ba_ge30', rep0_only=False):
    out = [vs_srs(R, c, ALL8, metric, rep0_only=rep0_only) for c in ('AE', 'S2')]
    return pd.concat(out, ignore_index=True)


def cv(d):
    """国家值：先在 (方法, 分类器, k[, cond]) 内对单元平均，再等权平均。"""
    keys = ['target', 'method', 'clf', 'k'] + (['cond'] if 'cond' in d and d.cond.nunique() > 1 else [])
    return d.groupby(keys).d.mean().groupby('target').mean()


def ts(v):
    return A.t_summary(v)


def bh(p):
    p = pd.Series(p, dtype=float)
    o = p.sort_values()
    m = len(p)
    q = o.to_numpy() * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    return pd.Series(np.minimum(q, 1), index=o.index).reindex(p.index)


def holm(p):
    return A.holm(pd.Series(p, dtype=float))
