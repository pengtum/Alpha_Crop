"""
F3b 的稳健性（只读；无训练）：
 (a) 重训模型换成第 200 轮（G 缓存里已有）：F3b 的增益是否与"少训练"重叠？
 (b) bal 头的设计权重开/关（predstrat 标签）；λ ∈ {0.02, 0.1, 0.5}
 (c) 逐国表（k=200，α=0.5）
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import numpy as np
import pandas as pd
from scipy import stats

__file__ = os.path.abspath(__file__)
exec(open(os.path.join(os.path.dirname(__file__), 'calib_diag.py')).read().split('rows = []')[0])


def lsm(L):
    L = L - L.max(1, keepdims=True)
    return L - np.log(np.exp(L).sum(1, keepdims=True))

df = pd.read_csv(os.path.join(OUT, 'frontier3_rows.csv'))
key = ['unit', 'target', 'clf', 'k']


def summ(j):
    c = j.groupby('target')[['dba', 'doa']].mean()
    out = []
    for col in ('dba', 'doa'):
        v = c[col]
        h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
        pos = int((v > 0).sum()) if col == 'dba' else int((v >= -0.3).sum())
        out.append(f'{col} {v.mean():+.2f} [{v.mean() - h:+.2f},{v.mean() + h:+.2f}] {pos}/11')
    return '  '.join(out)


def get(des, it, var, a, k):
    return df[(df.design == des) & (df.it == it) & (df['var'] == var) & (df.alpha == a) & (df.k == k)] \
        .set_index(key)[['ba', 'oa']]


def diff(x, y):
    return (x - y).dropna().rename(columns={'ba': 'dba', 'oa': 'doa'}).reset_index()


L = []
for des in ('srs_G', 'predstrat_AE'):
    for k in (50, 200):
        L.append(f'\n== {des} k={k} ==')
        r300 = get(des, 'i300', 'retrain', 0.0, k)
        r200 = get(des, 'i200', 'retrain', 0.0, k)
        L.append('  重训 i200 − 重训 i300               ' + summ(diff(r200, r300)))
        for a in (0.3, 0.4, 0.5):
            x = get(des, 'i200', 'F3b_l0.1_dw1', a, k)
            L.append(f'  F3b(i200) α={a} − 重训 i200        ' + summ(diff(x, r200)))
            L.append(f'  F3b(i200) α={a} − 重训 i300        ' + summ(diff(x, r300)))
        for lam in (0.02, 0.1, 0.5):
            for dw in ((1, 0) if des != 'srs_G' else (1,)):
                x = get(des, 'i300', f'F3b_l{lam}_dw{dw}', 0.5, k)
                L.append(f'  F3b λ={lam} 设计权重={dw} α=0.5 − 重训 i300  ' + summ(diff(x, r300)))

L.append('\n== 逐国（k=200，α=0.5，λ=0.1，i300）：ΔBA / ΔOA ==')
for des, ref_des in (('srs_G', 'srs_G'), ('predstrat_AE', 'predstrat_AE'), ('predstrat_AE', 'srs_G')):
    x = get(des, 'i300', 'F3b_l0.1_dw1', 0.5, 200)
    ref = get(ref_des, 'i300', 'retrain', 0.0, 200)
    j = diff(x, ref)
    if ref_des != des:
        j = j[j.unit.str.endswith('r0')]
    t = j.groupby('target')[['dba', 'doa']].mean().round(2).T
    L.append(f'F3b({des}) − 重训 {ref_des}\n' + t.to_string())
txt = '\n'.join(L)
open(os.path.join(OUT, 'frontier3_report.txt'), 'w').write(txt)
print(txt)
