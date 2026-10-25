"""(2) analyze_ML.load_all 的 LightGBM 行：θ_E1（ba_ge30）是否复现 G 的 −0.054 [−0.244, +0.136]。ML_OUT 指向空目录，不读任何新学习器的行。"""
import os, sys
os.environ['ML_OUT'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'empty_ml_out')
L = sys.argv[1]; sys.path.insert(0, L); os.chdir(L)
import numpy as np, pandas as pd
import analyze_ML as AM
import analyze_G as A
from guidance_G import E1
assert AM.OUT == os.environ['ML_OUT'], AM.OUT
R = AM.load_all()
print('learners', R.learner.value_counts().to_dict(), 'rows', len(R))
lg = R[R.learner == 'lgb']
key = A.UNIT + ['cond', 'method', 'clf', 'k']
print('重复键：', int(lg.duplicated(key).sum()))
print('lgb 行按 (cond, method) 计数：'); print(lg.groupby(['cond', 'method']).size().to_string())
print('k 分布：', lg.k.value_counts().sort_index().to_dict(), '；clf：', lg.clf.value_counts().to_dict())
print('每单元 AE 行数：', lg[lg.cond == 'AE'].groupby(A.UNIT).size().value_counts().to_dict())
for met in ('ba_ge30', 'oa'):
    d = A.deltas(lg, 'AE', 'S2', E1, met)
    v = A.country_values(d)
    s = A.t_summary(v)
    print(met, 'n_deltas', len(d), A.fmt(s), '→', A.verdict(s) if met == 'ba_ge30' else '', 'TOST p', round(AM.tost_p(s), 4))
# 与 analyze_G 在全部 G 行上的直接计算对照
RG = A.load_rows()
s0 = A.t_summary(A.country_values(A.deltas(RG, 'AE', 'S2', E1)))
print('analyze_G 全部行直接计算：', A.fmt(s0))
# 与 report_G.txt 对照
rep = open(os.path.join(A.OUT, 'report_G.txt')).read()
i = rep.find('θ_E1 =')
print('report_G.txt:', rep[i:i + 160].splitlines()[0])
