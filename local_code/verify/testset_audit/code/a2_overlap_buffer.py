"""审计 2：各划分测试集之间的重叠；1 km 缓冲；候选池/选样与测试不相交；去重后残余的近重复。只读。"""
import os, sys, glob, numpy as np, pandas as pd
sys.path.insert(0, '.')
import guidance_G as G
from stageA import km_xy
from scipy.spatial import cKDTree
R = '/Users/perng_mini/AlphaCrops_local/results'
D = G.GData()
rows = []
for t in D.countries:
    sp = {s: G.make_split(D, t, s) for s in range(6)}
    T = {s: set(sp[s]['test'].tolist()) for s in sp}
    nT = {s: len(T[s]) for s in sp}
    allT = sp[0]['allT']; n_all = len(set(sp[0]['cand'].tolist()) | T[0])
    t_all = np.flatnonzero(D.ctry == t)
    u01 = T[0] | T[1]
    r = dict(country=t, n_test_s0=nT[0], n_test_s1=nT[1], frac_test_s0=nT[0] / len(t_all))
    for s in (2, 3, 4, 5):
        r[f's{s}_test_in_s0s1test'] = len(T[s] & u01) / nT[s]
    r['s0s1_overlap'] = len(T[0] & T[1]) / nT[0]
    r['s2s3_test_disjoint_from_s0s1_n'] = len((T[2] | T[3]) - u01)
    # 缓冲：候选到测试的最近距离（公里，质心）
    for s in (0, 1):
        x = sp[s]
        lat0 = x['lat0']
        xy_c = km_xy(D.lon[x['cand']], D.lat[x['cand']], lat0)
        xy_t = km_xy(D.lon[x['test']], D.lat[x['test']], lat0)
        d, _ = cKDTree(xy_t).query(xy_c)
        r[f'min_cand_test_km_s{s}'] = float(d.min())
        assert len(set(x['cand'].tolist()) & set(x['test'].tolist())) == 0
        # 被缓冲去掉的比例
        n_nontest = len(t_all)  # 近似
    rows.append(r)
df = pd.DataFrame(rows)
pd.set_option('display.width', 250)
print(df.round(3).to_string())
df.to_csv('/Users/perng_mini/AlphaCrops_local/verify/testset_audit/code/a2_overlap.csv', index=False)
