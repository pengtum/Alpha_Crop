"""审计 3：检验 2 / 检验 3 / 提升空间诊断的测试集是否一致；与 G 的 s0/s1 测试集的重叠；随机划分下候选与测试的近邻距离。只读。"""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, '.')
import guidance_G as G
from stageA import km_xy
from scipy.spatial import cKDTree
from sklearn.model_selection import train_test_split
R = '/Users/perng_mini/AlphaCrops_local/results'
D = G.GData()
for t in D.countries:
    c3 = np.load(f'{R}/check3/{t}_split.npz')
    test3, pool3 = c3['test'], c3['pool']
    msg = f'{t}: check3 test {len(test3)}'
    p2 = f'{R}/preds/{t}_test_idx.npy'
    if os.path.exists(p2):
        test2 = np.load(p2)
        msg += f' | check2 test == check3 test: {np.array_equal(np.sort(test2), np.sort(test3))} (order eq {np.array_equal(test2, test3)})'
    # 重算（与 setup_target 相同的步骤）
    t_idx = np.flatnonzero(D.ctry == t)
    keep = ~pd.DataFrame({'a': D.lon[t_idx], 'b': D.lat[t_idx], 'c': D.area[t_idx]}).duplicated().to_numpy()
    pool, test = train_test_split(t_idx[keep], test_size=0.3, random_state=42)
    msg += f' | recomputed eq {np.array_equal(test, test3)}'
    # 与 G s0/s1 测试的重叠
    for s in (0, 1):
        g = np.load(f'{R}/guidance_G/stage0/{t}_s{s}.npz')
        msg += f' | in G s{s} test {np.isin(test3, g["test"]).mean():.3f}'
    # 随机划分：测试地块到最近候选的距离
    lat0 = D.lat[t_idx].mean()
    d, _ = cKDTree(km_xy(D.lon[pool3], D.lat[pool3], lat0)).query(km_xy(D.lon[test3], D.lat[test3], lat0))
    msg += f' | median d(test→pool) {np.median(d)*1000:.0f} m, <100 m {(d < 0.1).mean():.3f}, ==0 {(d == 0).sum()}'
    sp = np.load(f'{R}/check3/{t}_split_sp.npz')
    msg += f' | test_sp {len(sp["test_sp"])}; test_sp∩G s0 test {np.isin(sp["test_sp"], np.load(f"{R}/guidance_G/stage0/{t}_s0.npz")["test"]).mean():.3f}'
    print(msg, flush=True)
