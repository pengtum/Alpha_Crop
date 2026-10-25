"""审计 6 补充：可行性草算——从"从未当过 G/阶段 A 测试"的格子储备中切出两个互不重叠的新测试集。
只算大小与达到 30 个地块的类数，不训练、不看任何精度。只读。"""
import os, sys
LE = '/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops/label_expansion'
sys.path.insert(0, LE)
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
import guidance_G as G
from stageA import km_xy, CELL_KM, stable_hash

R = '/Users/perng_mini/AlphaCrops_local/results'
OUTD = '/Users/perng_mini/AlphaCrops_local/verify/testset_audit/empirical'
D = G.GData()
Y = np.asarray(D.y)
EARTH = 6371.0088
rows = []
for tgt in G.COUNTRIES_G:
    t_idx = np.flatnonzero(D.ctry == tgt)
    keep = ~pd.DataFrame({'a': D.lon[t_idx], 'b': D.lat[t_idx], 'c': D.area[t_idx]}).duplicated().to_numpy()
    t_idx = t_idx[keep]
    xy = km_xy(D.lon[t_idx], D.lat[t_idx], D.lat[t_idx].mean())
    cell = pd.factorize(pd.Series(np.floor(xy[:, 0] / CELL_KM).astype(int)).astype(str) + '_' +
                        pd.Series(np.floor(xy[:, 1] / CELL_KM).astype(int)).astype(str))[0]
    used = set()
    for s in (0, 1):
        used |= set(np.load(f'{R}/guidance_G/stage0/{tgt}_s{s}.npz')['test'].tolist())
    if tgt in ('PT', 'BE', 'SI'):
        used |= set(np.load(f'{R}/stageA/stage0/{tgt}_s2.npz')['test'].tolist())
    is_used = np.isin(t_idx, list(used))
    used_cells = set(cell[is_used])
    res = np.array(sorted(set(range(cell.max() + 1)) - used_cells))
    assert not (set(cell[is_used]) & set(res))
    rng = np.random.default_rng(stable_hash(f'fresh_reservoir_{tgt}') % 2 ** 32)
    res = rng.permutation(res)
    halves = {'F1': res[: len(res) // 2], 'F2': res[len(res) // 2:]}
    for h, hc in halves.items():
        is_test = np.isin(cell, hc)
        test = t_idx[is_test]
        ll_t = np.radians(np.c_[D.lat[test], D.lon[test]])
        d, _ = BallTree(ll_t, metric='haversine').query(np.radians(np.c_[D.lat[t_idx[~is_test]], D.lon[t_idx[~is_test]]]), k=1)
        cand = t_idx[~is_test][d[:, 0] * EARTH > 1.0]
        cnt = np.bincount(Y[test], minlength=10)
        rows.append(dict(tgt=tgt, half=h, n_cells_total=cell.max() + 1, n_res_cells=len(res), n_test_cells=len(hc),
                         frac_cells=len(hc) / (cell.max() + 1), n_test=len(test), n_cand=len(cand),
                         overlap_with_used_test=int(np.isin(test, list(used)).sum()), n_big_classes=int((cnt >= 30).sum())))
# 参照：s0 的 big 类数
for tgt in G.COUNTRIES_G:
    t = np.load(f'{R}/guidance_G/stage0/{tgt}_s0.npz')['test']
    rows.append(dict(tgt=tgt, half='ref_s0', n_test=len(t), n_big_classes=int((np.bincount(Y[t], minlength=10) >= 30).sum())))
df = pd.DataFrame(rows)
df.to_csv(f'{OUTD}/a5_fresh_reservoir.csv', index=False)
print(df.round(3).to_string())
