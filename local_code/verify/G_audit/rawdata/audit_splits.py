"""划分核对：(a) PT/BE/SI s0/s1 与阶段 A 逐位比对；(b) 22 个划分的结构性质独立检查（缓冲、国家、去重、E_far）。"""
import os

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

RES = '/Users/perng_mini/AlphaCrops_local/results/guidance_G'
RESA = '/Users/perng_mini/AlphaCrops_local/results/stageA'
CACHE = '/Users/perng_mini/AlphaCrops_local/data/cache_guidance_G'
y = np.load(os.path.join(CACHE, 'y.npy'))
ctry = np.load(os.path.join(CACHE, 'ctry.npy'))
lon = np.load(os.path.join(CACHE, 'lon.npy'))
lat = np.load(os.path.join(CACHE, 'lat.npy'))
area = np.load(os.path.join(CACHE, 'area.npy'))
countries = ['AT', 'BE', 'DE', 'EE', 'LT', 'LV', 'PT', 'RO', 'SE', 'SI', 'SK']

print('== (a) 与阶段 A 逐位比对')
for tgt in ('PT', 'BE', 'SI'):
    for s in (0, 1):
        G = np.load(os.path.join(RES, 'stage0', f'{tgt}_s{s}.npz'))
        A = np.load(os.path.join(RESA, 'stage0', f'{tgt}_s{s}.npz'))
        res = {k: bool(np.array_equal(G[k], A[k])) for k in ('foreign', 'cand', 'test', 'allT', 'lat0')}
        print(tgt, s, res, 'n', len(G['foreign']), len(G['cand']), len(G['test']))
        # 候选池（阶段 A 的 pool 目录如有）
print(sorted(os.listdir(os.path.join(RESA, 'pool')))[:10])


def kmxy(lo, la, lat0):
    return np.c_[lo * 111.32 * np.cos(np.radians(lat0)), la * 110.57]


print('== (b) 结构检查')
rows = []
for tgt in countries:
    for s in (0, 1):
        G = np.load(os.path.join(RES, 'stage0', f'{tgt}_s{s}.npz'))
        cand, test, allT, foreign, efar = G['cand'], G['test'], G['allT'], G['foreign'], G['efar']
        lat0 = float(G['lat0'])
        xy_c = kmxy(lon[cand], lat[cand], lat0)
        xy_t = kmxy(lon[test], lat[test], lat0)
        dmin, _ = cKDTree(xy_t).query(xy_c)
        dfar, _ = cKDTree(xy_c).query(xy_t)
        efar2 = dfar >= np.quantile(dfar, 2 / 3)
        t_all = np.flatnonzero(ctry == tgt)
        dup = pd.DataFrame({'a': lon[allT], 'b': lat[allT], 'c': area[allT]}).duplicated().sum()
        fc = pd.Series(ctry[foreign]).value_counts()
        # 测试格网比例
        xy_all = kmxy(lon[allT], lat[allT], lat0)
        cellid = pd.Series(np.floor(xy_all[:, 0] / 10).astype(int)).astype(str) + '_' + \
            pd.Series(np.floor(xy_all[:, 1] / 10).astype(int)).astype(str)
        n_cells_test = cellid[len(cand):].nunique()
        overlap_cells = len(set(cellid[:len(cand)]) & set(cellid[len(cand):]))
        yT = y[test]
        big = np.flatnonzero(np.bincount(yT, minlength=10) >= 30)
        rows.append(dict(
            tgt=tgt, s=s, n_cand=len(cand), n_test=len(test), n_tgt_all=len(t_all),
            allT_eq=bool(np.array_equal(allT, np.r_[cand, test])),
            cand_test_disj=len(np.intersect1d(cand, test)) == 0,
            cand_ctry=bool((ctry[cand] == tgt).all()), test_ctry=bool((ctry[test] == tgt).all()),
            foreign_no_tgt=bool((ctry[foreign] != tgt).all()), foreign_n=len(foreign),
            foreign_uniq=len(np.unique(foreign)), foreign_per_min=int(fc.min()), foreign_per_max=int(fc.max()),
            foreign_nctry=len(fc), min_dist_cand_test_km=float(dmin.min()), dup_allT=int(dup),
            efar_eq=bool(np.array_equal(efar2, efar)), efar_frac=float(efar.mean()),
            dfar_eq=bool(np.allclose(G['d_far'], dfar.astype(np.float32))),
            n_cells_test=n_cells_test, overlap_cells=overlap_cells, n_big=len(big),
            big_far_lt30=int(sum((yT[efar] == c).sum() < 30 for c in big)),
            big_far_min=int(min((yT[efar] == c).sum() for c in big)),
            PT_ae_rows=G['P_T_ae'].shape[0] == len(allT), G_ae_rows=G['G_ae'].shape[0] == len(allT)))
T = pd.DataFrame(rows)
pd.set_option('display.width', 250)
print(T.to_string())
T.to_csv('/Users/perng_mini/AlphaCrops_local/verify/G_audit/rawdata/split_checks.csv', index=False)
