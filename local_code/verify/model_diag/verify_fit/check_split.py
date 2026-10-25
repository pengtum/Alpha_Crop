"""泄漏核查：目标候选池（上限实验训练集）与测试集是否重叠 / 距离 / 特征完全相同的行。只读。"""
import os, sys
PROJ = '/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops/label_expansion'
sys.path.insert(0, PROJ); os.chdir(PROJ)
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
import guidance_G as G
from stageA import km_xy
GRES = '/Users/perng_mini/AlphaCrops_local/results/guidance_G'
D = G.GData()
rows = []
for tgt in D.countries:
    for split in (0, 1):
        S = np.load(os.path.join(GRES, 'stage0', f'{tgt}_s{split}.npz'))
        fr, test, allT, cand = S['foreign'], S['test'], S['allT'], S['cand']
        pp = G.pool_positions(S, split, 0)
        pool = allT[pp]
        lat0 = float(S['lat0'])
        xy_p = km_xy(D.lon[pool], D.lat[pool], lat0); xy_t = km_xy(D.lon[test], D.lat[test], lat0)
        dmin, _ = cKDTree(xy_t).query(xy_p)
        # 特征完全相同（AE 128 列）的池-测试对
        def dup(feat):
            A = np.ascontiguousarray(np.asarray(D.X[feat][pool]))
            B = np.ascontiguousarray(np.asarray(D.X[feat][test]))
            ha = set(map(bytes, A.view(np.uint8).reshape(len(A), -1)))
            return int(sum(bytes(b) in ha for b in B.view(np.uint8).reshape(len(B), -1)))
        rows.append(dict(tgt=tgt, split=split, n_pool=len(pool), n_test=len(test), n_cand=len(cand),
                         pool_in_test=int(np.isin(pool, test).sum()), pool_in_cand=int(np.isin(pool, cand).sum()),
                         foreign_in_target=int((D.ctry[fr] == tgt).sum()), min_dist_km=float(dmin.min()),
                         p1_dist_km=float(np.quantile(dmin, 0.01)), dupfeat_ae=dup('ae'), dupfeat_s2=dup('s2'),
                         test_ctry_ok=bool((D.ctry[test] == tgt).all())))
df = pd.DataFrame(rows)
out = '/Users/perng_mini/AlphaCrops_local/verify/model_diag/verify_fit/split_leak_check.csv'
df.to_csv(out, index=False)
print(df.to_string())
