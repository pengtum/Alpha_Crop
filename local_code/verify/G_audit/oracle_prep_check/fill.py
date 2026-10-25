"""检查 oracle_compute 对"训练折中缺失的类"用 sc.min() 填充造成的折号列。"""
import numpy as np, os, sys
sys.path.insert(0, '.')
import guidance_G as G
from sklearn.model_selection import KFold
D = G.GData()
tot = 0
for tgt in D.countries:
    for split in (0, 1):
        S = dict(np.load(os.path.join(G.OUT, 'stage0', f'{tgt}_s{split}.npz')))
        o = np.load(G.oracle_path(tgt, split)); Gm = o['G'].astype(np.float64)
        pp = G.pool_positions(S, split, 0); y = D.y[S['allT'][pp]]
        folds = np.zeros(len(y), int)
        for f, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=0).split(Gm)): folds[te] = f
        cnt = np.bincount(y, minlength=10)
        # 某列在某折内为常数 → 该折训练集中缺该类，被填充
        filled = []
        for c in range(10):
            nconst = sum(np.ptp(Gm[folds == f, c]) == 0 for f in range(5))
            if nconst: filled.append((c, int(cnt[c]), int(nconst)))
        v = Gm.var(0); share_fill = sum(v[c] for c, n, nf in filled) / v.sum()
        vz = np.ones(10)  # z 后每列方差 1
        share_fill_z = sum(1 for _ in filled) / 10
        print(f'{tgt}_s{split} filled(col,n_pool,n_folds)={filled} var_share_filled_cols raw={share_fill:.3f} after_z={share_fill_z:.2f}')
