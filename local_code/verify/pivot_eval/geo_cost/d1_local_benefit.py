"""
D1：目标国标签的"局部效应"——测试地块附近（R km 内）有同类已标注地块时，是否比同一选样下同类的其他测试地块改善更多。
只读 G 的 stage0 / sel / probs / rows 文件；不训练。线程 ≤2。
识别：在每个 (单元, 分类器, 真实类 c) 块内，对 correct[s, i] 做双向去均值（选样 s × 地块 i），
再回归到 near[s, i] = 1[选样 s 中存在类 c 的标签距地块 i ≤ R km]。选样×类 固定效应吸收"这批标签整体提高了类 c 召回"，
地块固定效应吸收地块本身难度与位置，剩下的是纯局部（空间邻近）效应。按 1/n_c 加权（与 BA 同口径）。
"""
import os, glob, sys
os.environ.setdefault('OMP_NUM_THREADS', '2')
import numpy as np, pandas as pd
from scipy.spatial import cKDTree

G = '/Users/perng_mini/AlphaCrops_local/results/guidance_G'
C = '/Users/perng_mini/AlphaCrops_local/data/cache_guidance_G'
y = np.load(f'{C}/y.npy'); lon = np.load(f'{C}/lon.npy'); lat = np.load(f'{C}/lat.npy')
RS = [1, 2, 5, 10, 20]

def km_xy(lo, la, lat0):
    return np.c_[lo * 111.32 * np.cos(np.radians(lat0)), la * 110.57]

out = []
for f0 in sorted(glob.glob(f'{G}/stage0/*.npz')):
    tgt, sp = os.path.basename(f0)[:-4].split('_s')
    S = np.load(f0)
    test = S['test']; lat0 = float(S['lat0'])
    xyT = km_xy(lon[test], lat[test], lat0); yT = y[test]
    big = np.flatnonzero(np.bincount(yT, minlength=10) >= 30)
    for rep in (0, 1):
        unit = f'{tgt}_s{sp}_r{rep}'
        rows = pd.concat([pd.read_csv(p) for p in glob.glob(f'{G}/rows_t*_{unit}.csv')])
        rows = rows[(rows.k > 0) & rows.clf.isin(['ae', 's2'])]
        for clf in ('ae', 's2'):
            rr = rows[rows.clf == clf].drop_duplicates('sel_hash')
            corr, sels, ks = [], [], []
            P0 = S[f'P0test_{clf}_i300'].astype(np.float32)
            corr.append(P0.argmax(1) == yT); sels.append(np.array([], int)); ks.append(0)
            for _, r in rr.iterrows():
                pp = f'{G}/probs/{unit}_{clf}_k{int(r.k)}_{r.sel_hash}.npz'
                sfile = f'{G}/sel/{unit}_{r.cond}_{r.method}_k{int(r.k)}.npz'
                if not (os.path.exists(pp) and os.path.exists(sfile)):
                    continue
                corr.append(np.load(pp)['i300'].argmax(1) == yT)
                sels.append(np.load(sfile)['sel']); ks.append(int(r.k))
            Cm = np.array(corr, dtype=np.float64)  # (n_s, n_test)
            nS = len(sels)
            for R in RS:
                num = den = 0.0; share_near = []
                num_any = den_any = 0.0
                for c in big:
                    ic = np.flatnonzero(yT == c)
                    near = np.zeros((nS, len(ic)))
                    near_any = np.zeros((nS, len(ic)))
                    for s, sel in enumerate(sels):
                        if len(sel) == 0:
                            continue
                        xs = km_xy(lon[sel], lat[sel], lat0)
                        m = y[sel] == c
                        if m.any():
                            d, _ = cKDTree(xs[m]).query(xyT[ic], k=1)
                            near[s] = d <= R
                        d2, _ = cKDTree(xs).query(xyT[ic], k=1)
                        near_any[s] = d2 <= R
                    Y = Cm[:, ic]
                    def dm(A):
                        return A - A.mean(0, keepdims=True) - A.mean(1, keepdims=True) + A.mean()
                    Yd, Nd, Ad = dm(Y), dm(near), dm(near_any)
                    w = 1.0 / len(ic)
                    num += w * (Nd * Yd).sum(); den += w * (Nd * Nd).sum()
                    # 对 near_any 另做一次（控制同类邻近之外的"任意标签邻近"）
                    num_any += w * (Ad * Yd).sum(); den_any += w * (Ad * Ad).sum()
                    share_near.append(near[np.array(ks) == 200].mean() if (np.array(ks) == 200).any() else np.nan)
                out.append(dict(target=tgt, split=int(sp), rep=rep, clf=clf, R=R, n_sel=nS,
                                gamma_same=100 * num / den if den > 0 else np.nan,
                                gamma_any=100 * num_any / den_any if den_any > 0 else np.nan,
                                var_near_same=den, share_near_same_k200=float(np.nanmean(share_near))))
            print(unit, clf, nS, [round(o['gamma_same'], 2) for o in out[-len(RS):]],
                  [round(o['share_near_same_k200'], 3) for o in out[-len(RS):]], flush=True)
pd.DataFrame(out).to_csv('/Users/perng_mini/AlphaCrops_local/verify/pivot_eval/geo_cost/d1_local_benefit.csv', index=False)
