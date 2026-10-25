"""审计 3–5：rep0/rep1 的测试集与候选池；make_split 与标签无关；隔离（缓冲、外国池、重复）。只读。"""
import os, sys, glob, re, hashlib
LE = '/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops/label_expansion'
sys.path.insert(0, LE)
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.neighbors import BallTree
import guidance_G as G
from stageA import km_xy, BUFFER_KM

R = '/Users/perng_mini/AlphaCrops_local/results'
OUTD = '/Users/perng_mini/AlphaCrops_local/verify/testset_audit/empirical'
D = G.GData()
EARTH_KM = 6371.0088


class Guard:
    """只暴露 make_split 需要的字段；访问 y 或 X 就报错，证明划分不读标签与特征。"""
    def __init__(self, D, y_perm):
        self.ctry, self.lon, self.lat, self.area, self.countries = D.ctry, D.lon, D.lat, D.area, D.countries
        self._y = y_perm

    def __getattr__(self, k):
        raise AttributeError(f'make_split 访问了 {k}')


def hav_min(src_ll, dst_ll):
    """src 每点到 dst 最近点的大圆距离（km）。"""
    t = BallTree(np.radians(dst_ll[:, ::-1]), metric='haversine')
    d, _ = t.query(np.radians(src_ll[:, ::-1]), k=1)
    return d[:, 0] * EARTH_KM


def rowhash(M):
    M = np.ascontiguousarray(M)
    return np.array([hashlib.md5(r.tobytes()).hexdigest()[:16] for r in M])


rows = []
rng = np.random.default_rng(12345)
y_perm = rng.permutation(np.asarray(D.y))
for tgt in G.COUNTRIES_G:
    for s in (0, 1):
        Z = np.load(f'{R}/guidance_G/stage0/{tgt}_s{s}.npz')
        S = {k: Z[k] for k in ('test', 'cand', 'foreign', 'allT', 'lat0')}
        test, cand, foreign, allT = S['test'], S['cand'], S['foreign'], S['allT']
        r = dict(tgt=tgt, split=s, n_test=len(test), n_cand=len(cand))
        # (4) 标签/特征无关：Guard 对象
        try:
            sp = G.make_split(Guard(D, y_perm), tgt, s)
            r['label_free'] = bool(np.array_equal(sp['test'], test) and np.array_equal(sp['cand'], cand)
                                   and np.array_equal(sp['foreign'], foreign))
        except AttributeError as e:
            r['label_free'] = f'FAIL {e}'
        # (3) 候选池
        pp0, pp1 = G.pool_positions(S, s, 0), G.pool_positions(S, s, 1)
        P0, P1 = set(allT[pp0]), set(allT[pp1])
        r.update(pool0=len(P0), pool1=len(P1), pools_identical=P0 == P1,
                 pool_jacc=len(P0 & P1) / len(P0 | P1), pool_in_cand=bool(pp0.max() < len(cand) and pp1.max() < len(cand)))
        tset = set(test)
        # 选样文件：都在对应 rep 的池内且不含测试地块
        nsel, bad_pool, bad_test = 0, 0, 0
        for f in glob.glob(f'{R}/guidance_G/sel/{tgt}_s{s}_r*_*.npz'):
            rep = int(re.search(r'_r(\d)_', os.path.basename(f)).group(1))
            z = np.load(f)
            sel = z['sel']
            nsel += 1
            P = P0 if rep == 0 else P1
            bad_pool += int(not set(sel) <= P)
            bad_test += int(bool(set(sel) & tset))
            if 'sel_pos' in z.files:
                bad_pool += int(not np.array_equal(allT[(pp0 if rep == 0 else pp1)[z['sel_pos']]], sel))
        r.update(n_sel_files=nsel, sel_not_in_pool=bad_pool, sel_hits_test=bad_test)
        # ORACLE 只在候选池（rep0）上拟合
        op = f'{R}/guidance_G/oracle/{tgt}_s{s}.npz'
        if os.path.exists(op):
            oz = np.load(op)
            r['oracle_rows_eq_pool0'] = bool(oz['P_T'].shape[0] == len(pp0) and oz['G'].shape[0] == len(pp0))
        # (5) 隔离
        lat0 = float(S['lat0'])
        xyT = km_xy(D.lon[test], D.lat[test], lat0)
        xyC = km_xy(D.lon[cand], D.lat[cand], lat0)
        d_proj, _ = cKDTree(xyT).query(xyC)
        llT = np.c_[D.lon[test], D.lat[test]]
        d_hav = hav_min(np.c_[D.lon[cand], D.lat[cand]], llT)
        r.update(min_d_proj_cand_test=float(d_proj.min()), min_d_hav_cand_test=float(d_hav.min()),
                 n_cand_hav_le_1km=int((d_hav <= 1.0).sum()), n_cand_hav_le_0p9=int((d_hav <= 0.9).sum()),
                 lat_min=float(D.lat[np.r_[test, cand]].min()), lat_max=float(D.lat[np.r_[test, cand]].max()), lat0=lat0)
        # 地块外缘：质心距离减去两块的等面积圆半径
        rad = lambda idx: np.sqrt(np.asarray(D.area[idx], float) * 1e4 / np.pi) / 1000.0
        t_i = BallTree(np.radians(llT[:, ::-1]), metric='haversine')
        dd, ii = t_i.query(np.radians(np.c_[D.lat[cand], D.lon[cand]]), k=1)
        edge = dd[:, 0] * EARTH_KM - rad(cand) - rad(test[ii[:, 0]])
        r['n_cand_edge_lt_0p5km'] = int((edge < 0.5).sum())
        r['min_edge_km'] = float(edge.min())
        # 外国池
        r['foreign_has_tgt'] = int((D.ctry[foreign] == tgt).sum())
        r['foreign_unique'] = bool(len(np.unique(foreign)) == len(foreign))
        d_fh = hav_min(np.c_[D.lon[foreign], D.lat[foreign]], llT)
        r['n_foreign_within_1km_of_test'] = int((d_fh <= 1.0).sum())
        r['min_d_foreign_test_km'] = float(d_fh.min())
        # 重复：坐标、坐标+面积、特征向量
        key = lambda idx: pd.Series(np.round(D.lon[idx], 7).astype(str)) + '_' + pd.Series(np.round(D.lat[idx], 7).astype(str))
        kT = set(key(test))
        train_idx = np.unique(np.r_[foreign, allT[pp0], allT[pp1]])
        r['coord_dup_train_test'] = len(kT & set(key(train_idx)))
        r['coord_dup_cand_test'] = len(kT & set(key(cand)))
        for f in ('ae', 's2'):
            hT = set(rowhash(np.asarray(D.X[f][test])))
            hTr = set(rowhash(np.asarray(D.X[f][train_idx])))
            r[f'featdup_{f}_train_test'] = len(hT & hTr)
        r['n_test_ids_unique'] = bool(len(np.unique(test)) == len(test))
        r['test_cand_disjoint'] = not bool(set(test) & set(cand))
        rows.append(r)
        print(r, flush=True)

df = pd.DataFrame(rows)
df.to_csv(f'{OUTD}/a3_pools_isolation.csv', index=False)
print(df.to_string())

# 阶段 A 保存的候选池与 pool_positions 一致；阶段 A 的 sel 不含测试地块
ra = []
for f in sorted(glob.glob(f'{R}/stageA/pool/*.npy')):
    tgt, s, rep = re.match(r'([A-Z]+)_s(\d)_r(\d)\.npy', os.path.basename(f)).groups()
    s, rep = int(s), int(rep)
    zA = np.load(f'{R}/stageA/stage0/{tgt}_s{s}.npz')
    S = dict(cand=zA['cand'], allT=zA['allT'])
    pool = np.load(f)
    same = np.array_equal(pool, zA['allT'][G.pool_positions(S, s, rep)])
    tset = set(zA['test'])
    hits = 0
    for g in glob.glob(f'{R}/stageA/sel/{tgt}_s{s}_r{rep}_*.npz'):
        hits += int(bool(set(np.load(g)['sel']) & tset))
    ra.append(dict(tgt=tgt, split=s, rep=rep, pool_matches_G_pool_positions=same, n_pool=len(pool), sel_hits_test=hits))
print(pd.DataFrame(ra).to_string())
pd.DataFrame(ra).to_csv(f'{OUTD}/a3_stageA_pools.csv', index=False)
