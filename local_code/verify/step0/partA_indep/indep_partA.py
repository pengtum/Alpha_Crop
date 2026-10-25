"""第 0 步 A 部分的独立复核实现（只依据 STEP0_SPEC_2026-09-29.md 的 A 部分；未读取 verify/step0/partA/）。

范围（按复核任务）：
  S0 ∈ {clus, rand, none} × n0 ∈ {100, 500, 2000}（none 只有 n0 = 0）；
  补采：口径 1 的 SRS、EXT-p（k = 200）；口径 2（B = 20，每簇 10 个）的 EXT、SPR（c2 = 2）、SPR=（c2 = 1）；
  估计量：E_S0、E_pool、E_S1、E_none、E_oracle；分类器特征 f ∈ {ae, s2}；R = 20。
不训练任何分类器；只在 stage0 的冻结概率 P_T_{f} 测试行上做先验重加权。
候选标签只用来模拟"采到的标签"，测试标签只用于评估（π_test 只进 E_oracle 与先验 L1）。

用法：python -B indep_partA.py [--centroid mass|square] [--out 目录]
"""
import os
import sys

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_v] = '2'
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import hashlib  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

L = '/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops/label_expansion'
sys.path.insert(0, L)
import guidance_G as G  # noqa: E402
from stageA import km_xy, class_weights_from, MIN_N  # noqa: E402
from check3_priors import reweight, laplace_prior  # noqa: E402

R_G = '/Users/perng_mini/AlphaCrops_local/results/guidance_G'
COUNTRIES = os.environ.get('INDEP_COUNTRIES', 'AT,BE,DE,EE,LT,LV,PT,RO,SE,SI,SK').split(',')
SPLITS = (0, 1)
C = 10
R = 20
N0_M = {100: 10, 500: 20, 2000: 25}
K = 200                  # 口径 1 标签预算
B_COST, M_B = 20, 10     # 口径 2：成本预算与每簇地块数
N_ZONE = 8
FEATS = ('ae', 's2')


def seed_of(tgt, split, scen, n0, r):
    """规格 A1：md5(f'step0A_{国家}_{划分}_{场景}_{n0}_{r}') 的前 8 位十六进制。场景 = S0 类型。"""
    return int(hashlib.md5(f'step0A_{tgt}_{split}_{scen}_{n0}_{r}'.encode()).hexdigest()[:8], 16)


def es_order(rng, w):
    """Efraimidis–Spirakis 加权随机排序：键 = u^(1/w)，按键降序；前 m 个即 PPS 不放回（逐次抽样）的 m 个。
    用 log(u)/w 避免下溢；w 必须 > 0。"""
    w = np.asarray(w, dtype=np.float64)
    assert np.all(w > 0)
    u = rng.random(len(w))
    key = np.log(u) / w
    return np.argsort(-key, kind='stable')


def score(yT, ph, big, nT_c):
    """测试集 OA 与 ba_ge30（百分数）。big：测试集中 ≥ MIN_N 的类。"""
    ok = ph == yT
    nc = np.bincount(yT[ok], minlength=C)
    return 100.0 * ok.mean(), 100.0 * float(np.mean(nc[big] / nT_c[big]))


class Unit:
    """一个（国家, 划分）单元：抽样框、格、调查区、底座概率与评估所需的常量。"""

    def __init__(self, D, tgt, split, centroid='mass'):
        from sklearn.cluster import KMeans
        S = np.load(os.path.join(R_G, 'stage0', f'{tgt}_s{split}.npz'))
        self.tgt, self.split = tgt, split
        cand, test, foreign = S['cand'], S['test'], S['foreign']
        n_cand = len(cand)
        assert np.array_equal(S['allT'][:n_cand], cand) and np.array_equal(S['allT'][n_cand:], test)
        self.n = n_cand
        self.yC = np.asarray(D.y[cand]).astype(np.int64)      # 候选标签：只用于模拟"采到的标签"
        self.yT = np.asarray(D.y[test]).astype(np.int64)      # 测试标签：只用于评估
        self.nT_c = np.bincount(self.yT, minlength=C)
        self.big = np.flatnonzero(self.nT_c >= MIN_N)
        self.pi_test = self.nT_c / len(self.yT)
        self.pi_frame = np.bincount(self.yC, minlength=C) / n_cand
        yF = np.asarray(D.y[foreign])
        cw = class_weights_from(yF)
        self.pi_train = np.bincount(yF, weights=cw[yF], minlength=C) / cw[yF].sum()   # = pi_train_of([], 0)
        self.P = {f: S[f'P_T_{f}'][n_cand:] for f in FEATS}                             # 温度校准，float32
        self.P0 = {f: S[f'P0test_{f}_i300'].astype(np.float32) for f in FEATS}          # 对账用
        self.S = S
        # ---- 格（10 km，与 make_split 相同的 km_xy 与 lat0）
        xy = km_xy(np.asarray(D.lon[cand]), np.asarray(D.lat[cand]), float(S['lat0']))
        ij = np.floor(xy / G.CELL_KM).astype(np.int64)
        uij, cell = np.unique(ij, axis=0, return_inverse=True)
        cell = cell.ravel()
        self.cell = cell
        self.cnt = np.bincount(cell)
        nc = len(self.cnt)
        order = np.argsort(cell, kind='stable')
        bounds = np.r_[0, np.cumsum(self.cnt)]
        self.members = [order[bounds[c]:bounds[c + 1]] for c in range(nc)]
        if centroid == 'mass':      # 格内候选地块坐标均值
            cen = np.c_[np.bincount(cell, xy[:, 0]), np.bincount(cell, xy[:, 1])] / self.cnt[:, None]
        else:                       # 10 km 方格的几何中心
            cen = (uij + 0.5) * G.CELL_KM
        fitm = self.cnt >= 10
        km = KMeans(n_clusters=N_ZONE, n_init=10, random_state=0).fit(cen[fitm])
        zc = np.empty(nc, np.int64)
        zc[fitm] = km.labels_
        if (~fitm).any():
            d2 = ((cen[~fitm, None, :] - km.cluster_centers_[None]) ** 2).sum(-1)
            zc[~fitm] = d2.argmin(1)
        # 核对：拟合格的 labels_ 与最近中心一致
        d2f = ((cen[fitm, None, :] - km.cluster_centers_[None]) ** 2).sum(-1)
        self.km_label_consistent = float((d2f.argmin(1) == km.labels_).mean())
        self.zone_cell = zc
        self.zone_par = zc[cell]
        self.zone_size = np.bincount(self.zone_par, minlength=N_ZONE)
        self.n_cells = nc

    # -------------------------------------------------------------- 先验修正与评估
    def eval_prior(self, f, pi_new, base='P'):
        P = (self.P if base == 'P' else self.P0)[f]
        q, _ = reweight(P, pi_new, self.pi_train)
        return score(self.yT, q.argmax(1), self.big, self.nT_c)

    def eval_none(self, f, base='P'):
        P = (self.P if base == 'P' else self.P0)[f]
        return score(self.yT, P.argmax(1), self.big, self.nT_c)


def draw_S0(U, rng, scen, n0):
    """返回 (S0 位置, 所选调查区, 所用格)。"""
    if scen == 'none':
        return np.array([], np.int64), np.array([], np.int64), np.array([], np.int64)
    if scen == 'rand':
        return np.sort(rng.choice(U.n, n0, replace=False)), np.array([], np.int64), np.array([], np.int64)
    M = N0_M[n0]
    ncl = n0 // M
    elig = U.cnt >= M
    zones = np.flatnonzero(U.zone_size > 0)
    zord = zones[es_order(rng, U.zone_size[zones])]
    m = 2
    while True:
        selz = zord[:m]
        n_el = int((elig & np.isin(U.zone_cell, selz)).sum())
        if n_el >= ncl + 20 or m >= len(zord):
            break
        m += 1
    cells_ok = np.flatnonzero(elig & np.isin(U.zone_cell, selz))
    assert len(cells_ok) >= ncl, (U.tgt, U.split, n0, len(cells_ok))
    cells = cells_ok[es_order(rng, U.cnt[cells_ok])[:ncl]]
    S0 = np.concatenate([rng.choice(U.members[c], M, replace=False) for c in cells])
    return S0, np.sort(selz), cells


def draw_S1(U, rng, scen, S0, selz, cells0):
    """返回 {臂: (S1 位置, 所得标签数)}。消耗随机数的顺序：SRS → EXT-p → EXT → SPR/SPR= 排序 → 簇内抽样。"""
    out = {}
    in_S0 = np.zeros(U.n, bool)
    in_S0[S0] = True
    # 口径 1：SRS（所有 S0）
    out['SRS'] = rng.choice(np.flatnonzero(~in_S0), K, replace=False)
    if scen != 'clus':
        return out
    # 口径 1：EXT-p（S0 所在区 = S0 抽样时所选的调查区）
    inz = np.isin(U.zone_par, selz)
    out['EXT-p'] = rng.choice(np.flatnonzero(inz & ~in_S0), K, replace=False)
    # 口径 2：可用簇 = 候选地块数 ≥ 10 且不是 S0 所用格
    avail = U.cnt >= M_B
    avail[cells0] = False
    zin_cell = np.isin(U.zone_cell, selz)
    # EXT：区内按地块数 PPS 不放回抽 B_COST 个簇（c1 = 1）
    a_in = np.flatnonzero(avail & zin_cell)
    assert len(a_in) >= B_COST, (U.tgt, U.split, len(a_in))
    ext_cells = a_in[es_order(rng, U.cnt[a_in])[:B_COST]]
    out['EXT'] = np.concatenate([rng.choice(U.members[c], M_B, replace=False) for c in ext_cells])
    # SPR / SPR=：全国可用簇一次 PPS 加权随机排序
    a_all = np.flatnonzero(avail)
    ordc = a_all[es_order(rng, U.cnt[a_all])]
    first = ordc[:B_COST]
    parts = [rng.choice(U.members[c], M_B, replace=False) for c in first]   # 簇内抽样只做一次，SPR 取前缀
    out['SPR='] = np.concatenate(parts)
    rem, n_take = B_COST, 0
    for c in ordc:
        cost = 1 if zin_cell[c] else 2
        if cost > rem:
            break
        rem -= cost
        n_take += 1
    assert n_take <= B_COST
    out['SPR'] = np.concatenate(parts[:n_take]) if n_take else np.array([], np.int64)
    out['_meta'] = dict(spr_clusters=n_take, spr_in=int(zin_cell[first[:n_take]].sum()),
                        sprq_in=int(zin_cell[first].sum()), ext_cells=ext_cells, first=first)
    return out


def check_draw(U, scen, n0, S0, selz, cells0, S1):
    """逐次抽样的结构核对（出错即断言失败）。"""
    if scen == 'clus':
        M = N0_M[n0]
        assert len(S0) == n0 and len(np.unique(S0)) == n0
        assert len(cells0) == n0 // M and np.all(U.cnt[cells0] >= M)
        assert set(np.unique(U.cell[S0])) == set(cells0)
        assert np.all(np.bincount(U.cell[S0], minlength=U.n_cells)[cells0] == M)
        assert np.all(np.isin(U.zone_cell[cells0], selz))
        assert int(((U.cnt >= M) & np.isin(U.zone_cell, selz)).sum()) >= n0 // M + 20 or len(selz) == N_ZONE
    elif scen == 'rand':
        assert len(S0) == n0 and len(np.unique(S0)) == n0
    for arm, s in S1.items():
        if arm == '_meta':
            continue
        assert len(np.unique(s)) == len(s)
        assert not np.isin(s, S0).any(), arm
        if arm in ('SRS', 'EXT-p', 'EXT', 'SPR='):
            assert len(s) == K, (arm, len(s))
        if arm == 'EXT-p':
            assert np.all(np.isin(U.zone_par[s], selz))
        if arm in ('EXT', 'SPR', 'SPR='):
            cs = np.unique(U.cell[s])
            assert not np.isin(cs, cells0).any() and np.all(U.cnt[cs] >= M_B)
            assert np.all(np.bincount(U.cell[s], minlength=U.n_cells)[cs] == M_B)
        if arm == 'EXT':
            assert np.all(np.isin(U.zone_par[s], selz))
    if 'SPR' in S1:
        m = S1['_meta']
        assert np.array_equal(S1['SPR'], S1['SPR='][:len(S1['SPR'])])
        cost = m['spr_in'] + 2 * (m['spr_clusters'] - m['spr_in'])
        assert cost <= B_COST
        if m['spr_clusters'] < B_COST:    # 停止时下一个簇付不起
            nxt = m['first'][m['spr_clusters']] if m['spr_clusters'] < len(m['first']) else None
            if nxt is not None:
                c_next = 1 if np.isin(U.zone_cell[nxt], selz) else 2
                assert cost + c_next > B_COST


def run(centroid, out_dir, log=print):
    D = G.GData()
    rows, units_meta, recon = [], [], []
    t0 = time.time()
    for tgt in COUNTRIES:
        for split in SPLITS:
            U = Unit(D, tgt, split, centroid)
            base = dict(target=tgt, split=split)
            # ---- 参照：E_none、E_oracle（与抽样无关）
            for f in FEATS:
                oa, ba = U.eval_none(f)
                rows.append(dict(base, feat=f, scen='ref', n0=0, r=-1, arm='-', est='E_none', oa=oa, ba_ge30=ba,
                                 prior_L1=float(np.abs(U.pi_train - U.pi_test).sum())))
                oa, ba = U.eval_prior(f, U.pi_test)
                rows.append(dict(base, feat=f, scen='ref', n0=0, r=-1, arm='-', est='E_oracle', oa=oa, ba_ge30=ba,
                                 prior_L1=0.0))
                # ---- 对账：P0test_i300 底座上的 E_none 与 G foreign_only 行；另用 G 的池先验复现 oa_P2
                for rep in (0, 1):
                    gp = os.path.join(R_G, f'rows_t1_{tgt}_s{split}_r{rep}.csv')
                    g = pd.read_csv(gp)
                    gr = g[(g.method == 'foreign_only') & (g.clf == f)].iloc[0]
                    P0 = U.P0[f].astype(np.float64)
                    P0 /= P0.sum(1, keepdims=True)
                    oa0, ba0 = score(U.yT, P0.argmax(1), U.big, U.nT_c)
                    pool_pos = G.pool_positions(U.S, split, rep)
                    pi_pool = np.bincount(np.asarray(D.y[U.S['allT'][pool_pos]]), minlength=C) / len(pool_pos)
                    q, _ = reweight(P0, pi_pool, U.pi_train)
                    oaP2, baP2 = score(U.yT, q.argmax(1), U.big, U.nT_c)
                    q, _ = reweight(P0, U.pi_test, U.pi_train)
                    oaO, baO = score(U.yT, q.argmax(1), U.big, U.nT_c)
                    # 与 G._scores 交叉核对评估函数
                    oa_g, ba_g, _, _ = G._scores(U.yT, P0.argmax(1), U.big)
                    recon.append(dict(base, rep=rep, feat=f, oa_none_P0=oa0, G_oa=gr.oa, ba_none_P0=ba0,
                                      G_ba=gr.ba_ge30, oa_poolprior_P0=oaP2, G_oa_P2=gr.oa_P2, ba_poolprior_P0=baP2,
                                      G_ba_P2=gr.ba_P2, oa_oracle_P0=oaO, ba_oracle_P0=baO,
                                      d_scores=max(abs(oa_g - oa0), abs(ba_g - ba0)),
                                      L1_pool_test=float(np.abs(pi_pool - U.pi_test).sum())))
            # ---- 抽样
            nz_used, empty_zone = [], []
            for scen, n0s in (('clus', (100, 500, 2000)), ('rand', (100, 500, 2000)), ('none', (0,))):
                for n0 in n0s:
                    for r in range(R):
                        rng = np.random.default_rng(seed_of(tgt, split, scen, n0, r))
                        S0, selz, cells0 = draw_S0(U, rng, scen, n0)
                        S1 = draw_S1(U, rng, scen, S0, selz, cells0)
                        check_draw(U, scen, n0, S0, selz, cells0, S1)
                        bias = float(np.abs(np.bincount(U.yC[S0], minlength=C) / n0 - U.pi_frame).sum()) \
                            if n0 else np.nan
                        n_z = len(selz) if scen == 'clus' else np.nan
                        if scen == 'clus':
                            nz_used.append(n_z)
                            empty_zone.append(int(len(set(selz) - set(U.zone_par[S0]))))
                        meta = S1.pop('_meta', None)
                        common = dict(base, scen=scen, n0=n0, r=r, bias_S0=bias, n_zones=n_z,
                                      spr_clusters=meta['spr_clusters'] if meta else np.nan,
                                      spr_in=meta['spr_in'] if meta else np.nan,
                                      sprq_in=meta['sprq_in'] if meta else np.nan)
                        ests = []
                        if n0:
                            ests.append(('-', 'E_S0', U.yC[S0]))
                        for arm, s1 in S1.items():
                            ests.append((arm, 'E_pool', U.yC[np.r_[S0, s1]]))
                            ests.append((arm, 'E_S1', U.yC[s1]))
                        for arm, est, ylab in ests:
                            pi_hat = laplace_prior(ylab, C, alpha=1)
                            L1 = float(np.abs(pi_hat - U.pi_test).sum())
                            for f in FEATS:
                                oa, ba = U.eval_prior(f, pi_hat)
                                rows.append(dict(common, feat=f, arm=arm, est=est, oa=oa, ba_ge30=ba, prior_L1=L1,
                                                 n_labels=len(ylab)))
            units_meta.append(dict(base, n_cand=U.n, n_test=len(U.yT), n_cells=U.n_cells,
                                   n_cells_ge10=int((U.cnt >= 10).sum()), zone_sizes=U.zone_size.tolist(),
                                   km_label_consistent=U.km_label_consistent,
                                   zones_used_mean=float(np.mean(nz_used)),
                                   zones_used_dist={int(k): int(v) for k, v in zip(*np.unique(nz_used, return_counts=True))},
                                   draws_with_empty_selected_zone=int(np.sum(np.array(empty_zone) > 0))))
            log(f'{tgt} s{split} 完成，累计 {time.time() - t0:.0f}s，行数 {len(rows):,}')
    df = pd.DataFrame(rows)
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, 'indep_rows.csv'), index=False)
    pd.DataFrame(units_meta).to_csv(os.path.join(out_dir, 'indep_units.csv'), index=False)
    pd.DataFrame(recon).to_csv(os.path.join(out_dir, 'indep_recon.csv'), index=False)
    return df


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--centroid', default='mass', choices=['mass', 'square'])
    ap.add_argument('--out', default='/Users/perng_mini/AlphaCrops_local/verify/step0/partA_indep')
    a = ap.parse_args()
    run(a.centroid, a.out)
