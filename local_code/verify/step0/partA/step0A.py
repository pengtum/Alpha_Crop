# -*- coding: utf-8 -*-
"""
第 0 步 A 部分：有偏初始样本下的先验估计（只走先验 / OA 通道）。

规格：label_expansion/STEP0_SPEC_2026-09-29.md（sha256 前 16 位 14c8de7365278bac，写任何第 0 步代码之前写定）。
术语：PAPER_PLAN §4.9 于 2026-09-29 新增的"标签预算 k""调查成本预算 B""初始样本 S0"。

做什么（全部是冻结概率上的后处理；不训练任何分类器）：
  单元 = 11 国 × 划分 {0,1}；抽样框 = 该单元全部缓冲候选地块 cand（stage0 文件）。
  每个单元、每个场景（S0 类型 × n0）抽 R = 20 次；对每次抽样：
    1. 抽 S0（clus：Betti 式按区、按格 PPS 抽簇；rand：简单随机；none：n0=0 参照）；
    2. 抽补采 S1（标签预算 k ∈ {50,200}：SRS / EXT-p / Betti-Admin(S0) / Betti-Img-ae(S0) / Betti-Img-s2(S0)；
       调查成本预算 B=20（只对 clus）：EXT / SPR(c2=2) / SPR=(c2=1)）；
    3. 用各先验估计量 p̂ 对底座概率做先验修正 check3_priors.reweight(P, p̂, π_train)，评估测试集 OA 与 ba_ge30。
  底座 = S['P_T_{f}'] 的测试行（温度校准，float32；D1 底座），f ∈ {ae, s2}。
  另在 P0test_{f}_i300（G foreign_only 底座）上只重算 E_none、E_oracle，与 G 的 foreign_only 行对账。

随机数方案（审计可逐位重算）：
  seed = int(md5(f'step0A_{国家}_{划分}_{场景}_{n0}_{r}').hexdigest()[:8], 16)，场景 ∈ {clus, rand, none}。
  S0：rng = np.random.default_rng(seed)；clus 依次消耗：8 个区的 E–S 键（rng.random(8)）→ 合格格 E–S 键
      （rng.random(合格格数)，合格格按格编号升序）→ 按入选顺序逐格 rng.choice(格内候选位置(升序), M, replace=False)。
      rand：rng.choice(n_cand, n0, replace=False)。
  每个补采臂一个独立子流：np.random.default_rng([seed, ARM_CODE[臂]])（见 ARM_CODE）。
    SRS / EXT-p：rng.choice(合格位置(升序), 200, replace=False)，k=50 取前 50 个（k=50 是 k=200 的前缀，仍是 SRS）。
    Betti 臂：贪心分配（确定性，k=50 的分配是 k=200 贪心的前 50 步）；按组编号 0..7 逐组
              rng.choice(组内非 S0 位置(升序), n_g(200), replace=False)，k=50 取每组前 n_g(50) 个。
    EXT：可用区内簇的 E–S 键 rng.random(可用簇数)（格编号升序）→ 前 20 簇 → 按顺序逐簇 rng.choice(…, 10)。
    SPR / SPR=：全国可用簇的 E–S 键 → 排序 → 前 20 簇按顺序逐簇 rng.choice(…, 10)；SPR 为其前缀。
  E–S（Efraimidis–Spirakis）键 = log(u)/w，按键降序（等价于 u^(1/w)），与逐次 PPS 不放回抽样同分布。

实现上的解释（规格未逐字写明处）：
  - 格的"质心" = 格内候选地块 km_xy 坐标的均值；KMeans 只在候选数 ≥10 的格上拟合，其余格归最近的区中心。
  - "S0 所在区" = clus 抽样第 1–2 步选中的区（含追加的区）。
  - 参照估计量 E_none / E_oracle / E_EM 与抽样无关，每单元 × 特征只写一行（s0_type='ref'，draw=-1）。
  - E_EM 在测试行（无标签）上运行 em_prior，与 G evaluate 的 P1 相同。

输出（本目录）：rows_A.csv、recon_A.csv、summary_A.txt、draws/{国家}_s{划分}.npz（S0、S1 的候选全局索引）、
      draws/{国家}_s{划分}_frame.npz（格、区、Img 分组）、run_A.log；冒烟：smoke/。

用法：
  python -B step0A.py --smoke              # AT s0，R=2，结构核对 + PPS 蒙特卡洛 + 对账，写 smoke/
  python -B step0A.py                      # 全量 22 单元 × R=20，写 rows_A.csv 等
  python -B step0A.py --summary-only       # 只从 rows_A.csv / recon_A.csv 重做 summary_A.txt
"""
import os
import sys

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_v] = '2'
sys.dont_write_bytecode = True  # 不在项目目录生成 __pycache__

import argparse  # noqa: E402
import hashlib  # noqa: E402
import time  # noqa: E402

LE_DIR = ('/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/'
          'Alpha_Crops/label_expansion')
SPEC = os.path.join(LE_DIR, 'STEP0_SPEC_2026-09-29.md')
SPEC_HASH16 = '14c8de7365278bac'
R_G = '/Users/perng_mini/AlphaCrops_local/results/guidance_G'
STAGE0 = os.path.join(R_G, 'stage0')
OUT_DIR = '/Users/perng_mini/AlphaCrops_local/verify/step0/partA'
sys.path.insert(0, LE_DIR)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402

import guidance_G as G  # noqa: E402  GData、pool_positions、_scores、MIN_N
from stageA import CELL_KM, class_weights_from, km_xy  # noqa: E402
from check3_priors import em_prior, laplace_prior, reweight  # noqa: E402

# ---------------------------------------------------------------- 设定（规格 A1–A5）
COUNTRIES = ['AT', 'BE', 'DE', 'EE', 'LT', 'LV', 'PT', 'RO', 'SE', 'SI', 'SK']
C9 = [c for c in COUNTRIES if c not in ('RO', 'PT')]
SPLITS = [0, 1]
R_FULL = 20
C = 10
FEATS = ['ae', 's2']
N0_M = {100: 10, 500: 20, 2000: 25}           # n0 → 每簇地块数 M
SCENARIOS = [('none', 0)] + [('clus', n) for n in N0_M] + [('rand', n) for n in N0_M]
KS = [50, 200]
B_COST, M_B, C1 = 20, 10, 1                   # 调查成本预算：20 个成本单位，每簇 10 个地块，区内成本 1
N_REG, N_IMG = 8, 8
KM_KW = dict(n_init=10, random_state=0)
MIN_CELL_REG = 10                              # 参与区 KMeans 的格：候选地块数 ≥10
EXTRA_ELIG = 20                                # 所选区合格格数须 ≥ 簇数 + 20
ARM_CODE = {'SRS': 1, 'EXT-p': 2, 'Betti-Admin': 3, 'Betti-Img-ae': 4, 'Betti-Img-s2': 5, 'EXT': 6, 'SPR': 7}
KEY_SAFE = {'SRS': 'SRS', 'EXT-p': 'EXTp', 'Betti-Admin': 'BAdmin', 'Betti-Img-ae': 'BImgae',
            'Betti-Img-s2': 'BImgs2', 'EXT': 'EXT', 'SPR': 'SPR', 'SPR=': 'SPReq'}
BETTI_ARMS = ['Betti-Admin', 'Betti-Img-ae', 'Betti-Img-s2']
ROW_COLS = ['target', 'split', 'feat', 'n0', 's0_type', 'draw', 'arm', 'budget_type', 'k_or_B', 'c2', 'estimator',
            'n_labels_S0', 'n_labels_S1', 's0_L1_frame', 'n_regions', 'prior_L1_test', 'oa', 'ba_ge30']


def spec_hash16():
    with open(SPEC, 'rb') as fh:
        return hashlib.sha256(fh.read()).hexdigest()[:16]


def seed_of(tgt, split, s0type, n0, r):
    return int(hashlib.md5(f'step0A_{tgt}_{split}_{s0type}_{n0}_{r}'.encode()).hexdigest()[:8], 16)


def es_order(rng, w):
    """Efraimidis–Spirakis：键 log(u)/w，降序 = 按权重 PPS 不放回的随机顺序。"""
    u = rng.random(len(w))
    key = np.log(u) / np.asarray(w, dtype=np.float64)
    return np.argsort(-key, kind='stable')


def lap(y):
    """PK50 同款 Laplace 平滑先验。"""
    return laplace_prior(np.asarray(y, dtype=np.int64), C, alpha=1)


def hajek_lap(y, g, N_g, n_g):
    """Betti 分配下的 Hájek 比例 h_c（权重 N_g/n_g）再平滑：p̂_c = (n·h_c + 1)/(n + 10)。"""
    y, g = np.asarray(y), np.asarray(g)
    w = N_g[g] / n_g[g]
    h = np.bincount(y, weights=w, minlength=C) / w.sum()
    n = len(y)
    return (n * h + 1) / (n + C)


def greedy_alloc(gamma, n0g, cap, snaps):
    """Betti Admin-Rep 加已有样本：每步把名额给 γ_g[(n⁰+n+1)^-½ − (n⁰+n+2)^-½] 最大、未满容量的组；平局取小编号。
    返回 {k: n_g}（k ∈ snaps；贪心是增量的，小 k 的分配是大 k 的前缀）。"""
    n = np.zeros(len(gamma), np.int64)
    out = {}
    for step in range(1, max(snaps) + 1):
        m = (n0g + n).astype(np.float64)
        gain = gamma * ((m + 1) ** -0.5 - (m + 2) ** -0.5)
        gain = np.where(n >= cap, -np.inf, gain)
        g = int(np.argmax(gain))
        assert np.isfinite(gain[g]), '所有组都已满容量'
        n[g] += 1
        if step in snaps:
            out[step] = n.copy()
    return out


def fast_scores(yT, ph, big):
    """与 guidance_G._scores 的 oa、ba_ge30 相同（冒烟中逐位核对）。"""
    tot = np.bincount(yT, minlength=C)
    cor = np.bincount(yT[ph == yT], minlength=C)
    rec = cor[big] / tot[big]
    return 100 * float((ph == yT).mean()), 100 * float(np.mean(rec))


def eval_prior(P, yT, big, pi_new, pi_train):
    q, _ = reweight(P, pi_new, pi_train)
    return fast_scores(yT, q.argmax(1), big)


# ---------------------------------------------------------------- 单元底座
def build_frame(D, tgt, split):
    S = np.load(os.path.join(STAGE0, f'{tgt}_s{split}.npz'))
    cand, test, fr = S['cand'], S['test'], S['foreign']
    n_cand = len(cand)
    assert np.array_equal(S['allT'][:n_cand], cand)
    y = np.asarray(D.y)
    yc, yT = y[cand], y[test]
    big = np.flatnonzero(np.bincount(yT, minlength=C) >= G.MIN_N)
    cw = class_weights_from(y[fr])
    pi_train = np.bincount(y[fr], weights=cw[y[fr]], minlength=C) / cw[y[fr]].sum()  # = pi_train_of([], 0)
    # 格：与 make_split 相同的 10 km 格（floor(x/10), floor(y/10)），只在候选地块上编号
    xy = km_xy(D.lon[cand], D.lat[cand], float(S['lat0']))
    cx = np.floor(xy[:, 0] / CELL_KM).astype(np.int64)
    cyy = np.floor(xy[:, 1] / CELL_KM).astype(np.int64)
    cell = pd.factorize(pd.Series(cx).astype(str) + '_' + pd.Series(cyy).astype(str))[0].astype(np.int64)
    n_cell = cell.max() + 1
    cell_cnt = np.bincount(cell, minlength=n_cell)
    cell_cen = np.c_[np.bincount(cell, xy[:, 0], n_cell) / cell_cnt, np.bincount(cell, xy[:, 1], n_cell) / cell_cnt]
    cell_cx = np.zeros(n_cell, np.int64); cell_cx[cell] = cx
    cell_cy = np.zeros(n_cell, np.int64); cell_cy[cell] = cyy
    fit = cell_cnt >= MIN_CELL_REG
    km = KMeans(N_REG, **KM_KW).fit(cell_cen[fit])
    d2 = ((cell_cen[:, None, :] - km.cluster_centers_[None]) ** 2).sum(-1)
    cell_reg = d2.argmin(1)
    cell_reg[fit] = km.labels_
    reg = cell_reg[cell]
    order = np.argsort(cell, kind='stable')
    members = np.split(order, np.cumsum(cell_cnt)[:-1])      # 每格候选位置（升序）
    img = {}
    for f in FEATS:
        img[f] = KMeans(N_IMG, **KM_KW).fit(np.asarray(S[f'G_{f}'][:n_cand], dtype=np.float64)).labels_.astype(np.int64)
    P, P0 = {}, {}
    for f in FEATS:
        p = S[f'P_T_{f}'][n_cand:].astype(np.float64)
        P[f] = p / p.sum(1, keepdims=True)
        p0 = S[f'P0test_{f}_i300'].astype(np.float32).astype(np.float64)   # 与 G evaluate 相同的转换
        P0[f] = p0 / p0.sum(1, keepdims=True)
    return dict(tgt=tgt, split=split, S=S, cand=cand, n_cand=n_cand, yc=yc, yT=yT, big=big, pi_train=pi_train,
                pi_test=np.bincount(yT, minlength=C) / len(yT), pi_frame=np.bincount(yc, minlength=C) / n_cand,
                cell=cell, cell_cnt=cell_cnt, cell_cen=cell_cen, cell_cx=cell_cx, cell_cy=cell_cy,
                cell_reg=cell_reg, reg=reg, reg_cnt=np.bincount(reg, minlength=N_REG), members=members,
                reg_centers=km.cluster_centers_, img=img, P=P, P0=P0, lat0=float(S['lat0']), test=test)


# ---------------------------------------------------------------- S0
def draw_S0(F, s0type, n0, rng):
    """返回 (S0 候选位置（抽取顺序）, info)。"""
    if s0type == 'none':
        return np.array([], np.int64), dict(regions=np.array([], np.int64), cells=np.array([], np.int64))
    if s0type == 'rand':
        return rng.choice(F['n_cand'], n0, replace=False).astype(np.int64), \
            dict(regions=np.array([], np.int64), cells=np.array([], np.int64))
    M = N0_M[n0]
    ncl = n0 // M
    elig = F['cell_cnt'] >= M
    ro = es_order(rng, F['reg_cnt'])                      # 1. 按各区候选地块数 PPS 不放回（E–S 顺序）
    nsel = 2
    while nsel < N_REG and (elig & np.isin(F['cell_reg'], ro[:nsel])).sum() < ncl + EXTRA_ELIG:
        nsel += 1                                         # 2. 合格格不足"簇数+20"时按 PPS 顺序追加一个区
    regions = ro[:nsel]
    pool = np.flatnonzero(elig & np.isin(F['cell_reg'], regions))
    assert len(pool) >= ncl, (F['tgt'], n0, len(pool))
    cells = pool[es_order(rng, F['cell_cnt'][pool])[:ncl]]  # 3. 合格格按格内候选数 PPS 不放回（E–S）
    S0 = np.concatenate([rng.choice(F['members'][c], M, replace=False) for c in cells])  # 4. 每格 SRS M 个
    return S0.astype(np.int64), dict(regions=regions.astype(np.int64), cells=cells.astype(np.int64),
                                     n_elig_sel=int(len(pool)))


# ---------------------------------------------------------------- S1（标签预算 k）
def arm_k(F, arm, S0, info, seed):
    """返回 {k: (S1 位置, 估计 E_S1 用的额外信息)}。"""
    rng = np.random.default_rng([seed, ARM_CODE[arm]])
    s0m = np.zeros(F['n_cand'], bool)
    s0m[S0] = True
    kmax = max(KS)
    if arm in ('SRS', 'EXT-p'):
        ok = ~s0m if arm == 'SRS' else (~s0m & np.isin(F['reg'], info['regions']))
        el = np.flatnonzero(ok)
        take = rng.choice(el, min(kmax, len(el)), replace=False)
        return {k: (take[:k], None) for k in KS}
    grp = F['reg'] if arm == 'Betti-Admin' else F['img'][arm.split('-')[-1]]
    gamma = np.bincount(grp, minlength=8) / F['n_cand']
    n0g = np.bincount(grp[S0], minlength=8)
    N_g = np.bincount(grp[~s0m], minlength=8)             # 容量 = 组内除 S0 外的地块数
    alloc = greedy_alloc(gamma, n0g, N_g, set(KS))
    draws = []
    for g in range(8):
        el = np.flatnonzero((grp == g) & ~s0m)
        draws.append(rng.choice(el, alloc[kmax][g], replace=False) if alloc[kmax][g] > 0 else np.array([], np.int64))
    out = {}
    for k in KS:
        sel = np.concatenate([draws[g][:alloc[k][g]] for g in range(8)]).astype(np.int64)
        out[k] = (sel, dict(grp=grp[sel], N_g=N_g, n_g=alloc[k], gamma=gamma, n0g=n0g))
    return out


# ---------------------------------------------------------------- S1（调查成本预算 B，只对 clus）
def arms_cost(F, S0, info, seed):
    in_reg_cell = np.isin(F['cell_reg'], info['regions'])
    avail = (F['cell_cnt'] >= M_B) & ~np.isin(np.arange(len(F['cell_cnt'])), info['cells'])
    # EXT：S0 所在区内可用簇，PPS 不放回 20 簇
    rng = np.random.default_rng([seed, ARM_CODE['EXT']])
    pool = np.flatnonzero(avail & in_reg_cell)
    ext_cells = pool[es_order(rng, F['cell_cnt'][pool])[:B_COST // C1]]
    ext = np.concatenate([rng.choice(F['members'][c], M_B, replace=False) for c in ext_cells]).astype(np.int64)
    # SPR / SPR=：全国可用簇一次 E–S 排序
    rng = np.random.default_rng([seed, ARM_CODE['SPR']])
    pool = np.flatnonzero(avail)
    order = pool[es_order(rng, F['cell_cnt'][pool])]
    eq_cells = order[:B_COST]                              # SPR=（c2=1）：前 20 簇
    eq_parts = [rng.choice(F['members'][c], M_B, replace=False).astype(np.int64) for c in eq_cells]
    rem, m = B_COST, 0
    for c in order:                                        # SPR（c2=2）：付不起即停
        cost = C1 if in_reg_cell[c] else 2
        if cost > rem:
            break
        rem -= cost
        m += 1
    assert m <= B_COST
    spr_cells = order[:m]
    spr = np.concatenate(eq_parts[:m]) if m else np.array([], np.int64)
    return dict(EXT=(ext, ext_cells), SPR=(spr, spr_cells, B_COST - rem, order[m] if m < len(order) else -1, rem),
                **{'SPR=': (np.concatenate(eq_parts), eq_cells)})


# ---------------------------------------------------------------- 一个单元
def ref_rows(F):
    """参照估计量（与抽样无关）：E_none、E_oracle、E_EM；另返回 EM 先验。"""
    rows = []
    base = dict(target=F['tgt'], split=F['split'], n0=0, s0_type='ref', draw=-1, arm='none', budget_type='none',
                k_or_B=0, c2=np.nan, n_labels_S0=0, n_labels_S1=0, s0_L1_frame=np.nan, n_regions=np.nan)
    for f in FEATS:
        P = F['P'][f]
        pi_em = em_prior(P, F['pi_train'])[0]
        for est, pin in (('E_none', F['pi_train']), ('E_oracle', F['pi_test']), ('E_EM', pi_em)):
            oa, ba = eval_prior(P, F['yT'], F['big'], pin, F['pi_train'])
            rows.append(dict(base, feat=f, estimator=est, prior_L1_test=float(np.abs(pin - F['pi_test']).sum()),
                             oa=oa, ba_ge30=ba))
    return rows


def recon_rows(F):
    """对账：P0test_i300 底座上 E_none、E_oracle，另用候选池先验（G 的 P2）复现 oa_P2；与 G foreign_only 行比较。"""
    out = []
    y = np.asarray(D_GLOBAL.y)
    for rep in (0, 1):
        g = pd.read_csv(os.path.join(R_G, f'rows_t1_{F["tgt"]}_s{F["split"]}_r{rep}.csv'))
        g = g[g.method == 'foreign_only'].set_index('clf')
        pool_pos = G.pool_positions(F['S'], F['split'], rep)
        idx_pool = F['S']['allT'][pool_pos]
        pi_pool = np.bincount(y[idx_pool], minlength=C) / len(idx_pool)
        for f in FEATS:
            P0 = F['P0'][f]
            oa_n, ba_n = eval_prior(P0, F['yT'], F['big'], F['pi_train'], F['pi_train'])
            oa_o, ba_o = eval_prior(P0, F['yT'], F['big'], F['pi_test'], F['pi_train'])
            oa_p, ba_p = eval_prior(P0, F['yT'], F['big'], pi_pool, F['pi_train'])
            oa_pt, ba_pt = fast_scores(F['yT'], F['P'][f].argmax(1), F['big'])
            gs = G._scores(F['yT'], P0.argmax(1), F['big'])
            out.append(dict(target=F['tgt'], split=F['split'], rep=rep, feat=f,
                            oa_Enone_P0=oa_n, ba_Enone_P0=ba_n, oa_Eoracle_P0=oa_o, ba_Eoracle_P0=ba_o,
                            oa_Epool_P0=oa_p, ba_Epool_P0=ba_p, oa_Enone_PT=oa_pt, ba_Enone_PT=ba_pt,
                            oa_G_scores_P0=gs[0], ba_G_scores_P0=gs[1],
                            G_oa=float(g.loc[f, 'oa']), G_ba=float(g.loc[f, 'ba_ge30']),
                            G_oa_P2=float(g.loc[f, 'oa_P2']), G_ba_P2=float(g.loc[f, 'ba_P2']),
                            L1_pool_vs_test=float(np.abs(pi_pool - F['pi_test']).sum()),
                            n_argmax_diff_PT_vs_P0=int((F['P'][f].argmax(1) != P0.argmax(1)).sum()),
                            n_test=len(F['yT'])))
    return out


def run_unit(F, R, draws_path, checks=None, alloc=None):
    """一个单元的全部场景 × 抽样。checks 为 list 时追加结构核对记录；alloc 为 list 时追加 Betti 分配记录。"""
    rows = ref_rows(F)
    yc = F['yc']
    store = {}
    for s0type, n0 in SCENARIOS:
        for r in range(R):
            seed = seed_of(F['tgt'], F['split'], s0type, n0, r)
            S0, info = draw_S0(F, s0type, n0, np.random.default_rng(seed))
            pre = f'{s0type}_{n0}_r{r:02d}'
            store[f'{pre}_seed'] = np.int64(seed)
            store[f'{pre}_S0'] = F['cand'][S0]
            store[f'{pre}_regions'] = info['regions']
            store[f'{pre}_s0cells'] = info['cells']
            assert len(S0) == n0 and len(np.unique(S0)) == n0
            s0set = set(S0.tolist())
            s0L1 = float(np.abs(np.bincount(yc[S0], minlength=C) / n0 - F['pi_frame']).sum()) if n0 else np.nan
            nreg = len(info['regions']) if s0type == 'clus' else np.nan
            ests = []   # (arm, budget_type, k_or_B, c2, estimator, prior, n1)
            if n0:
                ests.append(('none', 'none', 0, np.nan, 'E_S0', lap(yc[S0]), 0))
            arms = ['SRS'] + (['EXT-p'] if s0type == 'clus' else []) + BETTI_ARMS
            for arm in arms:
                res = arm_k(F, arm, S0, info, seed)
                for k in KS:
                    S1, ex = res[k]
                    assert len(S1) == k and len(np.unique(S1)) == k and not (set(S1.tolist()) & s0set), (arm, k)
                    store[f'{pre}_{KEY_SAFE[arm]}_k{k}'] = F['cand'][S1]
                    e_s1 = lap(yc[S1]) if ex is None else hajek_lap(yc[S1], ex['grp'], ex['N_g'], ex['n_g'])
                    ests.append((arm, 'k', k, np.nan, 'E_pool', lap(yc[np.r_[S0, S1]]), k))
                    ests.append((arm, 'k', k, np.nan, 'E_S1', e_s1, k))
                    if checks is not None:
                        checks.append(check_k(F, arm, k, S0, S1, ex, info, e_s1))
                    if alloc is not None and ex is not None:
                        alloc.append(dict(target=F['tgt'], split=F['split'], s0_type=s0type, n0=n0, draw=r, arm=arm,
                                          k=k, n_zero_groups=int((ex['n_g'] == 0).sum()),
                                          hajek_cover=float(ex['N_g'][ex['n_g'] > 0].sum() / ex['N_g'].sum()),
                                          n_g=' '.join(map(str, ex['n_g'])), n0_g=' '.join(map(str, ex['n0g'])),
                                          N_g=' '.join(map(str, ex['N_g']))))
            if s0type == 'clus':
                cres = arms_cost(F, S0, info, seed)
                for arm, c2 in (('EXT', np.nan), ('SPR', 2), ('SPR=', 1)):
                    S1 = cres[arm][0]
                    assert not (set(S1.tolist()) & s0set) and len(np.unique(S1)) == len(S1), arm
                    store[f'{pre}_{KEY_SAFE[arm]}_B{B_COST}'] = F['cand'][S1]
                    store[f'{pre}_{KEY_SAFE[arm]}_cells'] = cres[arm][1].astype(np.int64)
                    ests.append((arm, 'B', B_COST, c2, 'E_pool', lap(yc[np.r_[S0, S1]]), len(S1)))
                    ests.append((arm, 'B', B_COST, c2, 'E_S1', lap(yc[S1]), len(S1)))
                if checks is not None:
                    checks.append(check_cost(F, S0, info, cres))
            if checks is not None and s0type == 'clus':
                checks.append(check_S0(F, n0, S0, info))
            for arm, bt, kb, c2, est, pin, n1 in ests:
                L1 = float(np.abs(pin - F['pi_test']).sum())
                for f in FEATS:
                    oa, ba = eval_prior(F['P'][f], F['yT'], F['big'], pin, F['pi_train'])
                    rows.append(dict(target=F['tgt'], split=F['split'], feat=f, n0=n0, s0_type=s0type, draw=r,
                                     arm=arm, budget_type=bt, k_or_B=kb, c2=c2, estimator=est, n_labels_S0=n0,
                                     n_labels_S1=n1, s0_L1_frame=s0L1, n_regions=nreg, prior_L1_test=L1,
                                     oa=oa, ba_ge30=ba))
    np.savez_compressed(draws_path + '.tmp.npz', **store)
    os.replace(draws_path + '.tmp.npz', draws_path)
    return rows


def save_frame(F, path):
    np.savez_compressed(path + '.tmp.npz', cand=F['cand'], cell=F['cell'], cell_cnt=F['cell_cnt'],
                        cell_cx=F['cell_cx'], cell_cy=F['cell_cy'], cell_cen=F['cell_cen'], cell_reg=F['cell_reg'],
                        reg=F['reg'], reg_centers=F['reg_centers'], img_ae=F['img']['ae'], img_s2=F['img']['s2'],
                        lat0=F['lat0'])
    os.replace(path + '.tmp.npz', path)


# ---------------------------------------------------------------- 冒烟核对
def check_S0(F, n0, S0, info):
    M = N0_M[n0]
    cells, regions = info['cells'], info['regions']
    ok = dict(kind='S0', n0=n0)
    ok['cells_distinct'] = len(np.unique(cells)) == len(cells) == n0 // M
    ok['cells_eligible'] = bool((F['cell_cnt'][cells] >= M).all())
    ok['cells_in_regions'] = bool(np.isin(F['cell_reg'][cells], regions).all())
    cc = np.bincount(F['cell'][S0], minlength=len(F['cell_cnt']))
    ok['M_per_cell'] = bool((cc[cells] == M).all() and cc.sum() == n0)
    elig_sel = ((F['cell_cnt'] >= M) & np.isin(F['cell_reg'], regions)).sum()
    elig_prev = ((F['cell_cnt'] >= M) & np.isin(F['cell_reg'], regions[:-1])).sum()
    ok['region_rule'] = bool(elig_sel >= n0 // M + EXTRA_ELIG and (len(regions) == 2 or elig_prev < n0 // M + EXTRA_ELIG))
    ok['n_regions'] = len(regions)
    return ok


def check_k(F, arm, k, S0, S1, ex, info, e_s1):
    ok = dict(kind='k', arm=arm, k=k)
    ok['no_overlap'] = not (set(S0.tolist()) & set(S1.tolist()))
    if arm == 'EXT-p':
        ok['in_S0_regions'] = bool(np.isin(F['reg'][S1], info['regions']).all())
    if ex is None:
        n = np.bincount(F['yc'][S1], minlength=C)
        ok['laplace_eq_formula'] = bool(np.allclose(e_s1, (n + 1) / (n.sum() + C), atol=0, rtol=1e-15))
    else:
        # 独立重算 Hájek：逐地块权重 = 组内非 S0 数 / 组内样本数
        grp_all = F['reg'] if arm == 'Betti-Admin' else F['img'][arm.split('-')[-1]]
        s0m = np.zeros(F['n_cand'], bool); s0m[S0] = True
        g1 = grp_all[S1]
        wsum = np.zeros(C)
        for i, gi in enumerate(g1):
            Ng = int(((grp_all == gi) & ~s0m).sum()); ng = int((g1 == gi).sum())
            wsum[F['yc'][S1[i]]] += Ng / ng
        h = wsum / wsum.sum()
        ref = (len(S1) * h + 1) / (len(S1) + C)
        ok['hajek_max_abs_diff'] = float(np.abs(ref - e_s1).max())
        ok['alloc_sum'] = int(ex['n_g'].sum()) == k
        ok['alloc_cap'] = bool((ex['n_g'] <= ex['N_g']).all())
        ok['alloc_matches_counts'] = bool(np.array_equal(np.bincount(g1, minlength=8), ex['n_g']))
        # 贪心独立重算（逐步 Python 循环）
        n = [0] * 8
        for _ in range(k):
            best, bg = -1.0, -1
            for g in range(8):
                if n[g] >= ex['N_g'][g]:
                    continue
                m = ex['n0g'][g] + n[g]
                v = ex['gamma'][g] * ((m + 1) ** -0.5 - (m + 2) ** -0.5)
                if v > best:
                    best, bg = v, g
            n[bg] += 1
        ok['greedy_reimpl_equal'] = n == ex['n_g'].tolist()
        # 等权时 Hájek 平滑退化为 Laplace
        e_eq = hajek_lap(F['yc'][S1], np.zeros(len(S1), int), np.array([7.0]), np.array([len(S1)]))
        ok['hajek_equal_w_eq_laplace'] = bool(np.allclose(e_eq, lap(F['yc'][S1]), atol=1e-15))
    return ok


def check_cost(F, S0, info, cres):
    ok = dict(kind='cost')
    in_reg = np.isin(F['cell_reg'], info['regions'])
    s0cells = set(info['cells'].tolist())
    ext, ext_cells = cres['EXT']
    ok['EXT_n'] = len(ext)
    ok['EXT_cells_in_reg_not_S0'] = bool(in_reg[ext_cells].all() and not (set(ext_cells.tolist()) & s0cells))
    ok['EXT_cells_ge10'] = bool((F['cell_cnt'][ext_cells] >= M_B).all())
    spr, spr_cells, spent, nxt, rem = cres['SPR']
    eq, eq_cells = cres['SPR=']
    cost = np.where(in_reg[spr_cells], 1, 2)
    ok['SPR_cost_sum'] = int(cost.sum())
    ok['SPR_cost_le_B'] = int(cost.sum()) <= B_COST and int(cost.sum()) == spent
    ok['SPR_stop_rule'] = bool(nxt < 0 or (1 if in_reg[nxt] else 2) > B_COST - cost.sum())
    ok['SPR_prefix_of_SPReq'] = bool(np.array_equal(spr_cells, eq_cells[:len(spr_cells)]) and
                                     np.array_equal(spr, eq[:len(spr)]))
    ok['SPR_n_labels'] = len(spr)
    ok['SPR_n_in'] = int(in_reg[spr_cells].sum())
    ok['SPR_n_out'] = int((~in_reg[spr_cells]).sum())
    ok['SPReq_n'] = len(eq)
    ok['SPR_cells_not_S0'] = not (set(eq_cells.tolist()) & s0cells)
    ok['per_cell_10'] = bool((np.bincount(F['cell'][eq], minlength=len(F['cell_cnt']))[eq_cells] == M_B).all())
    return ok


def pps_montecarlo(F, lines, n_sim=20000):
    """E–S 顺序 = 逐次 PPS 不放回：区的首抽频率 ∝ 区大小；前 2 个的包含概率与逐次抽样理论值一致；
    格的首抽频率 ∝ 格内候选数（卡方拟合优度）；并用 draw_S0 本身的前两区统计核对。"""
    w = F['reg_cnt'].astype(float)
    p = w / w.sum()
    n_sim = 5 * n_sim                                      # 区的检验多做一些（100k 次）
    rng = np.random.default_rng(987654)
    first = np.zeros(N_REG); inc2 = np.zeros(N_REG)
    for _ in range(n_sim):
        o = es_order(rng, w)
        first[o[0]] += 1
        inc2[o[:2]] += 1
    th2 = np.array([p[i] + sum(p[j] * p[i] / (1 - p[j]) for j in range(N_REG) if j != i) for i in range(N_REG)])
    lines.append(f'[PPS 区] 首抽频率 vs γ：最大 |差| {np.abs(first / n_sim - p).max():.4f}（SE≈{np.sqrt(p.max() * (1 - p.max()) / n_sim):.4f}）；'
                 f'χ² p={stats.chisquare(first, p * n_sim).pvalue:.3f}')
    se2 = np.sqrt(th2 * (1 - th2) / n_sim)
    lines.append(f'[PPS 区] 前 2 包含概率 vs 逐次抽样理论值：最大 |差| {np.abs(inc2 / n_sim - th2).max():.4f}，'
                 f'最大 |z| {np.abs((inc2 / n_sim - th2) / se2).max():.2f}（8 个区，{n_sim} 次）')
    n_sim = n_sim // 5
    # 格：取 clus n0=500 的合格格（M=20），全体区上首抽频率
    wc = F['cell_cnt'][F['cell_cnt'] >= 20].astype(float)
    pc = wc / wc.sum()
    cnt = np.zeros(len(wc))
    for _ in range(n_sim):
        cnt[es_order(rng, wc)[0]] += 1
    lines.append(f'[PPS 格] {len(wc)} 个合格格（M=20）首抽频率 χ² p={stats.chisquare(cnt, pc * n_sim).pvalue:.3f}；'
                 f'频率与权重相关 r={np.corrcoef(cnt / n_sim, pc)[0, 1]:.4f}')
    # 用 draw_S0 本身：前两区的包含频率（种子与主运行无关）
    inc = np.zeros(N_REG); nsim2 = 10000
    for s in range(nsim2):
        _, info = draw_S0(F, 'clus', 100, np.random.default_rng(10 ** 6 + s))
        inc[info['regions'][:2]] += 1
    z2 = (inc / nsim2 - th2) / np.sqrt(th2 * (1 - th2) / nsim2)
    lines.append(f'[PPS draw_S0] n0=100：前两区包含频率 vs 理论最大 |差| {np.abs(inc / nsim2 - th2).max():.4f}，'
                 f'最大 |z| {np.abs(z2).max():.2f}（8 个区，{nsim2} 次）')
    # 簇内 SRS：同一格被抽中时，格内各地块被选频率应相等（取一个格做 χ²）
    c0 = int(np.flatnonzero(F['cell_cnt'] >= 40)[0])
    cnt = np.zeros(F['cell_cnt'][c0])
    pos = {v: i for i, v in enumerate(F['members'][c0])}
    for _ in range(4000):
        for v in rng.choice(F['members'][c0], 10, replace=False):
            cnt[pos[v]] += 1
    lines.append(f'[簇内 SRS] 格 {c0}（{F["cell_cnt"][c0]} 个地块，抽 10）等概率 χ² p={stats.chisquare(cnt).pvalue:.3f}')


def smoke(tgt='AT', split=0, R=2):
    out = os.path.join(OUT_DIR, 'smoke')
    os.makedirs(os.path.join(out, 'draws'), exist_ok=True)
    lines = [f'冒烟：{tgt} s{split}，R={R}；规格 sha256[:16]={spec_hash16()}（登记 {SPEC_HASH16}）']
    t0 = time.time()
    F = build_frame(D_GLOBAL, tgt, split)
    lines.append(f'底座：cand {F["n_cand"]:,}，test {len(F["yT"]):,}，格 {len(F["cell_cnt"])}，'
                 f'区大小 {F["reg_cnt"].tolist()}，Img-ae 组 {np.bincount(F["img"]["ae"]).tolist()}，'
                 f'Img-s2 组 {np.bincount(F["img"]["s2"]).tolist()}；π_train {np.round(F["pi_train"], 4).tolist()}；'
                 f'big {F["big"].tolist()}；{time.time() - t0:.1f}s')
    # KMeans labels_ 与最近中心一致性
    fit = F['cell_cnt'] >= MIN_CELL_REG
    d2 = ((F['cell_cen'][:, None, :] - F['reg_centers'][None]) ** 2).sum(-1)
    lines.append(f'区：拟合格 labels_ 与最近中心不一致的格数 {(d2[fit].argmin(1) != F["cell_reg"][fit]).sum()}；'
                 f'未拟合格 {int((~fit).sum())} 个按最近中心归区')
    checks, alloc = [], []
    rows = run_unit(F, R, os.path.join(out, 'draws', f'{tgt}_s{split}.npz'), checks, alloc)
    pd.DataFrame(alloc).to_csv(os.path.join(out, f'alloc_smoke_{tgt}_s{split}.csv'), index=False)
    save_frame(F, os.path.join(out, 'draws', f'{tgt}_s{split}_frame.npz'))
    df = pd.DataFrame(rows)[ROW_COLS]
    df.to_csv(os.path.join(out, f'rows_smoke_{tgt}_s{split}.csv'), index=False)
    lines.append(f'行数 {len(df)}；耗时 {time.time() - t0:.1f}s')
    ck = pd.DataFrame(checks)
    for kind, sub in ck.groupby('kind'):
        lines.append(f'--- 核对 {kind}（{len(sub)} 条）')
        for col in sub.columns:
            if col in ('kind',):
                continue
            v = sub[col].dropna()
            if len(v) == 0 or col in ('arm', 'n0', 'k'):
                continue
            if all(isinstance(x, (bool, np.bool_)) for x in v):
                lines.append(f'  {col}: 全部为真 = {bool(v.astype(bool).all())}（{int(v.astype(bool).sum())}/{len(v)}）')
            else:
                lines.append(f'  {col}: 最小 {v.min()} 最大 {v.max()} 值 {sorted(pd.unique(v))[:12]}')
    # fast_scores 与 G._scores 逐位核对
    diffs = []
    rng = np.random.default_rng(1)
    for f in FEATS:
        for _ in range(20):
            pin = rng.dirichlet(np.ones(C))
            q, _ = reweight(F['P'][f], pin, F['pi_train'])
            ph = q.argmax(1)
            a = fast_scores(F['yT'], ph, F['big'])
            b = G._scores(F['yT'], ph, F['big'])
            diffs.append(max(abs(a[0] - b[0]), abs(a[1] - b[1])))
    lines.append(f'fast_scores vs guidance_G._scores：40 个随机先验上最大 |差| {max(diffs):.3e}')
    # Laplace 与 PK50 同款：analyze_check3_capacity 用 laplace_prior(y, C)（alpha 默认 1）
    yy = F['yc'][:50]
    lines.append(f'lap(y) == laplace_prior(y, C)（PK50 调用）：{np.array_equal(lap(yy), laplace_prior(yy, C))}')
    pps_montecarlo(F, lines)
    rc = pd.DataFrame(recon_rows(F))
    rc.to_csv(os.path.join(out, f'recon_smoke_{tgt}_s{split}.csv'), index=False)
    for _, r in rc.iterrows():
        lines.append(f'[对账] r{r.rep} {r.feat}: E_none(P0) OA {r.oa_Enone_P0:.4f} vs G oa {r.G_oa:.4f}（差 {r.oa_Enone_P0 - r.G_oa:+.2e}）；'
                     f'BA {r.ba_Enone_P0:.4f} vs {r.G_ba:.4f}；候选池先验 OA {r.oa_Epool_P0:.4f} vs G oa_P2 {r.G_oa_P2:.4f}'
                     f'（差 {r.oa_Epool_P0 - r.G_oa_P2:+.2e}）；E_oracle(P0) OA {r.oa_Eoracle_P0:.4f}；'
                     f'P_T vs P0 argmax 不同 {r.n_argmax_diff_PT_vs_P0} 个')
    # 行级概览
    g = df[df.s0_type != 'ref'].groupby(['s0_type', 'n0', 'arm', 'k_or_B', 'estimator']).agg(
        n=('oa', 'size'), oa=('oa', 'mean'), L1=('prior_L1_test', 'mean'), n1=('n_labels_S1', 'mean'),
        s0L1=('s0_L1_frame', 'mean'), nreg=('n_regions', 'mean'))
    lines.append('--- 行级概览（冒烟单元均值，只用于核对流程，不解读）')
    lines.append(g.round(3).to_string())
    lines.append(df[df.s0_type == 'ref'].to_string())
    txt = '\n'.join(lines)
    open(os.path.join(out, f'smoke_checks_{tgt}_s{split}.txt'), 'w').write(txt + '\n')
    print(txt)


# ---------------------------------------------------------------- 汇总（规格 A6）
def summarize(rows, recon, R, alloc=None):
    rows = rows.copy()
    lines = []
    P = lines.append
    tq = {n: stats.t.ppf(0.975, n - 1) for n in (9, 11)}
    # 国家值：两划分 × R 次抽样平均（参照行：两划分平均）
    key = ['feat', 's0_type', 'n0', 'arm', 'k_or_B', 'estimator', 'target']
    CM = rows.groupby(key)[['oa', 'ba_ge30', 'prior_L1_test', 'n_labels_S1', 's0_L1_frame', 'n_regions']].mean()
    cnt = rows.groupby(key).size()
    assert (cnt[cnt.index.get_level_values('s0_type') != 'ref'] == 2 * R).all()
    assert (cnt[cnt.index.get_level_values('s0_type') == 'ref'] == 2).all()

    def V(metric, feat, s0, n0, arm, kb, est):
        if feat == 'avg':
            return (V(metric, 'ae', s0, n0, arm, kb, est) + V(metric, 's2', s0, n0, arm, kb, est)) / 2
        s = CM.loc[(feat, s0, n0, arm, kb, est), metric]
        return s.reindex(COUNTRIES)

    def REF(metric, feat, est):
        return V(metric, feat, 'ref', 0, 'none', 0, est)

    def st(s, cs):
        v = s.reindex(cs).to_numpy(float)
        n = len(v); m = v.mean(); h = tq[n] * v.std(ddof=1) / np.sqrt(n)
        return m, m - h, m + h, int((v > 0).sum()), n

    def fmt(s, d=2, pos=True):
        a = st(s, COUNTRIES); b = st(s, C9)
        f = f'{{:+.{d}f}}'
        o = f'11国 {f.format(a[0])} [{f.format(a[1])}, {f.format(a[2])}]'
        if pos:
            o += f' {a[3]}/11'
        o += f'  |  9国 {f.format(b[0])} [{f.format(b[1])}, {f.format(b[2])}]'
        if pos:
            o += f' {b[3]}/9'
        return o

    def recov(num, den):
        """恢复率 = 国家均值之比；分母（OA_oracle − OA_none 的国家均值）≤ 0 时加 †（比值不可解释）。"""
        o = []
        for tag, cs in (('11国', COUNTRIES), ('9国', C9)):
            d = den.reindex(cs).mean()
            o.append(f'{tag} {num.reindex(cs).mean() / d:.1%}' + ('†' if d <= 0 else ''))
        return '  |  '.join(o)

    FE = ['ae', 's2', 'avg']
    # 特征无关量核对：prior L1 在 ae 与 s2 行相同
    nr = rows[rows.s0_type != 'ref']
    l1d = (nr[nr.feat == 'ae'].prior_L1_test.to_numpy() - nr[nr.feat == 's2'].prior_L1_test.to_numpy())
    P('第 0 步 A 部分：有偏初始样本下的先验估计（只走先验 / OA 通道）—— 汇总')
    P(f'规格 STEP0_SPEC_2026-09-29.md sha256[:16] = {spec_hash16()}（登记值 {SPEC_HASH16}）')
    P(f'生成时间 {time.strftime("%Y-%m-%d %H:%M:%S")}；rows_A.csv {len(rows):,} 行；单元 {rows.groupby(["target", "split"]).ngroups} 个；R = {R}')
    P('底座：S[P_T_f] 测试行（温度校准，float32，D1 底座），f ∈ {ae, s2}；先验修正 check3_priors.reweight(P, p̂, π_train)，'
      'π_train = pi_train_of([], 0)（外国池类均衡加权分布）。')
    P('国家值 = 两个划分 × 20 次抽样的平均；11 国与去掉 RO、PT 的 9 国：均值、95% CI（t 分布，df=10 / 8）、>0 的国家数。')
    P('"avg" = 两种特征国家值的平均（附加）。全部为 C 级、描述性（测试集为 G s0/s1 已复用的测试集），不做判定。')
    P('先验 L1 = ‖p̂ − π_test‖₁（与特征无关；ae 与 s2 行最大差 %.1e）。OA、BA 单位：pp。' % np.abs(l1d).max())
    P('')
    # ---------------- 参照
    P('=' * 110)
    P('〇、参照（与抽样无关）：E_none（不修正）、E_oracle（测试集真实先验）、E_EM（零标签 EM）')
    P('-' * 110)
    for f in FE:
        for est in ('E_none', 'E_oracle', 'E_EM'):
            P(f'  {f:3s} {est:8s} OA {fmt(REF("oa", f, est), pos=False)}')
        for est in ('E_oracle', 'E_EM'):
            P(f'  {f:3s} {est:8s} − E_none：OA {fmt(REF("oa", f, est) - REF("oa", f, "E_none"))}')
            P(f'  {"":3s} {"":8s}           BA {fmt(REF("ba_ge30", f, est) - REF("ba_ge30", f, "E_none"))}')
        P(f'  {f:3s} E_EM 的 OA 恢复率（国家均值之比）：{recov(REF("oa", f, "E_EM") - REF("oa", f, "E_none"), REF("oa", f, "E_oracle") - REF("oa", f, "E_none"))}')
    for est in ('E_none', 'E_EM'):
        P(f'  先验 L1 {est:7s} {fmt(REF("prior_L1_test", "ae", est), 3, pos=False)}')
    P('')
    P('  ！注意（描述，不改规格）：P_T 底座上用测试集真实先验修正（E_oracle）并不普遍提高 OA。')
    temps = {f: pd.Series({c: np.mean([float(np.load(os.path.join(STAGE0, f'{c}_s{s_}.npz'))[f'temp_{f}'])
                                        for s_ in SPLITS]) for c in COUNTRIES}) for f in FEATS}
    rc0 = recon[recon.rep == 0].assign(d=lambda x: x.oa_Eoracle_P0 - x.oa_Enone_P0).groupby(['target', 'feat']).d.mean().unstack()
    for f in FEATS:
        dT = REF('oa', f, 'E_oracle') - REF('oa', f, 'E_none')
        P(f'    {f}: OA_oracle − OA_none 为正的国家 P_T 底座 {int((dT > 0).sum())}/11（9 国 {int((dT.reindex(C9) > 0).sum())}/9，'
          f'9 国均值 {dT.reindex(C9).mean():+.2f}）；P0test 底座（未校准）{int((rc0[f] > 0).sum())}/11'
          f'（9 国 {int((rc0[f].reindex(C9) > 0).sum())}/9，9 国均值 {rc0[f].reindex(C9).mean():+.2f}）；'
          f'P_T 的温度 s 范围 [{temps[f].min():.2f}, {temps[f].max():.2f}]（logit 乘 s，s<1 使概率变平，先验修正的作用被放大）')
    P('    因此：9 国的恢复率分母为负，恢复率比值不可解释（下文标 †）；11 国的分母主要由 RO、PT 贡献。OA 差值本身仍可描述。')
    P('  逐国参照（P_T 底座，两划分平均；OA/BA，pp）与 P0test 底座的 E_oracle − E_none（OA）：')
    tb = pd.DataFrame({f'{f}_{e}_{m}': REF(m, f, e) for f in FEATS for e in ('E_none', 'E_oracle', 'E_EM')
                       for m in ('oa', 'ba_ge30')})
    for f in FEATS:
        tb[f'{f}_P0_dOA_oracle'] = rc0[f]
        tb[f'{f}_temp'] = temps[f]
    P(tb.round(2).to_string())
    P('')
    # ---------------- S0 描述
    P('=' * 110)
    P('S0 描述：偏差强度 s0_L1_frame = ‖S0 类别构成 − 抽样框类别构成‖₁；clus 所用区数')
    P('-' * 110)
    for n0 in N0_M:
        for s0 in ('clus', 'rand'):
            P(f'  n0={n0:4d} {s0:4s} 偏差强度 {fmt(V("s0_L1_frame", "ae", s0, n0, "none", 0, "E_S0"), 3, pos=False)}')
        P(f'  n0={n0:4d} clus − rand 偏差强度 {fmt(V("s0_L1_frame", "ae", "clus", n0, "none", 0, "E_S0") - V("s0_L1_frame", "ae", "rand", n0, "none", 0, "E_S0"), 3)}')
        sub = rows[(rows.s0_type == 'clus') & (rows.n0 == n0) & (rows.estimator == 'E_S0') & (rows.feat == 'ae')]
        vc = sub.n_regions.value_counts().sort_index()
        P(f'  n0={n0:4d} clus 所用区数分布（440 次抽样）：{dict((int(k), int(v)) for k, v in vc.items())}；'
          f'国家均值 {", ".join(f"{c} {x:.2f}" for c, x in V("n_regions", "ae", "clus", n0, "none", 0, "E_S0").items())}')
    P('')
    # ---------------- A-c1
    P('=' * 110)
    P('一、A-c1 初始样本的偏差：每个 n0 下 E_S0 在 clus 与 rand 下的先验 L1 与 OA 恢复率')
    P('    恢复率 = (OA_E − OA_none)/(OA_oracle − OA_none)，只报国家均值之比')
    P('-' * 110)
    for n0 in N0_M:
        P(f'  -- n0 = {n0}（M = {N0_M[n0]}，{n0 // N0_M[n0]} 簇）')
        Lc = V('prior_L1_test', 'ae', 'clus', n0, 'none', 0, 'E_S0'); Lr = V('prior_L1_test', 'ae', 'rand', n0, 'none', 0, 'E_S0')
        P(f'     先验 L1  clus {fmt(Lc, 3, pos=False)}')
        P(f'              rand {fmt(Lr, 3, pos=False)}')
        P(f'              clus − rand {fmt(Lc - Lr, 3)}')
        for f in FE:
            on = REF('oa', f, 'E_none'); oo = REF('oa', f, 'E_oracle')
            oc = V('oa', f, 'clus', n0, 'none', 0, 'E_S0'); orr = V('oa', f, 'rand', n0, 'none', 0, 'E_S0')
            P(f'     {f:3s} OA 恢复率 clus {recov(oc - on, oo - on)}；rand {recov(orr - on, oo - on)}')
            P(f'     {f:3s} OA(E_S0) − OA(E_none)  clus {fmt(oc - on)}')
            P(f'     {"":3s}                        rand {fmt(orr - on)}')
            P(f'     {f:3s} OA(E_S0) clus − rand  {fmt(oc - orr)}')
    P('')
    # ---------------- A-c2
    P('=' * 110)
    P('二、A-c2 先验用哪些标签估（k = 200，SRS 补采）：OA(E_S1) − OA(E_pool) 与 OA(E_S1) − OA(E_S0)，按 n0；rand 作对照')
    P('-' * 110)
    for s0 in ('clus', 'rand'):
        for n0 in N0_M:
            P(f'  -- S0 = {s0}，n0 = {n0}')
            L = {e: V('prior_L1_test', 'ae', s0, n0, 'SRS', 200, e) for e in ('E_S1', 'E_pool')}
            L['E_S0'] = V('prior_L1_test', 'ae', s0, n0, 'none', 0, 'E_S0')
            P(f'     先验 L1：E_S1 − E_pool {fmt(L["E_S1"] - L["E_pool"], 3)}')
            P(f'              E_S1 − E_S0   {fmt(L["E_S1"] - L["E_S0"], 3)}')
            for f in FE:
                s1 = V('oa', f, s0, n0, 'SRS', 200, 'E_S1'); pl = V('oa', f, s0, n0, 'SRS', 200, 'E_pool')
                e0 = V('oa', f, s0, n0, 'none', 0, 'E_S0')
                P(f'     {f:3s} OA(E_S1) − OA(E_pool) {fmt(s1 - pl)}')
                P(f'     {"":3s} OA(E_S1) − OA(E_S0)   {fmt(s1 - e0)}')
    P('  -- 参照：S0 = none（n0 = 0），SRS k = 200 的 E_S1（即 200 个随机标签的 PK）')
    for f in FE:
        on = REF('oa', f, 'E_none'); oo = REF('oa', f, 'E_oracle')
        s1 = V('oa', f, 'none', 0, 'SRS', 200, 'E_S1')
        P(f'     {f:3s} OA − E_none {fmt(s1 - on)}；恢复率 {recov(s1 - on, oo - on)}')
    P('')
    # ---------------- A-c3
    P('=' * 110)
    P('三、A-c3 成本口径下补采放在哪里（clus，B = 20，M = 10）：EXT、SPR（c2=2）、SPR=（c2=1）')
    P('-' * 110)
    for n0 in N0_M:
        P(f'  -- n0 = {n0}')
        nl = V('n_labels_S1', 'ae', 'clus', n0, 'SPR', B_COST, 'E_S1')
        sub = rows[(rows.s0_type == 'clus') & (rows.n0 == n0) & (rows.arm == 'SPR') & (rows.estimator == 'E_S1') & (rows.feat == 'ae')]
        P(f'     所得标签数：EXT 200，SPR= 200；SPR 国家均值 {fmt(nl, 1, pos=False)}；抽样级范围 [{sub.n_labels_S1.min()}, {sub.n_labels_S1.max()}]')
        for est in ('E_pool', 'E_S1'):
            L = {a: V('prior_L1_test', 'ae', 'clus', n0, a, B_COST, est) for a in ('EXT', 'SPR', 'SPR=')}
            P(f'     [{est}] 先验 L1 水平：EXT {L["EXT"].mean():.3f}，SPR {L["SPR"].mean():.3f}，SPR= {L["SPR="].mean():.3f}（11 国均值）')
            P(f'     [{est}] 先验 L1 SPR − EXT  {fmt(L["SPR"] - L["EXT"], 3)}')
            P(f'     [{est}]         SPR= − EXT {fmt(L["SPR="] - L["EXT"], 3)}')
            P(f'     [{est}]         SPR= − SPR {fmt(L["SPR="] - L["SPR"], 3)}')
            for f in FE:
                O = {a: V('oa', f, 'clus', n0, a, B_COST, est) for a in ('EXT', 'SPR', 'SPR=')}
                on = REF('oa', f, 'E_none'); oo = REF('oa', f, 'E_oracle')
                P(f'     [{est}] {f:3s} OA 水平：EXT {O["EXT"].mean():.2f}，SPR {O["SPR"].mean():.2f}，SPR= {O["SPR="].mean():.2f}；'
                  f'恢复率 EXT {recov(O["EXT"] - on, oo - on)}；SPR {recov(O["SPR"] - on, oo - on)}；SPR= {recov(O["SPR="] - on, oo - on)}')
                P(f'     [{est}] {f:3s} OA SPR − EXT  {fmt(O["SPR"] - O["EXT"])}')
                P(f'     [{est}] {f:3s}    SPR= − EXT {fmt(O["SPR="] - O["EXT"])}')
                P(f'     [{est}] {f:3s}    SPR= − SPR {fmt(O["SPR="] - O["SPR"])}')
    P('')
    # ---------------- A-c4
    P('=' * 110)
    P('四、A-c4 Betti 分配与 SRS（E_S1，Betti 臂用 Hájek；k ∈ {50, 200}）；另报 AE 分组 − S2 分组（Betti-Img-ae − Betti-Img-s2）')
    P('-' * 110)
    if alloc is not None:
        P('  分配描述（alloc_A.csv；全部抽样平均）：得 0 名额的组数、Hájek 覆盖率 = Σ_{n_g>0} N_g / Σ N_g（E_S1 只代表有样本的组）')
        ag = alloc.groupby(['s0_type', 'n0', 'arm', 'k'])[['n_zero_groups', 'hajek_cover']].mean().unstack('arm')
        P(ag.round(3).to_string())
    scen = [('none', 0)] + [('clus', n) for n in N0_M] + [('rand', n) for n in N0_M]
    for k in KS:
        for s0, n0 in scen:
            P(f'  -- k = {k}，S0 = {s0}，n0 = {n0}')
            L = {a: V('prior_L1_test', 'ae', s0, n0, a, k, 'E_S1') for a in ['SRS'] + BETTI_ARMS}
            for a in BETTI_ARMS:
                P(f'     先验 L1 {a:12s} − SRS {fmt(L[a] - L["SRS"], 3)}')
            P(f'     先验 L1 Img-ae − Img-s2   {fmt(L["Betti-Img-ae"] - L["Betti-Img-s2"], 3)}')
            for f in FE:
                O = {a: V('oa', f, s0, n0, a, k, 'E_S1') for a in ['SRS'] + BETTI_ARMS}
                for a in BETTI_ARMS:
                    P(f'     {f:3s} OA {a:12s} − SRS {fmt(O[a] - O["SRS"])}')
                P(f'     {f:3s} OA Img-ae − Img-s2   {fmt(O["Betti-Img-ae"] - O["Betti-Img-s2"])}')
    P('')
    # ---------------- A-c5
    P('=' * 110)
    P('五、A-c5 各估计量下 BA 的代价：ba_ge30(E) − ba_ge30(E_none)（预期为负，是 BA 定义的推论，不作为发现）')
    P('-' * 110)
    items = [('E_oracle', None), ('E_EM', None), ('none SRS k50 E_S1', ('none', 0, 'SRS', 50, 'E_S1')),
             ('none SRS k200 E_S1', ('none', 0, 'SRS', 200, 'E_S1'))]
    for n0 in N0_M:
        for s0 in ('clus', 'rand'):
            items.append((f'{s0} n0={n0} E_S0', (s0, n0, 'none', 0, 'E_S0')))
            items.append((f'{s0} n0={n0} SRS k200 E_pool', (s0, n0, 'SRS', 200, 'E_pool')))
            items.append((f'{s0} n0={n0} SRS k200 E_S1', (s0, n0, 'SRS', 200, 'E_S1')))
        items.append((f'clus n0={n0} SPR= B20 E_S1', ('clus', n0, 'SPR=', B_COST, 'E_S1')))
    for f in FE:
        bn = REF('ba_ge30', f, 'E_none')
        for name, spec in items:
            b = REF('ba_ge30', f, name) if spec is None else V('ba_ge30', f, *spec)
            P(f'  {f:3s} {name:28s} ΔBA {fmt(b - bn)}')
    P('')
    # ---------------- A-c6
    P('=' * 110)
    P('六、A-c6 n0 的趋势（100 → 500 → 2000）：OA(E_pool) − OA(E_S1)（正 = 合并优于只用补采）')
    P('-' * 110)
    for s0 in ('clus', 'rand'):
        for arm, kb in (('SRS', 50), ('SRS', 200)) + ((('SPR=', B_COST), ('EXT', B_COST)) if s0 == 'clus' else ()):
            for f in FE:
                seg = []
                for n0 in N0_M:
                    d = V('oa', f, s0, n0, arm, kb, 'E_pool') - V('oa', f, s0, n0, arm, kb, 'E_S1')
                    a = st(d, COUNTRIES); b = st(d, C9)
                    seg.append(f'n0={n0}: {a[0]:+.2f} [{a[1]:+.2f}, {a[2]:+.2f}] {a[3]}/11; 9国 {b[0]:+.2f} [{b[1]:+.2f}, {b[2]:+.2f}] {b[3]}/9')
                P(f'  {s0} {arm} {"k" if arm == "SRS" else "B"}={kb} {f:3s} OA ' + ' || '.join(seg))
            seg = []
            for n0 in N0_M:
                d = V('prior_L1_test', 'ae', s0, n0, arm, kb, 'E_pool') - V('prior_L1_test', 'ae', s0, n0, arm, kb, 'E_S1')
                a = st(d, COUNTRIES)
                seg.append(f'n0={n0}: {a[0]:+.3f} [{a[1]:+.3f}, {a[2]:+.3f}] {a[3]}/11')
            P(f'  {s0} {arm} {"k" if arm == "SRS" else "B"}={kb} 先验 L1(E_pool) − L1(E_S1) ' + ' || '.join(seg))
    P('')
    # ---------------- 对账
    P('=' * 110)
    P('七、对账：P0test_i300 底座（G foreign_only 行的底座）上的 E_none、E_oracle 与 G 的 oa、oa_P2')
    P('-' * 110)
    rc = recon
    P(f'  单元 × 重复 × 特征 = {len(rc)} 行')
    P(f'  E_none(P0) OA − G oa：最大 |差| {np.abs(rc.oa_Enone_P0 - rc.G_oa).max():.2e}；BA − G ba_ge30：最大 |差| {np.abs(rc.ba_Enone_P0 - rc.G_ba).max():.2e}')
    P(f'  本脚本 fast_scores 与 guidance_G._scores（P0 argmax）：OA 最大 |差| {np.abs(rc.oa_Enone_P0 - rc.oa_G_scores_P0).max():.2e}')
    P(f'  候选池先验（G 的 P2 定义，本脚本用 check3_priors.reweight 复现）OA − G oa_P2：最大 |差| {np.abs(rc.oa_Epool_P0 - rc.G_oa_P2).max():.2e}；'
      f'BA − G ba_P2：最大 |差| {np.abs(rc.ba_Epool_P0 - rc.G_ba_P2).max():.2e}')
    d = rc.oa_Eoracle_P0 - rc.G_oa_P2
    P(f'  E_oracle(P0) OA − G oa_P2：均值 {d.mean():+.3f} pp，范围 [{d.min():+.3f}, {d.max():+.3f}]；'
      f'差异来源：P2 用候选池（3 万池，随重复变化）的类别比例，E_oracle 用测试集真实比例；'
      f'两者 L1 距离均值 {rc.L1_pool_vs_test.mean():.3f}（范围 {rc.L1_pool_vs_test.min():.3f}–{rc.L1_pool_vs_test.max():.3f}）')
    P(f'  P_T 底座与 P0test 底座：E_none 的 argmax 不同的测试地块数 最大 {rc.n_argmax_diff_PT_vs_P0.max()}'
      f'（占比最大 {(rc.n_argmax_diff_PT_vs_P0 / rc.n_test).max():.2e}）；OA 差 最大 |差| {np.abs(rc.oa_Enone_PT - rc.oa_Enone_P0).max():.3f} pp'
      f'（温度缩放不改 argmax，差异只来自 float16 存储的并列/舍入）')
    ccv = rc[rc.rep == 0].groupby(['target', 'feat'])[['oa_Enone_P0', 'G_oa', 'oa_Eoracle_P0', 'G_oa_P2']].mean().unstack('feat')
    P('  国家值（两划分平均，重复 0 的 G 行）：')
    P(ccv.round(3).to_string())
    P('')
    P('† 恢复率分母（OA_oracle − OA_none 的国家均值）≤ 0，比值不可解释。')
    P('说明：本汇总只做描述，不做判定；第 0 步数字不进论文摘要（规格"不做"节）。')
    return '\n'.join(lines)


# ---------------------------------------------------------------- 主程序
def main():
    global D_GLOBAL
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--smoke-unit', default='AT_0')
    ap.add_argument('--summary-only', action='store_true')
    ap.add_argument('--R', type=int, default=R_FULL)
    a = ap.parse_args()
    h = spec_hash16()
    assert h == SPEC_HASH16, f'规格哈希不符：{h}'
    if a.summary_only:
        rows = pd.read_csv(os.path.join(OUT_DIR, 'rows_A.csv'))
        recon = pd.read_csv(os.path.join(OUT_DIR, 'recon_A.csv'))
        alloc = pd.read_csv(os.path.join(OUT_DIR, 'alloc_A.csv'))
        txt = summarize(rows, recon, a.R, alloc)
        open(os.path.join(OUT_DIR, 'summary_A.txt'), 'w').write(txt + '\n')
        print(txt)
        return
    D_GLOBAL = G.GData()
    if a.smoke:
        tgt, sp = a.smoke_unit.split('_')
        smoke(tgt, int(sp), R=2)
        return
    os.makedirs(os.path.join(OUT_DIR, 'draws'), exist_ok=True)
    log = open(os.path.join(OUT_DIR, 'run_A.log'), 'a')
    all_rows, all_recon, all_alloc = [], [], []
    t00 = time.time()
    for tgt in COUNTRIES:
        for split in SPLITS:
            t0 = time.time()
            F = build_frame(D_GLOBAL, tgt, split)
            save_frame(F, os.path.join(OUT_DIR, 'draws', f'{tgt}_s{split}_frame.npz'))
            rows = run_unit(F, a.R, os.path.join(OUT_DIR, 'draws', f'{tgt}_s{split}.npz'), alloc=all_alloc)
            all_rows += rows
            all_recon += recon_rows(F)
            msg = f'{tgt} s{split}: cand {F["n_cand"]:,} test {len(F["yT"]):,} 行 {len(rows):,} {time.time() - t0:.0f}s'
            print(msg, flush=True)
            log.write(msg + '\n'); log.flush()
    rows = pd.DataFrame(all_rows)[ROW_COLS]
    recon = pd.DataFrame(all_recon)
    rows.to_csv(os.path.join(OUT_DIR, 'rows_A.csv.tmp'), index=False)
    os.replace(os.path.join(OUT_DIR, 'rows_A.csv.tmp'), os.path.join(OUT_DIR, 'rows_A.csv'))
    recon.to_csv(os.path.join(OUT_DIR, 'recon_A.csv'), index=False)
    alloc = pd.DataFrame(all_alloc)
    alloc.to_csv(os.path.join(OUT_DIR, 'alloc_A.csv'), index=False)
    txt = summarize(rows, recon, a.R, alloc)
    open(os.path.join(OUT_DIR, 'summary_A.txt'), 'w').write(txt + '\n')
    log.write(f'完成，总耗时 {time.time() - t00:.0f}s\n')
    print(f'完成，总耗时 {time.time() - t00:.0f}s')


D_GLOBAL = None
if __name__ == '__main__':
    main()
