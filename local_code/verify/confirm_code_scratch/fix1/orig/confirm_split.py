"""
确认运行 v1：储备格、两半、候选、候选池、外国池、big、调查区与选样清单（CONFIRM_PLAN_v5.md §1、§2）。
方案 sha256[:16]=5384fb7efa67f495（冻结）。只构建下标，不训练、不评估。

构建分两步：
  1. 无标签构建：D.y 与 D.X 换成"一访问就报错"的代理（断言 4），生成全部下标、调查区与选样清单；
     每个国家从头构建两次，逐数组比较 sha256（断言 7）；断言 1–3、5、6 同时执行，任何一条失败即停止。
  2. 全部国家的断言通过后，才读测试标签冻结 big（只数各类地块数）并判定单元可行性，再写文件。

种子：一律 np.random.default_rng(stable_hash(方案字符串) % 2**31)（stable_hash = stageA.stable_hash，md5）。
  两半 confirm_v1_halves_{国家}；候选池 confirm_v1_pool_{单元}；外国池 confirm_v1_foreign_{单元}；
  π_r confirm_v1_srs_{单元}_r{r}；S0 confirm_v1_S0_{单元}_r{r}；单元 = {国家}_F{1|2}。

实现上的约定（方案未逐字写明处，与第 0 步 A 的参考实现一致）：
  - 格 = (⌊x/10⌋, ⌊y/10⌋)，格编号按 (cx, cy) 字典序；储备格打乱前也按此顺序排列。
  - 大圆距离：三维单位向量上 cKDTree 找最近测试地块（弦长与大圆距离单调），再用 haversine 算该对的距离。
  - 调查区：格质心 = 格内候选地块 km_xy 的均值；区的地块数 = 区内候选地块数。
  - E–S 顺序：键 log(u)/w 降序（u = rng.random(n)）；S0 依次消耗：8 个区的键 → 合格格的键（格编号升序）→
    按入选顺序逐格 rng.choice(格内候选位置（升序）, 10, replace=False)。
  - "S0 所在区"（EXT）= S0 第 1 步选中的区（含追加的区）。
  - 候选池存为按候选集位置升序的全局下标；π_r 是候选池内位置 0..n_pool−1 的排列。
  - S1 按选入顺序存放（BA 按贪心步序），所以 k1 = 100 的数组就是 k1 = 300 数组的前 100 个。

用法（/Users/perng_mini/AlphaCrops_local/.venv/bin/python -B）：
  confirm_split.py --dry-run --countries AT,PT --root /Users/perng_mini/AlphaCrops_local/verify/confirm_code_scratch/dryrun
  confirm_split.py --root /Users/perng_mini/AlphaCrops_local/results/confirm_v1     # 正式：代码须已登记冻结、作者已批准
"""
import os
import sys
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '2'
sys.dont_write_bytecode = True  # 不在项目目录生成 __pycache__
import argparse  # noqa: E402
import hashlib  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LE = os.path.dirname(HERE)  # label_expansion
sys.path.insert(0, LE)
import guidance_G as G  # noqa: E402
from stageA import CELL_KM, km_xy, save_csv, stable_hash  # noqa: E402

# ---------------------------------------------------------------- 登记值
PLAN = os.path.join(LE, 'CONFIRM_PLAN_v5.md')
PLAN_SHA16 = '5384fb7efa67f495'
PLAN_HASHES = os.path.join(LE, 'plan_hashes.txt')
LOCAL = '/Users/perng_mini/AlphaCrops_local'
SCRATCH = os.path.join(LOCAL, 'verify', 'confirm_code_scratch')
R_G_STAGE0 = os.path.join(LOCAL, 'results', 'guidance_G', 'stage0')
R_A_STAGE0 = os.path.join(LOCAL, 'results', 'stageA', 'stage0')
# 复用的冻结模块（sha256 与 CONFIRM_PLAN_DRAFT_v2 附表、PC-1 摘要、plan_hashes.txt 的登记值一致）；
# common.py 是 stageA.lgb_params 的参数来源，一并锁定
FROZEN = {
    os.path.join(LE, 'guidance_G.py'): '47d729be62d5d2c9280a8a0e2f5f0f85a88e315255cf181321982d7b00e650b5',
    os.path.join(LE, 'stageA.py'): '5db47e2196d169d41217f72c36eca364761a993246e741ec611f5a23e9e2e6bb',
    os.path.join(LE, 'check3_priors.py'): 'b57282b95d5fc954de2a8f7ee2a9e78293e48ece50bb6c495eec5a5d9ef5cd73',
    os.path.join(LE, 'multilearner_G.py'): 'f32183af651d337a0809ffa0c678e65a9e6c6e30d0104eb904f8f1b8dc1150a4',
    os.path.join(LE, 'common.py'): '7624fe9b5aa0a9794d7b00fdd6204e808b59393a6b1bd06d0416bd866dd6545e',
    os.path.join(LOCAL, 'verify/step0/partB/step0B.py'):
        '21a8e5d770fce2b2037a4060cceddf1dfd49460b47e2ab94ac1ee08d64b3a7c1',
    os.path.join(LOCAL, 'verify/step0/partB2/step0B2.py'):
        'd7f37ca5f11c2e9ec630c9884fc3f675e50959bc3f8aa1830fa1e91dcbbbe46c',
}
SOURCE_STAMP = '/Users/perng_mini/AlphaCrops_local/data/dataset_v2.parquet|1485358544|1790223931'
LIBS = {'python': '3.12.13', 'numpy': '2.5.3', 'scipy': '1.18.1', 'sklearn': '1.9.1', 'lightgbm': '4.7.0'}

# ---------------------------------------------------------------- 方案常数
COUNTRIES = ['AT', 'BE', 'DE', 'EE', 'LT', 'LV', 'PT', 'RO', 'SE', 'SI', 'SK']
STAGEA_DEV = ('PT', 'BE', 'SI')
HALVES = (1, 2)
R_EARTH = 6371.0088
BUF_KM = 1.0
N_POOL = 30_000
N_FOREIGN_PER = 5_000
N_REG = 8
KM_KW = dict(n_init=10, random_state=0)
MIN_CELL = 10            # 合格格：候选地块 ≥ 10（KMeans 与 S0 共用）
S0_N_REG = 2             # 先抽 2 个区
S0_MIN_ELIG = 30         # 所选区合格格不足 30 个时追加区
S0_CELLS = 10            # PPS 抽 10 个格
S0_PER_CELL = 10         # 每格 SRS 10 个
S0_N = S0_CELLS * S0_PER_CELL
R = 3                    # 每单元抽样次数
K_A = (50, 100, 200, 400)
K1 = (100, 300)
STRATS = ('SRS', 'EXT', 'BA')
MIN_N = G.MIN_N          # big：测试集中 ≥ 30 个地块的类
MIN_TEST = 3_000
MIN_BIG = 2


def need(cond, msg):
    """断言（不受 python -O 影响）。"""
    if not cond:
        raise AssertionError(msg)


def sha16(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def seed_of(s):
    return stable_hash(s) % 2 ** 31


def rng_of(s):
    return np.random.default_rng(seed_of(s))


def arr_sha(a):
    a = np.ascontiguousarray(np.asarray(a))
    return hashlib.sha256(f'{a.dtype.str}|{a.shape}|'.encode() + a.tobytes()).hexdigest()


def content_hash(d):
    """字典全部数组的内容哈希（与 npz 文件的打包时间无关，存盘前后相同）。"""
    h = hashlib.sha256()
    for k in sorted(d):
        h.update(f'{k}|{arr_sha(d[k])}|'.encode())
    return h.hexdigest()


def code_hash():
    b = b''.join(open(os.path.join(HERE, n), 'rb').read() for n in ('confirm_split.py', 'confirm_run.py'))
    return hashlib.sha256(b).hexdigest()[:16]


def check_env():
    """方案哈希、冻结模块 sha256、数据 source stamp、库版本；不符即停止。返回说明行。"""
    import scipy
    import sklearn
    import lightgbm
    need(sha16(PLAN) == PLAN_SHA16, f'方案哈希 {sha16(PLAN)} ≠ 登记值 {PLAN_SHA16}')
    for p, h in FROZEN.items():
        with open(p, 'rb') as f:
            got = hashlib.sha256(f.read()).hexdigest()
        need(got == h, f'冻结模块已变：{p} {got[:16]} ≠ {h[:16]}')
    need(G.GData.source_stamp() == SOURCE_STAMP, f'数据 source stamp 已变：{G.GData.source_stamp()}')
    got = {'python': sys.version.split()[0], 'numpy': np.__version__, 'scipy': scipy.__version__,
           'sklearn': sklearn.__version__, 'lightgbm': lightgbm.__version__}
    need(got == LIBS, f'库版本不符：{got} ≠ {LIBS}')
    return [f'方案 {os.path.basename(PLAN)} sha256[:16]={PLAN_SHA16}；冻结模块 {len(FROZEN)} 个哈希一致；'
            f'source stamp 一致；库 {LIBS}']


def check_frozen_self():
    """正式运行前：confirm_split.py 与 confirm_run.py 的当前哈希必须已登记在 plan_hashes.txt（代码冻结后才能运行）。"""
    rec = open(PLAN_HASHES, encoding='utf-8').read()
    for n in ('confirm_split.py', 'confirm_run.py'):
        h = sha16(os.path.join(HERE, n))
        need(f'{n} sha256[:16]={h}' in rec, f'{n} 的当前哈希 {h} 未登记在 plan_hashes.txt（未冻结或冻结后被改动）')


# ---------------------------------------------------------------- 无标签数据视图（断言 4）
class Forbidden:
    """y / X 的代理：任何访问（下标、转数组、取长度、迭代、取属性）都报错。"""

    def __init__(self, name):
        object.__setattr__(self, '_name', name)

    def _boom(self, *a, **k):
        raise RuntimeError(f'断言 4：构建划分时访问了 {object.__getattribute__(self, "_name")}')

    __getitem__ = __array__ = __len__ = __iter__ = __contains__ = _boom

    def __getattr__(self, a):
        self._boom()


class LabelFree:
    """构建划分用的数据视图：只有国家、坐标、面积；y 与 X 一访问就报错。"""

    def __init__(self, D):
        self.ctry, self.lon, self.lat, self.area, self.countries = D.ctry, D.lon, D.lat, D.area, D.countries
        self.y, self.X = Forbidden('y'), Forbidden('X')


# ---------------------------------------------------------------- 几何
def unit_vec(lon, lat):
    la, lo = np.radians(np.asarray(lat, float)), np.radians(np.asarray(lon, float))
    return np.c_[np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]


def haversine_km(lon1, lat1, lon2, lat2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * R_EARTH * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


class Nearest:
    """到一组地块的最近大圆距离（km）。"""

    def __init__(self, lon, lat):
        from scipy.spatial import cKDTree
        self.lon, self.lat = np.asarray(lon, float), np.asarray(lat, float)
        self.tree = cKDTree(unit_vec(self.lon, self.lat))

    def km(self, lon, lat):
        _, j = self.tree.query(unit_vec(lon, lat))
        return haversine_km(np.asarray(lon, float), np.asarray(lat, float), self.lon[j], self.lat[j])


def min_km_indep(D, a, b):
    """断言 5 的独立实现（sklearn BallTree，haversine）：b 中地块到 a 的最近大圆距离的最小值（km）。"""
    from sklearn.neighbors import BallTree
    bt = BallTree(np.radians(np.c_[D.lat[a], D.lon[a]]), metric='haversine')
    d, _ = bt.query(np.radians(np.c_[D.lat[b], D.lon[b]]), k=1)
    return float(d.min() * R_EARTH)


def rows_isin(A, B):
    """A 的每一行是否逐字节出现在 B 中。"""
    A, B = np.ascontiguousarray(A), np.ascontiguousarray(B, dtype=np.asarray(A).dtype)
    v = np.dtype((np.void, A.dtype.itemsize * A.shape[1]))
    return np.isin(A.view(v).ravel(), B.view(v).ravel())


def cells_of(lon, lat, lat0):
    """10 km 格：返回 (格表 (n, 2)，按 (cx, cy) 字典序；每个地块的格编号；km_xy)。"""
    xy = km_xy(lon, lat, lat0)
    cxy = np.c_[np.floor(xy[:, 0] / CELL_KM), np.floor(xy[:, 1] / CELL_KM)].astype(np.int64)
    cells, cell = np.unique(cxy, axis=0, return_inverse=True)
    return cells, cell.ravel().astype(np.int64), xy


# ---------------------------------------------------------------- 国家：U、储备、两半
def stored_test(kind, tgt, split):
    d = R_G_STAGE0 if kind == 'G' else R_A_STAGE0
    with np.load(os.path.join(d, f'{tgt}_s{split}.npz')) as z:
        return z['test']


def old_splits(tgt):
    return [('G', 0), ('G', 1)] + ([('A', 0), ('A', 1), ('A', 2)] if tgt in STAGEA_DEV else [])


def check_halves(n_cells, U, halves):
    f1, f2 = halves[1], halves[2]
    need(not np.intersect1d(np.r_[f1, f2], U).size, '断言 1：储备格与 U 相交')
    need(not np.intersect1d(f1, f2).size, '断言 1：F1 与 F2 相交')
    need(np.array_equal(np.sort(np.r_[f1, f2]), np.setdiff1d(np.arange(n_cells), U)), '断言 1：F1 ∪ F2 ≠ 储备格')


def build_country(Dl, tgt, stored=stored_test):
    """去重（与 make_split 相同）、格、U（旧测试集所在的格）、储备与两半。断言 1、3。"""
    t_idx = np.flatnonzero(Dl.ctry == tgt)
    keep = ~pd.DataFrame({'a': Dl.lon[t_idx], 'b': Dl.lat[t_idx], 'c': Dl.area[t_idx]}).duplicated().to_numpy()
    t_idx = t_idx[keep]
    lat0 = float(Dl.lat[t_idx].mean())
    cells, cell, _ = cells_of(Dl.lon[t_idx], Dl.lat[t_idx], lat0)
    recomputed, olds = {}, []
    for kind, s in old_splits(tgt):
        if s not in recomputed:
            recomputed[s] = G.make_split(Dl, tgt, s)['test']
        ref = stored(kind, tgt, s)
        if ref is not None:
            need(np.array_equal(recomputed[s], ref), f'断言 3：{tgt} {kind} s{s} 重算的测试集与已存 stage0 不同')
        olds.append(recomputed[s])
    old_test = np.unique(np.concatenate(olds))
    pos = np.searchsorted(t_idx, old_test)
    need(np.array_equal(t_idx[np.minimum(pos, len(t_idx) - 1)], old_test), f'{tgt}：旧测试地块不在去重目标地块中')
    U = np.unique(cell[pos])
    reserve = np.setdiff1d(np.arange(len(cells)), U)          # 字典序
    sh = reserve[rng_of(f'confirm_v1_halves_{tgt}').permutation(len(reserve))]
    halves = {1: np.sort(sh[:len(sh) // 2]), 2: np.sort(sh[len(sh) // 2:])}
    check_halves(len(cells), U, halves)
    return dict(tgt=tgt, t_idx=t_idx, lat0=lat0, cells=cells, cell=cell, U=U, halves=halves, old_test=old_test)


# ---------------------------------------------------------------- 单元：测试、候选、池、外国池、调查区
def regions_of(D, cand, lat0):
    """调查区：候选 ≥ 10 的格的质心按 (cx, cy) 字典序输入 KMeans（8 区，n_init=10，random_state=0）；其余格归最近质心。"""
    from sklearn.cluster import KMeans
    cells, cand_cell, xy = cells_of(D.lon[cand], D.lat[cand], lat0)
    n = len(cells)
    cnt = np.bincount(cand_cell, minlength=n)
    cen = np.c_[np.bincount(cand_cell, xy[:, 0], n), np.bincount(cand_cell, xy[:, 1], n)] / cnt[:, None]
    fit = cnt >= MIN_CELL
    km = KMeans(N_REG, **KM_KW).fit(cen[fit])
    reg = ((cen[:, None, :] - km.cluster_centers_[None]) ** 2).sum(-1).argmin(1)
    reg[fit] = km.labels_
    reg = reg.astype(np.int64)
    return dict(cell_cx=cells[:, 0], cell_cy=cells[:, 1], cell_cnt=cnt, cell_cen=cen, cell_reg=reg,
                reg_centers=km.cluster_centers_, cand_cell=cand_cell, cand_reg=reg[cand_cell])


def build_unit(Dl, C, h):
    tgt = C['tgt']
    unit = f'{tgt}_F{h}'
    t_idx = C['t_idx']
    is_test = np.isin(C['cell'], C['halves'][h])
    test, rest = t_idx[is_test], t_idx[~is_test]
    near = Nearest(Dl.lon[test], Dl.lat[test])
    cand = rest[near.km(Dl.lon[rest], Dl.lat[rest]) > BUF_KM]
    pool_pos = np.sort(rng_of(f'confirm_v1_pool_{unit}').choice(len(cand), min(N_POOL, len(cand)), replace=False))
    rf = rng_of(f'confirm_v1_foreign_{unit}')
    foreign, n_excl = [], []
    for c in Dl.countries:
        if c == tgt:
            continue
        idx = np.flatnonzero(Dl.ctry == c)
        ok = near.km(Dl.lon[idx], Dl.lat[idx]) > BUF_KM       # 先剔除距本单元测试地块 ≤ 1 km 的地块
        n_excl.append(int((~ok).sum()))
        foreign.append(rf.choice(idx[ok], N_FOREIGN_PER, replace=False))
    sp = dict(unit=unit, country=tgt, half=f'F{h}', lat0=C['lat0'], test=test, cand=cand, pool=cand[pool_pos],
              pool_pos=pool_pos.astype(np.int64), foreign=np.concatenate(foreign).astype(np.int64),
              n_foreign_excl=np.array(n_excl, np.int64), test_cells=C['cells'][C['halves'][h]],
              n_U_cells=len(C['U']), n_reserve_cells=len(C['halves'][1]) + len(C['halves'][2]))
    sp.update(regions_of(Dl, cand, C['lat0']))
    return sp


# ---------------------------------------------------------------- 选样
def es_order(rng, w):
    """Efraimidis–Spirakis：键 log(u)/w 降序 = 按权重 PPS 不放回的随机顺序。"""
    u = rng.random(len(w))
    return np.argsort(-(np.log(u) / np.asarray(w, dtype=np.float64)), kind='stable')


def n_elig(sp, regions):
    return int(((sp['cell_cnt'] >= MIN_CELL) & np.isin(sp['cell_reg'], regions)).sum())


def draw_S0(sp, rng):
    """S0：按区候选数 PPS 抽 2 个区（不足 30 个合格格按同一顺序追加）→ 合格格按地块数 PPS 抽 10 个 → 每格 SRS 10 个。
    返回 (S0 的候选集位置（抽取顺序）, 所选区（E–S 顺序）, 所选格)。"""
    ro = es_order(rng, np.bincount(sp['cand_reg'], minlength=N_REG))
    nsel = S0_N_REG
    while nsel < N_REG and n_elig(sp, ro[:nsel]) < S0_MIN_ELIG:
        nsel += 1
    regions = ro[:nsel]
    cpool = np.flatnonzero((sp['cell_cnt'] >= MIN_CELL) & np.isin(sp['cell_reg'], regions))
    need(len(cpool) >= S0_CELLS, f'{sp["unit"]}：所选区只有 {len(cpool)} 个合格格')
    cells = cpool[es_order(rng, sp['cell_cnt'][cpool])[:S0_CELLS]]
    pos = np.concatenate([rng.choice(np.flatnonzero(sp['cand_cell'] == c), S0_PER_CELL, replace=False) for c in cells])
    return pos.astype(np.int64), regions.astype(np.int64), cells.astype(np.int64)


def ba_sequence(gamma, n0, cap, k):
    """Betti 分配：每步把名额给 γ_g[(n⁰_g+n_g+1)^−½ − (n⁰_g+n_g+2)^−½] 最大、未满容量的组，平局取编号小的组。
    返回长度 k 的组号序列（前 k' 步即 k' 的分配）。"""
    gamma, n0, cap = (np.asarray(a, dtype=np.float64) for a in (gamma, n0, cap))
    n = np.zeros(len(gamma))
    seq = np.empty(k, np.int64)
    for i in range(k):
        m = n0 + n
        gain = np.where(n >= cap, -np.inf, gamma * ((m + 1) ** -0.5 - (m + 2) ** -0.5))
        g = int(np.argmax(gain))
        need(np.isfinite(gain[g]), 'BA：所有组都已满容量')
        n[g] += 1
        seq[i] = g
    return seq


def draw_S1(sp, pi, S0_pos, regions):
    """S1（候选池内位置，按选入顺序）：SRS = π_r 顺序跳过 S0；EXT = 其中只取 S0 所在区；BA = 贪心步序、组内按 π_r。"""
    kmax = max(K1)
    pool_reg = sp['cand_reg'][sp['pool_pos']]
    s0m = np.isin(sp['pool_pos'], S0_pos)
    order = pi[~s0m[pi]]
    ext = order[np.isin(pool_reg[order], regions)]
    need(len(order) >= kmax and len(ext) >= kmax, f'{sp["unit"]}：可补采地块不足（全池 {len(order)}，S0 区 {len(ext)}）')
    gamma = np.bincount(pool_reg, minlength=N_REG) / len(pool_reg)
    n0 = np.bincount(sp['cand_reg'][S0_pos], minlength=N_REG)
    cap = np.bincount(pool_reg[~s0m], minlength=N_REG)
    seq = ba_sequence(gamma, n0, cap, kmax)
    queues = [order[pool_reg[order] == g] for g in range(N_REG)]
    ptr = np.zeros(N_REG, np.int64)
    ba = np.empty(kmax, np.int64)
    for i, g in enumerate(seq):
        ba[i] = queues[g][ptr[g]]
        ptr[g] += 1
    return dict(SRS=order[:kmax], EXT=ext[:kmax], BA=ba), dict(gamma=gamma, n0=n0, cap=cap, seq=seq)


def build_selection(sp):
    unit = str(sp['unit'])
    cand, pool = sp['cand'], sp['pool']
    sel = {}
    for r in range(R):
        pi = rng_of(f'confirm_v1_srs_{unit}_r{r}').permutation(len(pool)).astype(np.int64)
        S0_pos, regions, cells = draw_S0(sp, rng_of(f'confirm_v1_S0_{unit}_r{r}'))
        s1, ba = draw_S1(sp, pi, S0_pos, regions)
        sel.update({f'pi_r{r}': pi, f'S0_r{r}': cand[S0_pos], f'S0reg_r{r}': regions, f'S0cell_r{r}': cells,
                    f'BA_gamma_r{r}': ba['gamma'], f'BA_n0_r{r}': ba['n0'], f'BA_cap_r{r}': ba['cap']})
        for k1 in K1:
            for s in STRATS:
                sel[f'S1_{s}_k{k1}_r{r}'] = pool[s1[s][:k1]]
            sel[f'BA_n_k{k1}_r{r}'] = np.bincount(ba['seq'][:k1], minlength=N_REG)
    return sel


# ---------------------------------------------------------------- 断言 2、5、6
def check_selection(sp, sel):
    """断言 5 的选样部分：各选样 ⊂ 各自的抽样框，S0 ∩ S1 = ∅，k1=100 是 300 的前缀，S0/EXT/BA 的结构。"""
    u, cand, pool = str(sp['unit']), sp['cand'], sp['pool']
    pool_reg = sp['cand_reg'][sp['pool_pos']]
    for r in range(R):
        pi = sel[f'pi_r{r}']
        need(np.array_equal(np.sort(pi), np.arange(len(pool))), f'断言 5：{u} r{r} π_r 不是候选池的排列')
        S0, regs, cells = sel[f'S0_r{r}'], sel[f'S0reg_r{r}'], sel[f'S0cell_r{r}']
        need(len(S0) == S0_N and len(np.unique(S0)) == S0_N and np.isin(S0, cand).all(), f'断言 5：{u} r{r} S0 ⊄ 候选集')
        cc = np.bincount(sp['cand_cell'][np.searchsorted(cand, S0)], minlength=len(sp['cell_cnt']))
        need(len(np.unique(cells)) == S0_CELLS and (cc[cells] == S0_PER_CELL).all() and cc.sum() == S0_N,
             f'断言 5：{u} r{r} S0 不是 10 格 × 10')
        need((sp['cell_cnt'][cells] >= MIN_CELL).all() and np.isin(sp['cell_reg'][cells], regs).all(),
             f'断言 5：{u} r{r} S0 的格不合格或不在所选区')
        need(n_elig(sp, regs) >= S0_MIN_ELIG or len(regs) == N_REG, f'断言 5：{u} r{r} 所选区合格格不足')
        need(len(regs) == S0_N_REG or n_elig(sp, regs[:-1]) < S0_MIN_ELIG, f'断言 5：{u} r{r} 多追加了区')
        srs = pool[pi][~np.isin(pool[pi], S0)]
        exp = {'SRS': srs, 'EXT': srs[np.isin(pool_reg[np.searchsorted(pool, srs)], regs)]}
        for s in STRATS:
            a, b = sel[f'S1_{s}_k{K1[0]}_r{r}'], sel[f'S1_{s}_k{K1[1]}_r{r}']
            need(np.array_equal(a, b[:K1[0]]), f'断言 5：{u} r{r} {s} 的 k1={K1[0]} 不是 {K1[1]} 的前缀')
            need(len(b) == K1[1] and len(np.unique(b)) == K1[1] and np.isin(b, pool).all(),
                 f'断言 5：{u} r{r} {s} ⊄ 候选池')
            need(not np.isin(b, S0).any(), f'断言 5：{u} r{r} S0 ∩ S1({s}) ≠ ∅')
            if s in exp:
                need(np.array_equal(b, exp[s][:K1[1]]), f'断言 5：{u} r{r} {s} 与 π_r 顺序不符')
        for k1 in K1:
            nb = np.bincount(pool_reg[np.searchsorted(pool, sel[f'S1_BA_k{k1}_r{r}'])], minlength=N_REG)
            need(np.array_equal(nb, sel[f'BA_n_k{k1}_r{r}']) and (nb <= sel[f'BA_cap_r{r}']).all() and nb.sum() == k1,
                 f'断言 5：{u} r{r} BA 的分配与样本不符')


def check_unit(D, sp, sel, old_test):
    """断言 2、5、6（D 须含 X；只用坐标与特征，不用 y）。返回最小距离（披露）。"""
    u = str(sp['unit'])
    t, c, p, f = sp['test'], sp['cand'], sp['pool'], sp['foreign']
    need(not np.intersect1d(t, old_test).size, f'断言 2：{u} 测试集与旧测试集有共同地块')
    need(not np.intersect1d(t, c).size, f'断言 5：{u} 测试 ∩ 候选 ≠ ∅')
    need(np.array_equal(c[sp['pool_pos']], p) and len(np.unique(p)) == len(p), f'断言 5：{u} 候选池 ⊄ 候选集')
    others = [x for x in D.countries if x != sp['country']]
    fc = pd.Series(D.ctry[f]).value_counts()
    need(len(np.unique(f)) == len(f) and sorted(fc.index) == sorted(others) and (fc == N_FOREIGN_PER).all(),
         f'断言 5：{u} 外国池不是其余 10 国各 {N_FOREIGN_PER} 个')
    dc, df = min_km_indep(D, t, c), min_km_indep(D, t, f)
    need(dc > BUF_KM and df > BUF_KM, f'断言 5：{u} 到测试的最小大圆距离 候选 {dc:.4f} km、外国 {df:.4f} km')
    check_selection(sp, sel)
    o = np.r_[c, f]
    need(not rows_isin(np.c_[D.lon[t], D.lat[t]], np.c_[D.lon[o], D.lat[o]]).any(), f'断言 6：{u} 测试与候选/外国有相同坐标')
    for feat in ('ae', 's2'):
        X = D.X[feat]
        need(not rows_isin(np.asarray(X[t]), np.asarray(X[o])).any(), f'断言 6：{u} 测试与候选/外国有相同的 {feat} 特征行')
    return dict(dmin_cand_km=dc, dmin_foreign_km=df)


def check_same(a, b, what):
    """断言 7：两次构建的每个数组逐位相同。"""
    need(sorted(a) == sorted(b), f'断言 7：{what} 键不同')
    diff = [k for k in a if arr_sha(a[k]) != arr_sha(b[k])]
    need(not diff, f'断言 7：{what} 两次构建不同：{diff}')


# ---------------------------------------------------------------- big、可行性、写盘
def freeze_big(D, sp):
    """断言全部通过后才调用：只数测试集各类地块数。"""
    cnt = np.bincount(np.asarray(D.y)[sp['test']], minlength=10)
    big = np.flatnonzero(cnt >= MIN_N).astype(np.int64)
    return dict(big=big, n_test=len(sp['test']), feasible=bool(len(sp['test']) >= MIN_TEST and len(big) >= MIN_BIG))


def write_unit(root, sp, sel, overwrite=False):
    """原子写入；文件已存在且不允许覆盖时，只核对内容相同（不改规则重抽）。"""
    for sub, d in (('splits', sp), ('selection', sel)):
        os.makedirs(os.path.join(root, sub), exist_ok=True)
        path = os.path.join(root, sub, f'{sp["unit"]}.npz')
        if os.path.exists(path) and not overwrite:
            with np.load(path) as z:
                need(content_hash(dict(z)) == content_hash(d), f'{path} 已存在且内容不同')
            continue
        tmp = path + '.tmp.npz'
        np.savez_compressed(tmp, **d)
        os.replace(tmp, path)


def manifest_row(sp, sel):
    row = dict(unit=str(sp['unit']), country=str(sp['country']), half=str(sp['half']), n_test=int(sp['n_test']),
               n_cand=len(sp['cand']), n_pool=len(sp['pool']), n_foreign=len(sp['foreign']),
               n_foreign_excl=int(np.sum(sp['n_foreign_excl'])) if 'n_foreign_excl' in sp else -1,
               n_big=len(sp['big']), feasible=bool(sp['feasible']),
               dmin_cand_km=float(sp.get('dmin_cand_km', np.nan)), dmin_foreign_km=float(sp.get('dmin_foreign_km', np.nan)),
               S0_n_regions=' '.join(str(len(sel[f'S0reg_r{r}'])) for r in range(R)))
    row.update({f'{k}_sha': arr_sha(sp[k]) for k in ('test', 'cand', 'pool', 'foreign')})
    row.update(splits_hash=content_hash(sp), selection_hash=content_hash(sel))
    return row


def build_smoke_unit(D, tgt, split):
    """冒烟专用（confirm_run --smoke）：旧 G s{split} 划分（G stage0 的 test、cand、foreign、lat0；候选池 = G rep0 的池），
    调查区与选样用与正式构建相同的函数。单元名 {国家}_s{划分}。"""
    with np.load(os.path.join(R_G_STAGE0, f'{tgt}_s{split}.npz')) as S:
        test, cand, foreign, lat0 = S['test'], S['cand'], S['foreign'], float(S['lat0'])
    pool_pos = G.pool_positions({'cand': cand}, split, 0).astype(np.int64)
    sp = dict(unit=f'{tgt}_s{split}', country=tgt, half=f's{split}', lat0=lat0, test=test, cand=cand,
              pool=cand[pool_pos], pool_pos=pool_pos, foreign=foreign)
    sp.update(regions_of(D, cand, lat0))
    sel = build_selection(sp)
    check_selection(sp, sel)
    sp.update(freeze_big(D, sp))
    return sp, sel


# ---------------------------------------------------------------- 主程序
def build_all(Dl, tgt, stored=stored_test):
    C = build_country(Dl, tgt, stored)
    out = []
    for h in HALVES:
        sp = build_unit(Dl, C, h)
        out.append((sp, build_selection(sp)))
    return C, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--dry-run', action='store_true', help='开发期：只在 ≤2 个国家上测试断言，输出到 confirm_code_scratch/dryrun')
    ap.add_argument('--countries', default=None, help='只用于 --dry-run，逗号分隔')
    a = ap.parse_args()
    root = os.path.realpath(a.root)
    if a.dry_run:
        need(root.startswith(os.path.join(SCRATCH, 'dryrun')), f'--dry-run 只能写到 {SCRATCH}/dryrun 下')
        countries = (a.countries or 'AT,PT').split(',')
        need(len(countries) <= 2 and set(countries) <= set(COUNTRIES), '--dry-run 只在 ≤2 个国家上运行')
    else:
        need(a.countries is None and not root.startswith(SCRATCH), '正式构建：全部 11 国，不写到 scratch')
        check_frozen_self()
        countries = COUNTRIES
    os.makedirs(root, exist_ok=True)
    t0 = time.time()
    lines = check_env()
    lines.append(f'confirm_split.py sha256[:16]={sha16(os.path.join(HERE, "confirm_split.py"))}；'
                 f'{"DRY-RUN" if a.dry_run else "正式"}；国家 {",".join(countries)}；输出 {root}')
    print('\n'.join(lines), flush=True)
    D = G.GData()
    Dl = LabelFree(D)
    units = []
    for tgt in countries:                                             # 第 1 步：无标签构建 + 断言
        C, first = build_all(Dl, tgt)                                 # 断言 1、3、4
        _, second = build_all(Dl, tgt)                                # 断言 7：同一段代码从头再跑一次
        for (sp, sel), (sp2, sel2) in zip(first, second):
            check_same(sp, sp2, f'{sp["unit"]} 划分')
            check_same(sel, sel2, f'{sp["unit"]} 选样')
            sp.update(check_unit(D, sp, sel, C['old_test']))          # 断言 2、5、6
            units.append((sp, sel))
        msg = (f'[{time.time() - t0:.0f}s] {tgt}：格 {len(C["cells"])}，U {len(C["U"])}，储备 '
               f'{len(C["halves"][1]) + len(C["halves"][2])}（F1 {len(C["halves"][1])}，F2 {len(C["halves"][2])}）；'
               f'旧测试 {"、".join(f"{k}s{s}" for k, s in old_splits(tgt))}；断言 1–7 通过')
        lines.append(msg)
        print(msg, flush=True)
    rows = []
    for sp, sel in units:                                             # 第 2 步：冻结 big、可行性、写盘
        sp.update(freeze_big(D, sp))
        write_unit(root, sp, sel, overwrite=a.dry_run)
        rows.append(manifest_row(sp, sel))
    M = pd.DataFrame(rows)
    save_csv(rows, os.path.join(root, 'split_manifest.csv'))
    cols = ['unit', 'n_test', 'n_cand', 'n_pool', 'n_foreign', 'n_foreign_excl', 'dmin_cand_km', 'dmin_foreign_km',
            'n_big', 'feasible', 'S0_n_regions']
    lines += ['', M[cols].to_string(index=False), '',
              f'单元 {len(M)}；可行 {int(M.feasible.sum())}；剔除 {", ".join(M.unit[~M.feasible]) or "无"}',
              f'split_manifest.csv sha256[:16]={sha16(os.path.join(root, "split_manifest.csv"))}',
              f'总耗时 {time.time() - t0:.0f}s']
    print('\n'.join(lines[-6:]), flush=True)
    with open(os.path.join(root, 'split_log.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
