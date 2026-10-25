"""
G 实验运行器（ANALYSIS_PLAN_guidance.md）：只替换指导数据（几何 + 预测），比较 AE 指导与 S2 指导的选样扩充。

用法：
  python guidance_G.py stage0 --jobs 5            # 每个（国家, 划分）：AE 与 S2 外国模型、分组留出温度、稳健 PCA、E_far 掩码
  python guidance_G.py run --tier 1 --jobs 5      # 按方案第 9 节的顺序选样与训练；--tier 2（含 ORACLE 交叉拟合）/ 3 同
  python guidance_G.py all --jobs 5               # 阶段 0 → 层级 1 → 层级 2，一次跑完
每个任务 = (层级, 国家, 划分, 重复)，结果写入自己的 CSV（原子写入），可断点续跑。
输出目录：RESULTS_DIR/guidance_G/（GUIDE_OUT 可覆盖；GUIDE_TREES 可调小，只用于冒烟测试）。
不修改 stageA.py；划分、外国池、候选池的构造与阶段 A 逐位一致。
"""
import argparse
import hashlib
import inspect
import multiprocessing as mp
import os
import time

import numpy as np
import pandas as pd

from stageA import BUFFER_KM, CELL_KM, MIN_N, TEST_FRAC, class_weights_from, km_xy, lgb_params, save_csv, \
    stable_hash
import samplers.base as B
from samplers.base import X_of, finalize, register
import samplers.designs, samplers.uncertainty, samplers.coverage  # noqa: F401,E401 注册方法
from samplers.uncertainty import badge_core


def _out():
    from common import RESULTS_DIR
    return os.environ.get('GUIDE_OUT', os.path.join(RESULTS_DIR, 'guidance_G'))


OUT = _out()  # 模块级：spawn 启动的子进程也能拿到
PLAN = 'ANALYSIS_PLAN_guidance.md'
TREES = int(os.environ.get('GUIDE_TREES', 300))
ITERS = sorted({min(t, TREES) for t in (200, 250, 300)})  # 同一模型的中途预测轮数（稳健性分析）
COUNTRIES_G = os.environ.get('GUIDE_COUNTRIES', 'AT,BE,DE,EE,LT,LV,PT,RO,SE,SI,SK').split(',')
SPLITS, REPS = [0, 1], [0, 1]
N_POOL = 30_000
N_FOREIGN_PER = 5_000
N_PCA = 32
CLIP = 5.0
R_MULT = 50.0
N_GROUPS = int(os.environ.get('GUIDE_TEMP_GROUPS', 5))  # 削减方案第 1 步：改为 3
N_ORACLE_FOLDS = 5
GEO_W_S2XY = 2.0

# ---------------------------------------------------------------- 方法与条件
E1 = ['lpm2_nogeo', 'kcenter', 'typiclust', 'kherding']       # 只用几何
PRED = ['margin', 'predstrat_srs']                             # 只用预测
HYB = ['gist_margin', 'badge_c']                               # 几何 + 预测
FAMILIES = {'geom': E1, 'pred': PRED, 'hyb': HYB}              # srs 不属于任何家族
METHOD_D = {'lpm2_nogeo': 16, 'kcenter': 32, 'typiclust': 32, 'kherding': 32, 'gist_margin': 32, 'badge_c': 32,
            'kcenter_foreign': 32}
G_HP = {'lpm2_nogeo': {'d': 16, 'geo_w': 0.0}, 'kcenter': {'d': 32}, 'typiclust': {'d': 32}, 'kherding': {'d': 32},
        'gist_margin': {'d': 32}, 'badge_c': {'d': 32}, 'predstrat_srs': {'geo_w': 0.0}, 'margin': {}, 'srs': {},
        'kcenter_foreign': {'d': 32}}
# 条件 → (几何来源, 预测来源)。几何来源是 S 中的键（目标行），或 'xy' / 's2xy' / 'oracle'；预测同理。
COND = {
    'none': (None, None),
    'AE': ('G_ae', 'P_T_ae'), 'S2': ('G_s2', 'P_T_s2'),
    'XY': ('xy', None), 'S2XY': ('s2xy', None),
    'ORACLE': ('oracle', 'oracle'),
    'ORACLE2': ('oracle2', 'oracle'),  # rev1（看过结果后的修正）：几何只保留池中 ≥30 个地块的类的列，再逐列 z 截断
    'AEg_S2p': ('G_ae', 'P_T_s2'), 'S2g_AEp': ('G_s2', 'P_T_ae'),
    'RAWZ_AE': ('RZ_ae', None), 'RAWZ_S2': ('RZ_s2', None),
    'NATIVE_AE': ('NT_ae', None), 'NATIVE_S2': ('NT_s2', None),
}
F_GEOM = {'G_ae': 'GF_ae', 'G_s2': 'GF_s2'}  # kcenter_foreign 用的外国池几何（同一 PCA）


@register('badge_c')
def select_badge_c(ctx, k, seed, d=32):
    """G 专用 BADGE：h = X_of(ctx,'ae',d) 在候选池内居中后再做幻觉梯度 k-means++（对 T 的整体平移不变）。"""
    rng = np.random.default_rng(seed)
    h = X_of(ctx, 'ae', d)
    h = h - h.mean(0)
    idx = badge_core(ctx.P_T, h, k, rng=rng)
    return finalize(idx, k, ctx.N)


def check_geo_w():
    """签名里有 geo_w 的 G 方法都必须显式收到 0（ctx.T_xy=None，漏传会直接报错，这里提前断言）。"""
    for m in E1 + PRED + HYB + ['kcenter_foreign']:
        sig = inspect.signature(B.REGISTRY[m])
        if 'geo_w' in sig.parameters:
            assert G_HP[m].get('geo_w') == 0.0, m


def jobs_for(tier, split, rep):
    """返回 [(cond, method, k, clfs)]，顺序即运行顺序。"""
    J = []
    both = ('s2', 'ae')
    if tier == 1:
        for cond in ('AE', 'S2'):
            for m in E1 + (PRED + HYB if rep == 0 else []):
                for k in (50, 200):
                    J.append((cond, m, k, both))
        for k in (50, 200, 100, 400):
            J.append(('none', 'srs', k, both))
    elif tier == 2 and rep == 0:
        for cond in ('XY', 'S2XY'):
            for m in ('lpm2_nogeo', 'kcenter'):
                for k in (50, 200):
                    J.append((cond, m, k, both))
        for m in ('lpm2_nogeo', 'typiclust', 'margin'):
            for k in (50, 200):
                J.append(('ORACLE', m, k, both))
    elif tier == 4 and rep == 0:  # rev1（事后）：修正后的 ORACLE 几何
        J += [('ORACLE2', m, k, both) for m in ('lpm2_nogeo', 'typiclust') for k in (50, 200)]
    elif tier == 3:
        if rep == 0:
            J += [(c, m, 50, both) for c in ('RAWZ_AE', 'RAWZ_S2') for m in E1]
            J += [(c, m, k, both) for c in ('AEg_S2p', 'S2g_AEp') for m in HYB for k in (50, 200)]
            J += [(c, m, 50, both) for c in ('NATIVE_AE', 'NATIVE_S2') for m in ('lpm2_nogeo', 'typiclust')]
            J += [(c, m, 50, ('fus',)) for c in ('AE', 'S2') for m in E1]
        J += [(c, 'kcenter_foreign', k, both) for c in ('AE', 'S2') for k in (50, 200)]
        if rep == 0:
            J.append(('none', 'srs', 1000, both))
            J += [(c, m, 1000, both) for c in ('AE', 'S2') for m in ('lpm2_nogeo', 'typiclust')]
    return J


# ---------------------------------------------------------------- 数据
class GData:
    """与 stageA.Data 相同的数据，另外记录特征列名（NATIVE 预处理要用）。
    读整个 parquet 的内存峰值约 6 GB，所以主进程先把用到的数组存成 .npy（build_cache），
    工作进程以内存映射方式读取，特征矩阵在各进程间共享页缓存。缓存记录数据集的大小与修改时间，不一致时报错。"""
    KEYS = ('y', 'ctry', 'lon', 'lat', 'area', 'X_ae', 'X_s2')

    def __init__(self):
        import json
        meta = json.load(open(os.path.join(self.cache_dir(), 'meta.json')))
        assert meta['source'] == self.source_stamp(), '数据集已变化，请重新运行 GData.build_cache()'
        z = {k: np.load(os.path.join(self.cache_dir(), f'{k}.npy'), mmap_mode='r' if k.startswith('X_') else None)
             for k in self.KEYS}
        self.y, self.ctry, self.lon, self.lat, self.area = z['y'], z['ctry'], z['lon'], z['lat'], z['area']
        self.X = {'ae': z['X_ae'], 's2': z['X_s2']}
        self.classes, self.countries, self.cols = meta['classes'], meta['countries'], meta['cols']

    @staticmethod
    def cache_dir():
        from common import DATA_DIR
        return os.path.join(DATA_DIR, 'cache_guidance_G')

    @staticmethod
    def source_stamp():
        from common import DATASET_V2
        st = os.stat(DATASET_V2)
        return f'{DATASET_V2}|{st.st_size}|{int(st.st_mtime)}'

    @classmethod
    def build_cache(cls):
        import json
        import pyarrow.parquet as pq
        from stageA import Data
        from common import DATASET_V2, feature_cols
        path = os.path.join(cls.cache_dir(), 'meta.json')
        if os.path.exists(path) and json.load(open(path))['source'] == cls.source_stamp():
            return
        os.makedirs(cls.cache_dir(), exist_ok=True)
        d = Data()
        arr = dict(y=d.y, ctry=d.ctry.astype('U2'), lon=d.lon, lat=d.lat, area=d.area, X_ae=d.X['ae'], X_s2=d.X['s2'])
        for k in cls.KEYS:
            np.save(os.path.join(cls.cache_dir(), f'{k}.npy'), np.ascontiguousarray(arr[k]))
        names = pq.read_schema(DATASET_V2).names
        cols = {f: feature_cols(pd.DataFrame(columns=names), f) for f in ('ae', 's2')}
        json.dump(dict(source=cls.source_stamp(), classes=list(d.classes), countries=list(d.countries), cols=cols),
                  open(path, 'w'))

    def fusion(self):
        if 'fus' not in self.X:
            self.X['fus'] = np.hstack([self.X['ae'], self.X['s2']])
        return self.X['fus']


# ---------------------------------------------------------------- 阶段 0
def make_split(D, tgt, split):
    """与 stageA.stage0_one 逐位一致：去重、10 km 格网划分、1 km 缓冲、外国池（同一随机数流）。
    另外给出 E_far 掩码：到完整缓冲候选集最近距离最远的三分之一测试地块。"""
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(1000 + 17 * split + stable_hash(tgt) % 997)
    t_idx = np.flatnonzero(D.ctry == tgt)
    keep = ~pd.DataFrame({'a': D.lon[t_idx], 'b': D.lat[t_idx], 'c': D.area[t_idx]}).duplicated().to_numpy()
    t_idx = t_idx[keep]
    lat0 = D.lat[t_idx].mean()
    xy = km_xy(D.lon[t_idx], D.lat[t_idx], lat0)
    cell = pd.factorize(pd.Series(np.floor(xy[:, 0] / CELL_KM).astype(int)).astype(str) + '_' +
                        pd.Series(np.floor(xy[:, 1] / CELL_KM).astype(int)).astype(str))[0]
    cells = rng.permutation(cell.max() + 1)
    test_cells = cells[:int(round(TEST_FRAC * len(cells)))]
    is_test = np.isin(cell, test_cells)
    d_test, _ = cKDTree(xy[is_test]).query(xy[~is_test])
    cand_mask = np.zeros(len(t_idx), bool)
    cand_mask[np.flatnonzero(~is_test)[d_test > BUFFER_KM]] = True
    test, cand = t_idx[is_test], t_idx[cand_mask]
    foreign = np.concatenate([rng.choice(np.flatnonzero(D.ctry == c), N_FOREIGN_PER, replace=False)
                              for c in D.countries if c != tgt])
    d_far, _ = cKDTree(xy[cand_mask]).query(xy[is_test])
    efar = d_far >= np.quantile(d_far, 2 / 3)
    return dict(foreign=foreign, cand=cand, test=test, allT=np.concatenate([cand, test]), lat0=lat0, efar=efar,
                d_far=d_far.astype(np.float32))


def fold_groups(D, tgt, split, foreign, n_groups=None):
    """外国 10 国分 n_groups 组（分组留出温度用）；由 (tgt, split) 的哈希固定，AE 与 S2 共用。"""
    n_groups = n_groups or N_GROUPS
    others = [c for c in D.countries if c != tgt]
    perm = np.random.default_rng(stable_hash(f'G_groups_{tgt}_s{split}') % 2 ** 32).permutation(len(others))
    gid = {others[j]: i % n_groups for i, j in enumerate(perm)}
    return np.array([gid[c] for c in D.ctry[foreign]])


def preprocess_guidance(Z, clip=CLIP, n_comp=N_PCA):
    """稳健几何：按全部行的均值与标准差做 z，截断到 ±clip，再在全部行上拟合 PCA。
    AE 与 S2 走同一个函数；只看数值，不看列名。"""
    from sklearn.decomposition import PCA
    Z = np.asarray(Z, dtype=np.float64)
    Z = np.clip((Z - Z.mean(0)) / (Z.std(0) + 1e-9), -clip, clip)
    return PCA(n_comp, random_state=0).fit_transform(Z).astype(np.float32)


def preprocess_rawz(Z, n_comp=N_PCA):
    """阶段 A 的做法：z 后直接 PCA（层级 3 的 RAWZ 条件）。"""
    from sklearn.decomposition import PCA
    Z = np.asarray(Z, dtype=np.float64)
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-9)
    return PCA(n_comp, random_state=0).fit_transform(Z).astype(np.float32)


def preprocess_native(Z, kind, n_comp=N_PCA):
    """层级 3 的 NATIVE 条件：'l2' = 行向量 L2 单位化后 PCA（AE 均值组）；'robust' = 稳健 z 后 PCA（S2 均值组）。"""
    from sklearn.decomposition import PCA
    Z = np.asarray(Z, dtype=np.float64)
    if kind == 'l2':
        Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)
        return PCA(n_comp, random_state=0).fit_transform(Z).astype(np.float32)
    return preprocess_guidance(Z, n_comp=n_comp)


def apply_temp(P, s):
    z = s * np.log(np.clip(np.asarray(P, dtype=np.float64), 1e-12, None))
    z -= z.max(1, keepdims=True)
    Q = np.exp(z)
    return Q / Q.sum(1, keepdims=True)


def fit_temperature(P, y):
    """标量温度：在袋外概率上最小化 NLL，边界 (0.05, 20)（与阶段 A 同一目标）。"""
    from scipy.optimize import minimize_scalar
    y = np.asarray(y)

    def nll(s):
        return -np.log(np.clip(apply_temp(P, s)[np.arange(len(y)), y], 1e-300, None)).mean()
    return float(minimize_scalar(nll, bounds=(0.05, 20), method='bounded').x)


def ece_nll(P, y, n_bins=15):
    y = np.asarray(y)
    conf, pred = P.max(1), P.argmax(1)
    bins = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    ece = sum(abs((pred[bins == b] == y[bins == b]).mean() - conf[bins == b].mean()) * (bins == b).mean()
              for b in range(n_bins) if (bins == b).any())
    return float(ece), float(-np.log(np.clip(P[np.arange(len(y)), y], 1e-300, None)).mean())


def train_predict(X_tr, y_tr, w_tr, X_te, threads, seed, trees=None):
    """训练一次，返回 {轮数: 测试集概率 float32}（第 ITERS 轮，num_iteration，不额外训练）。"""
    import lightgbm as lgb
    trees = trees or TREES
    m = lgb.train(lgb_params(threads, seed), lgb.Dataset(X_tr, y_tr, weight=w_tr, free_raw_data=True),
                  num_boost_round=trees)
    return {it: m.predict(X_te, num_iteration=it).astype(np.float32) for it in sorted({*ITERS, trees}) if it <= trees}


def guidance_models(X, foreign, allT, yF, groups, threads, trees=None):
    """一种特征的指导预测：外国全量模型（种子 0）+ 分组留出袋外概率 + 标量温度。AE 与 S2 调用同一函数。"""
    trees = trees or TREES
    cw = class_weights_from(yF)
    full = train_predict(X[foreign], yF, cw[yF], X[allT], threads, 0, trees)
    P_F_oof = np.zeros((len(foreign), 10), np.float32)
    for g in np.unique(groups):
        te = groups == g
        P_F_oof[te] = train_predict(X[foreign[~te]], yF[~te], cw[yF[~te]], X[foreign[te]], threads, 0, trees)[trees]
    temp = fit_temperature(P_F_oof, yF)
    ece, nll = ece_nll(apply_temp(P_F_oof, temp), yF)
    return dict(P_T=apply_temp(full[trees], temp).astype(np.float32), P_raw=full, temp=temp, P_F_oof=P_F_oof,
                ece=ece, nll=nll)


def stage0_compute(D, tgt, split, threads, trees=None):
    sp = make_split(D, tgt, split)
    foreign, allT = sp['foreign'], sp['allT']
    yF = D.y[foreign]
    groups = fold_groups(D, tgt, split, foreign)
    out = dict(sp, groups=groups)
    n_cand = len(sp['cand'])
    for f in ('ae', 's2'):
        gm = guidance_models(D.X[f], foreign, allT, yF, groups, threads, trees)
        out[f'P_T_{f}'] = gm['P_T']
        out[f'temp_{f}'] = gm['temp']
        out[f'P_F_oof_{f}'] = gm['P_F_oof']
        out[f'ece_{f}'], out[f'nll_{f}'] = gm['ece'], gm['nll']
        for it, P in gm['P_raw'].items():  # k=0（只用外国标签）的测试集概率
            out[f'P0test_{f}_i{it}'] = P[n_cand:].astype(np.float16)
        Z = np.vstack([D.X[f][foreign], D.X[f][allT]])
        G = preprocess_guidance(Z)
        out[f'GF_{f}'], out[f'G_{f}'] = G[:len(foreign)], G[len(foreign):]
        out[f'RZ_{f}'] = preprocess_rawz(Z)[len(foreign):]
        mean_cols = [j for j, c in enumerate(D.cols[f]) if c.endswith('_mean')]
        out[f'NT_{f}'] = preprocess_native(Z[:, mean_cols], 'l2' if f == 'ae' else 'robust')[len(foreign):]
    return out


def stage0_one(D, tgt, split, threads):
    path = os.path.join(OUT, 'stage0', f'{tgt}_s{split}.npz')
    if os.path.exists(path):
        return
    t0 = time.time()
    S = stage0_compute(D, tgt, split, threads)
    tmp = path + '.tmp.npz'
    np.savez_compressed(tmp, **S)
    os.replace(tmp, path)
    print(f'[stage0] {tgt} s{split}: cand {len(S["cand"]):,} test {len(S["test"]):,} '
          f'temp ae {S["temp_ae"]:.2f} s2 {S["temp_s2"]:.2f} {time.time() - t0:.0f}s', flush=True)


# ---------------------------------------------------------------- ORACLE 指导（候选池内交叉拟合）
def oracle_compute(D, idx_pool, threads, trees=None, n_folds=N_ORACLE_FOLDS):
    """候选池内 n_folds 折交叉拟合（折由位置随机划分，不看标签）。每折的模型、类别权重、LDA 只用其余折。
    预测：LightGBM（AE⊕S2 原始特征），袋外概率 + 一个标量温度；几何：LDA decision_function（C 列，按类别对齐）。"""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.model_selection import KFold
    trees = trees or TREES
    y = D.y[idx_pool]
    Xr = np.hstack([D.X['ae'][idx_pool], D.X['s2'][idx_pool]])
    Xz = np.clip((Xr - Xr.mean(0)) / (Xr.std(0) + 1e-9), -CLIP, CLIP)
    P_oof = np.zeros((len(y), 10), np.float32)
    G = np.zeros((len(y), 10), np.float32)
    for tr, te in KFold(n_folds, shuffle=True, random_state=0).split(Xr):
        cw = class_weights_from(y[tr])
        P_oof[te] = train_predict(Xr[tr], y[tr], cw[y[tr]], Xr[te], threads, 0, trees)[trees]
        lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto').fit(Xz[tr], y[tr])
        sc = lda.decision_function(Xz[te])
        if sc.ndim == 1:  # 只有两类时 sklearn 返回一列
            sc = np.c_[-sc, sc]
        full = np.full((len(te), 10), sc.min())
        full[:, lda.classes_] = sc
        G[te] = full
    temp = fit_temperature(P_oof, y)
    return dict(P_T=apply_temp(P_oof, temp).astype(np.float32), G=G, temp=temp, P_oof=P_oof)


ORACLE2_MIN_N = 30


def oracle_geometry_v2(G_raw, y_pool, min_n=ORACLE2_MIN_N, clip=CLIP):
    """rev1 修正（ANALYSIS_PLAN_guidance_rev1.md）：原 ORACLE 几何有两处问题——训练折中缺席的类用折内最小值填充
    （该列只编码折号），且没有按方案做逐列 z 截断（稀有类的列尺度极大）。这里只保留候选池中 ≥min_n 个地块的类
    （5 折时每个训练折都至少有约 0.8·min_n 个，列都是真实的 LDA 分数），再按池内均值、标准差逐列 z，截断到 ±clip。"""
    keep = np.bincount(np.asarray(y_pool), minlength=10) >= min_n
    G = np.asarray(G_raw, dtype=np.float64)[:, keep]
    return np.clip((G - G.mean(0)) / (G.std(0) + 1e-9), -clip, clip).astype(np.float32)


def oracle_path(tgt, split):
    return os.path.join(OUT, 'oracle', f'{tgt}_s{split}.npz')


# ---------------------------------------------------------------- 候选池、ctx、选样
def pool_positions(S, split, rep):
    """与阶段 A 相同：从缓冲候选集中抽 N_POOL 个（cand 在 allT 的前 n_cand 个）。"""
    n_cand = len(S['cand'])
    rng = np.random.default_rng(10_000 * split + 100 * rep + 7)
    return np.sort(rng.choice(n_cand, min(N_POOL, n_cand), replace=False))


def _part(A, d):
    A = np.asarray(A, dtype=np.float64)[:, :d]
    return A / np.sqrt(A.var(0).sum() / d)


def _xy_std(xy):
    xy = xy - xy.mean(0)
    return xy / np.sqrt(xy.var(0).sum() / 2)


def guidance_arrays(D, S, pool_pos, cond, d=None, oracle=None):
    """按条件取指导数据：(几何 (N, ·), 预测 (N, C) 或 None, 外国几何或 None)。"""
    g_src, p_src = COND[cond]
    idx = S['allT'][pool_pos]
    xy = km_xy(D.lon[idx], D.lat[idx], float(S['lat0']))
    if g_src is None:
        geom = None
    elif g_src == 'xy':
        geom = xy - xy.mean(0)
    elif g_src == 's2xy':
        geom = np.hstack([_part(S['G_s2'][pool_pos], d), GEO_W_S2XY * _xy_std(xy)])
    elif g_src == 'oracle':
        geom = oracle['G']
    elif g_src == 'oracle2':
        geom = oracle['G2']
    else:
        geom = S[g_src][pool_pos]
    if p_src is None:
        pred = None
    elif p_src == 'oracle':
        pred = oracle['P_T']
    else:
        pred = S[p_src][pool_pos]
    F = S[F_GEOM[g_src]] if g_src in F_GEOM else None
    return geom, pred, F


def build_ctx_guided(D, S, pool_pos, cond, d=None, oracle=None, with_foreign=False):
    """ctx 只保留 N、C、T_ae 槽（指导几何）、F_y、P_T（指导预测）；其余与指导有关的字段一律为 None。"""
    geom, pred, F = guidance_arrays(D, S, pool_pos, cond, d, oracle)
    return B.Ctx(N=len(pool_pos), C=10, T_ae=None if geom is None else np.asarray(geom, dtype=np.float32),
                 T_s2=None, T_xy=None, T_nuts3=None, F_ae=F if with_foreign else None, F_s2=None, F_xy=None,
                 F_y=D.y[S['foreign']], F_country=None,
                 P_T=None if pred is None else np.asarray(pred, dtype=np.float64))


def select_one(D, S, pool_pos, cond, method, k, seed, oracle=None):
    """一次选样：返回 (候选池内位置, meta)。种子只由单元决定，与条件名、方法名、分类器无关。"""
    d = METHOD_D.get(method)
    ctx = build_ctx_guided(D, S, pool_pos, cond, d, oracle, with_foreign=(method == 'kcenter_foreign'))
    hp = dict(G_HP[method])
    if cond == 'S2XY':
        hp['d'] = d + 2  # 使 X_of 取到全部 d+2 列（S2 前 d 维 + 两列坐标）
    sel_pos = B.REGISTRY[method](ctx, k, seed, **hp)
    return sel_pos, ctx.meta.get(method, {})


def sel_hash(sel):
    return hashlib.md5(np.sort(np.asarray(sel, dtype=np.int64)).tobytes()).hexdigest()[:16]


# ---------------------------------------------------------------- 评估
def prior_correct(P, pi_new, pi_train):
    w = np.divide(pi_new, pi_train, out=np.zeros_like(pi_new), where=pi_train > 0)
    q = P * w
    s = q.sum(1, keepdims=True)
    return np.where(s > 0, q / np.where(s > 0, s, 1), P)


def _scores(yT, ph, big, mask=None):
    from sklearn.metrics import f1_score
    if mask is not None:
        yT, ph = yT[mask], ph[mask]
    rec = np.array([(ph[yT == c] == c).mean() if (yT == c).any() else np.nan for c in range(10)])
    return (100 * float((ph == yT).mean()), 100 * float(np.nanmean(rec[big])),
            100 * float(f1_score(yT, ph, labels=big, average='macro', zero_division=0)), rec)


def evaluate(P_it, yT, big, efar, pi_train, pi_pool, pi_des):
    """P_it：{轮数: 测试集概率}。主指标取最后一轮；中途轮数只算 P0。pi_des=None 时 P1 用零标签 EM。"""
    from check3_priors import em_prior
    last = max(P_it)
    P = P_it[last].astype(np.float64)
    P /= P.sum(1, keepdims=True)
    ph = P.argmax(1)
    oa, ba, f1, rec = _scores(yT, ph, big)
    out = dict(oa=oa, ba_ge30=ba, f1_ge30=f1, **{f'rec{c}': rec[c] for c in range(10)})
    for it, Pi in P_it.items():
        o, b, f, _ = _scores(yT, Pi.argmax(1), big)
        out.update({f'oa_i{it}': o, f'ba_i{it}': b, f'f1_i{it}': f})
    src = 'design'
    if pi_des is None:
        pi_des, src = em_prior(P, pi_train)[0], 'em'
    for tag, pin in (('P1', pi_des), ('P2', pi_pool)):
        o, b, _, _ = _scores(yT, prior_correct(P, pin, pi_train).argmax(1), big)
        out[f'oa_{tag}'], out[f'ba_{tag}'] = o, b
    o, b, _, _ = _scores(yT, ph, big, efar)
    out.update(prior_src=src, prior_L1=float(np.abs(pi_des - pi_pool).sum()), oa_far=o, ba_far=b,
               n_far=int(efar.sum()))
    return out


# ---------------------------------------------------------------- 一个单元
def run_unit(D, S, tgt, split, rep, tier, threads, out=None, trees=None, log=print, conds=None):
    """一个（层级, 国家, 划分, 重复）：对 jobs_for 给出的全部 (条件, 方法, k, 分类器) 选样、训练、评估。
    选样按 (条件, 方法, k) 保存；训练按 (分类器, k, 选样哈希) 缓存，跨条件复用。返回全部行。"""
    out = out or OUT
    trees = trees or TREES
    for sub in ('sel', 'probs', 'oracle'):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    unit = f'{tgt}_s{split}_r{rep}'
    pool_pos = pool_positions(S, split, rep)
    allT, test, fr = S['allT'], S['test'], S['foreign']
    idx_pool = allT[pool_pos]
    yT = D.y[test]
    big = np.flatnonzero(np.bincount(yT, minlength=10) >= MIN_N)
    efar = S['efar'].astype(bool)
    cw = class_weights_from(D.y[fr])
    pi_pool = np.bincount(D.y[idx_pool], minlength=10) / len(idx_pool)
    seed = 1000 * split + rep
    rows_path = os.path.join(out, f'rows_t{tier}_{unit}.csv')
    rows = pd.read_csv(rows_path).to_dict('records') if os.path.exists(rows_path) else []
    done = {(r_['cond'], r_['method'], r_['k'], r_['clf']) for r_ in rows}
    J = [j for j in jobs_for(tier, split, rep) if conds is None or j[0] in conds]
    oracle = None
    if any(c in ('ORACLE', 'ORACLE2') for c, *_ in J):
        op = os.path.join(out, 'oracle', f'{tgt}_s{split}.npz')
        if not os.path.exists(op):
            t0 = time.time()
            o = oracle_compute(D, idx_pool, threads, trees)
            np.savez_compressed(op + '.tmp.npz', **o)
            os.replace(op + '.tmp.npz', op)
            log(f'[oracle] {tgt} s{split}: temp {o["temp"]:.2f} {time.time() - t0:.0f}s')
        oracle = dict(np.load(op))
        oracle['G2'] = oracle_geometry_v2(oracle['G'], D.y[idx_pool])

    def pi_train_of(sel, r_mult):
        ys = D.y[sel]
        w = np.r_[cw[D.y[fr]], r_mult * cw[ys]]
        return np.bincount(np.r_[D.y[fr], ys], weights=w, minlength=10) / w.sum()

    # k=0：阶段 0 外国全量模型（只在层级 1 记录）
    if tier == 1:
        for clf in ('s2', 'ae'):
            if ('none', 'foreign_only', 0, clf) in done:
                continue
            pre = f'P0test_{clf}_i'
            P_it = {int(key[len(pre):]): S[key].astype(np.float32) for key in S if key.startswith(pre)}
            m = evaluate(P_it, yT, big, efar, pi_train_of(np.array([], int), 0.0), pi_pool, None)
            rows.append(dict(cond='none', geo_src='none', pred_src='none', method='foreign_only', clf=clf, k=0,
                             target=tgt, split=split, rep=rep, sel_hash='', t_select=0.0, t_train=0.0,
                             cache_hit=True, n_classes_sel=0, comp_L1=np.nan, **m))
            save_csv(rows, rows_path)

    for cond, method, k, clfs in J:
        todo = [c for c in clfs if (cond, method, k, c) not in done]
        if not todo:
            continue
        sp = os.path.join(out, 'sel', f'{unit}_{cond}_{method}_k{k}.npz')
        t0 = time.time()
        if os.path.exists(sp):
            z = np.load(sp)
            sel_pos, pi_sel = z['sel_pos'], z['pi_sel']
        else:
            sel_pos, meta = select_one(D, S, pool_pos, cond, method, k, seed, oracle)
            pi_sel = np.asarray(meta['pi'])[sel_pos] if 'pi' in meta else np.array([])
            np.savez_compressed(sp, sel_pos=sel_pos, sel=idx_pool[sel_pos], pi_sel=pi_sel)
        t_sel = time.time() - t0
        sel = idx_pool[sel_pos]
        h = sel_hash(sel)
        ys = D.y[sel]
        pi_des = None
        if len(pi_sel):
            ipw = 1 / pi_sel
            pi_des = np.bincount(ys, weights=ipw, minlength=10) / ipw.sum()  # Hájek
        for clf in todo:
            t1 = time.time()
            pp = os.path.join(out, 'probs', f'{unit}_{clf}_k{k}_{h}.npz')
            hit = os.path.exists(pp)
            if hit:
                z = np.load(pp)
                P_it = {int(key[1:]): z[key].astype(np.float32) for key in z.files}
            else:
                X = D.fusion() if clf == 'fus' else D.X[clf]
                w_tr = np.r_[cw[D.y[fr]], R_MULT * cw[ys]]
                P_it = train_predict(np.vstack([X[fr], X[sel]]), np.r_[D.y[fr], ys], w_tr, X[test], threads, seed,
                                     trees)
                np.savez_compressed(pp + '.tmp.npz', **{f'i{it}': P for it, P in P_it.items()})  # float32，缓存命中时指标逐位相同
                os.replace(pp + '.tmp.npz', pp)
            m = evaluate(P_it, yT, big, efar, pi_train_of(sel, R_MULT), pi_pool, pi_des)
            g_src, p_src = COND[cond]
            rows.append(dict(cond=cond, geo_src=g_src or 'none', pred_src=p_src or 'none', method=method, clf=clf,
                             k=k, target=tgt, split=split, rep=rep, sel_hash=h, t_select=round(t_sel, 2),
                             t_train=round(time.time() - t1, 1), cache_hit=hit, n_classes_sel=len(np.unique(ys)),
                             comp_L1=float(np.abs(np.bincount(ys, minlength=10) / k - pi_pool).sum()), **m))
            save_csv(rows, rows_path)
            log(f'{unit} t{tier} {cond} {method} k{k} {clf} sel {t_sel:.0f}s train {time.time() - t1:.0f}s'
                f'{" (cache)" if hit else ""}')
    return rows


# ---------------------------------------------------------------- 进程与主程序
def init_worker(conds=None):
    run_task.D = GData()
    run_task.conds = conds


def run_task(args):
    kind, tier, tgt, split, rep, threads, worker = args
    D = run_task.D
    if kind == 'stage0':
        stage0_one(D, tgt, split, threads)
        return f'stage0 {tgt} s{split}'
    S = dict(np.load(os.path.join(OUT, 'stage0', f'{tgt}_s{split}.npz')))
    rows = run_unit(D, S, tgt, split, rep, tier, threads, log=lambda s: print(f'[w{worker}] {s}', flush=True),
                    conds=run_task.conds)
    return f't{tier} {tgt} s{split} r{rep}: {len(rows)} rows'


def unit_order():
    """方案第 9 节：先 11 国的 (s0, r0)，再 (s1, r0)，再 (s0, r1)，再 (s1, r1)。"""
    return [(t, s, r) for s, r in ((0, 0), (1, 0), (0, 1), (1, 1)) for t in COUNTRIES_G]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['stage0', 'run', 'all'])
    ap.add_argument('--tier', type=int, default=1)
    ap.add_argument('--jobs', type=int, default=5)
    ap.add_argument('--conds', type=str, default=None, help='逗号分隔，只跑这些条件（如层级 3 的 RAWZ_AE,RAWZ_S2）')
    a = ap.parse_args()
    for sub in ('stage0', 'sel', 'probs', 'oracle'):
        os.makedirs(os.path.join(OUT, sub), exist_ok=True)
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), PLAN), 'rb') as f:
        print(f'{PLAN} sha256[:16] = {hashlib.sha256(f.read()).hexdigest()[:16]}；TREES={TREES}，ITERS={ITERS}，'
              f'温度分组 {N_GROUPS}，输出 {OUT}', flush=True)
    check_geo_w()
    GData.build_cache()
    threads = max(1, 10 // a.jobs)
    phases = []
    if a.mode in ('stage0', 'all'):
        phases.append([('stage0', 0, t, s, 0) for t in COUNTRIES_G for s in SPLITS])
    if a.mode == 'run':
        phases.append([('run', a.tier, t, s, r) for t, s, r in unit_order()])
    if a.mode == 'all':
        phases += [[('run', tier, t, s, r) for t, s, r in unit_order()] for tier in (1, 2)]
    conds = set(a.conds.split(',')) if a.conds else None
    with mp.Pool(a.jobs, initializer=init_worker, initargs=(conds,)) as pool:
        for ph in phases:
            jobs = [(*j, threads, i % a.jobs) for i, j in enumerate(ph)]
            for msg in pool.imap_unordered(run_task, jobs):
                print(f'task done: {msg}', flush=True)


if __name__ == '__main__':
    main()
