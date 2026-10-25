# -*- coding: utf-8 -*-
"""
迁移版 Betti 分配的覆盖诊断（不训练、不读标签）。

规格：label_expansion/BETTI_TRANSFER_DIAG_SPEC_2026-09-28.md（sha256 前 16 位 4c8da222b77b0f84，运行前写定）。

做什么：
  在 G 的阶段 0 几何（G_{f}：allT 行；GF_{f}：外国池 5 万行；同一 PCA 32 维）上，
  对每个单元（11 国 × 划分 {0,1} × 重复 {0,1}）、每个空间 f ∈ {ae, s2}：
    1. 候选池 = guidance_G.pool_positions(S, split, rep)（3 万个，与 G 选样相同）；
    2. KMeans(K, n_init=10, random_state=0) 在候选池上分组（K ∈ {8, 16}，每单元/空间/K 只拟合一次）；
    3. 外国样本归入最近质心，距离 ≤ r_g（组内候选成员到质心距离的 q95 / q80 / 不设阈值）才计入，
       m_g 为计入数，n⁰_g = w·m_g（w ∈ {1/10, 1/50, 1/200}）；
    4. 贪心分配 k ∈ {50, 200, 400} 个名额（边际增益 γ_g[(n⁰+n+1)^(-1/2) − (n⁰+n+2)^(-1/2)]，
       n_g < 容量，平局取编号小的组），对照 (a) n⁰=0 的同一贪心（检验 2 试点做法）、(b) 比例分配 γ_g·k；
    5. 指标：TV_a（迁移 vs 无初始，主指标）、TV_b（迁移 vs 比例）、外国计入率、覆盖不均（c_g 的变异系数）、
       得 0 名额的组数、单组最大名额占比；另每单元/空间一次的分离比
       （外国→最近候选池地块距离中位数 / 候选池内最近邻距离中位数）。

数据使用（硬性）：
  - 只读 npz 中的 cand、allT 的长度（只读 npy 头）、G_ae/G_s2 的前 n_cand 行（流式只解压到候选行为止，
    测试行的字节不读）、GF_ae/GF_s2。不碰任何 y、概率、温度等键。
  - 不训练模型，不看任何精度结果。
  - 线程 ≤ 2（在 import numpy/sklearn 之前设定环境变量）。

用法：
  python -B betti_transfer_diag.py            # 全部 44 单元，写 diag_rows.csv / diag_alloc.csv / diag_summary.txt
  python -B betti_transfer_diag.py --smoke --out <目录> [--units AT_s0_r0,SE_s1_r1]   # 冒烟测试
"""
import os
import sys

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_v] = '2'
sys.dont_write_bytecode = True  # 不在项目目录生成 __pycache__

import argparse  # noqa: E402
import hashlib  # noqa: E402
import time  # noqa: E402
import zipfile  # noqa: E402

LE_DIR = ('/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/'
          'Alpha_Crops/label_expansion')
SPEC = os.path.join(LE_DIR, 'BETTI_TRANSFER_DIAG_SPEC_2026-09-28.md')
SPEC_HASH16 = '4c8da222b77b0f84'
STAGE0 = '/Users/perng_mini/AlphaCrops_local/results/guidance_G/stage0'
OUT_DIR = '/Users/perng_mini/AlphaCrops_local/verify/betti_transfer'
sys.path.insert(0, LE_DIR)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402

from guidance_G import pool_positions, N_POOL  # noqa: E402  与 G 选样完全相同的候选池位置

# ---------------------------------------------------------------- 设定（规格）
COUNTRIES = ['AT', 'BE', 'DE', 'EE', 'LT', 'LV', 'PT', 'RO', 'SE', 'SI', 'SK']
SPLITS, REPS = [0, 1], [0, 1]
SPACES = ['ae', 's2']
KS = [8, 16]
THRS = ['q95', 'q80', 'none']
THR_Q = {'q95': 0.95, 'q80': 0.80, 'none': None}
WS = {'1/10': 1 / 10, '1/50': 1 / 50, '1/200': 1 / 200}
BUDGETS = [50, 200, 400]
N_FOREIGN = 50_000
MAIN = dict(K=8, thr='q95', w='1/50', k=200)
KM_KW = dict(n_init=10, random_state=0)


def spec_hash16():
    with open(SPEC, 'rb') as fh:
        return hashlib.sha256(fh.read()).hexdigest()[:16]


# ---------------------------------------------------------------- 读取（只读允许的键与行）
def _npy_header(fh):
    ver = np.lib.format.read_magic(fh)
    if ver == (1, 0):
        return np.lib.format.read_array_header_1_0(fh)
    return np.lib.format.read_array_header_2_0(fh)


def npy_shape(path, key):
    """只读 npz 成员的 npy 头，返回 shape（不解压数据）。"""
    with zipfile.ZipFile(path) as zf, zf.open(key + '.npy') as fh:
        shape, _, _ = _npy_header(fh)
    return shape


def npy_head_rows(path, key, n_rows):
    """流式读取 npz 成员的前 n_rows 行（C 顺序），读到为止，后面的字节（测试行）不解压。"""
    with zipfile.ZipFile(path) as zf, zf.open(key + '.npy') as fh:
        shape, fortran, dtype = _npy_header(fh)
        assert not fortran and len(shape) == 2 and n_rows <= shape[0], (key, shape, fortran)
        nbytes = n_rows * shape[1] * dtype.itemsize
        buf = fh.read(nbytes)
        assert len(buf) == nbytes
    return np.frombuffer(buf, dtype=dtype).reshape(n_rows, shape[1]), shape


def npy_full(path, key):
    with np.load(path, allow_pickle=False) as Z:
        return np.asarray(Z[key])


def load_file(ctry, split):
    """返回 dict(cand, n_allT, G={f: 候选行}, GF={f: 外国池})。"""
    path = os.path.join(STAGE0, f'{ctry}_s{split}.npz')
    cand = npy_full(path, 'cand')
    n_cand = len(cand)
    n_allT = npy_shape(path, 'allT')[0]
    out = dict(cand=cand, n_cand=n_cand, n_allT=n_allT, G={}, GF={})
    for f in SPACES:
        Gc, shp = npy_head_rows(path, f'G_{f}', n_cand)
        assert shp[0] == n_allT and shp[1] == 32, (f, shp, n_allT)
        out['G'][f] = Gc
        GF = npy_full(path, f'GF_{f}')
        assert GF.shape == (N_FOREIGN, 32), GF.shape
        out['GF'][f] = GF
    return out


# ---------------------------------------------------------------- 分配
def greedy_alloc(gamma, cap, k, n0=None):
    """逐个名额贪心（与 pilot_betti.allocate_parcels 相同，只多了初始样本 n⁰）。平局时 argmax 取编号小的组。"""
    gamma = np.asarray(gamma, dtype=np.float64)
    cap = np.asarray(cap, dtype=np.float64)
    n0 = np.zeros(len(gamma)) if n0 is None else np.asarray(n0, dtype=np.float64)
    n = np.zeros(len(gamma))
    for _ in range(int(min(k, cap.sum()))):
        gain = np.where(n < cap, gamma * ((n0 + n + 1) ** -0.5 - (n0 + n + 2) ** -0.5), -np.inf)
        n[int(np.argmax(gain))] += 1
    return n.astype(int)


def tv(a, b, k):
    return 0.5 * np.abs(np.asarray(a, float) - np.asarray(b, float)).sum() / k


# ---------------------------------------------------------------- 距离
def nn_dist(Q, P, exclude_self=False, chunk=512):
    """Q 每行到 P 的最近邻欧氏距离（分块暴力，float64）。exclude_self 时 Q 与 P 为同一数组，去掉自身。"""
    Q = np.asarray(Q, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    Pn = (P * P).sum(1)
    out = np.empty(len(Q))
    for i0 in range(0, len(Q), chunk):
        q = Q[i0:i0 + chunk]
        d2 = (q * q).sum(1)[:, None] + Pn[None, :] - 2.0 * (q @ P.T)
        if exclude_self:
            r = np.arange(len(q))
            d2[r, i0 + r] = np.inf
        out[i0:i0 + len(q)] = np.sqrt(np.maximum(d2.min(1), 0.0))
    return out


def sq_dist_to_centers(X, C):
    d2 = (X * X).sum(1)[:, None] + (C * C).sum(1)[None, :] - 2.0 * (X @ C.T)
    return np.maximum(d2, 0.0)


# ---------------------------------------------------------------- 单个单元 × 空间
def run_unit_space(Xp, XF, unit, space, log=print):
    """Xp：候选池几何 (30000, 32)；XF：外国池几何 (50000, 32)。返回 (rows, alloc_rows, info)。"""
    t0 = time.time()
    Xp = np.asarray(Xp, dtype=np.float64)
    XF = np.asarray(XF, dtype=np.float64)
    N, NF = len(Xp), len(XF)

    # 分离比（每单元、每空间一次）
    d_fp = nn_dist(XF, Xp)
    d_pp = nn_dist(Xp, Xp, exclude_self=True)
    med_fp, med_pp = float(np.median(d_fp)), float(np.median(d_pp))
    sep = dict(med_d_foreign_pool=med_fp, med_d_pool_pool=med_pp, sep_ratio=med_fp / med_pp)
    t_nn = time.time() - t0

    rows, arows, info = [], [], dict(sep, t_nn=t_nn)
    for K in KS:
        t1 = time.time()
        km = KMeans(n_clusters=K, **KM_KW).fit(Xp)  # 同一单元/空间/K 只拟合一次
        C = km.cluster_centers_.astype(np.float64)
        lab = km.labels_
        d2p = sq_dist_to_centers(Xp, C)
        lab_chk = float(np.mean(np.argmin(d2p, 1) == lab))
        d_own = np.sqrt(d2p[np.arange(N), lab])
        cap = np.bincount(lab, minlength=K)
        gamma = cap / N
        # 外国归组与距离（只算一次）
        d2f = sq_dist_to_centers(XF, C)
        gstar = np.argmin(d2f, 1)
        dstar = np.sqrt(d2f[np.arange(NF), gstar])
        # 对照：无初始（贪心）与比例
        n_none = {k: greedy_alloc(gamma, cap, k) for k in BUDGETS}
        n_prop = {k: gamma * k for k in BUDGETS}
        info[f'K{K}_label_consistency'] = lab_chk
        info[f'K{K}_min_gamma'] = float(gamma.min())
        info[f'K{K}_t'] = time.time() - t1
        for thr in THRS:
            q = THR_Q[thr]
            if q is None:
                r_g = np.full(K, np.inf)
            else:
                r_g = np.array([np.quantile(d_own[lab == g], q) for g in range(K)])
            counted = dstar <= r_g[gstar]
            m = np.bincount(gstar[counted], minlength=K).astype(np.float64)
            sm = m.sum()
            if sm > 0:
                c_g = (m / sm) / gamma
                cov_cv = float(np.std(c_g) / np.mean(c_g))
            else:
                c_g = np.full(K, np.nan)
                cov_cv = np.nan
            for wname, w in WS.items():
                n0 = w * m
                for k in BUDGETS:
                    n_tr = greedy_alloc(gamma, cap, k, n0)
                    assert n_tr.sum() == k and n_none[k].sum() == k
                    rows.append(dict(
                        unit=unit['name'], country=unit['country'], split=unit['split'], rep=unit['rep'],
                        space=space, K=K, thr=thr, w=wname, k=k,
                        TV_a=tv(n_tr, n_none[k], k), TV_b=tv(n_tr, n_prop[k], k),
                        TV_none_prop=tv(n_none[k], n_prop[k], k),
                        foreign_counted_rate=sm / NF, sum_m=sm, sum_n0=w * sm, cov_cv=cov_cv,
                        n_zero_groups=int((n_tr == 0).sum()), max_share=float(n_tr.max() / k),
                        n_zero_groups_none=int((n_none[k] == 0).sum()), max_share_none=float(n_none[k].max() / k),
                        n_pool=N, n_foreign=NF, **sep))
                    for g in range(K):
                        arows.append(dict(
                            unit=unit['name'], country=unit['country'], split=unit['split'], rep=unit['rep'],
                            space=space, K=K, thr=thr, w=wname, k=k, group=g,
                            gamma=gamma[g], capacity=int(cap[g]), r_g=(np.nan if q is None else r_g[g]),
                            m_g=int(m[g]), n0_g=n0[g], c_g=c_g[g],
                            n_transfer=int(n_tr[g]), n_none=int(n_none[k][g]), n_prop=n_prop[k][g]))
    info['t_total'] = time.time() - t0
    log(f'  [{unit["name"]} {space}] 分离比 {sep["sep_ratio"]:.3f} '
        f'(外国→池 {med_fp:.3f} / 池内 {med_pp:.3f}); '
        f'标签一致性 K8 {info["K8_label_consistency"]:.4f} K16 {info["K16_label_consistency"]:.4f}; '
        f'NN {t_nn:.0f}s KM8 {info["K8_t"]:.0f}s KM16 {info["K16_t"]:.0f}s', flush=True)
    return rows, arows, info


def all_units():
    return [dict(name=f'{c}_s{s}_r{r}', country=c, split=s, rep=r) for c in COUNTRIES for s in SPLITS for r in REPS]


def run(units, log=print):
    rows, arows = [], []
    by_file = {}
    for u in units:
        by_file.setdefault((u['country'], u['split']), []).append(u)
    for (c, s), us in by_file.items():
        t0 = time.time()
        F = load_file(c, s)
        S = {'cand': F['cand']}  # pool_positions 只用 len(S['cand'])
        log(f'[{c}_s{s}] n_cand {F["n_cand"]:,} n_allT {F["n_allT"]:,} 读取 {time.time() - t0:.0f}s', flush=True)
        for u in us:
            pos = pool_positions(S, u['split'], u['rep'])
            assert len(pos) == min(N_POOL, F['n_cand']) and pos.max() < F['n_cand']
            for f in SPACES:
                r, a, _ = run_unit_space(F['G'][f][pos], F['GF'][f], u, f, log)
                rows += r
                arows += a
        del F
    return pd.DataFrame(rows), pd.DataFrame(arows)


# ---------------------------------------------------------------- 汇总
def verdict(med):
    if med < 0.10:
        return '退化'
    if med < 0.20:
        return '弱偏离'
    return '明显偏离'


def q3(x):
    x = np.asarray(x, float)
    return np.median(x), np.percentile(x, 25), np.percentile(x, 75)


def sel(df, **kw):
    m = np.ones(len(df), bool)
    for c, v in kw.items():
        m &= (df[c] == v).to_numpy()
    return df[m]


def summarize(R, sha16, extra_lines=()):
    L = []
    P = L.append
    P(f'规格文件 sha256[:16] = {sha16}（登记值 {SPEC_HASH16}）')
    assert sha16 == SPEC_HASH16
    P('规格：label_expansion/BETTI_TRANSFER_DIAG_SPEC_2026-09-28.md')
    P(f'生成时间：{time.strftime("%Y-%m-%d %H:%M:%S")}；行数 {len(R)}；单元数 {R["unit"].nunique()}；'
      f'空间 {sorted(R["space"].unique())}')
    P('数据使用：只读 cand、allT 长度（npy 头）、G_* 的候选行（流式读到 n_cand 行为止）与 GF_*；未读任何标签/概率；'
      '未读测试行几何；未训练；未看精度结果。')
    P('')
    main = sel(R, K=MAIN['K'], thr=MAIN['thr'], w=MAIN['w'], k=MAIN['k'])

    P('=' * 100)
    P('一、主设定（K=8、95% 阈值、w=1/50、k=200）：每个空间 44 单元的 TV_a 与判定')
    P('   判定：TV_a 中位数 <0.10 退化；0.10–0.20 弱偏离；≥0.20 明显偏离（值得考虑训练，需作者批准）')
    P('-' * 100)
    P(f'{"空间":<5}{"n":>4}{"TV_a 中位数":>12}{"[Q25, Q75]":>18}{"均值":>8}{"最小":>8}{"最大":>8}'
      f'{"≥0.20 单元数":>14}{"判定":>8}')
    for f in SPACES:
        x = sel(main, space=f)['TV_a'].to_numpy()
        md, a, b = q3(x)
        P(f'{f:<5}{len(x):>4}{md:>12.4f}{f"[{a:.4f}, {b:.4f}]":>18}{x.mean():>8.4f}{x.min():>8.4f}{x.max():>8.4f}'
          f'{int((x >= 0.20).sum()):>14}{verdict(md):>8}')
    P('')
    P('主设定下的其他指标（44 单元中位数 [Q25, Q75]）：')
    for f in SPACES:
        d = sel(main, space=f)
        parts = []
        for col, nm in [('TV_b', 'TV_b(迁移 vs 比例)'), ('TV_none_prop', 'TV(无初始 vs 比例)'),
                        ('foreign_counted_rate', '外国计入率'), ('sum_n0', 'Σn⁰'), ('cov_cv', '覆盖不均 CV'),
                        ('n_zero_groups', '0 名额组数'), ('max_share', '单组最大占比'),
                        ('n_zero_groups_none', '0 名额组数(无初始)'), ('max_share_none', '单组最大占比(无初始)')]:
            md, a, b = q3(d[col].dropna())
            parts.append(f'{nm} {md:.4g} [{a:.4g}, {b:.4g}]')
        P(f'  {f}: ' + '；'.join(parts))
    P('')

    countries = [c for c in COUNTRIES if c in set(R['country'])]
    P('=' * 100)
    P('二、逐国主设定 TV_a（每国 4 单元：划分 {0,1} × 重复 {0,1}）')
    P('-' * 100)
    P(f'{"国家":<5}{"ae 中位数":>10}{"ae 4 单元":>34}{"s2 中位数":>10}{"s2 4 单元":>34}{"ae−s2 均值":>12}')
    for c in countries:
        va = sel(main, space='ae', country=c).sort_values(['split', 'rep'])['TV_a'].to_numpy()
        vs = sel(main, space='s2', country=c).sort_values(['split', 'rep'])['TV_a'].to_numpy()
        P(f'{c:<5}{np.median(va):>10.4f}{" ".join(f"{v:.3f}" for v in va):>34}'
          f'{np.median(vs):>10.4f}{" ".join(f"{v:.3f}" for v in vs):>34}{np.mean(va - vs):>12.4f}')
    P('')

    P('=' * 100)
    P('三、AE − S2 配对差（主设定；逐单元配对 TV_a(ae) − TV_a(s2)，先平均到国家，再对 11 国求均值与 95% CI，t 分布 df=10）')
    P('-' * 100)
    pa = sel(main, space='ae').set_index('unit')['TV_a']
    ps = sel(main, space='s2').set_index('unit')['TV_a']
    diff = (pa - ps.reindex(pa.index))
    ctry_of = sel(main, space='ae').set_index('unit')['country']
    dc = diff.groupby(ctry_of).mean().reindex(countries)
    n = len(dc)
    mu, sd = dc.mean(), dc.std(ddof=1)
    tq = stats.t.ppf(0.975, n - 1)
    lo, hi = mu - tq * sd / np.sqrt(n), mu + tq * sd / np.sqrt(n)
    P(f'国家层差值：' + '，'.join(f'{c} {v:+.4f}' for c, v in dc.items()))
    P(f'均值 {mu:+.4f}，95% CI [{lo:+.4f}, {hi:+.4f}]（n={n} 国，df={n - 1}，t={tq:.3f}，sd={sd:.4f}）；'
      f'单元层差值中位数 {diff.median():+.4f}，ae>s2 的单元 {int((diff > 0).sum())}/{len(diff)}')
    P('')

    P('=' * 100)
    P('四、分离比（外国样本到最近候选池地块的距离中位数 / 候选池内最近邻距离中位数；每单元、每空间一次）')
    P('-' * 100)
    sep = R.drop_duplicates(['unit', 'space'])[['unit', 'country', 'space', 'sep_ratio', 'med_d_foreign_pool',
                                                 'med_d_pool_pool']]
    for f in SPACES:
        d = sel(sep, space=f)
        md, a, b = q3(d['sep_ratio'])
        P(f'  {f}: 分离比中位数 {md:.3f} [Q25 {a:.3f}, Q75 {b:.3f}]，范围 [{d["sep_ratio"].min():.3f}, '
          f'{d["sep_ratio"].max():.3f}]；外国→池距离中位数的中位数 {d["med_d_foreign_pool"].median():.3f}，'
          f'池内最近邻距离中位数的中位数 {d["med_d_pool_pool"].median():.3f}')
    P('  逐国（4 单元中位数）：')
    P(f'  {"国家":<5}{"ae":>8}{"s2":>8}')
    for c in countries:
        P(f'  {c:<5}{sel(sep, space="ae", country=c)["sep_ratio"].median():>8.3f}'
          f'{sel(sep, space="s2", country=c)["sep_ratio"].median():>8.3f}')
    P('')

    P('=' * 100)
    P('五、敏感性（只作描述，不改变判定）')
    P('5a. 从主设定出发每次只改一个因素：TV_a 中位数 [Q25, Q75]（44 单元）')
    P('-' * 100)
    P(f'{"改动":<22}{"ae TV_a":>28}{"s2 TV_a":>28}{"ae 计入率":>12}{"s2 计入率":>12}')
    variants = [('主设定', {})] + [(f'K={v}', dict(K=v)) for v in KS if v != MAIN['K']] + \
               [(f'阈值={v}', dict(thr=v)) for v in THRS if v != MAIN['thr']] + \
               [(f'w={v}', dict(w=v)) for v in WS if v != MAIN['w']] + \
               [(f'k={v}', dict(k=v)) for v in BUDGETS if v != MAIN['k']]
    for nm, ch in variants:
        s = {**MAIN, **ch}
        cells = []
        rates = []
        for f in SPACES:
            d = sel(R, space=f, **s)
            md, a, b = q3(d['TV_a'])
            cells.append(f'{md:.4f} [{a:.4f}, {b:.4f}]')
            rates.append(d['foreign_counted_rate'].median())
        P(f'{nm:<22}{cells[0]:>28}{cells[1]:>28}{rates[0]:>12.4f}{rates[1]:>12.4f}')
    P('')
    P('5b. 全部设定：TV_a 中位数 [Q25, Q75]、TV_b 中位数、外国计入率中位数、Σn⁰ 中位数、判定标签（仅描述）')
    P('-' * 100)
    P(f'{"空间":<5}{"K":>4}{"阈值":>6}{"w":>7}{"k":>5}{"TV_a 中位数":>12}{"[Q25, Q75]":>18}{"TV_b":>8}'
      f'{"TV(无初始,比例)":>16}{"计入率":>8}{"Σn⁰":>9}{"0名额组":>8}{"最大占比":>9}{"标签":>8}')
    g = R.groupby(['space', 'K', 'thr', 'w', 'k'], sort=False)
    order = [(f, K, t, w, k) for f in SPACES for K in KS for t in THRS for w in WS for k in BUDGETS]
    for key in order:
        d = g.get_group(key)
        md, a, b = q3(d['TV_a'])
        f, K, t, w, k = key
        P(f'{f:<5}{K:>4}{t:>6}{w:>7}{k:>5}{md:>12.4f}{f"[{a:.4f}, {b:.4f}]":>18}{d["TV_b"].median():>8.4f}'
          f'{d["TV_none_prop"].median():>16.4f}{d["foreign_counted_rate"].median():>8.4f}'
          f'{d["sum_n0"].median():>9.1f}{d["n_zero_groups"].median():>8.1f}{d["max_share"].median():>9.3f}'
          f'{verdict(md):>8}')
    P('')
    P('5c. 各空间在全部 54 个（K × 阈值 × w × k）设定中 TV_a 中位数落入各判定区间的个数')
    for f in SPACES:
        meds = [q3(g.get_group(key)['TV_a'])[0] for key in order if key[0] == f]
        cnt = pd.Series([verdict(m) for m in meds]).value_counts()
        P(f'  {f}: ' + '，'.join(f'{v} {int(cnt.get(v, 0))}' for v in ['退化', '弱偏离', '明显偏离'])
          + f'（共 {len(meds)}）')
    P('')
    if extra_lines:
        P('=' * 100)
        P('六、运行记录')
        L.extend(extra_lines)
    return '\n'.join(L) + '\n'


def write_atomic_csv(df, path):
    tmp = path + '.tmp'
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def write_atomic_text(text, path):
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as fh:
        fh.write(text)
    os.replace(tmp, path)


# ---------------------------------------------------------------- 冒烟：手算小例子
def hand_checks(log=print):
    big = np.array([10 ** 9] * 3)
    a = greedy_alloc([0.5, 0.3, 0.2], big, 100, [400, 20, 0])
    b = greedy_alloc([0.5, 0.3, 0.2], big, 100)
    log(f'手算例 γ=[0.5,0.3,0.2], n⁰=[400,20,0], k=100 → {a.tolist()}（应为 [0, 48, 52]）')
    log(f'手算例 γ=[0.5,0.3,0.2], n⁰=0, k=100 → {b.tolist()}（应为 [45, 31, 24]）')
    assert a.tolist() == [0, 48, 52], a
    assert b.tolist() == [45, 31, 24], b
    # 与 pilot_betti.allocate_parcels 逐位比较（n⁰=0，含容量约束）
    from pilot_betti import allocate_parcels
    rng = np.random.default_rng(0)
    for t in range(300):
        K = int(rng.integers(2, 17))
        cap = rng.integers(0 if t % 3 == 0 else 1, 60, K)
        if cap.sum() == 0:
            cap[0] = 1
        gamma = cap / cap.sum()
        k = int(rng.integers(1, 450))
        assert np.array_equal(greedy_alloc(gamma, cap, k), allocate_parcels(gamma, cap, k)), (t, cap, k)
    log('greedy_alloc(n⁰=0) 与 pilot_betti.allocate_parcels 在 300 个随机例子（含容量约束与平局）上完全一致')
    # 容量约束下 n⁰ 版本不超容量
    cap = np.array([3, 100, 2])
    c = greedy_alloc([0.2, 0.5, 0.3], cap, 50, [0, 1000, 0])
    assert (c <= cap).all() and c.sum() == 50, c
    log(f'容量例 cap=[3,100,2], n⁰=[0,1000,0], k=50 → {c.tolist()}（前后两组封顶）')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--units', default='')
    ap.add_argument('--out', default=OUT_DIR)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    log_lines = []

    def log(msg, flush=True):
        print(msg, flush=flush)
        log_lines.append('  ' + msg)

    sha16 = spec_hash16()
    log(f'规格 sha256[:16] = {sha16}')
    assert sha16 == SPEC_HASH16, sha16
    try:
        from threadpoolctl import threadpool_info
        log('线程池：' + '；'.join(f'{i["internal_api"]}={i["num_threads"]}' for i in threadpool_info()))
    except Exception as e:  # noqa: BLE001
        log(f'threadpoolctl 不可用：{e}')
    hand_checks(log)

    units = all_units()
    if args.units:
        want = set(args.units.split(','))
        units = [u for u in units if u['name'] in want]
    elif args.smoke:
        units = [u for u in units if u['name'] in ('PT_s0_r0', 'PT_s0_r1')]
    assert units
    t0 = time.time()
    R, A = run(units, log)
    log(f'单元数 {len(units)}，总耗时 {time.time() - t0:.0f}s')
    exp = len(units) * len(SPACES) * len(KS) * len(THRS) * len(WS) * len(BUDGETS)
    assert len(R) == exp, (len(R), exp)
    write_atomic_csv(R, os.path.join(args.out, 'diag_rows.csv'))
    write_atomic_csv(A, os.path.join(args.out, 'diag_alloc.csv'))
    text = summarize(R, sha16, log_lines)
    write_atomic_text(text, os.path.join(args.out, 'diag_summary.txt'))
    print(text)


if __name__ == '__main__':
    main()
