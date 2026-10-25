"""
迁移版 Betti 分配覆盖诊断：独立复核实现（只依据 BETTI_TRANSFER_DIAG_SPEC_2026-09-28.md）。

范围（复核任务所限）：
  - 主设定：K=8、阈值 q95、w=1/50，k ∈ {50, 200, 400}；
  - 敏感性：K=8、阈值 none（外国样本全部计入）、w=1/50、k=200；
  - 两个空间 ae / s2，44 个单元（11 国 × 划分 {0,1} × 重复 {0,1}）；
  - 分离比：外国样本到最近候选池地块距离的中位数 / 候选池内最近邻距离的中位数（每单元 × 空间一次）。

数据纪律：
  - 不读任何标签或概率键；npz 只读 cand（索引）、allT 的头部形状、G_{f} 的前 n_cand 行、GF_{f}。
  - G_{f} 的测试行不解压：直接从 zip 流里读 .npy 头，再只读前 n_cand 行的字节（C 序，候选在前）。
  - 候选池位置用 guidance_G.pool_positions（唯一从项目导入的函数）。
  - 不训练模型，不读任何精度结果。
线程 ≤ 2（在导入 numpy/sklearn 之前设置）。
"""
import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_v] = '2'

import sys

sys.dont_write_bytecode = True
LE_DIR = ('/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/'
          'Alpha_Crops/label_expansion')
sys.path.insert(0, LE_DIR)

import hashlib
import time
import zipfile

import numpy as np
import pandas as pd
from numpy.lib import format as npf
from scipy import stats
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits

from guidance_G import pool_positions  # noqa: E402  候选池位置必须与 G 完全一致

SPEC = os.path.join(LE_DIR, 'BETTI_TRANSFER_DIAG_SPEC_2026-09-28.md')
SPEC_HASH16 = '4c8da222b77b0f84'
STAGE0 = '/Users/perng_mini/AlphaCrops_local/results/guidance_G/stage0'
OUT = '/Users/perng_mini/AlphaCrops_local/verify/betti_transfer/indep'
COUNTRIES = ['AT', 'BE', 'DE', 'EE', 'LT', 'LV', 'PT', 'RO', 'SE', 'SI', 'SK']
SPLITS, REPS = [0, 1], [0, 1]
SPACES = ['ae', 's2']
K = 8
W = 1 / 50
# (阈值名, 分位数或 None, 该设定要算的 k 列表)
SETTINGS = [('q95', 0.95, [50, 200, 400]), ('none', None, [200])]
MAIN_K = 200


# ---------------------------------------------------------------- 读取（只读允许的键）
def _read_header(f):
    ver = npf.read_magic(f)
    if ver == (1, 0):
        return npf.read_array_header_1_0(f)
    if ver == (2, 0):
        return npf.read_array_header_2_0(f)
    raise ValueError(f'未知 npy 版本 {ver}')


def npz_shape(z, key):
    """只读 .npy 头部，返回形状（用于 allT 长度，不读内容）。"""
    with z.open(key + '.npy') as f:
        shape, _, _ = _read_header(f)
    return shape


def npz_first_rows(z, key, n_rows=None):
    """从 zip 流中读 2 维数组的前 n_rows 行（C 序）；n_rows=None 读全部。后面的行不会被解压。"""
    with z.open(key + '.npy') as f:
        shape, fortran, dtype = _read_header(f)
        assert not fortran and len(shape) == 2, (key, shape, fortran)
        n = shape[0] if n_rows is None else n_rows
        assert n <= shape[0]
        need = n * shape[1] * dtype.itemsize
        chunks, got = [], 0
        while got < need:
            b = f.read(need - got)
            if not b:
                break
            chunks.append(b)
            got += len(b)
        assert got == need, (key, got, need)
        arr = np.frombuffer(b''.join(chunks), dtype=dtype).reshape(n, shape[1]).copy()
    return arr, shape


def npz_full_1d(z, key):
    with z.open(key + '.npy') as f:
        shape, fortran, dtype = _read_header(f)
        assert len(shape) == 1
        need = shape[0] * dtype.itemsize
        buf = f.read(need)
        assert len(buf) == need
        return np.frombuffer(buf, dtype=dtype).copy()


# ---------------------------------------------------------------- 距离
def sqdist(A, B):
    """A (n,d)、B (m,d) 之间的平方欧氏距离（float64），负数截为 0。"""
    A = np.asarray(A, np.float64)
    B = np.asarray(B, np.float64)
    D = (A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2.0 * A @ B.T
    np.maximum(D, 0.0, out=D)
    return D


def nn_dist(Q, P, exclude_self=False, chunk=1000):
    """Q 中每行到 P 的最近邻距离；exclude_self=True 时 Q 就是 P，排除自身（按位置）。"""
    P64 = np.asarray(P, np.float64)
    pn = (P64 * P64).sum(1)
    out = np.empty(len(Q))
    for i0 in range(0, len(Q), chunk):
        q = np.asarray(Q[i0:i0 + chunk], np.float64)
        D = (q * q).sum(1)[:, None] + pn[None, :] - 2.0 * q @ P64.T
        if exclude_self:
            r = np.arange(len(q))
            D[r, i0 + r] = np.inf
        out[i0:i0 + len(q)] = np.sqrt(np.maximum(D.min(1), 0.0))
    return out


# ---------------------------------------------------------------- 分配
def greedy_alloc(gamma, cap, n0, k):
    """逐个名额贪心：增益 γ_g[(n0+n+1)^-1/2 − (n0+n+2)^-1/2]，只在 n_g < 容量的组里选；
    np.argmax 在并列时取第一个（编号小的组）。"""
    n = np.zeros(len(gamma))
    steps = int(min(k, cap.sum()))
    for _ in range(steps):
        a = n0 + n
        gain = gamma * ((a + 1.0) ** -0.5 - (a + 2.0) ** -0.5)
        gain = np.where(n < cap, gain, -np.inf)
        n[int(np.argmax(gain))] += 1
    return n


def tv(a, b, k):
    return 0.5 * np.abs(a - b).sum() / k


# ---------------------------------------------------------------- 主流程
def unit_space(Xp, XF):
    """一个单元 × 空间：聚类、阈值、外国归组，返回各设定的行与分配明细，以及分离比。"""
    km = KMeans(n_clusters=K, n_init=10, random_state=0).fit(Xp)
    lab = km.labels_
    C = km.cluster_centers_
    # 自检：labels_ 应与"到最终质心最近"一致
    Dp = sqdist(Xp, C)
    n_mislab = int((Dp.argmin(1) != lab).sum())
    d_pool = np.sqrt(Dp[np.arange(len(Xp)), lab])
    cap = np.bincount(lab, minlength=K).astype(float)
    gamma = cap / cap.sum()

    DF = sqdist(XF, C)
    g_star = DF.argmin(1)
    d_star = np.sqrt(DF[np.arange(len(XF)), g_star])

    rows, alloc = [], []
    for thr_name, qq, ks in SETTINGS:
        if qq is None:
            counted = np.ones(len(XF), bool)
            r_g = np.full(K, np.inf)
        else:
            r_g = np.array([np.quantile(d_pool[lab == g], qq) for g in range(K)])
            counted = d_star <= r_g[g_star]
        m = np.bincount(g_star[counted], minlength=K).astype(float)
        n0 = W * m
        sm = m.sum()
        if sm > 0:
            c = (m / sm) / gamma
            cov_cv = float(c.std() / c.mean())
        else:
            cov_cv = np.nan
        for k in ks:
            n_t = greedy_alloc(gamma, cap, n0, k)
            n_a = greedy_alloc(gamma, cap, np.zeros(K), k)
            prop = gamma * k
            rows.append(dict(K=K, thr=thr_name, w=W, k=k, tv_a=tv(n_t, n_a, k), tv_b=tv(n_t, prop, k),
                             foreign_rate=sm / len(XF), cov_cv=cov_cv, n_zero=int((n_t == 0).sum()),
                             max_share=float(n_t.max() / k), tv_a_vs_b=tv(n_a, prop, k),
                             n_mislab=n_mislab, inertia=float(km.inertia_)))
            for g in range(K):
                alloc.append(dict(K=K, thr=thr_name, w=W, k=k, group=g, gamma=gamma[g], cap=int(cap[g]),
                                  r_g=r_g[g], m_g=int(m[g]), n0=n0[g], n_transfer=int(n_t[g]),
                                  n_noinit=int(n_a[g]), prop=prop[g]))
    # 分离比
    d_fp = nn_dist(XF, Xp)
    d_pp = nn_dist(Xp, Xp, exclude_self=True)
    sep = dict(med_foreign_to_pool=float(np.median(d_fp)), med_pool_nn=float(np.median(d_pp)))
    sep['sep_ratio'] = sep['med_foreign_to_pool'] / sep['med_pool_nn']
    return rows, alloc, sep


def verdict(med):
    if med < 0.10:
        return '退化'
    if med < 0.20:
        return '弱偏离'
    return '明显偏离'


def main():
    with open(SPEC, 'rb') as f:
        h = hashlib.sha256(f.read()).hexdigest()
    assert h[:16] == SPEC_HASH16, f'规格文件哈希不符：{h[:16]}'
    print(f'spec sha256[:16] = {h[:16]}', flush=True)
    os.makedirs(OUT, exist_ok=True)
    rows_all, alloc_all, sep_all = [], [], []
    t_all = time.time()
    for tgt in COUNTRIES:
        for split in SPLITS:
            path = os.path.join(STAGE0, f'{tgt}_s{split}.npz')
            with zipfile.ZipFile(path) as z:
                cand = npz_full_1d(z, 'cand')
                n_cand = len(cand)
                n_allT = npz_shape(z, 'allT')[0]
                geo = {}
                for f in SPACES:
                    Gc, shapeG = npz_first_rows(z, f'G_{f}', n_cand)  # 只解压候选行
                    assert shapeG[0] == n_allT, (tgt, split, f, shapeG, n_allT)
                    GF, _ = npz_first_rows(z, f'GF_{f}')
                    geo[f] = (Gc, GF)
            for rep in REPS:
                pos = pool_positions({'cand': cand}, split, rep)
                assert pos.max() < n_cand
                for f in SPACES:
                    t0 = time.time()
                    Gc, GF = geo[f]
                    Xp = Gc[pos]
                    rows, alloc, sep = unit_space(Xp, GF)
                    base = dict(target=tgt, split=split, rep=rep, space=f)
                    rows_all += [{**base, **r} for r in rows]
                    alloc_all += [{**base, **a} for a in alloc]
                    sep_all.append({**base, 'n_cand': n_cand, 'n_pool': len(pos), 'n_foreign': len(GF), **sep})
                    main_row = [r for r in rows if r['thr'] == 'q95' and r['k'] == MAIN_K][0]
                    print(f'{tgt} s{split} r{rep} {f}: pool {len(pos)} TV_a(k200) {main_row["tv_a"]:.3f} '
                          f'计入率 {main_row["foreign_rate"]:.3f} 分离比 {sep["sep_ratio"]:.2f} '
                          f'mislab {main_row["n_mislab"]} {time.time() - t0:.1f}s', flush=True)
    R = pd.DataFrame(rows_all)
    A = pd.DataFrame(alloc_all)
    S = pd.DataFrame(sep_all)
    R.to_csv(os.path.join(OUT, 'indep_rows.csv'), index=False)
    A.to_csv(os.path.join(OUT, 'indep_alloc.csv'), index=False)
    S.to_csv(os.path.join(OUT, 'indep_sep.csv'), index=False)
    write_summary(R, S, time.time() - t_all, h)


def write_summary(R, S, secs, h):
    L = [f'迁移版 Betti 覆盖诊断（独立复核实现） spec sha256[:16] = {h[:16]}',
         f'单元数：{R.groupby(["target", "split", "rep"]).ngroups}；空间 ae/s2；K={K}；w=1/50；耗时 {secs:.0f}s', '']
    L.append('== 主设定（K=8、q95、w=1/50） ==')
    for k in (50, 200, 400):
        for f in SPACES:
            d = R[(R.thr == 'q95') & (R.k == k) & (R.space == f)]
            L.append(f'k={k:3d} {f}: TV_a 中位数 {d.tv_a.median():.4f} [IQR {d.tv_a.quantile(.25):.4f}, '
                     f'{d.tv_a.quantile(.75):.4f}]  TV_b 中位数 {d.tv_b.median():.4f}  '
                     f'外国计入率中位数 {d.foreign_rate.median():.4f}  覆盖 CV 中位数 {d.cov_cv.median():.3f}  '
                     f'0 名额组数中位数 {d.n_zero.median():.1f}  最大组占比中位数 {d.max_share.median():.3f}  '
                     f'(n={len(d)})')
    L.append('')
    L.append('== 判定（主设定 k=200，44 单元 TV_a 中位数） ==')
    for f in SPACES:
        d = R[(R.thr == 'q95') & (R.k == MAIN_K) & (R.space == f)]
        L.append(f'{f}: 中位数 {d.tv_a.median():.4f} → {verdict(d.tv_a.median())}；'
                 f'最小 {d.tv_a.min():.4f}，最大 {d.tv_a.max():.4f}；TV_a ≥ 0.10 的单元 {(d.tv_a >= 0.10).sum()}，'
                 f'≥ 0.20 的单元 {(d.tv_a >= 0.20).sum()}')
    L.append('')
    L.append('== 敏感性（K=8、阈值 none、w=1/50、k=200；只作描述） ==')
    for f in SPACES:
        d = R[(R.thr == 'none') & (R.k == MAIN_K) & (R.space == f)]
        L.append(f'{f}: TV_a 中位数 {d.tv_a.median():.4f} [IQR {d.tv_a.quantile(.25):.4f}, '
                 f'{d.tv_a.quantile(.75):.4f}]  TV_b 中位数 {d.tv_b.median():.4f}  '
                 f'计入率中位数 {d.foreign_rate.median():.4f}  覆盖 CV 中位数 {d.cov_cv.median():.3f}')
    L.append('')
    L.append('== 无初始分配与比例分配之间的 TV（参照，主设定 k=200） ==')
    for f in SPACES:
        d = R[(R.thr == 'q95') & (R.k == MAIN_K) & (R.space == f)]
        L.append(f'{f}: TV(无初始, 比例) 中位数 {d.tv_a_vs_b.median():.4f}')
    L.append('')
    L.append('== 分离比（外国→候选池最近邻距离中位数 / 候选池内最近邻距离中位数） ==')
    for f in SPACES:
        d = S[S.space == f]
        L.append(f'{f}: 44 单元中位数 {d.sep_ratio.median():.3f} [IQR {d.sep_ratio.quantile(.25):.3f}, '
                 f'{d.sep_ratio.quantile(.75):.3f}]，范围 {d.sep_ratio.min():.3f}–{d.sep_ratio.max():.3f}')
    L.append('')
    L.append('== 各国 TV_a 中位数（主设定 k=200，4 个单元） ==')
    P = (R[(R.thr == 'q95') & (R.k == MAIN_K)].groupby(['target', 'space']).tv_a.median().unstack('space'))
    P2 = (R[(R.thr == 'q95') & (R.k == MAIN_K)].groupby(['target', 'space']).foreign_rate.median()
          .unstack('space').add_prefix('计入率_'))
    Ssep = S.groupby(['target', 'space']).sep_ratio.median().unstack('space').add_prefix('分离比_')
    T = pd.concat([P, P2, Ssep], axis=1)
    L.append(T.round(4).to_string())
    L.append('')
    # AE − S2 配对差（描述）
    d = R[(R.thr == 'q95') & (R.k == MAIN_K)].pivot_table(index=['target', 'split', 'rep'], columns='space',
                                                            values='tv_a')
    diff = (d['ae'] - d['s2']).groupby(level='target').mean()
    mu, se = diff.mean(), diff.std(ddof=1) / np.sqrt(len(diff))
    tq = stats.t.ppf(0.975, len(diff) - 1)
    L.append(f'== AE − S2 配对 TV_a 差（主设定 k=200，先平均到国家，n={len(diff)}，df={len(diff) - 1}） ==')
    L.append(f'均值 {mu:.4f}，95% CI [{mu - tq * se:.4f}, {mu + tq * se:.4f}]')
    L.append('')
    mis = R.drop_duplicates(['target', 'split', 'rep', 'space'])
    L.append(f'自检：KMeans labels_ 与 float64 重算的最近质心不一致的地块数，88 个单元×空间合计 '
             f'{int(mis.n_mislab.sum())}（float32 近似并列所致；分组一律以 labels_ 为准）')
    txt = '\n'.join(L)
    with open(os.path.join(OUT, 'indep_summary.txt'), 'w') as fo:
        fo.write(txt + '\n')
    print(txt, flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
