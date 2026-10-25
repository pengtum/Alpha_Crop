"""NC 主实验分析的独立实现（交叉核对用；冻结方案 NC_PLAN_v6.md 第 1、3、4 节）。

只依据方案文字编写，不读取、不 import nc_main_analyze.py 与 nc_metrics.py。
复用的冻结代码只有两处：
  analyze_confirm.unit_curves   每个臂的 τ 曲线 (OA[81], BA[81])，τ = −2..2 步长 0.05（RSE 已验证）
  prep_unified.crop_units       读取作物（RSE 确认运行）的测试集概率、标签与 big 类别
其余全部在本文件中实现：Δ_V（自由处置包络 BA⁺）、e(k)、g(k)、d、M、保序回归与 L_saved、
H1 精确二项、H2 OLS + HC3 + 单侧 t、Holm、M₁ 与按族分层 bootstrap、附录学习器、作物锚点。

输入（<root> 为 nc_main_run.py 的输出根目录）
  <root>/runs_*.csv、budgets.csv、removed.csv、probs/{task}/{domain}/{learner}_{feat}_{arm}_r{r}.npz
  ~/AlphaCrops_local/nc_main/splits/：targets.json、roles.parquet、orders.npz、ba_classes.json（只读）
  ~/AlphaCrops_local/nc_pilot/splits/：t1_samples.parquet、t2_samples.parquet（标签）
  ~/AlphaCrops_local/results/confirm_v1/：作物 probs、splits、selection（只读）
输出 <root>/indep/：domain_values.csv、curves_e_g.csv、confirm.json、appendix.json、crop_anchor.csv

用法：~/AlphaCrops_local/.venv/bin/python -B nc_main_indep.py --root <runs 根目录> [--no-crop]
"""
import sys

sys.dont_write_bytecode = True  # 不在项目目录写 __pycache__
import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '2'
import argparse
import glob
import hashlib
import json
import math

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
AC_ROOT = os.path.dirname(HERE)
LE = os.path.join(AC_ROOT, 'label_expansion')
UF = os.path.join(AC_ROOT, 'unified_figs')
LOCAL = '/Users/perng_mini/AlphaCrops_local'
MAIN_SPLITS = os.path.join(LOCAL, 'nc_main', 'splits')
FORMAL_RUNS = os.path.join(LOCAL, 'nc_main', 'runs')
PILOT_SPLITS = os.path.join(LOCAL, 'nc_pilot', 'splits')
CONF = os.path.join(LOCAL, 'results', 'confirm_v1')

# ---------------------------------------------------------------- 常量（方案第 2–4 节）
TASKS = ('lcz', 'lucas')
SAMPLES = {'lcz': ('t1_samples.parquet', 'city', 'row'), 'lucas': ('t2_samples.parquet', 'nuts0', 'id')}
FEATS = ('ae', 's2')
KS_ALL = (50, 100, 200, 400, 800, 1600)
R = 3
APPX = ('rf', 'lr')
M_LO, M_HI = 200, 250                     # 抽样顺序第 201–250 个（0 起算的切片）
CROP_KS = (50, 100, 200, 400)
N_TAU = 81
I_T0 = 40                                 # τ = −2 + 0.05·40 = 0
EPS = 1e-9
OFFS = np.array([round(-4 + 0.1 * j, 1) for j in range(41)])   # o_j = m + round(−4 + 0.1·j, 1)
ALPHA = 0.025
N_BOOT = 2000
BOOT_SEED = 'nc_main_v1_boot_M1'


# ---------------------------------------------------------------- 冻结代码的导入（导入链会 chdir，前后恢复 cwd）
def import_ac():
    cwd = os.getcwd()
    sys.path.insert(0, os.path.join(LE, 'confirm'))
    import analyze_confirm as AC  # noqa: E402
    os.chdir(cwd)
    assert len(AC.TAUS) == N_TAU and AC.TAUS[I_T0] == 0.0
    return AC


def import_pu():
    cwd = os.getcwd()
    sys.path.insert(0, UF)
    import prep_unified as PU  # noqa: E402
    os.chdir(cwd)
    return PU


# ---------------------------------------------------------------- 方案第 2 节：种子
def rng_of(s):
    return np.random.default_rng(int(hashlib.md5(s.encode('utf-8')).hexdigest(), 16) % (2 ** 31))


# ---------------------------------------------------------------- 方案第 3 节：Δ_V
def ba_plus(oa, ba, grid):
    """BA⁺(o)：τ 路径上 OA ≥ o − 1e-9 的所有点中的最高 BA。
    路径点 = 81 个顶点 + 相邻顶点间线性插值所得、OA 恰为 o 的点。"""
    oa = np.asarray(oa, float)
    ba = np.asarray(ba, float)
    grid = np.asarray(grid, float)
    # 顶点
    okv = oa[None, :] >= grid[:, None] - EPS
    best = np.where(okv, ba[None, :], -np.inf).max(1)
    # 段内 OA 恰为 o 的插值点
    o0, o1, b0, b1 = oa[:-1], oa[1:], ba[:-1], ba[1:]
    lo, hi = np.minimum(o0, o1), np.maximum(o0, o1)
    nonflat = o1 != o0
    inside = (grid[:, None] >= lo[None, :]) & (grid[:, None] <= hi[None, :]) & nonflat[None, :]
    den = np.where(nonflat, o1 - o0, 1.0)
    lam = np.clip((grid[:, None] - o0[None, :]) / den[None, :], 0.0, 1.0)
    bx = np.where(inside, b0[None, :] + lam * (b1 - b0)[None, :], -np.inf)
    return np.maximum(best, bx.max(1))


def delta_v(cA, cB):
    """Δ_V(A → B)：锚点 m = min(OA_A(τ=0), OA_B(τ=0))，41 个格点上 BA⁺_B − BA⁺_A 的平均。"""
    (oaA, baA), (oaB, baB) = cA, cB
    assert len(oaA) == N_TAU and len(oaB) == N_TAU
    m = min(float(oaA[I_T0]), float(oaB[I_T0]))
    grid = m + OFFS
    eA, eB = ba_plus(oaA, baA, grid), ba_plus(oaB, baB, grid)
    assert np.isfinite(eA).all() and np.isfinite(eB).all()   # τ=0 顶点 OA ≥ m ≥ o，必有可行点
    return float(np.mean(eB - eA))


# ---------------------------------------------------------------- 方案第 3 节：L_saved
def pav_inc(y):
    """等权保序回归（单调不减），合并相邻违例块。"""
    blocks = []
    for v in np.asarray(y, float):
        blocks.append([v, 1])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            (m1, n1), (m2, n2) = blocks[-2], blocks[-1]
            blocks[-2:] = [[(m1 * n1 + m2 * n2) / (n1 + n2), n1 + n2]]
    return np.concatenate([np.full(n, m) for m, n in blocks])


def l_saved(K, e):
    """K 升序（含 0），e 为对应 e(k)。ê = −pav(−e)（单调不增）。
    返回 (L_saved, censored, k_max, ê)；censored 时 L_saved 记 NaN，含义为 "> k_max"。"""
    K = np.asarray(K, float)
    eh = -pav_inc(-np.asarray(e, float))
    kmax = float(K[-1])
    if eh[0] <= 0:
        return 0.0, False, kmax, eh
    hit = np.flatnonzero(eh <= 0)
    if len(hit) == 0:
        return np.nan, True, kmax, eh
    j = int(hit[0])
    k0, k1, v0, v1 = K[j - 1], K[j], eh[j - 1], eh[j]
    return float(k0 + (0.0 - v0) / (v1 - v0) * (k1 - k0)), False, kmax, eh


# ---------------------------------------------------------------- 方案第 4 节：检验
def binom_greater_p(n_pos, n):
    """精确二项单侧 p = P(X ≥ n_pos | n, 1/2)。"""
    return float(sum(math.comb(n, x) for x in range(n_pos, n + 1)) / 2 ** n)


def ols_fit(y, X):
    """OLS；X 秩不足时返回 None。"""
    if np.linalg.matrix_rank(X) < X.shape[1]:
        return None
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ X.T @ y
    return b, XtX_inv


def hc3_se(y, X, b, XtX_inv):
    e = y - X @ b
    h = np.einsum('ij,jk,ik->i', X, XtX_inv, X)
    with np.errstate(divide='ignore', invalid='ignore'):
        w = e ** 2 / (1.0 - h) ** 2
        V = XtX_inv @ (X.T * w) @ X @ XtX_inv
        return np.sqrt(np.diag(V)), h


def design(fam, M):
    """g(200) = a + γ·[族 = LUCAS] + β·M。"""
    fam = np.asarray(fam)
    return np.column_stack([np.ones(len(fam)), (fam == 'lucas').astype(float), np.asarray(M, float)])


def holm2(p1, p2, alpha=ALPHA):
    """两个假设的 Holm：较小 p 在 alpha/2 下检验；拒绝后另一个在 alpha 下检验。p 为 NaN 视作不拒绝。"""
    ps = [p1 if np.isfinite(p1) else np.inf, p2 if np.isfinite(p2) else np.inf]
    order = [0, 1] if ps[0] <= ps[1] else [1, 0]
    rej, thr = [False, False], [None, None]
    thr[order[0]], thr[order[1]] = alpha / 2, alpha
    if ps[order[0]] <= alpha / 2:
        rej[order[0]] = True
        rej[order[1]] = ps[order[1]] <= alpha
    return rej, thr


def q_ci95(x):
    """逆经验分布分位数，不插值：排序后第 ceil(0.025·nb) 与第 ceil(0.975·nb) 个（1 起算），整数运算避免浮点误差。"""
    s = np.sort(np.asarray(x, float))
    nb = len(s)
    if nb == 0:
        return None
    lo_i, hi_i = -(-25 * nb // 1000), -(-975 * nb // 1000)
    return [float(s[lo_i - 1]), float(s[hi_i - 1])]


# ---------------------------------------------------------------- 读 NC 运行输出
def read_runs(root):
    fs = sorted(glob.glob(os.path.join(root, 'runs_*.csv')))
    assert fs, f'{root} 下没有 runs_*.csv'
    runs = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    assert runs.key.is_unique, 'runs 中有重复的拟合键'
    return runs


def planned_fits(ks, learners):
    """方案第 2 节：lgb 做 Z0 与全部 RT_k（3 次抽样）；附录学习器只做 Z0 与 RT200（3 次抽样）。两种特征。"""
    out = []
    for lr_ in learners:
        arms = [('Z0', 0)] + ([(f'RT{k}', r) for k in ks for r in range(R)] if lr_ == 'lgb'
                              else [('RT200', r) for r in range(R)])
        out += [(lr_, f, a, r) for f in FEATS for a, r in arms]
    return out


def load_task_meta(task):
    fn, dcol, scol = SAMPLES[task]
    S = pd.read_parquet(os.path.join(PILOT_SPLITS, fn))[[dcol, scol, 'y']]
    S = S.rename(columns={dcol: 'domain', scol: 'sid'})
    S['sid'] = S.sid.astype(np.int64)
    return S.set_index(['domain', 'sid']).y.astype(int)


def domain_inputs(task, d, root, Y, roles, orders, removed, ba_all, budgets):
    """一个域：标签、BA 类别、预算档、M 的 50 个样本；并对照划分文件独立核对运行输出的样本集合。"""
    rm = set(removed.loc[(removed.task == task) & (removed.domain == d), 'sid'].astype(np.int64))
    rl = roles[(roles.task == task) & (roles.domain == d)]
    cand = set(rl.loc[rl.role == 'cand', 'sid'].astype(np.int64))
    test_all = rl.loc[rl.role == 'test', 'sid'].astype(np.int64).values
    test_kept = np.sort(test_all[~np.isin(test_all, list(rm))])
    ords = []
    for r in range(R):
        o = np.asarray(orders[f'{task}/{d}/r{r}'], dtype=np.int64)
        assert set(o.tolist()) == cand and len(o) == len(cand), f'{task}/{d}/r{r}：抽样顺序不是候选集的排列'
        ords.append(o[~np.isin(o, list(rm))])             # 剔除缺失后保持相对次序
    n_c = len(ords[0])
    ks = [k for k in KS_ALL if k <= n_c]
    b = budgets[(budgets.task == task) & (budgets.domain == d)]
    assert len(b) == 1, f'{task}/{d}：budgets.csv 中缺失或重复'
    assert int(b.n_cand_after.iloc[0]) == n_c, f'{task}/{d}：剔除后候选数 {n_c} 与 budgets.csv 不符'
    assert [int(x) for x in str(b.ks.iloc[0]).split()] == ks, f'{task}/{d}：预算档与 budgets.csv 不符'
    ba = [int(c) for c in ba_all[task][d]]
    # 方案第 1 节：BA 类别为测试集（剔除缺失之前）≥ 10 个样本的类，且 ≥ 3 类（LCZ 的空间簇条件此处不重算）
    yt_all = Y.loc[[(d, s) for s in test_all]].values
    cnt = np.bincount(yt_all, minlength=max(ba) + 1)
    assert all(cnt[c] >= 10 for c in ba) and len(ba) >= 3, f'{task}/{d}：BA 类别与测试集计数不符'
    return dict(test=test_kept, ks=ks, ba=ba, m_sids=ords[0][M_LO:M_HI], n_cand_after=n_c)


def domain_values(task, d, root, Y, info, learners, AC):
    """一个域的全部量。返回 (行列表, 曲线行列表)。"""
    pdir = os.path.join(root, 'probs', task, d)
    ks = info['ks']
    P, sid_ref, Pm, sid_m = {}, None, None, None
    for lr_, f, a, r in planned_fits(ks, learners):
        z = np.load(os.path.join(pdir, f'{lr_}_{f}_{a}_r{r}.npz'))
        sid = z['sid_test'].astype(np.int64)
        if sid_ref is None:
            sid_ref = sid
            assert np.array_equal(np.sort(sid), info['test']), f'{task}/{d}：测试集与划分（剔除缺失后）不符'
        assert np.array_equal(sid, sid_ref), f'{task}/{d}：各臂测试集行序不一致'
        P[(lr_, f, a, r)] = z['P_test']
        if lr_ == 'lgb' and f == 'ae' and a == 'Z0':
            Pm, sid_m = np.asarray(z['P_m'], np.float64), z['sid_m'].astype(np.int64)
    yT = Y.loc[[(d, s) for s in sid_ref]].values             # 标签按运行输出的行序对齐
    CV = AC.unit_curves(yT, info['ba'], P)

    # M：AE 的 Z0 在 τ = 0（argmax，并列取首个）对第 201–250 个候选样本的正确率（%）
    assert np.array_equal(sid_m, info['m_sids']), f'{task}/{d}：M 的 50 个样本与抽样顺序第 201–250 个不符'
    ym = Y.loc[[(d, s) for s in sid_m]].values
    M = 100.0 * float(np.mean(np.argmax(Pm, axis=1) == ym))

    rows, curves = [], []
    for lr_ in learners:
        z0a, z0s = CV[(lr_, 'ae', 'Z0', 0)], CV[(lr_, 's2', 'Z0', 0)]
        kk = ks if lr_ == 'lgb' else [200]
        e = {0: delta_v(z0s, z0a)}
        g = {}
        for k in kk:
            e[k] = float(np.mean([delta_v(CV[(lr_, 's2', f'RT{k}', r)], z0a) for r in range(R)]))
            g[k] = float(np.mean([delta_v(z0a, CV[(lr_, 'ae', f'RT{k}', r)]) for r in range(R)]))
        row = dict(task=task, domain=d, learner=lr_, d=e[200], g200=g[200],
                   M=np.nan, L_saved=np.nan, censored=np.nan, k_max=np.nan)
        if lr_ == 'lgb':
            K = [0] + list(ks)
            L, cens, kmax, eh = l_saved(K, [e[k] for k in K])
            row.update(M=M, L_saved=L, censored=cens, k_max=kmax)
            row['_ehat'] = dict(zip(K, eh))
            row['_e'], row['_g'] = e, g
            curves += [dict(task=task, domain=d, learner=lr_, quantity='e', k=k, value=e[k]) for k in K]
            curves += [dict(task=task, domain=d, learner=lr_, quantity='g', k=k, value=g[k]) for k in ks]
        rows.append(row)
    return rows, curves


# ---------------------------------------------------------------- 作物锚点
def crop_values(AC):
    PU = import_pu()
    cwd = os.getcwd()
    sys.path.insert(0, LE)
    os.chdir(LE)
    from guidance_G import GData  # noqa: E402
    G = GData()
    os.chdir(cwd)
    y_all = np.asarray(G.y)
    units = []
    for U in PU.crop_units():
        u, P = U['unit'], {k: v for k, v in U['P'].items() if k[1] != 'REF'}
        CV = AC.unit_curves(U['yT'], U['big'], P)
        z0a, z0s = CV[('ae', 'Z0', 0)], CV[('s2', 'Z0', 0)]
        e = {0: delta_v(z0s, z0a)}
        g = {}
        for k in CROP_KS:
            e[k] = float(np.mean([delta_v(CV[('s2', f'RT{k}', r)], z0a) for r in range(R)]))
            g[k] = float(np.mean([delta_v(z0a, CV[('ae', f'RT{k}', r)]) for r in range(R)]))
        # M：RSE 已存 AE Z0 的候选池概率 P_pool，第 r = 0 次抽样顺序第 201–250 位，τ = 0 的正确率
        sp = np.load(os.path.join(CONF, 'splits', f'{u}.npz'))
        sel = np.load(os.path.join(CONF, 'selection', f'{u}.npz'))
        Pp = np.asarray(np.load(os.path.join(CONF, 'probs', u, 'lgb_Z0_ae_r0.npz'))['P_pool'], np.float64)
        pool, posm = sp['pool'], sel['pi_r0'][M_LO:M_HI]
        assert Pp.shape[0] == len(pool) and len(posm) == 50
        assert not np.isin(pool[posm], sp['test']).any()
        assert np.array_equal(U['yT'], y_all[sp['test']])
        M = 100.0 * float(np.mean(np.argmax(Pp[posm], axis=1) == y_all[pool[posm]]))
        units.append(dict(unit=u, domain=U['domain'], primary=bool(U['primary']), e=e, g=g, M=M))
    rows, curves = [], []
    K = [0] + list(CROP_KS)
    for cc in sorted({x['domain'] for x in units}):
        us = [x for x in units if x['domain'] == cc]
        assert len(us) == 2, f'作物 {cc}：应有 F1、F2 两半'
        e = {k: float(np.mean([x['e'][k] for x in us])) for k in K}
        g = {k: float(np.mean([x['g'][k] for x in us])) for k in CROP_KS}
        M = float(np.mean([x['M'] for x in us]))
        L, cens, kmax, eh = l_saved(K, [e[k] for k in K])
        rows.append(dict(domain=cc, primary=us[0]['primary'], d=e[200], g200=g[200], M=M, L_saved=L, censored=cens,
                         _k_max=kmax, _ehat=dict(zip(K, eh)), _e=e))
        curves += [dict(task='crop', domain=cc, learner='lgb', quantity='e', k=k, value=e[k]) for k in K]
        curves += [dict(task='crop', domain=cc, learner='lgb', quantity='g', k=k, value=g[k]) for k in CROP_KS]
    return rows, curves


# ---------------------------------------------------------------- 汇总（只报估计值）
def fnum(x):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else float(x)


def summ(v):
    v = np.asarray(v, float)
    return dict(n=int(len(v)), mean=fnum(float(np.mean(v))) if len(v) else None,
                median=fnum(float(np.median(v))) if len(v) else None,
                n_pos=int((v > 0).sum()), frac_pos=fnum(float((v > 0).mean())) if len(v) else None)


def lsaved_summary(rows):
    """L_saved 分布与各 k 上 L_saved > k（即 ê(k) > 0）的域比例；只在有该 k 的域上计算。"""
    allk = sorted({k for r in rows for k in r['_ehat']})
    frac = []
    for k in allk:
        have = [r for r in rows if k in r['_ehat']]
        npos = sum(r['_ehat'][k] > 0 for r in have)
        frac.append(dict(k=int(k), n_domains=len(have), n_gt=int(npos), frac=fnum(npos / len(have))))
    vals = [(r['domain'], ('> %d' % int(r.get('k_max', r.get('_k_max')))) if r['censored'] else fnum(r['L_saved']))
            for r in rows]
    arr = np.array([np.inf if r['censored'] else r['L_saved'] for r in rows], float)   # 截尾视作 +∞
    med = float(np.median(arr)) if len(arr) else np.nan
    return dict(n=len(rows), n_censored=int(sum(bool(r['censored']) for r in rows)),
                median_censored_as_inf=('censored' if med == np.inf else fnum(med)),
                values=vals, frac_L_saved_gt_k=frac)


def curve_summary(rows, key):
    allk = sorted({k for r in rows for k in r[key]})
    out = []
    for k in allk:
        v = [r[key][k] for r in rows if k in r[key]]
        out.append(dict(k=int(k), n_domains=len(v), mean=fnum(float(np.mean(v))), median=fnum(float(np.median(v)))))
    return out


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return fnum(float(o))
    return o


# ---------------------------------------------------------------- 主程序
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--no-crop', action='store_true')
    a = ap.parse_args()
    root = os.path.abspath(os.path.expanduser(a.root))
    out = os.path.join(root, 'indep')
    os.makedirs(out, exist_ok=True)
    AC = import_ac()

    runs = read_runs(root)
    budgets = pd.read_csv(os.path.join(root, 'budgets.csv'))
    removed = pd.read_csv(os.path.join(root, 'removed.csv'))
    removed['sid'] = removed.sid.astype(np.int64)
    targets = json.load(open(os.path.join(MAIN_SPLITS, 'targets.json')))
    roles = pd.read_parquet(os.path.join(MAIN_SPLITS, 'roles.parquet'))
    orders = np.load(os.path.join(MAIN_SPLITS, 'orders.npz'))
    ba_all = json.load(open(os.path.join(MAIN_SPLITS, 'ba_classes.json')))
    learners = ['lgb'] + [x for x in APPX if x in set(runs.learner)]
    formal = os.path.realpath(root) == os.path.realpath(FORMAL_RUNS)

    D, CURV, dropped = [], [], []
    for task in TASKS:
        Y = load_task_meta(task)
        doms = sorted(budgets.loc[budgets.task == task, 'domain'])
        if formal:
            assert doms == sorted(targets[task]), f'{task}：正式运行的域与登记名单不符'
        else:
            assert set(doms) <= set(targets[task])
            if set(doms) != set(targets[task]):
                print(f'[非正式根目录] {task}：只分析 {len(doms)}/{len(targets[task])} 个域')
        for d in doms:
            info = domain_inputs(task, d, root, Y, roles, orders, removed, ba_all, budgets)
            rd = runs[(runs.task == task) & (runs.domain == d)]
            st = {(x.learner, x.feat, x.arm, int(x.r)): x.status for x in rd.itertuples()}
            plan = planned_fits(info['ks'], learners)
            assert set(st) == set(plan), f'{task}/{d}：runs 记录与方案拟合计划不一致'
            bad = [k for k in plan if st[k] != 'ok']
            if bad:                                         # 方案第 2 节：重试后仍失败 → 剔除该域并报告
                dropped.append(dict(task=task, domain=d, failed=['|'.join(map(str, k)) for k in bad]))
                continue
            rows, curves = domain_values(task, d, root, Y, info, learners, AC)
            D += rows
            CURV += curves
            print(f'{task}/{d}: n_test {len(info["test"])}, ks {info["ks"]}, d {rows[0]["d"]:+.3f}, '
                  f'g200 {rows[0]["g200"]:+.3f}, M {rows[0]["M"]:.1f}', flush=True)

    cols = ['task', 'domain', 'learner', 'd', 'g200', 'M', 'L_saved', 'censored', 'k_max']
    DV = pd.DataFrame([{c: r[c] for c in cols} for r in D])
    DV.to_csv(os.path.join(out, 'domain_values.csv'), index=False)

    # ---------------- 确认性检验（LightGBM，两族合并）
    L = [r for r in D if r['learner'] == 'lgb']
    L.sort(key=lambda r: (TASKS.index(r['task']), r['domain']))
    fam = np.array([r['task'] for r in L])
    dv = np.array([r['d'] for r in L], float)
    gv = np.array([r['g200'] for r in L], float)
    Mv = np.array([r['M'] for r in L], float)
    n = len(L)
    n_pos = int((dv > 0).sum())
    p1 = binom_greater_p(n_pos, n)
    X = design(fam, Mv)
    fit = ols_fit(gv, X)
    if fit is None:
        b = np.full(3, np.nan)
        se = np.full(3, np.nan)
    else:
        b, XtX_inv = fit
        se, _ = hc3_se(gv, X, b, XtX_inv)
    df = n - 3
    with np.errstate(divide='ignore', invalid='ignore'):
        t = float(b[2] / se[2])
    p2 = float(stats.t.cdf(t, df)) if np.isfinite(t) and df > 0 else np.nan
    (rej1, rej2), (thr1, thr2) = holm2(p1, p2)

    # ---------------- M₁ 与按族分层 bootstrap
    def m1_of(bb):
        return {'lcz': (1 - bb[0]) / bb[2], 'lucas': (1 - bb[0] - bb[1]) / bb[2]}

    with np.errstate(divide='ignore', invalid='ignore'):
        m1_pt = m1_of(b)
    rng = rng_of(BOOT_SEED)
    idx = {}
    gi = {}
    for task in TASKS:                                       # 每族的域按名称升序；先 LCZ、后 LUCAS
        sel = sorted([i for i in range(n) if fam[i] == task], key=lambda i: L[i]['domain'])
        gi[task] = np.array(sel, int)
        idx[task] = rng.integers(0, len(sel), size=(N_BOOT, len(sel)))
    bet, m1b = np.full(N_BOOT, np.nan), {f: np.full(N_BOOT, np.nan) for f in TASKS}
    n_sing = 0
    for bi in range(N_BOOT):
        ii = np.concatenate([gi[f][idx[f][bi]] for f in TASKS])
        fb = ols_fit(gv[ii], design(fam[ii], Mv[ii]))
        if fb is None:
            n_sing += 1
            continue
        bb = fb[0]
        bet[bi] = bb[2]
        if bb[2] < 0:
            mm = m1_of(bb)
            for f in TASKS:
                m1b[f][bi] = mm[f]
    n_nonneg = int(np.sum(bet >= 0))
    q = n_nonneg / N_BOOT
    q_excl = (n_nonneg + n_sing) / N_BOOT                   # 不可估计（设计矩阵秩不足）的重抽样同样不计入
    keep = np.isfinite(bet) & (bet < 0)
    M1 = {}
    for f in TASKS:
        Mf = Mv[fam == f]
        v = m1_pt[f]
        ci = q_ci95(m1b[f][keep]) if q_excl <= 0.025 else None
        M1[f] = dict(value=fnum(float(v)), ci95=ci, unbounded=bool(q_excl > 0.025),
                     extrapolated=(bool(v < Mf.min() or v > Mf.max()) if np.isfinite(v) and len(Mf) else None),
                     M_range=[fnum(float(Mf.min())), fnum(float(Mf.max()))] if len(Mf) else None)

    claims = {}
    if rej1:
        claims['H1'] = '在多数目标地区，零标签的 AE 胜过 S2 加 200 个本地标签（附 L_saved 的分布）'
    if rej2:
        claims['H2'] = ('先用 50 个本地标签测零标签迁移精度。高于 M₁ 时，再采 200 个标签预期提升不足 1 pp；'
                        '低于 M₁ 时，应采标签。')
    est = {
        'd': {**{f: summ(dv[fam == f]) for f in TASKS}, 'pooled': summ(dv)},
        'g200': {**{f: summ(gv[fam == f]) for f in TASKS}, 'pooled': summ(gv)},
        'M': {f: summ(Mv[fam == f]) for f in TASKS},
        'L_saved': {**{f: lsaved_summary([r for r in L if r['task'] == f]) for f in TASKS}, 'pooled': lsaved_summary(L)},
        'e_curve': {**{f: curve_summary([r for r in L if r['task'] == f], '_e') for f in TASKS},
                    'pooled': curve_summary(L, '_e')},
        'g_curve': {**{f: curve_summary([r for r in L if r['task'] == f], '_g') for f in TASKS},
                    'pooled': curve_summary(L, '_g')},
    }
    confirm = dict(
        n=n, n_by_family={f: int((fam == f).sum()) for f in TASKS}, removed_domains=dropped,
        H1=dict(n_pos=n_pos, n=n, p=p1, rejected=bool(rej1), holm_alpha=thr1, test='精确二项单侧 P(X ≥ n_pos | n, 1/2)'),
        H2=dict(a=b[0], gamma=b[1], beta=b[2], se_hc3=se[2], t=t, df=df, p=p2, rejected=bool(rej2), holm_alpha=thr2,
                se_hc3_all=list(se), test='OLS g200 ~ 1 + [LUCAS] + M，HC3，单侧 t（β < 0）'),
        M1=M1, M1_q_beta_nonneg=q, M1_n_nonestimable=n_sing, M1_q_excluded=q_excl, M1_n_boot=N_BOOT,
        M1_n_used=int(keep.sum()), claims=claims, estimates=est,
    )

    # ---------------- 附录：随机森林、逻辑回归
    appx = {}
    for lr_ in APPX:
        A = [r for r in D if r['learner'] == lr_]
        if not A:
            appx[lr_] = None
            continue
        af = np.array([r['task'] for r in A])
        ad = np.array([r['d'] for r in A], float)
        ag = np.array([r['g200'] for r in A], float)
        appx[lr_] = dict(d={**{f: summ(ad[af == f]) for f in TASKS}, 'pooled': summ(ad)},
                         g200={**{f: summ(ag[af == f]) for f in TASKS}, 'pooled': summ(ag)},
                         frac_d_pos=fnum(float((ad > 0).mean())))
    fr = [appx[x]['frac_d_pos'] if appx.get(x) else None for x in APPX]
    consistent = bool(rej1 and all(v is not None and v > 0.5 for v in fr))
    appx['H1_rejected'] = bool(rej1)
    appx['wording'] = '三种学习器方向一致' if consistent else '结论限于 LightGBM'

    # ---------------- 作物锚点
    if not a.no_crop:
        CR, ccurves = crop_values(AC)
        CURV += ccurves
        pd.DataFrame([{c: r[c] for c in ['domain', 'primary', 'd', 'g200', 'M', 'L_saved', 'censored']} for r in CR]) \
            .to_csv(os.path.join(out, 'crop_anchor.csv'), index=False)
        crop_est = {}
        for nm, sel in (('9', [r for r in CR if r['primary']]), ('11', CR)):
            crop_est[nm] = dict(d=summ([r['d'] for r in sel]), g200=summ([r['g200'] for r in sel]),
                                M=summ([r['M'] for r in sel]), L_saved=lsaved_summary(sel),
                                e_curve=curve_summary(sel, '_e'))
        confirm['estimates']['crop'] = crop_est

    pd.DataFrame(CURV, columns=['task', 'domain', 'learner', 'quantity', 'k', 'value']) \
        .to_csv(os.path.join(out, 'curves_e_g.csv'), index=False)
    json.dump(clean(confirm), open(os.path.join(out, 'confirm.json'), 'w'), indent=1, ensure_ascii=False)
    json.dump(clean(appx), open(os.path.join(out, 'appendix.json'), 'w'), indent=1, ensure_ascii=False)
    print(json.dumps(clean(dict(H1=confirm['H1'], H2=confirm['H2'], M1=M1, q=q, n_sing=n_sing,
                                appendix=appx['wording'])), ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
