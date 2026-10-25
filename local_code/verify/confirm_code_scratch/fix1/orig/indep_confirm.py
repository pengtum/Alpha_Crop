"""
indep_confirm.py —— 确认运行方案 v5 第 7 节的独立复算：H1、H2（含 Holm）、A1、B1、X1；
另算 A3（RF 的 ΔFA(RT200)），因为回溯对账要复现它。

方案：label_expansion/CONFIRM_PLAN_v5.md（sha256[:16] = 5384fb7efa67f495，冻结）。
独立性：不导入、不读取 analyze_confirm.py 与 confirm_run.py；不导入 step0B.py / step0B2.py。
  τ 族、OA / ba_ge30、上包络插值、FA、两臂之差、D、国家值、t 推断、Holm 都在本文件内自写。
  只导入已冻结的 guidance_G.GData 读标签 y（启动时断言 guidance_G.py 的哈希）。

口径（方案第 3 节）
  概率      P_test 转 float64 后按行归一化；ℓ = log P（P = 0 → −inf）
  τ 族      ℓ − τ·log π̂；π̂ = 该臂测试集平均预测（不用标签；π̂_c = 0 的类偏移取 0）；
            τ = round(arange(−2, 2 + 1e-9, 0.05), 2)，81 点；argmax 并列取第一个
  指标      OA = 100·正确率；ba_ge30 = 100·big 类召回的平均（big = 测试集中 ≥ 30 个地块的类）
  包络      相邻 τ 点在 (OA, BA) 平面线性插值，找 OA = o 的点，取 BA 最大；
            OA 相同的平段取两端 BA 较大者；判断 o 是否落在段内的容差 1e-9
  窗口      o ∈ OA★ + {−4.0, −3.9, …, 0}（41 点）；OA★ = 同一学习器、同一特征 Z0 在 τ = 0 的 OA（两个场景共用）
  两臂之差  共同可达格点上逐点 BA 差的平均；共同可达 < 50% 记缺失
  D         锚点 = min(OA★_ae, OA★_s2)（LightGBM Z0）；两特征共同可达格点上 BA_ae − BA_s2 的平均；< 50% 记缺失
  国家值    抽样内配对求差 → 对 r 平均 → 对 F1、F2 平均（每层跳过缺失）；both = AE、S2 国家值的平均（任一缺失则缺失）
  推断      国家为单位，单样本 t（df = n − 1）；9 国 = 去掉 RO、PT；可判定：9 国至少 7 国、11 国至少 9 国有值
  Holm      H1（ΔFA(RT200)，both，9 国）与 H2（D(Z0)，11 国）的单侧 p（H0：≤ 0）；较小者在 0.0125 下检验，
            通过后另一条在 0.025 下检验；不可判定的一条按 p = 1 处理（不拒绝）。报 95% 与 97.5% CI
  B1 读法   95% CI ⊂ (−1, +1) → "补采位置的影响小于 1 pp"；否则 CI 不含 0 → 方向与大小；否则 "未能区分，区间为 [L, U]"

两种模式
  正式      --root 结果根目录（正式运行为 /Users/perng_mini/AlphaCrops_local/results/confirm_v1/）。读：
            splits/{unit}.npz 的 test、big；runs/{unit}.csv 的 learner、arm、feat、r、sel_hash、status；
            probs/{unit}/{learner}_{arm}_{feat}_r{r}.npz 的 P_test。unit = {国家}_F{1|2}，r = 0..2（Z0 写 r0）。
            单元可行性（n_test ≥ 3,000 且 |big| ≥ 2）独立重算；不可行单元剔除并披露。
            status 不为 ok 或文件缺失的拟合记为缺失，涉及它的差值为缺失，并在摘要中计数。
  --retro   G s0/s1 回溯（PC-1 raw 口径，同样的适配）：半区 ↔ 划分 s0/s1，抽样 r ↔ 重复 r0/r1（R = 2）；
            LightGBM Z0 = stage0 的 P0test_{f}_i300，RT-k = G 的 SRS(k) 重训 i300；RF Z0 = k0 的 P，RT200 = SRS k=200 的 P。
            选样对齐：G sel 文件的 sel_pos 经候选池位置映射等于 sel；sel_hash（自算）= rows_t1 / rows_rf 的 sel_hash。
            逐单元与 PC-1 的 pc1_rows.csv 核对（容差 1e-9），并核对方案第 6 节登记的 H1、H2、RF 数字。场景 B 无回溯值，不算。
输出（--out；正式默认 {root}/indep/，--retro 默认 verify/confirm_code_scratch/indep_retro/）
  indep_unit.csv      逐单元：臂的 OA★、FA、可达率、τ = 0 的 OA / BA；各差值与 D
  indep_country.csv   国家值
  indep_estimates.csv 汇总（n、均值、SD、95% / 97.5% CI、单侧 p、>0 国家数、可判定；B1 的 both 行另有读法）
  indep_tests.csv     H1、H2 的 Holm（单侧 p、检验水平、是否拒绝）
  indep_summary.txt   可读摘要（含 Holm、B1 读法；--retro 含对账）

用法：/Users/perng_mini/AlphaCrops_local/.venv/bin/python -B indep_confirm.py --root <结果根目录> [--out <目录>]
      /Users/perng_mini/AlphaCrops_local/.venv/bin/python -B indep_confirm.py --retro [--out <目录>]
"""
import os
import sys
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_v] = '2'
sys.dont_write_bytecode = True  # 不在项目目录生成 __pycache__
import argparse
import glob
import hashlib
import platform
import time

import numpy as np
import pandas as pd
import scipy
from scipy import stats

L = '/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops/label_expansion'
PLAN = os.path.join(L, 'CONFIRM_PLAN_v5.md')
PLAN_HASH = '5384fb7efa67f495'
GUIDE_PY = os.path.join(L, 'guidance_G.py')
GUIDE_HASH = '47d729be62d5d2c9'
R_G = '/Users/perng_mini/AlphaCrops_local/results/guidance_G'
R_ML = '/Users/perng_mini/AlphaCrops_local/results/guidance_G_ml'
PC1_ROWS = '/Users/perng_mini/AlphaCrops_local/verify/confirm_plan/pc1/pc1_rows.csv'
PC1_ROWS_HASH = '9e47b82a5490e9c6'
RETRO_OUT = '/Users/perng_mini/AlphaCrops_local/verify/confirm_code_scratch/indep_retro'

C = 10
MIN_N = 30                 # big：测试集中 ≥ 30 个地块的类
MIN_TEST = 3000            # 单元可行性
FEATS = ('ae', 's2')
COUNTRIES = ('AT', 'BE', 'DE', 'EE', 'LT', 'LV', 'PT', 'RO', 'SE', 'SI', 'SK')
DROP = ('RO', 'PT')
NEED = {9: 7, 11: 9}       # 可判定条件：该国家集合至少多少国有值
TAUS = np.round(np.arange(-2, 2 + 1e-9, 0.05), 2)
I_T0 = int(np.flatnonzero(TAUS == 0.0)[0])
OFFS = np.round(np.linspace(-4.0, 0.0, 41), 1)
EPS = 1e-9
MIN_COV = 0.5
KS = (50, 100, 200, 400)
K1S = (100, 300)
STRATS = ('SRS', 'EXT', 'BA')
R_DRAWS = 3
ALPHA_FAM = 0.025
TOL = 1e-9
assert len(TAUS) == 81 and TAUS[0] == -2.0 and TAUS[-1] == 2.0 and TAUS[I_T0] == 0.0
assert len(OFFS) == 41 and OFFS[0] == -4.0 and OFFS[-1] == 0.0 and np.all(np.diff(OFFS) > 0.09)

# 方案第 6 节登记的 PC-1 数字（回溯对账）：(量, 学习器, 特征, 国家集合) → 均值 [95% CI]
PC1_REGISTERED = {('dFA_RT200', 'lgb', 'both', 9): '+2.40 [+1.53, +3.27]',
                  ('D_Z0', 'lgb', 'ae-s2', 11): '+4.95 [+2.15, +7.74]',
                  ('rf_dFA_RT200', 'rf', 'both', 9): '+2.30 [+1.12, +3.48]'}


def sha(path, n=16):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()[:n]


def sel_hash(sel):
    """与 G 相同的选样哈希定义：排序后的 int64 下标的 md5 前 16 位。"""
    return hashlib.md5(np.sort(np.asarray(sel, dtype=np.int64)).tobytes()).hexdigest()[:16]


# ================================================================ 臂与估计量
def arms_of(scen_b):
    lgb = ['Z0'] + [f'RT{k}' for k in KS]
    if scen_b:
        lgb += ['B_S0'] + [f'B_{s}{k1}' for s in STRATS for k1 in K1S]
    return {'lgb': lgb, 'rf': ['Z0', 'RT200']}


def quantities(scen_b):
    """名称 → (学习器, 臂 a, 臂 b)；值 = FA(a) − FA(b)（共同可达格点上的逐点差平均）。"""
    Q = {f'dFA_RT{k}': ('lgb', f'RT{k}', 'Z0') for k in KS}      # A1；H1 = dFA_RT200
    Q['rf_dFA_RT200'] = ('rf', 'RT200', 'Z0')                    # A3（回溯对账）
    if scen_b:
        for k1 in K1S:                                           # B1
            Q[f'B1_SRS-EXT_{k1}'] = ('lgb', f'B_SRS{k1}', f'B_EXT{k1}')
            Q[f'B1_BA-SRS_{k1}'] = ('lgb', f'B_BA{k1}', f'B_SRS{k1}')
        Q['X1_S0-RT100'] = ('lgb', 'B_S0', 'RT100')              # X1
        for k1, ka in ((100, 200), (300, 400)):
            for s in STRATS:
                Q[f'X1_{s}{k1}-RT{ka}'] = ('lgb', f'B_{s}{k1}', f'RT{ka}')
    return Q


# ================================================================ 度量（自写）
def norm64(P):
    P = np.asarray(P, dtype=np.float64)
    return P / P.sum(axis=1, keepdims=True)


def tau_curve(P, yT, big):
    """P：(n, C) 归一化概率。返回 81 个 τ 上的 (OA, ba_ge30)，单位 %。"""
    with np.errstate(divide='ignore'):
        Z = np.log(P)
    pihat = P.mean(axis=0)
    lp = np.zeros(C)
    pos = pihat > 0
    lp[pos] = np.log(pihat[pos])
    n_big = np.array([np.count_nonzero(yT == c) for c in big], dtype=float)
    oa = np.empty(len(TAUS))
    ba = np.empty(len(TAUS))
    for i, t in enumerate(TAUS):
        pred = np.argmax(Z - t * lp, axis=1)
        hit = pred == yT
        oa[i] = 100.0 * np.count_nonzero(hit) / len(yT)
        n_hit = np.bincount(yT[hit], minlength=C)
        ba[i] = 100.0 * np.mean(n_hit[big] / n_big)
    return oa, ba


def envelope(oa, ba, grid):
    """逐段扫描：对每个 o，所有满足 min(o0,o1) − ε ≤ o ≤ max(o0,o1) + ε 的相邻 τ 段上插值 BA，取最大；不可达为 NaN。"""
    best = np.full(len(grid), -np.inf)
    for j in range(len(oa) - 1):
        o0, o1, b0, b1 = oa[j], oa[j + 1], ba[j], ba[j + 1]
        ins = (grid >= min(o0, o1) - EPS) & (grid <= max(o0, o1) + EPS)
        if not ins.any():
            continue
        if abs(o1 - o0) < EPS:
            v = np.full(len(grid), max(b0, b1))
        else:
            v = b0 + np.clip((grid - o0) / (o1 - o0), 0.0, 1.0) * (b1 - b0)
        best[ins] = np.maximum(best[ins], v[ins])
    best[np.isneginf(best)] = np.nan
    return best


def fa_cov(e):
    ok = np.isfinite(e)
    cov = float(ok.mean())
    return (float(e[ok].mean()) if cov >= MIN_COV else np.nan), cov


def pdiff(ea, eb):
    """两条包络在共同可达格点上的逐点差平均；共同可达 < 50% 记缺失。"""
    if ea is None or eb is None:
        return np.nan, 0.0
    ok = np.isfinite(ea) & np.isfinite(eb)
    cov = float(ok.mean())
    return (float((ea[ok] - eb[ok]).mean()) if cov >= MIN_COV else np.nan), cov


def big_of(yT):
    return np.flatnonzero(np.bincount(yT, minlength=C) >= MIN_N)


# ================================================================ 一个单元
def run_unit(meta, yT, big, load, reps, scen_b):
    """meta：country、half、unit；load(learner, arm, feat, r) → 归一化概率或 None（缺失）。返回逐单元行。"""
    Q = quantities(scen_b)
    cv = {}
    for lrn, arms in arms_of(scen_b).items():
        for f in FEATS:
            for arm in arms:
                for r in ((0,) if arm == 'Z0' else reps):
                    P = load(lrn, arm, f, r)
                    cv[(lrn, f, arm, r)] = None if P is None else tau_curve(P, yT, big)
    ostar = {(lrn, f): (cv[(lrn, f, 'Z0', 0)][0][I_T0] if cv[(lrn, f, 'Z0', 0)] is not None else np.nan)
             for lrn in arms_of(scen_b) for f in FEATS}
    rows, env = [], {}
    for (lrn, f, arm, r), c in cv.items():
        row = dict(meta, level='arm', learner=lrn, feat=f, r=r, name=arm, arm_a=arm, arm_b='',
                   OA_star=ostar[(lrn, f)], value=np.nan, cov=0.0, oa_t0=np.nan, ba_t0=np.nan, fit_missing=c is None)
        if c is not None:
            env[(lrn, f, arm, r)] = e = envelope(c[0], c[1], ostar[(lrn, f)] + OFFS)
            row['value'], row['cov'] = fa_cov(e)
            row['oa_t0'], row['ba_t0'] = c[0][I_T0], c[1][I_T0]
        rows.append(row)
    for name, (lrn, a, b) in Q.items():
        for f in FEATS:
            for r in reps:
                v, cov = pdiff(env.get((lrn, f, a, 0 if a == 'Z0' else r)), env.get((lrn, f, b, 0 if b == 'Z0' else r)))
                rows.append(dict(meta, level='q', learner=lrn, feat=f, r=r, name=name, arm_a=a, arm_b=b,
                                 OA_star=ostar[(lrn, f)], value=v, cov=cov))
    # D(Z0)：LightGBM，锚点 = 两特征 Z0 的 OA★ 较小者
    anc = min(ostar[('lgb', 'ae')], ostar[('lgb', 's2')]) if all(
        np.isfinite(ostar[('lgb', f)]) for f in FEATS) else np.nan
    ed = {f: (envelope(*cv[('lgb', f, 'Z0', 0)], anc + OFFS) if cv[('lgb', f, 'Z0', 0)] is not None else None)
          for f in FEATS}
    v, cov = pdiff(ed['ae'], ed['s2'])
    rows.append(dict(meta, level='D', learner='lgb', feat='ae-s2', r=0, name='D_Z0', arm_a='Z0', arm_b='Z0',
                     OA_star=anc, value=v, cov=cov))
    return rows


# ================================================================ 数据：正式模式
def confirm_units(root, y, reps, log):
    """逐单元产出 (meta, yT, big, load)；不可行单元只记入 excluded。"""
    excluded, n_missing_fit = [], []
    for tgt in COUNTRIES:
        for h in (1, 2):
            unit = f'{tgt}_F{h}'
            Z = np.load(os.path.join(root, 'splits', f'{unit}.npz'))
            test = np.asarray(Z['test'], dtype=np.int64)
            yT = np.asarray(y[test], dtype=np.int64)
            big = big_of(yT)
            if len(test) < MIN_TEST or len(big) < 2:
                excluded.append(f'{unit}（n_test {len(test)}，|big| {len(big)}）')
                continue
            assert np.array_equal(np.sort(np.asarray(Z['big'], dtype=np.int64)), big), (unit, '存档的 big 与重算不一致')
            runs = pd.read_csv(os.path.join(root, 'runs', f'{unit}.csv'), dtype={'sel_hash': str})
            ok = runs[runs.status == 'ok']
            ok_keys = set(zip(ok.learner, ok.arm, ok.feat, ok.r.astype(int)))
            # 同一抽样内：AE 与 S2 同一批标签；RF 与 LightGBM 的 RT200 同一批标签
            sel = ok[~ok.arm.isin(['Z0', 'TGT'])]
            nh = sel.groupby(['arm', 'r']).sel_hash.nunique()
            assert (nh == 1).all(), (unit, '同一臂同一抽样的 sel_hash 不唯一', nh[nh != 1])

            def load(lrn, arm, f, r, unit=unit, n=len(test), ok_keys=ok_keys):
                if (lrn, arm, f, r) not in ok_keys:
                    n_missing_fit.append((unit, lrn, arm, f, r))
                    return None
                P = np.load(os.path.join(root, 'probs', unit, f'{lrn}_{arm}_{f}_r{r}.npz'))['P_test']
                assert P.shape == (n, C) and np.isfinite(P).all() and (P >= 0).all(), (unit, lrn, arm, f, r)
                assert np.abs(P.astype(np.float64).sum(1) - 1).max() < 1e-3, (unit, lrn, arm, f, r, '行和')
                return norm64(P)
            log(f'{unit}：n_test {len(test)}，big {big.tolist()}')
            yield dict(country=tgt, half=h, unit=unit), yT, big, load
    confirm_units.excluded, confirm_units.missing = excluded, n_missing_fit


# ================================================================ 数据：回溯模式（G s0/s1）
def retro_units(y, log):
    GR = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(os.path.join(R_G, 'rows_t1_*.csv')))], ignore_index=True)
    GR = GR[(GR.cond == 'none') & (GR.method == 'srs')]
    MR = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(os.path.join(R_ML, 'rows_rf_*.csv')))], ignore_index=True)
    MR = MR[(MR.learner == 'rf') & (MR.cond == 'none') & (MR.method == 'srs') & (MR.k == 200)]
    n_assert = 0
    for tgt in COUNTRIES:
        for split in (0, 1):
            S = np.load(os.path.join(R_G, 'stage0', f'{tgt}_s{split}.npz'))
            allT, test, n_cand = S['allT'], S['test'], len(S['cand'])
            assert np.array_equal(allT[n_cand:], test)
            yT = np.asarray(y[test], dtype=np.int64)
            big = big_of(yT)
            H = {}
            for rep in (0, 1):
                rng = np.random.default_rng(10_000 * split + 100 * rep + 7)       # G 的候选池位置（与阶段 A 相同）
                pool_pos = np.sort(rng.choice(n_cand, min(30_000, n_cand), replace=False))
                for k in KS:
                    z = np.load(os.path.join(R_G, 'sel', f'{tgt}_s{split}_r{rep}_none_srs_k{k}.npz'))
                    assert len(z['sel']) == k and np.array_equal(allT[pool_pos][z['sel_pos']], z['sel'])
                    h = sel_hash(z['sel'])
                    for f in FEATS:
                        g = GR[(GR.target == tgt) & (GR.split == split) & (GR.rep == rep) & (GR.clf == f) & (GR.k == k)]
                        assert len(g) == 1 and g.sel_hash.iloc[0] == h, (tgt, split, rep, f, k)
                        n_assert += 1
                        if k == 200:
                            m = MR[(MR.target == tgt) & (MR.split == split) & (MR.rep == rep) & (MR.clf == f)]
                            assert len(m) == 1 and m.status.iloc[0] == 'ok' and m.sel_hash.iloc[0] == h
                            n_assert += 1
                    H[(rep, k)] = h

            def load(lrn, arm, f, r, S=S, H=H, tgt=tgt, split=split):
                if lrn == 'lgb' and arm == 'Z0':
                    return norm64(S[f'P0test_{f}_i300'])
                if lrn == 'lgb':
                    k = int(arm[2:])
                    return norm64(np.load(os.path.join(R_G, 'probs', f'{tgt}_s{split}_r{r}_{f}_k{k}_{H[(r, k)]}.npz'))
                                  ['i300'])
                if arm == 'Z0':
                    return norm64(np.load(os.path.join(R_ML, 'probs', 'rf', f'{tgt}_s{split}_{f}_k0.npz'))['P'])
                return norm64(np.load(os.path.join(R_ML, 'probs', 'rf',
                                                   f'{tgt}_s{split}_r{r}_{f}_k200_{H[(r, 200)]}.npz'))['P'])
            log(f'{tgt}_s{split}：n_test {len(test)}，big {big.tolist()}')
            yield dict(country=tgt, half=split, unit=f'{tgt}_s{split}'), yT, big, load
    retro_units.n_assert = n_assert


# ================================================================ 汇总（自写）
def country_table(U):
    """逐单元差值 → 国家值（行 = (name, learner, feat)，列 = 国家）；另加 both。"""
    X = U[U.level.isin(['q', 'D'])]
    per_half = X.groupby(['name', 'learner', 'feat', 'country', 'half']).value.mean()   # 对 r 平均
    cv = per_half.groupby(level=['name', 'learner', 'feat', 'country']).mean().unstack('country')
    cv = cv.reindex(columns=list(COUNTRIES))
    both = []
    for (name, lrn), g in cv.groupby(level=['name', 'learner']):
        if name == 'D_Z0':
            continue
        s = (g.xs('ae', level='feat').iloc[0] + g.xs('s2', level='feat').iloc[0]) / 2   # 任一缺失 → 缺失
        both.append(s.rename((name, lrn, 'both')))
    cv = pd.concat([cv, pd.DataFrame(both)])
    cv.index = pd.MultiIndex.from_tuples(cv.index, names=['name', 'learner', 'feat'])
    return cv


def tsum(v, nset):
    """v：以国家为索引的 Series。返回单样本 t 汇总（95% 与 97.5% 双侧 CI、单侧 p）。"""
    v = v.reindex([c for c in COUNTRIES if nset == 11 or c not in DROP]).dropna()
    x = v.to_numpy(dtype=float)
    n = len(x)
    out = dict(n=n, decidable=n >= NEED[nset], mean=np.nan, sd=np.nan, lo95=np.nan, hi95=np.nan,
               lo975=np.nan, hi975=np.nan, t=np.nan, p_one=np.nan, n_pos=int((x > 0).sum()))
    if n >= 2:
        m, sd = x.mean(), x.std(ddof=1)
        se = sd / np.sqrt(n)
        out.update(mean=m, sd=sd)
        for lv, tag in ((0.95, '95'), (0.975, '975')):
            q = stats.t.ppf(1 - (1 - lv) / 2, n - 1)
            out[f'lo{tag}'], out[f'hi{tag}'] = m - q * se, m + q * se
        if se > 0:
            out['t'] = m / se
            out['p_one'] = float(stats.t.sf(m / se, n - 1))
    elif n == 1:
        out['mean'] = x[0]
    return out


def holm(pvals):
    """pvals：名称 → 单侧 p（不可判定为 NaN，按 1 处理）。返回 名称 → (检验水平, 是否拒绝)。"""
    order = sorted(pvals, key=lambda k: (pvals[k] if np.isfinite(pvals[k]) else 1.0, k))
    res, go = {}, True
    for i, k in enumerate(order):
        a = ALPHA_FAM / (len(order) - i)
        rej = bool(go and np.isfinite(pvals[k]) and pvals[k] <= a)
        res[k] = (a, rej)
        go = rej
    return res


def read_b1(s):
    if not s['decidable']:
        return '不可判定'
    lo, hi = s['lo95'], s['hi95']
    if lo > -1 and hi < 1:
        return '补采位置的影响小于 1 pp'
    if lo > 0 or hi < 0:
        return f'{"正" if s["mean"] > 0 else "负"}向，{s["mean"]:+.2f} pp'
    return f'未能区分，区间为 [{lo:+.2f}, {hi:+.2f}]'


def fmt(s, ci='95'):
    if s['n'] < 2:
        return f'n={s["n"]}（不足 2 国）'
    tag = '' if s['decidable'] else '【不可判定】'
    return (f'{s["mean"]:+.2f} [{s[f"lo{ci}"]:+.2f}, {s[f"hi{ci}"]:+.2f}]，SD {s["sd"]:.2f}，'
            f'>0 {s["n_pos"]}/{s["n"]}{tag}')


# ================================================================ 回溯对账（PC-1）
def retro_check(U, CV, EST, P):
    assert sha(PC1_ROWS) == PC1_ROWS_HASH, 'pc1_rows.csv 的哈希与登记值不一致'
    R = pd.read_csv(PC1_ROWS, float_precision='round_trip')
    ok_all = True

    def cmp(mine, ref, keys, cols, label):
        nonlocal ok_all
        M = mine.merge(ref, on=keys, how='outer', suffixes=('', '_pc1'), indicator=True)
        n_unmatched = int((M._merge != 'both').sum())
        worst, nan_bad = 0.0, 0
        for c in cols:
            a, b = M[c].to_numpy(float), M[f'{c}_pc1'].to_numpy(float)
            nan_bad += int((np.isnan(a) != np.isnan(b)).sum())
            fin = np.isfinite(a) & np.isfinite(b)
            if fin.any():
                worst = max(worst, float(np.abs(a[fin] - b[fin]).max()))
        ok = n_unmatched == 0 and nan_bad == 0 and worst <= TOL
        ok_all &= ok
        P(f'  {label}：{len(M)} 行，未配对 {n_unmatched}，NaN 不一致 {nan_bad}，最大 |Δ| {worst:.2e}（{"/".join(cols)}）'
          f' → {"通过" if ok else "不通过"}')

    def rep_expand(X):
        """我的行（r-less 的 Z0 与 D 各复制到 r0、r1）→ PC-1 单元名 {国}_s{划分}_r{重复}。"""
        parts = []
        for rep in (0, 1):
            Y = X[(X.r == rep) | X.name.isin(['Z0', 'D_Z0'])].copy()
            Y['unit'] = Y.country + '_s' + Y.half.astype(str) + f'_r{rep}'
            parts.append(Y)
        return pd.concat(parts, ignore_index=True)

    P('== 回溯对账：与 PC-1（pc1_rows.csv，sha256[:16] ' + PC1_ROWS_HASH + '）逐单元核对，容差 1e-9 ==')
    A = rep_expand(U[U.level == 'arm']).rename(columns={'feat': 'clf'})
    RA = R[(R.level == 'arm') & (((R.learner == 'lgb') & R.name.isin(['Z0'] + [f'RT{k}' for k in KS])) |
                                 ((R.learner == 'rf') & R.name.isin(['Z0', 'RT200'])))]
    cols = ['OA_star', 'value', 'cov', 'oa_t0', 'ba_t0']
    cmp(A[['learner', 'unit', 'clf', 'name'] + cols], RA[['learner', 'unit', 'clf', 'name'] + cols],
        ['learner', 'unit', 'clf', 'name'], cols, '臂（OA★、FA、可达率、τ=0 的 OA / BA）')
    Qd = rep_expand(U[U.level == 'q']).rename(columns={'feat': 'clf'})
    Qd['name'] = Qd.name.str.replace('rf_', '', regex=False)
    RD = R[(R.level == 'diff') & (((R.learner == 'lgb') & R.name.isin([f'dFA_RT{k}' for k in KS])) |
                                  ((R.learner == 'rf') & (R.name == 'dFA_RT200')))]
    cols = ['value', 'cov']
    cmp(Qd[['learner', 'unit', 'clf', 'name'] + cols], RD[['learner', 'unit', 'clf', 'name'] + cols],
        ['learner', 'unit', 'clf', 'name'], cols, 'ΔFA(RT-k)（LightGBM）与 RF ΔFA(RT200)')
    Dd = rep_expand(U[U.level == 'D'])
    RDD = R[(R.level == 'D') & (R.learner == 'lgb') & (R.name == 'D_Z0')]
    cols = ['OA_star', 'value', 'cov']
    cmp(Dd[['learner', 'unit', 'name'] + cols], RDD[['learner', 'unit', 'name'] + cols],
        ['learner', 'unit', 'name'], cols, 'D(Z0)（LightGBM；锚点与值）')

    # 国家层：按 PC-1 的定义（该国可用单元的平均；both = 两特征平均）从 pc1_rows 重算，与本脚本的国家值比较
    P('  国家值与汇总（PC-1 定义：该国 4 个单元平均）：')
    items = [(f'dFA_RT{k}', 'lgb', f'dFA_RT{k}', 'diff') for k in KS] + [('rf_dFA_RT200', 'rf', 'dFA_RT200', 'diff'),
                                                                         ('D_Z0', 'lgb', 'D_Z0', 'D')]
    for mine, lrn, ref, lev in items:
        X = R[(R.level == lev) & (R.learner == lrn) & (R.name == ref)]
        if lev == 'D':
            ref_cv = {'ae-s2': X.groupby('target').value.mean()}
        else:
            ref_cv = {f: X[X.clf == f].groupby('target').value.mean() for f in FEATS}
            ref_cv['both'] = (ref_cv['ae'] + ref_cv['s2']) / 2
        for f, rv in ref_cv.items():
            mv = CV.loc[(mine, lrn, f)].reindex(list(COUNTRIES))
            rv = rv.reindex(list(COUNTRIES))
            d_cv = float(np.nanmax(np.abs(mv.to_numpy() - rv.to_numpy())))
            nan_bad = int((mv.isna() != rv.isna()).sum())
            d_est = 0.0
            for nset in (9, 11):
                a, b = tsum(mv, nset), tsum(rv, nset)
                d_est = max(d_est, *(abs(a[c] - b[c]) for c in ('mean', 'lo95', 'hi95', 'sd')))
            ok = d_cv <= TOL and d_est <= TOL and nan_bad == 0
            ok_all &= ok
            P(f'    {mine:13s} {lrn:3s} {f:5s} 国家值最大 |Δ| {d_cv:.2e}，汇总（均值 / SD / CI，9 与 11 国）最大 |Δ| '
              f'{d_est:.2e} → {"通过" if ok else "不通过"}')
    P('  方案第 6 节登记值（两位小数）：')
    for (name, lrn, f, nset), reg in PC1_REGISTERED.items():
        s = EST[(name, lrn, f, nset)]
        got = f'{s["mean"]:+.2f} [{s["lo95"]:+.2f}, {s["hi95"]:+.2f}]'
        ok = got == reg
        ok_all &= ok
        P(f'    {name:13s} {lrn:3s} {f:5s} {nset} 国：本脚本 {got}；登记 {reg} → {"一致" if ok else "不一致"}')
    P(f'  回溯对账结论：{"全部通过" if ok_all else "有不通过项"}')
    return ok_all


# ================================================================ 主程序
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', help='结果根目录（正式模式）')
    ap.add_argument('--retro', action='store_true', help='G s0/s1 回溯，复现 PC-1')
    ap.add_argument('--out', help='输出目录')
    ap.add_argument('--R', type=int, default=R_DRAWS, help='每单元抽样次数（方案 R = 3；只在测试时改）')
    a = ap.parse_args()
    assert a.retro != bool(a.root), '--root 与 --retro 二选一'
    out = a.out or (RETRO_OUT if a.retro else os.path.join(a.root, 'indep'))
    os.makedirs(out, exist_ok=True)
    assert sha(PLAN) == PLAN_HASH, '方案 v5 的哈希与登记值不一致'
    assert sha(GUIDE_PY) == GUIDE_HASH, 'guidance_G.py 的哈希与冻结值不一致'
    sys.path.insert(0, L)
    from guidance_G import GData
    t0 = time.time()

    def log(s):
        print(f'[{time.time() - t0:6.0f}s] {s}', flush=True)

    y = np.asarray(GData().y)
    reps = (0, 1) if a.retro else tuple(range(a.R))
    scen_b = not a.retro
    src = retro_units(y, log) if a.retro else confirm_units(a.root, y, reps, log)
    rows = []
    for meta, yT, big, load in src:
        rows += run_unit(meta, yT, big, load, reps, scen_b)
    U = pd.DataFrame(rows)
    CV = country_table(U)
    Q = quantities(scen_b)
    EST, est_rows = {}, []
    for (name, lrn, f) in CV.index:
        for nset in (9, 11):
            s = tsum(CV.loc[(name, lrn, f)], nset)
            EST[(name, lrn, f, nset)] = s
            est_rows.append(dict(name=name, learner=lrn, feat=f, set=nset, **s))

    lines = []
    P = lines.append
    me = os.path.abspath(__file__)
    P('indep_confirm.py：确认运行方案 v5 第 7 节的独立复算（H1、H2、A1、B1、X1；另算 A3 = RF ΔFA(RT200)）'
      + ('【--retro：G s0/s1 回溯，C 级，只作对账】' if a.retro else ''))
    P(f'方案 CONFIRM_PLAN_v5.md sha256[:16] {sha(PLAN)}；indep_confirm.py {sha(me)}；guidance_G.py {sha(GUIDE_PY)}'
      f'（只用 GData 读 y）；Python {platform.python_version()}、numpy {np.__version__}、scipy {scipy.__version__}、'
      f'pandas {pd.__version__}')
    P(f'输入：{"G s0/s1（" + R_G + "、" + R_ML + "）" if a.retro else a.root}；抽样 r = {list(reps)}')
    nU = U.unit.nunique()
    P(f'单元 {nU} 个')
    if a.retro:
        P(f'选样对齐断言通过 {retro_units.n_assert} 次（sel_pos 经候选池位置映射 = sel；自算 sel_hash = rows_t1 与 rows_rf）')
    else:
        P(f'剔除单元（n_test < {MIN_TEST} 或 |big| < 2）：{"、".join(confirm_units.excluded) or "无"}')
        miss = confirm_units.missing
        P(f'缺失拟合（runs 中 status ≠ ok 或无记录）：{len(miss)} 个' + (f'：{miss}' if miss else ''))
    qm = U[U.level.isin(['q', 'D'])]
    nm = qm.groupby('name').value.apply(lambda v: int(v.isna().sum()))
    P('逐单元差值缺失（共同可达 < 50% 或拟合缺失）：' + ('无' if nm.sum() == 0 else
                                               '，'.join(f'{k} {v}' for k, v in nm.items() if v)))
    P('格式：均值 [CI]，SD，>0 国家数 / 国家数；单位 pp。国家值 = 抽样内配对差 → 对 r 平均 → 对两个半区平均；'
      'both = AE、S2 国家值的平均。9 国 = 去掉 RO、PT。')
    P('')

    # ---- H1、H2 与 Holm
    P('######## 确认性检验（Holm，族 α = 0.025，单侧 H0：≤ 0）########')
    H = {'H1': ('dFA_RT200', 'lgb', 'both', 9), 'H2': ('D_Z0', 'lgb', 'ae-s2', 11)}
    pv = {h: (EST[k]['p_one'] if EST[k]['decidable'] else np.nan) for h, k in H.items()}
    HR = holm(pv)
    for h, (name, lrn, f, nset) in H.items():
        s = EST[(name, lrn, f, nset)]
        lab = 'ΔFA(RT200) = FA(RT200) − FA(Z0)，LightGBM，both' if h == 'H1' else 'D(Z0)（AE − S2），LightGBM'
        P(f'{h} {lab}，主集 {nset} 国：')
        P(f'  95% CI   {fmt(s, "95")}')
        P(f'  97.5% CI {fmt(s, "975")}')
        P(f'  t = {s["t"]:.4f}，df = {s["n"] - 1}，单侧 p = {s["p_one"]:.6g}；Holm 检验水平 {HR[h][0]:g} → '
          f'{"拒绝 H0" if HR[h][1] else ("不可判定" if not s["decidable"] else "不拒绝 H0")}')
        other = 11 if nset == 9 else 9
        P(f'  {other} 国同报：{fmt(EST[(name, lrn, f, other)])}')
        cvv = CV.loc[(name, lrn, f)]
        P('  逐国：' + '，'.join(f'{c} {cvv[c]:+.2f}' if np.isfinite(cvv[c]) else f'{c} 缺失' for c in COUNTRIES))
    P('')

    def block(title, names, feats=('both', 'ae', 's2'), reader=None):
        P(title)
        for name in names:
            lrn = Q[name][0]
            P(f'-- {name}（{Q[name][1]} − {Q[name][2]}，{lrn}）--')
            for f in feats:
                for nset in (9, 11):
                    s = EST[(name, lrn, f, nset)]
                    extra = f'   读法：{reader(s)}' if reader and f == 'both' else ''
                    P(f'  {f:4s} {nset} 国  {fmt(s)}{extra}')
        P('')

    block('######## A1 剂量曲线：ΔFA(RT-k)，LightGBM（95% CI；主 = 9 国）########', [f'dFA_RT{k}' for k in KS])
    block('######## A3（回溯对账用）：RF ΔFA(RT200)（95% CI）########', ['rf_dFA_RT200'])
    if scen_b:
        block('######## B1 补在哪里：FA(SRS) − FA(EXT)，FA(BA) − FA(SRS)（95% CI；主 = 9 国）########',
              [f'B1_{c}_{k1}' for k1 in K1S for c in ('SRS-EXT', 'BA-SRS')], reader=read_b1)
        block('######## X1 同标签总数的比较（95% CI；主 = 9 国）########',
              ['X1_S0-RT100'] + [f'X1_{s}{k1}-RT{ka}' for k1, ka in ((100, 200), (300, 400)) for s in STRATS])
    else:
        P('场景 B（B1、X1）没有回溯值，--retro 不计算。')
        P('')

    ok = True
    if a.retro:
        ok = retro_check(U, CV, EST, P)
        P('')

    ES = pd.DataFrame(est_rows)
    ES['reading'] = [read_b1(EST[(n, l, f, s)]) if n.startswith('B1_') and f == 'both' else ''
                     for n, l, f, s in zip(ES.name, ES.learner, ES.feat, ES.set)]
    TS = pd.DataFrame([dict(hyp=h, name=k[0], learner=k[1], feat=k[2], set=k[3], p_one=pv[h], alpha_step=HR[h][0],
                            reject=HR[h][1]) for h, k in H.items()])
    paths = {'indep_unit.csv': U, 'indep_country.csv': CV.reset_index(), 'indep_estimates.csv': ES, 'indep_tests.csv': TS}
    for fn, df in paths.items():
        df.to_csv(os.path.join(out, fn), index=False)
    P('== 输出文件 sha256 ==')
    for fn in paths:
        P(f'  {fn} {sha(os.path.join(out, fn), 64)}')
    P(f'总耗时 {time.time() - t0:.0f}s')
    sp = os.path.join(out, 'indep_summary.txt')
    open(sp, 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(f'indep_summary.txt sha256 {sha(sp, 64)}')
    if not ok:
        sys.exit(1)


if __name__ == '__main__':
    main()
