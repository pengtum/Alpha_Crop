"""
确认运行登记前的不训练分析 · AE 线 · ae_mechanism 流的独立复核实现（只依据规格，不读主实现及其输出）。
规格：label_expansion/PRECONF_ANALYSES_SPEC_2026-09-29.md（sha256[:16] 登记值 d0906bd1417839d2）。
事后分析，C 级、描述性；不训练任何作物分类器；只读已存概率与 stage0 几何；测试标签只用于评估。

本文件覆盖的量（编排脚本指定的独立复核范围）：
  AE2  检验 3 的国内（within）与跨境（loco）测试概率，c300（固定 300 棵）为主、早停（es）为次：
       各自做 τ 族；锚点 = 该模型两种特征 τ=0 时 OA 的较小者；W4 = [锚点−4, 锚点]，步长 0.1（41 点）；
       逐点 BA_ae(o) − BA_s2(o)，在两条曲线共同可达的点上取平均；共同可达 <50% 记缺失；
       得到 D_dom（within）、D_xb（loco）与 D_xb − D_dom；对照：τ=0 的 AE−S2（BA、OA）。
  AE6  类无关混合指数 M_all：划分 0 的 stage0 几何（G_f 的候选行 = 前 n_cand 行；GF_f = 外国池；32 维联合 PCA），
       目标候选与外国池各抽 2,000 个，合并后对每个目标样本找 10 个欧氏最近邻（排除自身），外国样本占比的平均；
       报各国 M_all(ae)、M_all(s2) 与 AE−S2。

复用（import，已审计，不改其文件）：
  step0B2.tau_preds_grid + TAUS_NEW（τ ∈ [−2, 2] 步长 0.05）；step0B.log_or_neginf（精确 log）、step0B.norm64、
  step0B.Scorer（OA 与 ba_ge30；类集合 = 测试集中 ≥ 30 个地块的类）、step0B.envelope（单曲线上包络 = OA=o 的各段交点取 BA 最大）、
  step0B.OFFS（W4 的 41 个偏移）。π̂ = 该臂测试集平均预测（与 step0B2 相同：exp(Z − 行最大) 归一化后取均值）。
本流自写：窗口比较（共同可达、平均差、占优判定）、国内/跨境配对、国家层汇总、M_all 近邻混合指数、全部核对。

用法：/Users/perng_mini/AlphaCrops_local/.venv/bin/python -B indep_ae_mechanism.py
"""
import os
import sys
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_v] = '2'
sys.dont_write_bytecode = True  # 不在项目目录或 partB/partB2 写 __pycache__
import hashlib
import json
import time

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
PARTB2 = '/Users/perng_mini/AlphaCrops_local/verify/step0/partB2'
PARTB = '/Users/perng_mini/AlphaCrops_local/verify/step0/partB'
sys.path.insert(0, PARTB2)
sys.path.insert(0, PARTB)
import step0B2 as B2  # noqa: E402  导入时 step0B 会 chdir 到项目目录（只读）并导入 guidance_G
B = B2.B
G = B.G

SPEC = os.path.join(B.L, 'PRECONF_ANALYSES_SPEC_2026-09-29.md')
SPEC_HASH = 'd0906bd1417839d2'
R3 = '/Users/perng_mini/AlphaCrops_local/results/check3'
R_G = '/Users/perng_mini/AlphaCrops_local/results/guidance_G'
CAP_CSV = '/Users/perng_mini/AlphaCrops_local/results/check3_capacity_arms.csv'
DATASET_V2 = '/Users/perng_mini/AlphaCrops_local/data/dataset_v2.parquet'
CURVES_B2 = os.path.join(PARTB2, 'curves_B2.csv')
INTEG_UNITS = ('/Users/perng_mini/AlphaCrops_local/verify/storyline/ae_s2_integration_20260929/'
               'ae_s2_evidence/matched_oa_units.csv')

C = 10
FEATS = ['ae', 's2']
COUNTRIES = ['AT', 'BE', 'DE', 'EE', 'LT', 'LV', 'PT', 'RO', 'SE', 'SI', 'SK']
DROP = ['RO', 'PT']
C9 = [c for c in COUNTRIES if c not in DROP]
TAUS = B2.TAUS_NEW
I_T0 = int(np.flatnonzero(TAUS == 0.0)[0])
MIN_COV = 0.5
EPS = 1e-9
CAPS = {'c300': '_c300', 'es': ''}          # c300 为主，es（早停）为次
MODELS = {'dom': 'within', 'xb': 'loco'}    # 国内模型 / 跨境模型
N_ALL = 2000                                # AE6 类无关版本每侧抽样数
K_NN = 10
N_ALT_SEEDS = 5                             # AE6 抽样噪声描述用的额外种子数（方案外）


def sha16(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()[:16]


# ================================================================ 通用：τ 曲线与窗口比较
def tau_curve(P, sc):
    """P：(n, C) 概率（float64、行和为 1）。返回 (oa (nτ,), ba (nτ,))，与 step0B2.run_unit 的曲线计算相同。"""
    Z = B.log_or_neginf(P)
    Pa = np.exp(Z - Z.max(1, keepdims=True))
    Pa /= Pa.sum(1, keepdims=True)
    pihat = Pa.mean(0)
    PH = B2.tau_preds_grid(Z, pihat, TAUS)
    return sc.curve(PH)


def offsets(width, step=0.1):
    return np.round(np.arange(-width, 0.0 + 1e-9, step), 1)


assert np.array_equal(offsets(4), B.OFFS)


def ba_on_grid(cv, grid):
    """单条曲线在 o 网格上的上包络 BA（不可达为 NaN）；直接用 step0B.envelope。"""
    return B.envelope([cv], grid)[0]


def ba_on_grid_own(cv, grid):
    """独立的逐段循环实现（只作核对）：OA=o 的所有相邻 τ 段交点线性插值取 BA 最大；OA 平段取两端 BA 较大者。"""
    oa, ba = cv
    out = np.full(len(grid), np.nan)
    for g, o in enumerate(grid):
        best = -np.inf
        for j in range(len(oa) - 1):
            a0, a1, b0, b1 = oa[j], oa[j + 1], ba[j], ba[j + 1]
            if not (min(a0, a1) - EPS <= o <= max(a0, a1) + EPS):
                continue
            if abs(a1 - a0) < EPS:
                v = max(b0, b1)
            else:
                lm = min(max((o - a0) / (a1 - a0), 0.0), 1.0)
                v = b0 + lm * (b1 - b0)
            best = max(best, v)
        if np.isfinite(best):
            out[g] = best
    return out


def window_compare(cv_ae, cv_s2, anchor, width=4):
    """同 OA 下 AE−S2：窗口 [anchor−width, anchor] 步长 0.1；两曲线共同可达 <50% 记缺失。"""
    grid = anchor + offsets(width)
    ba_a, ba_s = ba_on_grid(cv_ae, grid), ba_on_grid(cv_s2, grid)
    ok = np.isfinite(ba_a) & np.isfinite(ba_s)
    cov = float(ok.mean())
    if cov >= MIN_COV:
        d = float(np.mean(ba_a[ok] - ba_s[ok]))
        ae_dom = bool((ba_a[ok] >= ba_s[ok]).all())
        s2_dom = bool((ba_s[ok] >= ba_a[ok]).all())
    else:
        d, ae_dom, s2_dom = np.nan, np.nan, np.nan
    return dict(D=d, cov=cov, reach_ae=float(np.isfinite(ba_a).mean()), reach_s2=float(np.isfinite(ba_s).mean()),
                ae_dom=ae_dom, s2_dom=s2_dom), (grid, ba_a, ba_s)


# ================================================================ 汇总
def tsum(v):
    v = pd.Series(v, dtype=float).dropna()
    n = len(v)
    if n < 2:
        return dict(n=n, mean=np.nan, lo95=np.nan, hi95=np.nan, sd=np.nan, npos=int((v > 0).sum()), p=np.nan)
    m, sd = float(v.mean()), float(v.std(ddof=1))
    se = sd / np.sqrt(n)
    h = stats.t.ppf(0.975, n - 1) * se
    p = float(2 * stats.t.sf(abs(m / se), n - 1)) if se > 0 else np.nan
    return dict(n=n, mean=m, lo95=m - h, hi95=m + h, sd=sd, npos=int((v > 0).sum()), p=p)


def fmt(s, nd=2):
    if s['n'] < 2:
        return f'n={s["n"]}（不足 2 国）'
    return (f'{s["mean"]:+.{nd}f} [{s["lo95"]:+.{nd}f}, {s["hi95"]:+.{nd}f}]，>0 {s["npos"]}/{s["n"]}，'
            f'SD {s["sd"]:.2f}，p={s["p"]:.3f}')


def reading_ae2(s):
    if s['n'] < 2 or not np.isfinite(s['lo95']):
        return '无法判定'
    if s['lo95'] > 0:
        return '下界 > 0：AE 的前沿优势在跨境时放大'
    if s['hi95'] < 0:
        return '上界 < 0：跨境时缩小'
    return 'CI 覆盖 0：没有检出放大，写成"AE 在国内与跨境都抬高前沿"'


# ================================================================ AE2
def load_check3_probs(tgt, f, model, cap):
    P = np.load(os.path.join(R3, f'{tgt}_{f}_{model}{CAPS[cap]}_test.npy'))
    s = P.astype(np.float64).sum(1)
    assert np.all(np.abs(s - 1) < 1e-4), (tgt, f, model, cap)
    return B.norm64(P)


def run_ae2(log):
    rows, curve_rows, chk = [], [], []
    for tgt in COUNTRIES:
        sp = np.load(os.path.join(R3, f'{tgt}_split.npz'))
        yT = np.asarray(sp['y_test'])
        big = np.flatnonzero(np.bincount(yT, minlength=C) >= G.MIN_N)
        sc = B.Scorer(yT, big)
        for cap in CAPS:
            for mk, model in MODELS.items():
                cvs = {}
                for f in FEATS:
                    P = load_check3_probs(tgt, f, model, cap)
                    assert len(P) == len(yT)
                    cvs[f] = tau_curve(P, sc)
                    oa, ba = cvs[f]
                    curve_rows += [dict(target=tgt, cap=cap, model=mk, clf=f, tau=TAUS[i], oa=oa[i], ba_ge30=ba[i])
                                   for i in range(len(TAUS))]
                    trained = int((P.max(0) > 0).sum())
                    chk.append(dict(target=tgt, cap=cap, model=mk, clf=f, oa_t0=oa[I_T0], ba_t0=ba[I_T0],
                                    n_test=len(yT), n_big=len(big), n_trained=trained,
                                    oa_min=float(oa.min()), oa_max=float(oa.max()),
                                    tau_oamax=float(TAUS[int(oa.argmax())])))
                oa0 = {f: cvs[f][0][I_T0] for f in FEATS}
                ba0 = {f: cvs[f][1][I_T0] for f in FEATS}
                anchor = min(oa0['ae'], oa0['s2'])
                res, (grid, ba_a, ba_s) = window_compare(cvs['ae'], cvs['s2'], anchor, 4)
                # 独立逐段实现核对包络
                own_a, own_s = ba_on_grid_own(cvs['ae'], grid), ba_on_grid_own(cvs['s2'], grid)
                dif = np.r_[own_a - ba_a, own_s - ba_s]
                env_diff = float(np.nanmax(np.abs(dif))) if np.isfinite(dif).any() else 0.0
                env_nan_same = bool(np.array_equal(np.isnan(own_a), np.isnan(ba_a)) and
                                    np.array_equal(np.isnan(own_s), np.isnan(ba_s)))
                row = dict(target=tgt, cap=cap, model=mk, anchor=anchor, anchor_feat='ae' if oa0['ae'] <= oa0['s2'] else 's2',
                           oa0_ae=oa0['ae'], oa0_s2=oa0['s2'], ba0_ae=ba0['ae'], ba0_s2=ba0['s2'],
                           dBA_t0=ba0['ae'] - ba0['s2'], dOA_t0=oa0['ae'] - oa0['s2'],
                           env_own_maxdiff=env_diff, env_own_nan_same=env_nan_same, **res)
                # 方案外敏感性：窗口宽 2、8；锚点分别取 OA★_ae 与 OA★_s2
                for w in (2, 8):
                    row[f'D_w{w}'] = window_compare(cvs['ae'], cvs['s2'], anchor, w)[0]['D']
                row['D_anc_ae'] = window_compare(cvs['ae'], cvs['s2'], oa0['ae'], 4)[0]['D']
                row['D_anc_s2'] = window_compare(cvs['ae'], cvs['s2'], oa0['s2'], 4)[0]['D']
                rows.append(row)
        log(f'AE2 {tgt} 完成')
    return pd.DataFrame(rows), pd.DataFrame(curve_rows), pd.DataFrame(chk)


# ================================================================ AE6
def seed_of(s):
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16)


def mixing_index(A_t, A_f, n, seed, k=K_NN, check_kdtree=False):
    """目标与外国各抽 min(n, N) 个（先目标后外国，同一 rng，无放回）；合并后对每个目标样本找 k 个欧氏最近邻（排除自身），
    返回外国样本所占比例的平均。距离在 float64 上精确计算，并列按合并样本中的位置稳定排序。"""
    rng = np.random.default_rng(seed)
    it = rng.choice(len(A_t), min(n, len(A_t)), replace=False)
    iff = rng.choice(len(A_f), min(n, len(A_f)), replace=False)
    X = np.vstack([A_t[it], A_f[iff]]).astype(np.float64)
    nt = len(it)
    is_for = np.r_[np.zeros(nt), np.ones(len(iff))]
    Q = X[:nt]
    # 平方欧氏距离（float64 展开式；排序与精确距离的一致性由 cKDTree 核对）
    d2 = (Q ** 2).sum(1)[:, None] + (X ** 2).sum(1)[None, :] - 2.0 * Q @ X.T
    d2[np.arange(nt), np.arange(nt)] = np.inf  # 排除自身
    nn = np.argsort(d2, axis=1, kind='stable')[:, :k]
    frac = is_for[nn].mean(1)
    M = float(frac.mean())
    extra = {}
    if check_kdtree:
        dd, j = cKDTree(X).query(Q, k=k + 1)
        # 去掉自身（若有零距离重复点，自身不一定排第一，按索引剔除）
        fr2 = np.array([is_for[[q for q in row if q != i][:k]].mean() for i, row in enumerate(j)])
        nn_same = float(np.mean([set(nn[i]) == set([q for q in row if q != i][:k]) for i, row in enumerate(j)]))
        extra = dict(M_kdtree=float(fr2.mean()), nn_set_agree=nn_same,
                     n_zero_dist=int((dd == 0).sum() - nt))  # 零距离的非自身近邻数
    return M, extra


def run_ae6(log):
    rows = []
    for tgt in COUNTRIES:
        S = np.load(os.path.join(R_G, 'stage0', f'{tgt}_s0.npz'))
        n_cand = len(S['cand'])
        for f in FEATS:
            Gf, GF = S[f'G_{f}'], S[f'GF_{f}']
            assert Gf.shape == (len(S['allT']), 32) and GF.shape == (len(S['foreign']), 32)
            A_t = Gf[:n_cand]
            M, extra = mixing_index(A_t, GF, N_ALL, seed_of(f'AE6_{tgt}_{f}_all'), check_kdtree=True)
            alts = [mixing_index(A_t, GF, N_ALL, seed_of(f'AE6_{tgt}_{f}_all_alt{r}'))[0] for r in range(N_ALT_SEEDS)]
            rows.append(dict(target=tgt, clf=f, n_cand=n_cand, n_foreign=len(GF), M_all=M, **extra,
                             M_alt_mean=float(np.mean(alts)), M_alt_sd=float(np.std(alts, ddof=1)),
                             **{f'M_alt{r}': a for r, a in enumerate(alts)}))
        log(f'AE6 {tgt} 完成')
    return pd.DataFrame(rows)


# ================================================================ 核对
def check_labels():
    """测试标签来源核对：按 check3_decomposition.py 的取法（load_dataset：has_ae & has_s2，reset_index；类 = 排序后的 label）
    重建 y，比较 y[test] 与 split.npz 的 y_test，并确认 test 行都属于目标国。"""
    df = pd.read_parquet(DATASET_V2, columns=['label', 'country', 'has_ae', 'has_s2'])
    df = df[df['has_ae'] & df['has_s2']].reset_index(drop=True)
    classes = sorted(df['label'].unique())
    y = df['label'].map({c: i for i, c in enumerate(classes)}).to_numpy()
    ctry = df['country'].to_numpy()
    out = []
    for tgt in COUNTRIES:
        sp = np.load(os.path.join(R3, f'{tgt}_split.npz'))
        te = sp['test']
        out.append(dict(target=tgt, y_same=bool(np.array_equal(y[te], sp['y_test'])),
                        ctry_ok=bool((ctry[te] == tgt).all()),
                        classes_same=bool(list(sp['classes']) == classes),
                        disjoint=bool(len(np.intersect1d(te, sp['pool'])) == 0 and
                                      len(np.intersect1d(te, sp['foreign'])) == 0),
                        n_test=len(te)))
    return pd.DataFrame(out)


def ae1_z0_reconcile():
    """AE1 中 Z0 的 D（本流窗口比较代码 + curves_B2.csv），与 AE–S2 整合评估 matched_oa_units.csv 的 W4_min_d 对账。"""
    cur = pd.read_csv(CURVES_B2, float_precision='round_trip')
    z = cur[cur.arm == 'Z0'].sort_values(['unit', 'clf', 'tau'])
    rows = []
    for unit, g in z.groupby('unit'):
        cv = {}
        for f in FEATS:
            x = g[g.clf == f]
            assert np.allclose(x.tau.to_numpy(), TAUS)
            cv[f] = (x.oa.to_numpy(), x.ba_ge30.to_numpy())
        anchor = min(cv['ae'][0][I_T0], cv['s2'][0][I_T0])
        res, _ = window_compare(cv['ae'], cv['s2'], anchor, 4)
        rows.append(dict(unit=unit, target=g.target.iloc[0], split=int(g.split.iloc[0]), rep=int(g.rep.iloc[0]),
                         D=res['D'], cov=res['cov'], ae_dom=res['ae_dom'], s2_dom=res['s2_dom']))
    U = pd.DataFrame(rows)
    ref = pd.read_csv(INTEG_UNITS).set_index('unit')
    J = U.set_index('unit').join(ref[['W4_min_d', 'W4_min_cov']], how='inner')
    return U, J


def curve_reproduce_check():
    """本流 tau_curve 包装与 step0B2 曲线逐位一致：用 stage0 的 P_T 测试行重算 Z0 曲线，与 curves_B2.csv 比较（2 个单元）。"""
    D = G.GData()
    y = np.asarray(D.y)
    cur = pd.read_csv(CURVES_B2, float_precision='round_trip')
    out = []
    for tgt, split, rep in (('AT', 0, 0), ('RO', 1, 1)):
        S = np.load(os.path.join(R_G, 'stage0', f'{tgt}_s{split}.npz'))
        n_cand = len(S['cand'])
        yT = y[S['test']]
        sc = B.Scorer(yT, np.flatnonzero(np.bincount(yT, minlength=C) >= G.MIN_N))
        for f in FEATS:
            oa, ba = tau_curve(B.norm64(S[f'P_T_{f}'][n_cand:].astype(np.float64)), sc)
            x = cur[(cur.unit == f'{tgt}_s{split}_r{rep}') & (cur.clf == f) & (cur.arm == 'Z0')].sort_values('tau')
            out.append(dict(unit=f'{tgt}_s{split}_r{rep}', clf=f,
                            same=bool(np.array_equal(oa, x.oa.to_numpy()) and np.array_equal(ba, x.ba_ge30.to_numpy())),
                            maxdiff=float(max(np.abs(oa - x.oa.to_numpy()).max(), np.abs(ba - x.ba_ge30.to_numpy()).max()))))
    return pd.DataFrame(out)


# ================================================================ 主程序
def main():
    t0 = time.time()
    lines = []
    P = lines.append

    def log(s):
        print(f'[{time.time() - t0:.0f}s] {s}', flush=True)

    h = sha16(SPEC)
    assert h == SPEC_HASH, f'规格哈希 {h} 与登记值 {SPEC_HASH} 不一致'
    A2, CV2, K2 = run_ae2(log)
    A6 = run_ae6(log)
    LBL = check_labels()
    log('标签核对完成')
    U1, J1 = ae1_z0_reconcile()
    log('AE1 Z0 对账完成')
    CRC = curve_reproduce_check()
    log('曲线复现核对完成')

    A2.to_csv(os.path.join(HERE, 'ae2_units.csv'), index=False)
    CV2.to_csv(os.path.join(HERE, 'ae2_curves.csv'), index=False)
    K2.to_csv(os.path.join(HERE, 'ae2_curve_checks.csv'), index=False)
    A6.to_csv(os.path.join(HERE, 'ae6_mall.csv'), index=False)
    U1.to_csv(os.path.join(HERE, 'ae1_z0_units_recalc.csv'), index=False)

    P('确认运行登记前的不训练分析 · AE 线 · ae_mechanism 流独立复核（AE2 c300/es 的 D_dom、D_xb、D_xb−D_dom；AE6 的 M_all）')
    P('事后分析，C 级、描述性；不训练；测试标签只用于评估。本批数字不进摘要。')
    P(f'规格 PRECONF_ANALYSES_SPEC_2026-09-29.md sha256[:16]={h}（登记 {SPEC_HASH}）；'
      f'step0B2.py sha256[:16]={sha16(os.path.join(PARTB2, "step0B2.py"))}；step0B.py sha256[:16]={sha16(os.path.join(PARTB, "step0B.py"))}（只 import，未改）；'
      f'本脚本 sha256[:16]={sha16(os.path.abspath(__file__))}。')
    P(f'τ 网格 {TAUS[0]}..{TAUS[-1]} 步长 0.05（{len(TAUS)} 点）；W4 = [锚点−4, 锚点] 步长 0.1（41 点）；共同可达 <50% 记缺失；'
      f'ba_ge30 类集合 = 测试集中 ≥ {G.MIN_N} 个地块的类；国家值 = 该国唯一单元（检验 3 为每国一个随机 70/30 划分）。')
    P('')

    # ---------------------------------------------------------------- 核对
    P('== 核对 ==')
    P(f'(1) 测试标签来源（dataset_v2 按 check3_decomposition 的取法重建 y）：y[test] == y_test {int(LBL.y_same.sum())}/{len(LBL)}；'
      f'test 行均属目标国 {int(LBL.ctry_ok.sum())}/{len(LBL)}；类顺序一致 {int(LBL.classes_same.sum())}/{len(LBL)}；'
      f'test 与 pool、foreign 不相交 {int(LBL.disjoint.sum())}/{len(LBL)}')
    cap = pd.read_csv(CAP_CSV)
    arm_of = {'dom': 'W0', 'xb': 'L0'}
    K2m = K2.assign(arm=K2.model.map(arm_of)).rename(columns={'clf': 'feat'})
    J = K2m.merge(cap, on=['target', 'feat', 'cap', 'arm'], how='inner')
    P(f'(2) τ=0 的 OA / ba_ge30 与 results/check3_capacity_arms.csv（W0 = within，L0 = loco；es 与 c300）：{len(J)}/{len(K2)} 行对上，'
      f'最大 |Δ| OA {(J.oa_t0 - J.oa).abs().max():.2e}，BA {(J.ba_t0 - J.ba_ge30).abs().max():.2e}')
    P(f'(3) 包络：step0B.envelope 与本流逐段循环实现（AE2 全部 {len(A2)} 个 (国家, 设置, 模型) × 两特征 × 41 点）：'
      f'最大 |Δ| {A2.env_own_maxdiff.max():.2e}，不可达位置一致 {int(A2.env_own_nan_same.sum())}/{len(A2)}')
    P(f'(4) 本流 τ 曲线包装与 step0B2 的 Z0 曲线（stage0 P_T 重算 vs curves_B2.csv）：{int(CRC.same.sum())}/{len(CRC)} 条逐位相同，'
      f'最大 |Δ| {CRC.maxdiff.max():.2e}')
    d = (J1.D - J1.W4_min_d)
    both_nan = J1.D.isna() & J1.W4_min_d.isna()
    P(f'(5) 本流窗口比较代码在 AE1 Z0（curves_B2.csv，44 单元）上与整合评估 matched_oa_units.csv 的 W4_min_d：{len(J1)} 单元，'
      f'最大 |Δ| {d.abs().max():.2e}，缺失位置一致 {int(((J1.D.isna()) == (J1.W4_min_d.isna())).sum())}/{len(J1)}'
      f'（整合评估网格为 np.round(arange(锚点−4, 锚点, 0.1), 6)，本流为 锚点 + step0B.OFFS，二者只差浮点舍入）')
    P(f'(6) AE6 最近邻：精确 float64 距离 + 稳定排序 与 cKDTree(k=11，按索引剔除自身) 的 M_all 最大 |Δ| '
      f'{(A6.M_all - A6.M_kdtree).abs().max():.2e}，近邻集合一致比例最小 {A6.nn_set_agree.min():.4f}；'
      f'零距离的非自身近邻数合计 {int(A6.n_zero_dist.sum())}')
    P(f'(7) 训练类数（概率列非零）：within {sorted(int(v) for v in K2[K2.model == "dom"].n_trained.unique())}，loco {sorted(int(v) for v in K2[K2.model == "xb"].n_trained.unique())}；'
      f'ba_ge30 类数 {sorted(int(v) for v in K2.n_big.unique())}')
    P('')

    # ---------------------------------------------------------------- AE2
    P('######## AE2 国内与跨境：同 OA 下 AE−S2（pp）########')
    P('D_dom = within 模型的 D；D_xb = loco 模型的 D；锚点 = 该模型两特征 τ=0 OA 的较小者；D = W4 内共同可达点上 BA_ae − BA_s2 的平均。')
    RES = {}
    for cp in CAPS:
        W = A2[A2.cap == cp].set_index(['target', 'model'])
        T = pd.DataFrame({
            'D_dom': W.xs('dom', level='model').D, 'D_xb': W.xs('xb', level='model').D,
            'cov_dom': W.xs('dom', level='model')['cov'], 'cov_xb': W.xs('xb', level='model')['cov'],
            'dBA0_dom': W.xs('dom', level='model').dBA_t0, 'dBA0_xb': W.xs('xb', level='model').dBA_t0,
            'dOA0_dom': W.xs('dom', level='model').dOA_t0, 'dOA0_xb': W.xs('xb', level='model').dOA_t0,
            'anc_dom': W.xs('dom', level='model').anchor, 'anc_xb': W.xs('xb', level='model').anchor,
        }).loc[COUNTRIES]
        T['D_diff'] = T.D_xb - T.D_dom
        T['dBA0_diff'] = T.dBA0_xb - T.dBA0_dom
        T['dOA0_diff'] = T.dOA0_xb - T.dOA0_dom
        for w in (2, 8):
            T[f'D_dom_w{w}'] = W.xs('dom', level='model')[f'D_w{w}'].loc[COUNTRIES]
            T[f'D_xb_w{w}'] = W.xs('xb', level='model')[f'D_w{w}'].loc[COUNTRIES]
            T[f'D_diff_w{w}'] = T[f'D_xb_w{w}'] - T[f'D_dom_w{w}']
        for a in ('ae', 's2'):
            T[f'D_dom_anc{a}'] = W.xs('dom', level='model')[f'D_anc_{a}'].loc[COUNTRIES]
            T[f'D_xb_anc{a}'] = W.xs('xb', level='model')[f'D_anc_{a}'].loc[COUNTRIES]
            T[f'D_diff_anc{a}'] = T[f'D_xb_anc{a}'] - T[f'D_dom_anc{a}']
        T.to_csv(os.path.join(HERE, f'ae2_country_{cp}.csv'))
        RES[cp] = T
        P(f'==== 设置 {cp}（{"固定 300 棵，与 G 一致；主" if cp == "c300" else "早停；次"}）====')
        P('  国家   锚点dom  D_dom  cov | 锚点xb   D_xb   cov | D_xb−D_dom | τ=0 ΔBA dom / xb / 差 | τ=0 ΔOA dom / xb')
        for c in COUNTRIES:
            r = T.loc[c]
            P(f'  {c}   {r.anc_dom:6.2f} {r.D_dom:+6.2f} {r.cov_dom:.2f} | {r.anc_xb:6.2f} {r.D_xb:+6.2f} {r.cov_xb:.2f} | '
              f'{r.D_diff:+7.2f}    | {r.dBA0_dom:+6.2f} / {r.dBA0_xb:+6.2f} / {r.dBA0_diff:+6.2f} | '
              f'{r.dOA0_dom:+6.2f} / {r.dOA0_xb:+6.2f}')
        for col, name in (('D_dom', 'D_dom'), ('D_xb', 'D_xb'), ('D_diff', 'D_xb − D_dom'),
                          ('dBA0_dom', '对照 τ=0 ΔBA 国内'), ('dBA0_xb', '对照 τ=0 ΔBA 跨境'),
                          ('dBA0_diff', '对照 τ=0 ΔBA 跨境−国内'),
                          ('dOA0_dom', '对照 τ=0 ΔOA 国内'), ('dOA0_xb', '对照 τ=0 ΔOA 跨境'),
                          ('dOA0_diff', '对照 τ=0 ΔOA 跨境−国内')):
            P(f'  {name:22s} 9 国 {fmt(tsum(T.loc[C9, col]))} | 11 国 {fmt(tsum(T[col]))}')
        P('  -- 方案外敏感性（描述）：窗口宽 2 / 8；锚点取 OA★_ae / OA★_s2（宽 4）--')
        for col in ('D_dom_w2', 'D_xb_w2', 'D_diff_w2', 'D_dom_w8', 'D_xb_w8', 'D_diff_w8',
                    'D_dom_ancae', 'D_xb_ancae', 'D_diff_ancae', 'D_dom_ancs2', 'D_xb_ancs2', 'D_diff_ancs2'):
            P(f'  {col:14s} 9 国 {fmt(tsum(T.loc[C9, col]))} | 11 国 {fmt(tsum(T[col]))}')
        x = A2[A2.cap == cp]
        P(f'  共同可达：dom 全可达 {int((x[x.model == "dom"]["cov"] >= 1).sum())}/11、缺失 {int(x[x.model == "dom"].D.isna().sum())}；'
          f'xb 全可达 {int((x[x.model == "xb"]["cov"] >= 1).sum())}/11、缺失 {int(x[x.model == "xb"].D.isna().sum())}；'
          f'AE 全窗口占优 dom {int((x[x.model == "dom"].ae_dom == True).sum())}、xb {int((x[x.model == "xb"].ae_dom == True).sum())}；'  # noqa: E712
          f'S2 全窗口占优 dom {int((x[x.model == "dom"].s2_dom == True).sum())}、xb {int((x[x.model == "xb"].s2_dom == True).sum())}')  # noqa: E712
        P('')
    P('-- 读法（规格 AE2，描述；D_xb − D_dom 的 95% CI）--')
    for cp in CAPS:
        for tag, v in (('9 国', RES[cp].loc[C9, 'D_diff']), ('11 国', RES[cp]['D_diff'])):
            s = tsum(v)
            P(f'  {cp:4s} {tag:5s} D_xb − D_dom = {fmt(s)} → {reading_ae2(s)}')
    P('')

    # ---------------------------------------------------------------- AE6
    P('######## AE6 类无关混合指数 M_all（划分 0；目标候选与外国池各抽 2,000；10-NN 排除自身；外国占比平均）########')
    P('种子 = int(md5(f"AE6_{国家}_{特征}_all").hexdigest()[:8], 16)；同一 rng 先抽目标、后抽外国（均无放回）。'
      f'方案外：另用 {N_ALT_SEEDS} 个种子（…_all_alt{{r}}）重复，给出抽样 SD 作为与主实现对账时的容差参考。')
    M = A6.pivot(index='target', columns='clf', values='M_all').loc[COUNTRIES]
    Ma = A6.pivot(index='target', columns='clf', values='M_alt_mean').loc[COUNTRIES]
    Ms = A6.pivot(index='target', columns='clf', values='M_alt_sd').loc[COUNTRIES]
    M['d'] = M.ae - M.s2
    alt_d = pd.DataFrame({r: A6[A6.clf == 'ae'].set_index('target')[f'M_alt{r}'] -
                          A6[A6.clf == 's2'].set_index('target')[f'M_alt{r}'] for r in range(N_ALT_SEEDS)}).loc[COUNTRIES]
    P('  国家   M_all(ae)  M_all(s2)  AE−S2   | 替代种子均值 ae / s2（SD ae / s2）| AE−S2 替代种子均值（SD）')
    for c in COUNTRIES:
        P(f'  {c}     {M.loc[c, "ae"]:.4f}     {M.loc[c, "s2"]:.4f}   {M.loc[c, "d"]:+.4f}  | '
          f'{Ma.loc[c, "ae"]:.4f} / {Ma.loc[c, "s2"]:.4f}（{Ms.loc[c, "ae"]:.4f} / {Ms.loc[c, "s2"]:.4f}）| '
          f'{alt_d.loc[c].mean():+.4f}（{alt_d.loc[c].std(ddof=1):.4f}）')
    for col, name in (('ae', 'M_all(ae)'), ('s2', 'M_all(s2)'), ('d', 'M_all AE−S2')):
        P(f'  {name:12s} 9 国 {fmt(tsum(M.loc[C9, col]), 4)} | 11 国 {fmt(tsum(M[col]), 4)}')
    M.to_csv(os.path.join(HERE, 'ae6_mall_country.csv'))
    # 方案外描述：与 AE1 Z0 的 D（本流重算，国家 = 4 单元平均）的 Spearman
    Dz = U1.groupby('target').D.mean().loc[COUNTRIES]
    r = stats.spearmanr(M['d'], Dz)
    P(f'  （描述）M_all AE−S2 与 AE1 Z0 的 D（本流重算，W4，锚点 = min OA★）逐国 Spearman ρ = {r[0]:+.3f}（n = 11，p = {r[1]:.3f}）')
    P('  AE1 Z0 的 D 国家值（本流重算）：' + '，'.join(f'{c} {Dz[c]:+.2f}' for c in COUNTRIES))
    s9, s11 = tsum(Dz.loc[C9]), tsum(Dz)
    P(f'  AE1 Z0 的 D：9 国 {fmt(s9)} | 11 国 {fmt(s11)}；AE 全窗口占优 {int((U1.ae_dom == True).sum())}/44，'  # noqa: E712
      f'S2 全窗口占优 {int((U1.s2_dom == True).sum())}/44')
    P('')
    P(f'总耗时 {time.time() - t0:.0f}s')

    open(os.path.join(HERE, 'indep_ae_mechanism_summary.txt'), 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    # 结构化结果（供编排脚本读取）
    key = {}
    for cp in CAPS:
        for col in ('D_dom', 'D_xb', 'D_diff', 'dBA0_dom', 'dBA0_xb', 'dBA0_diff'):
            for tag, v in (('9', RES[cp].loc[C9, col]), ('11', RES[cp][col])):
                key[f'{cp}.{col}.{tag}'] = tsum(v)
    for col in ('ae', 's2', 'd'):
        for tag, v in (('9', M.loc[C9, col]), ('11', M[col])):
            key[f'M_all.{col}.{tag}'] = tsum(v)
    json.dump({k: {kk: (None if (isinstance(vv, float) and not np.isfinite(vv)) else vv) for kk, vv in v.items()}
               for k, v in key.items()}, open(os.path.join(HERE, 'indep_ae_mechanism_key.json'), 'w'),
              ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
