"""
PC-1：确认运行方案 v4 的回溯参照（raw 口径）—— 主实现。
规格：label_expansion/PC1_SPEC_2026-09-29.md（sha256[:16]=eb82a82b7acea883，写于任何 PC-1 代码与输出之前；本脚本不改动它）。
所服务的方案：verify/confirm_plan/CONFIRM_PLAN_DRAFT_v4.md（sha256[:16]=a634f4a83b1b3942）第 2–6 节。
C 级，只作参照，不进摘要（G s0/s1 测试集已多次复用）。不训练任何分类器，只读已存概率与 G 的选样文件。
候选标签只用于构造 p̂（PK-k）；测试标签只用于评估，另外按规格作为 PKo 的"真实比例"。

臂（每单元 × 特征；单元 = 11 国 × 划分 s0/s1 × 重复 r0/r1 = 44 个）
  LightGBM
    Z0    stage0/{国}_s{划分}.npz 的 P0test_{f}_i300（外国全量模型第 300 轮的未校准测试概率；float16 → float64，按行归一化）
    PK-k  check3_priors.reweight(Z0, laplace_prior(y_sel, 10, 1), π_train)；y_sel = G 的 SRS(k) 选样
          （sel/{单元}_none_srs_k{k}.npz 的 sel；其 sel_hash 与 rows_t1 中 method=srs、cond=none、该 k 与 clf 的 sel_hash 相同）
    PKo   同上，比例取测试集真实比例
    RT-k  rows_t1 中上述 sel_hash 对应的 probs/{单元}_{f}_k{k}_{sel_hash}.npz 的 i300；k ∈ {50, 100, 200, 400}
    π_train = 外国池按 class_weights_from 加权后的类别分布（= guidance_G.pi_train_of([], 0)）
  RF
    Z0    guidance_G_ml/probs/rf/{国}_s{划分}_{f}_k0.npz 的 P
    PK200 RF 的 Z0 经同一 reweight（p̂ 取 SRS k=200 的同一批标签）
    RT200 rows_rf_*.csv 中 method=srs、k=200 的 sel_hash 对应 probs/rf/{单元}_{f}_k200_{sel_hash}.npz 的 P
  TGT：测试概率没有保存，不计算（规格）。

度量（v4 第 3 节；复用第 0 步 B / B2 的函数，import，不改其文件）
  τ 族    ℓ = log P − τ·log π̂，π̂ = 该臂测试集平均预测；τ ∈ [−2, 2] 步长 0.05（81 点）；精确 log；argmax 并列取第一个
          （step0B2.tau_preds_grid = step0B.tau_preds 的源码；step0B.log_or_neginf）
  OA / ba_ge30  step0B.Scorer（ba_ge30 的类集合 = 该单元测试集中 ≥ 30 个地块的类）
  上包络  step0B.envelope（每臂一条 τ 曲线；相邻 τ 点在 (OA, BA) 平面线性插值）
  窗口    [OA★ − 4, OA★] 步长 0.1（41 点，= step0B.OFFS）；OA★ = 同一学习器、同一特征的 Z0 在 τ = 0 的 OA
  两臂之差  在两臂共同可达的格点上逐点求 BA 差再平均；共同可达 < 50% 记缺失
  D       锚点 = min(OA★_ae, OA★_s2)（同一学习器的 Z0）；窗口同上；两特征该臂曲线共同可达的格点上平均 BA_ae − BA_s2；
          共同可达 < 50% 记缺失（沿用上一条规则）
  国家值  单元内先配对求差 → 该国可用单元（2 划分 × 2 重复）平均；both = AE、S2 两特征国家值的平均（step0B.country_vals）
  汇报    9 国（去 RO、PT）与 11 国：均值、95% t CI、>0 的国家数（P6 为 90% CI）
计算项 P1–P10 见规格；P9 功效只用 rep0 单元的国家层 SD。另报各臂 τ = 0 时 argmax 并列的比例。

用法：/Users/perng_mini/AlphaCrops_local/.venv/bin/python -B pc1.py --smoke   # AT_s0_r0、RO_s1_r1，输出到 smoke/，做对账
      /Users/perng_mini/AlphaCrops_local/.venv/bin/python -B pc1.py           # 全量 44 个单元
"""
import os
import sys
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_v] = '2'
sys.dont_write_bytecode = True  # 不在项目目录、partB/、partB2/ 写 __pycache__
import glob
import hashlib
import time

import numpy as np
import pandas as pd
from scipy import optimize, stats

HERE = os.path.dirname(os.path.abspath(__file__))
PARTB2 = '/Users/perng_mini/AlphaCrops_local/verify/step0/partB2'
sys.path.insert(0, PARTB2)
import step0B2 as B2  # noqa: E402  导入链：step0B（chdir 到项目目录，只读；导入 guidance_G）
from check3_priors import laplace_prior, reweight  # noqa: E402

B = B2.B
G = B.G
SPEC = os.path.join(B.L, 'PC1_SPEC_2026-09-29.md')
SPEC_HASH = 'eb82a82b7acea883'
PLAN = '/Users/perng_mini/AlphaCrops_local/verify/confirm_plan/CONFIRM_PLAN_DRAFT_v4.md'
PLAN_HASH = 'a634f4a83b1b3942'
STEP0B_PY = B2.STEP0B_PY
STEP0B2_PY = os.path.join(PARTB2, 'step0B2.py')
GUIDE_PY = os.path.join(B.L, 'guidance_G.py')
PRIORS_PY = os.path.join(B.L, 'check3_priors.py')
R_G, R_ML = B.R_G, B.R_ML
# P10 连续性：旧口径的行文件
FA_B2 = os.path.join(PARTB2, 'fa_rows_B2.csv')
LABEL_ROWS = '/Users/perng_mini/AlphaCrops_local/verify/preconf/label/label_rows.csv'
AE1_ROWS = '/Users/perng_mini/AlphaCrops_local/verify/preconf/ae/ae_frontier_rows.csv'

C = B.C
FEATS = B.FEATS
COUNTRIES = B.COUNTRIES
DROP = B.DROP
MIN_COV = B.MIN_COV
TAUS = B2.TAUS_NEW
I_T0 = B2.I_T0
OFFS = B.OFFS
assert len(TAUS) == 81 and TAUS[0] == -2.0 and TAUS[-1] == 2.0 and TAUS[I_T0] == 0.0
assert len(OFFS) == 41 and OFFS[0] == -4.0 and OFFS[-1] == 0.0
KS = [50, 100, 200, 400]
LEARNERS = ['lgb', 'rf']
LNAME = {'lgb': 'LightGBM', 'rf': 'RF'}
ARMS = {'lgb': ['Z0'] + [f'PK{k}' for k in KS] + ['PKo'] + [f'RT{k}' for k in KS],
        'rf': ['Z0', 'PK200', 'RT200']}
DIFFS = {'lgb': {**{f'dFA_RT{k}': (f'RT{k}', 'Z0') for k in KS},
                 **{f'A_{k}': (f'PK{k}', 'Z0') for k in KS},
                 **{f'R_{k}': (f'RT{k}', f'PK{k}') for k in KS},
                 'A_o': ('PKo', 'Z0')},
         'rf': {'dFA_RT200': ('RT200', 'Z0'), 'A_200': ('PK200', 'Z0'), 'R_200': ('RT200', 'PK200')}}
D_ARMS = {'lgb': ['Z0', 'RT200'], 'rf': ['Z0']}
UK = ['target', 'split', 'rep', 'unit']
TOL = 1e-9
ALPHAS = (0.025, 0.0125)   # P9：单侧 α；0.0125 = Holm 首步


def sha16(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()[:16]


def sha64(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()


def pihat_of(Z):
    """π̂：该臂测试集平均预测概率（exp(Z − 行最大) 归一化后取列均值；与 step0B / B2 相同，不用标签）。"""
    Pa = np.exp(Z - Z.max(1, keepdims=True))
    Pa /= Pa.sum(1, keepdims=True)
    return Pa.mean(0)


def tie_frac_t0(Z):
    """τ = 0 时 argmax 所见的 ℓ = log P：最大值与次大值相等（并列）的测试行比例。"""
    s = np.sort(Z, axis=1)
    return float((s[:, -1] == s[:, -2]).mean())


def paired(ea, eb):
    """两条上包络在共同可达格点上的逐点差平均；共同可达 < 50% 记缺失。返回 (值, 共同可达率)。"""
    ok = np.isfinite(ea) & np.isfinite(eb)
    cov = float(ok.mean())
    return (float(np.mean(ea[ok] - eb[ok])) if cov >= MIN_COV else np.nan), cov


# ---------------------------------------------------------------- 数据
def load_G_rows():
    GR = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(R_G, 'rows_t1_*.csv')))], ignore_index=True)
    return GR[(GR.cond == 'none') & GR.method.isin(['srs', 'foreign_only'])]


def load_RF_rows():
    MR = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(R_ML, 'rows_rf_*.csv')))], ignore_index=True)
    MR = MR[(MR.learner == 'rf') & (MR.cond == 'none') &
            (((MR.method == 'srs') & (MR.k == 200)) | (MR.method == 'foreign_only'))]
    assert (MR.status == 'ok').all()
    return MR


# ---------------------------------------------------------------- 一个单元
def run_unit(D, S, GR, MR, tgt, split, rep, log=print):
    y = np.asarray(D.y)
    unit = f'{tgt}_s{split}_r{rep}'
    n_cand = len(S['cand'])
    allT, test, fr = S['allT'], S['test'], S['foreign']
    assert np.array_equal(allT[n_cand:], test)
    yT = y[test]
    nT = len(yT)
    big = np.flatnonzero(np.bincount(yT, minlength=C) >= G.MIN_N)
    sc = B.Scorer(yT, big)
    yF = y[fr]
    cw = G.class_weights_from(yF)
    pi_train = np.bincount(yF, weights=cw[yF], minlength=C) / cw[yF].sum()   # = guidance_G pi_train_of([], 0)
    pool_pos = G.pool_positions(S, split, rep)
    pi_test = np.bincount(yT, minlength=C) / nT                              # 只用于 PKo（规格定义）

    # SRS 选样：G 的 sel 文件；sel_pos 经 pool_positions 映射必须等于 sel
    sels = {}
    for k in KS:
        z = np.load(os.path.join(R_G, 'sel', f'{unit}_none_srs_k{k}.npz'))
        sel = z['sel']
        assert np.array_equal(allT[pool_pos][z['sel_pos']], sel) and len(sel) == k, (unit, k)
        sels[k] = (G.sel_hash(sel), laplace_prior(y[sel], C, alpha=1))

    rows, CV = [], {}
    n_hash = 0
    for f in FEATS:
        cell = dict(target=tgt, split=split, rep=rep, unit=unit, clf=f)
        Z, refs, fb = {}, {}, {}
        # ---- LightGBM
        P0 = B.norm64(S[f'P0test_{f}_i300'])      # float16 → float64，按行归一化
        Z[('lgb', 'Z0')] = B.log_or_neginf(P0)
        g0 = GR[(GR.method == 'foreign_only') & (GR.target == tgt) & (GR.split == split) & (GR.rep == rep) &
                (GR.clf == f)]
        assert len(g0) == 1, (unit, f)
        refs[('lgb', 'Z0')] = ('G_rows_foreign_only', g0.iloc[0].oa, g0.iloc[0].ba_ge30)
        for k in KS:
            h, p_hat = sels[k]
            g = GR[(GR.method == 'srs') & (GR.target == tgt) & (GR.split == split) & (GR.rep == rep) &
                   (GR.clf == f) & (GR.k == k)]
            assert len(g) == 1, (unit, f, k, len(g))
            g = g.iloc[0]
            assert g.sel_hash == h, (unit, f, k)
            n_hash += 1
            Pr = B.norm64(np.load(os.path.join(R_G, 'probs', f'{unit}_{f}_k{k}_{h}.npz'))['i300'])
            Z[('lgb', f'RT{k}')] = B.log_or_neginf(Pr)
            refs[('lgb', f'RT{k}')] = ('G_rows_srs', g.oa, g.ba_ge30)
            Q, fb[('lgb', f'PK{k}')] = reweight(P0, p_hat, pi_train)
            Z[('lgb', f'PK{k}')] = B.log_or_neginf(Q)
            # 对账用另一条路径：guidance_G.prior_correct + 概率 argmax + guidance_G._scores
            o_i, b_i = G._scores(yT, G.prior_correct(P0, p_hat, pi_train).argmax(1), big)[:2]
            refs[('lgb', f'PK{k}')] = ('prior_correct_path', o_i, b_i)
        Q, fb[('lgb', 'PKo')] = reweight(P0, pi_test, pi_train)
        Z[('lgb', 'PKo')] = B.log_or_neginf(Q)
        o_i, b_i = G._scores(yT, G.prior_correct(P0, pi_test, pi_train).argmax(1), big)[:2]
        refs[('lgb', 'PKo')] = ('prior_correct_path', o_i, b_i)
        # ---- RF
        q = (MR.target == tgt) & (MR.split == split) & (MR.rep == rep) & (MR.clf == f)
        m0, m2 = MR[q & (MR.method == 'foreign_only')], MR[q & (MR.method == 'srs')]
        assert len(m0) == 1 and len(m2) == 1, (unit, f)
        m0, m2 = m0.iloc[0], m2.iloc[0]
        h200, p200 = sels[200]
        assert m2.sel_hash == h200, (unit, f, 'rf')   # RF 的 SRS k=200 与 G 为同一批选样
        n_hash += 1
        Pk0 = B.norm64(np.load(os.path.join(R_ML, 'probs', 'rf', f'{tgt}_s{split}_{f}_k0.npz'))['P'])
        P2 = B.norm64(np.load(os.path.join(R_ML, 'probs', 'rf', f'{unit}_{f}_k200_{m2.sel_hash}.npz'))['P'])
        Z[('rf', 'Z0')] = B.log_or_neginf(Pk0)
        Z[('rf', 'RT200')] = B.log_or_neginf(P2)
        Q, fb[('rf', 'PK200')] = reweight(Pk0, p200, pi_train)
        Z[('rf', 'PK200')] = B.log_or_neginf(Q)
        refs[('rf', 'Z0')] = ('RF_rows_foreign_only', m0.oa, m0.ba_ge30)
        refs[('rf', 'RT200')] = ('RF_rows_srs', m2.oa, m2.ba_ge30)
        o_i, b_i = G._scores(yT, G.prior_correct(Pk0, p200, pi_train).argmax(1), big)[:2]
        refs[('rf', 'PK200')] = ('prior_correct_path', o_i, b_i)

        # ---- τ 曲线
        for (lrn, arm), Za in Z.items():
            PH = B2.tau_preds_grid(Za, pihat_of(Za), TAUS)
            CV[(lrn, f, arm)] = sc.curve(PH)

        # ---- 每个学习器：OA★、各臂上包络、臂行与差值行
        for lrn in LEARNERS:
            oa_star = CV[(lrn, f, 'Z0')][0][I_T0]
            grid = oa_star + OFFS
            env = {}
            for arm in ARMS[lrn]:
                oa, ba = CV[(lrn, f, arm)]
                best = B.envelope([(oa, ba)], grid)[0]
                env[arm] = best
                fa, cov = B.fa_of(best)
                src, o_ref, b_ref = refs.get((lrn, arm), ('', np.nan, np.nan))
                rows.append(dict(level='arm', learner=lrn, **cell, name=arm, arm_a=arm, arm_b='', OA_star=oa_star,
                                 value=fa, cov=cov, missing=bool(cov < MIN_COV), oa_t0=oa[I_T0], ba_t0=ba[I_T0],
                                 tie_t0=tie_frac_t0(Z[(lrn, arm)]), oa_min=float(oa.min()), oa_max=float(oa.max()),
                                 ref=src, oa_ref=o_ref, ba_ref=b_ref, fallback=fb.get((lrn, arm), 0)))
            for name, (a, b) in DIFFS[lrn].items():
                v, cov = paired(env[a], env[b])
                rows.append(dict(level='diff', learner=lrn, **cell, name=name, arm_a=a, arm_b=b, OA_star=oa_star,
                                 value=v, cov=cov, missing=bool(cov < MIN_COV)))

    # ---- D（AE − S2；锚点 = 同一学习器两特征 Z0 在 τ=0 的 OA 的较小者）
    for lrn in LEARNERS:
        anc = min(CV[(lrn, f, 'Z0')][0][I_T0] for f in FEATS)
        grid = anc + OFFS
        for arm in D_ARMS[lrn]:
            ea = B.envelope([CV[(lrn, 'ae', arm)]], grid)[0]
            es = B.envelope([CV[(lrn, 's2', arm)]], grid)[0]
            v, cov = paired(ea, es)
            rows.append(dict(level='D', learner=lrn, target=tgt, split=split, rep=rep, unit=unit, clf='ae-s2',
                             name=f'D_{arm}', arm_a=arm, arm_b=arm, OA_star=anc, value=v, cov=cov,
                             missing=bool(cov < MIN_COV), reach_ae=float(np.isfinite(ea).mean()),
                             reach_s2=float(np.isfinite(es).mean())))
    log(f'{unit} 完成')
    return rows, n_hash


# ---------------------------------------------------------------- 汇总工具
def tsum(v, level=0.95):
    v = np.asarray(pd.Series(v).dropna(), dtype=float)
    n = len(v)
    if n < 2:
        return dict(n=n, mean=float(v.mean()) if n else np.nan, sd=np.nan, lo=np.nan, hi=np.nan,
                    n_pos=int((v > 0).sum()))
    m, sd = v.mean(), v.std(ddof=1)
    h = stats.t.ppf(1 - (1 - level) / 2, n - 1) * sd / np.sqrt(n)
    return dict(n=n, mean=m, sd=sd, lo=m - h, hi=m + h, n_pos=int((v > 0).sum()))


def fmt(s, nd=2):
    if s['n'] < 2:
        return f'n={s["n"]}（不足 2 国）' + (f' 均值 {s["mean"]:+.{nd}f}' if s['n'] == 1 else '')
    return f'{s["mean"]:+.{nd}f} [{s["lo"]:+.{nd}f}, {s["hi"]:+.{nd}f}]，>0 {s["n_pos"]}/{s["n"]}'


def sel9(v, nine):
    return v.drop([c for c in DROP if c in v.index]) if nine else v


def mde80(sd, n, alpha):
    """单样本 t 检验，单侧 α，80% 功效的最小可检出效应（非中心 t 精确解；与 power_confirm.py 相同）。"""
    if not np.isfinite(sd) or sd <= 0 or n < 3:
        return np.nan
    df = n - 1
    tc = stats.t.ppf(1 - alpha, df)
    return optimize.brentq(lambda d: stats.nct.sf(tc, df, d / (sd / np.sqrt(n))) - 0.8, 1e-9, 50 * sd)


def power(delta, sd, n, alpha):
    if not np.isfinite(sd) or sd <= 0 or n < 3:
        return np.nan
    df = n - 1
    return float(stats.nct.sf(stats.t.ppf(1 - alpha, df), df, delta / (sd / np.sqrt(n))))


# ---------------------------------------------------------------- 主程序
def main():
    smoke = '--smoke' in sys.argv
    out = os.path.join(HERE, 'smoke') if smoke else HERE
    os.makedirs(out, exist_ok=True)
    assert sha16(SPEC) == SPEC_HASH, '规格文件哈希与登记值不一致'
    t0 = time.time()
    print(f'pc1.py sha256[:16]={sha16(os.path.abspath(__file__))}；step0B2.py {sha16(STEP0B2_PY)}；'
          f'step0B.py {sha16(STEP0B_PY)}', flush=True)
    D = G.GData()
    GR = load_G_rows()
    MR = load_RF_rows()
    units = ([('AT', 0, 0), ('RO', 1, 1)] if smoke else
             [(t, s, r) for t in COUNTRIES for s in (0, 1) for r in (0, 1)])
    ROWS, n_hash = [], 0
    cache = {}
    for tgt, split, rep in units:
        if (tgt, split) not in cache:
            cache.clear()
            cache[(tgt, split)] = np.load(os.path.join(R_G, 'stage0', f'{tgt}_s{split}.npz'))
        r_, h_ = run_unit(D, cache[(tgt, split)], GR, MR, tgt, split, rep,
                          log=lambda s: print(f'[{time.time() - t0:.0f}s] {s}', flush=True))
        ROWS += r_
        n_hash += h_
    R = pd.DataFrame(ROWS)
    rows_path = os.path.join(out, 'pc1_rows.csv')
    R.to_csv(rows_path, index=False)
    lines, PW, KEY = summarize(R, n_hash, smoke)
    pw_path = os.path.join(out, 'pc1_power.csv')
    PW.to_csv(pw_path, index=False)
    lines += ['', '== 输出文件 sha256 ==',
              f'  pc1_rows.csv  {sha64(rows_path)}',
              f'  pc1_power.csv {sha64(pw_path)}',
              '  （pc1_summary.txt 自身的 sha256 见运行日志末行）']
    sum_path = os.path.join(out, 'pc1_summary.txt')
    open(sum_path, 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print('KEY ' + ' | '.join(f'{k}: {v}' for k, v in KEY.items()))
    print(f'pc1_summary.txt sha256 {sha64(sum_path)}')
    print(f'总耗时 {time.time() - t0:.0f}s')


def summarize(R, n_hash, smoke):
    lines = []
    P = lines.append
    KEY = {}
    nU = R.unit.nunique()
    AR = R[R.level == 'arm']
    DF = R[R.level == 'diff']
    DD = R[R.level == 'D']

    P('PC-1：确认运行方案 v4 的回溯参照（raw 口径）—— C 级，只作参照，不进摘要（G s0/s1 测试集已多次复用）。'
      + ('【冒烟：2 个单元，数字无意义，只验证代码路径与对账】' if smoke else ''))
    P(f'规格 PC1_SPEC_2026-09-29.md sha256[:16]={sha16(SPEC)}（登记 {SPEC_HASH}）；方案 CONFIRM_PLAN_DRAFT_v4.md '
      f'sha256[:16]={sha16(PLAN)}（规格登记时 {PLAN_HASH}）；pc1.py {sha16(os.path.abspath(__file__))}；'
      f'import 复用（未改）：step0B2.py {sha16(STEP0B2_PY)}、step0B.py {sha16(STEP0B_PY)}、guidance_G.py {sha16(GUIDE_PY)}、'
      f'check3_priors.py {sha16(PRIORS_PY)}。')
    P(f'单元 {nU}（国家 × 划分 × 重复）× 特征 2；学习器 LightGBM（臂 {", ".join(ARMS["lgb"])}）与 RF（臂 {", ".join(ARMS["rf"])}）。'
      f'τ ∈ [{TAUS[0]}, {TAUS[-1]}] 步长 0.05（{len(TAUS)} 点）；窗口 [OA★ − 4, OA★] 步长 0.1（{len(OFFS)} 点），'
      'OA★ = 同一学习器、同一特征的 Z0（raw）在 τ = 0 的 OA；两臂之差 = 共同可达格点上的逐点 BA 差平均，共同可达 <50% 记缺失；'
      'D 的锚点 = min(OA★_ae, OA★_s2)，两特征共同可达格点上平均 BA_ae − BA_s2（<50% 记缺失）。')
    P('国家值 = 单元内配对差在该国可用单元（2 划分 × 2 重复）上的平均；both = AE、S2 两特征国家值的平均（任一缺失则 both 缺失）；'
      '9 国 = 去掉 RO、PT；格式：均值 [95% t CI]，>0 的国家数 / 国家数（P6 为 90% CI）；单位 pp。')
    P('TGT：测试概率没有保存，不计算（规格）。')
    P('')

    # ================================================================ 对账
    P('== 对账（τ = 0；容差 1e-9）==')
    ok_all = True

    def chk(mask, label):
        nonlocal ok_all
        x = AR[mask]
        do = (x.oa_t0 - x.oa_ref).abs().max()
        db = (x.ba_t0 - x.ba_ref).abs().max()
        ok = bool(len(x) > 0 and do < TOL and db < TOL)
        ok_all &= ok
        P(f'{label}：{len(x)} 行，最大 |Δ| OA {do:.2e}，ba_ge30 {db:.2e} → {"通过" if ok else "不通过"}')
        return ok

    for k in KS:
        chk((AR.learner == 'lgb') & (AR.name == f'RT{k}') & (AR.ref == 'G_rows_srs'),
            f'(1) RT{k}（LightGBM i300）与 G rows_t1（srs、cond=none）的 oa、ba_ge30')
    chk((AR.learner == 'rf') & (AR.name == 'Z0') & (AR.ref == 'RF_rows_foreign_only'),
        '(2) RF Z0（k0 的 P）与 guidance_G_ml rows_rf（foreign_only）')
    chk((AR.learner == 'rf') & (AR.name == 'RT200') & (AR.ref == 'RF_rows_srs'),
        '    RF RT200 与 guidance_G_ml rows_rf（srs、k=200）')
    chk((AR.learner == 'lgb') & (AR.name == 'Z0') & (AR.ref == 'G_rows_foreign_only'),
        '(3) LightGBM Z0（P0test_i300）与 G rows_t1（foreign_only，同一 i300 概率的 argmax）')
    chk(AR.ref == 'prior_correct_path',
        '(4) PK-k、PKo、RF PK200 与另一条路径（guidance_G.prior_correct + 概率 argmax + guidance_G._scores）')
    P(f'(5) sel_hash：G sel 文件（sel_pos 经 pool_positions 映射 = sel）与 rows_t1 的 srs sel_hash 逐一相同，RF srs k=200 的 '
      f'sel_hash 与 G 同单元 k=200 相同：共 {n_hash} 次断言通过')
    nfb = int(AR.fallback.fillna(0).sum())
    ok_all &= nfb == 0
    P(f'(6) reweight 回退行合计 {nfb}；臂行数 {len(AR)}（应 = {nU} × 2 × {len(ARMS["lgb"]) + len(ARMS["rf"])} = '
      f'{nU * 2 * (len(ARMS["lgb"]) + len(ARMS["rf"]))}）；差值行 {len(DF)}；D 行 {len(DD)}')
    ok_all &= len(AR) == nU * 2 * (len(ARMS['lgb']) + len(ARMS['rf']))
    P(f'对账结论：{"全部通过" if ok_all else "有不通过项，见上"}')
    KEY['对账'] = '全部通过' if ok_all else '有不通过项'
    P('')

    # ================================================================ 工具
    def unit_diff(lrn, name):
        x = DF[(DF.learner == lrn) & (DF.name == name)]
        return x[UK + ['clf']].assign(d=x.value.to_numpy())

    def feat_block(title, U, level=0.95, sets=(True, False), feats=('both', 'ae', 's2'), tag=None):
        cv = B.country_vals(U, 'd')
        P(title)
        for f in feats:
            for nine in sets:
                s = tsum(sel9(cv[f], nine), level)
                miss = ''
                if f != 'both' and not nine:
                    miss = f'   （缺失单元 {int(U[U.clf == f].d.isna().sum())}/{int((U.clf == f).sum())}）'
                P(f'  {f:4s} {"9 国" if nine else "11 国":5s} {fmt(s)}{miss}')
                if tag and f == 'both':
                    KEY[f'{tag} both {"9" if nine else "11"}国'] = fmt(s)
        return cv

    def d_country(lrn, name, reps=(0, 1)):
        x = DD[(DD.learner == lrn) & (DD.name == name) & DD.rep.isin(reps)]
        return x.groupby('target').value.mean(), int(x.value.isna().sum()), len(x)

    def d_block(title, lrn, name, sets=(False, True), tag=None):
        cv, nm, nn = d_country(lrn, name)
        P(title + f'   （缺失单元 {nm}/{nn}）')
        for nine in sets:
            s = tsum(sel9(cv, nine))
            P(f'  {"9 国" if nine else "11 国":5s} {fmt(s)}')
            if tag:
                KEY[f'{tag} {"9" if nine else "11"}国'] = fmt(s)
        return cv

    def t0_unit(lrn, a, b, met):
        x = AR[AR.learner == lrn]
        W = x.set_index(UK + ['clf', 'name'])[met].unstack('name').reset_index()
        return W[UK + ['clf']].assign(d=(W[a] - W[b]).to_numpy())

    def t0_feat(lrn, arm, met):
        x = AR[(AR.learner == lrn) & (AR.name == arm)]
        W = x.set_index(UK + ['clf'])[met].unstack('clf')
        return (W['ae'] - W['s2']).groupby('target').mean()

    # ================================================================ P1
    P('######## P1  ΔFA(RT200) = FA(RT200) − FA(Z0)，LightGBM，both（v4 H1；主 = 9 国，11 国同报）########')
    feat_block('-- ΔFA(RT200)，both --', unit_diff('lgb', 'dFA_RT200'), feats=('both',), tag='P1 ΔFA(RT200)')
    P('')
    # ================================================================ P2
    P('######## P2  D(Z0)（AE − S2，同 OA），LightGBM（v4 H2；主 = 11 国，9 国同报）########')
    d_block('-- LightGBM D(Z0) --', 'lgb', 'D_Z0', tag='P2 D(Z0)')
    P('')
    # ================================================================ P3
    P('######## P3  LightGBM 分解（v4 A1、A2）：A_k = FA(PK-k) − FA(Z0)；R_k = FA(RT-k) − FA(PK-k)；ΔFA(RT-k)；'
      'A_o = FA(PKo) − FA(Z0) ########')
    P('  注：三者都按"共同可达格点上的逐点差"定义，各自的共同可达集合不同，所以 A_k + R_k 与 ΔFA(RT-k) 不必严格相等。')
    for k in KS:
        feat_block(f'-- A_{k} --', unit_diff('lgb', f'A_{k}'), tag=f'P3 A_{k}')
        feat_block(f'-- R_{k} --', unit_diff('lgb', f'R_{k}'), tag=f'P3 R_{k}')
        feat_block(f'-- ΔFA(RT{k}) --', unit_diff('lgb', f'dFA_RT{k}'), tag=f'P3 ΔFA(RT{k})')
    feat_block('-- A_o = FA(PKo) − FA(Z0)（比例上限）--', unit_diff('lgb', 'A_o'), tag='P3 A_o')
    P('')
    # ================================================================ P4
    P('######## P4  RF（v4 A4；OA★ = RF Z0 在 τ=0 的 OA）：ΔFA(RT200)、A_200、R_200 ########')
    for name, lab in (('dFA_RT200', 'ΔFA(RT200)'), ('A_200', 'A_200'), ('R_200', 'R_200')):
        feat_block(f'-- RF {lab} --', unit_diff('rf', name), tag=f'P4 RF {lab}')
    P('')
    # ================================================================ P5
    P('######## P5  E1：D(RT200)（LightGBM；锚点 = LightGBM Z0 的 min(OA★_ae, OA★_s2)）；RF 的 D(Z0) ########')
    d_block('-- LightGBM D(RT200) --', 'lgb', 'D_RT200', tag='P5 D(RT200) lgb')
    d_block('-- RF D(Z0) --', 'rf', 'D_Z0', tag='P5 D(Z0) rf')
    P('')
    # ================================================================ P6
    P('######## P6  E2：AE − S2 交互量（国家值 ae − 国家值 s2），LightGBM，90% t CI ########')
    for name, lab in (('dFA_RT200', 'ΔFA(RT200)'), ('A_200', 'A_200')):
        cv = B.country_vals(unit_diff('lgb', name), 'd')
        inter = cv['ae'] - cv['s2']
        P(f'-- {lab}：AE − S2 --')
        for nine in (True, False):
            s = tsum(sel9(inter, nine), 0.90)
            P(f'  {"9 国" if nine else "11 国":5s} {fmt(s)}（90% CI）')
            KEY[f'P6 {lab} AE−S2 {"9" if nine else "11"}国 (90%CI)'] = fmt(s)
    P('')
    # ================================================================ P7
    P('######## P7  M1：固定工作点 τ = 0 的差，与 H1、H2 的同 OA 值并列（LightGBM）########')
    P('-- 同 OA（H1）：ΔFA(RT200)，both --')
    cv = B.country_vals(unit_diff('lgb', 'dFA_RT200'), 'd')
    for nine in (True, False):
        P(f'  {"9 国" if nine else "11 国":5s} {fmt(tsum(sel9(cv["both"], nine)))}')
    for met, lab in (('ba_t0', 'BA'), ('oa_t0', 'OA')):
        feat_block(f'-- τ=0：RT200 − Z0 的 {lab} 差 --', t0_unit('lgb', 'RT200', 'Z0', met),
                   tag=f'P7 τ0 RT200−Z0 {lab}')
    P('-- 同 OA（H2）：D(Z0) --')
    cvD, _, _ = d_country('lgb', 'D_Z0')
    for nine in (False, True):
        P(f'  {"9 国" if nine else "11 国":5s} {fmt(tsum(sel9(cvD, nine)))}')
    for met, lab in (('ba_t0', 'BA'), ('oa_t0', 'OA')):
        v = t0_feat('lgb', 'Z0', met)
        P(f'-- τ=0：Z0 的 AE − S2 {lab} 差 --')
        for nine in (False, True):
            s = tsum(sel9(v, nine))
            P(f'  {"9 国" if nine else "11 国":5s} {fmt(s)}')
            KEY[f'P7 τ0 Z0 AE−S2 {lab} {"9" if nine else "11"}国'] = fmt(s)
    P('')
    # ================================================================ P8
    P('######## P8  A 的 OA 通道：OA(PK-k, τ=0) − OA(Z0, τ=0)，LightGBM（主 = 11 国，9 国同报）########')
    for k in (50, 200):
        feat_block(f'-- PK{k} − Z0 的 OA 差（τ = 0）--', t0_unit('lgb', f'PK{k}', 'Z0', 'oa_t0'), sets=(False, True),
                   tag=f'P8 OA PK{k}−Z0')
    P('')

    # ================================================================ P9
    P('######## P9  功效（v4 第 6 节；确认运行 = 每国 2 个单元；国家层 SD 只用 rep0 单元（s0/r0、s1/r0））########')
    P('单样本 t（df = n − 1），单侧 α = 0.025 与 0.0125（Holm 首步）；MDE80 = 80% 功效的非中心 t 精确解；'
      '功效 = 真值取回溯均值（4 单元国家值的均值，即 P1 / P2）或其一半时的拒绝概率（非中心 t）。')
    P('假设：确认运行的国家集合与国家间异质性同本回溯；本回溯每单元只有 1 次 SRS 抽样（确认运行为 3 次平均），'
      '所以 rep0 的 SD 含单次抽样的变异。')
    pw_rows = []
    specs = [('H1', 'ΔFA(RT200) LightGBM both', '9', lambda reps: B.country_vals(
        unit_diff('lgb', 'dFA_RT200')[lambda u: u.rep.isin(reps)], 'd')['both']),
             ('H2', 'D(Z0) LightGBM', '11', lambda reps: d_country('lgb', 'D_Z0', reps)[0])]
    P('  假设  量                          国家  n   回溯均值  SD_r0 | α       MDE80  功效(真值=均值)  功效(真值=均值/2)')
    for hyp, qn, cs, fn in specs:
        nine = cs == '9'
        v4 = sel9(fn((0, 1)), nine).dropna()
        vr0 = sel9(fn((0,)), nine).dropna()
        n, sd, m = len(vr0), float(vr0.std(ddof=1)) if len(vr0) > 1 else np.nan, float(v4.mean())
        for a in ALPHAS:
            r = dict(hyp=hyp, quantity=qn, set=cs, n=n, mean4=m, sd_r0=sd, alpha_one_sided=a, mde80=mde80(sd, n, a),
                     power_at_mean=power(m, sd, n, a), power_at_half_mean=power(m / 2, sd, n, a))
            pw_rows.append(r)
            P(f'  {hyp:4s}  {qn:26s} {cs:>3s}  {n:2d}  {m:+8.2f}  {sd:5.2f} | {a:<7g} {r["mde80"]:5.2f}  '
              f'{r["power_at_mean"]:15.3f}  {r["power_at_half_mean"]:17.3f}')
            KEY[f'P9 {hyp} α={a:g}'] = (f'n={n} mean={m:+.2f} SD_r0={sd:.2f} MDE80={r["mde80"]:.2f} '
                                        f'power@mean={r["power_at_mean"]:.3f} power@half={r["power_at_half_mean"]:.3f}')
    P('  v4 第 6 节现有代理值（P_T 口径，供对照）：H1 9 国 SD ≈ 0.94、MDE80 ≈ 1.0–1.2、回溯均值 +2.10、真值一半时功效 ≈ 0.7；'
      'H2 11 国 SD ≈ 3.73、MDE80 ≈ 3.5–4.0、回溯均值 +5.25、真值一半时功效 ≈ 0.4。')
    P('')

    # ================================================================ P10
    P('######## P10  与已有口径的并列（连续性说明）：P1、P2、P3 ########')
    P('  列：PC-1（Z0 = raw P0test_i300；两臂之差 = 共同可达格点上的逐点差） | B2（partB2/fa_rows_B2.csv，新网格；Z0 = P_T 温度校准；'
      '差 = 两臂各自 FA 之差） | 确认前分析（P1/P3：preconf/label/label_rows.csv，LightGBM、W4，P_T；P2：preconf/ae/'
      'ae_frontier_rows.csv 的 AE1 D(Z0)，锚点 min、W4，P_T） | 均值差 = PC-1 − B2（P2 为 PC-1 − 确认前）')
    P('  RT-k 在三种口径下都是同一份 i300 未校准概率；差别来自 Z0 / PK 的底座（raw 对 P_T）、OA★ 锚点与差值的定义。')
    units_now = set(R.unit)
    FB = pd.read_csv(FA_B2, float_precision='round_trip')
    FB = FB[(FB.grid == 'new') & FB.unit.isin(units_now)]
    WB = FB.set_index(UK + ['clf', 'arm'])['FA'].unstack('arm').reset_index()
    LR_ = pd.read_csv(LABEL_ROWS, float_precision='round_trip')
    LR_ = LR_[(LR_.width == 4) & (LR_.learner == 'lgb') & LR_.unit.isin(units_now)]
    WL = LR_.set_index(UK + ['clf', 'arm'])['FA'].unstack('arm').reset_index()

    def old_cv(W, a, b):
        if a not in W or b not in W:
            return None
        return B.country_vals(W[UK + ['clf']].assign(d=(W[a] - W[b]).to_numpy()), 'd')['both']

    items = [('P1', 'ΔFA(RT200)', 'dFA_RT200', ('RT200', 'Z0'))]
    for k in KS:
        items += [('P3', f'A_{k}', f'A_{k}', (f'PK{k}', 'Z0')), ('P3', f'R_{k}', f'R_{k}', (f'RT{k}', f'PK{k}')),
                  ('P3', f'ΔFA(RT{k})', f'dFA_RT{k}', (f'RT{k}', 'Z0'))]
    items += [('P3', 'A_o', 'A_o', ('PKo', 'Z0'))]
    for pid, lab, name, (a, b) in items:
        cn = B.country_vals(unit_diff('lgb', name), 'd')['both']
        cb, cl = old_cv(WB, a, b), old_cv(WL, a, b)
        for nine in (True, False):
            sn = tsum(sel9(cn, nine))
            sb = tsum(sel9(cb, nine)) if cb is not None else None
            sl = tsum(sel9(cl, nine)) if cl is not None else None
            dm = f'{sn["mean"] - sb["mean"]:+.2f}' if sb is not None else '—'
            P(f'  {pid} {lab:11s} both {"9 国" if nine else "11 国":5s} {fmt(sn)} | '
              f'{fmt(sb) if sb is not None else "—（B2 无此臂）"} | {fmt(sl) if sl is not None else "—（无此臂）"} | {dm}')
    AE1 = pd.read_csv(AE1_ROWS, float_precision='round_trip')
    AE1 = AE1[(AE1.learner == 'lgb') & (AE1.arm == 'Z0') & (AE1.anchor == 'min') & (AE1.width == 4) &
              AE1.unit.isin(units_now)]
    ca = AE1.groupby('target').D.mean()
    cvD, _, _ = d_country('lgb', 'D_Z0')
    for nine in (False, True):
        sn, sa = tsum(sel9(cvD, nine)), tsum(sel9(ca, nine))
        P(f'  P2 D(Z0)       —    {"9 国" if nine else "11 国":5s} {fmt(sn)} | —（B2 无 D） | {fmt(sa)} | '
          f'{sn["mean"] - sa["mean"]:+.2f}')
    P('')

    # ================================================================ 另报：并列比例
    P('######## 另报：各臂 τ = 0 时 argmax 并列的比例（ℓ = log P 的最大值与次大值相等的测试行比例；单元 × 特征上的均值 / 最大；'
      '有并列的单元×特征数）########')
    for lrn in LEARNERS:
        for arm in ARMS[lrn]:
            parts = []
            for f in FEATS:
                x = AR[(AR.learner == lrn) & (AR.name == arm) & (AR.clf == f)].tie_t0
                parts.append(f'{f} {x.mean():.4f} / {x.max():.4f}（{int((x > 0).sum())}/{len(x)}）')
            P(f'  {LNAME[lrn]:8s} {arm:6s} ' + '；'.join(parts))
    P('  注：LightGBM Z0 存为 float16（P0test），PK-k、PKo 由它重加权得到；RT-k 与 RF 存为 float32。')
    P('')
    return lines, pd.DataFrame(pw_rows), KEY


if __name__ == '__main__':
    main()
