"""
PC-1 独立复核：只算 P1、P2、P3（k=200）、P4（raw 口径）。
规格：label_expansion/PC1_SPEC_2026-09-29.md（sha256[:16]=eb82a82b7acea883，不改）；所服务的方案 CONFIRM_PLAN_DRAFT_v4.md 第 2–6 节。
C 级、只作参照；不训练；只读 G stage0、G 选样与已存概率、guidance_G_ml 的 RF 概率与 rows；测试标签只用于评估（以及规格定义的 PKo 上限臂）。

独立性：不 import step0B / step0B2 / 主实现；τ 族、包络、FA 差、D、Laplace 与 PK 重加权都在本文件内重写。
guidance_G 只当数据层用：GData（标签 y）、pool_positions、sel_hash、class_weights_from、MIN_N。

口径（逐条对应规格）：
  Z0（LightGBM） stage0/{国}_s{划分}.npz 的 P0test_{f}_i300（float16 → float64，按行归一化）
  RT200（LightGBM）rows_t1 中 method=srs、cond=none、k=200、clf=f 的 sel_hash → probs/{单元}_{f}_k200_{hash}.npz 的 i300
  PK200          reweight(Z0, laplace(y_sel, 10, α=1), π_train)；y_sel = 同一批 SRS(200) 选样的标签
  PKo            reweight(Z0, 测试集真实比例, π_train)
  π_train        外国池按 class_weights_from 加权的类分布（= guidance_G.pi_train_of([], 0)）
  RF             Z0 / PK200 的底座 = guidance_G_ml/probs/rf/{国}_s{划分}_{f}_k0.npz 的 P；
                 RT200 = rows_rf_*.csv 中 method=srs、k=200 的 sel_hash → probs/rf/{单元}_{f}_k200_{hash}.npz 的 P
  τ 族            ℓ = log P − τ·log π̂，π̂ = 该臂测试集平均预测；τ ∈ [−2, 2] 步长 0.05；精确 log（0 → −inf）；argmax 并列取第一个
  包络            (OA, BA) 平面上相邻 τ 点线性插值；OA = o 的所有段取 BA 最大；OA 相同的平段取两端较大者；无段经过 → 不可达
  FA 差           窗口 [OA★ − 4, OA★] 步长 0.1（41 点），OA★ = 同一学习器、同一特征的 Z0 在 τ=0 的 OA；
                 两臂之差在共同可达格点上逐点平均；共同可达 < 50% 记缺失
  D              锚点 = min(OA★_ae, OA★_s2)；两特征该臂的包络在共同可达格点上平均 BA_ae − BA_s2；< 50% 记缺失
  国家值          逐单元（2 划分 × 2 重复）先算配对差，再对该国可用单元取平均；both = AE、S2 两特征国家值的平均
  报告            9 国（去 RO、PT）与 11 国：均值、95% t CI（df = n−1）、>0 的国家数

用法：/Users/perng_mini/AlphaCrops_local/.venv/bin/python -B indep_pc1.py
"""
import os
import sys
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_v] = '2'
sys.dont_write_bytecode = True  # 不在项目目录写 __pycache__
import glob
import hashlib
import time

import numpy as np
import pandas as pd
from scipy import stats

L = '/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops/label_expansion'
sys.path.insert(0, L)
import guidance_G as G  # noqa: E402  只用数据层（只读）

HERE = os.path.dirname(os.path.abspath(__file__))
SPEC = os.path.join(L, 'PC1_SPEC_2026-09-29.md')
SPEC_HASH = 'eb82a82b7acea883'
R_G = '/Users/perng_mini/AlphaCrops_local/results/guidance_G'
R_ML = '/Users/perng_mini/AlphaCrops_local/results/guidance_G_ml'
assert os.path.realpath(G.OUT) == os.path.realpath(R_G), G.OUT

C = 10
K = 200
FEATS = ['ae', 's2']
COUNTRIES = ['AT', 'BE', 'DE', 'EE', 'LT', 'LV', 'PT', 'RO', 'SE', 'SI', 'SK']
DROP = ['RO', 'PT']
TAUS = np.round(np.linspace(-2.0, 2.0, 81), 2)
I_T0 = int(np.flatnonzero(TAUS == 0.0)[0])
OFFS = np.round(np.linspace(-4.0, 0.0, 41), 1)
EPS = 1e-9           # 包络判定"段经过 o"的浮点容差（OA 取值间隔 ≈ 100/n ≫ 1e-9）
MIN_COV = 0.5
assert len(TAUS) == 81 and TAUS[I_T0] == 0.0 and len(OFFS) == 41 and OFFS[0] == -4.0 and OFFS[-1] == 0.0


def sha16(path):
    with open(path, 'rb') as fh:
        return hashlib.sha256(fh.read()).hexdigest()[:16]


# ---------------------------------------------------------------- 概率与先验
def rownorm(P):
    P = np.asarray(P, dtype=np.float64)
    return P / P.sum(axis=1, keepdims=True)


def exact_log(P):
    with np.errstate(divide='ignore'):
        return np.log(P)


def laplace(ys):
    """(n_c + 1) / (k + C)，对全部 10 类平滑。"""
    return (np.bincount(ys, minlength=C).astype(np.float64) + 1.0) / (len(ys) + C)


def reweight(P, pi_new, pi_train):
    """q ∝ p · pi_new / pi_train；pi_train = 0 的类权重为 0；行和为 0 时保留原 p。返回 (q, 回退行数)。"""
    w = np.zeros(C)
    ok = pi_train > 0
    w[ok] = pi_new[ok] / pi_train[ok]
    q = P * w[None, :]
    s = q.sum(axis=1)
    bad = ~(s > 0) | ~np.isfinite(s)
    out = P.copy()
    good = ~bad
    out[good] = q[good] / s[good, None]
    return out, int(bad.sum())


# ---------------------------------------------------------------- τ 族与评分
def tau_curve(P, yT, big):
    """P：已归一化的 (n, C) 概率。返回 OA (81,)、BA_ge30 (81,)、τ=0 的预测。
    π̂_c = 0 的类整列概率为 0（log = −inf），其偏移取 0（避免 −inf·0）。"""
    lp = exact_log(P)
    pihat = P.mean(axis=0)
    lpi = np.zeros(C)
    pos = pihat > 0
    lpi[pos] = np.log(pihat[pos])
    masks = [yT == c for c in big]
    oa = np.empty(len(TAUS))
    ba = np.empty(len(TAUS))
    pred0 = None
    for i, t in enumerate(TAUS):
        pred = np.argmax(lp - t * lpi[None, :], axis=1)   # 并列取第一个
        hit = pred == yT
        oa[i] = 100.0 * hit.mean()
        ba[i] = 100.0 * np.mean([hit[m].mean() for m in masks])
        if i == I_T0:
            pred0 = pred
    return oa, ba, pred0


def envelope(oa, ba, grid):
    """单条 τ 曲线在 grid 上的 BA 上包络；不可达为 NaN。"""
    a0, a1, b0, b1 = oa[:-1], oa[1:], ba[:-1], ba[1:]
    lo = np.minimum(a0, a1) - EPS
    hi = np.maximum(a0, a1) + EPS
    flat = np.abs(a1 - a0) < EPS
    out = np.full(len(grid), np.nan)
    for g, o in enumerate(grid):
        on = (o >= lo) & (o <= hi)
        if not on.any():
            continue
        vals = []
        for j in np.flatnonzero(on):
            if flat[j]:
                vals.append(max(b0[j], b1[j]))
            else:
                lam = min(max((o - a0[j]) / (a1[j] - a0[j]), 0.0), 1.0)
                vals.append(b0[j] + lam * (b1[j] - b0[j]))
        out[g] = max(vals)
    return out


def pair_diff(eA, eB):
    """逐点差在共同可达格点上的平均；共同可达 < 50% 记缺失。返回 (差, 共同可达率)。"""
    m = np.isfinite(eA) & np.isfinite(eB)
    cov = float(m.mean())
    return (float(np.mean(eA[m] - eB[m])) if cov >= MIN_COV else np.nan), cov


def reach_fa(e):
    cov = float(np.isfinite(e).mean())
    return (float(np.nanmean(e)) if cov >= MIN_COV else np.nan), cov


# ---------------------------------------------------------------- 数据
def load_rows():
    GR = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(os.path.join(R_G, 'rows_t1_*.csv')))], ignore_index=True)
    GR = GR[(GR.cond == 'none') & (((GR.method == 'srs') & (GR.k == K)) | (GR.method == 'foreign_only'))]
    RF = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(os.path.join(R_ML, 'rows_rf_*.csv')))], ignore_index=True)
    RF = RF[(RF.learner == 'rf') & (RF.cond == 'none') &
            (((RF.method == 'srs') & (RF.k == K)) | (RF.method == 'foreign_only'))]
    assert (RF.status == 'ok').all()
    return GR, RF


def one_row(T, **kw):
    m = np.ones(len(T), bool)
    for a, v in kw.items():
        m &= (T[a] == v).to_numpy()
    x = T[m]
    assert len(x) == 1, (kw, len(x))
    return x.iloc[0]


# ---------------------------------------------------------------- 主循环
def main():
    t0 = time.time()
    assert sha16(SPEC) == SPEC_HASH, '规格哈希与登记值不一致'
    D = G.GData()
    y = np.asarray(D.y)
    GR, RF = load_rows()
    est_rows, arm_rows, chk_rows = [], [], []

    for tgt in COUNTRIES:
        for split in (0, 1):
            S = np.load(os.path.join(R_G, 'stage0', f'{tgt}_s{split}.npz'))
            cand, test, allT, fr = S['cand'], S['test'], S['allT'], S['foreign']
            assert np.array_equal(allT[len(cand):], test)
            yT = y[test]
            nT = len(yT)
            big = np.flatnonzero(np.bincount(yT, minlength=C) >= G.MIN_N)
            yF = y[fr]
            cw = G.class_weights_from(yF)
            pi_train = np.bincount(yF, weights=cw[yF], minlength=C) / cw[yF].sum()
            pi_test = np.bincount(yT, minlength=C) / nT
            # 与重复无关的底座（每 (国家, 划分, 特征) 一次）
            base = {}
            for f in FEATS:
                Z0 = rownorm(S[f'P0test_{f}_i300'])
                Zrf = rownorm(np.load(os.path.join(R_ML, 'probs', 'rf', f'{tgt}_s{split}_{f}_k0.npz'))['P'])
                PKo, nfb_o = reweight(Z0, pi_test, pi_train)
                base[f] = dict(Z0=Z0, Zrf=Zrf, cv={('lgb', 'Z0'): tau_curve(Z0, yT, big),
                                                   ('lgb', 'PKo'): tau_curve(PKo, yT, big),
                                                   ('rf', 'Z0'): tau_curve(Zrf, yT, big)}, nfb_o=nfb_o)
            for rep in (0, 1):
                unit = f'{tgt}_s{split}_r{rep}'
                pool_pos = G.pool_positions(S, split, rep)
                zs = np.load(os.path.join(R_G, 'sel', f'{unit}_none_srs_k{K}.npz'))
                sel = zs['sel']
                assert np.array_equal(allT[pool_pos][zs['sel_pos']], sel), unit
                assert len(sel) == K
                h = G.sel_hash(sel)
                ys = y[sel]
                p_hat = laplace(ys)
                CV, OASTAR = {}, {}
                for f in FEATS:
                    cell = dict(target=tgt, split=split, rep=rep, unit=unit, clf=f)
                    b = base[f]
                    g2 = one_row(GR, target=tgt, split=split, rep=rep, clf=f, method='srs', k=K)
                    g0 = one_row(GR, target=tgt, split=split, rep=rep, clf=f, method='foreign_only')
                    r2 = one_row(RF, target=tgt, split=split, rep=rep, clf=f, method='srs', k=K)
                    r0 = one_row(RF, target=tgt, split=split, rep=rep, clf=f, method='foreign_only')
                    assert g2.sel_hash == h and r2.sel_hash == h, (unit, f)
                    Prt = rownorm(np.load(os.path.join(R_G, 'probs', f'{unit}_{f}_k{K}_{h}.npz'))['i300'])
                    Prf = rownorm(np.load(os.path.join(R_ML, 'probs', 'rf', f'{unit}_{f}_k{K}_{h}.npz'))['P'])
                    PK, nfb = reweight(b['Z0'], p_hat, pi_train)
                    PKrf, nfb_rf = reweight(b['Zrf'], p_hat, pi_train)
                    cv = dict(b['cv'])
                    cv[('lgb', 'RT200')] = tau_curve(Prt, yT, big)
                    cv[('lgb', 'PK200')] = tau_curve(PK, yT, big)
                    cv[('rf', 'RT200')] = tau_curve(Prf, yT, big)
                    cv[('rf', 'PK200')] = tau_curve(PKrf, yT, big)
                    CV[f] = cv
                    # τ = 0 对账
                    for (lrn, arm), ref in ((('lgb', 'RT200'), g2), (('lgb', 'Z0'), g0),
                                            (('rf', 'RT200'), r2), (('rf', 'Z0'), r0)):
                        chk_rows.append(dict(cell, learner=lrn, arm=arm, oa=cv[(lrn, arm)][0][I_T0],
                                             ba_ge30=cv[(lrn, arm)][1][I_T0], oa_ref=float(ref.oa),
                                             ba_ref=float(ref.ba_ge30),
                                             ref='G_rows_t1' if lrn == 'lgb' else 'G_ml_rows_rf'))
                    # FA 差（每学习器用自己的 Z0 锚点）
                    for lrn, arms, pairs in (
                            ('lgb', ['Z0', 'PK200', 'PKo', 'RT200'],
                             {'dFA_RT200': ('RT200', 'Z0'), 'A_200': ('PK200', 'Z0'),
                              'R_200': ('RT200', 'PK200'), 'A_o': ('PKo', 'Z0')}),
                            ('rf', ['Z0', 'PK200', 'RT200'],
                             {'dFA_RT200': ('RT200', 'Z0'), 'A_200': ('PK200', 'Z0'),
                              'R_200': ('RT200', 'PK200')})):
                        oa_star = cv[(lrn, 'Z0')][0][I_T0]
                        OASTAR[(lrn, f)] = oa_star
                        grid = oa_star + OFFS
                        env = {a: envelope(cv[(lrn, a)][0], cv[(lrn, a)][1], grid) for a in arms}
                        for a in arms:
                            fa1, rch = reach_fa(env[a])
                            arm_rows.append(dict(cell, learner=lrn, arm=a, OA_star=oa_star,
                                                 oa_t0=cv[(lrn, a)][0][I_T0], ba_t0=cv[(lrn, a)][1][I_T0],
                                                 oa_min=float(cv[(lrn, a)][0].min()),
                                                 oa_max=float(cv[(lrn, a)][0].max()),
                                                 reach=rch, FA_single=fa1,
                                                 pk_fallback={'PK200': nfb if lrn == 'lgb' else nfb_rf,
                                                              'PKo': b['nfb_o']}.get(a, np.nan)))
                        for est, (a, b_) in pairs.items():
                            d, cov = pair_diff(env[a], env[b_])
                            est_rows.append(dict(cell, learner=lrn, est=est, value=d, cov=cov))
                # D(Z0)，LightGBM：锚点 = 两特征 OA★ 较小者
                anchor = min(OASTAR[('lgb', 'ae')], OASTAR[('lgb', 's2')])
                grid = anchor + OFFS
                e_ae = envelope(*CV['ae'][('lgb', 'Z0')][:2], grid)
                e_s2 = envelope(*CV['s2'][('lgb', 'Z0')][:2], grid)
                d, cov = pair_diff(e_ae, e_s2)
                est_rows.append(dict(target=tgt, split=split, rep=rep, unit=unit, clf='ae-s2', learner='lgb',
                                     est='D_Z0', value=d, cov=cov, anchor=anchor))
                print(f'[{time.time() - t0:.0f}s] {unit} 完成', flush=True)

    E = pd.DataFrame(est_rows)
    A = pd.DataFrame(arm_rows)
    K_ = pd.DataFrame(chk_rows)
    CT = country_table(E)
    outs = {'indep_unit_est.csv': E, 'indep_unit_arms.csv': A, 'indep_checks.csv': K_, 'indep_country.csv': CT}
    for name, df in outs.items():
        df.to_csv(os.path.join(HERE, name), index=False)
    lines = summarize(E, A, K_, CT)
    with open(os.path.join(HERE, 'indep_summary.txt'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    hl = [f'{n} sha256[:16]={sha16(os.path.join(HERE, n))}'
          for n in ['indep_pc1.py', 'indep_summary.txt'] + list(outs)]
    with open(os.path.join(HERE, 'indep_hashes.txt'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(hl) + '\n')
    print('\n'.join(hl))
    print(f'总耗时 {time.time() - t0:.0f}s')


# ---------------------------------------------------------------- 汇总
def country_table(E):
    """国家值：逐 (学习器, 估计量, 特征) 对该国可用单元取平均；both = ae、s2 两特征国家值的平均（任一缺失则缺失）。"""
    rows = []
    for (lrn, est), x in E.groupby(['learner', 'est']):
        cv = {c: g.groupby('target')['value'].mean() for c, g in x.groupby('clf')}
        nu = {c: g.groupby('target')['value'].apply(lambda s: int(s.notna().sum())) for c, g in x.groupby('clf')}
        if 'ae' in cv and 's2' in cv:
            both = pd.concat([cv['ae'], cv['s2']], axis=1).mean(axis=1, skipna=False)
            cv['both'] = both
        for c, s in cv.items():
            for t in COUNTRIES:
                rows.append(dict(learner=lrn, est=est, clf=c, target=t, value=s.get(t, np.nan),
                                 n_units=int(nu[c].get(t, 0)) if c in nu else np.nan))
    return pd.DataFrame(rows)


def t_sum(v):
    v = np.asarray(pd.Series(v).dropna(), dtype=float)
    n = len(v)
    if n < 2:
        return dict(n=n, mean=np.nan, lo=np.nan, hi=np.nan, sd=np.nan, npos=int((v > 0).sum()))
    m, sd = float(v.mean()), float(v.std(ddof=1))
    q = stats.t.ppf(0.975, n - 1)
    return dict(n=n, mean=m, lo=m - q * sd / np.sqrt(n), hi=m + q * sd / np.sqrt(n), sd=sd, npos=int((v > 0).sum()))


def fmt(s):
    if s['n'] < 2:
        return f'n={s["n"]}（不足 2 国）'
    return f'{s["mean"]:+.2f} [{s["lo"]:+.2f}, {s["hi"]:+.2f}]，>0 {s["npos"]}/{s["n"]}'


def summarize(E, A, K_, CT):
    lines = []
    P = lines.append
    P('PC-1 独立复核（只算 P1、P2、P3（k=200）、P4；raw 口径）—— C 级，只作参照，不进摘要。')
    P(f'规格 PC1_SPEC_2026-09-29.md sha256[:16]={sha16(SPEC)}（登记 {SPEC_HASH}）；本脚本 indep_pc1.py '
      f'sha256[:16]={sha16(os.path.abspath(__file__))}；未 import step0B / step0B2 / 主实现。')
    P(f'单元 {E.unit.nunique()}（11 国 × 2 划分 × 2 重复）× 特征 2；τ ∈ [−2, 2] 步长 0.05（{len(TAUS)} 点）；'
      f'W = [OA★ − 4, OA★] 步长 0.1（{len(OFFS)} 点）；两臂之差在共同可达格点上逐点平均，共同可达 < 50% 记缺失。')
    P('国家值 = 该国可用单元的平均（配对差先在单元内算）；both = AE、S2 国家值的平均；9 国 = 去 RO、PT；95% t CI（df = n−1）。')
    P('')
    P('== 核对（τ = 0）==')
    for (lrn, arm), x in K_.groupby(['learner', 'arm']):
        P(f'  {lrn:3s} {arm:6s} 与 {x.ref.iloc[0]}：{len(x)} 行，最大 |Δ| OA {(x.oa - x.oa_ref).abs().max():.2e}，'
          f'ba_ge30 {(x.ba_ge30 - x.ba_ref).abs().max():.2e}')
    P('  sel_hash：G sel 文件（sel 与 allT[pool_pos][sel_pos] 相同）、rows_t1（srs k=200，ae/s2）、rows_rf（srs k=200，ae/s2）'
      '三方一致（断言通过）。')
    pk = A[A.arm.isin(['PK200', 'PKo'])]
    P(f'  PK 重加权回退行合计 {int(pk.pk_fallback.sum())}。')
    P('')

    def block(title, lrn, est, feats=('both', 'ae', 's2')):
        P(title)
        x = E[(E.learner == lrn) & (E.est == est)]
        for c in feats:
            s = CT[(CT.learner == lrn) & (CT.est == est) & (CT.clf == c)].set_index('target')['value']
            s9 = s.drop(DROP)
            miss = int(x[x.clf == c].value.isna().sum()) if c != 'both' else None
            P(f'  {c:5s} 9 国 {fmt(t_sum(s9))} | 11 国 {fmt(t_sum(s))}'
              + (f'   （缺失单元 {miss}/{int((x.clf == c).sum())}）' if miss is not None else ''))

    P('######## P1  ΔFA(RT200) = FA(RT200) − FA(Z0)，LightGBM（H1；主 = both、9 国）########')
    block('', 'lgb', 'dFA_RT200')
    P('')
    P('######## P2  D(Z0) = AE − S2，LightGBM（H2；主 = 11 国）########')
    block('', 'lgb', 'D_Z0', feats=('ae-s2',))
    P('')
    P('######## P3（k = 200）LightGBM：A_200、R_200、ΔFA(RT200)；A_o = FA(PKo) − FA(Z0)（A1、A2）########')
    block('-- A_200 = FA(PK200) − FA(Z0) --', 'lgb', 'A_200')
    block('-- R_200 = FA(RT200) − FA(PK200) --', 'lgb', 'R_200')
    block('-- ΔFA(RT200)（同 P1）--', 'lgb', 'dFA_RT200')
    block('-- A_o = FA(PKo) − FA(Z0) --', 'lgb', 'A_o')
    P('')
    P('######## P4  RF（锚点 = RF 自身 Z0 在 τ=0 的 OA）：ΔFA(RT200)、A_200、R_200（A4）########')
    block('-- ΔFA(RT200) --', 'rf', 'dFA_RT200')
    block('-- A_200 --', 'rf', 'A_200')
    block('-- R_200 --', 'rf', 'R_200')
    P('')
    P('######## 国家值（pp）########')
    cols = [('lgb', 'dFA_RT200', 'both'), ('lgb', 'D_Z0', 'ae-s2'), ('lgb', 'A_200', 'both'), ('lgb', 'R_200', 'both'),
            ('lgb', 'A_o', 'both'), ('rf', 'dFA_RT200', 'both'), ('rf', 'A_200', 'both'), ('rf', 'R_200', 'both')]
    names = ['P1 dRT200', 'P2 D(Z0)', 'A_200', 'R_200', 'A_o', 'RF dRT200', 'RF A_200', 'RF R_200']
    P('  国家 ' + ' '.join(f'{n:>10s}' for n in names))
    piv = CT.set_index(['learner', 'est', 'clf', 'target'])['value']
    for t in COUNTRIES:
        P(f'  {t}   ' + ' '.join(f'{piv.get((l_, e_, c_, t), np.nan):+10.2f}' for l_, e_, c_ in cols))
    P('')
    P('######## 可达情况（描述，逐单元 × 特征；锚点为本学习器 Z0 的 OA★）########')
    for (lrn, arm), x in A.groupby(['learner', 'arm'], sort=False):
        P(f'  {lrn:3s} {arm:6s} 全可达 {int((x.reach >= 1).sum()):3d}，部分 {int(((x.reach >= MIN_COV) & (x.reach < 1)).sum()):3d}，'
          f'单臂可达 < 50% {int((x.reach < MIN_COV).sum()):3d}（共 {len(x)}）')
    for (lrn, est), x in E.groupby(['learner', 'est']):
        P(f'  {lrn:3s} {est:9s} 共同可达率：最小 {x["cov"].min():.3f}，<100% {int((x["cov"] < 1).sum())}，'
          f'<50%（缺失）{int((x["cov"] < MIN_COV).sum())}（共 {len(x)}）')
    return lines


if __name__ == '__main__':
    main()
