"""
标签线独立复核（L1 与 L2 的部分量）——规格 label_expansion/PRECONF_ANALYSES_SPEC_2026-09-29.md
（sha256[:16]=d0906bd1417839d2，不修改）。事后分析，C 级、描述性；不训练任何分类器，只读已存概率与 G 的选样文件。

本流只依据规格独立实现（不读同事在 verify/preconf/label/ 下以 label_ 开头的任何文件）：
  L1（LightGBM，44 单元 × {ae, s2}）：RTPK-k = RT-k 的测试概率用同一批 k 个 SRS 标签的 p̂（Laplace，α=1）重加权，
      相对的训练先验 = 该重训模型的实际训练先验（外国样本权重 cw[y]、目标标签权重 R_MULT·cw[y] 下的类分布，
      即 guidance_G.run_unit 的 pi_train_of(sel, R_MULT)）；
      R*_k = FA(RTPK-k) − FA(PK-k)；FA(RTPK-k) − FA(RT-k)；可达率。k ∈ {50, 200, 400}（k=200 为主）。
  L2（LR、MLP，k=200）：Z0 = 本学习器 k=0 概率；PK200 = Z0 用同一批 200 个标签的 p̂ 重加权（训练先验 = 外国池先验
      pi_train_of([], 0)）；RT200 = 本学习器 SRS k=200 概率；RTPK200 = RT200 用 p̂ 相对 pi_train_of(sel, R_MULT) 重加权；
      锚点 = 本学习器 Z0 在 τ=0 的 OA；A_200 = FA(PK200) − FA(Z0)，R_200 = FA(RT200) − FA(PK200)，
      R*_200 = FA(RTPK200) − FA(PK200)，ΔFA(RT200) = FA(RT200) − FA(Z0)。

口径复用（import 已审计的 step0B / step0B2，不改其文件）：
  τ 族（step0B2.tau_preds_grid，网格 −2..2 步长 0.05）、π̂（该臂测试集平均预测，与 step0B2 相同的计算路径）、
  精确 log（step0B.log_or_neginf）、OA / ba_ge30（step0B.Scorer）、上包络与 FA（step0B.envelope、step0B.fa_of；
  W4 = [锚点−4, 锚点] 步长 0.1；可达率 <50% 记缺失）。
本流自写：类权重与训练先验（含重训模型的 R_MULT 先验）、Laplace 先验、重加权、锚点与窗口、国家汇总与 t 区间；
  另有一份独立的包络 / FA 实现，只用于与 step0B.envelope 对账。

用法：/Users/perng_mini/AlphaCrops_local/.venv/bin/python -B indep_label.py
"""
import os
import sys
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_v] = '2'
sys.dont_write_bytecode = True
import glob
import hashlib
import time

import numpy as np
import pandas as pd
from scipy import stats

OUT = os.path.dirname(os.path.abspath(__file__))
PARTB = '/Users/perng_mini/AlphaCrops_local/verify/step0/partB'
PARTB2 = '/Users/perng_mini/AlphaCrops_local/verify/step0/partB2'
sys.path.insert(0, PARTB)
sys.path.insert(0, PARTB2)
import step0B as B  # noqa: E402  （导入时 chdir 到项目目录，只读）
import step0B2 as B2  # noqa: E402
import check3_priors as CP  # noqa: E402  只用于对账（重加权与 Laplace 的逐位比较）

G = B.G
SPEC = os.path.join(B.L, 'PRECONF_ANALYSES_SPEC_2026-09-29.md')
SPEC_HASH = 'd0906bd1417839d2'
R_G = '/Users/perng_mini/AlphaCrops_local/results/guidance_G'
R_ML = '/Users/perng_mini/AlphaCrops_local/results/guidance_G_ml'
C = 10
FEATS = ['ae', 's2']
COUNTRIES = 'AT,BE,DE,EE,LT,LV,PT,RO,SE,SI,SK'.split(',')
DROP = ['RO', 'PT']
KS = [50, 200, 400]
R_MULT = 50.0
N_POOL = 30_000
TAUS = B2.TAUS_NEW                     # −2..2 步长 0.05
I_T0 = int(np.flatnonzero(TAUS == 0.0)[0])
IDX_OLD = B2.IDX_OLD                   # 旧网格（−0.5..2）在新网格中的列，只用于与 partB 对账
OFFS = np.round(np.arange(-4.0, 0.0 + 1e-9, 0.1), 1)   # W4，41 点
assert np.array_equal(OFFS, B.OFFS)
MIN_COV = 0.5
UKEY = ['learner', 'target', 'split', 'rep', 'unit', 'clf']


def sha16(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:16]


# ---------------------------------------------------------------- 本流自写的先验与重加权
def my_class_weights(yF):
    """外国池类别权重：n / (出现类数 × n_c)；外国没有的类取 1（与 stageA.class_weights_from 同一规则，另行实现）。"""
    n = np.bincount(yF, minlength=C).astype(float)
    have = n > 0
    w = np.ones(C)
    w[have] = len(yF) / (have.sum() * n[have])
    return w


def my_train_prior(yF, ys, cw, r_mult):
    """训练样本权重下的类分布：外国样本权重 cw[y]，目标标签权重 r_mult·cw[y]（逐类计数相乘，不走 bincount 加权）。"""
    nF = np.bincount(yF, minlength=C).astype(float)
    nS = np.bincount(ys, minlength=C).astype(float) if len(ys) else np.zeros(C)
    mass = cw * (nF + r_mult * nS)
    return mass / mass.sum()


def my_laplace(ys, alpha=1.0):
    n = np.bincount(ys, minlength=C).astype(float)
    return (n + alpha) / (len(ys) + alpha * C)


def my_reweight(P, pi_new, pi_train):
    """q ∝ p·π_new/π_train（π_train=0 的类 q=0）；行和为 0 时保留原行。返回 (q, 回退行数)。"""
    ratio = np.zeros(C)
    m = pi_train > 0
    ratio[m] = pi_new[m] / pi_train[m]
    Q = P * ratio[None, :]
    s = Q.sum(1)
    bad = ~(s > 0) | ~np.isfinite(s)
    Q[~bad] /= s[~bad, None]
    Q[bad] = P[bad]
    return Q, int(bad.sum())


def my_sel_hash(sel):
    return hashlib.md5(np.sort(np.asarray(sel, dtype=np.int64)).tobytes()).hexdigest()[:16]


def my_pool_positions(n_cand, split, rep):
    rng = np.random.default_rng(10_000 * split + 100 * rep + 7)
    return np.sort(rng.choice(n_cand, min(N_POOL, n_cand), replace=False))


# ---------------------------------------------------------------- 曲线与 FA
def curve_of(Z, sc):
    """与 step0B2 相同的计算路径：π̂ = softmax(Z) 的测试集平均；τ 族 argmax；OA 与 ba_ge30。"""
    Pa = np.exp(Z - Z.max(1, keepdims=True))
    Pa /= Pa.sum(1, keepdims=True)
    pihat = Pa.mean(0)
    PH = B2.tau_preds_grid(Z, pihat, TAUS)
    return sc.curve(PH)


def my_envelope(oa, ba, grid, eps=1e-9):
    """独立的上包络实现（对账用）：逐段线性插值；OA 相同的平段取两端 BA 较大者；多段经过取最大。"""
    best = np.full(len(grid), -np.inf)
    for j in range(len(oa) - 1):
        o0, o1, b0, b1 = oa[j], oa[j + 1], ba[j], ba[j + 1]
        lo, hi = min(o0, o1) - eps, max(o0, o1) + eps
        ins = (grid >= lo) & (grid <= hi)
        if not ins.any():
            continue
        if abs(o1 - o0) < eps:
            v = np.full(ins.sum(), max(b0, b1))
        else:
            lam = np.clip((grid[ins] - o0) / (o1 - o0), 0.0, 1.0)
            v = b0 + lam * (b1 - b0)
        best[ins] = np.maximum(best[ins], v)
    best[~np.isfinite(best)] = np.nan
    return best


def fa_row(cv, anchor, grid_idx=None):
    oa, ba = cv
    if grid_idx is not None:
        oa, ba = oa[grid_idx], ba[grid_idx]
    grid = anchor + OFFS
    best, *_ = B.envelope([(oa, ba)], grid)
    fa, cov = B.fa_of(best)
    b2 = my_envelope(oa, ba, grid)
    cov2 = float(np.isfinite(b2).mean())
    fa2 = float(np.nanmean(b2)) if cov2 >= MIN_COV else np.nan
    return dict(FA=fa, reach=cov, partial=bool(MIN_COV <= cov < 1.0), fa_missing=bool(cov < MIN_COV),
                FA_own=fa2, reach_own=cov2,
                env_maxdiff=float(np.nanmax(np.abs(best - b2))) if np.isfinite(best).any() else 0.0,
                env_nan_same=bool(np.array_equal(np.isnan(best), np.isnan(b2))))


# ---------------------------------------------------------------- 数据
def load_rows():
    GR = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(R_G, 'rows_t1_*.csv'))], ignore_index=True)
    GR = GR[(GR.cond == 'none') & GR.method.isin(['srs', 'foreign_only'])]
    MR = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(R_ML, 'rows_*_*.csv'))], ignore_index=True)
    MR = MR[MR.learner.isin(['lr', 'mlp']) & (MR.cond == 'none') &
            (((MR.method == 'srs') & (MR.k == 200)) | (MR.method == 'foreign_only'))]
    assert (MR.status == 'ok').all()
    return GR, MR


def norm64(P):
    P = np.asarray(P, dtype=np.float64)
    return P / P.sum(1, keepdims=True)


def one(df, **kw):
    m = np.ones(len(df), bool)
    for a, b in kw.items():
        m &= (df[a] == b).to_numpy()
    x = df[m]
    assert len(x) == 1, (kw, len(x))
    return x.iloc[0]


# ---------------------------------------------------------------- 一个单元
def run_unit(D, S, GR, MR, tgt, split, rep):
    y = np.asarray(D.y)
    unit = f'{tgt}_s{split}_r{rep}'
    n_cand = len(S['cand'])
    allT, test, fr = S['allT'], S['test'], S['foreign']
    assert np.array_equal(allT[n_cand:], test)
    yT = y[test]
    big = np.flatnonzero(np.bincount(yT, minlength=C) >= G.MIN_N)
    sc = B.Scorer(yT, big)
    yF = y[fr]
    cw = my_class_weights(yF)
    assert np.array_equal(cw, G.class_weights_from(yF))
    pi_tr0 = my_train_prior(yF, np.array([], int), cw, 0.0)
    pool_pos = my_pool_positions(n_cand, split, rep)
    assert np.array_equal(pool_pos, G.pool_positions(S, split, rep))
    idx_pool = allT[pool_pos]
    pi_pool = np.bincount(y[idx_pool], minlength=C) / len(idx_pool)

    # 同一批 SRS 标签（G 的选样文件；与特征、学习器无关）
    sels = {}
    for k in KS:
        z = np.load(os.path.join(R_G, 'sel', f'{unit}_none_srs_k{k}.npz'))
        sel = np.asarray(z['sel'])
        assert np.array_equal(sel, idx_pool[z['sel_pos']])
        assert not np.intersect1d(sel, test).size and not np.intersect1d(sel, fr).size
        ys = y[sel]
        sels[k] = dict(sel=sel, h=my_sel_hash(sel), ys=ys, p_hat=my_laplace(ys),
                       pi_trk=my_train_prior(yF, ys, cw, R_MULT))

    curve_rows, fa_rows, chk = [], [], []
    for f in FEATS:
        base = dict(target=tgt, split=split, rep=rep, unit=unit, clf=f)
        # ================= LightGBM（L1）
        cell = dict(base, learner='lgb')
        PT = norm64(S[f'P_T_{f}'][n_cand:])
        Z = {'Z0': B.log_or_neginf(PT)}
        extra = {}
        for k in KS:
            s_ = sels[k]
            g = one(GR, method='srs', target=tgt, split=split, rep=rep, clf=f, k=k)
            assert g.sel_hash == s_['h'], (unit, f, k)
            Pr = norm64(np.load(os.path.join(R_G, 'probs', f'{unit}_{f}_k{k}_{g.sel_hash}.npz'))['i300'])
            Qpk, fb1 = my_reweight(PT, s_['p_hat'], pi_tr0)
            Qrt, fb2 = my_reweight(Pr, s_['p_hat'], s_['pi_trk'])
            Z[f'PK{k}'] = B.log_or_neginf(Qpk)
            Z[f'RT{k}'] = B.log_or_neginf(Pr)
            Z[f'RTPK{k}'] = B.log_or_neginf(Qrt)
            extra[f'RTPK{k}'] = dict(fallback=fb2, L1_trk_tr0=float(np.abs(s_['pi_trk'] - pi_tr0).sum()),
                                     L1_phat_trk=float(np.abs(s_['p_hat'] - s_['pi_trk']).sum()))
            extra[f'PK{k}'] = dict(fallback=fb1, L1_phat_trk=float(np.abs(s_['p_hat'] - pi_tr0).sum()))
            # 对账：训练先验 → G rows 的 P2（prior_correct(P, π_pool, pi_train_of(sel, R_MULT))）
            o2, b2 = G._scores(yT, my_reweight(Pr, pi_pool, s_['pi_trk'])[0].argmax(1), big)[:2]
            chk.append(dict(cell, k=k, what='P2_trainprior_RT', mine_oa=o2, ref_oa=g.oa_P2, mine_ba=b2, ref_ba=g.ba_P2))
            # 对账：本流重加权 / Laplace 与 check3_priors 的逐位比较
            q_ref, _ = CP.reweight(Pr, CP.laplace_prior(s_['ys'], C, alpha=1), s_['pi_trk'])
            chk.append(dict(cell, k=k, what='reweight_vs_check3', maxdiff=float(np.abs(q_ref - Qrt).max()),
                            phat_diff=float(np.abs(CP.laplace_prior(s_['ys'], C, alpha=1) - s_['p_hat']).max())))
        g0 = one(GR, method='foreign_only', target=tgt, split=split, rep=rep, clf=f, k=0)
        P0i = norm64(S[f'P0test_{f}_i300'])
        o2, b2 = G._scores(yT, my_reweight(P0i, pi_pool, pi_tr0)[0].argmax(1), big)[:2]
        chk.append(dict(cell, k=0, what='P2_trainprior_k0', mine_oa=o2, ref_oa=g0.oa_P2, mine_ba=b2, ref_ba=g0.ba_P2))
        CV = {a: curve_of(Za, sc) for a, Za in Z.items()}
        anchor = CV['Z0'][0][I_T0]
        for a, (oa, ba) in CV.items():
            for i, t in enumerate(TAUS):
                curve_rows.append(dict(cell, arm=a, tau=t, oa=oa[i], ba_ge30=ba[i]))
            fa_rows.append(dict(cell, arm=a, anchor=anchor, grid='new', oa_t0=oa[I_T0], ba_t0=ba[I_T0],
                                **fa_row(CV[a], anchor), **extra.get(a, {})))

        # ================= LR、MLP（L2）
        s_ = sels[200]
        for lrn in ('lr', 'mlp'):
            cell = dict(base, learner=lrn)
            m2 = one(MR, learner=lrn, method='srs', target=tgt, split=split, rep=rep, clf=f, k=200)
            m0 = one(MR, learner=lrn, method='foreign_only', target=tgt, split=split, rep=rep, clf=f, k=0)
            assert m2.sel_hash == s_['h'], (unit, f, lrn)
            P0 = norm64(np.load(os.path.join(R_ML, 'probs', lrn, f'{tgt}_s{split}_{f}_k0.npz'))['P'])
            P2 = norm64(np.load(os.path.join(R_ML, 'probs', lrn, f'{unit}_{f}_k200_{m2.sel_hash}.npz'))['P'])
            assert len(P0) == len(yT) == len(P2)
            Qpk, fb1 = my_reweight(P0, s_['p_hat'], pi_tr0)
            Qrt, fb2 = my_reweight(P2, s_['p_hat'], s_['pi_trk'])
            Z = {'Z0': B.log_or_neginf(P0), 'PK200': B.log_or_neginf(Qpk), 'RT200': B.log_or_neginf(P2),
                 'RTPK200': B.log_or_neginf(Qrt)}
            extra = {'PK200': dict(fallback=fb1), 'RTPK200': dict(fallback=fb2)}
            # 对账：训练先验（k=0 与 k=200）经 P2 与 guidance_G_ml rows 比
            for kk, PP, pit, mrow in ((0, P0, pi_tr0, m0), (200, P2, s_['pi_trk'], m2)):
                o2, b2 = G._scores(yT, my_reweight(PP, pi_pool, pit)[0].argmax(1), big)[:2]
                chk.append(dict(cell, k=kk, what='P2_trainprior_ml', mine_oa=o2, ref_oa=mrow.oa_P2, mine_ba=b2,
                                ref_ba=mrow.ba_P2))
            CV = {a: curve_of(Za, sc) for a, Za in Z.items()}
            for kk, a, mrow in ((0, 'Z0', m0), (200, 'RT200', m2)):
                chk.append(dict(cell, k=kk, what='t0_vs_ml_rows', mine_oa=CV[a][0][I_T0], ref_oa=mrow.oa,
                                mine_ba=CV[a][1][I_T0], ref_ba=mrow.ba_ge30))
            anchor = CV['Z0'][0][I_T0]
            for a, (oa, ba) in CV.items():
                for i, t in enumerate(TAUS):
                    curve_rows.append(dict(cell, arm=a, tau=t, oa=oa[i], ba_ge30=ba[i]))
                fa_rows.append(dict(cell, arm=a, anchor=anchor, grid='new', oa_t0=oa[I_T0], ba_t0=ba[I_T0],
                                    **fa_row(CV[a], anchor), **extra.get(a, {})))
            # 对账：旧网格（−0.5..2）上 RT200 在本学习器锚点下的 FA ↔ partB fa_rows.csv（anchor=own_{lrn}）
            fa_rows.append(dict(cell, arm='RT200', anchor=anchor, grid='old',
                                **fa_row(CV['RT200'], anchor, IDX_OLD)))
    return curve_rows, fa_rows, chk


# ---------------------------------------------------------------- 汇总（本流自写）
def country_vals(U, col):
    """国家值 = 该国可用单元的平均（逐特征）；both = 两特征国家值的平均（任一缺失则缺失）。"""
    out = {f: U[U.clf == f].groupby('target')[col].mean().reindex(COUNTRIES) for f in FEATS}
    out['both'] = (out['ae'] + out['s2']) / 2.0
    return out


def tci(v):
    v = np.asarray(pd.Series(v).dropna(), dtype=float)
    n = len(v)
    if n < 2:
        return dict(n=n, mean=np.nan, lo=np.nan, hi=np.nan, npos=int((v > 0).sum()), sd=np.nan)
    m, sd = v.mean(), v.std(ddof=1)
    h = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
    return dict(n=n, mean=m, lo=m - h, hi=m + h, npos=int((v > 0).sum()), sd=sd)


def fmt(s):
    if s['n'] < 2:
        return f'n={s["n"]}（不足 2 国）'
    return f'{s["mean"]:+.2f} [{s["lo"]:+.2f}, {s["hi"]:+.2f}]，>0 {s["npos"]}/{s["n"]}'


def sub9(v, nine):
    return v.drop(DROP) if nine else v


def main():
    t0 = time.time()
    assert sha16(SPEC)[:16] == SPEC_HASH, '规格哈希不一致'
    D = G.GData()
    GR, MR = load_rows()
    CR, FR, KR = [], [], []
    for tgt in COUNTRIES:
        for split in (0, 1):
            S = np.load(os.path.join(R_G, 'stage0', f'{tgt}_s{split}.npz'))
            for rep in (0, 1):
                a, b, c = run_unit(D, S, GR, MR, tgt, split, rep)
                CR += a
                FR += b
                KR += c
                print(f'[{time.time() - t0:.0f}s] {tgt}_s{split}_r{rep}', flush=True)
    CRd, FRd, KRd = map(pd.DataFrame, (CR, FR, KR))
    CRd.to_csv(os.path.join(OUT, 'indep_curves.csv'), index=False)
    FRd.to_csv(os.path.join(OUT, 'indep_fa_rows.csv'), index=False)
    KRd.to_csv(os.path.join(OUT, 'indep_check_rows.csv'), index=False)
    lines, est = summarize(CRd, FRd, KRd)
    open(os.path.join(OUT, 'indep_summary.txt'), 'w').write('\n'.join(lines) + '\n')
    est.to_csv(os.path.join(OUT, 'indep_estimates.csv'), index=False)
    print('\n'.join(lines))
    print(f'总耗时 {time.time() - t0:.0f}s')


def summarize(CRd, FRd, KRd):
    L_ = []
    P = L_.append
    P('标签线独立复核（L1：LightGBM R*_k、FA(RTPK-k) − FA(RT-k)；L2：LR、MLP 的 A_200、R_200、R*_200、ΔFA(RT200)）'
      '—— 事后分析，C 级、描述性；本批数字不进摘要。')
    P(f'规格 PRECONF_ANALYSES_SPEC_2026-09-29.md sha256[:16]={sha16(SPEC)}（登记 {SPEC_HASH}）；'
      f'indep_label.py sha256[:16]={sha16(os.path.abspath(__file__))}；step0B.py {sha16(os.path.join(PARTB, "step0B.py"))}；'
      f'step0B2.py {sha16(os.path.join(PARTB2, "step0B2.py"))}（只 import，未改）。')
    P(f'τ 网格 {TAUS[0]}..{TAUS[-1]} 步长 0.05（{len(TAUS)} 点）；W4 = [锚点−4, 锚点] 步长 0.1（{len(OFFS)} 点）；'
      f'可达率 <{MIN_COV:.0%} 记缺失。L1 锚点 = LightGBM Z0（P_T 测试行）τ=0 的 OA；L2 锚点 = 本学习器 Z0（k=0）τ=0 的 OA。')
    P('国家值 = 该国可用单元（2 划分 × 2 重复）的平均；both = 两特征国家值平均（主）；95% t CI（df = n−1）；9 国 = 去 RO、PT（主）。')
    P('')

    # ======================== 对账
    P('== 对账 ==')
    ok_all = True
    x = KRd[KRd.what.str.startswith('P2_')]
    d = max((x.mine_oa - x.ref_oa).abs().max(), (x.mine_ba - x.ref_ba).abs().max())
    ok = d < 1e-9
    ok_all &= ok
    for w in ('P2_trainprior_k0', 'P2_trainprior_RT', 'P2_trainprior_ml'):
        xx = x[x.what == w]
        P(f'(1) 训练先验（本流逐类计数实现）→ prior_correct(P, π_pool, π_train) 的 OA/BA 与 rows 的 oa_P2/ba_P2 [{w}]：'
          f'{len(xx)} 行，最大 |Δ| OA {(xx.mine_oa - xx.ref_oa).abs().max():.2e}，BA {(xx.mine_ba - xx.ref_ba).abs().max():.2e}')
    P(f'    → {"通过" if ok else "不通过"}（即本流的 pi_train_of([],0) 与 pi_train_of(sel, R_MULT) 与 G / multilearner 的运行时先验一致）')
    x = KRd[KRd.what == 'reweight_vs_check3']
    ok = x.maxdiff.max() < 1e-12 and x.phat_diff.max() == 0
    ok_all &= ok
    P(f'(2) 本流 my_reweight / my_laplace 与 check3_priors.reweight / laplace_prior：{len(x)} 行，Q 最大差 {x.maxdiff.max():.2e}，'
      f'p̂ 最大差 {x.phat_diff.max():.2e} → {"通过" if ok else "不通过"}')
    x = KRd[KRd.what == 't0_vs_ml_rows']
    d = max((x.mine_oa - x.ref_oa).abs().max(), (x.mine_ba - x.ref_ba).abs().max())
    ok_all &= d < 1e-9
    P(f'(3) LR/MLP 的 Z0（k=0）与 RT200 在 τ=0 的 OA/BA 与 guidance_G_ml rows：{len(x)} 行，最大 |Δ| {d:.2e}')
    # 与 B2 曲线逐位比较
    CB = pd.read_csv(os.path.join(PARTB2, 'curves_B2.csv'), float_precision='round_trip')
    CB = CB[CB.arm.isin(['Z0'] + [f'PK{k}' for k in KS] + [f'RT{k}' for k in KS])]
    CB['ti'] = np.rint(CB.tau * 100).astype(int)
    CN = CRd[(CRd.learner == 'lgb') & CRd.arm.isin(CB.arm.unique())].copy()
    CN['ti'] = np.rint(CN.tau * 100).astype(int)
    J = CN.merge(CB, on=['unit', 'clf', 'arm', 'ti'], suffixes=('', '_B'), how='inner')
    same = np.array_equal(J.oa.to_numpy(), J.oa_B.to_numpy()) and np.array_equal(J.ba_ge30.to_numpy(),
                                                                                  J.ba_ge30_B.to_numpy())
    ok_all &= same and len(J) == len(CN) == len(CB)
    P(f'(4) LightGBM Z0、PK-k、RT-k 曲线与 partB2/curves_B2.csv：{len(J)}/{len(CB)} 行对上，OA/BA '
      f'{"逐位相同" if same else "不同"}（最大 |Δ| OA {(J.oa - J.oa_B).abs().max():.2e}，BA {(J.ba_ge30 - J.ba_ge30_B).abs().max():.2e}）')
    FB = pd.read_csv(os.path.join(PARTB2, 'fa_rows_B2.csv'), float_precision='round_trip')
    FB = FB[(FB.grid == 'new') & FB.arm.isin(CB.arm.unique())]
    FN = FRd[(FRd.learner == 'lgb') & (FRd.grid == 'new')]
    J = FN.merge(FB, on=['unit', 'clf', 'arm'], suffixes=('', '_B'), how='inner')
    same = (np.array_equal(J.FA.to_numpy(float), J.FA_B.to_numpy(float), equal_nan=True) and
            np.array_equal(J.reach.to_numpy(), J.reach_B.to_numpy()) and
            np.array_equal(J.anchor.to_numpy(), J.OA_star.to_numpy()))
    ok_all &= same and len(J) == len(FB)
    P(f'(5) LightGBM Z0、PK-k、RT-k 的锚点、FA、可达率与 partB2/fa_rows_B2.csv（grid=new）：{len(J)}/{len(FB)} 行，'
      f'{"逐位相同" if same else "不同"}')
    FBo = pd.read_csv(os.path.join(PARTB, 'fa_rows.csv'), float_precision='round_trip')
    rows = []
    for lrn in ('lr', 'mlp'):
        a = FRd[(FRd.learner == lrn) & (FRd.grid == 'old')]
        b = FBo[(FBo.anchor == f'own_{lrn}') & (FBo.arm == f'RT200{lrn}')]
        j = a.merge(b, on=['unit', 'clf'], suffixes=('', '_B'))
        rows.append((lrn, len(j), np.array_equal(j.FA.to_numpy(float), j.FA_B.to_numpy(float), equal_nan=True),
                     np.array_equal(j.anchor.to_numpy(), j.OA_star.to_numpy()),
                     float(np.nanmax(np.abs(j.FA - j.FA_B)))))
    ok_all &= all(r[2] and r[3] and r[1] == 88 for r in rows)
    P('(6) LR/MLP 的 RT200 在本学习器锚点、旧网格（−0.5..2）上的 FA 与 partB/fa_rows.csv（anchor=own_lr/own_mlp）：' +
      '；'.join(f'{r[0]} {r[1]} 行，锚点{"相同" if r[3] else "不同"}，FA {"逐位相同" if r[2] else "不同"}（最大 |Δ| {r[4]:.2e}）'
               for r in rows))
    x = FRd[FRd.grid == 'new']
    dfa = (x.FA - x.FA_own).abs().max()
    same_nan = (x.FA.isna() == x.FA_own.isna()).all() and x.env_nan_same.all()
    ok_all &= dfa < 1e-9 and same_nan
    P(f'(7) 独立包络实现 vs step0B.envelope：{len(x)} 行（学习器 × 单元 × 特征 × 臂），FA 最大 |Δ| {dfa:.2e}，'
      f'包络点最大 |Δ| {x.env_maxdiff.max():.2e}，可达 / 缺失模式{"一致" if same_nan else "不一致"}')
    nfb = int(FRd.fallback.fillna(0).sum())
    P(f'(8) 重加权回退行合计 {nfb}；曲线行数每 (学习器, 臂) '
      f'{sorted(set(CRd.groupby(["learner", "arm"]).size().tolist()))}（应 = 44 × 2 × {len(TAUS)} = {88 * len(TAUS)}）')
    P(f'对账结论：{"全部通过" if ok_all else "有不通过项，见上"}')
    P('')

    # ======================== 可达率
    FN = FRd[FRd.grid == 'new']
    P('######## 可达率（单元 × 特征；全可达 / 部分可达 / 缺失；缺失所在国家）########')
    for lrn in ('lgb', 'lr', 'mlp'):
        arms = (['Z0'] + [f'PK{k}' for k in KS] + [f'RT{k}' for k in KS] + [f'RTPK{k}' for k in KS]
                if lrn == 'lgb' else ['Z0', 'PK200', 'RT200', 'RTPK200'])
        for a in arms:
            for f in FEATS:
                x = FN[(FN.learner == lrn) & (FN.arm == a) & (FN.clf == f)]
                rel = x.oa_t0 - x.anchor
                P(f'  {lrn:3s} {a:8s} {f}  {int((x.reach >= 1).sum()):2d} / {int(x.partial.sum()):2d} / '
                  f'{int(x.fa_missing.sum()):2d}  可达率均值 {x.reach.mean():.3f}  τ=0 的 OA − 锚点 中位 {rel.median():+.2f} '
                  f'[P10 {rel.quantile(.1):+.2f}, P90 {rel.quantile(.9):+.2f}]  缺失国家 '
                  f'{",".join(sorted(x[x.fa_missing].target.unique())) or "—"}')
    P('')

    # ======================== 估计量
    W = FN.set_index(UKEY + ['arm'])['FA'].unstack('arm').reset_index()
    T0o = FN.set_index(UKEY + ['arm'])['oa_t0'].unstack('arm').reset_index()
    T0b = FN.set_index(UKEY + ['arm'])['ba_t0'].unstack('arm').reset_index()
    EST = []
    CVS = {}

    def est(name, lrn, a, b, Wt=None, show=True):
        Wt = W if Wt is None else Wt
        x = Wt[Wt.learner == lrn]
        U = x[UKEY].copy()
        U['d'] = x[a].to_numpy() - x[b].to_numpy()
        cv = country_vals(U, 'd')
        CVS[(lrn, name)] = cv
        nmiss = {f: int(U[U.clf == f].d.isna().sum()) for f in FEATS}
        if show:
            P(f'- {name}（{lrn}）')
        for f in ('both', 'ae', 's2'):
            for nine in (True, False):
                s = tci(sub9(cv[f], nine))
                EST.append(dict(learner=lrn, est=name, feat=f, set='9' if nine else '11', **s,
                                miss_units=nmiss.get(f, np.nan)))
                if show:
                    P(f'    {f:4s} {"9 国" if nine else "11 国":5s} {fmt(s)}' +
                      (f'   （缺失单元 {nmiss[f]}/44）' if f != 'both' and not nine else ''))
        return cv

    P('######## L1（LightGBM；锚点 = Z0 τ=0 的 OA；pp）########')
    for k in KS:
        est(f'R*_{k} = FA(RTPK{k}) − FA(PK{k})', 'lgb', f'RTPK{k}', f'PK{k}')
    for k in KS:
        est(f'FA(RTPK{k}) − FA(RT{k})', 'lgb', f'RTPK{k}', f'RT{k}')
    P('-- 参照（与 B2 同口径复算）--')
    est('A_200 = FA(PK200) − FA(Z0)', 'lgb', 'PK200', 'Z0')
    est('R_200 = FA(RT200) − FA(PK200)', 'lgb', 'RT200', 'PK200')
    est('ΔFA(RT200) = FA(RT200) − FA(Z0)', 'lgb', 'RT200', 'Z0')
    est('ΔFA(RTPK200) = FA(RTPK200) − FA(Z0)', 'lgb', 'RTPK200', 'Z0')
    P('-- 描述：τ = 0 的 RTPK200 − RT200（OA、BA）--')
    est('τ=0 OA：RTPK200 − RT200', 'lgb', 'RTPK200', 'RT200', Wt=T0o)
    est('τ=0 BA：RTPK200 − RT200', 'lgb', 'RTPK200', 'RT200', Wt=T0b)
    x = FN[(FN.learner == 'lgb') & FN.arm.isin([f'RTPK{k}' for k in KS])]
    P('-- 描述：重训模型训练先验相对外国池先验的偏移 ‖π_train(k) − π_train(0)‖₁ 与 ‖p̂ − π_train(k)‖₁（单元×特征均值）：' +
      '；'.join(f'k={k}: {x[x.arm == f"RTPK{k}"].L1_trk_tr0.mean():.3f} / {x[x.arm == f"RTPK{k}"].L1_phat_trk.mean():.3f}'
               for k in KS) + f'；参照 ‖p̂ − π_train(0)‖₁（PK200）{FN[(FN.learner == "lgb") & (FN.arm == "PK200")].L1_phat_trk.mean():.3f}')
    P('')
    P('######## L2（LR、MLP，k=200；锚点 = 本学习器 Z0 τ=0 的 OA；pp）########')
    for lrn in ('lr', 'mlp'):
        est('A_200 = FA(PK200) − FA(Z0)', lrn, 'PK200', 'Z0')
        est('R_200 = FA(RT200) − FA(PK200)', lrn, 'RT200', 'PK200')
        est('R*_200 = FA(RTPK200) − FA(PK200)', lrn, 'RTPK200', 'PK200')
        est('ΔFA(RT200) = FA(RT200) − FA(Z0)', lrn, 'RT200', 'Z0')
        est('FA(RTPK200) − FA(RT200)', lrn, 'RTPK200', 'RT200')
    P('')
    # 恒等式检查
    for lrn in ('lgb', 'lr', 'mlp'):
        x = W[W.learner == lrn]
        i1 = ((x.PK200 - x.Z0) + (x.RT200 - x.PK200) - (x.RT200 - x.Z0)).abs().max()
        i2 = ((x.RTPK200 - x.PK200) - (x.RT200 - x.PK200) - (x.RTPK200 - x.RT200)).abs().max()
        P(f'恒等式（单元层）{lrn}：A + R − ΔFA 最大 |·| {i1:.1e}；R* − R − (FA(RTPK) − FA(RT)) 最大 |·| {i2:.1e}')
    P('')

    # ======================== 国家值
    P('######## 国家值（both；pp）########')
    cols = [('lgb', 'R*_200 = FA(RTPK200) − FA(PK200)', 'lgb R*200'), ('lgb', 'FA(RTPK200) − FA(RT200)', 'lgb RTPK−RT'),
            ('lgb', 'R_200 = FA(RT200) − FA(PK200)', 'lgb R200'),
            ('lr', 'A_200 = FA(PK200) − FA(Z0)', 'lr A200'), ('lr', 'R_200 = FA(RT200) − FA(PK200)', 'lr R200'),
            ('lr', 'R*_200 = FA(RTPK200) − FA(PK200)', 'lr R*200'),
            ('mlp', 'A_200 = FA(PK200) − FA(Z0)', 'mlp A200'), ('mlp', 'R_200 = FA(RT200) − FA(PK200)', 'mlp R200'),
            ('mlp', 'R*_200 = FA(RTPK200) − FA(PK200)', 'mlp R*200')]
    P('  国家 ' + ' '.join(f'{c[2]:>12s}' for c in cols))
    for t in COUNTRIES:
        P(f'  {t}   ' + ' '.join(f'{CVS[(c[0], c[1])]["both"].get(t, np.nan):+12.2f}' for c in cols))
    P('')

    # ======================== 读法（L1，k=200，9 国，both）
    E = pd.DataFrame(EST)
    r = E[(E.learner == 'lgb') & (E.est == 'R*_200 = FA(RTPK200) − FA(PK200)') & (E.feat == 'both') & (E.set == '9')].iloc[0]
    P('######## 读法（规格 L1 事先写定：k = 200、9 国、both）########')
    P(f'  R*_200 = {r["mean"]:+.2f} [{r.lo:+.2f}, {r.hi:+.2f}]，>0 {r.npos}/{r.n}')
    if r.lo > 0:
        P('  → R*_200 的 95% CI 下界 > 0：重训在比例之外有增益（C 级、描述）。')
    else:
        P('  → R*_200 的 95% CI 下界 ≤ 0：未检出重训在比例之外的增益。' +
          ('（CI 上界 < 0：两边都用比例时，重训反而低于只做先验修正；读法未单列，按字面归入"未检出"。）' if r.hi < 0 else ''))
    return L_, E


if __name__ == '__main__':
    main()
