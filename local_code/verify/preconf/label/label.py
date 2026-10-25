"""
确认运行登记前的不训练分析 —— 标签线补充 L1–L5 与 AE7（界值与功效）。
规格：label_expansion/PRECONF_ANALYSES_SPEC_2026-09-29.md（sha256[:16]=d0906bd1417839d2，登记于 plan_hashes.txt，未改动）。
事后分析，C 级、描述性；只用已复用的 G s0/s1 测试集；不训练任何作物分类器，只读已存概率、G 选样与 stage0。
候选标签只用于估比例（p̂）、拟合 BCTS 头（HD-k）与 L4 选 τ；测试标签只用于评估，不用来选择任何参数。

口径复用（import 第 0 步补遗 B2 的 step0B2 与第 0 步 B 的 step0B，均不改其文件）：
  τ 族      step0B2.tau_preds_grid（step0B.tau_preds 源码），τ ∈ [−2, 2] 步长 0.05（81 点，step0B2.TAUS_NEW）
  π̂        该臂测试集平均预测概率（softmax(Z) 的列均值，不用标签）
  log       step0B.log_or_neginf（精确 log，概率恰为 0 → −inf）
  OA / BA   step0B.Scorer（ba_ge30 的类集合 = 测试集中 ≥ G.MIN_N=30 个地块的类）
  包络 / FA step0B.envelope、step0B.fa_of（可达率 <100% 为部分可达，<50% FA 记缺失）
  BCTS 头   step0B.fit_bcts / step0B.weights（λ=0.1，类均衡权重；与 B2 的 HD-k 相同）
  汇总      step0B.country_vals、step0B.t_summary、step0B.f95（国家值 = 该国可用单元平均；both = 两特征国家值平均）
窗口：W_w = [锚点 − w, 锚点]，步长 0.1；主窗口 w = 4（与 step0B.OFFS 逐位相同），L3 另取 w = 2、8。

臂（每单元 × 特征）：
  LightGBM  Z0 = P_T 测试行；PK-k = Z0 经 reweight(P, p̂_k, π_train_F)；RT-k = G SRS(k) 重训 i300；HD-k = 同一批选样上的 BCTS 头；
            RTPK-k（L1 新增）= RT-k 经 reweight(P_RT, p̂_k, π_train_RT)，π_train_RT = guidance_G.run_unit 中的
            pi_train_of(sel, R_MULT)（外国池类权重 cw + R_MULT·cw[标签]）。p̂_k = laplace_prior(同一批 k 个标签, 10, α=1)。
  LR / MLP（L2）以及 RF（方案外，只作描述与 AE7 附表）：
            Z0 = 本学习器 k=0 的已存概率；PK200 = Z0 经 reweight(P, p̂_200, π_train_F)（π_train_F 同 LightGBM 外国池先验）；
            RT200 = 本学习器 SRS k=200 的已存概率（按 rows_{learner}_*.csv 的 sel_hash 取文件）；RTPK200 同 L1。
  FA 锚点：LightGBM = Z0 在 τ=0 的 OA（OA★）；LR/MLP/RF = 本学习器 Z0 在 τ=0 的 OA。
估计量：
  L1  R*_k = FA(RTPK-k) − FA(PK-k)；FA(RTPK-k) − FA(RT-k)；可达率；k ∈ {50, 200, 400}。读法（k=200，9 国，both）。
  L2  LR、MLP：A_200 = FA(PK200) − FA(Z0)；R_200 = FA(RT200) − FA(PK200)；R*_200 = FA(RTPK200) − FA(PK200)；
      ΔFA(RT200) = FA(RT200) − FA(Z0)。
  L3  w ∈ {2, 4, 8}：ΔFA(RT200)、A_200、R_200、R*_200（LightGBM）；AE1 型 D（Z0、RT200）。
      D_arm = 窗口 [min(OA★_ae, OA★_s2) − w, min(...)] 内 BA_ae(o) − BA_s2(o) 的平均（两特征各用自己的该臂单曲线；
      共同可达 <50% 记缺失）。
  L4  在 k 个 SRS 标签上算各 τ 的 BA_k、OA_k，取 OA_k(τ) ≥ OA_k(0) − 2 条件下 BA_k 最大的 τ̂（Z0、PK-k、HD-k；
      PK、HD 为样本内）；RT-k 只报固定 τ ∈ {0, 0.25, 0.5}；报测试集 BA、OA 与相对 Z0（同一规则）的差。
  L5  model_diag 中是否有 tgt300 测试概率：核查；不可得时只报已有的 τ=0 差距。
  AE7 界值：B2 的 A_200、R_200、ΔFA(RT200) 与 L1 的 R*_200 的 AE−S2 交互量（国家值 ae − s2），90% t CI 与 ±1 pp TOST p；
      功效：只用 rep0 单元的国家层 SD，确认运行 22 单元（每国 2 单元），H_R1（同 OA 下 AE−S2 > 0，单侧 α=0.025）
      与 H_R2（交互量 ±1 pp TOST，α=0.05）的功效 / MDE80，9 国与 11 国。

用法：/Users/perng_mini/AlphaCrops_local/.venv/bin/python -B label.py --smoke   # AT_s0_r0、RO_s1_r1，输出到 smoke/
      /Users/perng_mini/AlphaCrops_local/.venv/bin/python -B label.py           # 全量 44 个单元
"""
import os
import sys
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_v] = '2'
sys.dont_write_bytecode = True  # 不在项目目录、partB/、partB2/ 写 __pycache__
import glob
import hashlib
import json
import time

import numpy as np
import pandas as pd
from scipy import integrate, optimize, stats

HERE = os.path.dirname(os.path.abspath(__file__))
PARTB2 = '/Users/perng_mini/AlphaCrops_local/verify/step0/partB2'
sys.path.insert(0, PARTB2)
import step0B2 as B2  # noqa: E402  （它再 import step0B：chdir 到项目目录（只读）、导入 guidance_G、取出 fit_bcts / weights）
from check3_priors import laplace_prior, reweight  # noqa: E402

B = B2.B
G = B.G
SPEC = os.path.join(B.L, 'PRECONF_ANALYSES_SPEC_2026-09-29.md')
SPEC_HASH = 'd0906bd1417839d2'
STEP0B2_PY = os.path.join(PARTB2, 'step0B2.py')
STEP0B_PY = B2.STEP0B_PY
CURVES_B2 = os.path.join(PARTB2, 'curves_B2.csv')
FA_B2 = os.path.join(PARTB2, 'fa_rows_B2.csv')
MATCHED = ('/Users/perng_mini/AlphaCrops_local/verify/storyline/ae_s2_integration_20260929/ae_s2_evidence/'
           'matched_oa_units.csv')
MODEL_DIAG = '/Users/perng_mini/AlphaCrops_local/verify/model_diag'
TGT_JSON = os.path.join(MODEL_DIAG, 'fit', 'jobs', 'tgt_{t}_0_{f}_k0_L63_R500.json')
UB_CSV = os.path.join(MODEL_DIAG, 'fit', 'table_upper_bound.csv')

C = B.C
FEATS = B.FEATS
COUNTRIES = B.COUNTRIES
DROP = B.DROP
KS = B.KS
MIN_COV = B.MIN_COV
TAUS = B2.TAUS_NEW
I_T0 = B2.I_T0
I_FIX = {t: int(np.flatnonzero(np.abs(TAUS - t) < 1e-9)[0]) for t in (0.0, 0.25, 0.5)}
WIDTHS = [2, 4, 8]
OFFS_W = {w: np.round(np.arange(-float(w), 0.0 + 1e-9, 0.1), 1) for w in WIDTHS}
assert np.array_equal(OFFS_W[4], B.OFFS)
ML = ['lr', 'mlp', 'rf']          # L2 只含 lr、mlp；rf 为方案外描述与 AE7 附表
L2_LRN = ['lr', 'mlp']
PK = [f'PK{k}' for k in KS]
RT = [f'RT{k}' for k in KS]
HD = [f'HD{k}' for k in KS]
RTPK = [f'RTPK{k}' for k in KS]
LGB_ARMS = ['Z0'] + PK + RT + HD + RTPK
ML_ARMS = ['Z0', 'PK200', 'RT200', 'RTPK200']
D_ARMS = {'lgb': ['Z0', 'PK200', 'RT200', 'HD200'], **{l_: ['Z0', 'RT200'] for l_ in ML}}
UKEY = ['target', 'split', 'rep', 'unit', 'clf']
OA_SLACK = 2.0   # L4：OA_k(τ) ≥ OA_k(0) − 2
TOL = 1e-9       # L4 比较的浮点容差


def sha16(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()[:16]


def softmax_mean(Z):
    P = np.exp(Z - Z.max(1, keepdims=True))
    P /= P.sum(1, keepdims=True)
    return P.mean(0)


def lab_scores(PH, yl):
    """L4：k 个标签上的 OA_k、BA_k（BA 的类集合 = 标签中出现的类，逐类召回平均）。PH：(nτ, k)。"""
    corr = PH == yl[None, :]
    oa = 100.0 * corr.mean(1)
    cls = np.unique(yl)
    ba = 100.0 * np.stack([corr[:, yl == c].mean(1) for c in cls], 1).mean(1)
    return oa, ba, len(cls)


def pick_tau(oa_k, ba_k):
    """事先写定的规则：OA_k(τ) ≥ OA_k(0) − 2 的 τ 中取 BA_k 最大者；并列取 |τ| 最小者，再并列取较小的 τ。"""
    feas = oa_k >= oa_k[I_T0] - OA_SLACK - TOL
    cand = np.flatnonzero(feas)
    bmax = ba_k[cand].max()
    top = cand[ba_k[cand] >= bmax - 1e-12]
    order = np.lexsort((TAUS[top], np.abs(TAUS[top])))
    return int(top[order[0]]), int(len(top))


def fa_width(curves, oa_star, w):
    grid = oa_star + OFFS_W[w]
    best, *_ = B.envelope(curves, grid)
    fa, cov = B.fa_of(best)
    return fa, cov


def d_width(cv_ae, cv_s2, anchor, w, grid=None):
    """AE1 型：两特征各用自己的单曲线，在 [anchor − w, anchor] 上逐点 BA_ae − BA_s2 平均；共同可达 <50% 记缺失。
    grid 缺省 = anchor + 偏移（与 step0B / B2 的 wgrid = OA★ + OFFS 相同）；只在对账时传入整合评估脚本的网格。"""
    if grid is None:
        grid = anchor + OFFS_W[w]
    ba_a = B.envelope([cv_ae], grid)[0]
    ba_s = B.envelope([cv_s2], grid)[0]
    ok = np.isfinite(ba_a) & np.isfinite(ba_s)
    cov = float(ok.mean())
    if cov < MIN_COV:
        return np.nan, cov, np.nan, np.nan
    dd = ba_a[ok] - ba_s[ok]
    return float(dd.mean()), cov, bool((dd >= 0).all()), bool((dd <= 0).all())


# ---------------------------------------------------------------- 数据
def load_ML_rows():
    MR = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(B.R_ML, 'rows_*_*.csv'))], ignore_index=True)
    MR = MR[MR.learner.isin(ML) & (MR.cond == 'none') &
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
    big = np.flatnonzero(np.bincount(yT, minlength=C) >= G.MIN_N)
    sc = B.Scorer(yT, big)
    yF = y[fr]
    cw = G.class_weights_from(yF)
    pi_train = np.bincount(yF, weights=cw[yF], minlength=C) / cw[yF].sum()   # = G pi_train_of([], 0)
    pool_pos = G.pool_positions(S, split, rep)
    yP = y[allT[pool_pos]]

    def pi_train_of(sel):
        """与 guidance_G.run_unit 内 pi_train_of(sel, R_MULT) 相同的公式（重训模型的实际训练先验）。"""
        ys_ = y[sel]
        w_ = np.r_[cw[yF], G.R_MULT * cw[ys_]]
        return np.bincount(np.r_[yF, ys_], weights=w_, minlength=C) / w_.sum()

    sels = {}
    for k in KS:
        zsel = np.load(os.path.join(B.R_G, 'sel', f'{unit}_none_srs_k{k}.npz'))
        sp, sel = zsel['sel_pos'], zsel['sel']
        assert np.array_equal(allT[pool_pos][sp], sel), (unit, k)
        sels[k] = (sp, sel, laplace_prior(yP[sp], C, alpha=1), pi_train_of(sel))

    curve_rows, fa_rows, chk_rows, l4_rows, meta_rows = [], [], [], [], []
    CV = {}
    for f in FEATS:
        cell = dict(target=tgt, split=split, rep=rep, unit=unit, clf=f)
        P_all = S[f'P_T_{f}']
        PT = P_all[n_cand:].astype(np.float64)
        PP = P_all[pool_pos].astype(np.float64)
        P0 = B.norm64(PT)
        Z = {'Z0': B.log_or_neginf(P0)}
        ZTc = np.log(np.clip(PT, 1e-12, None))
        ZPc = np.log(np.clip(PP, 1e-12, None))
        LAB = {}
        for k in KS:
            g = GR[(GR.method == 'srs') & (GR.target == tgt) & (GR.split == split) & (GR.rep == rep) &
                   (GR.clf == f) & (GR.k == k)]
            assert len(g) == 1, (unit, f, k, len(g))
            g = g.iloc[0]
            sp, sel, p_hat, pit_rt = sels[k]
            assert G.sel_hash(sel) == g.sel_hash, (unit, f, k)
            ys = yP[sp]
            Pr = B.norm64(np.load(os.path.join(B.R_G, 'probs', f'{unit}_{f}_k{k}_{g.sel_hash}.npz'))['i300'])
            Z[f'RT{k}'] = B.log_or_neginf(Pr)
            s, b = B.fit_bcts(ZPc[sp], ys, B.weights(ys, np.ones(len(ys)), 'bal'), B.LAM)
            Z[f'HD{k}'] = s * ZTc + b
            Qp, nfb_p = reweight(P0, p_hat, pi_train)
            Z[f'PK{k}'] = B.log_or_neginf(Qp)
            Qr, nfb_r = reweight(Pr, p_hat, pit_rt)
            Z[f'RTPK{k}'] = B.log_or_neginf(Qr)
            meta_rows.append(dict(cell, learner='lgb', k=k, pk_fallback=nfb_p, rtpk_fallback=nfb_r,
                                  L1_phat_vs_pitrain_rt=float(np.abs(p_hat - pit_rt).sum()),
                                  L1_phat_vs_pitrain_F=float(np.abs(p_hat - pi_train).sum()),
                                  L1_pitrain_rt_vs_F=float(np.abs(pit_rt - pi_train).sum()),
                                  n_classes_sel=int(len(np.unique(ys))), hd_s=s, big=','.join(map(str, big))))
            # 对账用独立路径：全局索引取标签、手写 Laplace 与训练先验、guidance_G.prior_correct + argmax + _scores
            ys2 = y[sel]
            ph2 = (np.bincount(ys2, minlength=C) + 1.0) / (len(ys2) + C)
            w2 = np.concatenate([cw[yF], 50.0 * cw[ys2]])
            pit2 = np.bincount(np.concatenate([yF, ys2]), weights=w2, minlength=C) / w2.sum()
            chk_rows.append(dict(cell, learner='lgb', arm=f'RTPK{k}', ref='hand_prior_correct',
                                 ref_vals=G._scores(yT, G.prior_correct(Pr, ph2, pit2).argmax(1), big)[:2],
                                 prior_maxdiff=float(max(np.abs(ph2 - p_hat).max(), np.abs(pit2 - pit_rt).max()))))
            chk_rows.append(dict(cell, learner='lgb', arm=f'RT{k}', ref='G_rows', ref_vals=(g.oa, g.ba_ge30)))
            # L4：k 个标签行上的各臂 Z（P_T 候选行；PK 与 HD 为样本内）
            Psel = B.norm64(PP[sp])
            LAB[k] = dict(ys=ys, Z0=B.log_or_neginf(Psel),
                          PK=B.log_or_neginf(reweight(Psel, p_hat, pi_train)[0]), HD=s * ZPc[sp] + b)
        # ---- 曲线
        pih = {}
        for arm in LGB_ARMS:
            pih[arm] = softmax_mean(Z[arm])
            PH = B2.tau_preds_grid(Z[arm], pih[arm], TAUS)
            CV[('lgb', f, arm)] = sc.curve(PH)
        # ---- 其他学习器（LR、MLP；RF 方案外）
        for lrn in ML:
            m0 = MR[(MR.learner == lrn) & (MR.method == 'foreign_only') & (MR.target == tgt) & (MR.split == split) &
                    (MR.rep == rep) & (MR.clf == f)]
            m2 = MR[(MR.learner == lrn) & (MR.method == 'srs') & (MR.target == tgt) & (MR.split == split) &
                    (MR.rep == rep) & (MR.clf == f)]
            assert len(m0) == 1 and len(m2) == 1, (lrn, unit, f)
            m0, m2 = m0.iloc[0], m2.iloc[0]
            sp, sel, p_hat, pit_rt = sels[200]
            assert m2.sel_hash == G.sel_hash(sel), (lrn, unit, f)
            Pk0 = B.norm64(np.load(os.path.join(B.R_ML, 'probs', lrn, f'{tgt}_s{split}_{f}_k0.npz'))['P'])
            P2 = B.norm64(np.load(os.path.join(B.R_ML, 'probs', lrn, f'{unit}_{f}_k200_{m2.sel_hash}.npz'))['P'])
            ZL = {'Z0': B.log_or_neginf(Pk0), 'RT200': B.log_or_neginf(P2)}
            Qp, nfb_p = reweight(Pk0, p_hat, pi_train)
            ZL['PK200'] = B.log_or_neginf(Qp)
            Qr, nfb_r = reweight(P2, p_hat, pit_rt)
            ZL['RTPK200'] = B.log_or_neginf(Qr)
            ties = {a: float(((P == P.max(1, keepdims=True)).sum(1) > 1).mean()) for a, P in (('Z0', Pk0), ('RT200', P2))}
            meta_rows.append(dict(cell, learner=lrn, k=200, pk_fallback=nfb_p, rtpk_fallback=nfb_r,
                                  tie_Z0=ties['Z0'], tie_RT200=ties['RT200'],
                                  zero_Z0=int((Pk0 == 0).sum()), zero_RT200=int((P2 == 0).sum())))
            for arm in ML_ARMS:
                PH = B2.tau_preds_grid(ZL[arm], softmax_mean(ZL[arm]), TAUS)
                CV[(lrn, f, arm)] = sc.curve(PH)
            chk_rows.append(dict(cell, learner=lrn, arm='Z0', ref='ML_rows', ref_vals=(m0.oa, m0.ba_ge30)))
            chk_rows.append(dict(cell, learner=lrn, arm='RT200', ref='ML_rows', ref_vals=(m2.oa, m2.ba_ge30)))
            chk_rows.append(dict(cell, learner=lrn, arm='PK200', ref='hand_prior_correct',
                                 ref_vals=G._scores(yT, G.prior_correct(Pk0, p_hat, pi_train).argmax(1), big)[:2],
                                 prior_maxdiff=0.0))
            chk_rows.append(dict(cell, learner=lrn, arm='RTPK200', ref='hand_prior_correct',
                                 ref_vals=G._scores(yT, G.prior_correct(P2, p_hat, pit_rt).argmax(1), big)[:2],
                                 prior_maxdiff=0.0))

        # ---- 曲线行与 FA 行（三种窗口宽度）
        for lrn in ['lgb'] + ML:
            arms = LGB_ARMS if lrn == 'lgb' else ML_ARMS
            oa_star = CV[(lrn, f, 'Z0')][0][I_T0]
            for arm in arms:
                oa, ba = CV[(lrn, f, arm)]
                curve_rows += [dict(cell, learner=lrn, arm=arm, tau=t, oa=oa[i], ba_ge30=ba[i]) for i, t in enumerate(TAUS)]
                for w in WIDTHS:
                    fa, cov = fa_width([(oa, ba)], oa_star, w)
                    fa_rows.append(dict(cell, learner=lrn, arm=arm, width=w, OA_star=oa_star, FA=fa, reach=cov,
                                        partial=bool(MIN_COV <= cov < 1.0), fa_missing=bool(cov < MIN_COV),
                                        oa_t0=oa[I_T0], ba_t0=ba[I_T0], oa_t05=oa[I_FIX[0.5]], ba_t05=ba[I_FIX[0.5]],
                                        oa_min=float(oa.min()), oa_max=float(oa.max())))
        # τ=0 对账记录补值
        for r in chk_rows:
            if r['unit'] == unit and r['clf'] == f and 'oa' not in r:
                cv = CV[(r['learner'], f, r['arm'])]
                r['oa'], r['ba_ge30'] = cv[0][I_T0], cv[1][I_T0]

        # ---- L4：k 个标签上选 τ̂（Z0、PK-k、HD-k）；RT-k 固定 τ
        for k in KS:
            lab = LAB[k]
            for arm, key, test_arm in (('Z0', 'Z0', 'Z0'), (f'PK{k}', 'PK', f'PK{k}'), (f'HD{k}', 'HD', f'HD{k}')):
                PHk = B2.tau_preds_grid(lab[key], pih[test_arm], TAUS)
                oa_k, ba_k, ncls = lab_scores(PHk, lab['ys'])
                j, nties = pick_tau(oa_k, ba_k)
                oa, ba = CV[('lgb', f, test_arm)]
                l4_rows.append(dict(cell, k=k, arm=arm, mode='rule', tau=float(TAUS[j]), n_ties=nties,
                                    n_cls_lab=ncls, oa_lab=oa_k[j], ba_lab=ba_k[j], oa_lab0=oa_k[I_T0], ba_lab0=ba_k[I_T0],
                                    oa_test=oa[j], ba_test=ba[j], oa_test0=oa[I_T0], ba_test0=ba[I_T0]))
            oa, ba = CV[('lgb', f, f'RT{k}')]
            for t, j in I_FIX.items():
                l4_rows.append(dict(cell, k=k, arm=f'RT{k}', mode=f'fixed{t:g}', tau=t, oa_test=oa[j], ba_test=ba[j],
                                    oa_test0=oa[I_T0], ba_test0=ba[I_T0]))
            # 同一固定 τ 下的 Z0（方案外对照）
            oa, ba = CV[('lgb', f, 'Z0')]
            for t, j in I_FIX.items():
                l4_rows.append(dict(cell, k=k, arm='Z0', mode=f'fixed{t:g}', tau=t, oa_test=oa[j], ba_test=ba[j],
                                    oa_test0=oa[I_T0], ba_test0=ba[I_T0]))

    # ---- AE1 型 D（两特征合一；锚点 = 本学习器 Z0 在 τ=0 的 OA 两特征较小者）
    d_rows = []
    for lrn, arms in D_ARMS.items():
        oa_ae, oa_s2 = CV[(lrn, 'ae', 'Z0')][0][I_T0], CV[(lrn, 's2', 'Z0')][0][I_T0]
        anchor = min(oa_ae, oa_s2)
        for arm in arms:
            for w in WIDTHS:
                d, cov, dom_ae, dom_s2 = d_width(CV[(lrn, 'ae', arm)], CV[(lrn, 's2', arm)], anchor, w)
                d_rows.append(dict(target=tgt, split=split, rep=rep, unit=unit, learner=lrn, arm=arm, width=w,
                                   anchor_kind='Z0min', anchor=anchor, oa_star_ae=oa_ae, oa_star_s2=oa_s2, D=d, cov=cov,
                                   ae_dominates=dom_ae, s2_dominates=dom_s2))
            if lrn == 'lgb' and arm != 'Z0':
                # 只用于对账：整合评估 ae_s2_matched_oa_arms.py 的锚点 = 该臂自身 τ=0 OA 的两特征较小者
                own = min(CV[(lrn, 'ae', arm)][0][I_T0], CV[(lrn, 's2', arm)][0][I_T0])
                d, cov, dom_ae, dom_s2 = d_width(CV[(lrn, 'ae', arm)], CV[(lrn, 's2', arm)], own, 4)
                g_r = np.round(np.arange(own - 4, own + 1e-9, 0.1), 6)   # 整合评估脚本的网格写法
                d_r = d_width(CV[(lrn, 'ae', arm)], CV[(lrn, 's2', arm)], own, 4, grid=g_r)[0]
                d_rows.append(dict(target=tgt, split=split, rep=rep, unit=unit, learner=lrn, arm=arm, width=4,
                                   anchor_kind='own_check', anchor=own, oa_star_ae=oa_ae, oa_star_s2=oa_s2, D=d, cov=cov,
                                   ae_dominates=dom_ae, s2_dominates=dom_s2, D_roundgrid=d_r))
    log(f'{unit} 完成')
    return curve_rows, fa_rows, chk_rows, l4_rows, meta_rows, d_rows


# ---------------------------------------------------------------- 主程序
def main():
    smoke = '--smoke' in sys.argv
    out = os.path.join(HERE, 'smoke') if smoke else HERE
    os.makedirs(out, exist_ok=True)
    assert sha16(SPEC) == SPEC_HASH, '规格文件哈希与登记值不一致'
    assert sha16(B2.ADDENDUM) == B2.ADD_HASH and sha16(B.SPEC) == B.SPEC_HASH
    t0 = time.time()
    print(f'label.py sha256[:16]={sha16(os.path.abspath(__file__))}', flush=True)
    D = G.GData()
    GR = B.load_G_rows()
    MR = load_ML_rows()
    units = ([('AT', 0, 0), ('RO', 1, 1)] if smoke else
             [(t, s, r) for t in COUNTRIES for s in (0, 1) for r in (0, 1)])
    acc = [[] for _ in range(6)]
    cache = {}
    for tgt, split, rep in units:
        if (tgt, split) not in cache:
            cache.clear()
            cache[(tgt, split)] = np.load(os.path.join(B.R_G, 'stage0', f'{tgt}_s{split}.npz'))
        res = run_unit(D, cache[(tgt, split)], GR, MR, tgt, split, rep,
                       log=lambda s: print(f'[{time.time() - t0:.0f}s] {s}', flush=True))
        for a, r in zip(acc, res):
            a += r
    CRd, FRd, KRd, L4d, MEd, DRall = map(pd.DataFrame, acc)
    DRd = DRall[DRall.anchor_kind == 'Z0min'].reset_index(drop=True)
    KRd['oa_ref'] = KRd.ref_vals.map(lambda v: float(v[0]))
    KRd['ba_ref'] = KRd.ref_vals.map(lambda v: float(v[1]))
    KRd = KRd.drop(columns='ref_vals')
    CRd.to_csv(os.path.join(out, 'label_curves.csv'), index=False)
    FRd.to_csv(os.path.join(out, 'label_rows.csv'), index=False)
    KRd.to_csv(os.path.join(out, 'label_check_rows.csv'), index=False)
    L4d.to_csv(os.path.join(out, 'label_l4_rows.csv'), index=False)
    MEd.to_csv(os.path.join(out, 'label_meta_rows.csv'), index=False)
    DRall.to_csv(os.path.join(out, 'label_d_rows.csv'), index=False)
    lines, extra = summarize(CRd, FRd, KRd, L4d, MEd, DRd, smoke, DRall)
    for name, df in extra.items():
        df.to_csv(os.path.join(out, f'label_{name}.csv'), index=False)
    open(os.path.join(out, 'label_summary.txt'), 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(f'总耗时 {time.time() - t0:.0f}s')


# ---------------------------------------------------------------- 统计工具
def sel9(v, drop):
    return v.drop([c for c in DROP if c in v.index]) if drop else v


def ci90(v):
    v = np.asarray(pd.Series(v).dropna(), dtype=float)
    n = len(v)
    if n < 2:
        return dict(n=n, mean=np.nan, sd=np.nan, lo90=np.nan, hi90=np.nan, p_tost=np.nan, n_pos=int((v > 0).sum()),
                    min_margin=np.nan)
    m, sd = v.mean(), v.std(ddof=1)
    se = sd / np.sqrt(n)
    q = stats.t.ppf(0.95, n - 1)
    p_lo = stats.t.sf((m + 1.0) / se, n - 1)     # H0: μ ≤ −1
    p_hi = stats.t.cdf((m - 1.0) / se, n - 1)    # H0: μ ≥ +1
    return dict(n=n, mean=m, sd=sd, lo90=m - q * se, hi90=m + q * se, p_tost=max(p_lo, p_hi),
                n_pos=int((v > 0).sum()), min_margin=max(abs(m - q * se), abs(m + q * se)))


def power_1s(delta, sd, n, alpha=0.025):
    """单样本 t，单侧 H1: μ > 0，α 单侧；非中心 t 精确功效。"""
    df = n - 1
    return float(stats.nct.sf(stats.t.ppf(1 - alpha, df), df, delta * np.sqrt(n) / sd))


def mde80_1s(sd, n, alpha=0.025, power=0.8):
    return float(optimize.brentq(lambda d: power_1s(d, sd, n, alpha) - power, 1e-9, 50 * sd))


def tost_power(mu, sd, n, margin=1.0, alpha=0.05):
    """单样本 TOST（两个单侧 t 检验各 α，即 90% CI 落在 ±margin 内）的精确功效：对 S 的分布数值积分。"""
    df = n - 1
    tc = stats.t.ppf(1 - alpha, df)
    se = sd / np.sqrt(n)
    qmax = df * (margin * np.sqrt(n) / (tc * sd)) ** 2   # S 超过此值时区间必然越界

    def f(q):
        half = margin - tc * sd * np.sqrt(q / df) / np.sqrt(n)
        if half <= 0:
            return 0.0
        return (stats.norm.cdf((half - mu) / se) - stats.norm.cdf((-half - mu) / se)) * stats.chi2.pdf(q, df)
    upper = min(qmax, stats.chi2.isf(1e-15, df))   # χ² 尾部质量 <1e-15 之外可忽略；避免积分区间过宽漏掉主质量
    return float(integrate.quad(f, 0, upper, limit=400, points=[df] if df < upper else None)[0])


def tost_margin80(sd, n, mu=0.0, alpha=0.05, power=0.8):
    return float(optimize.brentq(lambda m: tost_power(mu, sd, n, m, alpha) - power, 1e-6, 50 * sd + 10))


def tost_power_mc(mu, sd, n, margin=1.0, alpha=0.05, sims=200_000, seed=42):
    rng = np.random.default_rng(seed)
    x = rng.normal(mu, sd, (sims, n))
    m, se = x.mean(1), x.std(1, ddof=1) / np.sqrt(n)
    q = stats.t.ppf(1 - alpha, n - 1)
    return float(np.mean((m - q * se > -margin) & (m + q * se < margin)))


# ---------------------------------------------------------------- 汇总
def summarize(CRd, FRd, KRd, L4d, MEd, DRd, smoke, DRall=None):
    lines = []
    P = lines.append
    extra = {}
    nU = FRd.unit.nunique()
    P('确认运行登记前的不训练分析 —— 标签线补充 L1–L5 与 AE7（事后分析，C 级、描述性；G s0/s1 测试集已多次复用；'
      '本批数字不进摘要，只用于写确认运行的登记方案）。' + ('【冒烟：2 个单元】' if smoke else ''))
    P(f'规格 PRECONF_ANALYSES_SPEC_2026-09-29.md sha256[:16]={sha16(SPEC)}（登记 {SPEC_HASH}）；补遗 B2 {sha16(B2.ADDENDUM)}'
      f'（登记 {B2.ADD_HASH}）；step0B2.py {sha16(STEP0B2_PY)}、step0B.py {sha16(STEP0B_PY)}（只 import，未改）；'
      f'label.py {sha16(os.path.abspath(__file__))}；fit_bcts/weights 源码哈希 {B.D1_FUNC_HASH}。')
    P(f'单元 {nU} × 特征 2；τ ∈ [{TAUS[0]}, {TAUS[-1]}] 步长 0.05（{len(TAUS)} 点）；主窗口 W4 = [锚点−4, 锚点] 步长 0.1'
      f'（{len(OFFS_W[4])} 点），另 W2（{len(OFFS_W[2])} 点）、W8（{len(OFFS_W[8])} 点）；可达率 <{MIN_COV:.0%} 记 FA 缺失。')
    P('国家值 = 该国可用单元（2 划分 × 2 重复）的平均；both = AE、S2 两特征国家值的平均（主）；推断单位 = 国家，95% CI 为 t '
      '区间（df = n−1）；9 国 = 去掉 RO、PT；格式"均值 [95% CI]，>0 国家数/国家数"；单位 pp。')
    P('')

    # ================================================================ 对账
    P('== 对账 ==')
    ok_all = True
    # (1) LightGBM 曲线与 curves_B2.csv
    CB = pd.read_csv(CURVES_B2, float_precision='round_trip')
    arms_b2 = ['Z0'] + PK + RT + HD
    CB = CB[CB.arm.isin(arms_b2)].copy()
    CB['ti'] = np.rint(CB.tau * 100).astype(int)
    CN = CRd[(CRd.learner == 'lgb') & CRd.arm.isin(arms_b2)].copy()
    CN['ti'] = np.rint(CN.tau * 100).astype(int)
    J = CN.merge(CB, on=UKEY + ['arm', 'ti'], suffixes=('', '_B'), how='inner')
    same = (len(J) == len(CN) and np.array_equal(J.oa.to_numpy(), J.oa_B.to_numpy()) and
            np.array_equal(J.ba_ge30.to_numpy(), J.ba_ge30_B.to_numpy()))
    ok_all &= same
    P(f'(1) LightGBM Z0、PK-k、RT-k、HD-k 曲线与 partB2/curves_B2.csv：{len(J)}/{len(CN)} 行对上，OA、BA '
      f'{"逐位相同" if same else "不同"}（最大 |Δ| OA {(J.oa - J.oa_B).abs().max():.2e}，BA {(J.ba_ge30 - J.ba_ge30_B).abs().max():.2e}）')
    # (2) W4 FA 与 fa_rows_B2.csv（grid=new）
    FB = pd.read_csv(FA_B2, float_precision='round_trip')
    FB = FB[(FB.grid == 'new') & FB.arm.isin(arms_b2)]
    FN = FRd[(FRd.learner == 'lgb') & (FRd.width == 4) & FRd.arm.isin(arms_b2)]
    J = FN.merge(FB, on=UKEY + ['arm'], suffixes=('', '_B'), how='inner')
    res = []
    for col in ('OA_star', 'FA', 'reach'):
        a, b = J[col].to_numpy(float), J[f'{col}_B'].to_numpy(float)
        s_ = np.array_equal(a, b, equal_nan=True)
        ok_all &= s_
        res.append(f'{col} {"逐位相同" if s_ else "不同"}')
    ok_all &= len(J) == len(FN)
    P(f'(2) W4 下 LightGBM Z0、PK-k、RT-k、HD-k 的 FA 与 partB2/fa_rows_B2.csv（grid=new）：{len(J)}/{len(FN)} 行对上；'
      + '；'.join(res))
    # (3) RT-k τ=0 与 G rows；RTPK τ=0 与手算
    for no, lrn_set, ref, txt in (
            (3, ['lgb'], 'G_rows', 'RT-k τ=0 与 G rows（srs，i300）'),
            (4, ['lgb'], 'hand_prior_correct', 'RTPK-k τ=0 与手算（全局索引取标签、手写 (n_c+1)/(k+10) 与 pi_train_of(sel, 50)、'
                                               'guidance_G.prior_correct(P_RT, p̂, π_train_RT).argmax + _scores）'),
            (5, ML, 'ML_rows', 'LR/MLP/RF 的 Z0（k=0）与 RT200（SRS k=200）τ=0 与 guidance_G_ml rows'),
            (6, ML, 'hand_prior_correct', 'LR/MLP/RF 的 PK200、RTPK200 τ=0 与 guidance_G.prior_correct 路径')):
        x = KRd[KRd.learner.isin(lrn_set) & (KRd.ref == ref)]
        do, db = (x.oa - x.oa_ref).abs().max(), (x.ba_ge30 - x.ba_ref).abs().max()
        pm = x.prior_maxdiff.max() if 'prior_maxdiff' in x else 0.0
        ok = do < 1e-9 and db < 1e-9 and (not np.isfinite(pm) or pm < 1e-12)
        ok_all &= ok
        P(f'({no}) {txt}：{len(x)} 行，最大 |Δ| OA {do:.2e}，BA {db:.2e}' +
          (f'；先验最大差 {pm:.2e}' if no == 4 else '') + f' → {"通过" if ok else "不通过"}')
        if ref == 'ML_rows':
            for l_ in L2_LRN + ['rf']:
                xx = x[x.learner == l_]
                P(f'      {l_}: {len(xx)} 行，最大 |Δ| OA {(xx.oa - xx.oa_ref).abs().max():.2e}，BA {(xx.ba_ge30 - xx.ba_ref).abs().max():.2e}')
    # (7) D(Z0) 与 AE–S2 整合评估 matched_oa_units.csv
    if os.path.exists(MATCHED):
        MU = pd.read_csv(MATCHED).set_index('unit')
        dz = DRd[(DRd.learner == 'lgb') & (DRd.arm == 'Z0')]
        for w, col in ((4, 'W4_min'), (8, 'W8_min')):
            x = dz[dz.width == w].set_index('unit').join(MU[[f'{col}_d', f'{col}_cov', 'oa0_ae', 'oa0_s2']], how='inner')
            dd = (x.D - x[f'{col}_d']).abs().max()
            dc = (x['cov'] - x[f'{col}_cov']).abs().max()
            da = max((x.oa_star_ae - x.oa0_ae).abs().max(), (x.oa_star_s2 - x.oa0_s2).abs().max())
            ok = dd < 1e-6 and dc < 1e-9 and da < 1e-9
            ok_all &= ok
            P(f'(7) AE1 型 D(Z0, W{w}) 与 ae_s2_evidence/matched_oa_units.csv 的 {col}_d：{len(x)} 单元，最大 |Δ| D {dd:.2e}，'
              f'共同可达 {dc:.2e}，OA★ {da:.2e}（该脚本网格为 np.round(arange(锚点−w, 锚点, 0.1), 6)，与本实现 锚点+round(偏移,1) '
              f'在 1e-6 量级可能不同） → {"通过" if ok else "不通过"}')
            if w == 4:
                n_ae = int(x.ae_dominates.fillna(False).astype(bool).sum())
                n_s2 = int(x.s2_dominates.fillna(False).astype(bool).sum())
                P(f'      全窗口占优单元数：AE {n_ae}/{len(x)}（整合评估 {int(MU.W4_min_ae_dominates.reindex(x.index).sum())}），'
                  f'S2 {n_s2}/{len(x)}（整合评估 {int(MU.W4_min_s2_dominates.reindex(x.index).sum())}）')
    # (7b) 其他臂的 D 与整合评估 matched_oa_by_arm.csv（该脚本锚点 = 该臂自身 τ=0 OA 的较小者，与规格 AE1 的 Z0 锚点不同）
    MA = os.path.join(os.path.dirname(MATCHED), 'matched_oa_by_arm.csv')
    if os.path.exists(MA) and DRall is not None and not smoke:
        ma = pd.read_csv(MA)
        ma = ma[ma.quantity == 'd_matched'].set_index('arm')
        res = []
        for arm in ('RT200', 'HD200'):
            x = DRall[(DRall.learner == 'lgb') & (DRall.arm == arm) & (DRall.anchor_kind == 'own_check')]
            cvr = x.groupby('target').D_roundgrid.mean()
            m11, m9 = cvr.mean(), sel9(cvr, True).mean()
            dmax = max(abs(m11 - ma.loc[arm, 'mean_11']), abs(m9 - ma.loc[arm, 'mean_9']))
            ok = dmax < 1e-3
            ok_all &= ok
            cvn = x.groupby('target').D.mean()
            ndiff = int(((x.D - x.D_roundgrid).abs() > 1e-4).sum())
            xz = DRd[(DRd.learner == 'lgb') & (DRd.arm == arm) & (DRd.width == 4)].groupby('target').D.mean()
            res.append(f'{arm}: 自身锚点 + 其网格 9 国 {m9:+.4f} / 11 国 {m11:+.4f}（整合评估 {ma.loc[arm, "mean_9"]:+.4f} / '
                       f'{ma.loc[arm, "mean_11"]:+.4f}，{"一致" if ok else "不一致"}）；自身锚点 + B2 网格 9 国 '
                       f'{sel9(cvn, True).mean():+.4f}（两种网格差 >1e-4 的单元 {ndiff}）；规格 Z0 锚点 9 国 {sel9(xz, True).mean():+.4f}')
        P('(7b) LightGBM 其他臂的 D 与 ae_s2_evidence/matched_oa_by_arm.csv：该脚本锚点取该臂自身 τ=0 OA 的两特征较小者，'
          '网格为 np.round(arange(锚点−4, 锚点, 0.1), 6)；规格 AE1 取零标签 Z0 的 OA★ 较小者，网格沿用 B2 的 锚点 + 偏移。'
          '按其锚点与网格复算对账（4 位小数）；本批估计量用规格锚点与 B2 网格。网格写法只在锚点恰为曲线顶点、BA 在该点跳变时有影响。')
        P('      ' + '；'.join(res))
    # (8) L4 自洽：Z0 在标签行 τ=0 的预测 = P_T 候选行 argmax（间接：oa_lab0 与 rule 下 τ̂ 的可行性）
    r4 = L4d[L4d['mode'] == 'rule']
    feas_ok = bool((r4.oa_lab >= r4.oa_lab0 - OA_SLACK - TOL).all() and (r4.ba_lab >= r4.ba_lab0 - 1e-12).all())
    ok_all &= feas_ok
    P(f'(8) L4 规则自洽：{len(r4)} 行 τ̂ 均满足 OA_k(τ̂) ≥ OA_k(0) − 2，且 BA_k(τ̂) ≥ BA_k(0)（τ=0 本身可行）：'
      f'{"是" if feas_ok else "否"}；BA_k 并列（>1 个 τ 取到最大）的行 {int((r4.n_ties > 1).sum())}/{len(r4)}')
    # (9) 回退与零概率
    P(f'(9) reweight 回退行合计：PK {int(MEd.pk_fallback.sum())}，RTPK {int(MEd.rtpk_fallback.sum())}；'
      f'LR/MLP/RF 已存概率中恰为 0 的元素：' + '，'.join(
        f'{l_} Z0 {int(MEd[MEd.learner == l_].zero_Z0.sum())} / RT200 {int(MEd[MEd.learner == l_].zero_RT200.sum())}'
        for l_ in ML))
    P(f'    RF 的 argmax 并列（τ=0，按原样取第一个最大类）比例：Z0 均值 {MEd[MEd.learner == "rf"].tie_Z0.mean():.4f}'
      f'（最大 {MEd[MEd.learner == "rf"].tie_Z0.max():.4f}），RT200 均值 {MEd[MEd.learner == "rf"].tie_RT200.mean():.4f}'
      f'（最大 {MEd[MEd.learner == "rf"].tie_RT200.max():.4f}）；LR {MEd[MEd.learner == "lr"].tie_Z0.max():.4f}，'
      f'MLP {MEd[MEd.learner == "mlp"].tie_Z0.max():.4f}（最大）')
    nz = CRd.groupby(['learner', 'arm']).size()
    P(f'(10) 曲线行数：每 (学习器, 臂) {sorted(set(int(v) for v in nz.values))}（应 = {nU} × 2 × {len(TAUS)} = {nU * 2 * len(TAUS)}）')
    P(f'对账结论：{"全部通过" if ok_all else "有不通过项，见上"}')
    P('')

    # ---------------------------------------------------------------- 工具
    def wide(lrn, w):
        x = FRd[(FRd.learner == lrn) & (FRd.width == w)]
        return x.set_index(UKEY + ['arm'])['FA'].unstack('arm').reset_index()

    def udiff(W, a, b):
        U = W[UKEY].copy()
        U['d'] = W[a] - W[b]
        return U

    def block(title, U, show_ae_s2=True, show=True):
        cv = B.country_vals(U, 'd')
        if show:
            P(title)
            for f in (('both', 'ae', 's2') if show_ae_s2 else ('both',)):
                for drop in (True, False):
                    P(f'  {f:4s} {"9 国" if drop else "11 国":5s} {B.f95(B.t_summary(sel9(cv[f], drop)))}' +
                      (f'   （缺失单元 {int(U[U.clf == f].d.isna().sum())}/{int((U.clf == f).sum())}）'
                       if f != 'both' and not drop else ''))
        return cv

    def reach_line(lrn, arm, w=4):
        x = FRd[(FRd.learner == lrn) & (FRd.arm == arm) & (FRd.width == w)]
        parts = []
        for f in FEATS:
            xf = x[x.clf == f]
            miss = sorted(xf[xf.fa_missing].target.unique())
            parts.append(f'{f}: 全 {int((xf.reach >= 1).sum())} / 部分 {int(xf.partial.sum())} / 缺失 {int(xf.fa_missing.sum())}'
                         + (f'（{",".join(miss)}）' if miss else ''))
        return f'  {lrn:4s} {arm:8s} W{w}  ' + '；'.join(parts)

    EST = {}

    # ================================================================ L1
    P('######## L1 重训后再按同一比例修正：RTPK-k（LightGBM；锚点 = Z0 的 OA★；W4）########')
    P('RTPK-k = RT-k 的 i300 测试概率经 reweight(P_RT, p̂_k, π_train_RT)；p̂_k = laplace_prior(同一批 G SRS(k) 标签, 10, α=1)；'
      'π_train_RT = guidance_G.run_unit 的 pi_train_of(sel, R_MULT=50)（外国池 cw + 50·cw[标签]）。')
    me = MEd[MEd.learner == 'lgb']
    for k in KS:
        x = me[me.k == k]
        P(f'  k={k}：‖p̂ − π_train_RT‖₁ 均值 {x.L1_phat_vs_pitrain_rt.mean():.3f}（‖p̂ − π_train_F‖₁ {x.L1_phat_vs_pitrain_F.mean():.3f}；'
          f'‖π_train_RT − π_train_F‖₁ {x.L1_pitrain_rt_vs_F.mean():.3f}）')
    P('-- 可达率（单元 × 特征；W4）--')
    for arm in PK + RT + RTPK:
        P(reach_line('lgb', arm))
    W4 = wide('lgb', 4)
    for k in KS:
        EST[f'Rstar{k}'] = block(f'-- R*_{k} = FA(RTPK{k}) − FA(PK{k}) --', udiff(W4, f'RTPK{k}', f'PK{k}'))
    for k in KS:
        EST[f'RTPK-RT{k}'] = block(f'-- FA(RTPK{k}) − FA(RT{k}) --', udiff(W4, f'RTPK{k}', f'RT{k}'))
    P('-- 描述：ΔFA(RTPK-k) = FA(RTPK-k) − FA(Z0)（both；9 国 | 11 国）--')
    for k in KS:
        cv = B.country_vals(udiff(W4, f'RTPK{k}', 'Z0'), 'd')
        EST[f'dRTPK{k}'] = cv
        P(f'  k={k:<3d} {B.f95(B.t_summary(sel9(cv["both"], True)))} | {B.f95(B.t_summary(cv["both"]))}')
    P('-- 描述：τ=0 原始工作点上 RTPK-k − RT-k 的差（both；9 国 | 11 国）--')
    F4 = FRd[(FRd.learner == 'lgb') & (FRd.width == 4)]
    T0 = F4.set_index(UKEY + ['arm'])[['oa_t0', 'ba_t0']].unstack('arm')
    for k in KS:
        for met, mn in (('ba_t0', 'BA'), ('oa_t0', 'OA')):
            U = T0[met][[f'RTPK{k}', f'RT{k}']].reset_index()
            U['d'] = U[f'RTPK{k}'] - U[f'RT{k}']
            cv = B.country_vals(U, 'd')
            P(f'  k={k:<3d} {mn} {B.f95(B.t_summary(sel9(cv["both"], True)))} | {B.f95(B.t_summary(cv["both"]))}')
    # 读法
    sR = B.t_summary(sel9(EST['Rstar200']['both'], True))
    if not np.isfinite(sR['lo95']):
        rtxt, rcode = '无法判定（国家数不足）', 0
    elif sR['lo95'] > 0:
        rtxt, rcode = '第一条：R*_200 的 95% CI 下界 > 0 —— 重训在比例之外有增益', 1
    else:
        rtxt, rcode = '第二条（否则）：未检出重训在比例之外的增益', 2
    P('-- L1 读法（事先写定；k = 200、9 国、both）--')
    P(f'  R*_200 = {B.f95(sR)}')
    P(f'  → 适用读法：{rtxt}。' + ('（冒烟只有 2 个单元，判定无意义。）' if smoke else ''))
    for f in ('ae', 's2'):
        P(f'    分特征（描述）{f}: R*_200 {B.f95(B.t_summary(sel9(EST["Rstar200"][f], True)))}')
    P(f'    11 国（描述）: R*_200 {B.f95(B.t_summary(EST["Rstar200"]["both"]))}')
    P(f'READING_L1_K200_9C_BOTH={rcode}')
    P('')

    # ================================================================ L2
    P('######## L2 LR、MLP 下的同一分解（锚点 = 本学习器 Z0 在 τ=0 的 OA；W4）########')
    P('Z0 = 本学习器 k=0 已存概率；PK200 = Z0 经 reweight(P, p̂_200, π_train_F)；RT200 = 本学习器 SRS k=200 已存概率；'
      'RTPK200 = RT200 经 reweight(P, p̂_200, π_train_RT)。')
    for lrn in L2_LRN + ['rf']:
        tag = '' if lrn in L2_LRN else '（方案外，描述；规格 L2 只含 LR、MLP）'
        P(f'==== {lrn.upper()} {tag}====')
        for arm in ML_ARMS:
            P(reach_line(lrn, arm))
        Wl = wide(lrn, 4)
        EST[f'{lrn}_A200'] = block(f'-- {lrn} A_200 = FA(PK200) − FA(Z0) --', udiff(Wl, 'PK200', 'Z0'))
        EST[f'{lrn}_R200'] = block(f'-- {lrn} R_200 = FA(RT200) − FA(PK200) --', udiff(Wl, 'RT200', 'PK200'))
        EST[f'{lrn}_Rstar200'] = block(f'-- {lrn} R*_200 = FA(RTPK200) − FA(PK200) --', udiff(Wl, 'RTPK200', 'PK200'))
        EST[f'{lrn}_dRT200'] = block(f'-- {lrn} ΔFA(RT200) = FA(RT200) − FA(Z0) --', udiff(Wl, 'RT200', 'Z0'))
        m = Wl[['Z0', 'PK200', 'RT200']].notna().all(axis=1)
        ca = B.country_vals(udiff(Wl[m], 'PK200', 'Z0'), 'd')['both']
        cd = B.country_vals(udiff(Wl[m], 'RT200', 'Z0'), 'd')['both']
        P(f'  描述：A_200 占 ΔFA(RT200)（国家均值之比，both）9 国 {sel9(ca, True).mean() / sel9(cd, True).mean():.1%} | '
          f'11 国 {ca.mean() / cd.mean():.1%}（单元集合 = Z0、PK200、RT200 三者 FA 都非缺失）')
    P('')

    # ================================================================ L3
    P('######## L3 窗口敏感性（w = 2 / 4 / 8；LightGBM；锚点同上）########')
    P('  量                  特征 国家 | W2 | W4 | W8')
    L3Q = {'ΔFA(RT200)': ('RT200', 'Z0'), 'A_200': ('PK200', 'Z0'), 'R_200': ('RT200', 'PK200'),
           'R*_200': ('RTPK200', 'PK200')}
    l3_rows = []
    for name, (a, b) in L3Q.items():
        cvs = {w: B.country_vals(udiff(wide('lgb', w), a, b), 'd') for w in WIDTHS}
        for f in ('both', 'ae', 's2'):
            for drop in (True, False):
                ss = {w: B.t_summary(sel9(cvs[w][f], drop)) for w in WIDTHS}
                P(f'  {name:18s} {f:4s} {"9" if drop else "11":>2s}  | ' + ' | '.join(B.f95(ss[w]) for w in WIDTHS))
                for w in WIDTHS:
                    l3_rows.append(dict(quantity=name, learner='lgb', feat=f, set='9' if drop else '11', width=w, **ss[w]))
        miss = {w: int(udiff(wide('lgb', w), a, b).d.isna().sum()) for w in WIDTHS}
        P(f'  {"":18s} 缺失单元×特征：' + '，'.join(f'W{w} {miss[w]}' for w in WIDTHS))
    P('-- AE1 型 D = 同 OA 下 BA_ae − BA_s2（锚点 = min(OA★_ae, OA★_s2)，两特征 Z0 在 τ=0 的 OA；单元值即两特征合一）--')
    for lrn, arm in (('lgb', 'Z0'), ('lgb', 'RT200')):
        for drop in (True, False):
            ss = {}
            for w in WIDTHS:
                x = DRd[(DRd.learner == lrn) & (DRd.arm == arm) & (DRd.width == w)]
                ss[w] = B.t_summary(sel9(x.groupby('target').D.mean(), drop))
                l3_rows.append(dict(quantity=f'D({arm})', learner=lrn, feat='both', set='9' if drop else '11', width=w, **ss[w]))
            P(f'  D({arm:5s}) {lrn}   {"9" if drop else "11":>2s}  | ' + ' | '.join(B.f95(ss[w]) for w in WIDTHS))
        miss = {w: int(DRd[(DRd.learner == lrn) & (DRd.arm == arm) & (DRd.width == w)].D.isna().sum()) for w in WIDTHS}
        P(f'  {"":18s} 缺失单元：' + '，'.join(f'W{w} {miss[w]}' for w in WIDTHS))
    P('-- 方案外（描述）：LR、MLP 的 L2 量在 W2 / W4 / W8（both）--')
    for lrn in L2_LRN:
        for name, (a, b) in L3Q.items():
            cvs = {w: B.country_vals(udiff(wide(lrn, w), a, b), 'd') for w in WIDTHS}
            for drop in (True, False):
                P(f'  {lrn:3s} {name:12s} {"9" if drop else "11":>2s} | ' +
                  ' | '.join(B.f95(B.t_summary(sel9(cvs[w]['both'], drop))) for w in WIDTHS))
    extra['l3_table'] = pd.DataFrame(l3_rows)
    P('')

    # ================================================================ L4
    P('######## L4 只用 k 个标签选 τ 的可部署规则（LightGBM）########')
    P(f'规则（事先写定）：在 k 个 SRS 标签（G 的同一批选样，候选行）上算各 τ 的 OA_k、BA_k（BA 的类集合 = 标签中出现的类），'
      f'取 OA_k(τ) ≥ OA_k(0) − {OA_SLACK:g} 条件下 BA_k 最大的 τ̂（并列取 |τ| 最小者，再并列取较小 τ；比较容差 {TOL:g}）。'
      '标签行上各臂的 τ 族用该臂测试集的 π̂。Z0：标签不参与拟合；PK-k、HD-k：标签参与了 p̂ / 头的估计，τ̂ 为样本内选择。'
      'RT-k 的标签在训练集中，只报固定 τ ∈ {0, 0.25, 0.5}。')
    R4 = L4d.copy()
    key4 = ['target', 'split', 'rep', 'unit', 'clf', 'k']
    z_rule = R4[(R4.arm == 'Z0') & (R4['mode'] == 'rule')].set_index(key4)
    l4_sum = []
    for k in KS:
        P(f'==== k = {k} ====')
        zr = z_rule.xs(k, level='k')
        P(f'  τ̂ 分布（单元 × 特征）：Z0 中位 {zr.tau.median():+.2f} [P10 {zr.tau.quantile(.1):+.2f}, P90 {zr.tau.quantile(.9):+.2f}]，'
          f'τ̂=0 {int((zr.tau == 0).sum())}/{len(zr)}，τ̂ 在网格端点 {int(zr.tau.abs().ge(2).sum())}；'
          + '；'.join(f'{a}: 中位 {R4[(R4.arm == a) & (R4["mode"] == "rule")].tau.median():+.2f}，τ̂=0 '
                     f'{int((R4[(R4.arm == a) & (R4["mode"] == "rule")].tau == 0).sum())}' for a in (f'PK{k}', f'HD{k}')))
        P(f'  标签中出现的类数中位 {zr.n_cls_lab.median():.0f}')
        # 水平
        for arm in ('Z0', f'PK{k}', f'HD{k}'):
            x = R4[(R4.arm == arm) & (R4['mode'] == 'rule') & (R4.k == k)]
            for met, mn in (('ba_test', 'BA'), ('oa_test', 'OA')):
                U = x[UKEY].copy()
                U['d'] = x[met].to_numpy()
                cv = B.country_vals(U, 'd')
                P(f'  {arm:6s}(τ̂) 测试 {mn} 水平   both 9 国 {B.t_summary(sel9(cv["both"], True))["mean"]:6.2f} | '
                  f'11 国 {B.t_summary(cv["both"])["mean"]:6.2f}')
        def add_sum(contrast, mn, cv):
            row = dict(k=k, contrast=contrast, metric=mn)
            for f in ('both', 'ae', 's2'):
                for s_ in ('9', '11'):
                    ts = B.t_summary(sel9(cv[f], s_ == '9'))
                    row.update({f'{f}_{s_}_mean': ts['mean'], f'{f}_{s_}_lo95': ts['lo95'], f'{f}_{s_}_hi95': ts['hi95'],
                                f'{f}_{s_}_npos': ts['n_pos'], f'{f}_{s_}_n': ts['n']})
            l4_sum.append(row)

        # 规则相对原始工作点（Z0 自身，描述）
        x = R4[(R4.arm == 'Z0') & (R4['mode'] == 'rule') & (R4.k == k)]
        for met, mn in (('ba', 'BA'), ('oa', 'OA')):
            U = x[UKEY].copy()
            U['d'] = (x[f'{met}_test'] - x[f'{met}_test0']).to_numpy()
            add_sum('Z0(tauhat)-Z0(0)', mn, block(f'  - Z0(τ̂) − Z0(τ=0) 测试 {mn}（规则相对原始工作点，描述）', U))
        # 相对 Z0（同一规则、同一批 k 个标签）
        for arm, mode in [(f'PK{k}', 'rule'), (f'HD{k}', 'rule')] + [(f'RT{k}', f'fixed{t:g}') for t in (0.0, 0.25, 0.5)]:
            x = R4[(R4.arm == arm) & (R4['mode'] == mode) & (R4.k == k)]
            lab = f'{arm}(τ̂)' if mode == 'rule' else f'{arm}(τ={mode[5:]})'
            for met, mn in (('ba_test', 'BA'), ('oa_test', 'OA')):
                U = x[key4].copy()
                U['d'] = x[met].to_numpy() - z_rule[met].reindex(pd.MultiIndex.from_frame(x[key4])).to_numpy()
                add_sum(f'{lab}-Z0(tauhat)', mn, block(f'  - {lab} − Z0(τ̂) 测试 {mn}' +
                                                       ('（样本内选 τ）' if mode == 'rule' else ''), U))
        P('  -- 方案外对照：RT-k 与 Z0 在同一固定 τ 的差（both；9 国 | 11 国）--')
        for t in (0.0, 0.25, 0.5):
            xr = R4[(R4.arm == f'RT{k}') & (R4['mode'] == f'fixed{t:g}') & (R4.k == k)].set_index(key4)
            xz = R4[(R4.arm == 'Z0') & (R4['mode'] == f'fixed{t:g}') & (R4.k == k)].set_index(key4)
            for met, mn in (('ba_test', 'BA'), ('oa_test', 'OA')):
                U = xr.reset_index()[key4].copy()
                U['d'] = xr[met].to_numpy() - xz[met].reindex(xr.index).to_numpy()
                cv = B.country_vals(U, 'd')
                P(f'    RT{k}(τ={t:g}) − Z0(τ={t:g}) {mn}: {B.f95(B.t_summary(sel9(cv["both"], True)))} | '
                  f'{B.f95(B.t_summary(cv["both"]))}')
    extra['l4_summary'] = pd.DataFrame(l4_sum)
    P('')

    # ================================================================ L5
    P('######## L5 只用目标池训练的模型（tgt300）########')
    npz = [p for p in glob.glob(os.path.join(MODEL_DIAG, '**', '*'), recursive=True)
           if os.path.isfile(p) and 'tgt' in os.path.basename(p).lower() and not p.endswith('.json')]
    probs_like = [p for p in glob.glob(os.path.join(MODEL_DIAG, '**', '*.np*'), recursive=True)]
    js = sorted(glob.glob(os.path.join(MODEL_DIAG, 'fit', 'jobs', 'tgt_*_0_*_k0_L63_R500.json')))
    P(f'核查：verify/model_diag 下文件名含 "tgt" 的非 JSON 文件 {len(npz)} 个；全部 .npy/.npz {len(probs_like)} 个'
      f'（{sorted(set(os.path.relpath(os.path.dirname(p), MODEL_DIAG) for p in probs_like))}，均为 τ 曲线或 MLP/LGB 种子曲线，'
      f'不含 tgt 模型）；tgt 作业只有 fit/jobs/tgt_*_0_*_k0_L63_R500.json（{len(js)} 个），其中只存逐检查点的 OA、BA、NLL、'
      f'逐类召回，fitdiag.py 的 keep_probs 只用于 k0/k200 核对，tgt 的测试概率未保存。')
    P('→ tgt300 测试概率：不可得。FA(tgt300) − FA(Z0) 不计算；按规格只报已有的 τ = 0 差距（划分 0；不另行训练）。')
    tg_rows = []
    z0 = FRd[(FRd.learner == 'lgb') & (FRd.width == 4) & (FRd.split == 0)]
    for pth in js:
        j = json.load(open(pth))
        t_, f_ = j['job']['tgt'], j['job']['feat']
        m300 = j['metrics']['test']['300']
        zz = z0[(z0.target == t_) & (z0.clf == f_)]
        if not len(zz):
            continue
        z0r = zz[zz.arm == 'Z0'].iloc[0]
        bigG = MEd[(MEd.learner == 'lgb') & (MEd.split == 0) & (MEd.target == t_) & (MEd.clf == f_)].big.unique()
        assert len(bigG) == 1
        row = dict(target=t_, clf=f_, tgt300_oa=m300['oa'], tgt300_ba=m300['ba'],
                   big_same=','.join(map(str, m300['big'])) == bigG[0],
                   z0_oa_t0=z0r.oa_t0, z0_ba_t0=z0r.ba_t0, n_train=j.get('n_train'))
        for k in KS:
            rr = zz[zz.arm == f'RT{k}']
            row[f'rt{k}_ba_t0'] = rr.ba_t0.mean()
            row[f'rt{k}_oa_t0'] = rr.oa_t0.mean()
        Wz = wide('lgb', 4)
        Wz = Wz[(Wz.split == 0) & (Wz.target == t_) & (Wz.clf == f_)]
        for k in KS:
            row[f'dFA_RT{k}_s0'] = float((Wz[f'RT{k}'] - Wz['Z0']).mean())
        tg_rows.append(row)
    TG = pd.DataFrame(tg_rows)
    if len(TG):
        TG['gap_ba'] = TG.tgt300_ba - TG.z0_ba_t0
        TG['gap_oa'] = TG.tgt300_oa - TG.z0_oa_t0
        if os.path.exists(UB_CSV):
            ub = pd.read_csv(UB_CSV).rename(columns={'feat': 'clf', 'tgt': 'target'})
            jj = TG.merge(ub[['clf', 'target', 'tgt300_ba', 'tgt300_oa', 'k0_ba', 'k0_oa']], on=['clf', 'target'],
                          suffixes=('', '_ub'))
            P(f'  对账：tgt300 的 OA/BA 与 model_diag/fit/table_upper_bound.csv（保留 2 位）最大 |Δ| '
              f'{max((jj.tgt300_ba - jj.tgt300_ba_ub).abs().max(), (jj.tgt300_oa - jj.tgt300_oa_ub).abs().max()):.3f}；'
              f'Z0(τ=0) 与其 k0（P0test_i300 原始概率）最大 |Δ| BA {(jj.z0_ba_t0 - jj.k0_ba).abs().max():.3f}，'
              f'OA {(jj.z0_oa_t0 - jj.k0_oa).abs().max():.3f}（表中为 2 位小数）；json 中 ba 的类集合与 G 测试集 ≥30 类相同：'
              f'{int(TG.big_same.sum())}/{len(TG)}。')
        extra['l5_tgt300'] = TG
        P('  划分 0；国家值 = 该国（ae / s2）单个值；both = 两特征平均。τ=0 的差 X − Z0(τ=0)，单位 pp：')
        for name, colfn in (('tgt300 − Z0 BA', lambda d: d.gap_ba), ('tgt300 − Z0 OA', lambda d: d.gap_oa),
                            *[(f'RT{k} − Z0 BA（τ=0）', (lambda k_: (lambda d: d[f'rt{k_}_ba_t0'] - d.z0_ba_t0))(k)) for k in KS],
                            *[(f'RT{k} − Z0 OA（τ=0）', (lambda k_: (lambda d: d[f'rt{k_}_oa_t0'] - d.z0_oa_t0))(k)) for k in KS],
                            *[(f'ΔFA(RT{k})（划分 0，W4，同口径对照）', (lambda k_: (lambda d: d[f'dFA_RT{k_}_s0']))(k)) for k in KS]):
            U = TG[['target', 'clf']].copy()
            U['d'] = colfn(TG).to_numpy()
            cv = B.country_vals(U, 'd')
            P(f'  - {name}')
            for f in ('both', 'ae', 's2'):
                P(f'    {f:4s} 9 国 {B.f95(B.t_summary(sel9(cv[f], True)))} | 11 国 {B.f95(B.t_summary(cv[f]))}')
        P('  RT-k 的 τ=0 差与 ΔFA(RT-k) 取划分 0 的两个重复平均；tgt300 为 fitdiag 的 tgt 作业（目标候选池 pool_positions(S, 0, 0)，'
          '30,000 个，池频率类权重，L63，第 300 轮），与 G 划分 0 同一测试集。')
    P('')

    # ================================================================ AE7
    P('######## AE7 界值与功效 ########')
    FB2 = pd.read_csv(FA_B2, float_precision='round_trip')
    WB = FB2[FB2.grid == 'new'].set_index(UKEY + ['arm'])['FA'].unstack('arm').reset_index()
    INT = {'A_200': udiff(WB, 'PK200', 'Z0'), 'R_200': udiff(WB, 'RT200', 'PK200'),
           'ΔFA(RT200)': udiff(WB, 'RT200', 'Z0'), 'R*_200（L1）': udiff(W4, 'RTPK200', 'PK200')}
    # 与本实现 W4 的一致性（B2 源 vs 本脚本重算）
    for nm, (a, b) in (('A_200', ('PK200', 'Z0')), ('R_200', ('RT200', 'PK200')), ('ΔFA(RT200)', ('RT200', 'Z0'))):
        mine = udiff(W4, a, b).set_index(UKEY).d
        src = INT[nm].set_index(UKEY).d.reindex(mine.index)
        assert np.array_equal(mine.to_numpy(), src.to_numpy(), equal_nan=True), nm
    P('-- 界值：AE−S2 交互量 = 国家值(ae) − 国家值(s2)（国家值 = 4 单元平均；A_200、R_200、ΔFA(RT200) 取自 partB2/fa_rows_B2.csv，'
      '已断言与本实现 W4 逐位相同）；90% t CI（df = n−1）；TOST ±1 pp 的 p = max(两个单侧 t 检验 p) --')
    bnd_rows = []
    for nm, U in INT.items():
        cv = B.country_vals(U, 'd')
        dd = cv['ae'] - cv['s2']
        for drop in (True, False):
            s_ = ci90(sel9(dd, drop))
            s95 = B.t_summary(sel9(dd, drop))
            bnd_rows.append(dict(quantity=nm, set='9' if drop else '11', **s_, lo95=s95['lo95'], hi95=s95['hi95']))
            P(f'  {nm:12s} {"9 国" if drop else "11 国":5s} 均值 {s_["mean"]:+.2f}，90% CI [{s_["lo90"]:+.2f}, {s_["hi90"]:+.2f}]，'
              f'95% CI [{s95["lo95"]:+.2f}, {s95["hi95"]:+.2f}]，TOST(±1) p = {s_["p_tost"]:.3f}，'
              f'能通过 TOST 的最小界值 {s_["min_margin"]:.2f}，SD {s_["sd"]:.2f}，>0 {s_["n_pos"]}/{s_["n"]}')
    extra['ae7_bounds'] = pd.DataFrame(bnd_rows)

    P('-- 功效：国家层 SD 只用 rep0 单元（每国 s0/r0、s1/r0 两个单元的平均，与确认运行每国 2 单元同设计）；'
      '确认运行 22 单元 = 11 国 × 2；n = 9 或 11 国；单样本 t。--')
    P('   H_R1：同 OA 下 AE−S2 > 0（D，W4，锚点 min(OA★_ae, OA★_s2)），单侧 α = 0.025；功效按"真值 = 本回溯 4 单元国家均值"'
      '（赢家诅咒，偏乐观）；MDE80 = 80% 功效的最小真值。')
    P('   H_R2：交互量 ±1 pp TOST，α = 0.05（90% CI 落在 ±1 内）；功效分"真值 = 0"与"真值 = 本回溯均值"；'
      '另给真值 = 0 时 80% 功效所需的最小界值。TOST 功效为对 S 分布数值积分的精确解。')
    P('   两种 SD 都来自复用测试集，不含确认运行换新测试集的额外变异，偏乐观。')
    pw_rows = []
    P('  H_R1         学习器 臂      国家  n  均值(4单元)  SD_r0  SD_4 | 功效(真值=均值) | MDE80')
    for lrn, arms in D_ARMS.items():
        for arm in arms:
            x = DRd[(DRd.learner == lrn) & (DRd.arm == arm) & (DRd.width == 4)]
            for drop in (True, False):
                v4 = sel9(x.groupby('target').D.mean(), drop).dropna()
                vr0 = sel9(x[x.rep == 0].groupby('target').D.mean(), drop).dropna()
                n = len(vr0)
                if n < 3:
                    continue
                sd0, sd4 = vr0.std(ddof=1), v4.std(ddof=1)
                pw = power_1s(v4.mean(), sd0, n)
                md = mde80_1s(sd0, n)
                pw_rows.append(dict(hyp='H_R1', quantity=f'D({arm})', learner=lrn, set='9' if drop else '11', n=n,
                                    mean4=v4.mean(), mean_r0=vr0.mean(), sd_r0=sd0, sd4=sd4, power_at_mean=pw, mde80=md))
                P(f'  H_R1 D       {lrn:4s}  {arm:6s}  {"9" if drop else "11":>2s}  {n:2d}  {v4.mean():+7.2f}     {sd0:5.2f}  '
                  f'{sd4:5.2f} | {pw:6.3f}          | {md:5.2f}' + ('' if lrn == 'lgb' and arm == 'Z0' else '   （附）'))
    P('  H_R2         交互量        国家  n  均值(4单元)  SD_r0  SD_4 | TOST 功效(真值=0) | (真值=均值) | 80% 功效所需界值(真值=0)')
    for nm, U in INT.items():
        cv0 = B.country_vals(U[U.rep == 0], 'd')
        cv4 = B.country_vals(U, 'd')
        for drop in (True, False):
            d0 = sel9(cv0['ae'] - cv0['s2'], drop).dropna()
            d4 = sel9(cv4['ae'] - cv4['s2'], drop).dropna()
            n = len(d0)
            if n < 3:
                continue
            sd0, sd4 = d0.std(ddof=1), d4.std(ddof=1)
            p0 = tost_power(0.0, sd0, n)
            pm = tost_power(d4.mean(), sd0, n)
            mg = tost_margin80(sd0, n)
            pw_rows.append(dict(hyp='H_R2', quantity=nm, learner='lgb', set='9' if drop else '11', n=n, mean4=d4.mean(),
                                mean_r0=d0.mean(), sd_r0=sd0, sd4=sd4, tost_power_true0=p0, tost_power_at_mean=pm,
                                margin80_true0=mg))
            P(f'  H_R2 {nm:12s}        {"9" if drop else "11":>2s}  {n:2d}  {d4.mean():+7.2f}     {sd0:5.2f}  {sd4:5.2f} | '
              f'{p0:6.3f}            | {pm:6.3f}      | {mg:5.2f}')
    # TOST 精确积分与蒙特卡罗核对
    x0 = [r for r in pw_rows if r['hyp'] == 'H_R2']
    if x0:
        r0 = x0[0]
        mc = tost_power_mc(0.0, r0['sd_r0'], r0['n'])
        P(f'  （核对：{r0["quantity"]} {r0["set"]} 国 TOST 功效 精确 {r0["tost_power_true0"]:.4f} vs 蒙特卡罗 20 万次 {mc:.4f}）')
    extra['ae7_power'] = pd.DataFrame(pw_rows)
    P('')

    # ================================================================ 国家值
    P('######## 国家值（both；LightGBM W4；pp）########')
    cols = ['Rstar50', 'Rstar200', 'Rstar400', 'RTPK-RT200', 'lr_Rstar200', 'mlp_Rstar200', 'lr_R200', 'mlp_R200',
            'lr_A200', 'mlp_A200']
    P('  国家 ' + ' '.join(f'{c:>12s}' for c in cols) + '     D(Z0)W4   D(RT200)W4')
    dz = DRd[(DRd.learner == 'lgb') & (DRd.width == 4)]
    for t in COUNTRIES:
        dz0 = dz[(dz.arm == 'Z0') & (dz.target == t)].D.mean()
        dr = dz[(dz.arm == 'RT200') & (dz.target == t)].D.mean()
        P(f'  {t}   ' + ' '.join(f'{EST[c]["both"].get(t, np.nan):+12.2f}' for c in cols) + f'  {dz0:+10.2f} {dr:+10.2f}')
    P('')
    return lines, extra


if __name__ == '__main__':
    main()
