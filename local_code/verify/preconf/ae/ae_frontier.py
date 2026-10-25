"""
确认运行登记前的不训练分析 · AE 线：AE1、AE3、AE4。
规格：label_expansion/PRECONF_ANALYSES_SPEC_2026-09-29.md（sha256[:16]=d0906bd1417839d2，写于本批任何代码与输出之前）。
事后分析，C 级、描述性：只读 G s0/s1 的已存概率（阶段 0 的 P_T、B2 的 τ 曲线、guidance_G_ml 的 LR/MLP/RF 概率）；
不训练任何分类器；测试标签只用于评估；本脚本不使用候选标签（AE1 的 PK200/HD200 曲线由 B2 用候选标签算好，这里直接读）。

口径（与第 0 步补遗 B2 的 step0B2.py 相同，import 复用，不改其文件）：
  τ 族      step0B2.tau_preds_grid（= step0B.tau_preds 源码）；τ ∈ [−2, 2] 步长 0.05（81 点）
  π̂        该臂测试集平均预测概率（exp(Z − 行最大) 归一化后取均值，不用标签）
  log       step0B.log_or_neginf（精确 log，概率恰为 0 → −inf）
  OA / BA   step0B.Scorer（ba_ge30 的类集合 = 该单元测试集中 ≥ 30 个地块的类）
  上包络    step0B.envelope（相邻 τ 点在 (OA, BA) 平面线性插值；多段经过 o 取 BA 最大）
  窗口      W_w = [锚点 − w, 锚点] 步长 0.1；主口径 w = 4（= step0B.OFFS）
  汇总      国家值 = 该国可用单元（2 划分 × 2 重复）的平均；11 国与 9 国（去 RO、PT）；均值、95% t CI、>0 的国家数
            （step0B.t_summary / f95）

AE1  同 OA 下 AE−S2 前沿差 D：每个单元，两种特征各用自己的臂曲线；锚点 = 两种特征零标签参照在 τ=0 的 OA 的较小者；
     窗口内逐点 BA_ae(o) − BA_s2(o)（单曲线上包络），只在两者共同可达的 o 上平均；共同可达 <50% 记缺失。
     LightGBM：臂 Z0、PK200、RT200、HD200 直接读 partB2/curves_B2.csv，锚点 = Z0 的 OA★。
     LR/MLP/RF：臂 K0（k=0）与 S200（SRS k=200，rows_{learner}_*.csv 的 sel_hash），在同一 τ 网格上算 τ 族；
                锚点 = 本学习器 K0 在 τ=0 的 OA 的较小者。RF 的 argmax 并列按原样（numpy 取第一个）并报并列比例。
     另报：窗口宽度 2/4/8；锚点取 OA★_ae 或 OA★_s2 的版本；τ=0.5（可部署点）与 τ=0 的 AE−S2 BA/OA；
           AE / S2 在整个窗口占优（共同可达点上 BA_ae ≥ BA_s2 处处成立 / 反之）的单元数。
     方案外（对账与描述）：锚点取该臂自身 τ=0 OA 的较小者（与 AE–S2 整合评估 matched_oa_by_arm 的口径相同）。
AE3  LightGBM Z0；o_mid = 锚点 − 2（锚点同 AE1）；两种特征各自在曲线上取 OA = o_mid 的点（段号与插值系数取自上包络），
     逐类召回按同一系数在相邻 τ 之间线性插值；每类召回 AE−S2，只计该单元测试集中该类 ≥ 30 的单元，先平均到国家，
     再报国家均值（及 95% t CI、>0 国家数）与国家数；对照 τ=0 的逐类差。
AE4  核面积 π·30² m²；测试地块按 面积/核面积 分 5 组 [0,0.5)、[0.5,1)、[1,2)、[2,4)、[4,∞)；每单元每组：
     组占比；τ=0 与 AE3 同 OA 点（各特征自己的 τ 段与插值系数）上组内 OA 的 AE−S2；
     组内 BA 的 AE−S2（只计该单元该组中 ≥ 30 个地块的类；<2 类记缺失）。

用法：python -B ae_frontier.py --smoke   # AT_s0_r0、RO_s1_r1 两个单元，输出到 ae_frontier_smoke/，做对账
      python -B ae_frontier.py           # 全量 44 个单元
"""
import os
import sys
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_v] = '2'
sys.dont_write_bytecode = True  # 不在项目目录、partB、partB2 写 __pycache__
import ast
import glob
import hashlib
import re
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PARTB2 = '/Users/perng_mini/AlphaCrops_local/verify/step0/partB2'
sys.path.insert(0, PARTB2)
import step0B2 as B2  # noqa: E402  导入链：step0B（chdir 到项目目录，只读；导入 guidance_G）
from scipy import stats  # noqa: E402

B = B2.B
G = B2.G
SPEC = os.path.join(B.L, 'PRECONF_ANALYSES_SPEC_2026-09-29.md')
SPEC_HASH = 'd0906bd1417839d2'
CURVES_B2 = os.path.join(PARTB2, 'curves_B2.csv')
STEP0B2_PY = os.path.join(PARTB2, 'step0B2.py')
STEP0B_PY = B2.STEP0B_PY
CURVES_B = B2.CB_CSV
INTEG = '/Users/perng_mini/AlphaCrops_local/verify/storyline/ae_s2_integration_20260929'
INTEG_UNITS = f'{INTEG}/ae_s2_evidence/matched_oa_units.csv'
INTEG_ARMS = f'{INTEG}/ae_s2_evidence/matched_oa_by_arm_report.txt'
INTEG_ML = f'{INTEG}/ae_s2_rse_design_20260929/matched_oa_lr_mlp_out.txt'
ML_REPORT = os.path.join(B.R_ML, 'report_ML_rev3.txt')
R_ML = B.R_ML

C = B.C
FEATS = B.FEATS
COUNTRIES = B.COUNTRIES
DROP = B.DROP
MIN_COV = B.MIN_COV
TAUS = B2.TAUS_NEW
I_T0 = B2.I_T0
I_T05 = int(np.flatnonzero(np.abs(TAUS - 0.5) < 1e-9)[0])
IDX_OLD = B2.IDX_OLD
WIDTHS = (2, 4, 8)
OFFS_W = {w: np.round(np.arange(-float(w), 0.0 + 1e-9, 0.1), 1) for w in WIDTHS}
assert np.array_equal(OFFS_W[4], B.OFFS)
LGB_ARMS = ['Z0', 'PK200', 'RT200', 'HD200']
LEARNERS = ['lr', 'mlp', 'rf']
ML_ARMS = ['K0', 'S200']
SETS = [('lgb', a) for a in LGB_ARMS] + [(l, a) for l in LEARNERS for a in ML_ARMS]
REF_ARM = {'lgb': 'Z0', 'lr': 'K0', 'mlp': 'K0', 'rf': 'K0'}
LNAME = {'lgb': 'LightGBM', 'lr': 'LR', 'mlp': 'MLP', 'rf': 'RF'}
CORE_HA = np.pi * 30.0 ** 2 / 1e4          # 核面积 ≈ 2,827 m² = 0.2827 ha（GData.area 单位为 ha）
SIZE_CUTS = [0.5, 1.0, 2.0, 4.0]           # np.digitize（right=False）：[0,0.5)→0，[0.5,1)→1，[1,2)→2，[2,4)→3，[4,∞)→4
SIZE_LAB = ['<0.5', '0.5–1', '1–2', '2–4', '≥4']
NG = len(SIZE_LAB)
UKEY = ['target', 'split', 'rep', 'unit']


def sha16(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()[:16]


def interp(arr, seg, lam):
    """与 step0B.transfer 相同：点 = (1−lam)·arr[seg] + lam·arr[seg+1] 的写法 arr[seg] + lam·(arr[seg+1] − arr[seg])。"""
    if seg < 0:
        return np.nan
    return arr[seg] + lam * (arr[seg + 1] - arr[seg])


def pihat_of(Z):
    Pa = np.exp(Z - Z.max(1, keepdims=True))
    Pa /= Pa.sum(1, keepdims=True)
    return Pa.mean(0)


# ---------------------------------------------------------------- AE1：一对特征曲线在一个窗口上的 D
def frontier_diff(cv_ae, cv_s2, grid):
    ba_a = B.envelope([cv_ae], grid)[0]
    ba_s = B.envelope([cv_s2], grid)[0]
    ok = np.isfinite(ba_a) & np.isfinite(ba_s)
    cov = float(ok.mean())
    has = cov >= MIN_COV
    d = ba_a[ok] - ba_s[ok]
    return dict(reach_ae=float(np.isfinite(ba_a).mean()), reach_s2=float(np.isfinite(ba_s).mean()), cov=cov,
                missing=not has, D=float(d.mean()) if has else np.nan,
                lvl_ae=float(ba_a[ok].mean()) if has else np.nan, lvl_s2=float(ba_s[ok].mean()) if has else np.nan,
                ae_dom=float((d >= 0).all()) if has else np.nan, s2_dom=float((d <= 0).all()) if has else np.nan,
                n_ae_gt=int((d > 0).sum()), n_s2_gt=int((d < 0).sum()), n_co=int(ok.sum()))


def ae1_unit(cell, CV):
    """CV：{(learner, arm): {f: (oa, ba)}}（新网格 81 点）。返回 AE1 行。"""
    rows = []
    for lrn, arm in SETS:
        cv = CV[(lrn, arm)]
        ref = CV[(lrn, REF_ARM[lrn])]
        star = {f: ref[f][0][I_T0] for f in FEATS}
        own = {f: cv[f][0][I_T0] for f in FEATS}
        anchors = {'min': min(star.values()), 'ae': star['ae'], 's2': star['s2']}
        if arm != REF_ARM[lrn]:
            anchors['own'] = min(own.values())  # 方案外：该臂自身 τ=0 OA 的较小者
        pts = {}
        for f in FEATS:
            pts.update({f'oa_t0_{f}': cv[f][0][I_T0], f'ba_t0_{f}': cv[f][1][I_T0],
                        f'oa_t05_{f}': cv[f][0][I_T05], f'ba_t05_{f}': cv[f][1][I_T05],
                        f'oastar_{f}': star[f]})
        for an, anc in anchors.items():
            for w in WIDTHS:
                if an == 'own' and w != 4:
                    continue
                r = dict(cell, learner=lrn, arm=arm, anchor=an, width=w, anc=anc, **pts)
                r.update(frontier_diff(cv['ae'], cv['s2'], anc + OFFS_W[w]))
                rows.append(r)
    return rows


# ---------------------------------------------------------------- LR / MLP / RF 曲线（新网格）
def load_ML_rows():
    MR = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(R_ML, 'rows_*_*.csv'))], ignore_index=True)
    MR = MR[MR.learner.isin(LEARNERS) & (MR.cond == 'none') &
            (((MR.method == 'srs') & (MR.k == 200)) | (MR.method == 'foreign_only'))]
    assert (MR.status == 'ok').all()
    return MR


def ml_curves(yT, sc, MR, GR, tgt, split, rep):
    unit = f'{tgt}_s{split}_r{rep}'
    CV, chk, ties = {}, [], []
    for lrn in LEARNERS:
        for f in FEATS:
            q = (MR.learner == lrn) & (MR.target == tgt) & (MR.split == split) & (MR.rep == rep) & (MR.clf == f)
            m0, m2 = MR[q & (MR.method == 'foreign_only')], MR[q & (MR.method == 'srs')]
            assert len(m0) == 1 and len(m2) == 1, (lrn, unit, f)
            m0, m2 = m0.iloc[0], m2.iloc[0]
            g = GR[(GR.method == 'srs') & (GR.target == tgt) & (GR.split == split) & (GR.rep == rep) & (GR.clf == f) &
                   (GR.k == 200)]
            same_sel = bool(len(g) == 1 and g.iloc[0].sel_hash == m2.sel_hash)
            for arm, path, ref in (('K0', os.path.join(R_ML, 'probs', lrn, f'{tgt}_s{split}_{f}_k0.npz'), m0),
                                   ('S200', os.path.join(R_ML, 'probs', lrn, f'{unit}_{f}_k200_{m2.sel_hash}.npz'), m2)):
                P = B.norm64(np.load(path)['P'])
                Z = B.log_or_neginf(P)
                pihat = pihat_of(Z)
                PH = B2.tau_preds_grid(Z, pihat, TAUS)
                oa, ba = sc.curve(PH)
                CV.setdefault((lrn, arm), {})[f] = (oa, ba)
                chk.append(dict(unit=unit, target=tgt, split=split, rep=rep, clf=f, learner=lrn, arm=arm, ref='ML_rows',
                                oa=oa[I_T0], ba_ge30=ba[I_T0], oa_ref=ref.oa, ba_ref=ref.ba_ge30,
                                same_sel_as_G_lgb=same_sel if arm == 'S200' else np.nan))
                # 并列比例：τ=0（= 原始概率的 argmax）与整个 τ 网格的平均
                lp = np.where(pihat > 0, np.log(np.where(pihat > 0, pihat, 1.0)), 0.0)
                tie_t = np.empty(len(TAUS))
                for i, t in enumerate(TAUS):
                    srt = np.sort(Z - t * lp, axis=1)
                    tie_t[i] = float((srt[:, -1] == srt[:, -2]).mean())
                ties.append(dict(unit=unit, target=tgt, split=split, rep=rep, clf=f, learner=lrn, arm=arm,
                                 tie_t0=tie_t[I_T0], tie_mean=float(tie_t.mean()), tie_max=float(tie_t.max()),
                                 n_zero_prob_frac=float((P == 0).mean())))
    return CV, chk, ties


# ---------------------------------------------------------------- AE3 / AE4：Z0 的逐类与分组明细
def z0_detail(D, S, cell, yT, big, sc, CV_lgb_z0, class_names):
    y = np.asarray(D.y)
    n_cand = len(S['cand'])
    test = S['test']
    grp = np.digitize(np.asarray(D.area)[test] / CORE_HA, SIZE_CUTS)
    ny = np.bincount(yT, minlength=C)
    det, chk = {}, []
    for f in FEATS:
        P0 = B.norm64(S[f'P_T_{f}'][n_cand:].astype(np.float64))   # 与 step0B2 的 Z0 相同
        Z = B.log_or_neginf(P0)
        PH = B2.tau_preds_grid(Z, pihat_of(Z), TAUS)
        oa, ba = sc.curve(PH)
        corr = PH == yT[None, :]
        rec = np.full((len(TAUS), C), np.nan)
        for c in range(C):
            if ny[c]:
                rec[:, c] = corr[:, yT == c].mean(1)
        goa = np.full((len(TAUS), NG), np.nan)
        grec = {}
        for j in range(NG):
            mj = grp == j
            if mj.any():
                goa[:, j] = corr[:, mj].mean(1)
            for c in range(C):
                mjc = mj & (yT == c)
                if mjc.sum() >= G.MIN_N:
                    grec[(j, c)] = corr[:, mjc].mean(1)
        # 与 curves_B2.csv 的 Z0 逐位对账
        ref_oa, ref_ba = CV_lgb_z0[f]
        chk.append(dict(cell, clf=f, check='Z0_curve_vs_curves_B2', max_abs_oa=float(np.max(np.abs(oa - ref_oa))),
                        max_abs_ba=float(np.max(np.abs(ba - ref_ba))),
                        bitwise=bool(np.array_equal(oa, ref_oa) and np.array_equal(ba, ref_ba))))
        det[f] = dict(oa=oa, ba=ba, rec=rec, goa=goa, grec=grec)
    star = {f: det[f]['oa'][I_T0] for f in FEATS}
    anc = min(star.values())
    o_mid = anc - 2.0
    pt = {}
    for f in FEATS:
        best, cid, seg, lam = B.envelope([(det[f]['oa'], det[f]['ba'])], np.array([o_mid]))
        s_, l_ = int(seg[0]), float(lam[0])
        tau_mid = (1 - l_) * TAUS[s_] + l_ * TAUS[s_ + 1] if s_ >= 0 else np.nan
        pt[f] = (s_, l_, float(best[0]), tau_mid)
        if s_ >= 0:
            oa_i = interp(det[f]['oa'], s_, l_)
            ba_i = 100.0 * np.nanmean([interp(det[f]['rec'][:, c], s_, l_) for c in big])
            chk.append(dict(cell, clf=f, check='AE3_point', oa_interp_minus_omid=float(oa_i - o_mid),
                            ba_from_recalls_minus_envelope=float(ba_i - best[0])))
    # ---- AE3 行（逐类）
    cls_rows = []
    for c in range(C):
        r = dict(cell, cls=c, cls_name=class_names[c], n_test=int(ny[c]), ge30=bool(ny[c] >= G.MIN_N),
                 anchor=anc, o_mid=o_mid)
        for f in FEATS:
            s_, l_, b_, t_ = pt[f]
            r[f'tau_mid_{f}'] = t_
            r[f'rec_mid_{f}'] = 100.0 * interp(det[f]['rec'][:, c], s_, l_) if ny[c] else np.nan
            r[f'rec_t0_{f}'] = 100.0 * det[f]['rec'][I_T0, c] if ny[c] else np.nan
        r['d_mid'] = r['rec_mid_ae'] - r['rec_mid_s2']
        r['d_t0'] = r['rec_t0_ae'] - r['rec_t0_s2']
        cls_rows.append(r)
    # ---- AE4 行（逐组）
    size_rows = []
    nT = len(yT)
    for j in range(NG):
        mj = grp == j
        cls_ok = [c for c in range(C) if (j, c) in det['ae']['grec']]
        assert cls_ok == [c for c in range(C) if (j, c) in det['s2']['grec']]
        r = dict(cell, grp=j, grp_lab=SIZE_LAB[j], n_grp=int(mj.sum()), share=float(mj.mean()), n_cls_ge30=len(cls_ok),
                 ba_missing=len(cls_ok) < 2)
        for f in FEATS:
            s_, l_, b_, t_ = pt[f]
            goa = det[f]['goa'][:, j]
            r[f'oa_t0_{f}'] = 100.0 * goa[I_T0] if mj.any() else np.nan
            r[f'oa_mid_{f}'] = 100.0 * interp(goa, s_, l_) if mj.any() else np.nan
            if len(cls_ok) >= 2:
                r[f'ba_t0_{f}'] = 100.0 * float(np.mean([det[f]['grec'][(j, c)][I_T0] for c in cls_ok]))
                r[f'ba_mid_{f}'] = 100.0 * float(np.mean([interp(det[f]['grec'][(j, c)], s_, l_) for c in cls_ok]))
            else:
                r[f'ba_t0_{f}'] = r[f'ba_mid_{f}'] = np.nan
        for q in ('oa_t0', 'oa_mid', 'ba_t0', 'ba_mid'):
            r[f'd_{q}'] = r[f'{q}_ae'] - r[f'{q}_s2']
        size_rows.append(r)
    # 分组对账：Σ 占比 × 组 OA = 总 OA（τ=0 与同 OA 点）
    for f in FEATS:
        sh = np.array([r['share'] for r in size_rows])
        o0 = np.array([r[f'oa_t0_{f}'] for r in size_rows])
        om = np.array([r[f'oa_mid_{f}'] for r in size_rows])
        chk.append(dict(cell, clf=f, check='AE4_share_weighted_oa',
                        t0_diff=float(np.nansum(sh * o0) - det[f]['oa'][I_T0]),
                        mid_diff=float(np.nansum(sh * om) - o_mid) if pt[f][0] >= 0 else np.nan,
                        share_sum_minus1=float(sh.sum() - 1.0), n_test=nT))
    return cls_rows, size_rows, chk


# ---------------------------------------------------------------- 主程序
def read_curves_B2():
    CB = pd.read_csv(CURVES_B2, float_precision='round_trip')
    CB = CB[CB.arm.isin(LGB_ARMS)].copy()
    CB['ti'] = np.rint(CB.tau * 100).astype(int)
    out = {}
    for (u, f, a), g in CB.groupby(['unit', 'clf', 'arm']):
        g = g.sort_values('ti')
        assert np.array_equal(g.ti.to_numpy(), np.rint(TAUS * 100).astype(int)), (u, f, a)
        out[(u, f, a)] = (g.oa.to_numpy(), g.ba_ge30.to_numpy())
    return out


def main():
    smoke = '--smoke' in sys.argv
    out = os.path.join(HERE, 'ae_frontier_smoke') if smoke else HERE
    os.makedirs(out, exist_ok=True)
    assert sha16(SPEC) == SPEC_HASH, '规格文件哈希与登记值不一致'
    t0 = time.time()
    print(f'ae_frontier.py sha256[:16]={sha16(os.path.abspath(__file__))}；step0B2.py {sha16(STEP0B2_PY)}；'
          f'step0B.py {sha16(STEP0B_PY)}；curves_B2.csv {sha16(CURVES_B2)}', flush=True)
    D = G.GData()
    GR = B.load_G_rows()
    MR = load_ML_rows()
    CB2 = read_curves_B2()
    class_names = list(D.classes)
    units = ([('AT', 0, 0), ('RO', 1, 1)] if smoke else
             [(t, s, r) for t in COUNTRIES for s in (0, 1) for r in (0, 1)])
    A1, CL, SZ, KR, TI, MLC, B2C = [], [], [], [], [], [], []
    cache = {}
    y = np.asarray(D.y)
    for tgt, split, rep in units:
        if (tgt, split) not in cache:
            cache.clear()
            cache[(tgt, split)] = np.load(os.path.join(B.R_G, 'stage0', f'{tgt}_s{split}.npz'))
        S = cache[(tgt, split)]
        unit = f'{tgt}_s{split}_r{rep}'
        cell = dict(target=tgt, split=split, rep=rep, unit=unit)
        yT = y[S['test']]
        big = np.flatnonzero(np.bincount(yT, minlength=C) >= G.MIN_N)
        sc = B.Scorer(yT, big)
        CV = {('lgb', a): {f: CB2[(unit, f, a)] for f in FEATS} for a in LGB_ARMS}
        cvm, chk, ties = ml_curves(yT, sc, MR, GR, tgt, split, rep)
        CV.update(cvm)
        KR += chk
        TI += ties
        for (lrn, arm), d in cvm.items():
            for f, (oa, ba) in d.items():
                MLC += [dict(cell, learner=lrn, arm=arm, clf=f, tau=t, oa=oa[i], ba_ge30=ba[i]) for i, t in enumerate(TAUS)]
        A1 += ae1_unit(cell, CV)
        cr, sr, kc = z0_detail(D, S, cell, yT, big, sc, CV[('lgb', 'Z0')], class_names)
        CL += cr
        SZ += sr
        KR += kc
        # 对账：用 step0B2.run_unit 重算本单元全部 B2 曲线，与 curves_B2.csv 逐位比较（Z0、PK200、RT200、HD200）
        cr2, _, _, _ = B2.run_unit(D, S, GR, tgt, split, rep, log=lambda s: None)
        C2 = pd.DataFrame(cr2)
        for (f, a), g in C2[C2.arm.isin(LGB_ARMS)].groupby(['clf', 'arm']):
            oa_r, ba_r = CB2[(unit, f, a)]
            B2C.append(dict(cell, clf=f, arm=a, bitwise=bool(np.array_equal(g.oa.to_numpy(), oa_r) and
                                                            np.array_equal(g.ba_ge30.to_numpy(), ba_r)),
                            max_abs=float(max(np.abs(g.oa.to_numpy() - oa_r).max(), np.abs(g.ba_ge30.to_numpy() - ba_r).max()))))
        print(f'[{time.time() - t0:.0f}s] {unit} 完成', flush=True)
    A1d, CLd, SZd, TId, MLCd, B2Cd = map(pd.DataFrame, (A1, CL, SZ, TI, MLC, B2C))
    KRd = pd.DataFrame(KR)
    A1d.to_csv(os.path.join(out, 'ae_frontier_rows.csv'), index=False)
    CLd.to_csv(os.path.join(out, 'ae_frontier_class_rows.csv'), index=False)
    SZd.to_csv(os.path.join(out, 'ae_frontier_size_rows.csv'), index=False)
    TId.to_csv(os.path.join(out, 'ae_frontier_ties.csv'), index=False)
    MLCd.to_csv(os.path.join(out, 'ae_frontier_curves_ml.csv'), index=False)
    KRd.to_csv(os.path.join(out, 'ae_frontier_checks.csv'), index=False)
    B2Cd.to_csv(os.path.join(out, 'ae_frontier_checks_B2rerun.csv'), index=False)
    lines, country = summarize(A1d, CLd, SZd, TId, MLCd, KRd, B2Cd, CB2, smoke)
    country.to_csv(os.path.join(out, 'ae_frontier_country.csv'), index=False)
    open(os.path.join(out, 'ae_frontier_summary.txt'), 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(f'总耗时 {time.time() - t0:.0f}s')


# ---------------------------------------------------------------- 汇总工具
def sel9(v, drop):
    return v.drop([c for c in DROP if c in v.index]) if drop else v


def cvals(U, col):
    """国家值 = 该国可用单元的平均（NaN 不计）。"""
    return U.groupby('target')[col].mean().reindex([c for c in COUNTRIES if c in set(U.target)])


def s95(v, drop):
    return B.f95(B.t_summary(sel9(v, drop)))


def both_sets(v):
    return f'9 国 {s95(v, True)} | 11 国 {s95(v, False)}'


def parse_dicts(path, pat):
    txt = open(path, encoding='utf-8').read()
    return [ast.literal_eval(m) for m in re.findall(pat, txt)]


# ---------------------------------------------------------------- 汇总
def summarize(A1d, CLd, SZd, TId, MLCd, KRd, B2Cd, CB2, smoke):
    lines = []
    P = lines.append
    nU = A1d.unit.nunique()
    P('确认运行登记前的不训练分析 · AE 线（AE1、AE3、AE4）—— 事后分析，C 级、描述性（G s0/s1 测试集已多次复用；本批数字不进摘要，'
      '只用于写确认运行的登记方案与正文描述）。' + ('【冒烟：2 个单元，判定无意义，只验证代码路径与对账】' if smoke else ''))
    P(f'规格 PRECONF_ANALYSES_SPEC_2026-09-29.md sha256[:16]={sha16(SPEC)}（登记 {SPEC_HASH}）；'
      f'ae_frontier.py {sha16(os.path.abspath(__file__))}；step0B2.py {sha16(STEP0B2_PY)}（只 import，未改）；'
      f'step0B.py {sha16(STEP0B_PY)}（只 import，未改）；curves_B2.csv {sha16(CURVES_B2)}。')
    P(f'单元 {nU}（11 国 × 2 划分 × 2 重复）；τ ∈ [{TAUS[0]}, {TAUS[-1]}] 步长 0.05（{len(TAUS)} 点）；窗口步长 0.1；'
      f'共同可达 <{MIN_COV:.0%} 记缺失。国家值 = 该国可用单元平均；推断单位 = 国家，95% t CI（df = n−1）；9 国 = 去 RO、PT。')
    P('注意：零标签参照（LightGBM Z0、LR/MLP/RF 的 K0）与测试集只随 (国家, 划分) 变化，同一划分的两个重复完全相同；'
      '"44 个单元"中这些臂实际只有 22 个不同的值（RT200/PK200/HD200/S200 随重复变化）。')
    P('特征维度：AE1/AE3/AE4 的估计量本身就是 AE − S2 的配对差，"两特征国家值平均"不适用；ae、s2 分列处给出各自的水平'
      '（AE1 = 共同可达点上的平均 BA；AE3 = 召回；AE4 = 组内 OA/BA），差 = 水平之差。')
    P('')

    # ================================================================ 对账
    P('== 对账 ==')
    ok_all = True
    z = KRd[KRd.check == 'Z0_curve_vs_curves_B2']
    ok = bool(z.bitwise.all())
    ok_all &= ok
    P(f'(1) 本脚本独立重算的 LightGBM Z0 τ 曲线（阶段 0 P_T 测试行，逐类召回与分组用同一 PH）与 curves_B2.csv：{len(z)} 条曲线，'
      f'逐位相同 {int(z.bitwise.sum())}/{len(z)}（最大 |Δ| OA {z.max_abs_oa.max():.2e}，BA {z.max_abs_ba.max():.2e}）')
    ok = bool(B2Cd.bitwise.all())
    ok_all &= ok
    P(f'(2) 用 step0B2.run_unit 重算 Z0、PK200、RT200、HD200 曲线，与 curves_B2.csv：{len(B2Cd)} 条曲线（单元 × 特征 × 臂），'
      f'逐位相同 {int(B2Cd.bitwise.sum())}/{len(B2Cd)}（最大 |Δ| {B2Cd.max_abs.max():.2e}）')
    ml = KRd[KRd.ref == 'ML_rows']
    d_oa, d_ba = (ml.oa - ml.oa_ref).abs().max(), (ml.ba_ge30 - ml.ba_ref).abs().max()
    ok = d_oa < 1e-9 and d_ba < 1e-9
    ok_all &= ok
    P(f'(3) LR/MLP/RF 的 K0 与 S200 在 τ=0 的 OA、BA 与 guidance_G_ml rows（foreign_only k=0；srs k=200）：{len(ml)} 行，'
      f'最大 |Δ| OA {d_oa:.2e}，BA {d_ba:.2e} → {"通过" if ok else "不通过"}；'
      + '，'.join(f'{l} {(ml[ml.learner == l].oa - ml[ml.learner == l].oa_ref).abs().max():.1e}/'
                 f'{(ml[ml.learner == l].ba_ge30 - ml[ml.learner == l].ba_ref).abs().max():.1e}' for l in LEARNERS))
    ss = ml[ml.arm == 'S200']
    P(f'    S200 的 sel_hash 与 G（LightGBM）同一单元 SRS k=200 的 sel_hash 相同：{int(ss.same_sel_as_G_lgb.sum())}/{len(ss)}')
    # LR/MLP 旧网格子集与 partB/curves_B.csv 逐位比较
    CBo = pd.read_csv(CURVES_B, float_precision='round_trip')
    amap = {('lr', 'K0'): 'K0lr', ('mlp', 'K0'): 'K0mlp', ('lr', 'S200'): 'RT200lr', ('mlp', 'S200'): 'RT200mlp'}
    CBo = CBo[CBo.arm.isin(amap.values())].copy()
    CBo['ti'] = np.rint(CBo.tau * 100).astype(int)
    M = MLCd[MLCd.learner.isin(['lr', 'mlp'])].copy()
    M['arm_b'] = [amap[(l, a)] for l, a in zip(M.learner, M.arm)]
    M['ti'] = np.rint(M.tau * 100).astype(int)
    M = M[M.ti.isin(np.rint(B2.TAUS_OLD * 100).astype(int))]
    J = M.merge(CBo, left_on=UKEY + ['clf', 'arm_b', 'ti'], right_on=UKEY + ['clf', 'arm', 'ti'], suffixes=('', '_B'))
    same = np.array_equal(J.oa.to_numpy(), J.oa_B.to_numpy()) and np.array_equal(J.ba_ge30.to_numpy(), J.ba_ge30_B.to_numpy())
    ok_all &= same and len(J) == len(M)
    P(f'(4) LR/MLP 新网格曲线中属于旧网格（τ ∈ [−0.5, 2]）的点与 partB/curves_B.csv（K0lr、K0mlp、RT200lr、RT200mlp）：'
      f'{len(J)}/{len(M)} 行对上，OA、BA {"逐位相同" if same else "不同"}'
      f'（最大 |Δ| {max((J.oa - J.oa_B).abs().max(), (J.ba_ge30 - J.ba_ge30_B).abs().max()) if len(J) else np.nan:.2e}）')
    # AE1 Z0 与 AE–S2 整合评估 matched_oa_units.csv
    IU = pd.read_csv(INTEG_UNITS)
    z0 = A1d[(A1d.learner == 'lgb') & (A1d.arm == 'Z0')]
    res = []
    for my_an, my_w, col in (('min', 4, 'W4_min'), ('min', 8, 'W8_min'), ('s2', 4, 'W4_s2'), ('ae', 4, 'W4_ae')):
        x = z0[(z0.anchor == my_an) & (z0.width == my_w)].merge(IU[['unit', f'{col}_d', f'{col}_cov',
                                                                   f'{col}_ae_dominates', f'{col}_s2_dominates']], on='unit')
        dd = (x.D - x[f'{col}_d']).abs()
        miss_same = bool((x.D.isna() == x[f'{col}_d'].isna()).all())
        dom_same = bool(((x.ae_dom.fillna(-1) == x[f'{col}_ae_dominates'].map({True: 1.0, False: 0.0}).fillna(-1)) &
                         (x.s2_dom.fillna(-1) == x[f'{col}_s2_dominates'].map({True: 1.0, False: 0.0}).fillna(-1))).all())
        res.append(f'{col}：{len(x)} 单元，最大 |ΔD| {dd.max():.1e}，最大 |Δ覆盖| {(x["cov"] - x[f"{col}_cov"]).abs().max():.3f}，'
                   f'缺失标记{"相同" if miss_same else "不同"}，占优标记{"相同" if dom_same else "不同"}')
        ok_all &= bool(dd.max() < 1e-4) and miss_same and dom_same
    P('(5) AE1 LightGBM Z0 的 D 与 AE–S2 整合评估 matched_oa_units.csv（其窗口格点为 round(arange(锚点−w, 锚点, 0.1), 6)，'
      '本脚本为 锚点 + round(arange(−w, 0, 0.1), 1)，格点差 ≤ 5e-7）：' + '；'.join(res))
    # 国家层 2 位小数：LightGBM 各臂（该臂自身 τ=0 OA 的较小者为锚点，新网格）与 matched_oa_by_arm_report.txt。
    # 该报告与下面 (7) 的脚本用 round(arange(锚点−4, 锚点, 0.1), 6) 构造窗口格点；这里两种格点都算：
    # "整合格点"应与报告逐位（2 位小数）一致，"本格点"（锚点 + OFFS，规格口径）的差只来自窗口顶端格点 ≤5e-7 的平移。
    def integ_grid(anc):
        return np.round(np.arange(anc - 4, anc + 1e-9, 0.1), 6)

    def cmp2(rows_m, rows_i, ref):
        vm, vi = cvals(pd.DataFrame(rows_m), 'D'), cvals(pd.DataFrame(rows_i), 'D')
        ref = pd.Series(ref).reindex(vm.index)
        return float((vi.round(2) - ref).abs().max()), float((vm.round(2) - ref).abs().max()), \
            [t for t in vm.index if abs(round(vm[t], 2) - ref[t]) > 0.005 + 1e-9]

    arms_i = ['Z0', 'RT50', 'RT200', 'RT400', 'HD50', 'HD200', 'HD400']
    dicts = parse_dicts(INTEG_ARMS, r'逐国 d_matched：(\{[^}]*\})')
    res = []
    for arm in ('Z0', 'RT200', 'HD200'):
        rows_m, rows_i = [], []
        for u in A1d.unit.unique():
            cv = {f: CB2[(u, f, arm)] for f in FEATS}
            anc = min(cv[f][0][I_T0] for f in FEATS)
            rows_m.append(dict(unit=u, target=u[:2], **frontier_diff(cv['ae'], cv['s2'], anc + B.OFFS)))
            rows_i.append(dict(unit=u, target=u[:2], **frontier_diff(cv['ae'], cv['s2'], integ_grid(anc))))
        di, dm, bad = cmp2(rows_m, rows_i, dicts[arms_i.index(arm)])
        res.append(f'{arm} 整合格点 {di:.3f} / 本格点 {dm:.3f}' + (f'（本格点不一致的国家 {",".join(bad)}）' if bad else ''))
        ok_all &= di <= 0.0050001 or smoke
    P('(6) LightGBM 国家层 D（锚点 = 该臂自身 τ=0 OA 的较小者；Z0 即主口径锚点）与 matched_oa_by_arm_report.txt 逐国值（2 位小数）'
      '最大 |Δ|：' + '；'.join(res) + '。整合格点 = round(arange(锚点−4, 锚点, 0.1), 6)（该报告的构造）；本格点 = 锚点 + OFFS'
      '（规格/B2 口径）；两者只在窗口格点上差 ≤5e-7，个别单元在近乎竖直的曲线段上 BA 对此敏感。主口径 Z0 锚点下的 RT200/HD200 为新量。')
    # LR/MLP：旧网格上的 D 与 matched_oa_lr_mlp_out.txt（K0 用本学习器 K0 锚点；RT200 用自身锚点）
    arms_l = re.findall(r'^## (\S+)（', open(INTEG_ML, encoding='utf-8').read(), flags=re.M)
    dl = parse_dicts(INTEG_ML, r'逐国同OA ΔBA：(\{[^}]*\})')
    res = []
    MC = {(u, l, a, f): (g.oa.to_numpy(), g.ba_ge30.to_numpy())
          for (u, l, a, f), g in MLCd.sort_values('tau').groupby(['unit', 'learner', 'arm', 'clf'])}
    i0_old = int(np.flatnonzero(B2.TAUS_OLD == 0)[0])
    for lrn in ('lr', 'mlp'):
        for arm, name in (('K0', f'K0{lrn}'), ('S200', f'RT200{lrn}')):
            rows_m, rows_i = [], []
            for u in A1d.unit.unique():
                cv = {f: (MC[(u, lrn, arm, f)][0][IDX_OLD], MC[(u, lrn, arm, f)][1][IDX_OLD]) for f in FEATS}
                anc = min(cv[f][0][i0_old] for f in FEATS)   # K0：= 本学习器 K0 锚点；RT200：自身锚点（与该脚本相同）
                rows_m.append(dict(unit=u, target=u[:2], **frontier_diff(cv['ae'], cv['s2'], anc + B.OFFS)))
                rows_i.append(dict(unit=u, target=u[:2], **frontier_diff(cv['ae'], cv['s2'], integ_grid(anc))))
            di, dm, bad = cmp2(rows_m, rows_i, dl[arms_l.index(name)])
            res.append(f'{name} 整合格点 {di:.3f} / 本格点 {dm:.3f}' + (f'（本格点不一致的国家 {",".join(bad)}）' if bad else ''))
            ok_all &= di <= 0.0050001 or smoke
    P('(7) LR/MLP 在旧网格（τ ∈ [−0.5, 2]）上重算的国家层 D 与 matched_oa_lr_mlp_out.txt 逐国值（2 位小数）最大 |Δ|：'
      + '；'.join(res) + '（K0 锚点 = 本学习器 K0；RT200 锚点 = 自身，与该脚本相同；格点说明同 (6)）')
    # τ=0 的 AE−S2（国家均值）与 report_ML_rev3.txt 的 Δ_feat
    rep = open(ML_REPORT, encoding='utf-8').read().split('================')
    sec = {'ba': [s for s in rep if 'Δ_feat(L, k) = ba_ge30' in s], 'oa': [s for s in rep if 'Δ_feat(L, k) = oa' in s]}
    lname = {'lgb': 'LightGBM（G 原结果）', 'lr': 'Logistic 回归', 'mlp': 'MLP 集成', 'rf': '随机森林'}
    res, dmax_all, n_cmp = [], 0.0, 0
    for met in ('ba', 'oa'):
        body = rep[rep.index(sec[met][0]) + 1]
        for lrn in ('lgb', 'lr', 'mlp', 'rf'):
            # LightGBM 的 k=0 行是 G foreign_only（阶段 0 的 P0test i300，未经分组留出温度），不是 Z0（P_T），不对账
            pairs = [('RT200', 200)] if lrn == 'lgb' else [('K0', 0), ('S200', 200)]
            for arm, k in pairs:
                x = A1d[(A1d.learner == lrn) & (A1d.arm == arm) & (A1d.anchor == 'min') & (A1d.width == 4)].copy()
                x['d'] = x[f'{met}_t0_ae'] - x[f'{met}_t0_s2']
                v = cvals(x, 'd')
                for drop, tag in ((False, '11 国'), (True, '去 RO、PT')):
                    m = re.search(rf'{re.escape(lname[lrn])}\s+k={k}\s+{tag}\s+([+-]\d+\.\d+) pp', body)
                    if m and not smoke:
                        dmax_all = max(dmax_all, abs(round(sel9(v, drop).mean(), 3) - float(m.group(1))))
                        n_cmp += 1
    ok_all &= (dmax_all <= 0.0010001 and n_cmp == 28) or smoke
    P(f'(8) τ=0 的 AE−S2 国家均值（BA、OA；LR/MLP/RF 的 k=0 与 SRS k=200，LightGBM 的 SRS k=200；11 国与 9 国；共 {n_cmp} 个数）'
      f'与 report_ML_rev3.txt 的 Δ_feat：最大 |Δ| {dmax_all:.4f}（报告为 3 位小数）' + ('（冒烟跳过）' if smoke else '') +
      '；LightGBM k=0 不比：报告中的 k=0 是 G foreign_only（阶段 0 P0test），本项的零标签参照是 Z0（P_T）')
    a3 = KRd[KRd.check == 'AE3_point']
    a4 = KRd[KRd.check == 'AE4_share_weighted_oa']
    P(f'(9) AE3 取点：插值 OA − o_mid 最大 |Δ| {a3.oa_interp_minus_omid.abs().max():.1e}；由插值逐类召回算出的 BA − 上包络 BA 最大 |Δ| '
      f'{a3.ba_from_recalls_minus_envelope.abs().max():.1e}（{len(a3)} 个单元×特征）')
    P(f'(10) AE4：组占比和 − 1 最大 |Δ| {a4.share_sum_minus1.abs().max():.1e}；Σ 占比×组 OA − 总 OA：τ=0 最大 |Δ| {a4.t0_diff.abs().max():.1e}，'
      f'同 OA 点最大 |Δ| {a4.mid_diff.abs().max():.1e}')
    wsum = SZd.assign(w=SZd.share * SZd.d_oa_mid).groupby('unit').w.sum()
    P(f'     推论：同 OA 点上两特征总 OA 都等于 o_mid，故每个单元 Σ 占比 × 组内 OA 差 = 0：最大 |Σ| {wsum.abs().max():.1e}'
      f'（{len(wsum)} 个单元）。AE4 同 OA 点的组内 OA 差因此只反映组间的相对分布。')
    ok_all &= bool(a3.oa_interp_minus_omid.abs().max() < 1e-8 and a3.ba_from_recalls_minus_envelope.abs().max() < 1e-8 and
                   a4.t0_diff.abs().max() < 1e-8 and a4.mid_diff.abs().max() < 1e-8 and wsum.abs().max() < 1e-8)
    P(f'对账结论：{"全部通过" if ok_all else "有不通过项，见上"}')
    P('')

    # ================================================================ AE1
    country = []

    def sub(lrn, arm, an='min', w=4):
        return A1d[(A1d.learner == lrn) & (A1d.arm == arm) & (A1d.anchor == an) & (A1d.width == w)]

    def label(lrn, arm):
        return f'{LNAME[lrn]} {arm}'

    P('######## AE1 同 OA 下 AE−S2 前沿差 D（pp；主口径：W4，锚点 = min(OA★_ae, OA★_s2)，OA★ = 本学习器零标签参照在 τ=0 的 OA）########')
    P('  LightGBM 的零标签参照 = Z0（四个臂共用 Z0 锚点）；LR/MLP/RF 的零标签参照 = 本学习器 K0（K0 与 S200 共用）。')
    P('  D = 共同可达 o 上 BA_ae(o) − BA_s2(o) 的平均；水平 = 同一批 o 上各特征 BA 的平均；占优 = 共同可达点上处处 ≥。')
    for lrn, arm in SETS:
        x = sub(lrn, arm)
        cv = cvals(x, 'D')
        la, ls = cvals(x, 'lvl_ae'), cvals(x, 'lvl_s2')
        P(f'-- {label(lrn, arm)} --')
        P(f'  D(AE−S2)  {both_sets(cv)}')
        P(f'  水平 ae   9 国 {sel9(la, True).mean():.2f} | 11 国 {la.mean():.2f}；水平 s2 9 国 {sel9(ls, True).mean():.2f} | '
          f'11 国 {ls.mean():.2f}')
        P(f'  缺失单元 {int(x.missing.sum())}/{len(x)}；共同可达 <100% 的单元 {int((x["cov"] < 1).sum())}；'
          f'AE 全窗口占优 {int(np.nansum(x.ae_dom))}/{len(x)}，S2 全窗口占优 {int(np.nansum(x.s2_dom))}/{len(x)}'
          + (f'（缺失 {int(x.missing.sum())}）' if x.missing.any() else ''))
        for t in COUNTRIES:
            if t in cv.index:
                country.append(dict(target=t, learner=lrn, arm=arm, quantity='D_W4_min', value=cv.get(t, np.nan),
                                    lvl_ae=la.get(t, np.nan), lvl_s2=ls.get(t, np.nan)))
    P('')
    P('-- 另报：窗口宽度 2 / 4 / 8（锚点 min）--')
    for lrn, arm in SETS:
        for w in WIDTHS:
            x = sub(lrn, arm, 'min', w)
            cv = cvals(x, 'D')
            P(f'  {label(lrn, arm):14s} W{w}  {both_sets(cv)}   （缺失 {int(x.missing.sum())}；AE 占优 {int(np.nansum(x.ae_dom))}，'
              f'S2 占优 {int(np.nansum(x.s2_dom))}）')
            if w != 4:  # W4 已在主口径中记录
                for t in cv.index:
                    country.append(dict(target=t, learner=lrn, arm=arm, quantity=f'D_W{w}_min', value=cv[t]))
    P('-- 另报：锚点取 OA★_ae 或 OA★_s2（W4）--')
    for lrn, arm in SETS:
        for an in ('ae', 's2'):
            x = sub(lrn, arm, an, 4)
            cv = cvals(x, 'D')
            P(f'  {label(lrn, arm):14s} 锚点 OA★_{an}  {both_sets(cv)}   （缺失 {int(x.missing.sum())}/{len(x)}，'
              f'缺失所在国家 {",".join(sorted(x[x.missing].target.unique())) or "—"}）')
            for t in cv.index:
                country.append(dict(target=t, learner=lrn, arm=arm, quantity=f'D_W4_{an}', value=cv[t]))
    P('-- 另报：可部署点 τ = 0.5 与原始工作点 τ = 0 的 AE−S2（BA、OA）；水平为 11 国均值 ae / s2 --')
    for lrn, arm in SETS:
        x = sub(lrn, arm).copy()
        for tt, tag in (('t05', 'τ=0.5'), ('t0', 'τ=0  ')):
            for met, mn in (('ba', 'BA'), ('oa', 'OA')):
                x['d'] = x[f'{met}_{tt}_ae'] - x[f'{met}_{tt}_s2']
                cv = cvals(x, 'd')
                la, ls = cvals(x, f'{met}_{tt}_ae'), cvals(x, f'{met}_{tt}_s2')
                P(f'  {label(lrn, arm):14s} {tag} Δ{mn}  {both_sets(cv)}   （水平 {la.mean():.2f} / {ls.mean():.2f}）')
                for t in cv.index:
                    country.append(dict(target=t, learner=lrn, arm=arm, quantity=f'd{mn}_{tt}', value=cv[t],
                                        lvl_ae=la[t], lvl_s2=ls[t]))
    P('-- 另报：44 个单元中 AE / S2 在整个窗口占优的单元数（W4，锚点 min；"严格" = 共同可达点上处处 >）--')
    for lrn, arm in SETS:
        x = sub(lrn, arm)
        st_a = int(((x.n_ae_gt == x.n_co) & ~x.missing).sum())
        st_s = int(((x.n_s2_gt == x.n_co) & ~x.missing).sum())
        ae_c = sorted(x[x.ae_dom == 1].target.unique())
        s2_c = sorted(x[x.s2_dom == 1].target.unique())
        P(f'  {label(lrn, arm):14s} AE 占优 {int(np.nansum(x.ae_dom))}/{len(x)}（严格 {st_a}），S2 占优 {int(np.nansum(x.s2_dom))}/{len(x)}'
          f'（严格 {st_s}），其余（交叉）{int((~x.missing).sum() - np.nansum(x.ae_dom) - np.nansum(x.s2_dom))}；'
          f'S2 占优所在国家 {",".join(s2_c) or "—"}；不属于 AE 占优的国家 {",".join(sorted(set(x.target) - set(ae_c))) or "—"}')
    P('-- 另报：argmax 并列比例（LR/MLP/RF；测试行中最大分数被 ≥2 个类同时取到的比例；τ=0 即原始概率；并列按原样取第一个类）--')
    for lrn in LEARNERS:
        for arm in ML_ARMS:
            x = TId[(TId.learner == lrn) & (TId.arm == arm)]
            for f in FEATS:
                xf = x[x.clf == f]
                P(f'  {LNAME[lrn]:4s} {arm:4s} {f}: τ=0 并列 均值 {xf.tie_t0.mean():.4f}（最大 {xf.tie_t0.max():.4f}）；'
                  f'81 个 τ 平均 {xf.tie_mean.mean():.4f}；τ≠0 时 τ·log π̂ 的偏移打破大部分并列；概率恰为 0 的格占比 '
                  f'{xf.n_zero_prob_frac.mean():.3f}（log → −inf，任何 τ 下不被预测）')
    P('-- 方案外（对账口径，描述）：锚点 = 该臂自身 τ=0 OA 的较小者（W4）--')
    for lrn, arm in SETS:
        if arm == REF_ARM[lrn]:
            continue
        x = sub(lrn, arm, 'own', 4)
        P(f'  {label(lrn, arm):14s} {both_sets(cvals(x, "D"))}   （缺失 {int(x.missing.sum())}）')
    P('')
    CT = pd.DataFrame(country)
    P('-- AE1 国家值 D（W4，锚点 min；pp）--')
    piv = CT[CT.quantity == 'D_W4_min'].pivot(index='target', columns=['learner', 'arm'], values='value').reindex(COUNTRIES)
    hdr = [f'{LNAME[l]}:{a}' for l, a in SETS]
    P('  国家 ' + ' '.join(f'{h:>12s}' for h in hdr))
    for t in COUNTRIES:
        P(f'  {t}   ' + ' '.join(f'{piv.loc[t, (l, a)] if (l, a) in piv.columns else np.nan:+12.2f}' for l, a in SETS))
    P('')
    # 跨学习器描述
    P('-- 描述：零标签参照的 D（W4 min）跨学习器 --')
    for lrn in ('lgb',) + tuple(LEARNERS):
        s9 = B.t_summary(sel9(cvals(sub(lrn, REF_ARM[lrn]), 'D'), True))
        s11 = B.t_summary(cvals(sub(lrn, REF_ARM[lrn]), 'D'))
        P(f'  {LNAME[lrn]:8s} {REF_ARM[lrn]}：9 国 CI 下界 {s9["lo95"]:+.2f}（{"> 0" if s9["lo95"] > 0 else "≤ 0"}），'
          f'11 国 CI 下界 {s11["lo95"]:+.2f}（{"> 0" if s11["lo95"] > 0 else "≤ 0"}）')
    P('')

    # ================================================================ AE3
    P('######## AE3 按类拆分（LightGBM Z0；同 OA 点 o_mid = min(OA★_ae, OA★_s2) − 2；召回 pp）########')
    tm = CLd.drop_duplicates('unit')
    P(f'  同 OA 点所在 τ（单元 × 特征，中位 [P10, P90]）：ae {tm.tau_mid_ae.median():+.2f} [{tm.tau_mid_ae.quantile(.1):+.2f}, '
      f'{tm.tau_mid_ae.quantile(.9):+.2f}]，s2 {tm.tau_mid_s2.median():+.2f} [{tm.tau_mid_s2.quantile(.1):+.2f}, '
      f'{tm.tau_mid_s2.quantile(.9):+.2f}]；o_mid 不可达的单元×特征 {int(tm.tau_mid_ae.isna().sum() + tm.tau_mid_s2.isna().sum())}')
    P('  每类：只计该单元测试集中该类 ≥ 30 个地块的单元，先平均到国家；报国家均值、95% t CI、>0 国家数/国家数；水平为 11 国（可用国家）均值 ae / s2。')
    for c in sorted(CLd.cls.unique()):
        x = CLd[(CLd.cls == c) & CLd.ge30]
        name = CLd[CLd.cls == c].cls_name.iloc[0]
        if not len(x):
            P(f'  [{c}] {name}: 无单元满足 ≥30')
            continue
        v = cvals(x, 'd_mid')
        v0 = cvals(x, 'd_t0')
        la, ls = cvals(x, 'rec_mid_ae'), cvals(x, 'rec_mid_s2')
        P(f'  [{c}] {name}（国家数 11 国口径 {v.notna().sum()}，9 国口径 {sel9(v, True).notna().sum()}；'
          f'国家：{",".join(v.dropna().index)}）')
        P(f'      同 OA 点 AE−S2  {both_sets(v)}；水平 {la.mean():.1f} / {ls.mean():.1f}')
        P(f'      τ=0     AE−S2  {both_sets(v0)}；水平 {cvals(x, "rec_t0_ae").mean():.1f} / {cvals(x, "rec_t0_s2").mean():.1f}')
        for t in v.index:
            country.append(dict(target=t, learner='lgb', arm='Z0', quantity=f'AE3_rec_mid_{name}', value=v[t],
                                lvl_ae=la[t], lvl_s2=ls[t]))
            country.append(dict(target=t, learner='lgb', arm='Z0', quantity=f'AE3_rec_t0_{name}', value=v0[t]))
    P('  -- 国家 × 类：同 OA 点召回 AE−S2（pp；空 = 该国无单元该类 ≥ 30）--')
    cls_list = sorted(CLd.cls.unique())
    names = {c: CLd[CLd.cls == c].cls_name.iloc[0] for c in cls_list}
    P('  国家 ' + ' '.join(f'{names[c][:10]:>10s}' for c in cls_list))
    for t in COUNTRIES:
        vals = []
        for c in cls_list:
            x = CLd[(CLd.cls == c) & CLd.ge30 & (CLd.target == t)]
            vals.append(f'{x.d_mid.mean():+10.1f}' if len(x) and x.d_mid.notna().any() else f'{"":>10s}')
        P(f'  {t}   ' + ' '.join(vals))
    P('')

    # ================================================================ AE4
    P('######## AE4 地块大小（核面积 π·30² ≈ 2,827 m²；组 = 面积/核面积；LightGBM Z0；同 OA 点同 AE3）########')
    P('  组占比（国家值 = 单元平均；11 国均值 | 9 国均值；逐国见 CSV）')
    for j in range(NG):
        x = SZd[SZd.grp == j]
        v = cvals(x, 'share')
        P(f'  {SIZE_LAB[j]:6s} 占比 11 国 {v.mean():.3f} | 9 国 {sel9(v, True).mean():.3f}；逐国 '
          + ' '.join(f'{t}:{v[t]:.2f}' for t in v.index))
    P('  组内 AE−S2（OA 与 BA；BA 只计该单元该组 ≥ 30 个地块的类，<2 类记缺失）；水平为 11 国（可用国家）均值 ae / s2。')
    P('  注：同 OA 点上两特征的总 OA 相等（= o_mid），各组 OA 差按组占比加权后和为 0（见对账 (10) 推论），'
      '所以"同 OA 点的组内 OA 差"描述的是 AE 与 S2 把正确判读分配在哪些大小的地块上，不是总体高低。')
    for j in range(NG):
        x = SZd[SZd.grp == j]
        P(f'  -- 组 {SIZE_LAB[j]}（BA 缺失单元 {int(x.ba_missing.sum())}/{len(x)}；每单元满足 ≥30 的类数中位数 '
          f'{x.n_cls_ge30.median():.0f}；组内地块数中位数 {x.n_grp.median():.0f}，<30 的单元 {int((x.n_grp < 30).sum())}'
          f'（{",".join(sorted(x[x.n_grp < 30].target.unique())) or "—"}）--')
        for q, qn in (('oa_t0', 'OA τ=0    '), ('oa_mid', 'OA 同OA点 '), ('ba_t0', 'BA τ=0    '), ('ba_mid', 'BA 同OA点 ')):
            v = cvals(x, f'd_{q}')
            la, ls = cvals(x, f'{q}_ae'), cvals(x, f'{q}_s2')
            P(f'    {qn} AE−S2 {both_sets(v)}；水平 {la.mean():.2f} / {ls.mean():.2f}')
            for t in v.index:
                country.append(dict(target=t, learner='lgb', arm='Z0', quantity=f'AE4_{q}_g{j}', value=v[t],
                                    lvl_ae=la[t], lvl_s2=ls[t]))
        xs = x[x.n_grp >= G.MIN_N]
        P(f'    方案外：OA 同OA点 只计组内 ≥30 个地块的单元  {both_sets(cvals(xs, "d_oa_mid"))}')
    P('  -- 方案外（描述）：最大组与最小组之差 [d(≥4) − d(<0.5)]，单元内配对后平均到国家（两组都非缺失的单元）--')
    W = SZd.pivot_table(index=UKEY, columns='grp', values=['d_oa_mid', 'd_ba_mid', 'd_oa_t0', 'd_ba_t0'])
    for q in ('d_oa_t0', 'd_oa_mid', 'd_ba_t0', 'd_ba_mid'):
        if (q, 4) in W.columns and (q, 0) in W.columns:
            U = (W[(q, 4)] - W[(q, 0)]).rename('x').reset_index()
            P(f'    {q:9s} {both_sets(cvals(U, "x"))}')
    P('  -- 国家 × 组：同 OA 点组内 BA 的 AE−S2（pp）--')
    P('  国家 ' + ' '.join(f'{s:>8s}' for s in SIZE_LAB))
    for t in COUNTRIES:
        vals = []
        for j in range(NG):
            x = SZd[(SZd.grp == j) & (SZd.target == t)]
            vals.append(f'{x.d_ba_mid.mean():+8.2f}' if x.d_ba_mid.notna().any() else f'{"—":>8s}')
        P(f'  {t}   ' + ' '.join(vals))
    P('  -- 国家 × 组：同 OA 点组内 OA 的 AE−S2（pp）--')
    for t in COUNTRIES:
        vals = []
        for j in range(NG):
            x = SZd[(SZd.grp == j) & (SZd.target == t)]
            vals.append(f'{x.d_oa_mid.mean():+8.2f}' if x.d_oa_mid.notna().any() else f'{"—":>8s}')
        P(f'  {t}   ' + ' '.join(vals))
    P('')
    P('######## 读法 ########')
    P('  规格对 AE1、AE3、AE4 没有事先写定的判定阈值（AE2 与 L1 才有）；以下全部是按上文数字自动生成的描述，不是判定。')
    pos9 = [f'{label(l, a)}' for l, a in SETS if B.t_summary(sel9(cvals(sub(l, a), 'D'), True))['lo95'] > 0]
    pos11 = [f'{label(l, a)}' for l, a in SETS if B.t_summary(cvals(sub(l, a), 'D'))['lo95'] > 0]
    P(f'  AE1：W4、锚点 min 下，D 的 95% CI 下界 > 0 的臂 9 国 {len(pos9)}/{len(SETS)}（{", ".join(pos9) or "—"}），'
      f'11 国 {len(pos11)}/{len(SETS)}。')
    wtxt = []
    for l, a in SETS:
        m = [sel9(cvals(sub(l, a, 'min', w), 'D'), True).mean() for w in WIDTHS]
        wtxt.append(f'{label(l, a)} {m[0]:+.2f}/{m[1]:+.2f}/{m[2]:+.2f}')
    P('  AE1 窗口宽度 2/4/8 的 9 国均值：' + '；'.join(wtxt) + '。')
    cpos, cneg = [], []
    for c in sorted(CLd.cls.unique()):
        x = CLd[(CLd.cls == c) & CLd.ge30]
        if not len(x):
            continue
        st = B.t_summary(sel9(cvals(x, 'd_mid'), True))
        nm = CLd[CLd.cls == c].cls_name.iloc[0]
        if st['n'] >= 2 and st['lo95'] > 0:
            cpos.append(f'{nm} {st["mean"]:+.1f}')
        elif st['n'] >= 2 and st['hi95'] < 0:
            cneg.append(f'{nm} {st["mean"]:+.1f}')
    P(f'  AE3：同 OA 点召回 AE−S2 的 9 国 CI 下界 > 0 的类：{", ".join(cpos) or "—"}；CI 上界 < 0 的类：{", ".join(cneg) or "—"}'
      f'（同 OA 点 AE 的 τ 中位数 {tm.tau_mid_ae.median():+.2f}，S2 {tm.tau_mid_s2.median():+.2f}）。')
    g_ba = [sel9(cvals(SZd[SZd.grp == j], 'd_ba_mid'), True).mean() for j in range(NG)]
    g_oa = [sel9(cvals(SZd[SZd.grp == j], 'd_oa_mid'), True).mean() for j in range(NG)]
    g_ba0 = [sel9(cvals(SZd[SZd.grp == j], 'd_ba_t0'), True).mean() for j in range(NG)]
    P('  AE4（9 国均值，组 ' + '/'.join(SIZE_LAB) + '）：同 OA 点组内 BA 差 ' + '/'.join(f'{v:+.2f}' for v in g_ba) +
      '；同 OA 点组内 OA 差 ' + '/'.join(f'{v:+.2f}' for v in g_oa) + '；τ=0 组内 BA 差 ' + '/'.join(f'{v:+.2f}' for v in g_ba0) + '。')
    P('  AE4 的用途是描述 AE 的优势是否随地块变大而改变，用来检验"30 m 核混入邻田"这一说法：该说法预期 AE−S2 在小地块组更小或为负、'
      '在大地块组更大；上面的组序列可直接对照这一预期（描述，C 级；组与国家的地块大小构成混杂，PT、SI、AT 小地块占比高）。')
    return lines, pd.DataFrame(country)


if __name__ == '__main__':
    main()
