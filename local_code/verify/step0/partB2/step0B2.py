"""
第 0 步补遗 B2：同 OA 下标签增益的分解（先验信息 vs 其余）。
规格：label_expansion/STEP0_ADDENDUM_B2_2026-09-29.md（sha256[:16]=0c4b31ebc681b90f）；补遗未改动的部分沿用
STEP0_SPEC_2026-09-29.md B 部分（sha256[:16]=14c8de7365278bac）与其主实现 partB/step0B.py。
事后分析（补遗写于看到第 0 步 B 结果之后、做任何 B2 计算之前），C 级、描述性；不训练任何分类器，只读 G 的已存概率与选样。
候选标签只用于构造 p̂（PK-k）与拟合 BCTS 头（HD-k）；测试标签只用于评估；PKo 用测试先验，是补遗定义的上界臂，只作描述。

口径复用（import step0B，不改其文件）：
  τ 族      step0B.tau_preds；网格以参数传入（调用时临时替换其模块全局 TAUS，返回前恢复）
  π̂        该臂测试集平均预测概率（exp(Z − 行最大) 归一化后取均值，与 step0B 相同，不用标签）
  log 取法  step0B.log_or_neginf（精确 log，概率恰为 0 → −inf；即 step0B 主实现的取法）
  OA / BA   step0B.Scorer（ba_ge30 的类集合 = 测试集中 ≥ G.MIN_N 个地块的类）
  包络 / FA step0B.envelope、step0B.fa_of；OA★ = Z0 在 τ=0 的 OA；W = [OA★−4, OA★] 步长 0.1（step0B.OFFS）；
            可达率 <100% 标部分可达，<50% FA 记缺失
  BCTS 头   step0B.fit_bcts / step0B.weights（从 calib_diag.py 按 D1 源码哈希取出；λ=0.1，类均衡权重）
  汇总      step0B.country_vals、step0B.t_summary、step0B.f95（国家值 = 可用单元平均；both = 两特征国家值平均）
补遗改动：
  τ 网格    −2.0..2.0 步长 0.05（81 点），所有臂一律用。另在旧网格（−0.5..2.0，51 点，是新网格的子集，直接取对应 τ 列）
            上重算，只用于与 partB 逐位对账，以及"对照"一节的新旧并列。
  零标签参照 主参照 Z0（P_T 测试行的 τ 族，不含 EM 点）；Z0+（Z0 与 EM 修正曲线的上包络）另列。
  PK-k      y_sel = 与 RT-k 同一批 G SRS(k) 选样（rows_t1：method=srs、cond=none、对应 k 与 clf 的 sel_hash）的标签；
            p̂ = check3_priors.laplace_prior(y_sel, 10, alpha=1)；P_T 测试行经 check3_priors.reweight(P, p̂, π_train)
            （π_train 同 STEP0_SPEC A1 = 外国池按 class_weights_from 加权的类分布）后走 τ 族。
  PKo       p̂ = 测试集真实类别比例，其余同 PK-k。
估计量（逐单元 × 特征计算，再平均到国家）：
  A_k = FA(PK-k) − FA(Z0)；A_o = FA(PKo) − FA(Z0)；R_k = FA(RT-k) − FA(PK-k)；ΔFA(RT-k) = FA(RT-k) − FA(Z0) = A_k + R_k；
  A_k 占 ΔFA(RT-k) 的比例 = 国家均值之比（只报，不推断）；H_k = FA(HD-k) − FA(PK-k)；
  对照：新网格 ΔFA(RT-k)、ΔFA(HD-k)、FA(RT-k) − FA(HD-k)，与旧网格（partB）并列；各臂可达率。

用法：python -B step0B2.py --smoke   # AT_s0_r0、RO_s1_r1 两个单元，输出到 partB2/smoke/，做对账
      python -B step0B2.py           # 全量 44 个单元
"""
import os
import sys
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_v] = '2'
sys.dont_write_bytecode = True  # 不在项目目录或 partB/ 写 __pycache__
import hashlib
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PARTB = os.path.join(os.path.dirname(HERE), 'partB')
sys.path.insert(0, PARTB)
import step0B as B  # noqa: E402  导入时：chdir 到项目目录（只读）、导入 guidance_G、按 D1 源码哈希取出 fit_bcts / weights
from check3_priors import em_prior, laplace_prior, reweight  # noqa: E402

G = B.G
ADDENDUM = os.path.join(B.L, 'STEP0_ADDENDUM_B2_2026-09-29.md')
ADD_HASH = '0c4b31ebc681b90f'
STEP0B_PY = os.path.join(PARTB, 'step0B.py')
FB_CSV = os.path.join(PARTB, 'fa_rows.csv')
CB_CSV = os.path.join(PARTB, 'curves_B.csv')

C = B.C
FEATS = B.FEATS
COUNTRIES = B.COUNTRIES
DROP = B.DROP
KS = B.KS
OFFS = B.OFFS
MIN_COV = B.MIN_COV
TAUS_NEW = np.round(np.arange(-2.0, 2.0 + 1e-9, 0.05), 2)   # 补遗网格，81 点
TAUS_OLD = B.TAUS.copy()                                      # 第 0 步 B 网格，51 点
IDX_OLD = np.array([int(np.flatnonzero(np.abs(TAUS_NEW - t) < 1e-9)[0]) for t in TAUS_OLD])
assert np.array_equal(TAUS_NEW[IDX_OLD], TAUS_OLD) and len(TAUS_NEW) == 81
I_T0 = int(np.flatnonzero(TAUS_NEW == 0.0)[0])
GRIDS = {'new': np.arange(len(TAUS_NEW)), 'old': IDX_OLD}
TAU_OF = {'new': TAUS_NEW, 'old': TAUS_OLD}

RT = [f'RT{k}' for k in KS]
HD = [f'HD{k}' for k in KS]
PK = [f'PK{k}' for k in KS]
CURVE_ARMS = ['Z0', 'Z0EM'] + PK + ['PKo'] + RT + HD
ENVS = {'Z0': ['Z0'], 'Z0EM': ['Z0EM'], 'Z0+': ['Z0', 'Z0EM'], **{a: [a] for a in PK + ['PKo'] + RT + HD}}
OLD_CMP = ['Z0', 'Z0EM', 'Z0+'] + RT + HD      # 旧网格上与 partB/fa_rows.csv 逐位对账的包络
UKEY = ['target', 'split', 'rep', 'unit', 'clf']


def sha16(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()[:16]


def tau_preds_grid(Z, pihat, taus):
    """复用 step0B.tau_preds 的同一份源码；τ 网格以参数传入（临时替换 step0B 的模块全局 TAUS，返回前恢复）。单线程调用。"""
    saved = B.TAUS
    B.TAUS = taus
    try:
        return B.tau_preds(Z, pihat)
    finally:
        B.TAUS = saved


# ---------------------------------------------------------------- 一个单元
def run_unit(D, S, GR, tgt, split, rep, log=print):
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
    pi_train = np.bincount(yF, weights=cw[yF], minlength=C) / cw[yF].sum()  # = G pi_train_of([], 0)（STEP0_SPEC A1）
    pool_pos = G.pool_positions(S, split, rep)
    yP = y[allT[pool_pos]]
    pi_test = np.bincount(yT, minlength=C) / nT  # 只用于 PKo（补遗定义的上界臂）与描述量 prior_L1

    curve_rows, fa_rows, chk_rows, env_rows = [], [], [], []
    for f in FEATS:
        cell = dict(target=tgt, split=split, rep=rep, unit=unit, clf=f)
        P_all = S[f'P_T_{f}']
        PT = P_all[n_cand:].astype(np.float64)
        PP = P_all[pool_pos].astype(np.float64)
        Z, meta = {}, {}
        # ---- Z0 与 Z0EM（与 step0B 相同）
        P0 = B.norm64(PT)
        Z['Z0'] = B.log_or_neginf(P0)
        pi_em, it_em, conv_em, _ = em_prior(P0, pi_train)
        Q, nfb = reweight(P0, pi_em, pi_train)
        Z['Z0EM'] = B.log_or_neginf(Q)
        meta['Z0EM'] = meta['Z0+'] = dict(em_iter=it_em, em_conv=conv_em, em_fallback=nfb)
        # ---- RT-k、HD-k（与 step0B 相同）与 PK-k（同一批 SRS(k) 选样的标签）
        ZTc = np.log(np.clip(PT, 1e-12, None))
        ZPc = np.log(np.clip(PP, 1e-12, None))
        indep = {}
        for k in KS:
            g = GR[(GR.method == 'srs') & (GR.target == tgt) & (GR.split == split) & (GR.rep == rep) &
                   (GR.clf == f) & (GR.k == k)]
            assert len(g) == 1, (unit, f, k, len(g))
            g = g.iloc[0]
            Pr = B.norm64(np.load(os.path.join(B.R_G, 'probs', f'{unit}_{f}_k{k}_{g.sel_hash}.npz'))['i300'])
            Z[f'RT{k}'] = B.log_or_neginf(Pr)
            zsel = np.load(os.path.join(B.R_G, 'sel', f'{unit}_none_srs_k{k}.npz'))
            sp = zsel['sel_pos']
            assert G.sel_hash(allT[pool_pos][sp]) == g.sel_hash == G.sel_hash(zsel['sel']), (unit, f, k)
            ys = yP[sp]
            s, b = B.fit_bcts(ZPc[sp], ys, B.weights(ys, np.ones(len(ys)), 'bal'), B.LAM)
            Z[f'HD{k}'] = s * ZTc + b
            meta[f'HD{k}'] = dict(s=s, bmax=float(np.abs(b).max()))
            p_hat = laplace_prior(ys, C, alpha=1)
            Qk, nfb_k = reweight(P0, p_hat, pi_train)
            Z[f'PK{k}'] = B.log_or_neginf(Qk)
            meta[f'PK{k}'] = dict(n_sel=len(ys), n_classes_sel=int(len(np.unique(ys))), pk_fallback=nfb_k,
                                  prior_L1=float(np.abs(p_hat - pi_test).sum()),
                                  prior_L1_train=float(np.abs(pi_train - pi_test).sum()))
            # 对账用的独立路径：全局索引取标签、手写 Laplace、guidance_G.prior_correct + argmax + guidance_G._scores
            ys2 = y[zsel['sel']]
            ph2 = (np.bincount(ys2, minlength=C) + 1.0) / (len(ys2) + C)
            o_i, b_i = G._scores(yT, G.prior_correct(P0, ph2, pi_train).argmax(1), big)[:2]
            indep[f'PK{k}'] = (o_i, b_i, float(np.abs(ph2 - p_hat).max()))
            chk_rows.append(dict(cell, arm=f'RT{k}', ref='G_rows', oa_ref=g.oa, ba_ref=g.ba_ge30))
        Qo, nfb_o = reweight(P0, pi_test, pi_train)
        Z['PKo'] = B.log_or_neginf(Qo)
        meta['PKo'] = dict(pk_fallback=nfb_o, prior_L1=0.0, prior_L1_train=float(np.abs(pi_train - pi_test).sum()))
        o_i, b_i = G._scores(yT, G.prior_correct(P0, pi_test, pi_train).argmax(1), big)[:2]
        indep['PKo'] = (o_i, b_i, 0.0)

        # ---- 每条曲线在新网格上计算；旧网格 = 新网格的对应 τ 列
        CV = {}
        for arm in CURVE_ARMS:
            Za = Z[arm]
            Pa = np.exp(Za - Za.max(1, keepdims=True))
            Pa /= Pa.sum(1, keepdims=True)
            pihat = Pa.mean(0)
            PH = tau_preds_grid(Za, pihat, TAUS_NEW)
            CV[arm] = sc.curve(PH)
            oa, ba = CV[arm]
            for i, t in enumerate(TAUS_NEW):
                curve_rows.append(dict(cell, arm=arm, tau=t, oa=oa[i], ba_ge30=ba[i]))

        # ---- τ = 0 对账记录
        for k in KS:
            r = [r for r in chk_rows if r['unit'] == unit and r['clf'] == f and r['arm'] == f'RT{k}'][0]
            r['oa'], r['ba_ge30'] = CV[f'RT{k}'][0][I_T0], CV[f'RT{k}'][1][I_T0]
            chk_rows.append(dict(cell, arm=f'HD{k}', ref='D1_srs_G_check', oa=CV[f'HD{k}'][0][I_T0],
                                 ba_ge30=CV[f'HD{k}'][1][I_T0]))
        chk_rows.append(dict(cell, arm='Z0', ref='D1_base_PT', oa=CV['Z0'][0][I_T0], ba_ge30=CV['Z0'][1][I_T0]))
        for arm, (o_i, b_i, dp) in indep.items():
            chk_rows.append(dict(cell, arm=arm, ref='indep_PT_laplace_reweight', oa=CV[arm][0][I_T0],
                                 ba_ge30=CV[arm][1][I_T0], oa_ref=o_i, ba_ref=b_i, prior_maxdiff=dp))

        # ---- FA：两种网格；锚点均为 Z0 在 τ=0 的 OA
        oa_star = CV['Z0'][0][I_T0]
        wgrid = oa_star + OFFS
        for gname, gi in GRIDS.items():
            taus = TAU_OF[gname]
            cvg = {a: (CV[a][0][gi], CV[a][1][gi]) for a in CURVE_ARMS}
            for e, comps in ENVS.items():
                best, cid, seg, lam = B.envelope([cvg[a] for a in comps], wgrid)
                fa, cov = B.fa_of(best)
                allo = np.concatenate([cvg[a][0] for a in comps])
                unr = ~np.isfinite(best)
                row = dict(cell, grid=gname, arm=e, OA_star=oa_star, FA=fa, reach=cov,
                           partial=bool(MIN_COV <= cov < 1.0), fa_missing=bool(cov < MIN_COV),
                           n_miss_top=int((unr & (wgrid > allo.max())).sum()),
                           n_miss_bot=int((unr & (wgrid < allo.min())).sum()),
                           oa_min=float(allo.min()), oa_max=float(allo.max()))
                if len(comps) == 1:
                    oa, ba = cvg[comps[0]]
                    j0 = int(np.flatnonzero(taus == 0.0)[0])
                    row.update(oa_t0=oa[j0], ba_t0=ba[j0], tau_oamax=float(taus[int(oa.argmax())]),
                               oamax_rising_at_lo=bool(oa[0] >= oa.max() and oa[0] > oa[1]))
                row.update(meta.get(e, {}))
                fa_rows.append(row)
                if gname == 'new':
                    env_rows += [dict(cell, arm=e, off=OFFS[q], o=wgrid[q], ba=best[q]) for q in range(len(wgrid))]
    log(f'{unit} 完成')
    return curve_rows, fa_rows, chk_rows, env_rows


# ---------------------------------------------------------------- 主程序
def main():
    smoke = '--smoke' in sys.argv
    out = os.path.join(HERE, 'smoke') if smoke else HERE
    os.makedirs(out, exist_ok=True)
    assert sha16(ADDENDUM) == ADD_HASH, '补遗文件哈希与登记值不一致'
    assert sha16(B.SPEC) == B.SPEC_HASH, '第 0 步规格哈希与登记值不一致'
    t0 = time.time()
    print(f'step0B2.py sha256[:16]={sha16(os.path.abspath(__file__))}；step0B.py sha256[:16]={sha16(STEP0B_PY)}', flush=True)
    D = G.GData()
    GR = B.load_G_rows()
    units = ([('AT', 0, 0), ('RO', 1, 1)] if smoke else
             [(t, s, r) for t in COUNTRIES for s in (0, 1) for r in (0, 1)])
    CR, FR, KR, ER = [], [], [], []
    cache = {}
    for tgt, split, rep in units:
        if (tgt, split) not in cache:
            cache.clear()
            cache[(tgt, split)] = np.load(os.path.join(B.R_G, 'stage0', f'{tgt}_s{split}.npz'))
        a, b, c, d = run_unit(D, cache[(tgt, split)], GR, tgt, split, rep,
                              log=lambda s: print(f'[{time.time() - t0:.0f}s] {s}', flush=True))
        CR += a
        FR += b
        KR += c
        ER += d
    CRd, FRd, KRd, ERd = map(pd.DataFrame, (CR, FR, KR, ER))
    FRd.to_csv(os.path.join(out, 'fa_rows_B2.csv'), index=False)
    CRd.to_csv(os.path.join(out, 'curves_B2.csv'), index=False)
    ERd.to_csv(os.path.join(out, 'envelope_B2.csv'), index=False)
    KRd.to_csv(os.path.join(out, 'check_rows_B2.csv'), index=False)
    lines = summarize(CRd, FRd, KRd, ERd, smoke)
    open(os.path.join(out, 'summary_B2.txt'), 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(f'总耗时 {time.time() - t0:.0f}s')


# ---------------------------------------------------------------- 汇总
def wide(FRd, grid):
    return FRd[FRd.grid == grid].set_index(UKEY + ['arm'])['FA'].unstack('arm').reset_index()


def udiff(W, a, b):
    U = W[UKEY].copy()
    U['d'] = W[a] - W[b]
    return U


def sel9(v, drop):
    return v.drop([c for c in DROP if c in v.index]) if drop else v


def summarize(CRd, FRd, KRd, ERd, smoke):
    lines = []
    P = lines.append
    nU = FRd.unit.nunique()
    P('第 0 步补遗 B2：同 OA 下标签增益的分解（先验信息 vs 其余）—— 事后分析，C 级、描述性（G s0/s1 测试集已多次复用；'
      '数字不进论文摘要）。' + ('【冒烟：2 个单元】' if smoke else ''))
    P(f'补遗 STEP0_ADDENDUM_B2_2026-09-29.md sha256[:16]={sha16(ADDENDUM)}（登记 {ADD_HASH}）；'
      f'STEP0_SPEC_2026-09-29.md sha256[:16]={sha16(B.SPEC)}（登记 {B.SPEC_HASH}）；'
      f'step0B.py sha256[:16]={sha16(STEP0B_PY)}（只 import，未改）；step0B2.py sha256[:16]={sha16(os.path.abspath(__file__))}；'
      f'calib_diag.py sha256[:16]={sha16(B.CALIB)}；fit_bcts/weights 源码哈希 {B.D1_FUNC_HASH}（与 D1 相同）。')
    P(f'单元 {nU} × 特征 {FRd.clf.nunique()}；只用 LightGBM；k ∈ {KS}。τ 网格（新）{TAUS_NEW[0]}..{TAUS_NEW[-1]} 步长 0.05'
      f'（{len(TAUS_NEW)} 点），所有臂一律使用；旧网格 {TAUS_OLD[0]}..{TAUS_OLD[-1]}（{len(TAUS_OLD)} 点）只用于对账与并列。'
      f'W = [OA★−4, OA★] 步长 0.1（{len(OFFS)} 点），OA★ = Z0 在 τ=0 的 OA；可达率 <{MIN_COV:.0%} 记 FA 缺失。')
    P('臂：Z0 = P_T 测试行（主参照）；Z0+ = Z0 与 EM 修正曲线的上包络（另列）；PK-k = P_T 测试行经 reweight(P, p̂, π_train)，'
      'p̂ = laplace_prior(与 RT-k 同一批 G SRS(k) 选样的标签, 10, α=1)；PKo = 同上但 p̂ = 测试集真实类别比例（上界臂，只作描述）；'
      'RT-k = G SRS(k) 重训 LightGBM 的 i300；HD-k = 同一批选样上的 BCTS 头（λ=0.1，类均衡）。')
    P('国家值 = 该国可用单元（2 划分 × 2 重复）的平均；both = AE、S2 两特征国家值的平均（主）；推断单位 = 国家，'
      '95% CI 为 t 区间（df = n−1）；9 国 = 去掉 RO、PT（主），11 国为次。')
    P('')

    # ================================================================ 对账
    P('== 对账 ==')
    ok_all = True
    ind = KRd[KRd.ref == 'indep_PT_laplace_reweight']
    d_oa = (ind.oa - ind.oa_ref).abs().max()
    d_ba = (ind.ba_ge30 - ind.ba_ref).abs().max()
    ok = d_oa < 1e-9 and d_ba < 1e-9 and ind.prior_maxdiff.max() < 1e-15
    ok_all &= ok
    P(f'(1) PK-k / PKo 在 τ=0 与独立路径（全局索引取 y_sel、手写 (n_c+1)/(k+10)、guidance_G.prior_correct(P_T, p̂, π_train)'
      f'.argmax + guidance_G._scores）：{len(ind)} 行，最大 |Δ| OA {d_oa:.2e}，BA {d_ba:.2e}；p̂ 最大差 '
      f'{ind.prior_maxdiff.max():.2e} → {"通过" if ok else "不通过"}')
    FB = pd.read_csv(FB_CSV, float_precision='round_trip')
    FB = FB[(FB.anchor == 'Z0') & FB.arm.isin(OLD_CMP)]
    FO = FRd[(FRd.grid == 'old') & FRd.arm.isin(OLD_CMP)]
    J = FO.merge(FB, on=UKEY + ['arm'], suffixes=('', '_B'), how='inner')
    exp_n = len(FO)
    res = []
    for col in ('OA_star', 'FA', 'reach', 'oa_min', 'oa_max'):
        a, b = J[col].to_numpy(float), J[f'{col}_B'].to_numpy(float)
        same = np.array_equal(a, b, equal_nan=True)
        res.append(f'{col} {"逐位相同" if same else "不同"}（最大 |Δ| {np.nanmax(np.abs(a - b)) if len(a) else np.nan:.2e}）')
        ok_all &= same
    same_flags = (J.partial.astype(bool).to_numpy() == J.partial_B.astype(bool).to_numpy()).all() and \
                 (J.fa_missing.astype(bool).to_numpy() == J.fa_missing_B.astype(bool).to_numpy()).all()
    ok_all &= bool(same_flags) and len(J) == exp_n
    P(f'(2) 旧网格上 {",".join(OLD_CMP)} 的包络与 partB/fa_rows.csv（anchor=Z0）逐行比：{len(J)}/{exp_n} 行对上；'
      + '；'.join(res) + f'；partial / fa_missing 标记{"相同" if same_flags else "不同"}')
    for arm in RT + HD:
        x = J[J.arm == arm]
        P(f'    {arm:6s} {len(x):3d} 行：FA 逐位相同 {int(np.sum((x.FA == x.FA_B) | (x.FA.isna() & x.FA_B.isna())))}/{len(x)}，'
          f'reach 逐位相同 {int((x.reach == x.reach_B).sum())}/{len(x)}')
    CB = pd.read_csv(CB_CSV, float_precision='round_trip')
    CB = CB[CB.arm.isin(['Z0', 'Z0EM'] + RT + HD)].copy()
    CB['ti'] = np.rint(CB.tau * 100).astype(int)
    CN = CRd[CRd.arm.isin(['Z0', 'Z0EM'] + RT + HD)].copy()
    CN['ti'] = np.rint(CN.tau * 100).astype(int)
    JC = CN.merge(CB, on=UKEY + ['arm', 'ti'], suffixes=('', '_B'), how='inner')
    n_exp = int(CN.ti.isin(np.rint(TAUS_OLD * 100).astype(int)).sum())
    same_c = np.array_equal(JC.oa.to_numpy(), JC.oa_B.to_numpy()) and np.array_equal(JC.ba_ge30.to_numpy(),
                                                                                     JC.ba_ge30_B.to_numpy())
    ok_all &= same_c and len(JC) == n_exp
    P(f'(3) 曲线点（Z0、Z0EM、RT-k、HD-k，新网格中属于旧网格的 τ）与 partB/curves_B.csv：{len(JC)}/{n_exp} 行对上，'
      f'OA、BA {"逐位相同" if same_c else "不同"}（最大 |Δ| OA {(JC.oa - JC.oa_B).abs().max():.2e}，BA {(JC.ba_ge30 - JC.ba_ge30_B).abs().max():.2e}）')
    rt = KRd[KRd.ref == 'G_rows']
    P(f'(4) RT-k τ=0 与 G rows（srs，i300）：{len(rt)} 行，最大 |Δ| OA {(rt.oa - rt.oa_ref).abs().max():.2e}，'
      f'BA {(rt.ba_ge30 - rt.ba_ref).abs().max():.2e}；sel_hash 三方一致（断言通过）')
    d1 = pd.read_csv(B.D1_ROWS)
    key = ['unit', 'clf']
    hd = KRd[KRd.ref == 'D1_srs_G_check'].copy()
    hd['k'] = hd.arm.str[2:].astype(int)
    jj = hd.set_index(key + ['k'])[['oa', 'ba_ge30']].join(
        d1[d1.design == 'srs_G_check'].set_index(key + ['k'])[['oa', 'ba_ge30']], rsuffix='_d1', how='inner')
    z0 = KRd[KRd.ref == 'D1_base_PT'].set_index(key)[['oa', 'ba_ge30']].join(
        d1[d1.design == 'base_PT'].set_index(key)[['oa', 'ba_ge30']], rsuffix='_d1', how='inner')
    P(f'(5) HD-k τ=0 与 D1 srs_G_check：{len(jj)} 行，最大 |Δ| OA {(jj.oa - jj.oa_d1).abs().max():.2e}，'
      f'BA {(jj.ba_ge30 - jj.ba_ge30_d1).abs().max():.2e}；Z0 τ=0 与 D1 base_PT：{len(z0)} 行，最大 |Δ| OA '
      f'{(z0.oa - z0.oa_d1).abs().max():.2e}，BA {(z0.ba_ge30 - z0.ba_ge30_d1).abs().max():.2e}')
    # 不变量：新网格曲线段集是旧网格的超集 → 在两者都可达的 o 上，新网格包络 BA ≥ 旧网格
    viol, npts, viol_rows = 0, 0, []
    for (u, f), cg in CRd.groupby(['unit', 'clf']):
        cvs = {a: (g_.oa.to_numpy(), g_.ba_ge30.to_numpy()) for a, g_ in cg.groupby('arm')}
        oa_star = cvs['Z0'][0][I_T0]
        wg = oa_star + OFFS
        for e, comps in ENVS.items():
            bn, *_ = B.envelope([cvs[a] for a in comps], wg)
            bo, *_ = B.envelope([(cvs[a][0][IDX_OLD], cvs[a][1][IDX_OLD]) for a in comps], wg)
            m = np.isfinite(bn) & np.isfinite(bo)
            npts += int(m.sum())
            v = int((bn[m] < bo[m] - 1e-12).sum())
            viol += v
            if v:
                viol_rows.append((u, f, e, v))
            # 新网格包络点与 envelope_B2.csv 一致
            sub = ERd[(ERd.unit == u) & (ERd.clf == f) & (ERd.arm == e)].ba.to_numpy()
            assert np.array_equal(sub, bn, equal_nan=True), (u, f, e)
    ok_all &= viol == 0
    P(f'(6) 不变量：在新旧网格都可达的 {npts} 个 (单元, 特征, 包络, o) 点上，新网格包络 BA ≥ 旧网格包络 BA：违反 {viol} 个'
      + (f' {viol_rows[:5]}' if viol else ''))
    em = FRd[(FRd.grid == 'new') & (FRd.arm == 'Z0EM')]
    pk = FRd[(FRd.grid == 'new') & FRd.arm.isin(PK + ['PKo'])]
    P(f'(7) EM：收敛 {int(em.em_conv.sum())}/{len(em)}，迭代数中位数 {em.em_iter.median():.0f}，回退行合计 '
      f'{int(em.em_fallback.sum())}；PK 臂 reweight 回退行合计 {int(pk.pk_fallback.sum())}')
    nz = CRd.groupby('arm').size()
    P(f'(8) 曲线行数：每臂 {sorted(set(int(v) for v in nz.values))}（应 = {nU} × 2 × {len(TAUS_NEW)} = {nU * 2 * len(TAUS_NEW)}）')
    P(f'对账结论：{"全部通过" if ok_all else "有不通过项，见上"}')
    P('')

    WN, WO = wide(FRd, 'new'), wide(FRd, 'old')
    FN = FRd[FRd.grid == 'new']
    FO_ = FRd[FRd.grid == 'old']

    # ================================================================ 可达率
    P('######## 可达率（新网格；锚点 = Z0 的 OA★；每臂 单元×特征）########')
    P('  臂      特征  全可达 部分 缺失 | 不可达点：窗口顶端 / 底端 | OA 最大在 τ=−2.0 且仍上升 | τ=0 的 OA − OA★ 中位 [P10, P90] | '
      '缺失所在国家 | 旧网格：全 / 部分 / 缺失 / OA 最大在 τ=−0.5 且仍上升')
    for arm in ['Z0', 'Z0EM', 'Z0+'] + PK + ['PKo'] + RT + HD:
        for f in FEATS + ['all']:
            x = FN[(FN.arm == arm) & ((FN.clf == f) if f != 'all' else True)]
            xo = FO_[(FO_.arm == arm) & ((FO_.clf == f) if f != 'all' else True)]
            rel = x.oa_t0 - x.OA_star
            miss = sorted(x[x.fa_missing].target.unique())
            reltxt = (f'{rel.median():+6.2f} [{rel.quantile(.1):+6.2f}, {rel.quantile(.9):+6.2f}]'
                      if rel.notna().any() else '       —（包络）       ')
            rising = f'{int(x.oamax_rising_at_lo.sum()):3d}/{len(x)}' if x.oamax_rising_at_lo.notna().any() else '  —   '
            P(f'  {arm:7s} {f:4s}  {int((x.reach >= 1).sum()):3d}  {int(x.partial.sum()):3d}  {int(x.fa_missing.sum()):3d} | '
              f'{int(x.n_miss_top.sum()):4d} / {int(x.n_miss_bot.sum()):4d} | {rising} | {reltxt} | '
              f'{",".join(miss) if miss else "—"} | {int((xo.reach >= 1).sum())} / {int(xo.partial.sum())} / '
              f'{int(xo.fa_missing.sum())} / ' + (f'{int(xo.oamax_rising_at_lo.sum())}' if xo.oamax_rising_at_lo.notna().any() else '—'))
    P('-- 新网格上非全可达（部分可达 + 缺失）的原因，单元×特征计数：')
    P('   截断 = 缺窗口顶端且 OA 最大点在 τ=−2.0、仍上升；天花板 = 缺窗口顶端且 OA 最大点在网格内部（该臂 OA 上限 < OA★）；'
      '缺底 = 缺窗口底端（该臂在 τ ≤ 2.0 的 OA 下限 > OA★−4）')
    for arm in ['Z0EM'] + PK + ['PKo'] + RT + HD:
        x = FN[(FN.arm == arm) & (FN.reach < 1)]
        top = x.n_miss_top > 0
        trunc = top & x.oamax_rising_at_lo.astype(bool)
        P(f'  {arm:6s} 非全可达 {len(x):3d}：截断 {int(trunc.sum())}，天花板 {int((top & ~trunc).sum())}，'
          f'缺底 {int((x.n_miss_bot > 0).sum())}；国家 {",".join(sorted(x.target.unique())) or "—"}')
    P('')

    # ================================================================ 估计量
    def block(title, U, col='d', show=True):
        cv = B.country_vals(U, col)
        if show:
            P(title)
            nmiss = {f: int(U[U.clf == f][col].isna().sum()) for f in FEATS}
            for f in ('both', 'ae', 's2'):
                for drop in (True, False):
                    P(f'  {f:4s} {"9 国" if drop else "11 国":5s} {B.f95(B.t_summary(sel9(cv[f], drop)))}' +
                      (f'   （11 国缺失单元 {nmiss[f]}/{int((U.clf == f).sum())}）' if f != 'both' and not drop else ''))
        return cv

    EST = {}
    P('######## 估计量（新网格；主参照 Z0；pp）########')
    P('-- 先验信息的价值 A_k = FA(PK-k) − FA(Z0)；A_o = FA(PKo) − FA(Z0) --')
    for k in KS:
        EST[f'A{k}'] = block(f'- A_{k}', udiff(WN, f'PK{k}', 'Z0'))
    EST['Ao'] = block('- A_o（PKo：测试集真实先验，上界臂）', udiff(WN, 'PKo', 'Z0'))
    P('-- 扣掉先验信息后重训多出的部分 R_k = FA(RT-k) − FA(PK-k) --')
    for k in KS:
        EST[f'R{k}'] = block(f'- R_{k}', udiff(WN, f'RT{k}', f'PK{k}'))
    P('-- ΔFA(RT-k) = FA(RT-k) − FA(Z0) = A_k + R_k --')
    for k in KS:
        EST[f'dRT{k}'] = block(f'- ΔFA(RT{k})', udiff(WN, f'RT{k}', 'Z0'))
    P('-- 扣掉先验后决策头多出的部分 H_k = FA(HD-k) − FA(PK-k) --')
    for k in KS:
        EST[f'H{k}'] = block(f'- H_{k}', udiff(WN, f'HD{k}', f'PK{k}'))
    P('')

    # ---- 占比：只报国家均值之比；在 FA(Z0)、FA(PK-k)、FA(RT-k) 都非缺失的单元上算，使 A + R = ΔFA 在国家层严格成立
    P('-- A_k 占 ΔFA(RT-k) 的比例（国家均值之比，不推断；单元集合 = Z0、PK-k、RT-k 三者 FA 都非缺失）--')
    P('  k    特征 国家   mean(A_k)  mean(R_k)  mean(ΔFA)   A_k/ΔFA   | 参照：mean(A_o)/mean(ΔFA)  | 排除单元')
    SHARE = {}
    for k in KS:
        m = WN[['Z0', f'PK{k}', f'RT{k}']].notna().all(axis=1)
        UA, UR, UD, UO = (udiff(WN[m], f'PK{k}', 'Z0'), udiff(WN[m], f'RT{k}', f'PK{k}'), udiff(WN[m], f'RT{k}', 'Z0'),
                          udiff(WN[m & WN.PKo.notna()], 'PKo', 'Z0'))
        cA, cR, cD, cO = (B.country_vals(U, 'd') for U in (UA, UR, UD, UO))
        for f in ('both', 'ae', 's2'):
            for drop in (True, False):
                a, r, d, o = (sel9(c[f], drop).dropna() for c in (cA, cR, cD, cO))
                idx = a.index.intersection(d.index)
                ma, mr, md = a[idx].mean(), r.reindex(idx).mean(), d[idx].mean()
                mo = o.reindex(idx).mean()
                SHARE[(k, f, drop)] = ma / md
                P(f'  {k:<4d} {f:4s} {"9 国" if drop else "11 国":5s} {ma:+9.2f}  {mr:+9.2f}  {md:+9.2f}   {ma / md:8.1%}   | '
                  f'{mo / md:8.1%}                  | {int((~m).sum())}')
    P('')

    # ================================================================ 对照
    P('######## 对照：新网格重算的 ΔFA(RT-k)、ΔFA(HD-k)、FA(RT-k) − FA(HD-k)，与第 0 步 B 旧网格并列（pp）########')
    P('  列：新网格·参照 Z0（本补遗主口径） | 新网格·参照 Z0+ | 旧网格·参照 Z0 | 旧网格·参照 Z0+（= partB summary_B 已报的口径）')
    CMP = {}
    for name, (a, b) in {**{f'ΔFA(RT{k})': (f'RT{k}', None) for k in KS}, **{f'ΔFA(HD{k})': (f'HD{k}', None) for k in KS},
                         **{f'FA(RT{k})−FA(HD{k})': (f'RT{k}', f'HD{k}') for k in KS}}.items():
        P(f'-- {name} --')
        if b is None:
            cols = [(WN, 'Z0'), (WN, 'Z0+'), (WO, 'Z0'), (WO, 'Z0+')]
        else:
            cols = [(WN, b), (WO, b)]
        cvs = [B.country_vals(udiff(W_, a, ref), 'd') for W_, ref in cols]
        CMP[name] = cvs
        for f in ('both', 'ae', 's2'):
            for drop in (True, False):
                txt = ' | '.join(B.f95(B.t_summary(sel9(cv[f], drop))) for cv in cvs)
                P(f'  {f:4s} {"9 国" if drop else "11 国":5s} {txt}')
    P('  （RT − HD 与参照无关，只有新 | 旧两列。）')
    P('-- 描述：FA(Z0+) − FA(Z0)（EM 对零标签包络的贡献），新网格 | 旧网格 --')
    cvs = [B.country_vals(udiff(W_, 'Z0+', 'Z0'), 'd') for W_ in (WN, WO)]
    for f in ('both', 'ae', 's2'):
        for drop in (True, False):
            P(f'  {f:4s} {"9 国" if drop else "11 国":5s} ' + ' | '.join(B.f95(B.t_summary(sel9(cv[f], drop))) for cv in cvs))
    P('-- 描述：各臂 FA 的网格效应 FA(新) − FA(旧)（单元 × 特征层，两网格都非缺失；均值 / 最小 / 最大 / 非零个数）--')
    for arm in ['Z0', 'Z0+'] + PK + ['PKo'] + RT + HD:
        dd = (WN[arm] - WO[arm]).dropna()
        P(f'  {arm:6s} n={len(dd):3d}  {dd.mean():+.3f} / {dd.min():+.3f} / {dd.max():+.3f}  非零 {int((dd.abs() > 1e-12).sum())}')
    P('')

    # ================================================================ 国家值
    P('######## 国家值（both = 两特征平均；新网格；参照 Z0；pp）########')
    cols = ['A50', 'A200', 'A400', 'Ao', 'R50', 'R200', 'R400', 'H50', 'H200', 'H400', 'dRT200']
    P('  国家 ' + ' '.join(f'{c:>8s}' for c in cols))
    for t in COUNTRIES:
        P(f'  {t}   ' + ' '.join(f'{EST[c]["both"].get(t, np.nan):+8.2f}' for c in cols))
    P('')

    # ================================================================ 方案外 / 描述
    P('######## 方案外（描述）：共同可达口径（只在两包络都可达的 o 上平均 BA 差；共同可达 <50% 记缺失）########')
    EW = ERd.set_index(UKEY + ['off', 'arm'])['ba'].unstack('arm')
    for name, (a, b) in {**{f'A_{k}': (f'PK{k}', 'Z0') for k in KS}, 'A_o': ('PKo', 'Z0'),
                         **{f'R_{k}': (f'RT{k}', f'PK{k}') for k in KS},
                         **{f'H_{k}': (f'HD{k}', f'PK{k}') for k in KS}}.items():
        m = EW[a].notna() & EW[b].notna()
        g = (EW[a] - EW[b]).where(m).groupby(UKEY)
        U = pd.DataFrame({'d': g.mean(), 'cov': m.groupby(UKEY).mean()}).reset_index()
        U.loc[U['cov'] < MIN_COV, 'd'] = np.nan
        cv = B.country_vals(U, 'd')
        P(f'  {name:5s} both 9 国 {B.f95(B.t_summary(sel9(cv["both"], True)))} | 11 国 {B.f95(B.t_summary(cv["both"]))}'
          f'   （共同可达 <100% 的单元×特征 {int((U["cov"] < 1).sum())}）')
    P('-- 描述：FA(RT-k) − FA(PKo)（重训相对"测试真实先验"上界臂）；A_k / A_o（k 个标签实现的先验价值占上界的比例，国家均值之比）--')
    for k in KS:
        cv = B.country_vals(udiff(WN, f'RT{k}', 'PKo'), 'd')
        m = WN[['Z0', f'PK{k}', 'PKo']].notna().all(axis=1)
        ca = B.country_vals(udiff(WN[m], f'PK{k}', 'Z0'), 'd')
        co = B.country_vals(udiff(WN[m], 'PKo', 'Z0'), 'd')
        r9 = sel9(ca['both'], True).mean() / sel9(co['both'], True).mean()
        r11 = ca['both'].dropna().mean() / co['both'].dropna().mean()
        P(f'  RT{k}−PKo both 9 国 {B.f95(B.t_summary(sel9(cv["both"], True)))} | 11 国 {B.f95(B.t_summary(cv["both"]))}；'
          f'A_{k}/A_o 9 国 {r9:.1%} | 11 国 {r11:.1%}（PKo 缺失单元已排除）')
    P('-- 描述：原始工作点 τ=0 的差（PK-k − Z0；PKo − Z0），both --')
    T0 = FN[FN.arm.isin(['Z0'] + PK + ['PKo'])].set_index(UKEY + ['arm'])[['oa_t0', 'ba_t0']].unstack('arm')
    for arm in PK + ['PKo']:
        for met, mn in (('oa_t0', 'OA'), ('ba_t0', 'BA')):
            U = T0[met][[arm, 'Z0']].reset_index()
            U['d'] = U[arm] - U['Z0']
            cv = B.country_vals(U, 'd')
            P(f'  {arm:5s} {mn}  9 国 {B.f95(B.t_summary(sel9(cv["both"], True)))} | 11 国 {B.f95(B.t_summary(cv["both"]))}')
    P('-- 描述：p̂ 与测试先验的 L1（单元 × 特征均值；9 国 | 11 国），参照 π_train 的 L1 --')
    for arm in PK:
        x = FN[FN.arm == arm]
        x9 = x[~x.target.isin(DROP)]
        P(f'  {arm:5s} ‖p̂ − π_test‖₁ {x9.prior_L1.mean():.3f} | {x.prior_L1.mean():.3f}；'
          f'‖π_train − π_test‖₁ {x9.prior_L1_train.mean():.3f} | {x.prior_L1_train.mean():.3f}；'
          f'选样类数中位数 {x.n_classes_sel.median():.0f}')
    P('')

    # ================================================================ 读法
    P('######## 读法（补遗事先写定；适用对象：k = 200、9 国、both = 两特征国家值平均）########')
    sR = B.t_summary(sel9(EST['R200']['both'], True))
    sA = B.t_summary(sel9(EST['A200']['both'], True))
    sD = B.t_summary(sel9(EST['dRT200']['both'], True))
    sh = SHARE[(200, 'both', True)]
    if not np.isfinite(sR['lo95']):
        reading, rtxt = 0, '无法判定（国家数不足）'
    elif sR['lo95'] > 0:
        reading, rtxt = 1, '第一条：R_200 的 95% CI 下界 > 0 —— 扣掉先验信息后，重训仍把前沿外推；可写成"标签补上了一部分域残差"（C 级）'
    elif sR['lo95'] <= 0 <= sR['hi95'] and sh >= 0.70:
        reading, rtxt = 2, ('第二条：R_200 的 95% CI 覆盖 0，且 A_200 占 ΔFA(RT200) ≥ 70% —— 同 OA 下的增益主要来自类别先验信息，'
                            '"标签主要提供类别比例"')
    else:
        reading, rtxt = 3, '第三条（其余情况）：不确定，两者都有贡献'
        if sR['hi95'] < 0:
            rtxt += '（注：R_200 的 95% CI 上界 < 0，即扣掉先验后重训低于 PK；读法未单列此情形，按字面归入第三条）'
        elif sh < 0.70:
            rtxt += f'（R_200 的 CI 覆盖 0，但 A_200 占比 {sh:.1%} < 70%）'
    P(f'  R_200 = {B.f95(sR)}')
    P(f'  A_200 = {B.f95(sA)}')
    P(f'  ΔFA(RT200) = {B.f95(sD)}（参照 Z0，新网格）')
    P(f'  A_200 / ΔFA(RT200)（国家均值之比）= {sh:.1%}')
    P(f'  → 适用读法：{rtxt}。')
    if smoke:
        P('  （冒烟只有 2 个单元，上面的判定无意义，只验证代码路径。）')
    P('  分特征（描述，不参与判定）：')
    for f in ('ae', 's2'):
        s_ = B.t_summary(sel9(EST['R200'][f], True))
        P(f'    {f}: R_200 {B.f95(s_)}；A_200 占比 {SHARE[(200, f, True)]:.1%}')
    s11 = B.t_summary(EST['R200']['both'])
    P(f'  11 国（次，描述）：R_200 {B.f95(s11)}；A_200 占比 {SHARE[(200, "both", False)]:.1%}')
    P('')
    P(f'READING_K200_9C_BOTH={reading}')
    return lines


if __name__ == '__main__':
    main()
