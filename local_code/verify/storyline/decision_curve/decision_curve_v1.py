"""
rev4 第 D1 节：决策层的标签效率曲线（只读 G 阶段 0 概率与 G 行；不训练任何分类器）。

规格：label_expansion/ANALYSIS_PLAN_guidance_rev4.md（sha256[:16]=7e90ccd5b1c3828f，已冻结）。
- 底座：G 阶段 0 外国全量 LightGBM，温度校准后的 S['P_T_{f}']，f ∈ {ae, s2}；候选池行与测试行来自同一模型。
- 标签子样本：guidance_G.pool_positions(S, split, rep) 给出的候选池内简单随机抽 k 个，k ∈ {50,100,200,400,1000}，
  每个 k 5 个重复，种子 = int(md5(f'D1_{tgt}_{split}_{rep}_{k}_{r}').hexdigest()[:8], 16)，
  抽样器 np.random.default_rng(seed).choice(N, k, replace=False)（与 G 的惯例相同）；另加"全池"一次。
- 决策头：calib_diag.fit_bcts（λ = 0.1）+ calib_diag.weights(mode='bal')（类均衡）。两个函数用 ast 从
  calib_diag.py 原样取出执行（该文件有顶层循环，不能直接 import），保证实现逐字相同。
- 指标：测试集 ba_ge30（类集合 = 测试集中 ≥ MIN_N=30 个地块的类）与 OA，用 guidance_G._scores。
- 对照：G 行 SRS(k) − foreign_only（同单元、同 clf），k ∈ {50,100,200,400}。
附加（非规格要求，只作核对 / 敏感性，报告中单列）：
- 'srs_G_check'：用 G 实际的 SRS 选样拟合同一个头，与 calib_diag 的 calib_rows.csv（srs_G, bal_l0.1）逐单元对比，核对实现。
- 'fullpool_lam1e-4'：全池 + λ=1e-4（calib_diag 'allpool bal_full' 的设置），看 λ=0.1 对全池上界的压缩。
输出（本目录）：decision_curve_rows.csv、decision_curve_summary.txt、decision_curve_country.csv、fig_decision_curve.png
事后分析，测试集已被多次复用 → C 级、描述性。
"""
import os
for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[v] = '2'
import ast
import hashlib
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize

L = '/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops/label_expansion'
sys.path.insert(0, L)
os.chdir(L)
import guidance_G as G  # noqa: E402
import analyze_G as AG  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
R_G = '/Users/perng_mini/AlphaCrops_local/results/guidance_G'
assert os.path.realpath(G.OUT) == os.path.realpath(R_G), G.OUT  # 只读 guidance_G，不碰 guidance_G_ml
CALIB = '/Users/perng_mini/AlphaCrops_local/verify/pivot_eval/prior_shift/calib_diag.py'
CALIB_ROWS = '/Users/perng_mini/AlphaCrops_local/verify/pivot_eval/prior_shift/calib_rows.csv'
PLAN = os.path.join(L, 'ANALYSIS_PLAN_guidance_rev4.md')
C = 10
KS = [50, 100, 200, 400, 1000]
KS_RET = [50, 100, 200, 400]
NREP = 5
LAM = 0.1
FEATS = ['ae', 's2']
COUNTRIES = 'AT,BE,DE,EE,LT,LV,PT,RO,SE,SI,SK'.split(',')
DROP = ['RO', 'PT']


def sha16(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()[:16]


# ---------------------------------------------------------------- 从 calib_diag.py 原样取出 fit_bcts 与 weights
def load_calib_funcs():
    src = open(CALIB, encoding='utf-8').read()
    tree = ast.parse(src)
    ns = dict(np=np, minimize=minimize, C=C)
    got = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in ('fit_bcts', 'weights'):
            code = ast.get_source_segment(src, node)
            exec(compile(code, CALIB, 'exec'), ns)
            got[node.name] = hashlib.sha256(code.encode()).hexdigest()[:16]
        if isinstance(node, ast.Assign) and any(getattr(t, 'id', None) == 'LAM0' for t in node.targets):
            assert ast.literal_eval(node.value) == LAM, 'calib_diag.LAM0 与 rev4 的 λ=0.1 不一致'
        if isinstance(node, ast.Assign) and any(getattr(t, 'id', None) == 'C' for t in node.targets):
            assert ast.literal_eval(node.value) == C
    assert set(got) == {'fit_bcts', 'weights'}
    return ns['fit_bcts'], ns['weights'], got


fit_bcts, weights, FUNC_HASH = load_calib_funcs()


def d1_seed(tgt, split, rep, k, r):
    return int(hashlib.md5(f'D1_{tgt}_{split}_{rep}_{k}_{r}'.encode()).hexdigest()[:8], 16)


def head_eval(ZP_sel, ys, ZT, yT, big, lam):
    s, b = fit_bcts(ZP_sel, ys, weights(ys, np.ones(len(ys)), 'bal'), lam)
    oa, ba, _, _ = G._scores(yT, (s * ZT + b).argmax(1), big)
    return dict(oa=oa, ba_ge30=ba, s=s, **{f'b{c}': b[c] for c in range(C)})


# ---------------------------------------------------------------- 逐单元计算
def compute_rows():
    D = G.GData()
    y = np.asarray(D.y)
    rows, base_rows = [], []
    t0 = time.time()
    for tgt in COUNTRIES:
        for split in (0, 1):
            S = np.load(os.path.join(R_G, 'stage0', f'{tgt}_s{split}.npz'))
            n_cand = len(S['cand'])
            allT, test = S['allT'], S['test']
            assert np.array_equal(allT[n_cand:], test)
            yT = y[test]
            big = np.flatnonzero(np.bincount(yT, minlength=C) >= G.MIN_N)
            P_all = {f: S[f'P_T_{f}'] for f in FEATS}
            P0 = {f: S[f'P0test_{f}_i300'] for f in FEATS}
            for rep in (0, 1):
                unit = f'{tgt}_s{split}_r{rep}'
                pool_pos = G.pool_positions(S, split, rep)
                yP = y[allT[pool_pos]]
                N = len(pool_pos)
                picks = {(k, r): np.sort(np.random.default_rng(d1_seed(tgt, split, rep, k, r)).choice(N, k, replace=False))
                         for k in KS for r in range(NREP)}
                gsel = {k: np.load(os.path.join(R_G, 'sel', f'{unit}_none_srs_k{k}.npz'))['sel_pos'] for k in KS_RET}
                for f in FEATS:
                    PP = P_all[f][pool_pos].astype(np.float64)
                    PT = P_all[f][n_cand:].astype(np.float64)
                    ZP, ZT = np.log(np.clip(PP, 1e-12, None)), np.log(np.clip(PT, 1e-12, None))
                    base = dict(target=tgt, split=split, rep=rep, unit=unit, clf=f, n_pool=N, n_test=len(yT),
                                n_big=len(big))
                    # 两种底座
                    ph_T = PT.argmax(1)
                    oa0, ba0, _, _ = G._scores(yT, ph_T, big)
                    Q = P0[f].astype(np.float64)
                    Q /= Q.sum(1, keepdims=True)
                    ph_0 = Q.argmax(1)
                    oa0g, ba0g, _, _ = G._scores(yT, ph_0, big)
                    rows.append(dict(base, design='base_PT', k=0, r=-1, seed=-1, lam=np.nan, oa=oa0, ba_ge30=ba0))
                    rows.append(dict(base, design='base_P0test_recomp', k=0, r=-1, seed=-1, lam=np.nan, oa=oa0g,
                                     ba_ge30=ba0g))
                    base_rows.append(dict(base, n_argmax_diff=int((ph_T != ph_0).sum()),
                                          n_argmax_diff_bigtrue=int(((ph_T != ph_0) & np.isin(yT, big)).sum())))
                    # D1 主曲线
                    for k in KS:
                        for r in range(NREP):
                            sp = picks[(k, r)]
                            m = head_eval(ZP[sp], yP[sp], ZT, yT, big, LAM)
                            rows.append(dict(base, design='srs_D1', k=k, r=r, seed=d1_seed(tgt, split, rep, k, r),
                                             lam=LAM, n_classes_sel=len(np.unique(yP[sp])), **m))
                    # 全池（规格：λ=0.1）与敏感性（λ=1e-4）
                    for lam, dsg in ((LAM, 'fullpool'), (1e-4, 'fullpool_lam1e-4')):
                        m = head_eval(ZP, yP, ZT, yT, big, lam)
                        rows.append(dict(base, design=dsg, k=N, r=-1, seed=-1, lam=lam,
                                         n_classes_sel=len(np.unique(yP)), **m))
                    # 核对：G 实际 SRS 选样 + 同一头（应复现 calib_rows.csv 的 srs_G / bal_l0.1）
                    for k in KS_RET:
                        sp = gsel[k]
                        m = head_eval(ZP[sp], yP[sp], ZT, yT, big, LAM)
                        rows.append(dict(base, design='srs_G_check', k=k, r=-1, seed=-1, lam=LAM,
                                         n_classes_sel=len(np.unique(yP[sp])), **m))
            print(f'{tgt} s{split} {time.time() - t0:.0f}s', flush=True)
    return pd.DataFrame(rows), pd.DataFrame(base_rows)


# ---------------------------------------------------------------- 汇总
def tsum(v):
    v = pd.Series(v).dropna()
    return AG.t_summary(v.to_numpy())


def f95(s, nd=2):
    return (f'{s["mean"]:+.{nd}f} [{s["lo95"]:+.{nd}f}, {s["hi95"]:+.{nd}f}]，p={s["p"]:.3g}，'
            f'>0 {s["n_pos"]}/{s["n"]}')


def build_unit_table(df, GR):
    """每 (单元, clf, k)：head(k)（5 重复平均）、两种底座、G 的 foreign_only 与 SRS(k)、全池。"""
    key = ['target', 'split', 'rep', 'unit', 'clf']
    out = []
    head = df[df.design == 'srs_D1'].groupby(key + ['k'])[['oa', 'ba_ge30']].mean().reset_index()
    bPT = df[df.design == 'base_PT'].set_index(key)[['oa', 'ba_ge30']]
    full = df[df.design == 'fullpool'].set_index(key)[['oa', 'ba_ge30']]
    full4 = df[df.design == 'fullpool_lam1e-4'].set_index(key)[['oa', 'ba_ge30']]
    gchk = df[df.design == 'srs_G_check'].set_index(key + ['k'])[['oa', 'ba_ge30']]
    GU = ['target', 'split', 'rep', 'clf']
    gF = GR[(GR.cond == 'none') & (GR.method == 'foreign_only')].set_index(GU)[['oa', 'ba_ge30']]
    gS = GR[(GR.cond == 'none') & (GR.method == 'srs')].set_index(GU + ['k'])[['oa', 'ba_ge30']]
    assert not gF.index.duplicated().any() and not gS.index.duplicated().any()
    for _, h in head.iterrows():
        ku = tuple(h[c] for c in key)
        gu = (h.target, h.split, h.rep, h.clf)
        for met in ('ba_ge30', 'oa'):
            row = dict(zip(key, ku), k=int(h.k), metric=met, head=h[met], base_PT=bPT.loc[ku, met],
                       base_G=gF.loc[gu, met], full=full.loc[ku, met], full_lam1e4=full4.loc[ku, met])
            if h.k in KS_RET:
                row['retrain'] = gS.loc[gu + (int(h.k),), met]
                row['head_Gsel'] = gchk.loc[ku + (int(h.k),), met]
            out.append(row)
    U = pd.DataFrame(out)
    U['dhead_PT'] = U['head'] - U.base_PT
    U['dhead_G'] = U['head'] - U.base_G
    U['dret'] = U.retrain - U.base_G
    U['diff_PT'] = U.dhead_PT - U.dret
    U['diff_G'] = U.dhead_G - U.dret  # = head − retrain
    U['dfull_PT'] = U.full - U.base_PT
    U['dfull_G'] = U.full - U.base_G
    U['dfull4_PT'] = U.full_lam1e4 - U.base_PT
    U['dheadGsel_PT'] = U.head_Gsel - U.base_PT
    U['diffGsel'] = U.head_Gsel - U.retrain  # 同一批标签：头 − 重训
    return U


def country_table(U, cols):
    """国家值 = 4 个单元（划分 × 重复）的平均（head 已先在 5 个子样本重复上平均）；'both' = 两种特征国家值的平均。"""
    cv = U.groupby(['metric', 'clf', 'k', 'target'])[cols].mean().reset_index()
    both = cv.groupby(['metric', 'k', 'target'])[cols].mean().reset_index().assign(clf='both')
    return pd.concat([cv, both], ignore_index=True)


def main():
    t0 = time.time()
    df, bd = compute_rows()
    df.to_csv(os.path.join(OUT, 'decision_curve_rows.csv'), index=False)
    bd.to_csv(os.path.join(OUT, 'decision_curve_base_argmax.csv'), index=False)
    GR = AG.load_rows(R_G)
    assert not AG.NOISE
    U = build_unit_table(df, GR)
    U.to_csv(os.path.join(OUT, 'decision_curve_units.csv'), index=False)
    cols = ['dhead_PT', 'dhead_G', 'dret', 'diff_PT', 'diff_G', 'dfull_PT', 'dfull_G', 'dfull4_PT', 'dheadGsel_PT',
            'diffGsel', 'base_PT', 'base_G', 'head', 'retrain', 'full']
    CV = country_table(U, cols)
    CV.to_csv(os.path.join(OUT, 'decision_curve_country.csv'), index=False)

    lines = []
    P = lines.append
    P('D1 决策层标签效率曲线（rev4 第 D1 节）—— 探索性（C 级）：G s0/s1 测试集已被多次复用；只作描述。')
    P(f'规格 {os.path.basename(PLAN)} sha256[:16]={sha16(PLAN)}（plan_hashes.txt 记录 7e90ccd5b1c3828f）；'
      f'calib_diag.py sha256[:16]={sha16(CALIB)}；取出的函数源码 sha256[:16]：{FUNC_HASH}')
    P(f'λ={LAM}（全池同）；权重 = calib_diag.weights(y, 1, "bal")（每个出现的类总权重相同，类内均匀）；'
      f'k ∈ {KS} × {NREP} 重复；种子 = int(md5("D1_{{tgt}}_{{split}}_{{rep}}_{{k}}_{{r}}")[:8], 16)，'
      f'np.random.default_rng(seed).choice(N, k, replace=False)，N = 该单元候选池大小。')
    P(f'单元数 {df.unit.nunique()}；国家 {len(COUNTRIES)}；推断单位 = 国家（df = n − 1），95% CI 为 t 区间。')
    P('')

    # ---------------- 核对
    P('== 核对 ==')
    chk = pd.read_csv(CALIB_ROWS)
    cr = chk[(chk.design == 'srs_G') & (chk['head'] == 'bal_l0.1')].set_index(['unit', 'clf', 'k'])[['oa', 'ba']]
    mine = df[df.design == 'srs_G_check'].set_index(['unit', 'clf', 'k'])[['oa', 'ba_ge30']]
    j = mine.join(cr, rsuffix='_calib', how='inner')
    P(f'(1) 用 G 实际 SRS 选样拟合同一头，与 calib_rows.csv（srs_G, bal_l0.1）对比：{len(j)} 行，'
      f'最大 |Δ| OA {(j.oa - j.oa_calib).abs().max():.2e}，BA {(j.ba_ge30 - j.ba).abs().max():.2e}')
    ca = chk[(chk.design == 'allpool') & (chk['head'] == 'bal_full')].set_index(['unit', 'clf'])[['oa', 'ba']]
    m4 = df[df.design == 'fullpool_lam1e-4'].set_index(['unit', 'clf'])[['oa', 'ba_ge30']].join(
        ca, rsuffix='_calib', how='inner')
    P(f'(2) 全池 λ=1e-4 与 calib_rows.csv（allpool, bal_full，只有 rep 0）对比：{len(m4)} 行，'
      f'最大 |Δ| OA {(m4.oa - m4.oa_calib).abs().max():.2e}，BA {(m4.ba_ge30 - m4.ba).abs().max():.2e}')
    rec = df[df.design == 'base_P0test_recomp'].set_index(['target', 'split', 'rep', 'clf'])[['oa', 'ba_ge30']]
    gF = GR[(GR.cond == 'none') & (GR.method == 'foreign_only')].set_index(['target', 'split', 'rep', 'clf'])[
        ['oa', 'ba_ge30']]
    jj = rec.join(gF, rsuffix='_G', how='inner')
    P(f'(3) 由 P0test_i300 重算的 foreign_only 与 G 行对比：{len(jj)} 行，最大 |Δ| OA '
      f'{(jj.oa - jj.oa_G).abs().max():.2e}，BA {(jj.ba_ge30 - jj.ba_ge30_G).abs().max():.2e}')
    P('')

    # ---------------- 两种底座
    P('== 两种底座的差异（k=0）==')
    P('P_T 底座 = argmax(温度校准后的 P_T 测试行，float32)；G 底座 = G foreign_only 行（未校准的 P0test_i300，float16）。')
    P('标量温度 s>0 不改变 argmax，所以差异只能来自 float16 舍入（并列或次序翻转）。')
    tot = bd.n_test.sum()
    P(f'测试行 argmax 不同的总数（按 44 单元 × 2 特征累计；同一划分的两个重复共用测试集，故每行计两次）：'
      f'{bd.n_argmax_diff.sum()} / {tot}（{100 * bd.n_argmax_diff.sum() / tot:.4f}%），'
      f'其中真值属 ≥30 类的 {bd.n_argmax_diff_bigtrue.sum()}；有差异的 (单元, 特征) {int((bd.n_argmax_diff > 0).sum())}/{len(bd)}，'
      f'单个组合最多 {bd.n_argmax_diff.max()} 行。')
    b0 = U[U.k == 50][['unit', 'clf', 'metric', 'base_PT', 'base_G']].copy()
    b0['d'] = b0.base_PT - b0.base_G
    for met in ('ba_ge30', 'oa'):
        x = b0[b0.metric == met].d
        P(f'  {met}：P_T 底座 − G 底座，单元级 最大 |Δ| {x.abs().max():.4f} pp，平均 {x.mean():+.5f} pp')
    for met in ('ba_ge30', 'oa'):
        for f in FEATS:
            c = CV[(CV.metric == met) & (CV.clf == f) & (CV.k == 50)]
            P(f'  k=0 水平（国家均值）{met} {f}：P_T 底座 {c.base_PT.mean():.2f}，G 底座 {c.base_G.mean():.2f}')
    P('')

    # ---------------- 主表
    def block(title, col, ks, met, sets=('ae', 's2', 'both')):
        P(title)
        for f in sets:
            for drop in (False, True):
                tag = '去 RO/PT' if drop else '11 国'
                for k in ks:
                    c = CV[(CV.metric == met) & (CV.clf == f) & (CV.k == k)]
                    if drop:
                        c = c[~c.target.isin(DROP)]
                    s = tsum(c[col])
                    P(f'  {f:4s} {tag:6s} k={k:<5d} {f95(s)}')
        P('')

    for met in ('ba_ge30', 'oa'):
        P(f'######## 指标 {met}（pp）########')
        block(f'-- Δhead(k)：只调决策层，相对 P_T 底座（主报告）--', 'dhead_PT', KS, met)
        block(f'-- Δretrain(k)：G 行 SRS(k) − foreign_only（同单元、同 clf）--', 'dret', KS_RET, met)
        block(f'-- Δhead(k) − Δretrain(k)（Δhead 用 P_T 底座，Δretrain 用 G 底座）--', 'diff_PT', KS_RET, met)
        block(f'-- 共同底座版（都以 G foreign_only 为底座）：Δhead_G(k) = head − G foreign_only --', 'dhead_G', KS,
              met)
        block(f'-- 共同底座版差值：Δhead_G(k) − Δretrain(k) = head(k) − retrain(k) --', 'diff_G', KS_RET, met)
        P(f'-- Δhead(全池)（λ=0.1，每单元一次）--')
        for f in ('ae', 's2', 'both'):
            for drop in (False, True):
                c = CV[(CV.metric == met) & (CV.clf == f) & (CV.k == 50)]
                if drop:
                    c = c[~c.target.isin(DROP)]
                tag = '去 RO/PT' if drop else '11 国'
                P(f'  {f:4s} {tag:6s} P_T 底座 {f95(tsum(c.dfull_PT))}；G 底座 {f95(tsum(c.dfull_G))}')
        P('')
        P('-- 比例（国家均值之比，描述，不推断）：Δhead(k)/Δhead(全池)；附 Δretrain(k)/Δhead(全池) --')
        for f in ('ae', 's2', 'both'):
            for drop in (False, True):
                tag = '去 RO/PT' if drop else '11 国'
                c = CV[(CV.metric == met) & (CV.clf == f)]
                if drop:
                    c = c[~c.target.isin(DROP)]
                den = c[c.k == 50].dfull_PT.mean()
                hs = '  '.join(f'k{k}:{c[c.k == k].dhead_PT.mean() / den:.2f}' for k in KS)
                rs = '  '.join(f'k{k}:{c[c.k == k].dret.mean() / den:.2f}' for k in KS_RET)
                P(f'  {f:4s} {tag:6s} 全池均值 {den:+.2f} | 头 {hs} | 重训 {rs}')
        if met == 'oa':
            P('  注意：OA 上全池头的增益接近 0 或为负（类均衡头面向 BA），上面的 OA 比例没有解释意义，只为与规格一致而列出。')
        P('')

    # ---------------- 敏感性 / 附加
    P('######## 附加（非规格要求；只作核对与敏感性）########')
    for met in ('ba_ge30', 'oa'):
        P(f'-- {met}：全池 λ=1e-4（calib_diag allpool 设置）相对 P_T 底座，对比 λ=0.1 --')
        for f in ('ae', 's2', 'both'):
            c = CV[(CV.metric == met) & (CV.clf == f) & (CV.k == 50)]
            P(f'  {f:4s} 11 国 λ=1e-4 {f95(tsum(c.dfull4_PT))}；λ=0.1 均值 {c.dfull_PT.mean():+.2f}')
        P(f'-- {met}：同一批标签（G 实际 SRS 选样）上 头 − 重训（标签配对版；D1 主表中头与重训用的是不同子样本）--')
        for f in ('ae', 's2', 'both'):
            for k in KS_RET:
                c = CV[(CV.metric == met) & (CV.clf == f) & (CV.k == k)]
                P(f'  {f:4s} k={k:<4d} {f95(tsum(c.diffGsel))}')
        P('')
    # 子样本噪声：单元内 5 个重复的 SD
    sd = df[df.design == 'srs_D1'].groupby(['unit', 'clf', 'k'])[['ba_ge30', 'oa']].std(ddof=1).reset_index()
    P('-- 单元内 5 个子样本重复间 SD 的平均（pp；描述头对抽样的敏感度）--')
    for f in FEATS:
        s_ = sd[sd.clf == f].groupby('k')[['ba_ge30', 'oa']].mean()
        P(f'  {f}：' + '  '.join(f'k{k}: BA {s_.loc[k, "ba_ge30"]:.2f} / OA {s_.loc[k, "oa"]:.2f}' for k in KS))
    s_med = df[df.design == 'srs_D1'].groupby('k')['s'].median()
    P('-- 拟合的温度乘子 s 中位数：' + '  '.join(f'k{k}:{s_med.loc[k]:.2f}' for k in KS) +
      f'  全池:{df[df.design == "fullpool"].s.median():.2f}  全池 λ=1e-4:{df[df.design == "fullpool_lam1e-4"].s.median():.2f}')
    bcols = [f'b{c}' for c in range(C)]
    bn = df[df.design.isin(['srs_D1', 'fullpool', 'fullpool_lam1e-4'])].assign(
        bmax=lambda x: x[bcols].abs().max(1))
    P('-- 偏置 max|b| 中位数：' + '  '.join(
        f'k{k}:{bn[(bn.design == "srs_D1") & (bn.k == k)].bmax.median():.2f}' for k in KS) +
      f'  全池:{bn[bn.design == "fullpool"].bmax.median():.2f}  全池 λ=1e-4:{bn[bn.design == "fullpool_lam1e-4"].bmax.median():.2f}')
    P('')

    # ---------------- 国家值
    P('######## 国家值（pp；Δhead 用 P_T 底座，Δretrain 用 G 行）########')
    for met in ('ba_ge30', 'oa'):
        for f in FEATS:
            P(f'-- {met} {f} --')
            P('  国家  ' + ' '.join(f'头{k:<5d}' for k in KS) + '  全池   ' + ' '.join(f'重训{k:<4d}' for k in KS_RET))
            for t in COUNTRIES:
                c = CV[(CV.metric == met) & (CV.clf == f) & (CV.target == t)].set_index('k')
                P(f'  {t}    ' + ' '.join(f'{c.loc[k, "dhead_PT"]:+6.2f}' for k in KS) +
                  f'  {c.loc[50, "dfull_PT"]:+6.2f}  ' + ' '.join(f'{c.loc[k, "dret"]:+7.2f}' for k in KS_RET))
        P('')
    # ---------------- 描述性结论（数字由上面的表计算）
    def cm(met, col, k, f='both', drop=False):
        c = CV[(CV.metric == met) & (CV.clf == f) & (CV.k == k)]
        return (c[~c.target.isin(DROP)] if drop else c)[col]
    def npos(v):
        return f'{int((v > 0).sum())}/{len(v)}'
    P('######## 描述性结论（C 级，测试集复用；两特征平均）########')
    P(f'(1) 只调决策层（类均衡、λ=0.1）时，50 个随机目标标签使 ba_ge30 提高 {cm("ba_ge30", "dhead_PT", 50).mean():+.2f} pp'
      f'（{npos(cm("ba_ge30", "dhead_PT", 50))} 国为正），400 个为 {cm("ba_ge30", "dhead_PT", 400).mean():+.2f} pp，'
      f'同一 λ 下全池为 {cm("ba_ge30", "dfull_PT", 50).mean():+.2f} pp；去掉 RO、PT 后依次为 '
      f'{cm("ba_ge30", "dhead_PT", 50, drop=True).mean():+.2f} / {cm("ba_ge30", "dhead_PT", 400, drop=True).mean():+.2f} / '
      f'{cm("ba_ge30", "dfull_PT", 50, drop=True).mean():+.2f} pp。')
    P(f'(2) 同样数量的标签并入重训，BA 增益为 {cm("ba_ge30", "dret", 50).mean():+.2f}（k=50）至 '
      f'{cm("ba_ge30", "dret", 400).mean():+.2f} pp（k=400），去掉 RO、PT 后为 {cm("ba_ge30", "dret", 50, drop=True).mean():+.2f} 至 '
      f'{cm("ba_ge30", "dret", 400, drop=True).mean():+.2f} pp；头领先重训 {min(cm("ba_ge30", "diff_PT", k).mean() for k in KS_RET):+.2f} 至 '
      f'{max(cm("ba_ge30", "diff_PT", k).mean() for k in KS_RET):+.2f} pp。')
    P(f'(3) 这一 BA 收益以 OA 为代价：去掉 RO、PT 的 9 国里，头的 OA 变化为 {cm("oa", "dhead_PT", 50, drop=True).mean():+.2f}（k=50）至 '
      f'{cm("oa", "dhead_PT", 1000, drop=True).mean():+.2f} pp（k=1000），重训为 {cm("oa", "dret", 50, drop=True).mean():+.2f} 至 '
      f'{cm("oa", "dret", 400, drop=True).mean():+.2f} pp。')
    P(f'(4) 全池头受 λ=0.1 约束：λ=1e-4 时全池 BA 增益为 {cm("ba_ge30", "dfull4_PT", 50).mean():+.2f} pp，k=400 的头只达到它的 '
      f'{cm("ba_ge30", "dhead_PT", 400).mean() / cm("ba_ge30", "dfull4_PT", 50).mean():.0%}。')
    P('')
    txt = '\n'.join(lines)
    open(os.path.join(OUT, 'decision_curve_summary.txt'), 'w').write(txt + '\n')
    print(txt)
    plot(CV)
    print(f'done {time.time() - t0:.0f}s')


# ---------------------------------------------------------------- 图
def plot(CV):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'Heiti SC']
    plt.rcParams['axes.unicode_minus'] = False
    INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
    COL = {'head': '#2a78d6', 'ret': '#eb6834'}
    LS = {'ae': '-', 's2': (0, (5, 3))}
    MK = {'ae': 'o', 's2': 's'}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), facecolor=SURF)
    for ax, met, name in zip(axes, ('ba_ge30', 'oa'), ('平衡精度 ba_ge30', '总体精度 OA（头为类均衡目标，面向 BA）')):
        ax.set_facecolor(SURF)
        ax.axhline(0, color=INK2, lw=0.8)
        for f in FEATS:
            c = CV[(CV.metric == met) & (CV.clf == f)]
            for kind, col, ks, off in (('head', 'dhead_PT', KS, 0.97), ('ret', 'dret', KS_RET, 1.03)):
                st = [tsum(c[c.k == k][col]) for k in ks]
                m = np.array([s['mean'] for s in st])
                lo = np.array([s['lo95'] for s in st])
                hi = np.array([s['hi95'] for s in st])
                x = np.array(ks) * (off if f == 'ae' else off ** 2)
                ax.errorbar(x, m, yerr=[m - lo, hi - m], color=COL[kind], ls=LS[f], lw=2, marker=MK[f], ms=6,
                            mfc=COL[kind] if f == 'ae' else SURF, mec=COL[kind], mew=1.5, elinewidth=0.8,
                            capsize=0, alpha=0.95)
            full = c[c.k == 50].dfull_PT.mean()
            ax.axhline(full, color=COL['head'], ls=(0, (1, 2)), lw=1.4 if f == 'ae' else 1.0,
                       alpha=0.9 if f == 'ae' else 0.6)
            ax.annotate(f'全池头 {f.upper()} {full:+.1f}', (2900, full), textcoords='offset points',
                        xytext=(0, 3 if f == 'ae' else -10), fontsize=8, color=INK2, ha='right')
        ax.set_xscale('log')
        ax.set_xticks(KS)
        ax.set_xticklabels([str(k) for k in KS])
        ax.minorticks_off()
        ax.set_xlim(38, 3000)
        ax.set_xlabel('目标国标签数 k（对数轴）', color=INK)
        ax.set_ylabel(f'相对 k=0 的增益（pp）', color=INK)
        ax.set_title(name, color=INK, fontsize=11)
        ax.grid(axis='y', color=GRID, lw=0.6)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        for sp in ('left', 'bottom'):
            ax.spines[sp].set_color(INK2)
        ax.tick_params(colors=INK2)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=COL['head'], lw=2, label='只调决策层（温度+类偏置，λ=0.1，类均衡）'),
         Line2D([], [], color=COL['ret'], lw=2, label='并入重训（G 的 SRS 行，r=50）'),
         Line2D([], [], color=INK2, lw=2, ls=LS['ae'], marker='o', label='AE 特征'),
         Line2D([], [], color=INK2, lw=2, ls=LS['s2'], marker='s', mfc=SURF, label='S2 特征'),
         Line2D([], [], color=COL['head'], lw=1.4, ls=(0, (1, 2)), label='全池决策头（每单元约 3 万标签）')]
    fig.legend(handles=h, loc='lower center', ncol=3, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.01))
    fig.text(0.01, 0.955, 'D1 决策层标签效率曲线：11 国均值，误差线为国家层 t 95% CI。头相对 argmax(P_T)；重训 = G 的 SRS(k) − foreign_only。'
             'G s0/s1 测试集复用（C 级，描述性）',
             fontsize=9, color=INK2)
    fig.tight_layout(rect=(0, 0.1, 1, 0.94))
    fig.savefig(os.path.join(OUT, 'fig_decision_curve.png'), dpi=200, facecolor=SURF)


if __name__ == '__main__':
    main()
