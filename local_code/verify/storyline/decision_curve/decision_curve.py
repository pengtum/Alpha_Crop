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
方案外（非规格要求，只作核对 / 敏感性，报告中单列、标为"方案外，描述"）：
- 'srs_G_check'：用 G 实际的 SRS 选样拟合同一个头，与 calib_diag 的 calib_rows.csv（srs_G, bal_l0.1）逐单元对比，
  核对实现；同时给出"标签配对版"头 − 重训（同一批 G 选样）。
- 'fullpool_lam1e-4'：全池 + λ=1e-4（calib_diag 'allpool bal_full' 的设置）。
- λ 敏感性：读复核者 d1_review/lam_sens_rows.csv（同一 D1 子样本种子、独立实现，λ ∈ {1e-4, 1e-2}），只作描述，不用于选 λ。
- 表征层剩余（rev4 术语的操作化）：model_diag tgt300（只用目标池训练，划分 0，与 D1 的 s0/r0 单元用同一 3 万候选池）
  − D1 全池头（λ=0.1，s0/r0）。
- 对照读数：H1 附缓冲格网上界（reposition/decision_vs_representation/oracle_G_buffered_rows.csv），只用于定义表中的区分。
输出（本目录）：decision_curve_rows.csv、decision_curve_summary.txt、decision_curve_country.csv、fig_decision_curve.png
事后分析，测试集已被多次复用 → C 级、描述性。

v2（2026-09-27，按 D1 复核意见修订；v1 另存为 *_v1.*）：
- summary 的解读改写：λ=0.1 下约 +6.5 pp 的平台是预先固定的 BA–OA 工作点；删去"k=400 只达到 λ=1e-4 全池 71%"；
  λ 敏感性只作方案外描述。
- 国家差异描述按复核意见更正（由国家表自动生成）。
- 图改为 2×3：原 11 国 BA/OA 两面板 + 去 RO、PT 的 BA/OA 两面板 + 标签配对版 头 − 重训 面板（两行各一）；图注写明
  RO/LV 的 ba_ge30 类数、"both" 口径、两位小数。
- summary 末尾加"术语定义表草稿"。
- 计算部分（compute_rows、build_unit_table、country_table）与 v1 完全相同；加 --from-csv 可只从已存行重做汇总与图。
"""
import os
import sys
for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[v] = '2'
sys.dont_write_bytecode = True  # 不在项目目录 L 的 __pycache__ 里写任何文件
import ast
import hashlib
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
LAM_SENS = '/Users/perng_mini/AlphaCrops_local/verify/storyline/d1_review/lam_sens_rows.csv'
TGT300 = '/Users/perng_mini/AlphaCrops_local/verify/model_diag/fit/table_upper_bound.csv'
ORACLE = '/Users/perng_mini/AlphaCrops_local/verify/reposition/decision_vs_representation/oracle_G_buffered_rows.csv'
FRONTIER2 = '/Users/perng_mini/AlphaCrops_local/verify/pivot_eval/prior_shift/frontier2_report.txt'
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


# ---------------------------------------------------------------- 逐单元计算（与 v1 相同）
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


def kset_text(ks_hit, ks_all):
    """把"重训超过头"的 k 集合写成文字：各 k / k≥x / 仅 k=x / 列举。"""
    ks_hit = sorted(ks_hit)
    if not ks_hit:
        return ''
    if ks_hit == sorted(ks_all):
        return '各 k'
    if len(ks_hit) == 1:
        return f'仅 k={ks_hit[0]}'
    tail = [k for k in ks_all if k >= ks_hit[0]]
    if ks_hit == tail:
        return f'k≥{ks_hit[0]}'
    return 'k=' + '、'.join(str(k) for k in ks_hit)


def main():
    t0 = time.time()
    if '--from-csv' in sys.argv:
        df = pd.read_csv(os.path.join(OUT, 'decision_curve_rows.csv'))
        bd = pd.read_csv(os.path.join(OUT, 'decision_curve_base_argmax.csv'))
        print('从已存 decision_curve_rows.csv 重做汇总与图（不重算头）')
    else:
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

    nbig = df[df.design == 'fullpool'].groupby('target').n_big.agg(['min', 'max'])
    assert (nbig['min'] == nbig['max']).all()
    nbig = nbig['min']
    npool = df[df.design == 'fullpool'].drop_duplicates('unit').n_pool
    others = nbig.drop(['RO', 'LV'])

    lines = []
    P = lines.append
    P('D1 决策层标签效率曲线（rev4 第 D1 节）—— 探索性（C 级）：G s0/s1 测试集已被多次复用；只作描述。')
    P('版本 v2（2026-09-27）：按 D1 复核意见修订解读、国家差异描述与图，另加术语定义表草稿；计算与 v1 相同（v1 见 *_v1.*）。')
    P(f'规格 {os.path.basename(PLAN)} sha256[:16]={sha16(PLAN)}（plan_hashes.txt 记录 7e90ccd5b1c3828f）；'
      f'calib_diag.py sha256[:16]={sha16(CALIB)}；取出的函数源码 sha256[:16]：{FUNC_HASH}')
    P(f'λ={LAM}（全池同）；权重 = calib_diag.weights(y, 1, "bal")（每个出现的类总权重相同，类内均匀）；'
      f'k ∈ {KS} × {NREP} 重复；种子 = int(md5("D1_{{tgt}}_{{split}}_{{rep}}_{{k}}_{{r}}")[:8], 16)，'
      f'np.random.default_rng(seed).choice(N, k, replace=False)，N = 该单元候选池大小'
      f'（{npool.min():,}–{npool.max():,}，即"3 万候选池"）。')
    P(f'单元数 {df.unit.nunique()}；国家 {len(COUNTRIES)}；推断单位 = 国家（df = n − 1），95% CI 为 t 区间。')
    P(f'ba_ge30 的类数（测试集中 ≥30 个地块的类）：RO {nbig["RO"]}、LV {nbig["LV"]}，其余 {others.min()}–{others.max()}'
      f'（{"、".join(f"{t} {nbig[t]}" for t in COUNTRIES)}）。RO 只有 {nbig["RO"]} 类，却主导 11 国均值，所以每项都另报去掉 RO、PT 的版本。')
    P('汇总口径："both" = AE、S2 两种特征国家值的平均，是作者选定的汇总口径（rev4 没有定义跨特征合并）；'
      '逐特征的值（ae、s2）同时列出。数值保留两位小数。')
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
        block(f'-- Δhead(k) − Δretrain(k)（Δhead 用 P_T 底座，Δretrain 用 G 底座；头与重训用不同子样本）--', 'diff_PT',
              KS_RET, met)
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

    # ---------------- 方案外：核对与敏感性
    P('######## 方案外，描述（非 rev4 规格；只作核对与敏感性；不用于选 λ，也不替换主结果）########')
    for met in ('ba_ge30', 'oa'):
        P(f'-- {met}：同一批标签（G 实际 SRS 选样）上 头 − 重训（标签配对版，共同底座 G foreign_only；'
          f'rev4 主表中头与重训用的是不同子样本）--')
        for f in ('ae', 's2', 'both'):
            for drop in (False, True):
                tag = '去 RO/PT' if drop else '11 国'
                for k in KS_RET:
                    c = CV[(CV.metric == met) & (CV.clf == f) & (CV.k == k)]
                    if drop:
                        c = c[~c.target.isin(DROP)]
                    P(f'  {f:4s} {tag:6s} k={k:<4d} {f95(tsum(c.diffGsel))}')
        P('')

    # λ 敏感性：D1 λ=0.1（本脚本）+ 复核者 lam_sens（λ=1e-2、1e-4）
    LS = pd.read_csv(LAM_SENS)
    lcv = (LS.groupby(['lam', 'k', 'target', 'clf'])[['dba', 'doa']].mean().reset_index())
    lcv_both = lcv.groupby(['lam', 'k', 'target'])[['dba', 'doa']].mean().reset_index()
    # 与本脚本 λ=1e-4 全池的一致性（两套独立实现）
    f4 = CV[(CV.k == 50) & CV.clf.isin(FEATS)].pivot_table(index=['target', 'clf'], columns='metric', values='dfull4_PT')
    l4 = lcv[(lcv.lam == 1e-4) & (lcv.k == -1)].set_index(['target', 'clf'])
    jl = f4.join(l4, how='inner')
    P(f'-- λ 敏感性（方案外，描述）：λ=0.1 为本脚本（rev4 预先固定）；λ=1e-2、1e-4 读自复核者 {os.path.relpath(LAM_SENS, OUT)}'
      f'（事后计算，同一 D1 子样本种子规则，独立实现，只拟合头；k ∈ {{50,200,400,1000}} 与全池）。C 级；不得用于选 λ。')
    P(f'   一致性：复核者 λ=1e-4 全池与本脚本 fullpool_lam1e-4 的国家 × 特征值，{len(jl)} 对，'
      f'最大 |Δ| BA {(jl.ba_ge30 - jl.dba).abs().max():.2e}，OA {(jl.oa - jl.doa).abs().max():.2e}')

    def lam_row(lam, drop, k):
        """国家均值（两特征平均）。k=-1 为全池。"""
        if lam == LAM:
            c = CV[(CV.clf == 'both') & (CV.k == (50 if k == -1 else k))]
            if drop:
                c = c[~c.target.isin(DROP)]
            col = 'dfull_PT' if k == -1 else 'dhead_PT'
            return (c[c.metric == 'ba_ge30'][col].mean(), c[c.metric == 'oa'][col].mean())
        c = lcv_both[(lcv_both.lam == lam) & (lcv_both.k == k)]
        if drop:
            c = c[~c.target.isin(DROP)]
        return c.dba.mean(), c.doa.mean()

    LSK = [50, 200, 400, 1000, -1]
    LAMS = [LAM, 1e-2, 1e-4]
    lam_tab = {}
    P('   两特征国家值平均；每格为 BA / OA（pp，相对 k=0）；最后一列 = Δhead(k=400) BA ÷ 同 λ 全池 BA（国家均值之比）')
    for drop in (False, True):
        tag = '去 RO/PT' if drop else '11 国'
        for lam in LAMS:
            vals = {k: lam_row(lam, drop, k) for k in LSK}
            lam_tab[(lam, drop)] = vals
            P(f'  λ={lam:<6g} {tag:6s} ' + '  '.join(
                f'{"全池" if k == -1 else "k=" + str(k)}: {vals[k][0]:+.2f} / {vals[k][1]:+.2f}' for k in LSK) +
              f'  | k400/全池 {vals[400][0] / vals[-1][0]:.2f}')
    P('')

    P('-- 全池 λ=1e-4（calib_diag allpool 设置，本脚本计算）相对 P_T 底座，对比 λ=0.1（方案外，描述）--')
    for met in ('ba_ge30', 'oa'):
        for f in ('ae', 's2', 'both'):
            c = CV[(CV.metric == met) & (CV.clf == f) & (CV.k == 50)]
            P(f'  {met:7s} {f:4s} 11 国 λ=1e-4 {f95(tsum(c.dfull4_PT))}；λ=0.1 均值 {c.dfull_PT.mean():+.2f}')
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
        bmax=lambda x: x[bcols].abs().max(axis=1))
    P('-- 偏置 max|b| 中位数：' + '  '.join(
        f'k{k}:{bn[(bn.design == "srs_D1") & (bn.k == k)].bmax.median():.2f}' for k in KS) +
      f'  全池:{bn[bn.design == "fullpool"].bmax.median():.2f}  全池 λ=1e-4:{bn[bn.design == "fullpool_lam1e-4"].bmax.median():.2f}')
    P('   λ 加在类均衡加权后的平均 NLL 上，正则强度不随 k 变小；各 k 的 max|b| 相近，说明 λ=0.1 在所有 k 下都约束偏置幅度。')
    P('')

    # 表征层剩余（rev4 术语的操作化；方案外，描述）
    T3 = pd.read_csv(TGT300).rename(columns={'tgt': 'target', 'feat': 'clf'})
    f00 = df[(df.design == 'fullpool') & (df.split == 0) & (df.rep == 0)].set_index(['target', 'clf'])[
        ['oa', 'ba_ge30']]
    b00 = df[(df.design == 'base_PT') & (df.split == 0) & (df.rep == 0)].set_index(['target', 'clf'])[
        ['oa', 'ba_ge30']]
    RR = T3.set_index(['target', 'clf'])[['k0_ba', 'k0_oa', 'tgt300_ba', 'tgt300_oa']].join(
        f00.rename(columns={'oa': 'full_oa', 'ba_ge30': 'full_ba'})).join(
        b00.rename(columns={'oa': 'base_oa', 'ba_ge30': 'base_ba'})).reset_index()
    assert len(RR) == 22 and RR.notna().all().all()
    RR['gap_ba'] = RR.tgt300_ba - RR.base_ba
    RR['gap_oa'] = RR.tgt300_oa - RR.base_oa
    RR['dec_ba'] = RR.full_ba - RR.base_ba
    RR['dec_oa'] = RR.full_oa - RR.base_oa
    RR['res_ba'] = RR.tgt300_ba - RR.full_ba
    RR['res_oa'] = RR.tgt300_oa - RR.full_oa
    RR.to_csv(os.path.join(OUT, 'decision_curve_repr_residual_s0r0.csv'), index=False)
    P('-- 表征层剩余（rev4 术语的操作化；方案外，描述）：划分 0、重复 0 单元，tgt300 − D1 全池头（λ=0.1）--')
    P(f'   tgt300 = model_diag 只用目标池训练的 LightGBM（300 轮，{os.path.relpath(TGT300, OUT)}，只有划分 0），'
      '训练集 = pool_positions(S, 0, 0)，与 D1 s0/r0 单元的全池头用同一批约 3 万候选池标签。')
    P(f'   核对：table_upper_bound 的 k0 与本脚本 P_T 底座（s0/r0）最大差 BA {(RR.k0_ba - RR.base_ba).abs().max():.3f}、'
      f'OA {(RR.k0_oa - RR.base_oa).abs().max():.3f}（该表保留两位小数）。')
    P('   恒等式：tgt300 − k0 = Δhead(全池) + 表征层剩余。类均衡头的 OA 不是 OA 的决策层上界，所以 OA 行的"剩余"含工作点代价。')
    RRb = RR.groupby('target')[['gap_ba', 'dec_ba', 'res_ba', 'gap_oa', 'dec_oa', 'res_oa']].mean().reset_index().assign(
        clf='both')
    RRall = pd.concat([RR, RRb], ignore_index=True)
    res_tab = {}
    for f in ('ae', 's2', 'both'):
        for drop in (False, True):
            tag = '去 RO/PT' if drop else '11 国'
            c = RRall[RRall.clf == f]
            if drop:
                c = c[~c.target.isin(DROP)]
            res_tab[(f, drop)] = c
            P(f'  {f:4s} {tag:6s} BA：tgt300−k0 {c.gap_ba.mean():+.2f}，全池头 {c.dec_ba.mean():+.2f}，'
              f'剩余 {f95(tsum(c.res_ba))} | OA：tgt300−k0 {c.gap_oa.mean():+.2f}，全池头 {c.dec_oa.mean():+.2f}，'
              f'剩余 {c.res_oa.mean():+.2f}')
    P('   逐国（BA 剩余，ae / s2）：' + '  '.join(
        f'{t} {RR[(RR.target == t) & (RR.clf == "ae")].res_ba.iloc[0]:+.2f}/{RR[(RR.target == t) & (RR.clf == "s2")].res_ba.iloc[0]:+.2f}'
        for t in COUNTRIES))
    P('')

    # H1 附缓冲格网上界（读数，只用于定义表区分）
    OR = pd.read_csv(ORACLE)
    orc = OR.groupby(['target', 'feat'])[['dec_oa', 'dec_ba']].mean().reset_index()
    ora = {f: orc[orc.feat == f][['dec_oa', 'dec_ba']].mean() for f in FEATS}
    ora_n = OR.drop_duplicates(['target', 'split']).shape[0]
    P(f'-- 读数：H1 附缓冲格网决策层上界（{os.path.relpath(ORACLE, OUT)}，两划分平均后的 11 国均值）：'
      f'OA AE {ora["ae"].dec_oa:+.2f} / S2 {ora["s2"].dec_oa:+.2f}；BA AE {ora["ae"].dec_ba:+.2f} / S2 {ora["s2"].dec_ba:+.2f}。'
      f'拟合方法与 D1 不同（见定义表），数值不与 D1 相除或并列当同一量。')
    P('')

    # ---------------- 国家值
    P('######## 国家值（pp；Δhead 用 P_T 底座、5 个 D1 子样本平均；Δretrain 用 G 行、每单元 1 次选样）########')
    for met in ('ba_ge30', 'oa'):
        for f in FEATS:
            P(f'-- {met} {f} --')
            P('  国家  ' + ' '.join(f'头{k:<5d}' for k in KS) + '  全池   ' + ' '.join(f'重训{k:<4d}' for k in KS_RET))
            for t in COUNTRIES:
                c = CV[(CV.metric == met) & (CV.clf == f) & (CV.target == t)].set_index('k')
                P(f'  {t}    ' + ' '.join(f'{c.loc[k, "dhead_PT"]:+6.2f}' for k in KS) +
                  f'  {c.loc[50, "dfull_PT"]:+6.2f}  ' + ' '.join(f'{c.loc[k, "dret"]:+7.2f}' for k in KS_RET))
        P('')

    # ---------------- 国家差异（由国家表自动生成）
    def cval(met, f, t, k, col):
        c = CV[(CV.metric == met) & (CV.clf == f) & (CV.target == t) & (CV.k == k)]
        return float(c[col].iloc[0])

    P('######## 国家差异（描述；ba_ge30；由上面的国家表自动生成）########')
    hit = {}
    for t in COUNTRIES:
        for f in FEATS:
            ks = [k for k in KS_RET if cval('ba_ge30', f, t, k, 'dret') > cval('ba_ge30', f, t, k, 'dhead_PT')]
            if ks:
                hit[(t, f)] = ks
    P('(i) 非配对主表中重训超过头的 (国家, 特征, k)：' + '；'.join(
        f'{t} {f.upper()} k={"、".join(map(str, ks))}' for (t, f), ks in hit.items()))
    ctry_hit = [t for t in COUNTRIES if any((t, f) in hit for f in FEATS)]
    hit_txt = '与 '.join(
        f'{t}（' + '；'.join(f'{f.upper()} {kset_text(hit[(t, f)], KS_RET)}' for f in FEATS if (t, f) in hit) + '）'
        for t in ctry_hit)
    P(f'   即：{hit_txt}是重训超过头的国家，其余国家在各 k 上都是头更高。')
    for t in ctry_hit:
        for f in FEATS:
            P(f'   {t} {f.upper()}：头 ' + ' / '.join(f'{cval("ba_ge30", f, t, k, "dhead_PT"):+.2f}' for k in KS_RET) +
              '；重训 ' + ' / '.join(f'{cval("ba_ge30", f, t, k, "dret"):+.2f}' for k in KS_RET) + '（k=50/100/200/400）')
    hitp = {}
    for t in COUNTRIES:
        for f in FEATS:
            ks = [k for k in KS_RET if cval('ba_ge30', f, t, k, 'diffGsel') < 0]
            if ks:
                hitp[(t, f)] = ks
    P('   标签配对版（方案外）中头 − 重训 < 0 的 (国家, 特征, k)：' + ('；'.join(
        f'{t} {f.upper()} {kset_text(ks, KS_RET)}' for (t, f), ks in hitp.items()) or '无'))
    ro = {f: [cval('ba_ge30', f, 'RO', k, 'dhead_PT') for k in KS] for f in FEATS}
    P(f'(ii) RO 主导 11 国均值：AE 头在 k=50 已达 {ro["ae"][0]:+.2f}，之后在 {min(ro["ae"]):+.2f}–{max(ro["ae"]):+.2f} 之间；'
      f'S2 为 {ro["s2"][0]:+.2f}，之后 {min(ro["s2"]):+.2f}–{max(ro["s2"]):+.2f}（RO 的 ba_ge30 只含 {nbig["RO"]} 类）。')
    se = {f: [cval('ba_ge30', f, 'SE', k, 'dhead_PT') for k in KS] for f in FEATS}
    sef = {f: cval('ba_ge30', f, 'SE', 50, 'dfull_PT') for f in FEATS}
    P(f'(iii) SE 要到 k≥200 才得到大部分增益：AE k=100 {se["ae"][1]:+.2f} → k=200 {se["ae"][2]:+.2f}'
      f'（全池 {sef["ae"]:+.2f}，比例 {se["ae"][1] / sef["ae"]:.2f} → {se["ae"][2] / sef["ae"]:.2f}）；'
      f'S2 k=100 {se["s2"][1]:+.2f} → k=200 {se["s2"][2]:+.2f}（全池 {sef["s2"]:+.2f}，比例 '
      f'{se["s2"][1] / sef["s2"]:.2f} → {se["s2"][2] / sef["s2"]:.2f}）。')
    below = []
    for t in COUNTRIES:
        for f in FEATS:
            hv = {k: cval('ba_ge30', f, t, k, 'dhead_PT') for k in KS}
            fv = cval('ba_ge30', f, t, 50, 'dfull_PT')
            kmax = max(hv, key=hv.get)
            if hv[kmax] - fv > 0.5:
                below.append((t, f, kmax, hv[kmax], fv))
    P('(iv) 全池不是逐国严格上界：全池值比某个 k 的头低 0.5 pp 以上的 (国家, 特征)：' + '；'.join(
        f'{t} {f.upper()}（全池 {fv:+.2f}，k={km} 为 {hv:+.2f}）' for t, f, km, hv, fv in below) + '。')
    b1000 = CV[(CV.metric == 'ba_ge30') & (CV.clf == 'both') & (CV.k == 1000)].dhead_PT.mean()
    bfull = CV[(CV.metric == 'ba_ge30') & (CV.clf == 'both') & (CV.k == 50)].dfull_PT.mean()
    P(f'     11 国均值上，k=1000 的头（{b1000:+.2f}）也高于全池（{bfull:+.2f}）。')
    P('')

    # ---------------- 描述性结论（数字由上面的表计算）
    def cm(met, col, k, f='both', drop=False):
        c = CV[(CV.metric == met) & (CV.clf == f) & (CV.k == k)]
        return (c[~c.target.isin(DROP)] if drop else c)[col]

    def npos(v):
        return f'{int((v > 0).sum())}/{len(v)}'

    lt = lam_tab
    P('######## 描述性结论（C 级，测试集复用；"both" 口径，逐特征值见上）########')
    P(f'(1) 只调决策层（类均衡、λ=0.1）时，50 个随机目标标签使 ba_ge30 提高 {cm("ba_ge30", "dhead_PT", 50).mean():+.2f} pp'
      f'（{npos(cm("ba_ge30", "dhead_PT", 50))} 国为正），400 个为 {cm("ba_ge30", "dhead_PT", 400).mean():+.2f} pp，'
      f'同一 λ 下全池为 {cm("ba_ge30", "dfull_PT", 50).mean():+.2f} pp；去掉 RO、PT 后依次为 '
      f'{cm("ba_ge30", "dhead_PT", 50, drop=True).mean():+.2f} / {cm("ba_ge30", "dhead_PT", 400, drop=True).mean():+.2f} / '
      f'{cm("ba_ge30", "dfull_PT", 50, drop=True).mean():+.2f} pp。')
    P(f'(2) 同样数量的标签并入重训，BA 增益为 {cm("ba_ge30", "dret", 50).mean():+.2f}（k=50）至 '
      f'{cm("ba_ge30", "dret", 400).mean():+.2f} pp（k=400），去掉 RO、PT 后为 {cm("ba_ge30", "dret", 50, drop=True).mean():+.2f} 至 '
      f'{cm("ba_ge30", "dret", 400, drop=True).mean():+.2f} pp；头领先重训 {min(cm("ba_ge30", "diff_PT", k).mean() for k in KS_RET):+.2f} 至 '
      f'{max(cm("ba_ge30", "diff_PT", k).mean() for k in KS_RET):+.2f} pp。标签配对版（方案外，同一批 G 选样）为 '
      f'{min(cm("ba_ge30", "diffGsel", k).mean() for k in KS_RET):+.2f} 至 {max(cm("ba_ge30", "diffGsel", k).mean() for k in KS_RET):+.2f} pp，'
      f'k=50 时 {f95(tsum(cm("ba_ge30", "diffGsel", 50)))}。头与重训各自的非配对 CI 大幅重叠，'
      f'但配对差值的 CI 窄，差异应以配对差值读。')
    P(f'(3) 这一 BA 收益以 OA 为代价：去掉 RO、PT 的 9 国里，头的 OA 变化为 {cm("oa", "dhead_PT", 50, drop=True).mean():+.2f}（k=50）至 '
      f'{cm("oa", "dhead_PT", 1000, drop=True).mean():+.2f} pp（k=1000），重训为 {cm("oa", "dret", 50, drop=True).mean():+.2f} 至 '
      f'{cm("oa", "dret", 400, drop=True).mean():+.2f} pp；标签配对版 头 − 重训 的 OA 在 11 国为 '
      f'{cm("oa", "diffGsel", 50).mean():+.2f}（k=50）至 {cm("oa", "diffGsel", 400).mean():+.2f} pp（k=400）。'
      f'11 国 OA 均值被 RO、PT 主导（两国头与重训的 OA 都大幅为正），看不到其余 9 国的方向，所以 OA 须看去掉 RO、PT 的版本。')
    # λ 敏感性的文字由数字判断生成，避免措辞与数字不符
    ba400 = [lt[(lam, False)][400][0] for lam in LAMS]           # λ = 0.1, 1e-2, 1e-4
    oa400 = [lt[(lam, False)][400][1] for lam in LAMS]
    oa400d = [lt[(lam, True)][400][1] for lam in LAMS]
    ba50 = [lt[(lam, False)][50][0] for lam in LAMS]
    ratio = [lt[(lam, False)][400][0] / lt[(lam, False)][-1][0] for lam in LAMS]
    oa_mono = all(np.diff(oa400) < 0) and all(np.diff(oa400d) < 0)
    ba_mono = all(np.diff(ba400) > 0)
    ba50_mono = all(np.diff(ba50) > 0)
    low_ba = [b for b in ba400[1:]]
    P(f'(4) 平台的含义：曲线在 k≈200–400 变平（Δhead(k)/同 λ 全池：k=200 为 '
      f'{cm("ba_ge30", "dhead_PT", 200).mean() / cm("ba_ge30", "dfull_PT", 50).mean():.2f}，k=400 为 '
      f'{cm("ba_ge30", "dhead_PT", 400).mean() / cm("ba_ge30", "dfull_PT", 50).mean():.2f}）。λ=0.1 下约 '
      f'{cm("ba_ge30", "dfull_PT", 50).mean():+.2f} pp 的平台，是 rev4 预先固定的 λ=0.1 所对应的一个 BA–OA 工作点'
      f'（同一全池头在去掉 RO、PT 的 9 国 OA 为 {cm("oa", "dfull_PT", 50, drop=True).mean():+.2f} pp），'
      f'不应读成"标签已用尽"，也不是决策层的能力上限。'
      f'方案外敏感性（复核者 lam_sens，事后计算，C 级）显示 λ 决定平台高度与 OA 代价：k=400、11 国，'
      f'λ=0.1 / 1e-2 / 1e-4 的 BA 为 {ba400[0]:+.2f} / {ba400[1]:+.2f} / {ba400[2]:+.2f}，'
      f'OA 为 {oa400[0]:+.2f} / {oa400[1]:+.2f} / {oa400[2]:+.2f}；去掉 RO、PT 的 9 国 OA 为 '
      f'{oa400d[0]:+.2f} / {oa400d[1]:+.2f} / {oa400d[2]:+.2f}。'
      + (f'λ 低于 0.1 时 BA 平台升到 {min(low_ba):+.2f}–{max(low_ba):+.2f}'
         + ('、并随 λ 降低单调上升' if ba_mono else '（1e-2 与 1e-4 之间不单调）'))
      + ('；OA 代价随 λ 降低单调加大。' if oa_mono else '；OA 代价随 λ 的变化不单调。')
      + f'"k≈400 追平同 λ 全池"在三种 λ 下都成立：k=400 ÷ 同 λ 全池 = {ratio[0]:.2f} / {ratio[1]:.2f} / {ratio[2]:.2f}。'
      f'k=50 时三种 λ 的 BA 为 {ba50[0]:+.2f} / {ba50[1]:+.2f} / {ba50[2]:+.2f}'
      + ('（随 λ 降低单调上升）。' if ba50_mono else '（不随 λ 单调）。')
      + '这些敏感性数字只作描述，不用来选 λ；主结果仍是 rev4 预先固定的 λ=0.1。')
    P('(5) 因此"少量标签能修正决策层"只能限定为：在 λ=0.1 这个预先固定的工作点上，面向平衡精度的决策层，'
      '约 200–400 个随机目标标签即可达到同一设置下 3 万候选池标签的水平（方案外：λ=1e-2、1e-4 下 k=400 也达到同 λ 全池水平）。'
      '它不涉及 OA：同一个头在去掉 RO、PT 的 9 国使 OA 下降；更低 λ 下更高的 BA 平台是以更大的 OA 代价换来的另一个工作点。')
    P('')

    # ---------------- 图注
    P('######## 图注（fig_decision_curve.png）########')
    P(caption_text(nbig, npool))
    P('')

    # ---------------- 术语定义表草稿
    rb = res_tab
    P('######## 术语定义表草稿（供论文方法节；数值均为 C 级、G s0/s1 测试集复用；"方案外"项只作描述）########')
    P('| 术语 | 定义 | 本研究的操作化 | 数值（ba_ge30 / OA，pp，相对 k=0） | 不能与之混用的量 |')
    P('|---|---|---|---|---|')
    P('| **决策层可挽回部分**（rev4 D1） | 在冻结的外国模型上，用目标标签只重新拟合温度与类别偏置所能得到的增益 | '
      '冻结 G 阶段 0 外国 LightGBM（300 轮 + 标量温度，P_T）；logits = s·log p + b；目标 = **类均衡**加权 NLL + (λ/2)‖b‖²，'
      '**λ=0.1 预先固定**，s∈[1e-3,50]，b∈[−30,30]；拟合集 = 该单元 **3 万候选池**（pool_positions，'
      f'{npool.min():,}–{npool.max():,}）中简单随机 k 个或全部；只用头打分。报告 Δhead(k)，k=50…1000 与全池 | '
      f'Δhead(400) {cm("ba_ge30", "dhead_PT", 400).mean():+.2f} / {cm("oa", "dhead_PT", 400).mean():+.2f}；'
      f'Δhead(全池) {cm("ba_ge30", "dfull_PT", 50).mean():+.2f} / {cm("oa", "dfull_PT", 50).mean():+.2f}'
      f'（AE {cm("ba_ge30", "dfull_PT", 50, f="ae").mean():+.2f}、S2 {cm("ba_ge30", "dfull_PT", 50, f="s2").mean():+.2f}）；'
      f'去 RO/PT 全池 {cm("ba_ge30", "dfull_PT", 50, drop=True).mean():+.2f} / {cm("oa", "dfull_PT", 50, drop=True).mean():+.2f} | '
      '它是 λ=0.1 工作点上的量，不是决策层能力上限；类均衡头的 OA 不是 OA 的决策层上界；全池也不是逐国严格上界 |')
    P('| **表征层剩余**（rev4 D1） | 只用目标池全部标签训练的模型（tgt300），与决策层上界之间的差距 | '
      'tgt300（model_diag，只用目标池训练，只有划分 0，训练集 = pool_positions(S,0,0)）− D1 全池头（λ=0.1，s0/r0 单元，'
      '同一批约 3 万标签）。方案外计算，描述 | '
      f'BA {rb[("both", False)].res_ba.mean():+.2f}（AE {rb[("ae", False)].res_ba.mean():+.2f}、S2 {rb[("s2", False)].res_ba.mean():+.2f}；'
      f'去 RO/PT {rb[("both", True)].res_ba.mean():+.2f}）；OA {rb[("both", False)].res_oa.mean():+.2f}'
      f'（去 RO/PT {rb[("both", True)].res_oa.mean():+.2f}，含类均衡工作点的 OA 代价）| '
      '只有划分 0、每国 1 个单元；逐国可为负（tgt300 的 BA 可低于类均衡头），不是严格非负的"剩余" |')
    P('| **域残差**（检验 3 链式分解） | 链式替换中去掉类别构成（OA）或类别集合（ba_ge30）与样本量之后，'
      '只剩"数据来自哪个国家"的差距 | rev4：域残差包含决策层可挽回部分与表征层剩余两部分。G 缓冲格网划分 0 上能算的是'
      '**总缺口**的恒等式 tgt300 − k0 = Δhead(全池) + 表征层剩余；Δhead(全池) 的偏置同时吸收先验（类别构成）差异，'
      '所以它不等于"域残差中的决策层部分" | '
      f'划分 0 总缺口 BA {rb[("both", False)].gap_ba.mean():+.2f} = {rb[("both", False)].dec_ba.mean():+.2f}（头）+ '
      f'{rb[("both", False)].res_ba.mean():+.2f}（剩余） | 检验 3 的分解在随机 70/30、无缓冲划分上，BA 中类别集合占 13–15%、'
      'OA 中类别构成占 44–67%（CHECK3_RESULTS），与 G 缓冲格网上的数不能直接相加减；检验 3 的 BIAS_ORACLE'
      '（大量本国标签只调温度与偏置：OA 可挽回 11–14、ba_ge30 6–7 pp）是第三种操作化，同样不与 D1 并列。'
      '"域残差 = 两部分之和"在 BA 上与检验 3 大致相容，在 OA 上的表述需作者决定 |')
    P('| 缓冲格网决策层上界（H1 附；另一种操作化） | 用候选集全部标签拟合温度与偏置的上界 | '
      'oracle_G_buffered.py：拟合集 = **全部缓冲候选集**（每单元 27,673–65,357 个，不是 3 万池）；**自然先验**（不加类均衡权重）；'
      '**无正则**（λ=0，b 无界）；BA 版在拟合后按候选集先验**改权**到出现类的均匀先验 | '
      f'OA AE {ora["ae"].dec_oa:+.2f} / S2 {ora["s2"].dec_oa:+.2f}；BA AE {ora["ae"].dec_ba:+.2f} / S2 {ora["s2"].dec_ba:+.2f} | '
      '与 D1 的全池头（λ=0.1、类均衡 NLL、3 万池）不是同一个量：拟合集、权重、正则、BA 的得到方式都不同，不相除、不并列 |')
    P('| 类均衡决策头 − 重训（D1 标签配对版） | 同一批 G SRS 选样上，只用头与并入重训之差 | 方案外核对 | '
      f'k=200：BA {cm("ba_ge30", "diffGsel", 200).mean():+.2f} / OA {cm("oa", "diffGsel", 200).mean():+.2f} | '
      'H5 的 +0.94 是 F3b α=0.4 的对数混合头（(1−α)·log P_retrain + α·log softmax(类均衡头)）在 k=200 相对重训的 BA 增益'
      '（frontier2_report.txt），不是纯类均衡头，也不是"决策层可兑现量"，不能与 D1 的数字并列 |')
    P('| λ 工作点（方案外，描述） | 岭罚 λ 决定头在 BA 与 OA 之间的位置 | rev4 预先固定 λ=0.1；λ=1e-2、1e-4 只作敏感性描述（复核者 lam_sens） | '
      f'k=400，11 国：λ=0.1 {lt[(LAM, False)][400][0]:+.2f} / {lt[(LAM, False)][400][1]:+.2f}；'
      f'λ=1e-2 {lt[(1e-2, False)][400][0]:+.2f} / {lt[(1e-2, False)][400][1]:+.2f}；'
      f'λ=1e-4 {lt[(1e-4, False)][400][0]:+.2f} / {lt[(1e-4, False)][400][1]:+.2f} | 不用于选 λ；不把低 λ 的平台写成"决策层上界" |')
    P('')

    txt = '\n'.join(lines)
    open(os.path.join(OUT, 'decision_curve_summary.txt'), 'w').write(txt + '\n')
    print(txt)
    plot(CV, nbig, npool)
    print(f'done {time.time() - t0:.0f}s')


# ---------------------------------------------------------------- 图注
def caption_text(nbig, npool):
    others = nbig.drop(['RO', 'LV'])
    return ('D1 决策层标签效率曲线（C 级，描述性；G s0/s1 测试集复用）。上行 11 国，下行去掉 RO、PT 的 9 国；'
            '左、中两列为相对 k=0 的增益（左 ba_ge30，中 OA），右列为标签配对版 头 − 重训。'
            '蓝 = 只调决策层（温度 + 类偏置，λ=0.1 预先固定，类均衡 NLL），每单元 5 个 D1 子样本平均，相对 argmax(P_T)；'
            '橙 = 并入重训（G 的 SRS(k) − foreign_only，r=50，每单元 1 次选样）；左、中两列中头与重训用不同子样本（非配对）。'
            f'蓝色点线 = 同 λ 全池头（每单元 {npool.min():,}–{npool.max():,} 个候选池标签），标注为国家均值。'
            '右列（方案外核对）：同一批 G SRS 选样上拟合同一头，减去同单元重训，共同底座 G foreign_only；绿 = ba_ge30，黄 = OA。'
            '实线 / 实心 = AE 特征，虚线 / 空心 = S2 特征；误差线为国家层 t 95% CI（11 国 df=10，9 国 df=8）。'
            f'ba_ge30 = 测试集中 ≥30 个地块的类的平均召回：RO 只含 {nbig["RO"]} 类、LV 只含 {nbig["LV"]} 类，其余国家 '
            f'{others.min()}–{others.max()} 类。正文与表中的"both" = 两特征国家值的平均（作者选定口径，rev4 未定义），'
            '图中两特征分开画。数值保留两位小数。')


def wrap_cjk(text, width):
    """按显示宽度折行（全角字符计 2，其余计 1）。"""
    import unicodedata
    out, cur, w = [], '', 0
    for ch in text:
        cw = 2 if unicodedata.east_asian_width(ch) in ('W', 'F') else 1
        if w + cw > width:
            out.append(cur)
            cur, w = '', 0
        cur += ch
        w += cw
    if cur:
        out.append(cur)
    return out


# ---------------------------------------------------------------- 图
def plot(CV, nbig, npool):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.ticker  # noqa: F401
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'Heiti SC']
    plt.rcParams['axes.unicode_minus'] = False
    INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
    COL = {'head': '#2a78d6', 'ret': '#eb6834', 'ba_ge30': '#1baf7a', 'oa': '#eda100'}  # 参考色板 1–4 号，按序
    LSTY = {'ae': '-', 's2': (0, (5, 3))}
    MK = {'ae': 'o', 's2': 's'}
    XOFF = {'ae': 0.965, 's2': 1.035}

    def style(ax, xlim, ks):
        ax.set_facecolor(SURF)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xscale('log')
        ax.set_xticks(ks)
        ax.set_xticklabels([str(k) for k in ks])
        ax.minorticks_off()
        ax.set_xlim(*xlim)
        ax.set_xlabel('目标国标签数 k（对数轴）', color=INK)
        ax.grid(axis='y', color=GRID, lw=0.6)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        for sp in ('left', 'bottom'):
            ax.spines[sp].set_color(INK2)
        ax.tick_params(colors=INK2)

    def sel(met, f, drop):
        c = CV[(CV.metric == met) & (CV.clf == f)]
        return c[~c.target.isin(DROP)] if drop else c

    def eb(ax, ks, st, col, f, xoff):
        m = np.array([s['mean'] for s in st])
        lo = np.array([s['lo95'] for s in st])
        hi = np.array([s['hi95'] for s in st])
        ax.errorbar(np.array(ks) * xoff, m, yerr=[m - lo, hi - m], color=col, ls=LSTY[f], lw=2, marker=MK[f], ms=6,
                    mfc=col if f == 'ae' else SURF, mec=col, mew=1.5, elinewidth=0.8, capsize=0, alpha=0.95)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10.2), facecolor=SURF)
    letters = iter('abcdef')
    for row, drop in enumerate((False, True)):
        rtag = '去掉 RO、PT（9 国）' if drop else '11 国'
        for ci, (met, name) in enumerate((('ba_ge30', '平衡精度 ba_ge30'), ('oa', '总体精度 OA'))):
            ax = axes[row, ci]
            style(ax, (38, 3000), KS)
            for f in FEATS:
                c = sel(met, f, drop)
                eb(ax, KS, [tsum(c[c.k == k].dhead_PT) for k in KS], COL['head'], f, XOFF[f])
                eb(ax, KS_RET, [tsum(c[c.k == k].dret) for k in KS_RET], COL['ret'], f, XOFF[f])
                full = c[c.k == 50].dfull_PT.mean()
                ax.axhline(full, color=COL['head'], ls=(0, (1, 2)), lw=1.4 if f == 'ae' else 1.0,
                           alpha=0.9 if f == 'ae' else 0.6)
                ax.annotate(f'全池头 {f.upper()} {full:+.2f}', (2950, full), textcoords='offset points',
                            xytext=(0, 3 if f == 'ae' else -10), fontsize=8, color=INK2, ha='right')
            ax.set_ylabel('相对 k=0 的增益（pp）', color=INK)
            ax.set_title(f'({next(letters)}) {rtag}：{name}', color=INK, fontsize=11, loc='left')
        # 配对差值面板
        ax = axes[row, 2]
        style(ax, (38, 720), KS_RET)
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        for met, lab in (('ba_ge30', 'ba_ge30'), ('oa', 'OA')):
            for f in FEATS:
                c = sel(met, f, drop)
                st = [tsum(c[c.k == k].diffGsel) for k in KS_RET]
                eb(ax, KS_RET, st, COL[met], f, XOFF[f])
            # 直接标注（身份不只靠颜色）
            yb = np.mean([sel(met, f, drop)[lambda x: x.k == 400].diffGsel.mean() for f in FEATS])
            ax.annotate(lab, (400 * 1.13, yb), fontsize=9, color=INK, va='center', ha='left')
        ax.set_ylabel('头 − 重训（pp，同一批标签）', color=INK)
        ax.set_title(f'({next(letters)}) {rtag}：头 − 重训（标签配对，方案外）', color=INK, fontsize=11, loc='left')

    h = [Line2D([], [], color=COL['head'], lw=2, label='只调决策层（温度+类偏置，λ=0.1，类均衡 NLL）'),
         Line2D([], [], color=COL['ret'], lw=2, label='并入重训（G 的 SRS 行，r=50）'),
         Line2D([], [], color=COL['head'], lw=1.4, ls=(0, (1, 2)), label='同 λ 全池决策头（每单元约 3 万候选池标签）'),
         Line2D([], [], color=COL['ba_ge30'], lw=2, label='配对差值：ba_ge30'),
         Line2D([], [], color=COL['oa'], lw=2, label='配对差值：OA'),
         Line2D([], [], color=INK2, lw=2, ls=LSTY['ae'], marker='o', label='AE 特征'),
         Line2D([], [], color=INK2, lw=2, ls=LSTY['s2'], marker='s', mfc=SURF, label='S2 特征')]
    fig.legend(handles=h, loc='upper center', ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.962))
    fig.text(0.01, 0.985, 'D1 决策层标签效率曲线（rev4 D1；v2）', fontsize=13, color=INK, va='top')
    cap = '\n'.join(wrap_cjk(caption_text(nbig, npool), 252))
    fig.text(0.01, 0.005, cap, fontsize=8.2, color=INK2, va='bottom', ha='left', linespacing=1.35)
    fig.tight_layout(rect=(0, 0.1, 1, 0.915), h_pad=2.2, w_pad=1.6)
    fig.savefig(os.path.join(OUT, 'fig_decision_curve.png'), dpi=200, facecolor=SURF)


if __name__ == '__main__':
    main()
