"""
第 0 步 B 部分：工作点对齐回溯（规格 label_expansion/STEP0_SPEC_2026-09-29.md，sha256[:16]=14c8de7365278bac，B 节）。

只读 G 阶段 0 概率、G 的 SRS 重训概率（i300）、guidance_G_ml 的 LR/MLP 概率与 G 选样文件；不训练任何分类器。
BCTS 决策头只在冻结的 P_T 候选行上拟合温度与偏置（calib_diag.fit_bcts，λ=0.1，类均衡权重，与 D1 head_eval 相同）。
候选标签只用于模拟"采到的标签"（HD-k 拟合）；测试标签只用于评估。C 级、描述性（G s0/s1 测试集已多次复用）。

B1 单元与臂：44 个单元（11 国 × 划分 × 重复）× 特征 f ∈ {ae, s2}。
  Z0      = P_T_{f} 测试行（温度校准，D1 底座）
  Z0EM    = Z0 经 em_prior（π_train = 外国池按 class_weights_from 加权的类分布，与 G pi_train_of([],0) 相同）先验修正
  RT{k}   = G 的 SRS(k) 选样重训 LightGBM 的 i300 测试概率，k ∈ {50,200,400}
  HD{k}   = 同一批 SRS(k) 选样（sel/{单元}_none_srs_k{k}.npz 的 sel_pos）在 P_T 候选行上拟合 BCTS 头，再用于测试行
  RT200lr / RT200mlp、K0lr / K0mlp（次要）= guidance_G_ml/probs/{lr,mlp} 中 SRS k=200 与 k=0 的 P；
  K0{lr,mlp}EM = K0 经同一 em_prior 修正（只用于"本学习器零标签包络"的方案外对照）
B2 工作点族：ℓ = log P − τ·log π̂，π̂ = 该臂测试集平均预测概率（不用标签），τ ∈ [−0.5, 2.0] 步长 0.05。
  OA★ = Z0 在 τ=0 的 OA（每单元 × 特征）；W = [OA★−4, OA★] 步长 0.1（41 点）；
  BA 上包络：相邻 τ 点在 (OA, BA) 平面线性插值，多段经过 o 取最大；FA = 可达 o 上包络 BA 的平均；
  可达率 = W 中可达点的比例；<100% 标"部分可达"，<50% FA 记缺失。Z0+ = Z0 与 Z0EM 两条曲线的上包络。
B3 估计量：ΔFA(RT-k)、ΔFA(HD-k)、FA(RT-k)−FA(HD-k)；τ=0.5 点的 BA/OA 差；11 国与 9 国（去 RO、PT）。
B4 功效：国家层 SD → 确认运行（22 单元、只用 rep0）的预期 95% CI 半宽与 80% 功效最小可检出效应。
B5 split-half：测试集按 md5(f'step0B_{单元}') 随机两分；在 a 上定 τ(o)（含插值系数），用到 b，与 b 上直接包络比较；a→b、b→a 平均。

用法：python -B step0B.py --smoke   # 2 个单元，输出到 partB/smoke/，并做对账
      python -B step0B.py           # 全量
"""
import os
import sys
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_v] = '2'
sys.dont_write_bytecode = True  # 不在项目目录写 __pycache__
import ast
import glob
import hashlib
import time

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import brentq, minimize

L = '/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops/label_expansion'
sys.path.insert(0, L)
os.chdir(L)  # 与 D1 相同（guidance_G 的导入链依赖项目目录）；只读
import guidance_G as G  # noqa: E402
from check3_priors import em_prior, reweight  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SPEC = os.path.join(L, 'STEP0_SPEC_2026-09-29.md')
SPEC_HASH = '14c8de7365278bac'
R_G = '/Users/perng_mini/AlphaCrops_local/results/guidance_G'
R_ML = '/Users/perng_mini/AlphaCrops_local/results/guidance_G_ml'
D1_DIR = os.path.join(L, 'storyline', 'decision_curve')
D1_ROWS = os.path.join(D1_DIR, 'decision_curve_rows.csv')
CALIB = '/Users/perng_mini/AlphaCrops_local/verify/pivot_eval/prior_shift/calib_diag.py'
D1_FUNC_HASH = {'fit_bcts': '265d1037406c408b', 'weights': '70a88077203bb3c5'}  # D1 summary 记录的函数源码哈希
assert os.path.realpath(G.OUT) == os.path.realpath(R_G), G.OUT

C = 10
LAM = 0.1
FEATS = ['ae', 's2']
COUNTRIES = 'AT,BE,DE,EE,LT,LV,PT,RO,SE,SI,SK'.split(',')
DROP = ['RO', 'PT']
KS = [50, 200, 400]
TAUS = np.round(np.arange(-0.5, 2.0 + 1e-9, 0.05), 2)
I_T0 = int(np.flatnonzero(TAUS == 0.0)[0])
I_T05 = int(np.flatnonzero(TAUS == 0.5)[0])
OFFS = np.round(np.arange(-4.0, 0.0 + 1e-9, 0.1), 1)  # W 相对 OA★ 的偏移，41 点
EPS = 1e-9
MIN_COV = 0.5
MAIN_ARMS = ['Z0', 'Z0EM'] + [f'RT{k}' for k in KS] + [f'HD{k}' for k in KS]
ML_ARMS = ['RT200lr', 'RT200mlp', 'K0lr', 'K0lrEM', 'K0mlp', 'K0mlpEM']
CURVE_ARMS = MAIN_ARMS + ML_ARMS
# 包络（FA 对象）：名称 → 组成曲线
ENVS = {'Z0': ['Z0'], 'Z0EM': ['Z0EM'], 'Z0+': ['Z0', 'Z0EM'],
        **{f'RT{k}': [f'RT{k}'] for k in KS}, **{f'HD{k}': [f'HD{k}'] for k in KS},
        'RT200lr': ['RT200lr'], 'RT200mlp': ['RT200mlp'], 'K0lr': ['K0lr'], 'K0mlp': ['K0mlp'],
        'K0lr+': ['K0lr', 'K0lrEM'], 'K0mlp+': ['K0mlp', 'K0mlpEM']}
# 锚点 → (OA★ 取哪条曲线在 τ=0 的 OA, 在该锚点下计算 FA 的包络, 参照包络)
ANCHORS = {'Z0': ('Z0', ['Z0', 'Z0EM', 'Z0+'] + [f'RT{k}' for k in KS] + [f'HD{k}' for k in KS] +
                  ['RT200lr', 'RT200mlp', 'K0lr+', 'K0mlp+'], 'Z0+'),
           'own_lr': ('K0lr', ['K0lr+', 'RT200lr'], 'K0lr+'),       # 方案外：本学习器自己的零标签锚点
           'own_mlp': ('K0mlp', ['K0mlp+', 'RT200mlp'], 'K0mlp+')}  # 方案外
SH_ENVS = ['Z0', 'Z0+'] + [f'RT{k}' for k in KS] + [f'HD{k}' for k in KS]  # B5 所做的包络（锚点 Z0）


def sha16(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()[:16]


# ---------------------------------------------------------------- 从 calib_diag.py 原样取出 fit_bcts 与 weights（同 D1）
def load_calib_funcs():
    src = open(CALIB, encoding='utf-8').read()
    ns = dict(np=np, minimize=minimize, C=C)
    got = {}
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name in ('fit_bcts', 'weights'):
            code = ast.get_source_segment(src, node)
            exec(compile(code, CALIB, 'exec'), ns)
            got[node.name] = hashlib.sha256(code.encode()).hexdigest()[:16]
    assert got == D1_FUNC_HASH, got
    return ns['fit_bcts'], ns['weights']


fit_bcts, weights = load_calib_funcs()


# ---------------------------------------------------------------- 工作点族
def log_or_neginf(P):
    with np.errstate(divide='ignore'):
        return np.log(P)  # 概率恰为 0 的类 → −inf，任何 τ 下都不会被预测


def tau_preds(Z, pihat):
    """Z：(n, C) 对数概率（或同一行常数平移的 logit）；pihat：π̂（该臂测试集平均预测概率）。
    返回 (nτ, n) 的 argmax 预测。π̂_c = 0 的类，其列本身为 −inf，偏移取 0 以避免 −inf·0。"""
    lp = np.where(pihat > 0, np.log(np.where(pihat > 0, pihat, 1.0)), 0.0)
    return np.stack([(Z - t * lp).argmax(1) for t in TAUS]).astype(np.int8)


class Scorer:
    """与 guidance_G._scores 相同的 OA 与 ba_ge30（类集合 big 取全测试集中 ≥ MIN_N 的类），可限定在测试子集上。"""

    def __init__(self, yT, big):
        self.yT, self.big = yT, big

    def curve(self, PH, mask=None):
        yT = self.yT if mask is None else self.yT[mask]
        ph = PH if mask is None else PH[:, mask]
        corr = ph == yT[None, :]
        oa = 100.0 * corr.mean(1)
        rec = []
        for c in self.big:
            m = yT == c
            rec.append(corr[:, m].mean(1) if m.any() else np.full(len(PH), np.nan))
        ba = 100.0 * np.nanmean(np.stack(rec, 1), 1)
        return oa, ba


def envelope(curves, grid):
    """curves：[(oa (nτ,), ba (nτ,)), ...]（按 τ 排序）；grid：o 值。
    对每个 o：在所有曲线的相邻 τ 段上找 OA = o 的点（线性插值；OA 相同的平段取两端 BA 较大者），取 BA 最大。
    返回 best (G,)（不可达为 NaN）、所属曲线 cid、段号 seg、插值系数 lam（点 = (1−lam)·τ_seg + lam·τ_{seg+1}）。"""
    ng = len(grid)
    best = np.full(ng, -np.inf)
    cid = np.full(ng, -1)
    seg = np.full(ng, -1)
    lam = np.full(ng, np.nan)
    for ci, (oa, ba) in enumerate(curves):
        o0, o1, b0, b1 = oa[:-1], oa[1:], ba[:-1], ba[1:]
        lo, hi = np.minimum(o0, o1) - EPS, np.maximum(o0, o1) + EPS
        d = o1 - o0
        flat = np.abs(d) < EPS
        inside = (grid[:, None] >= lo[None]) & (grid[:, None] <= hi[None])
        lm = np.clip((grid[:, None] - o0[None]) / np.where(flat, 1.0, d)[None], 0.0, 1.0)
        lm = np.where(flat[None], (b1 > b0).astype(float)[None], lm)
        val = np.where(inside, b0[None] + lm * (b1 - b0)[None], -np.inf)
        j = val.argmax(1)
        v = val[np.arange(ng), j]
        upd = v > best
        best[upd], cid[upd], seg[upd], lam[upd] = v[upd], ci, j[upd], lm[np.arange(ng), j][upd]
    best[~np.isfinite(best)] = np.nan
    return best, cid, seg, lam


def fa_of(best):
    cov = float(np.isfinite(best).mean())
    fa = float(np.nanmean(best)) if cov >= MIN_COV else np.nan
    return fa, cov


def transfer(curves_dst, cid, seg, lam):
    """把源半集上找到的 (曲线, 段, 插值系数) 用到目标半集的同一曲线上，得到目标半集的 BA 与 OA。"""
    out_b = np.full(len(cid), np.nan)
    out_o = np.full(len(cid), np.nan)
    for g in np.flatnonzero(cid >= 0):
        oa, ba = curves_dst[cid[g]]
        j, lm = seg[g], lam[g]
        out_b[g] = ba[j] + lm * (ba[j + 1] - ba[j])
        out_o[g] = oa[j] + lm * (oa[j + 1] - oa[j])
    return out_b, out_o


def half_seed(unit):
    return int(hashlib.md5(f'step0B_{unit}'.encode()).hexdigest()[:8], 16)


# ---------------------------------------------------------------- 数据
def load_G_rows():
    GR = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(R_G, 'rows_t*_*.csv'))], ignore_index=True)
    GR = GR[(GR.cond == 'none') & GR.method.isin(['srs', 'foreign_only'])]
    return GR


def load_ML_rows():
    MR = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(R_ML, 'rows_*_*.csv'))], ignore_index=True)
    MR = MR[MR.learner.isin(['lr', 'mlp']) & (MR.cond == 'none') &
            (((MR.method == 'srs') & (MR.k == 200)) | (MR.method == 'foreign_only'))]
    assert (MR.status == 'ok').all()
    return MR


def norm64(P):
    P = np.asarray(P, dtype=np.float64)
    return P / P.sum(1, keepdims=True)


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
    sc = Scorer(yT, big)
    yF = y[fr]
    cw = G.class_weights_from(yF)
    pi_train = np.bincount(yF, weights=cw[yF], minlength=C) / cw[yF].sum()  # = G pi_train_of([], 0)
    pool_pos = G.pool_positions(S, split, rep)
    yP = y[allT[pool_pos]]
    rng = np.random.default_rng(half_seed(unit))
    perm = rng.permutation(nT)
    half_a = np.zeros(nT, bool)
    half_a[perm[:nT // 2]] = True
    halves = {'a': half_a, 'b': ~half_a}

    curve_rows, fa_rows, sh_rows, chk_rows, env_rows = [], [], [], [], []
    for f in FEATS:
        cell = dict(target=tgt, split=split, rep=rep, unit=unit, clf=f)
        P_all = S[f'P_T_{f}']
        PT = P_all[n_cand:].astype(np.float64)
        PP = P_all[pool_pos].astype(np.float64)
        Z = {}      # 臂 → (n, C) 对数概率 / logit
        meta = {}
        # Z0 与 Z0EM
        P0 = norm64(PT)
        Z['Z0'] = log_or_neginf(P0)
        pi_em, it_em, conv_em, _ = em_prior(P0, pi_train)
        Q, nfb = reweight(P0, pi_em, pi_train)
        Z['Z0EM'] = log_or_neginf(Q)
        meta['em'] = dict(em_iter=it_em, em_conv=conv_em, em_fallback=nfb)
        # RT-k（G 的 SRS 选样，LightGBM i300）与 HD-k（同一批选样上的 BCTS 头）
        ZTc = np.log(np.clip(PT, 1e-12, None))          # 与 D1 head_eval 相同的输入
        ZPc = np.log(np.clip(PP, 1e-12, None))
        for k in KS:
            g = GR[(GR.method == 'srs') & (GR.target == tgt) & (GR.split == split) & (GR.rep == rep) &
                   (GR.clf == f) & (GR.k == k)]
            assert len(g) == 1, (unit, f, k, len(g))
            g = g.iloc[0]
            pp = os.path.join(R_G, 'probs', f'{unit}_{f}_k{k}_{g.sel_hash}.npz')
            Pr = norm64(np.load(pp)['i300'])
            Z[f'RT{k}'] = log_or_neginf(Pr)
            zsel = np.load(os.path.join(R_G, 'sel', f'{unit}_none_srs_k{k}.npz'))
            sp = zsel['sel_pos']
            h_pos = G.sel_hash(allT[pool_pos][sp])
            assert h_pos == g.sel_hash == G.sel_hash(zsel['sel']), (unit, f, k)
            ys = yP[sp]
            s, b = fit_bcts(ZPc[sp], ys, weights(ys, np.ones(len(ys)), 'bal'), LAM)
            Z[f'HD{k}'] = s * ZTc + b                  # logit；与 log softmax 只差行常数
            meta[f'HD{k}'] = dict(s=s, bmax=float(np.abs(b).max()))
            chk_rows.append(dict(cell, arm=f'RT{k}', ref='G_rows', oa_ref=g.oa, ba_ref=g.ba_ge30))
        # LR / MLP（次要）
        for lrn in ('lr', 'mlp'):
            m2 = MR[(MR.learner == lrn) & (MR.method == 'srs') & (MR.target == tgt) & (MR.split == split) &
                    (MR.rep == rep) & (MR.clf == f)]
            m0 = MR[(MR.learner == lrn) & (MR.method == 'foreign_only') & (MR.target == tgt) & (MR.split == split) &
                    (MR.rep == rep) & (MR.clf == f)]
            assert len(m2) == 1 and len(m0) == 1
            P2 = norm64(np.load(os.path.join(R_ML, 'probs', lrn, f'{unit}_{f}_k200_{m2.iloc[0].sel_hash}.npz'))['P'])
            Pk0 = norm64(np.load(os.path.join(R_ML, 'probs', lrn, f'{tgt}_s{split}_{f}_k0.npz'))['P'])
            Z[f'RT200{lrn}'] = log_or_neginf(P2)
            Z[f'K0{lrn}'] = log_or_neginf(Pk0)
            pe, _, _, _ = em_prior(Pk0, pi_train)
            Z[f'K0{lrn}EM'] = log_or_neginf(reweight(Pk0, pe, pi_train)[0])
            chk_rows.append(dict(cell, arm=f'RT200{lrn}', ref='ML_rows', oa_ref=m2.iloc[0].oa,
                                 ba_ref=m2.iloc[0].ba_ge30))
            chk_rows.append(dict(cell, arm=f'K0{lrn}', ref='ML_rows', oa_ref=m0.iloc[0].oa, ba_ref=m0.iloc[0].ba_ge30))

        # 每条曲线：全测试集与两个半集
        CV = {}
        CVh = {h: {} for h in halves}
        pihats = {}
        for arm in CURVE_ARMS:
            Za = Z[arm]
            Pa = np.exp(Za - Za.max(1, keepdims=True))
            Pa /= Pa.sum(1, keepdims=True)
            pihat = Pa.mean(0)
            pihats[arm] = pihat
            PH = tau_preds(Za, pihat)
            CV[arm] = sc.curve(PH)
            for h, m in halves.items():
                CVh[h][arm] = sc.curve(PH, m)
            oa, ba = CV[arm]
            for i, t in enumerate(TAUS):
                curve_rows.append(dict(cell, arm=arm, tau=t, oa=oa[i], ba_ge30=ba[i]))
        # τ=0 对账记录
        for r in chk_rows:
            if r['unit'] == unit and r['clf'] == f and 'oa' not in r:
                r['oa'], r['ba_ge30'] = CV[r['arm']][0][I_T0], CV[r['arm']][1][I_T0]
        chk_rows.append(dict(cell, arm='Z0', ref='D1_base_PT', oa=CV['Z0'][0][I_T0], ba_ge30=CV['Z0'][1][I_T0]))
        for k in KS:
            chk_rows.append(dict(cell, arm=f'HD{k}', ref='D1_srs_G_check', oa=CV[f'HD{k}'][0][I_T0],
                                 ba_ge30=CV[f'HD{k}'][1][I_T0]))

        # FA（各锚点）
        for anc, (anc_arm, envs, ref_env) in ANCHORS.items():
            oa_star = CV[anc_arm][0][I_T0]
            grid = oa_star + OFFS
            bests = {}
            for e in envs:
                comps = ENVS[e]
                best, cid, seg, lam = envelope([CV[a] for a in comps], grid)
                bests[e] = best
                env_rows += [dict(cell, anchor=anc, arm=e, off=OFFS[g], o=grid[g], ba=best[g]) for g in range(len(grid))]
                fa, cov = fa_of(best)
                one = comps[0] if len(comps) == 1 else None
                allo = np.concatenate([CV[a][0] for a in comps])
                row = dict(cell, anchor=anc, arm=e, OA_star=oa_star, FA=fa, reach=cov,
                           partial=bool(MIN_COV <= cov < 1.0), fa_missing=bool(cov < MIN_COV),
                           oa_min=float(allo.min()), oa_max=float(allo.max()),
                           oa_t0=CV[one][0][I_T0] if one else np.nan, ba_t0=CV[one][1][I_T0] if one else np.nan,
                           oa_t05=CV[one][0][I_T05] if one else np.nan, ba_t05=CV[one][1][I_T05] if one else np.nan)
                if e.startswith('HD'):
                    row.update(meta[e])
                if e in ('Z0EM', 'Z0+'):
                    row.update(meta['em'])
                fa_rows.append(row)
            # 方案外：共同可达部分上的 ΔFA（相对参照包络）
            rb = bests[ref_env]
            for r in fa_rows[-len(envs):]:
                bb = bests[r['arm']]
                m = np.isfinite(bb) & np.isfinite(rb)
                r['cs_cov'] = float(m.mean())
                r['dFA_cs'] = float(np.mean(bb[m] - rb[m])) if m.mean() >= MIN_COV else np.nan

        # B5 split-half（锚点 Z0；各半集用自己的 OA★）
        for src, dst in (('a', 'b'), ('b', 'a')):
            grid_s = CVh[src]['Z0'][0][I_T0] + OFFS
            grid_d = CVh[dst]['Z0'][0][I_T0] + OFFS
            for e in SH_ENVS:
                comps = ENVS[e]
                best_s, cid, seg, lam = envelope([CVh[src][a] for a in comps], grid_s)
                fa_s, cov_s = fa_of(best_s)
                tb, to = transfer([CVh[dst][a] for a in comps], cid, seg, lam)
                fa_tr = float(np.nanmean(tb)) if cov_s >= MIN_COV else np.nan
                best_d, *_ = envelope([CVh[dst][a] for a in comps], grid_d)
                fa_d, cov_d = fa_of(best_d)
                sh_rows.append(dict(cell, arm=e, dir=f'{src}2{dst}', FA_src=fa_s, cov_src=cov_s, FA_transfer=fa_tr,
                                    FA_direct=fa_d, cov_dst=cov_d, bias=fa_d - fa_tr,
                                    oa_shift=float(np.mean(to[cid >= 0] - grid_d[cid >= 0]))
                                    if cov_s >= MIN_COV else np.nan))
    log(f'{unit} 完成')
    return curve_rows, fa_rows, sh_rows, chk_rows, env_rows


# ---------------------------------------------------------------- 汇总工具
def t_summary(v):
    v = np.asarray(pd.Series(v).dropna(), dtype=float)
    n = len(v)
    if n < 2:
        return dict(n=n, mean=np.nan, lo95=np.nan, hi95=np.nan, sd=np.nan, n_pos=int((v > 0).sum()))
    m, sd = v.mean(), v.std(ddof=1)
    q = stats.t.ppf(0.975, n - 1)
    return dict(n=n, mean=m, sd=sd, lo95=m - q * sd / np.sqrt(n), hi95=m + q * sd / np.sqrt(n),
                n_pos=int((v > 0).sum()))


def f95(s, nd=2):
    if s['n'] < 2:
        return f'n={s["n"]}（不足 2 国）'
    return f'{s["mean"]:+.{nd}f} [{s["lo95"]:+.{nd}f}, {s["hi95"]:+.{nd}f}]，>0 {s["n_pos"]}/{s["n"]}'


def country_vals(U, col):
    """U：单元级表（target, split, rep, clf, col）。国家值 = 可用单元的平均（逐特征）；both = 两特征国家值的平均
    （任一特征缺失则该国 both 缺失）。返回 {feat: Series(index=target)}。"""
    out = {}
    for f in FEATS:
        out[f] = U[U.clf == f].groupby('target')[col].mean()
    both = pd.concat([out['ae'], out['s2']], axis=1)
    out['both'] = both.mean(axis=1, skipna=False)
    return out


def mde80(sd, n, alpha=0.05, power=0.8):
    """单样本双侧 t 检验（df = n−1）80% 功效下的最小可检出效应（非中心 t 精确解）。"""
    if not np.isfinite(sd) or n < 3 or sd <= 0:
        return np.nan
    df = n - 1
    tc = stats.t.ppf(1 - alpha / 2, df)

    def pw(d):
        nc = d * np.sqrt(n) / sd
        return stats.nct.sf(tc, df, nc) + stats.nct.cdf(-tc, df, nc) - power
    hi = sd
    while pw(hi) < 0:
        hi *= 1.5
    return brentq(pw, 1e-9, hi)


# ---------------------------------------------------------------- 主程序
def main():
    smoke = '--smoke' in sys.argv
    out = os.path.join(HERE, 'smoke') if smoke else HERE
    os.makedirs(out, exist_ok=True)
    assert sha16(SPEC) == SPEC_HASH, '规格文件哈希与登记值不一致'
    t0 = time.time()
    D = G.GData()
    GR = load_G_rows()
    MR = load_ML_rows()
    units = ([('AT', 0, 0), ('RO', 1, 1)] if smoke else
             [(t, s, r) for t in COUNTRIES for s in (0, 1) for r in (0, 1)])
    CR, FR, SR, KR, ER = [], [], [], [], []
    cache = {}
    for tgt, split, rep in units:
        if (tgt, split) not in cache:
            cache.clear()
            cache[(tgt, split)] = np.load(os.path.join(R_G, 'stage0', f'{tgt}_s{split}.npz'))
        S = cache[(tgt, split)]
        a, b, c, d, e = run_unit(D, S, GR, MR, tgt, split, rep,
                              log=lambda s: print(f'[{time.time() - t0:.0f}s] {s}', flush=True))
        CR += a
        FR += b
        SR += c
        KR += d
        ER += e
    CRd, FRd, SRd, KRd, ERd = map(pd.DataFrame, (CR, FR, SR, KR, ER))
    ERd.to_csv(os.path.join(out, 'envelope_B.csv'), index=False)
    CRd.to_csv(os.path.join(out, 'curves_B.csv'), index=False)
    FRd.to_csv(os.path.join(out, 'fa_rows.csv'), index=False)
    SRd.to_csv(os.path.join(out, 'splithalf_rows.csv'), index=False)
    KRd.to_csv(os.path.join(out, 'check_rows.csv'), index=False)
    lines = summarize(CRd, FRd, SRd, KRd, ERd, smoke)
    open(os.path.join(out, 'summary_B.txt'), 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(f'总耗时 {time.time() - t0:.0f}s')


def summarize(CRd, FRd, SRd, KRd, ERd, smoke):
    lines = []
    P = lines.append
    P('第 0 步 B 部分：工作点对齐回溯 —— C 级、描述性（G s0/s1 测试集已多次复用；数字不进论文摘要，只用于写确认运行的登记方案）。')
    P(f'规格 STEP0_SPEC_2026-09-29.md sha256[:16]={sha16(SPEC)}（登记 {SPEC_HASH}）；calib_diag.py sha256[:16]={sha16(CALIB)}；'
      f'fit_bcts/weights 源码哈希与 D1 相同 {D1_FUNC_HASH}。')
    P(f'单元 {FRd.unit.nunique()} × 特征 {FRd.clf.nunique()}；τ 网格 {TAUS[0]}..{TAUS[-1]} 步长 0.05（{len(TAUS)} 点）；'
      f'W = [OA★−4, OA★] 步长 0.1（{len(OFFS)} 点）；可达率 <{MIN_COV:.0%} 记 FA 缺失。' + ('【冒烟：2 个单元】' if smoke else ''))
    P('臂：Z0 = P_T 测试行；Z0EM = Z0 经 em_prior 先验修正；Z0+ = Z0 与 Z0EM 两条 τ 曲线的上包络；RT-k = G SRS(k) 重训 i300；'
      'HD-k = 同一批 SRS(k) 选样上拟合的 BCTS 头（λ=0.1，类均衡）；RT200lr/RT200mlp、K0lr/K0mlp = guidance_G_ml 的 SRS k=200 / k=0。')
    P('国家值 = 该国可用单元（2 划分 × 2 重复）的平均；both = AE、S2 两特征国家值的平均（D1 口径）；推断单位 = 国家，95% CI 为 t 区间（df = n−1）。')
    P('')

    # ---------------- 对账
    P('== 对账（τ = 0）==')
    d1 = pd.read_csv(D1_ROWS)
    key = ['unit', 'clf']
    z0 = KRd[KRd.ref == 'D1_base_PT'].set_index(key)[['oa', 'ba_ge30']]
    b = d1[d1.design == 'base_PT'].set_index(key)[['oa', 'ba_ge30']]
    j = z0.join(b, rsuffix='_d1', how='inner')
    P(f'(1) Z0 τ=0 与 D1 base_PT：{len(j)} 行，最大 |Δ| OA {(j.oa - j.oa_d1).abs().max():.2e}，BA {(j.ba_ge30 - j.ba_ge30_d1).abs().max():.2e}')
    rt = KRd[KRd.ref == 'G_rows']
    P(f'(2) RT-k τ=0 与 G rows（srs，i300 的 oa、ba_ge30）：{len(rt)} 行，最大 |Δ| OA {(rt.oa - rt.oa_ref).abs().max():.2e}，'
      f'BA {(rt.ba_ge30 - rt.ba_ref).abs().max():.2e}；sel/ 的 sel_pos 经 pool_positions 映射后的 sel_hash 与 G rows 逐一相同（断言通过）')
    hd = KRd[KRd.ref == 'D1_srs_G_check'].copy()
    hd['k'] = hd.arm.str[2:].astype(int)
    b = d1[d1.design == 'srs_G_check'].set_index(key + ['k'])[['oa', 'ba_ge30']]
    j = hd.set_index(key + ['k'])[['oa', 'ba_ge30']].join(b, rsuffix='_d1', how='inner')
    P(f'(3) HD-k τ=0 与 D1 srs_G_check：{len(j)} 行，最大 |Δ| OA {(j.oa - j.oa_d1).abs().max():.2e}，BA {(j.ba_ge30 - j.ba_ge30_d1).abs().max():.2e}')
    ml = KRd[KRd.ref == 'ML_rows']
    P(f'(4) LR/MLP 的 SRS k=200 与 k=0 τ=0 与 guidance_G_ml rows：{len(ml)} 行，最大 |Δ| OA {(ml.oa - ml.oa_ref).abs().max():.2e}，'
      f'BA {(ml.ba_ge30 - ml.ba_ref).abs().max():.2e}')
    em = FRd[(FRd.anchor == 'Z0') & (FRd.arm == 'Z0EM')]
    P(f'(5) EM：收敛 {int(em.em_conv.sum())}/{len(em)}，迭代数中位数 {em.em_iter.median():.0f}，回退行合计 {int(em.em_fallback.sum())}')
    nz = CRd.groupby('arm').size()
    P(f'(6) 曲线行数：每臂 {sorted(set(nz.values))}（应 = 单元 × 特征 × {len(TAUS)}）')
    P('')

    FZ = FRd[FRd.anchor == 'Z0'].copy()
    ukey = ['target', 'split', 'rep', 'unit', 'clf']
    W = FZ.set_index(ukey + ['arm'])['FA'].unstack('arm').reset_index()
    T05 = FZ.set_index(ukey + ['arm'])[['ba_t05', 'oa_t05', 'ba_t0', 'oa_t0']].unstack('arm')
    CS = FZ.set_index(ukey + ['arm'])['dFA_cs'].unstack('arm').reset_index()

    # ---------------- 可达率分布
    P('== B2 可达率分布（锚点 = Z0 的 OA★；每臂 88 个单元 × 特征）==')
    P('  臂        特征  可达率 最小 / P10 / 中位 / 均值   全可达  部分可达  缺失   τ=0 的 OA 相对 OA★（中位 [P10, P90]）  缺失所在国家')
    for arm in ['Z0', 'Z0EM', 'Z0+'] + [f'RT{k}' for k in KS] + [f'HD{k}' for k in KS] + ['RT200lr', 'RT200mlp', 'K0lr+',
                                                                                      'K0mlp+']:
        for f in FEATS + ['all']:
            x = FZ[(FZ.arm == arm) & ((FZ.clf == f) if f != 'all' else True)]
            rel = x.oa_t0 - x.OA_star
            miss = sorted(x[x.fa_missing].target.unique())
            reltxt = (f'{rel.median():+6.2f} [{rel.quantile(.1):+6.2f}, {rel.quantile(.9):+6.2f}]'
                      if rel.notna().any() else '      —（包络）      ')
            P(f'  {arm:9s} {f:4s}  {x.reach.min():.2f} / {x.reach.quantile(.1):.2f} / {x.reach.median():.2f} / '
              f'{x.reach.mean():.2f}   {int((x.reach >= 1).sum()):3d}    {int(x.partial.sum()):3d}     '
              f'{int(x.fa_missing.sum()):3d}    {reltxt}   {",".join(miss) if miss else "—"}')
    P('')

    # ---------------- B3 估计量
    def unit_diff(a, b_):
        U = W[ukey].copy()
        U['d'] = W[a] - W[b_]
        return U

    def block(title, U, col='d'):
        P(title)
        cv = country_vals(U, col)
        nmiss = {f: int(U[U.clf == f][col].isna().sum()) for f in FEATS}
        for f in ('ae', 's2', 'both'):
            for drop in (False, True):
                v = cv[f]
                if drop:
                    v = v.drop([c for c in DROP if c in v.index])
                tag = '9 国' if drop else '11 国'
                P(f'  {f:4s} {tag:5s} {f95(t_summary(v))}' +
                  (f'   （缺失单元 {nmiss[f]}/{int((U.clf == f).sum())}）' if f != 'both' and not drop else ''))
        return cv

    P('######## B3 估计量（锚点 = Z0 的 OA★；pp）########')
    EST = {}
    for k in KS:
        EST[f'dFA_RT{k}'] = block(f'-- ΔFA(RT{k}) = FA(RT{k}) − FA(Z0+) --', unit_diff(f'RT{k}', 'Z0+'))
    for k in KS:
        EST[f'dFA_HD{k}'] = block(f'-- ΔFA(HD{k}) = FA(HD{k}) − FA(Z0+) --', unit_diff(f'HD{k}', 'Z0+'))
    for k in KS:
        EST[f'RT{k}-HD{k}'] = block(f'-- FA(RT{k}) − FA(HD{k}) --', unit_diff(f'RT{k}', f'HD{k}'))
    P('')
    P('-- 次要学习器（LR、MLP，k=200）：相对 LightGBM 的 Z0+（同一 OA★ 窗口）--')
    for lrn in ('lr', 'mlp'):
        EST[f'dFA_RT200{lrn}'] = block(f'  ΔFA(RT200{lrn}) = FA(RT200{lrn}) − FA(Z0+)', unit_diff(f'RT200{lrn}', 'Z0+'))
        block(f'  FA(K0{lrn}+) − FA(Z0+)（{lrn} 自身零标签包络，描述）', unit_diff(f'K0{lrn}+', 'Z0+'))
    P('')
    P('-- 方案外：LR、MLP 以本学习器 k=0 在 τ=0 的 OA 为锚点，FA(RT200) − FA(K0+)（同学习器零标签包络）--')
    for lrn in ('lr', 'mlp'):
        Fo = FRd[FRd.anchor == f'own_{lrn}'].set_index(ukey + ['arm'])['FA'].unstack('arm').reset_index()
        U = Fo[ukey].copy()
        U['d'] = Fo[f'RT200{lrn}'] - Fo[f'K0{lrn}+']
        block(f'  {lrn}：FA(RT200{lrn}) − FA(K0{lrn}+)', U)
        x = FRd[(FRd.anchor == f'own_{lrn}') & (FRd.arm == f'RT200{lrn}')]
        P(f'    RT200{lrn} 在本锚点下：全可达 {int((x.reach >= 1).sum())}，部分 {int(x.partial.sum())}，缺失 {int(x.fa_missing.sum())}（共 {len(x)}）')
    P('')
    P('-- 描述：EM 对零标签包络的贡献 FA(Z0+) − FA(Z0) --')
    block('', unit_diff('Z0+', 'Z0'))
    P('')

    # τ=0.5 与 τ=0 点
    def pt_block(title, tag, a, b_):
        P(title)
        for met, mn in (('ba', 'BA'), ('oa', 'OA')):
            U = T05[f'{met}_{tag}'][[a, b_]].reset_index()
            U['d'] = U[a] - U[b_]
            cv = country_vals(U, 'd')
            for f in ('ae', 's2', 'both'):
                for drop in (False, True):
                    v = cv[f]
                    if drop:
                        v = v.drop([c for c in DROP if c in v.index])
                    P(f'  {mn} {f:4s} {"9 国" if drop else "11 国":5s} {f95(t_summary(v))}')

    P('######## B3 附：τ = 0.5（可部署点，τ 预先固定、不看标签）的 BA 与 OA 差；对照 = Z0 在 τ=0.5 ########')
    for k in KS:
        pt_block(f'-- RT{k}(τ=.5) − Z0(τ=.5) --', 't05', f'RT{k}', 'Z0')
    for k in KS:
        pt_block(f'-- HD{k}(τ=.5) − Z0(τ=.5) --', 't05', f'HD{k}', 'Z0')
    for k in KS:
        pt_block(f'-- RT{k}(τ=.5) − HD{k}(τ=.5) --', 't05', f'RT{k}', f'HD{k}')
    for lrn in ('lr', 'mlp'):
        pt_block(f'-- RT200{lrn}(τ=.5) − Z0(τ=.5)（次要）--', 't05', f'RT200{lrn}', 'Z0')
    P('')
    P('######## 参照：τ = 0（原始工作点）的差，与 G/D1 已有量对应 ########')
    for k in KS:
        pt_block(f'-- RT{k}(τ=0) − Z0(τ=0) --', 't0', f'RT{k}', 'Z0')
        pt_block(f'-- HD{k}(τ=0) − Z0(τ=0) --', 't0', f'HD{k}', 'Z0')
    P('')

    P('######## 方案外：共同可达部分上的 ΔFA（只在两臂都可达的 o 上平均 BA 差；共同可达 <50% 记缺失）########')
    for arm in [f'RT{k}' for k in KS] + [f'HD{k}' for k in KS]:
        U = CS[ukey].copy()
        U['d'] = CS[arm]
        block(f'-- {arm} − Z0+（共同可达）--', U)
    EZ = ERd[ERd.anchor == 'Z0'].set_index(ukey + ['off', 'arm'])['ba'].unstack('arm')
    for k in KS:
        m = EZ[f'RT{k}'].notna() & EZ[f'HD{k}'].notna()
        g = (EZ[f'RT{k}'] - EZ[f'HD{k}']).where(m).groupby(ukey)
        U = pd.DataFrame({'d': g.mean(), 'cov': m.groupby(ukey).mean()}).reset_index()
        U.loc[U['cov'] < MIN_COV, 'd'] = np.nan
        block(f'-- RT{k} − HD{k}（共同可达）--', U)
    P('')

    P('######## 国家值（both = 两特征平均；pp；锚点 Z0）########')
    cols = ([f'dFA_RT{k}' for k in KS] + [f'dFA_HD{k}' for k in KS] + [f'RT{k}-HD{k}' for k in KS])
    P('  国家 ' + ' '.join(f'{c:>12s}' for c in cols))
    for t in COUNTRIES:
        P(f'  {t}   ' + ' '.join(f'{EST[c]["both"].get(t, np.nan):+12.2f}' for c in cols))
    P('')

    # ---------------- B4 功效
    P('######## B4 功效（国家层；确认运行 = 22 个单元 = 11 国 × 2 划分，只用 rep0）########')
    P('假设：确认运行的国家集合与本回溯相同（11 国或去 RO、PT 的 9 国）；国家间异质性与划分间变异与本回溯相同；国家值 = 该国 2 个单元'
      '（s0/r0、s1/r0）的平均；推断用单样本 t（df = n−1），双侧 α = 0.05；MDE 为 80% 功效下的精确非中心 t 解。')
    P('三种 SD：SD4 = 本回溯国家值（4 单元）的 SD；SDr0 = 只用 rep0 单元（2 单元，与确认运行同设计）的国家值 SD；'
      'SDproj = √(SD4² + σ̂²_rep/4)，σ̂²_rep = 同一 (国家, 划分) 两个重复之差的方差（(x_r0 − x_r1)²/2 的平均），'
      '即把每国 4 单元换成 2 单元时增加的重复间方差。三种 SD 都不含"确认运行换新测试集"的额外变异。')
    P('  估计量              特征 国家  n   SD4   SDr0  SDproj | 半宽(SD4 / SDr0 / SDproj) | MDE80(SD4 / SDr0 / SDproj)')
    pw_rows = []
    for name, (a, b_) in {**{f'dFA_RT{k}': (f'RT{k}', 'Z0+') for k in KS},
                          **{f'dFA_HD{k}': (f'HD{k}', 'Z0+') for k in KS},
                          **{f'RT{k}-HD{k}': (f'RT{k}', f'HD{k}') for k in KS}}.items():
        U = unit_diff(a, b_)
        cv4 = country_vals(U, 'd')
        cvr0 = country_vals(U[U.rep == 0], 'd')
        # 重复间方差：逐特征；both 用单元级两特征平均
        Ub = U.set_index(['target', 'split', 'rep', 'clf'])['d'].unstack('clf')
        Ub['both'] = Ub[['ae', 's2']].mean(axis=1, skipna=False)
        for f in ('ae', 's2', 'both'):
            for drop in (False, True):
                cs = [c for c in COUNTRIES if not (drop and c in DROP)]
                v4 = cv4[f].reindex(cs).dropna()
                vr0 = cvr0[f].reindex(cs).dropna()
                x = Ub[f].unstack('rep')
                x = x[x.index.get_level_values('target').isin(cs)].dropna()
                s2rep = float(((x[0] - x[1]) ** 2 / 2).mean()) if len(x) else np.nan
                sd4, sdr0 = v4.std(ddof=1), vr0.std(ddof=1)
                sdp = np.sqrt(sd4 ** 2 + s2rep / 4)
                n = len(v4)
                q = stats.t.ppf(0.975, n - 1)
                hw = [q * s / np.sqrt(n) for s in (sd4, sdr0, sdp)]
                md = [mde80(s, n) for s in (sd4, sdr0, sdp)]
                pw_rows.append(dict(est=name, feat=f, set='9' if drop else '11', n=n, sd4=sd4, sdr0=sdr0, sdproj=sdp,
                                    s2rep=s2rep, hw4=hw[0], hwr0=hw[1], hwproj=hw[2], mde4=md[0], mder0=md[1],
                                    mdeproj=md[2], mean=v4.mean()))
                P(f'  {name:18s} {f:4s} {"9" if drop else "11":>3s}  {n:2d}  {sd4:5.2f} {sdr0:5.2f} {sdp:6.2f} | '
                  f'{hw[0]:5.2f} / {hw[1]:5.2f} / {hw[2]:5.2f} | {md[0]:5.2f} / {md[1]:5.2f} / {md[2]:5.2f}   '
                  f'（本回溯均值 {v4.mean():+.2f}）')
    P('')

    # ---------------- B5 split-half
    P('######## B5 包络偏差（split-half；bias = FA_b（在 b 上直接画包络） − FA_{a→b}（τ(o) 在 a 上定、用到 b）；a→b 与 b→a 平均）########')
    P('做法：测试集按 md5(f"step0B_{单元}")[:8] 的种子随机两分；每个半集用自己的 OA★（Z0 在该半集 τ=0 的 OA）定窗口；'
      'τ(o) 以"曲线 + 相邻 τ 段 + 插值系数"表示，用到另一半的同一段、同一系数；π̂ 用全测试集（无标签）；ba_ge30 的类集合用全测试集的。')
    SRd = SRd.copy()
    SB = SRd.groupby(ukey + ['arm'])[['bias', 'FA_direct', 'FA_transfer']].mean().reset_index()
    P('  臂      单元×特征 偏差：均值 / 中位 / P10 / P90 / 最大 | 国家层 11 国 | 9 国')
    for arm in SH_ENVS:
        x = SB[SB.arm == arm]
        cv = country_vals(x, 'bias')
        P(f'  {arm:6s}  n={x.bias.notna().sum():3d}  {x.bias.mean():+.2f} / {x.bias.median():+.2f} / '
          f'{x.bias.quantile(.1):+.2f} / {x.bias.quantile(.9):+.2f} / {x.bias.max():+.2f} | '
          f'{f95(t_summary(cv["both"]))} | {f95(t_summary(cv["both"].drop(DROP, errors="ignore")))}')
    P('  -- 单方向（未平均）的差 FA_direct − FA_transfer：SD / P10 / P90；以及转移点在目标半集上的 OA 落点偏移（相对目标窗口，均值 / SD）--')
    for arm in SH_ENVS:
        x = SRd[SRd.arm == arm]
        P(f'  {arm:6s}  n={x.bias.notna().sum():3d}  SD {x.bias.std():.2f} / P10 {x.bias.quantile(.1):+.2f} / '
          f'P90 {x.bias.quantile(.9):+.2f}；OA 落点偏移 {x.oa_shift.mean():+.2f} / SD {x.oa_shift.std():.2f}')
    P('  -- ΔFA 的偏差（臂的偏差 − Z0+ 的偏差；即直接包络的 ΔFA 相对转移 ΔFA 的高估）--')
    piv = SRd.set_index(ukey + ['dir', 'arm'])[['FA_direct', 'FA_transfer']].unstack('arm')
    for arm in [f'RT{k}' for k in KS] + [f'HD{k}' for k in KS]:
        dd = (piv['FA_direct'][arm] - piv['FA_direct']['Z0+']) - (piv['FA_transfer'][arm] - piv['FA_transfer']['Z0+'])
        dd = dd.groupby(ukey).mean().rename('bias').reset_index()
        cv = country_vals(dd, 'bias')
        P(f'  Δ{arm:6s} n={dd.bias.notna().sum():3d}  均值 {dd.bias.mean():+.2f}，中位 {dd.bias.median():+.2f}，'
          f'P10 {dd.bias.quantile(.1):+.2f}，P90 {dd.bias.quantile(.9):+.2f} | 国家层 11 国 {f95(t_summary(cv["both"]))} | '
          f'9 国 {f95(t_summary(cv["both"].drop(DROP, errors="ignore")))}')
    for k in KS:
        a_, b_ = f'RT{k}', f'HD{k}'
        dd = (piv['FA_direct'][a_] - piv['FA_direct'][b_]) - (piv['FA_transfer'][a_] - piv['FA_transfer'][b_])
        dd = dd.groupby(ukey).mean().rename('bias').reset_index()
        cv = country_vals(dd, 'bias')
        P(f'  RT{k}−HD{k} n={dd.bias.notna().sum():3d}  均值 {dd.bias.mean():+.2f}，中位 {dd.bias.median():+.2f} | '
          f'国家层 11 国 {f95(t_summary(cv["both"]))} | 9 国 {f95(t_summary(cv["both"].drop(DROP, errors="ignore")))}')
    P('  半集覆盖：源半集可达率 <50% 的 (单元, 特征, 方向, 臂) 数 ' +
      '，'.join(f'{a}:{int((SRd[SRd.arm == a].cov_src < MIN_COV).sum())}' for a in SH_ENVS))
    P('')
    return lines


if __name__ == '__main__':
    main()
