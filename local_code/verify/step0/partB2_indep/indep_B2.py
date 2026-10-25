"""
第 0 步补遗 B2 的独立复核实现（规格 label_expansion/STEP0_ADDENDUM_B2_2026-09-29.md，sha256[:16]=0c4b31ebc681b90f；
所引第 0 步规格 STEP0_SPEC_2026-09-29.md B 部分，sha256[:16]=14c8de7365278bac）。

只依据补遗规格独立编写：τ 族、包络插值、FA、PK 重加权都在本文件内自写；未读 partB2/ 下任何文件。
check3_priors.reweight / laplace_prior 按规格为定义，本文件另写一份并逐单元断言两者一致。
不训练任何分类器；BCTS 决策头只在冻结 P_T 候选行上拟合温度与偏置（calib_diag.fit_bcts，源码哈希与 D1 相同）。
候选标签只用于构造 p̂ 与拟合头；测试标签只用于评估（PKo 用测试先验，是规格定义的上界臂，只作描述）。
C 级、描述性。

臂（每单元 × 特征 f ∈ {ae, s2}）：
  Z0      P_T_f 测试行（float64 行归一）
  Z0EM    Z0 经 em_prior 先验修正（只用于次要参照 Z0+ = Z0 ∪ Z0EM 的上包络）
  PK{k}   同 RT{k} 那批 G SRS 选样的标签 → p̂ = laplace_prior(y_sel, 10, 1)；Z0 的概率按 p̂/π_train 重加权
  PKo     p̂ = 测试集真实类别比例（上界，与 k 无关）
  RT{k}   G SRS(k) 选样重训 LightGBM 的 i300 测试概率
  HD{k}   同一批选样在 P_T 候选行上拟合 BCTS（λ=0.1，类均衡权重），用到测试行
τ 族：ℓ = log P − τ·log π̂，π̂ = 该臂测试集平均预测概率；精确 log（P=0 → −inf；π̂_c=0 → 偏移取 0）；
      τ ∈ [−2, 2] 步长 0.05（81 点）。旧网格 [−0.5, 2] 是其子集，用于与 partB 对账。
FA：OA★ = Z0 在 τ=0 的 OA；W = OA★ + {−4.0, −3.9, …, 0}；每个 o 取所有相邻 τ 段（线性插值）上 BA 的最大值；
    FA = 可达 o 的平均；可达率 <1 标部分可达，<0.5 记缺失。

用法：python -B indep_B2.py --smoke   # 2 个单元
      python -B indep_B2.py           # 全量 44 单元
"""
import os
import sys
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_v] = '2'
sys.dont_write_bytecode = True
import ast
import glob
import hashlib
import time

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

PROJ = '/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops/label_expansion'
sys.path.insert(0, PROJ)
os.chdir(PROJ)  # guidance_G 的导入链依赖项目目录；只读
import guidance_G as G  # noqa: E402
import check3_priors as CP  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
ADD = os.path.join(PROJ, 'STEP0_ADDENDUM_B2_2026-09-29.md')
SPEC = os.path.join(PROJ, 'STEP0_SPEC_2026-09-29.md')
ADD_HASH, SPEC_HASH = '0c4b31ebc681b90f', '14c8de7365278bac'
RG = '/Users/perng_mini/AlphaCrops_local/results/guidance_G'
PARTB = '/Users/perng_mini/AlphaCrops_local/verify/step0/partB'
CALIB = '/Users/perng_mini/AlphaCrops_local/verify/pivot_eval/prior_shift/calib_diag.py'
BCTS_HASH = {'fit_bcts': '265d1037406c408b', 'weights': '70a88077203bb3c5'}

NC = 10
KS = (50, 200, 400)
FEATS = ('ae', 's2')
COUNTRIES = ('AT', 'BE', 'DE', 'EE', 'LT', 'LV', 'PT', 'RO', 'SE', 'SI', 'SK')
NINE = tuple(c for c in COUNTRIES if c not in ('RO', 'PT'))
TAU = np.round(np.linspace(-2.0, 2.0, 81), 2)          # 新网格
assert len(TAU) == 81 and TAU[0] == -2.0 and TAU[-1] == 2.0
T0 = int(np.flatnonzero(TAU == 0.0)[0])
OLDMASK = TAU >= -0.5 - 1e-12                           # 旧网格 [−0.5, 2]（51 点）
assert OLDMASK.sum() == 51
WOFF = np.arange(-40, 1) / 10.0                         # 窗口偏移（41 点）
TOL = 1e-9
ARMS = (['Z0', 'Z0EM', 'PKo'] + [f'PK{k}' for k in KS] + [f'RT{k}' for k in KS] + [f'HD{k}' for k in KS])
ENVS = {a: [a] for a in ARMS if a != 'Z0EM'}
ENVS['Z0+'] = ['Z0', 'Z0EM']


def sha16(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:16]


def get_bcts():
    """从 calib_diag.py 取出 fit_bcts、weights 的源码并执行（与 D1、step0B 同源），核对源码哈希。"""
    src = open(CALIB, encoding='utf-8').read()
    ns = {'np': np, 'minimize': minimize, 'C': NC}
    h = {}
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name in BCTS_HASH:
            seg = ast.get_source_segment(src, node)
            h[node.name] = hashlib.sha256(seg.encode()).hexdigest()[:16]
            exec(compile(seg, CALIB, 'exec'), ns)
    assert h == BCTS_HASH, h
    return ns['fit_bcts'], ns['weights']


FIT_BCTS, WEIGHTS = get_bcts()


# ------------------------------------------------------------------ 自写：先验重加权、τ 族、曲线、包络
def my_reweight(P, pnew, ptr):
    """q ∝ P·pnew/ptr；ptr=0 的类 q=0；行和为 0 的行保留原 P。返回 (q, 回退行数)。"""
    ratio = np.zeros(NC)
    ok = ptr > 0
    ratio[ok] = pnew[ok] / ptr[ok]
    q = P * ratio[None, :]
    rs = q.sum(axis=1)
    fb = ~(rs > 0) | ~np.isfinite(rs)
    q[~fb] /= rs[~fb, None]
    q[fb] = P[fb]
    return q, int(fb.sum())


def my_laplace(y, alpha=1.0):
    cnt = np.bincount(y, minlength=NC).astype(float)
    return (cnt + alpha) / (len(y) + alpha * NC)


def exact_log(P):
    out = np.full(P.shape, -np.inf)
    pos = P > 0
    out[pos] = np.log(P[pos])
    return out


def softmax_rows(Z):
    m = Z.max(axis=1, keepdims=True)
    E = np.exp(Z - m)
    return E / E.sum(axis=1, keepdims=True)


def family_preds(logp, pihat):
    """logp：(n, C) 对数概率（或只差行常数的 logit）。返回 (len(TAU), n) 的预测类。"""
    shift = np.zeros(NC)
    pos = pihat > 0
    shift[pos] = np.log(pihat[pos])
    preds = np.empty((len(TAU), logp.shape[0]), dtype=np.int16)
    for i, t in enumerate(TAU):
        preds[i] = np.argmax(logp - t * shift[None, :], axis=1)
    return preds


def curve_metrics(preds, yT, big):
    hit = preds == yT[None, :]
    oa = 100.0 * hit.mean(axis=1)
    rec = np.column_stack([hit[:, yT == c].mean(axis=1) for c in big])
    ba = 100.0 * rec.mean(axis=1)
    return oa, ba


def env_one(oa, ba, grid):
    """一条曲线在各 o 上的上包络 BA：遍历相邻 τ 段，OA 落在段内（容差 TOL）则线性插值；
    OA 差 ≤ TOL 的平段取两端 BA 的最大值；多段取最大。不可达为 NaN。"""
    res = np.full(len(grid), np.nan)
    a0, a1, c0, c1 = oa[:-1], oa[1:], ba[:-1], ba[1:]
    lo, hi = np.minimum(a0, a1), np.maximum(a0, a1)
    for gi, o in enumerate(grid):
        s = np.flatnonzero((o >= lo - TOL) & (o <= hi + TOL))
        if len(s) == 0:
            continue
        d = a1[s] - a0[s]
        flat = np.abs(d) <= TOL
        lam = np.clip((o - a0[s]) / np.where(flat, 1.0, d), 0.0, 1.0)
        v = np.where(flat, np.maximum(c0[s], c1[s]), c0[s] + lam * (c1[s] - c0[s]))
        res[gi] = v.max()
    return res


def env_multi(curves, grid):
    E = np.vstack([env_one(o, b, grid) for o, b in curves])
    with np.errstate(all='ignore'):
        out = np.where(np.isnan(E).all(0), np.nan, np.nanmax(np.where(np.isnan(E), -np.inf, E), 0))
    return out


def fa_from_env(e):
    reach = float(np.isfinite(e).mean())
    fa = float(np.nanmean(e)) if reach >= 0.5 else np.nan
    return fa, reach


# ------------------------------------------------------------------ 一个单元
def do_unit(D, S, GR, tgt, split, rep):
    unit = f'{tgt}_s{split}_r{rep}'
    y = np.asarray(D.y)
    cand, allT, test, fr = S['cand'], S['allT'], S['test'], S['foreign']
    ncand = len(cand)
    assert np.array_equal(allT[ncand:], test) and np.array_equal(allT[:ncand], cand)
    yT = y[test]
    nT = len(yT)
    big = np.flatnonzero(np.bincount(yT, minlength=NC) >= G.MIN_N)
    yF = y[fr]
    cw = G.class_weights_from(yF)
    pi_tr = np.bincount(yF, weights=cw[yF], minlength=NC) / cw[yF].sum()
    pi_test = np.bincount(yT, minlength=NC) / nT
    pos_of = {int(g): i for i, g in enumerate(allT[:ncand])}   # 全局行号 → P_T 行位置（候选部分）
    pool_pos = G.pool_positions(S, split, rep)

    # 选样（与 ae/s2 无关；逐特征核对 rows_t1 的 sel_hash）
    sels = {}
    for k in KS:
        z = np.load(os.path.join(RG, 'sel', f'{unit}_none_srs_k{k}.npz'))
        sel = np.asarray(z['sel'], dtype=np.int64)
        assert len(sel) == k
        h = G.sel_hash(sel)
        for f in FEATS:
            g = GR[(GR.target == tgt) & (GR.split == split) & (GR.rep == rep) & (GR.clf == f) & (GR.k == k)]
            assert len(g) == 1 and g.iloc[0].sel_hash == h, (unit, f, k)
        p = np.array([pos_of[int(s)] for s in sel])
        assert np.array_equal(np.sort(p), np.sort(pool_pos[z['sel_pos']]))  # 与 pool_positions 路线一致
        sels[k] = dict(sel=sel, pos=p, y=y[sel], hash=h)

    rows, crow, chk = [], [], []
    for f in FEATS:
        base = dict(target=tgt, split=split, rep=rep, unit=unit, clf=f)
        PTall = S[f'P_T_{f}'].astype(np.float64)
        PTt = PTall[ncand:]
        P0 = PTt / PTt.sum(1, keepdims=True)
        LP, meta = {}, {}
        LP['Z0'] = exact_log(P0)
        pe, it, conv, _ = CP.em_prior(P0, pi_tr)
        Qe, fbe = CP.reweight(P0, pe, pi_tr)
        LP['Z0EM'] = exact_log(Qe)
        meta['Z0EM'] = dict(em_iter=it, em_conv=conv, fallback=fbe)
        # PK：先验重加权（自写，与 check3_priors 核对）
        priors = {'PKo': pi_test}
        for k in KS:
            ph = CP.laplace_prior(sels[k]['y'], NC, alpha=1)
            assert np.allclose(ph, my_laplace(sels[k]['y']), atol=0, rtol=1e-15)
            priors[f'PK{k}'] = ph
        for name, ph in priors.items():
            Q, fb = my_reweight(P0, ph, pi_tr)
            Qr, fbr = CP.reweight(P0, ph, pi_tr)
            dmax = float(np.abs(Q - Qr).max())
            assert fb == fbr and dmax < 1e-12, (unit, f, name, dmax)
            LP[name] = exact_log(Q)
            meta[name] = dict(fallback=fb, prior_L1_test=float(np.abs(ph - pi_test).sum()),
                              n_zero_prior=int((ph == 0).sum()))
            chk.append(dict(base, check=f'reweight_{name}', value=dmax))
        # RT 与 HD
        LTc = np.log(np.clip(PTt, 1e-12, None))
        for k in KS:
            s_ = sels[k]
            g = GR[(GR.target == tgt) & (GR.split == split) & (GR.rep == rep) & (GR.clf == f) & (GR.k == k)].iloc[0]
            Pr = np.load(os.path.join(RG, 'probs', f'{unit}_{f}_k{k}_{s_["hash"]}.npz'))['i300'].astype(np.float64)
            Pr /= Pr.sum(1, keepdims=True)
            LP[f'RT{k}'] = exact_log(Pr)
            meta[f'RT{k}'] = dict(g_oa=g.oa, g_ba=g.ba_ge30)
            Lc = np.log(np.clip(PTall[s_['pos']], 1e-12, None))
            sc, bb = FIT_BCTS(Lc, s_['y'], WEIGHTS(s_['y'], np.ones(k), 'bal'), 0.1)
            LP[f'HD{k}'] = sc * LTc + bb
            meta[f'HD{k}'] = dict(s=float(sc), bmax=float(np.abs(bb).max()))

        CV = {}
        for a in ARMS:
            L = LP[a]
            Pa = softmax_rows(np.where(np.isneginf(L), -np.inf, L)) if a.startswith('HD') else np.exp(L)
            pihat = Pa.mean(0)
            oa, ba = curve_metrics(family_preds(L, pihat), yT, big)
            CV[a] = (oa, ba)
            for i, t in enumerate(TAU):
                crow.append(dict(base, arm=a, tau=float(t), oa=oa[i], ba_ge30=ba[i]))
        oa_star = CV['Z0'][0][T0]
        grid = oa_star + WOFF
        for e, comps in ENVS.items():
            for gname, msk in (('new', np.ones(len(TAU), bool)), ('old', OLDMASK)):
                env = env_multi([(CV[c][0][msk], CV[c][1][msk]) for c in comps], grid)
                fa, reach = fa_from_env(env)
                oa_all = np.concatenate([CV[c][0][msk] for c in comps])
                r = dict(base, arm=e, grid=gname, OA_star=oa_star, FA=fa, reach=reach,
                         full=bool(reach >= 1.0), partial=bool(0.5 <= reach < 1.0), missing=bool(reach < 0.5),
                         oa_min=float(oa_all.min()), oa_max=float(oa_all.max()),
                         n_top_miss=int(np.isnan(env[grid > oa_all.max() + TOL]).sum()),
                         n_bot_miss=int(np.isnan(env[grid < oa_all.min() - TOL]).sum()))
                if len(comps) == 1:
                    o_, b_ = CV[e]
                    r.update(oa_t0=o_[T0], ba_t0=b_[T0],
                             tau_argmax_oa=float(TAU[msk][np.argmax(o_[msk])]))
                for c in comps:
                    if c in meta:
                        r.update({f'{kk}': vv for kk, vv in meta[c].items()})
                rows.append(r)
        # 对账：RT τ=0 与 G rows
        for k in KS:
            chk.append(dict(base, check=f'RT{k}_t0_oa_vs_G', value=abs(CV[f'RT{k}'][0][T0] - meta[f'RT{k}']['g_oa'])))
            chk.append(dict(base, check=f'RT{k}_t0_ba_vs_G', value=abs(CV[f'RT{k}'][1][T0] - meta[f'RT{k}']['g_ba'])))
    return rows, crow, chk


# ------------------------------------------------------------------ 汇总
def tci(v):
    v = np.asarray(pd.Series(v, dtype=float).dropna())
    n = len(v)
    if n < 2:
        return dict(n=n, mean=np.nan, lo=np.nan, hi=np.nan, npos=int((v > 0).sum()))
    m, sd = v.mean(), v.std(ddof=1)
    h = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
    return dict(n=n, mean=m, lo=m - h, hi=m + h, npos=int((v > 0).sum()), sd=sd)


def fmt(s):
    if s['n'] < 2:
        return f'n={s["n"]}'
    return f'{s["mean"]:+.2f} [{s["lo"]:+.2f}, {s["hi"]:+.2f}] {s["npos"]}/{s["n"]}'


def cvals(U, col):
    """U：单元 × 特征级表。国家值 = 该国可用单元均值（逐特征）；both = 两特征国家值均值（任一缺失则缺失）。"""
    out = {f: U[U.clf == f].groupby('target')[col].mean().reindex(list(COUNTRIES)) for f in FEATS}
    out['both'] = pd.concat([out['ae'], out['s2']], axis=1).mean(axis=1, skipna=False)
    return out


def main():
    smoke = '--smoke' in sys.argv
    od = os.path.join(OUT, 'smoke') if smoke else OUT
    os.makedirs(od, exist_ok=True)
    assert sha16(ADD) == ADD_HASH and sha16(SPEC) == SPEC_HASH
    t0 = time.time()
    D = G.GData()
    GR = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(os.path.join(RG, 'rows_t1_*.csv')))], ignore_index=True)
    GR = GR[(GR.method == 'srs') & (GR.cond == 'none') & GR.k.isin(KS)]
    units = [('AT', 0, 0), ('RO', 1, 1)] if smoke else [(t, s, r) for t in COUNTRIES for s in (0, 1) for r in (0, 1)]
    R, CR, K = [], [], []
    cur = None
    for tgt, sp, rp in units:
        if cur is None or cur[0] != (tgt, sp):
            cur = ((tgt, sp), np.load(os.path.join(RG, 'stage0', f'{tgt}_s{sp}.npz')))
        a, b, c = do_unit(D, cur[1], GR, tgt, sp, rp)
        R += a
        CR += b
        K += c
        print(f'[{time.time() - t0:.0f}s] {tgt}_s{sp}_r{rp}', flush=True)
    FR, CRd, KR = pd.DataFrame(R), pd.DataFrame(CR), pd.DataFrame(K)
    FR.to_csv(os.path.join(od, 'indep_fa_rows.csv'), index=False)
    CRd.to_csv(os.path.join(od, 'indep_curves.csv'), index=False)
    KR.to_csv(os.path.join(od, 'indep_checks.csv'), index=False)
    lines = summarize(FR, CRd, KR, smoke)
    open(os.path.join(od, 'indep_summary.txt'), 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(f'耗时 {time.time() - t0:.0f}s')


def summarize(FR, CRd, KR, smoke):
    L = []
    P = L.append
    uk = ['target', 'split', 'rep', 'unit', 'clf']
    P('第 0 步补遗 B2 —— 独立复核实现（C 级、描述性；事后分析，规格在任何 B2 计算前登记）')
    P(f'补遗 sha256[:16]={sha16(ADD)}（登记 {ADD_HASH}）；第 0 步规格 {sha16(SPEC)}（登记 {SPEC_HASH}）；'
      f'fit_bcts/weights 源码哈希 {BCTS_HASH}')
    P(f'单元 {FR.unit.nunique()} × 特征 {FR.clf.nunique()}；τ ∈ [−2, 2] 步长 0.05（81 点）；W = [OA★−4, OA★] 步长 0.1（41 点）'
      + ('【冒烟】' if smoke else ''))
    P('')
    P('== 核对 ==')
    for pre in ('reweight_', 'RT'):
        x = KR[KR.check.str.startswith(pre)]
        P(f'  {pre}*：{len(x)} 项，最大 {x.value.max():.2e}')
    em = FR[(FR.arm == 'Z0+') & (FR.grid == 'new')]
    P(f'  EM：收敛 {int(em.em_conv.sum())}/{len(em)}；重加权回退行合计：'
      + '，'.join(f'{a} {int(FR[(FR.arm == a) & (FR.grid == "new")].fallback.sum())}'
                 for a in ['PKo'] + [f'PK{k}' for k in KS]))
    P(f'  PKo 中测试集为 0 的类（p̂=0）：单元 × 特征中 {int((FR[(FR.arm == "PKo") & (FR.grid == "new")].n_zero_prior > 0).sum())} 个有此情形')
    # 与 partB 对账（旧网格子集）
    fb = pd.read_csv(os.path.join(PARTB, 'fa_rows.csv'))
    fb = fb[fb.anchor == 'Z0']
    old = FR[FR.grid == 'old'].merge(fb[uk + ['arm', 'FA', 'reach', 'OA_star']], on=uk + ['arm'], suffixes=('', '_B'))
    for a in ['Z0', 'Z0+'] + [f'RT{k}' for k in KS] + [f'HD{k}' for k in KS]:
        x = old[old.arm == a]
        dfa = (x.FA - x.FA_B).abs()
        nm = int((x.FA.isna() != x.FA_B.isna()).sum())
        P(f'  旧网格子集 vs partB fa_rows {a:5s}：{len(x)} 行，FA 最大 |Δ| {dfa.max():.2e}，可达率最大 |Δ| '
          f'{(x.reach - x.reach_B).abs().max():.2e}，缺失不一致 {nm}，OA★ 最大 |Δ| {(x.OA_star - x.OA_star_B).abs().max():.2e}')
    cb = pd.read_csv(os.path.join(PARTB, 'curves_B.csv'))
    cm = CRd.merge(cb, on=uk + ['arm', 'tau'], suffixes=('', '_B'))
    for a in ['Z0', 'Z0EM', 'RT200', 'HD200']:
        x = cm[cm.arm == a]
        P(f'  曲线 vs partB curves_B {a:5s}（τ ∈ [−0.5,2]）：{len(x)} 点，OA 最大 |Δ| {(x.oa - x.oa_B).abs().max():.2e}，'
          f'BA 最大 |Δ| {(x.ba_ge30 - x.ba_ge30_B).abs().max():.2e}')
    P('')

    N = FR[FR.grid == 'new']
    O = FR[FR.grid == 'old']
    W = N.set_index(uk + ['arm'])['FA'].unstack('arm').reset_index()
    Wo = O.set_index(uk + ['arm'])['FA'].unstack('arm').reset_index()

    P('== 新网格可达率（88 单元 × 特征；全可达 / 部分 / 缺失；缺失或部分所在国家；窗口顶端/底端未达点合计）==')
    for a in ['Z0', 'Z0+', 'PKo'] + [f'PK{k}' for k in KS] + [f'RT{k}' for k in KS] + [f'HD{k}' for k in KS]:
        for tagd, X in (('新', N), ('旧', O)):
            x = X[X.arm == a]
            bad = sorted(x[~x.full].target.unique())
            P(f'  {a:5s} {tagd}  {int(x.full.sum()):3d} / {int(x.partial.sum()):3d} / {int(x.missing.sum()):3d}   '
              f'顶端未达 {int(x.n_top_miss.sum()):4d} 底端未达 {int(x.n_bot_miss.sum()):4d}   {",".join(bad) if bad else "—"}')
    P('')

    def diff(Wd, a, b):
        U = Wd[uk].copy()
        U['d'] = Wd[a] - Wd[b]
        return U

    EST = {}
    specs = [('A_o', 'PKo', 'Z0')]
    for k in KS:
        specs += [(f'A_{k}', f'PK{k}', 'Z0'), (f'R_{k}', f'RT{k}', f'PK{k}'), (f'dFA_RT{k}', f'RT{k}', 'Z0'),
                  (f'H_{k}', f'HD{k}', f'PK{k}'), (f'dFA_HD{k}', f'HD{k}', 'Z0'), (f'RT{k}-HD{k}', f'RT{k}', f'HD{k}'),
                  (f'dFA_RT{k}_vsZ0+', f'RT{k}', 'Z0+'), (f'dFA_HD{k}_vsZ0+', f'HD{k}', 'Z0+'),
                  (f'R_o{k}', f'RT{k}', 'PKo')]
    specs += [('Z0+-Z0', 'Z0+', 'Z0')]
    P('== 估计量（pp；国家值 = 可用单元均值；both = 两特征国家值均值；均值 [95% t CI] >0 国家数/国家数）==')
    P(f'  {"估计量":22s} {"特征":4s} | {"9 国（新网格）":32s} | {"11 国（新网格）":32s} | 9 国（旧网格） | 11 国（旧网格）')
    rows_out = []
    for name, a, b in specs:
        cvn = cvals(diff(W, a, b), 'd')
        cvo = cvals(diff(Wo, a, b), 'd')
        EST[name] = cvn
        EST[name + '_old'] = cvo
        nmiss = int(diff(W, a, b).d.isna().sum())
        for f in ('both', 'ae', 's2'):
            s9, s11 = tci(cvn[f].reindex(list(NINE))), tci(cvn[f])
            o9, o11 = tci(cvo[f].reindex(list(NINE))), tci(cvo[f])
            P(f'  {name:22s} {f:4s} | {fmt(s9):32s} | {fmt(s11):32s} | {fmt(o9)} | {fmt(o11)}'
              + (f'   （新网格缺失单元×特征 {nmiss}）' if f == 'both' else ''))
            for tag, s in (('9_new', s9), ('11_new', s11), ('9_old', o9), ('11_old', o11)):
                rows_out.append(dict(est=name, feat=f, set=tag, **s))
    pd.DataFrame(rows_out).to_csv(os.path.join(OUT if not smoke else os.path.join(OUT, 'smoke'),
                                               'indep_estimates.csv'), index=False)
    P('')

    P('== A_k 占 ΔFA(RT-k) 的比例（国家均值之比；只在 A_k 与 ΔFA 都非缺失的国家上）==')
    share = {}
    for k in KS:
        for f in ('both', 'ae', 's2'):
            for tag, cs in (('9 国', NINE), ('11 国', COUNTRIES)):
                a_ = EST[f'A_{k}'][f].reindex(list(cs))
                d_ = EST[f'dFA_RT{k}'][f].reindex(list(cs))
                ao = EST['A_o'][f].reindex(list(cs))
                m = a_.notna() & d_.notna()
                share[(k, f, tag)] = a_[m].mean() / d_[m].mean()
                m2 = ao.notna() & d_.notna()
                P(f'  k={k:3d} {f:4s} {tag:5s}  A_{k}/ΔFA = {a_[m].mean():+.3f}/{d_[m].mean():+.3f} = '
                  f'{share[(k, f, tag)]:.3f}（{int(m.sum())} 国）   A_o/ΔFA = {ao[m2].mean() / d_[m2].mean():.3f}')
    P('')
    # 可加性
    Uadd = W[uk].copy()
    Uadd['gap'] = (W['RT200'] - W['Z0']) - ((W['PK200'] - W['Z0']) + (W['RT200'] - W['PK200']))
    P(f'  可加性（单元层）ΔFA(RT200) − (A_200 + R_200) 最大 |.| = {Uadd.gap.abs().max():.2e}；'
      f'国家层 both 9 国：ΔFA − (A+R) = '
      f'{(EST["dFA_RT200"]["both"] - EST["A_200"]["both"] - EST["R_200"]["both"]).reindex(list(NINE)).abs().max():.2e}')
    P('')

    P('== 读法（事先写定；k=200，9 国，both）==')
    r9 = tci(EST['R_200']['both'].reindex(list(NINE)))
    sh = share[(200, 'both', '9 国')]
    if r9['lo'] > 0:
        verdict = 'R_200 的 95% CI 下界 > 0 → 扣掉先验信息后重训仍把前沿外推（"标签补上了一部分域残差"，C 级）'
    elif r9['lo'] <= 0 <= r9['hi'] and sh >= 0.70:
        verdict = 'R_200 CI 覆盖 0 且 A_200 占比 ≥ 70% → 同 OA 下增益主要来自类别先验信息'
    else:
        verdict = '其余情况 → 不确定，两者都有贡献'
    P(f'  R_200 = {fmt(r9)}；A_200 占比 = {sh:.3f} → {verdict}')
    P('')

    P('== 逐国值（both，新网格，pp）==')
    cols = ['A_o', 'A_50', 'A_200', 'A_400', 'R_50', 'R_200', 'R_400', 'dFA_RT200', 'H_200', 'dFA_HD200', 'RT200-HD200']
    P('  国家 ' + ' '.join(f'{c:>11s}' for c in cols))
    for t in COUNTRIES:
        P(f'  {t}   ' + ' '.join(f'{EST[c]["both"].get(t, np.nan):+11.2f}' for c in cols))
    return L


if __name__ == '__main__':
    main()
