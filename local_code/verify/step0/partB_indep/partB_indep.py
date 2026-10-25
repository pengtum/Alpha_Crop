"""
第 0 步 B 部分的独立复核实现（只依据 label_expansion/STEP0_SPEC_2026-09-29.md 的 B 部分；未读取 verify/step0/partB/）。

范围（按复核任务）：臂 Z0（不含 EM 点）、Z0+（含 EM）、RT-200、HD-200（LightGBM 底座），f ∈ {ae, s2}，44 单元；
FA 与可达率；ΔFA(RT-200)、ΔFA(HD-200)、FA(RT-200) − FA(HD-200) 的国家层结果（11 国与去 RO、PT 的 9 国）；
B5 split-half 包络偏差（只对 Z0+ 与 RT-200）。另附（方案外、只作核对）：τ=0.5 可部署点差值、共同可达子窗口版 ΔFA、
全测试集锚点版 split-half 偏差。

实现要点（全部自写：τ 族、上包络插值、FA、可达率、split-half）：
- 工作点族：ℓ = log P − τ·log π̂，π̂ = 该臂在测试集上的平均预测概率（不用标签）；τ ∈ {−0.50, −0.45, …, 2.00}（51 点）。
  log 前把概率与 π̂ 裁到 ≥ 1e-12；HD 臂直接用对数 softmax（s·log P_T + b − logsumexp），不经过裁剪。
- 锚点 OA★ = Z0 在 τ=0 的测试 OA（每单元、每特征）。窗口格点 o_j = OA★ − 0.1·(40 − j)，j = 0..40（41 点，含两端）。
- 上包络：曲线按 τ 顺序连成折线；对每个 o，取所有经过 o 的相邻段上线性插值的 BA 的最大值；水平段（两端 OA 相等）取两端 BA 较大者。
- 可达率 = 窗口 41 个格点中被曲线 OA 范围覆盖的比例（折线连续，覆盖 ⇔ o ∈ [min OA, max OA]）；
  <100% 标"部分可达"，FA 只在可达格点上平均；<50% 时 FA 缺失。
- Z0+：Z0 曲线与"EM 修正后再走同一 τ 族"的曲线（τ=0 即 EM 点）两条折线取上包络；可达格点为两者之并。
- RT-200：G 的 SRS k=200 选样（rows_t1 中 cond=none、method=srs）对应 probs 文件的 i300。
- HD-200：同一批 SRS 选样的候选行 P_T 上拟合 calib_diag.fit_bcts（λ=0.1，weights(mode='bal') 类均衡），应用到测试行。
- 国家值 = 该国 4 个单元（划分 × 重复）的平均（缺失单元剔除）；推断为国家层单样本 t（df = 国家数 − 1）。
- split-half：测试集按 md5(f'step0B_{单元}') 前 8 位十六进制作种子，default_rng(seed).permutation 前一半为 a、后一半为 b；
  π̂ 仍用整个测试集（无标签），ba_ge30 的类集合沿用整个测试集的 ≥30 类；
  主版本锚点在各半内重算（a→b 用 OA★_a 定窗口并在 a 上求 τ(o)；FA_b 直接在 b 上用 OA★_b 求包络）；
  偏差 = FA_b(直接包络) − FA_{a→b}(交叉)，两个方向平均；正值 = 包络的乐观偏差。
标签用途：候选标签只用于拟合 HD 头（模拟采到的标签）；测试标签只用于计算 OA/BA 曲线与包络（即 FA 的定义本身）。
不训练任何分类器；只读 G 的结果文件。
"""
import os
import sys

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_v] = '2'
sys.dont_write_bytecode = True

import ast  # noqa: E402
import hashlib  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from scipy.optimize import minimize  # noqa: E402
from scipy.special import logsumexp  # noqa: E402

L = '/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops/label_expansion'
sys.path.insert(0, L)
import guidance_G as G  # noqa: E402
from check3_priors import em_prior, reweight  # noqa: E402

R_G = '/Users/perng_mini/AlphaCrops_local/results/guidance_G'
OUT = '/Users/perng_mini/AlphaCrops_local/verify/step0/partB_indep'
CALIB = '/Users/perng_mini/AlphaCrops_local/verify/pivot_eval/prior_shift/calib_diag.py'
D1_ROWS = os.path.join(L, 'storyline', 'decision_curve', 'decision_curve_rows.csv')  # 只用于核对 τ=0 点
SPEC = os.path.join(L, 'STEP0_SPEC_2026-09-29.md')

C = 10
LAM = 0.1
K = 200
FEATS = ['ae', 's2']
COUNTRIES = 'AT,BE,DE,EE,LT,LV,PT,RO,SE,SI,SK'.split(',')
DROP = ['RO', 'PT']
TAUS = np.round(np.arange(-0.5, 2.0 + 1e-9, 0.05), 2)
NW, WSTEP = 41, 0.1  # 窗口 [OA★ − 4, OA★]，步长 0.1
EPS = 1e-9
CLIP = 1e-12
ARMS = ['Z0', 'Z0+', 'RT-200', 'HD-200']


# ---------------------------------------------------------------- fit_bcts / weights：按 D1 的做法从 calib_diag.py 原样取出
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
    assert set(got) == {'fit_bcts', 'weights'}
    return ns['fit_bcts'], ns['weights'], got


fit_bcts, weights, FUNC_HASH = load_calib_funcs()


# ---------------------------------------------------------------- 指标与 τ 族
def oa_ba(y, ph, big):
    """OA 与 ba_ge30（百分数）；big 中某类在子集里不存在时对其余类取平均（整测试集上不会发生）。"""
    cm = np.bincount(y * C + ph, minlength=C * C).reshape(C, C)
    n = cm.sum(1)
    rec = np.where(n[big] > 0, np.diag(cm)[big] / np.maximum(n[big], 1), np.nan)
    return 100.0 * np.trace(cm) / len(y), 100.0 * float(np.nanmean(rec))


def tau_argmax(Z, logpi, taus):
    """(n_tau, n) 的预测类：argmax(Z − τ·log π̂)。"""
    return np.stack([(Z - t * logpi).argmax(1) for t in taus])


def curve_from_preds(PH, y, big, idx=None):
    O, Bv = np.empty(len(PH)), np.empty(len(PH))
    yy = y if idx is None else y[idx]
    for i, ph in enumerate(PH):
        O[i], Bv[i] = oa_ba(yy, ph if idx is None else ph[idx], big)
    return O, Bv


def window(anchor):
    return anchor - WSTEP * np.arange(NW - 1, -1, -1)  # 末点精确等于 anchor


def envelope(O, Bv, taus, grid):
    """折线（按 τ 顺序）上 OA = o 处 BA 的上包络。返回 (env, tau_o)；不可达处为 nan。"""
    env = np.full(len(grid), -np.inf)
    tau_o = np.full(len(grid), np.nan)
    for i in range(len(O) - 1):
        o1, o2, b1, b2, t1, t2 = O[i], O[i + 1], Bv[i], Bv[i + 1], taus[i], taus[i + 1]
        m = (grid >= min(o1, o2) - EPS) & (grid <= max(o1, o2) + EPS)
        if not m.any():
            continue
        if abs(o2 - o1) < 1e-12:
            val = np.full(m.sum(), max(b1, b2))
            tt = np.full(m.sum(), t1 if b1 >= b2 else t2)
        else:
            w = np.clip((grid[m] - o1) / (o2 - o1), 0.0, 1.0)
            val = b1 + w * (b2 - b1)
            tt = t1 + w * (t2 - t1)
        idx = np.flatnonzero(m)
        better = val > env[idx] + 1e-12
        env[idx[better]] = val[better]
        tau_o[idx[better]] = tt[better]
    env[~np.isfinite(env)] = np.nan
    return env, tau_o


def envelope_multi(curves, grid):
    """多条曲线 [(O, B, taus)] 的上包络；返回 (env, tau_o, 曲线编号)。"""
    env = np.full(len(grid), np.nan)
    tau_o = np.full(len(grid), np.nan)
    cid = np.full(len(grid), -1)
    for j, (O, Bv, taus) in enumerate(curves):
        e, t = envelope(O, Bv, taus, grid)
        better = np.isfinite(e) & (~np.isfinite(env) | (e > env + 1e-12))
        env[better], tau_o[better], cid[better] = e[better], t[better], j
    return env, tau_o, cid


def fa_of(env):
    cov = float(np.isfinite(env).mean())
    fa_raw = float(np.nanmean(env)) if cov > 0 else np.nan
    return dict(cov=cov, partial=bool(cov < 1.0), fa_raw=fa_raw, FA=fa_raw if cov >= 0.5 else np.nan)


# ---------------------------------------------------------------- 一个臂的表示
class Arm:
    """logits Z（n_test, C，对数概率）与 log π̂（π̂ = 测试集平均预测概率）。"""

    def __init__(self, logP):
        self.Z = logP
        P = np.exp(logP - logsumexp(logP, axis=1, keepdims=True))
        self.pi = P.mean(0)
        self.logpi = np.log(np.clip(self.pi, CLIP, None))
        self.PH = tau_argmax(self.Z, self.logpi, TAUS)

    def pred_at(self, tau):
        return (self.Z - tau * self.logpi).argmax(1)


def logp_of(P):
    P = np.asarray(P, dtype=np.float64)
    P = P / P.sum(1, keepdims=True)
    return np.log(np.clip(P, CLIP, None))


def split_halves(unit, n):
    seed = int(hashlib.md5(f'step0B_{unit}'.encode()).hexdigest()[:8], 16)
    perm = np.random.default_rng(seed).permutation(n)
    return np.sort(perm[:n // 2]), np.sort(perm[n // 2:]), seed


def cross_fa(arms, y, big, sel_idx, sel_anchor, ev_idx):
    """在 sel_idx（半 a）上求曲线与窗口（锚点 sel_anchor），得 τ(o)，用到 ev_idx（半 b）上算 BA，平均。
    arms：一条或多条曲线（Z0+ 为 [Z0, EM]）。返回 (FA_cross, 覆盖率)。"""
    curves = []
    for a in arms:
        O, Bv = curve_from_preds(a.PH, y, big, sel_idx)
        curves.append((O, Bv, TAUS))
    grid = window(sel_anchor)
    env, tau_o, cid = envelope_multi(curves, grid)
    ok = np.isfinite(env)
    cov = float(ok.mean())
    if cov < 0.5:
        return np.nan, cov
    ba_b = [oa_ba(y[ev_idx], arms[cid[j]].pred_at(tau_o[j])[ev_idx], big)[1] for j in np.flatnonzero(ok)]
    return float(np.mean(ba_b)), cov


def direct_fa(arms, y, big, idx, anchor):
    curves = []
    for a in arms:
        O, Bv = curve_from_preds(a.PH, y, big, idx)
        curves.append((O, Bv, TAUS))
    env, _, _ = envelope_multi(curves, window(anchor))
    return fa_of(env)['FA']


def anchor_on(z0, y, big, idx):
    return oa_ba(y[idx], z0.PH[TAUS == 0.0][0][idx], big)[0]


# ---------------------------------------------------------------- 主计算
def compute():
    t0 = time.time()
    D = G.GData()
    y_all = np.asarray(D.y)
    assert np.sum(TAUS == 0.0) == 1 and np.sum(np.isclose(TAUS, 0.5)) == 1 and len(TAUS) == 51
    GR = pd.concat([pd.read_csv(os.path.join(R_G, f'rows_t1_{t}_s{s}_r{r}.csv'))
                    for t in COUNTRIES for s in (0, 1) for r in (0, 1)], ignore_index=True)
    SRS = GR[(GR.cond == 'none') & (GR.method == 'srs') & (GR.k == K)].set_index(['target', 'split', 'rep', 'clf'])
    assert len(SRS) == 88 and not SRS.index.duplicated().any()
    d1 = pd.read_csv(D1_ROWS)
    d1b = d1[d1.design == 'base_PT'].set_index(['unit', 'clf'])
    d1h = d1[(d1.design == 'srs_G_check') & (d1.k == K)].set_index(['unit', 'clf'])
    rows, checks = [], []
    for tgt in COUNTRIES:
        for split in (0, 1):
            S = np.load(os.path.join(R_G, 'stage0', f'{tgt}_s{split}.npz'))
            n_cand = len(S['cand'])
            allT, test, fr = S['allT'], S['test'], S['foreign']
            assert np.array_equal(allT[n_cand:], test)
            yT = y_all[test]
            big = np.flatnonzero(np.bincount(yT, minlength=C) >= G.MIN_N)
            yF = y_all[fr]
            cw = G.class_weights_from(yF)
            pi_train = np.bincount(yF, weights=cw[yF], minlength=C) / cw[yF].sum()  # = pi_train_of([], 0)
            for f in FEATS:
                PTall = S[f'P_T_{f}']
                PT = PTall[n_cand:].astype(np.float64)
                PTn = PT / PT.sum(1, keepdims=True)
                z0 = Arm(logp_of(PTn))
                pi_em, n_it, conv, _ = em_prior(PTn, pi_train)
                Pem, n_bad = reweight(PTn, pi_em, pi_train)
                zem = Arm(logp_of(Pem))
                O_z0, B_z0 = curve_from_preds(z0.PH, yT, big)
                O_em, B_em = curve_from_preds(zem.PH, yT, big)
                i0 = int(np.flatnonzero(TAUS == 0.0)[0])
                i5 = int(np.flatnonzero(np.isclose(TAUS, 0.5))[0])
                oa_star = O_z0[i0]
                grid = window(oa_star)
                for rep in (0, 1):
                    unit = f'{tgt}_s{split}_r{rep}'
                    key = (tgt, split, rep, f)
                    g = SRS.loc[key]
                    pool_pos = G.pool_positions(S, split, rep)
                    z = np.load(os.path.join(R_G, 'sel', f'{unit}_none_srs_k{K}.npz'))
                    sel_pos = z['sel_pos']
                    sel = allT[pool_pos][sel_pos]
                    assert np.array_equal(sel, z['sel']) and G.sel_hash(sel) == g.sel_hash, unit
                    # RT-200
                    Prt = np.load(os.path.join(R_G, 'probs', f'{unit}_{f}_k{K}_{g.sel_hash}.npz'))['i300']
                    rt = Arm(logp_of(Prt))
                    # HD-200：候选行 P_T（与 D1 相同：float64 直接取对数，不重新归一化）
                    ZP = np.log(np.clip(PTall[pool_pos][sel_pos].astype(np.float64), CLIP, None))
                    ys = y_all[sel]
                    s_, b_ = fit_bcts(ZP, ys, weights(ys, np.ones(len(ys)), 'bal'), LAM)
                    ZT_raw = np.log(np.clip(PT, CLIP, None))
                    lg = s_ * ZT_raw + b_
                    hd = Arm(lg - logsumexp(lg, axis=1, keepdims=True))
                    O_rt, B_rt = curve_from_preds(rt.PH, yT, big)
                    O_hd, B_hd = curve_from_preds(hd.PH, yT, big)
                    # 核对
                    chk = dict(unit=unit, clf=f,
                               z0_oa_vs_D1=O_z0[i0] - d1b.loc[(unit, f), 'oa'],
                               z0_ba_vs_D1=B_z0[i0] - d1b.loc[(unit, f), 'ba_ge30'],
                               rt_oa_vs_G=O_rt[i0] - g.oa, rt_ba_vs_G=B_rt[i0] - g.ba_ge30,
                               rt_oa_vs_G_i300=O_rt[i0] - g.oa_i300,
                               hd_oa_vs_D1=O_hd[i0] - d1h.loc[(unit, f), 'oa'],
                               hd_ba_vs_D1=B_hd[i0] - d1h.loc[(unit, f), 'ba_ge30'],
                               hd_s_vs_D1=s_ - d1h.loc[(unit, f), 's'],
                               em_iter=n_it, em_conv=conv, em_bad=n_bad,
                               n_test=len(yT), n_big=len(big), n_sel_classes=len(np.unique(ys)))
                    checks.append(chk)
                    # 全测试集 FA
                    curves = {'Z0': [(O_z0, B_z0, TAUS)], 'Z0+': [(O_z0, B_z0, TAUS), (O_em, B_em, TAUS)],
                              'RT-200': [(O_rt, B_rt, TAUS)], 'HD-200': [(O_hd, B_hd, TAUS)]}
                    tau0 = {'Z0': (O_z0[i0], B_z0[i0], O_z0[i5], B_z0[i5]),
                            'Z0+': (O_em[i0], B_em[i0], O_em[i5], B_em[i5]),  # Z0+ 行记 EM 曲线的 τ=0（EM 点）与 τ=0.5
                            'RT-200': (O_rt[i0], B_rt[i0], O_rt[i5], B_rt[i5]),
                            'HD-200': (O_hd[i0], B_hd[i0], O_hd[i5], B_hd[i5])}
                    envs = {}
                    for arm in ARMS:
                        env, _, cid = envelope_multi(curves[arm], grid)
                        envs[arm] = env
                        allO = np.concatenate([c[0] for c in curves[arm]])
                        r = dict(target=tgt, split=split, rep=rep, unit=unit, clf=f, arm=arm, oa_star=oa_star,
                                 oa_tau0=tau0[arm][0], ba_tau0=tau0[arm][1], oa_tau05=tau0[arm][2],
                                 ba_tau05=tau0[arm][3], oa_min=allO.min(), oa_max=allO.max(), **fa_of(env),
                                 n_em_points=int((cid == 1).sum()) if arm == 'Z0+' else np.nan,
                                 hd_s=s_ if arm == 'HD-200' else np.nan, sel_hash=g.sel_hash)
                        rows.append(r)
                    # 共同可达子窗口（方案外核对）：各对比在两臂都可达的格点上平均差值
                    pairs = {'RT-200': ('Z0+', 'HD-200'), 'HD-200': ('Z0+',)}
                    for r in rows[-4:]:
                        a = r['arm']
                        for other in pairs.get(a, ()):
                            m = np.isfinite(envs[a]) & np.isfinite(envs[other])
                            r[f'dFA_common_vs_{other}'] = float(np.mean(envs[a][m] - envs[other][m])) \
                                if m.mean() >= 0.5 else np.nan
                    # split-half（只对 Z0+ 与 RT-200）
                    ia, ib, seed = split_halves(unit, len(yT))
                    anc = {'a': anchor_on(z0, yT, big, ia), 'b': anchor_on(z0, yT, big, ib)}
                    idx = {'a': ia, 'b': ib}
                    for arm, arms in (('Z0+', [z0, zem]), ('RT-200', [rt])):
                        r = next(x for x in rows[-4:] if x['arm'] == arm)
                        r['sh_seed'] = seed
                        r['sh_anchor_a'], r['sh_anchor_b'] = anc['a'], anc['b']
                        bias, bias_full = [], []
                        for s_from, s_to in (('a', 'b'), ('b', 'a')):
                            fa_x, cov_x = cross_fa(arms, yT, big, idx[s_from], anc[s_from], idx[s_to])
                            fa_d = direct_fa(arms, yT, big, idx[s_to], anc[s_to])
                            r[f'sh_{s_from}{s_to}_cross'], r[f'sh_{s_from}{s_to}_direct'] = fa_x, fa_d
                            r[f'sh_{s_from}{s_to}_cov'] = cov_x
                            bias.append(fa_d - fa_x)
                            # 敏感性：两半都用整个测试集的 OA★
                            fa_x2, _ = cross_fa(arms, yT, big, idx[s_from], oa_star, idx[s_to])
                            fa_d2 = direct_fa(arms, yT, big, idx[s_to], oa_star)
                            bias_full.append(fa_d2 - fa_x2)
                        r['sh_bias'] = float(np.mean(bias))
                        r['sh_bias_fullanchor'] = float(np.mean(bias_full))
            print(f'{tgt} s{split} {time.time() - t0:.0f}s', flush=True)
    return pd.DataFrame(rows), pd.DataFrame(checks)


# ---------------------------------------------------------------- 汇总
def tsum(v):
    v = np.asarray(pd.Series(v).dropna(), dtype=float)
    n = len(v)
    if n < 2:
        return dict(n=n, mean=np.nan, lo=np.nan, hi=np.nan, p=np.nan, npos=int((v > 0).sum()), sd=np.nan)
    m, sd = v.mean(), v.std(ddof=1)
    se = sd / np.sqrt(n)
    q = stats.t.ppf(0.975, n - 1)
    p = 2 * stats.t.sf(abs(m / se), n - 1) if se > 0 else np.nan
    return dict(n=n, mean=m, lo=m - q * se, hi=m + q * se, p=p, npos=int((v > 0).sum()), sd=sd)


def fmt(s):
    return (f'{s["mean"]:+.2f} [{s["lo"]:+.2f}, {s["hi"]:+.2f}]，p={s["p"]:.3g}，>0 {s["npos"]}/{s["n"]}，'
            f'国家 SD {s["sd"]:.2f}')


def country_vals(U, col, feats):
    """单元值 → 国家值：先在每个特征内对该国单元取平均，再对所选特征取平均。"""
    x = U[U.clf.isin(feats)].groupby(['target', 'clf'])[col].mean().groupby('target').mean()
    return x.reindex(COUNTRIES)


def summarize(R, CK):
    L_ = []
    P = L_.append
    P('第 0 步 B 部分独立复核（partB_indep）——只依据规格 B 部分；未读取 verify/step0/partB/')
    P(f'规格 sha256 前 16 位：{hashlib.sha256(open(SPEC, "rb").read()).hexdigest()[:16]}；'
      f'fit_bcts/weights 源码 sha256[:16]：{FUNC_HASH}')
    P(f'τ 网格：{TAUS[0]} … {TAUS[-1]}，{len(TAUS)} 点；窗口 41 格点（步长 0.1）；λ={LAM}；k={K}；log 裁剪 {CLIP}')
    P('C 级、描述性；测试集已被多次复用。国家值 = 4 个单元（划分 × 重复）的平均；9 国 = 去掉 RO、PT。')
    P('')
    P('== 0. 核对 ==')
    for c in ['z0_oa_vs_D1', 'z0_ba_vs_D1', 'rt_oa_vs_G', 'rt_ba_vs_G', 'rt_oa_vs_G_i300', 'hd_oa_vs_D1',
              'hd_ba_vs_D1', 'hd_s_vs_D1']:
        P(f'  {c}: max|差| = {CK[c].abs().max():.2e}（{len(CK)} 个 单元×特征）')
    P(f'  EM：未收敛 {int((~CK.em_conv).sum())}/{len(CK)}，迭代数中位 {CK.em_iter.median():.0f}，'
      f'回退行总数 {int(CK.em_bad.sum())}')
    P(f'  SRS k=200 选样类别数：最少 {CK.n_sel_classes.min()}，中位 {CK.n_sel_classes.median():.0f}')
    Z = R[R.arm == 'Z0']
    P(f'  Z0 在 τ=0 的 OA 即 OA★：max|oa_tau0 − oa_star| = {(Z.oa_tau0 - Z.oa_star).abs().max():.2e}')
    W = R.pivot_table(index=['unit', 'clf'], columns='arm', values='fa_raw')
    P(f'  包络单调性：FA_raw(Z0+) − FA_raw(Z0) 最小值 {(W["Z0+"] - W["Z0"]).min():+.3f}（Z0 全可达时应 ≥ 0）')
    P('')
    P('== 1. 可达率（每臂 88 个 单元×特征；另按特征分列） ==')
    for arm in ARMS:
        for feats in (['ae'], ['s2'], FEATS):
            x = R[(R.arm == arm) & R.clf.isin(feats)]
            full = int((x['cov'] >= 1.0).sum())
            part = int(((x['cov'] < 1.0) & (x['cov'] >= 0.5)).sum())
            miss = int((x['cov'] < 0.5).sum())
            P(f'  {arm:7s} {"+".join(feats):6s} n={len(x)}：全可达 {full}，部分可达 {part}，缺失 {miss}；'
              f'可达率均值 {x["cov"].mean():.3f}，最小 {x["cov"].min():.3f}，分位 [10/50/90] '
              f'{np.percentile(x["cov"], 10):.3f}/{np.percentile(x["cov"], 50):.3f}/{np.percentile(x["cov"], 90):.3f}')
        x = R[(R.arm == arm) & (R['cov'] < 1.0)]
        if len(x):
            P(f'      非全可达：' + '，'.join(f'{u}/{c}({v:.2f})' for u, c, v in zip(x.unit, x.clf, x['cov'])))
    P('')
    P('== 2. 各臂 FA 水平（国家均值 [95% CI]） ==')
    for arm in ARMS:
        for feats in (['ae'], ['s2'], FEATS):
            u = R[R.arm == arm][['target', 'clf', 'FA']]
            cv = country_vals(u, 'FA', feats)
            for nm, sub in (('11', cv), ('9', cv.drop(DROP))):
                s = tsum(sub)
                P(f'  FA({arm}) {"+".join(feats):6s} [{nm}]：{s["mean"]:.2f} [{s["lo"]:.2f}, {s["hi"]:.2f}]')
    P('')
    # 对比
    Wd = R.pivot_table(index=['target', 'split', 'rep', 'unit', 'clf'], columns='arm', values='FA').reset_index()
    Wd['dRT'] = Wd['RT-200'] - Wd['Z0+']
    Wd['dHD'] = Wd['HD-200'] - Wd['Z0+']
    Wd['RTmHD'] = Wd['RT-200'] - Wd['HD-200']
    Wd['dRT_Z0'] = Wd['RT-200'] - Wd['Z0']
    Wd['dHD_Z0'] = Wd['HD-200'] - Wd['Z0']
    Wd['Z0p_m_Z0'] = Wd['Z0+'] - Wd['Z0']
    cm_rt = R[R.arm == 'RT-200'].set_index(['unit', 'clf'])
    cm_hd = R[R.arm == 'HD-200'].set_index(['unit', 'clf'])
    Wd = Wd.set_index(['unit', 'clf'])
    Wd['dRT_common'] = cm_rt['dFA_common_vs_Z0+']
    Wd['RTmHD_common'] = cm_rt['dFA_common_vs_HD-200']
    Wd['dHD_common'] = cm_hd['dFA_common_vs_Z0+']
    Wd = Wd.reset_index()
    P('== 3. 主对比（B3；单元差值 → 国家均值；mean [95% CI]，t 检验 p，>0 国家数）==')
    names = [('dRT', 'ΔFA(RT-200) = FA(RT-200) − FA(Z0+)'), ('dHD', 'ΔFA(HD-200) = FA(HD-200) − FA(Z0+)'),
             ('RTmHD', 'FA(RT-200) − FA(HD-200)')]
    extra = [('dRT_Z0', '〔方案外〕FA(RT-200) − FA(Z0)'), ('dHD_Z0', '〔方案外〕FA(HD-200) − FA(Z0)'),
             ('Z0p_m_Z0', '〔方案外〕FA(Z0+) − FA(Z0)'),
             ('dRT_common', '〔方案外〕ΔFA(RT-200) 共同可达子窗口'), ('dHD_common', '〔方案外〕ΔFA(HD-200) 共同可达子窗口'),
             ('RTmHD_common', '〔方案外〕RT−HD 共同可达子窗口')]
    CT = []
    for col, lab in names + extra:
        P(f'  {lab}')
        for feats in (['ae'], ['s2'], FEATS):
            cv = country_vals(Wd, col, feats)
            n_unit = int(Wd[Wd.clf.isin(feats)][col].notna().sum())
            for nm, sub in (('11', cv), ('9', cv.drop(DROP))):
                s = tsum(sub)
                P(f'    {"+".join(feats):6s} [{nm:>2s}] {fmt(s)}（非缺失单元 {n_unit}）')
                CT.append(dict(contrast=col, feats='+'.join(feats), set=nm, **s))
            P('      国家值：' + '，'.join(f'{c} {v:+.2f}' for c, v in cv.items()))
    P('')
    P('== 4. 〔方案外、只作核对〕τ = 0.5 可部署点的差值（Z0 与 EM 曲线各自在 τ=0.5；RT/HD 在 τ=0.5）==')
    T5 = R.pivot_table(index=['target', 'split', 'rep', 'unit', 'clf'], columns='arm',
                       values=['ba_tau05', 'oa_tau05']).reset_index()
    for met in ('ba_tau05', 'oa_tau05'):
        for a, b_ in (('RT-200', 'Z0'), ('HD-200', 'Z0'), ('RT-200', 'HD-200'), ('Z0+', 'Z0')):
            d = pd.DataFrame(dict(target=T5['target'], clf=T5['clf'], v=T5[(met, a)] - T5[(met, b_)]))
            lab = f'{met[:2].upper()}(τ=.5) {a if a != "Z0+" else "EM曲线"} − {b_}'
            for feats in (['ae'], ['s2'], FEATS):
                cv = country_vals(d, 'v', feats)
                P(f'  {lab:28s} {"+".join(feats):6s} [11] {fmt(tsum(cv))} | [9] {fmt(tsum(cv.drop(DROP)))}')
    P('')
    P('== 5. B5 split-half 包络偏差（偏差 = 直接包络 FA − 交叉 FA，两方向平均；正 = 乐观）==')
    SH = R[R.arm.isin(['Z0+', 'RT-200'])].copy()
    for arm in ('Z0+', 'RT-200'):
        for feats in (['ae'], ['s2'], FEATS):
            x = SH[(SH.arm == arm) & SH.clf.isin(feats)]
            v = x.sh_bias.dropna()
            P(f'  {arm:7s} {"+".join(feats):6s} 单元层 n={len(v)}：均值 {v.mean():+.3f}，SD {v.std():.3f}，'
              f'分位 [5/25/50/75/95] ' + '/'.join(f'{np.percentile(v, q):+.2f}' for q in (5, 25, 50, 75, 95)) +
              f'，>0 {int((v > 0).sum())}/{len(v)}')
            cv = country_vals(x, 'sh_bias', ['ae', 's2'])
            P(f'      国家层 [11] {fmt(tsum(cv))} | [9] {fmt(tsum(cv.drop(DROP)))}')
            v2 = x.sh_bias_fullanchor.dropna()
            cv2 = country_vals(x, 'sh_bias_fullanchor', ['ae', 's2'])
            P(f'      〔敏感性：锚点用整个测试集 OA★〕单元均值 {v2.mean():+.3f}；国家层 [11] {fmt(tsum(cv2))}')
    # ΔFA(RT-200) 的偏差 = 偏差(RT) − 偏差(Z0+)
    B2 = SH.pivot_table(index=['target', 'unit', 'clf'], columns='arm', values='sh_bias').reset_index()
    B2['d'] = B2['RT-200'] - B2['Z0+']
    for feats in (['ae'], ['s2'], FEATS):
        x = B2[B2.clf.isin(feats)]
        cv = country_vals(x, 'd', feats)
        P(f'  偏差差（RT-200 − Z0+，即 ΔFA(RT-200) 的偏差） {"+".join(feats):6s} 单元均值 {x.d.mean():+.3f}；'
          f'国家层 [11] {fmt(tsum(cv))} | [9] {fmt(tsum(cv.drop(DROP)))}')
    covx = SH[[c for c in SH.columns if c.endswith('_cov') and c.startswith('sh_')]]
    P(f'  split-half 交叉方向的可达率：最小 {np.nanmin(covx.values):.3f}；<1 的 (单元×方向) 数 '
      f'{int((covx.values < 1).sum())}/{covx.size}；缺失偏差 {int(SH.sh_bias.isna().sum())}')
    return '\n'.join(L_), pd.DataFrame(CT)


def main():
    R, CK = compute()
    R.to_csv(os.path.join(OUT, 'indep_fa_rows.csv'), index=False)
    CK.to_csv(os.path.join(OUT, 'indep_checks.csv'), index=False)
    txt, CT = summarize(R, CK)
    CT.to_csv(os.path.join(OUT, 'indep_country_contrasts.csv'), index=False)
    open(os.path.join(OUT, 'indep_summary.txt'), 'w', encoding='utf-8').write(txt + '\n')
    print(txt)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--summary':
        R = pd.read_csv(os.path.join(OUT, 'indep_fa_rows.csv'))
        CK = pd.read_csv(os.path.join(OUT, 'indep_checks.csv'))
        txt, CT = summarize(R, CK)
        CT.to_csv(os.path.join(OUT, 'indep_country_contrasts.csv'), index=False)
        open(os.path.join(OUT, 'indep_summary.txt'), 'w', encoding='utf-8').write(txt + '\n')
        print(txt)
    else:
        main()
