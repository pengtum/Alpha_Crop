"""
确认运行登记前的不训练分析 —— AE 线的 AE2、AE5、AE6（机制描述）。
规格：label_expansion/PRECONF_ANALYSES_SPEC_2026-09-29.md（sha256[:16]=d0906bd1417839d2，登记后未改）。
事后分析，C 级、描述性；不训练任何作物分类器，只读已存概率与 stage0 几何。测试标签只用于评估；
候选标签与外国标签只用于 AE6 的类条件距离（按类分组抽样）。

口径复用（import 第 0 步补遗 B2 的 step0B2.py，它又 import step0B.py；两者都不改）：
  τ 族      step0B2.tau_preds_grid（= step0B.tau_preds 源码，网格 TAUS_NEW：−2.0..2.0 步长 0.05，81 点）
  π̂        该臂测试集平均预测（exp(Z − 行最大) 归一化后取均值，不用标签）
  log 取法  step0B.log_or_neginf（精确 log，概率恰为 0 → −inf）
  OA / BA   step0B.Scorer（ba_ge30 的类集合 = 测试集中 ≥ 30 个地块的类；检验 3 的 MIN_N 同为 30）
  包络      step0B.envelope（单曲线：同一 o 的多个交点取 BA 最大；相邻 τ 线性插值）
  窗口      W_w = [锚点 − w, 锚点] 步长 0.1（w = 4 为主；w = 4 时与 step0B.OFFS 逐位相同）
  共同可达  两条曲线在窗口内都可达的 o 的比例 < 50% → 记缺失（同 AE1 第 4 条）
  汇总      step0B.t_summary / f95（国家层 t 区间，df = n − 1）；9 国 = 去掉 RO、PT

AE2  检验 3（随机 70/30、无缓冲；{T}_split.npz 的 test / y_test，与 analyze_check3.py、analyze_check3_capacity.py 相同）
     国内 = within（W0），跨境 = loco（L0）；设置 c300（固定 300 棵，主）与 es（早停，次）。
     每个模型：两特征各自 τ 族；锚点 = 该模型两特征 τ=0 OA 的较小者；D = 窗口内共同可达点上 BA_ae(o) − BA_s2(o) 的平均。
     报 D_dom、D_xb、D_xb − D_dom；对照：τ=0 的 AE − S2（BA、OA）。每国只有一个单元（检验 3 一个划分），国家值 = 该值。
AE5  G 的 P_T（温度校准的外国模型，测试行）：π̂_EM = check3_priors.em_prior(P_T 测试行, π_train)（= B2 的 Z0EM 所用 EM，
     π_train = 外国池按 class_weights_from 加权的类分布）；L1 = ‖π̂_EM − π_test‖₁。国家值 = 4 个单元平均。
     关联（描述）：检验 3 的 EM（em_prior(L0 的 pool 预测, 训练类上均匀先验)，同 analyze_check3.py）的 L1 与 OA 恢复率
     (PEM − L0)/(PO − L0)，es 为检验 3 原口径，c300 另列。
AE6  stage0（划分 0）的 G_f 候选行（allT 的前 n_cand 行）与 GF_f（外国池）；每类（两边都 ≥ 100）各抽 min(500, n)，
     种子 = int(md5(f'AE6_{国家}_{特征}_{类号}')[:8], 16)；对每个目标样本在合并样本中找 10 个欧氏最近邻（排除自身），
     外国样本比例的平均 = M_c。类无关版本：两边各抽 2,000，种子串 f'AE6_{国家}_{特征}_all'。
     报每国 M_c 的中位数与 M_all、两者的 AE − S2，以及与 AE1 的 Z0 D（由 curves_B2.csv 按 AE1 口径复算）的 Spearman。

用法：python -B ae_mechanism.py --smoke   # AE2/AE6：AT、RO；AE5：AT_s0_r0、RO_s1_r1；输出到 smoke/
      python -B ae_mechanism.py           # 全量
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
from scipy import stats
from scipy.spatial.distance import cdist

HERE = os.path.dirname(os.path.abspath(__file__))
PARTB2 = '/Users/perng_mini/AlphaCrops_local/verify/step0/partB2'
sys.path.insert(0, PARTB2)
import step0B2 as B2  # noqa: E402  导入链：step0B（chdir 到项目目录，只读）→ guidance_G
B = B2.B
G = B.G
from check3_priors import as64, trained_mask, uniform_prior, reweight, em_prior  # noqa: E402

SPEC = os.path.join(B.L, 'PRECONF_ANALYSES_SPEC_2026-09-29.md')
SPEC_HASH = 'd0906bd1417839d2'
R3 = '/Users/perng_mini/AlphaCrops_local/results/check3'
C3_ARMS = '/Users/perng_mini/AlphaCrops_local/results/check3_arms.csv'
C3_CAP = '/Users/perng_mini/AlphaCrops_local/results/check3_capacity_arms.csv'
NUMBERS = os.path.join(B.L, 'storyline', 'numbers_table', 'NUMBERS.csv')
CURVES_B2 = os.path.join(PARTB2, 'curves_B2.csv')
FA_B2 = os.path.join(PARTB2, 'fa_rows_B2.csv')
MATCHED_UNITS = ('/Users/perng_mini/AlphaCrops_local/verify/storyline/ae_s2_integration_20260929/ae_s2_evidence/'
                 'matched_oa_units.csv')

C = 10
FEATS = ['ae', 's2']
COUNTRIES = B.COUNTRIES
DROP = B.DROP
C9 = [c for c in COUNTRIES if c not in DROP]
TAUS = B2.TAUS_NEW
I_T0 = B2.I_T0
I_T05 = int(np.flatnonzero(TAUS == 0.5)[0])
WIDTHS = (2, 4, 8)
OFFS_W = {w: np.round(np.arange(-float(w), 0.0 + 1e-9, 0.1), 1) for w in WIDTHS}
assert np.array_equal(OFFS_W[4], B.OFFS)
MIN_COV = B.MIN_COV
C3_MIN_N = 30
assert G.MIN_N == C3_MIN_N
SETTINGS = {'c300': '_c300', 'es': ''}          # 检验 3 文件后缀；c300 为主
MODELS3 = {'dom': 'within', 'xb': 'loco'}       # 国内 = within（W0），跨境 = loco（L0）
C3_ARM = {'dom': 'W0', 'xb': 'L0'}
AE6_MIN = 100
AE6_NC = 500
AE6_NALL = 2000
AE6_K = 10


def sha16(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()[:16]


def md5seed(s):
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16)


# ---------------------------------------------------------------- 通用：τ 族曲线与同 OA 比较
def tau_curve(P, sc):
    """P：(n, C) 概率（float64，行已归一化）。与 step0B2.run_unit 的曲线循环相同：Z = 精确 log，π̂ = 该臂平均预测。"""
    Z = B.log_or_neginf(P)
    Pa = np.exp(Z - Z.max(1, keepdims=True))
    Pa /= Pa.sum(1, keepdims=True)
    pihat = Pa.mean(0)
    PH = B2.tau_preds_grid(Z, pihat, TAUS)
    return sc.curve(PH), pihat


def matched(cv_a, cv_s, anchor, offs):
    """AE1 口径：两特征各用自己的曲线；窗口 anchor + offs；共同可达点上逐点 BA_ae − BA_s2 取平均；共同可达 <50% 记缺失。"""
    grid = anchor + offs
    ba_a = B.envelope([cv_a], grid)[0]
    ba_s = B.envelope([cv_s], grid)[0]
    ok = np.isfinite(ba_a) & np.isfinite(ba_s)
    cov = float(ok.mean())
    miss = cov < MIN_COV
    diff = ba_a[ok] - ba_s[ok]
    return dict(cov=cov, reach_ae=float(np.isfinite(ba_a).mean()), reach_s2=float(np.isfinite(ba_s).mean()),
                D=np.nan if miss else float(diff.mean()),
                lev_ae=np.nan if miss else float(ba_a[ok].mean()),
                lev_s2=np.nan if miss else float(ba_s[ok].mean()),
                ae_dom=np.nan if miss else bool((diff >= 0).all()),
                s2_dom=np.nan if miss else bool((diff <= 0).all()),
                d_min=np.nan if miss else float(diff.min()), d_max=np.nan if miss else float(diff.max()))


# ---------------------------------------------------------------- AE2
def run_ae2(D, countries, log):
    y_all = np.asarray(D.y)
    curve_rows, rows, chk = [], [], []
    for tgt in countries:
        sp = np.load(os.path.join(R3, f'{tgt}_split.npz'))
        test, yT = sp['test'], sp['y_test'].astype(np.int64)
        assert np.array_equal(y_all[test], yT), f'{tgt}：检验 3 的 y_test 与数据集标签不一致'
        big = np.flatnonzero(np.bincount(yT, minlength=C) >= C3_MIN_N)
        sc = B.Scorer(yT, big)
        for st, suf in SETTINGS.items():
            CV = {}
            for mk, mname in MODELS3.items():
                for f in FEATS:
                    P = as64(np.load(os.path.join(R3, f'{tgt}_{f}_{mname}{suf}_test.npy')))
                    (oa, ba), pihat = tau_curve(P, sc)
                    CV[(mk, f)] = (oa, ba)
                    cell = dict(target=tgt, setting=st, model=mk, clf=f)
                    curve_rows += [dict(cell, tau=t, oa=oa[i], ba_ge30=ba[i]) for i, t in enumerate(TAUS)]
                    extra = {}
                    if mk == 'dom':  # 只用于与 NUMBERS H1.total.oa 对账（OA 链的本国模型是不加权的 wnat，不进入 AE2 计算）
                        Pw = as64(np.load(os.path.join(R3, f'{tgt}_{f}_wnat{suf}_test.npy')))
                        extra['oa_wnat'] = G._scores(yT, Pw.argmax(1), big)[0]
                    chk.append(dict(cell, n_test=len(yT), n_big=len(big), n_untrained=int((~trained_mask(P)).sum()),
                                    oa=oa[I_T0], ba_ge30=ba[I_T0], **extra))
            for mk in MODELS3:
                ca, cs = CV[(mk, 'ae')], CV[(mk, 's2')]
                anchor = min(ca[0][I_T0], cs[0][I_T0])
                base = dict(target=tgt, setting=st, model=mk, anchor=anchor,
                            oa0_ae=ca[0][I_T0], oa0_s2=cs[0][I_T0], ba0_ae=ca[1][I_T0], ba0_s2=cs[1][I_T0],
                            oamax_ae=float(ca[0].max()), oamax_s2=float(cs[0].max()),
                            oamin_ae=float(ca[0].min()), oamin_s2=float(cs[0].min()))
                for w in WIDTHS:
                    rows.append(dict(base, width=w, **matched(ca, cs, anchor, OFFS_W[w])))
        log(f'AE2 {tgt} 完成')
    return pd.DataFrame(curve_rows), pd.DataFrame(rows), pd.DataFrame(chk)


# ---------------------------------------------------------------- AE5（G 的 P_T）
def run_ae5(D, units, log):
    y_all = np.asarray(D.y)
    rows, curve_rows = [], []
    cache = {}
    for tgt, split, rep in units:
        if (tgt, split) not in cache:
            cache.clear()
            cache[(tgt, split)] = np.load(os.path.join(B.R_G, 'stage0', f'{tgt}_s{split}.npz'))
        S = cache[(tgt, split)]
        unit = f'{tgt}_s{split}_r{rep}'
        n_cand = len(S['cand'])
        allT, test, fr = S['allT'], S['test'], S['foreign']
        assert np.array_equal(allT[n_cand:], test)
        yT = y_all[test]
        nT = len(yT)
        big = np.flatnonzero(np.bincount(yT, minlength=C) >= G.MIN_N)
        sc = B.Scorer(yT, big)
        yF = y_all[fr]
        cw = G.class_weights_from(yF)
        pi_train = np.bincount(yF, weights=cw[yF], minlength=C) / cw[yF].sum()   # = G pi_train_of([], 0)
        pool_pos = G.pool_positions(S, split, rep)
        pi_pool = np.bincount(y_all[allT[pool_pos]], minlength=C) / len(pool_pos)
        pi_test = np.bincount(yT, minlength=C) / nT
        for f in FEATS:
            cell = dict(target=tgt, split=split, rep=rep, unit=unit, clf=f)
            P0 = B.norm64(S[f'P_T_{f}'][n_cand:].astype(np.float64))
            pi_em, it, conv, ll = em_prior(P0, pi_train)
            Q, nfb = reweight(P0, pi_em, pi_train)
            (oa0, ba0), pih0 = tau_curve(P0, sc)
            (oaE, baE), pihE = tau_curve(Q, sc)
            for arm, (oa, ba) in (('Z0', (oa0, ba0)), ('Z0EM', (oaE, baE))):
                curve_rows += [dict(cell, arm=arm, tau=t, oa=oa[i], ba_ge30=ba[i]) for i, t in enumerate(TAUS)]
            Qo, _ = reweight(P0, pi_test, pi_train)            # PKo（测试真实先验，只作描述）
            o_o, b_o = G._scores(yT, Qo.argmax(1), big)[:2]
            # 次：G 自己的 P1 臂（foreign_only，未温度校准的 P0test_i300，EM），与 G rows 的 prior_L1（相对 π_pool）对账
            Praw = S[f'P0test_{f}_i300'].astype(np.float32).astype(np.float64)
            Praw /= Praw.sum(1, keepdims=True)
            pi_raw, it_r, conv_r, _ = em_prior(Praw, pi_train)
            rows.append(dict(cell, n_test=nT,
                             L1_em=float(np.abs(pi_em - pi_test).sum()),
                             L1_train=float(np.abs(pi_train - pi_test).sum()),
                             L1_mean=float(np.abs(pih0 - pi_test).sum()),
                             L1_em_pool=float(np.abs(pi_em - pi_pool).sum()),
                             L1_raw_em=float(np.abs(pi_raw - pi_test).sum()),
                             L1_raw_em_pool=float(np.abs(pi_raw - pi_pool).sum()),
                             em_iter=it, em_conv=conv, em_fallback=nfb,
                             em_ll_monotone=bool(np.all(np.diff(ll) >= -1e-8 * np.abs(ll[:-1]))),
                             raw_em_iter=it_r, raw_em_conv=conv_r,
                             oa_t0_Z0=oa0[I_T0], ba_t0_Z0=ba0[I_T0], oa_t0_EM=oaE[I_T0], ba_t0_EM=baE[I_T0],
                             oa_t0_PKo=o_o, ba_t0_PKo=b_o,
                             **{f'pi_em_{c}': pi_em[c] for c in range(C)},
                             **{f'pi_test_{c}': pi_test[c] for c in range(C)},
                             **{f'pi_train_{c}': pi_train[c] for c in range(C)}))
        log(f'AE5 {unit} 完成')
    return pd.DataFrame(rows), pd.DataFrame(curve_rows)


def run_ae5_c3(countries):
    """检验 3 的 EM（同 analyze_check3.py：em_prior(L0 的 pool 预测, uniform_prior(trained_mask(L0 测试预测)))）。"""
    rows = []
    for tgt in countries:
        sp = np.load(os.path.join(R3, f'{tgt}_split.npz'))
        y_pool, y_test = sp['y_pool'].astype(np.int64), sp['y_test'].astype(np.int64)
        pi_pool = np.bincount(y_pool, minlength=C) / len(y_pool)
        pi_test = np.bincount(y_test, minlength=C) / len(y_test)
        big = np.flatnonzero(np.bincount(y_test, minlength=C) >= C3_MIN_N)
        for f in FEATS:
            for st, suf in SETTINGS.items():
                Lt = as64(np.load(os.path.join(R3, f'{tgt}_{f}_loco{suf}_test.npy')))
                Lp = as64(np.load(os.path.join(R3, f'{tgt}_{f}_loco{suf}_pool.npy')))
                pi_tr = uniform_prior(trained_mask(Lt))
                pem, it, conv, _ = em_prior(Lp, pi_tr)
                arms = {'L0': Lt, 'PEM': reweight(Lt, pem, pi_tr)[0], 'PO': reweight(Lt, pi_pool, pi_tr)[0]}
                m = {a: G._scores(y_test, p.argmax(1), big)[:2] for a, p in arms.items()}
                rows.append(dict(target=tgt, clf=f, setting=st,
                                 L1_em=float(np.abs(pem - pi_test).sum()),
                                 L1_em_pool=float(np.abs(pem - pi_pool).sum()),
                                 L1_train=float(np.abs(pi_tr - pi_test).sum()),
                                 L1_mean=float(np.abs(Lt.mean(0) - pi_test).sum()),
                                 em_iter=it, em_conv=conv,
                                 **{f'oa_{a}': v[0] for a, v in m.items()}, **{f'ba_{a}': v[1] for a, v in m.items()},
                                 rec_oa=(m['PEM'][0] - m['L0'][0]) / (m['PO'][0] - m['L0'][0]),
                                 **{f'pi_em_{c}': pem[c] for c in range(C)}))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- AE6
def mix_index(Xt, Xf, k=AE6_K):
    """对每个目标样本，在合并样本 [Xt; Xf] 中找 k 个欧氏最近邻（排除自身；距离并列按合并样本下标顺序，目标在前），
    返回外国样本比例的平均、以及第 k 与第 k+1 近邻距离并列的目标样本数（并列会让结果依赖下标顺序）。"""
    X = np.vstack([Xt, Xf]).astype(np.float64)
    nT = len(Xt)
    d = cdist(X[:nT], X)
    d[np.arange(nT), np.arange(nT)] = np.inf
    order = np.argsort(d, axis=1, kind='stable')[:, :k + 1]
    frac = (order[:, :k] >= nT).mean(1)
    dk = np.take_along_axis(d, order, 1)
    return float(frac.mean()), int((dk[:, k - 1] == dk[:, k]).sum()), int((dk[:, :k] == 0).any(1).sum())


def run_ae6(D, countries, log):
    y_all = np.asarray(D.y)
    rows = []
    for tgt in countries:
        S = np.load(os.path.join(B.R_G, 'stage0', f'{tgt}_s0.npz'))
        cand, fr, allT = S['cand'], S['foreign'], S['allT']
        n_cand = len(cand)
        assert np.array_equal(allT[:n_cand], cand)
        yC, yF = y_all[cand], y_all[fr]
        GC = {f: S[f'G_{f}'][:n_cand] for f in FEATS}
        GF = {f: S[f'GF_{f}'] for f in FEATS}
        assert all(GF[f].shape == (len(fr), 32) and GC[f].shape == (n_cand, 32) for f in FEATS)
        for f in FEATS:
            for c in list(range(C)) + ['all']:
                if c == 'all':
                    iT, iF = np.arange(n_cand), np.arange(len(fr))
                    cap = AE6_NALL
                else:
                    iT, iF = np.flatnonzero(yC == c), np.flatnonzero(yF == c)
                    cap = AE6_NC
                row = dict(target=tgt, clf=f, cls=str(c), cls_name='all' if c == 'all' else D.classes[c],
                           n_cand=len(iT), n_foreign=len(iF))
                elig = c == 'all' or (len(iT) >= AE6_MIN and len(iF) >= AE6_MIN)
                row['eligible'] = bool(elig)
                if elig:
                    rng = np.random.default_rng(md5seed(f'AE6_{tgt}_{f}_{c}'))
                    sT = rng.choice(iT, min(cap, len(iT)), replace=False)
                    sF = rng.choice(iF, min(cap, len(iF)), replace=False)
                    M, ties, zero = mix_index(GC[f][sT], GF[f][sF])
                    # 方案外（描述）：两特征用同一批地块（种子串不含特征），看特征间抽样差异的影响
                    rng2 = np.random.default_rng(md5seed(f'AE6pair_{tgt}_{c}'))
                    pT = rng2.choice(iT, min(cap, len(iT)), replace=False)
                    pF = rng2.choice(iF, min(cap, len(iF)), replace=False)
                    Mp, _, _ = mix_index(GC[f][pT], GF[f][pF])
                    E = len(sF) / (len(sT) + len(sF) - 1)   # 完全混合时的期望（两边样本量不等时 ≠ 0.5）
                    row.update(n_sT=len(sT), n_sF=len(sF), M=M, E_mix=E, M_norm=M / E, M_pair=Mp,
                               n_tie_k=ties, n_zero_dist=zero,
                               sel_hash_T=G.sel_hash(cand[sT]), sel_hash_F=G.sel_hash(fr[sF]))
                rows.append(row)
        log(f'AE6 {tgt} 完成')
    return pd.DataFrame(rows)


def d_z0_from_b2(CB):
    """AE1 的 Z0 D（LightGBM）：直接读 curves_B2.csv 的 Z0 曲线；锚点 = min(OA★_ae, OA★_s2)；W4、步长 0.1。"""
    z = CB[CB.arm == 'Z0']
    rows = []
    for unit, g in z.groupby('unit'):
        cv = {}
        for f in FEATS:
            x = g[g.clf == f].sort_values('tau')
            assert np.allclose(x.tau.to_numpy(), TAUS, atol=1e-9)
            cv[f] = (x.oa.to_numpy(), x.ba_ge30.to_numpy())
        anchor = min(cv['ae'][0][I_T0], cv['s2'][0][I_T0])
        for w in WIDTHS:
            rows.append(dict(target=g.target.iloc[0], split=int(g.split.iloc[0]), rep=int(g.rep.iloc[0]), unit=unit,
                             width=w, anchor=anchor, **matched(cv['ae'], cv['s2'], anchor, OFFS_W[w])))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 主程序
def main():
    smoke = '--smoke' in sys.argv
    out = os.path.join(HERE, 'smoke') if smoke else HERE
    os.makedirs(out, exist_ok=True)
    assert sha16(SPEC) == SPEC_HASH, '规格文件哈希与登记值不一致'
    t0 = time.time()

    def log(s):
        print(f'[{time.time() - t0:.0f}s] {s}', flush=True)
    log(f'ae_mechanism.py sha256[:16]={sha16(os.path.abspath(__file__))}；规格 {sha16(SPEC)}')
    D = G.GData()
    ctrs = ['AT', 'RO'] if smoke else COUNTRIES
    units = [('AT', 0, 0), ('RO', 1, 1)] if smoke else [(t, s, r) for t in COUNTRIES for s in (0, 1) for r in (0, 1)]
    A2c, A2, A2k = run_ae2(D, ctrs, log)
    A5, A5c = run_ae5(D, units, log)
    A5x = run_ae5_c3(ctrs)
    A6 = run_ae6(D, ctrs, log)
    CB = pd.read_csv(CURVES_B2, float_precision='round_trip')
    DZ = d_z0_from_b2(CB)
    pre = os.path.join(out, 'ae_mechanism_')
    A2c.to_csv(pre + 'ae2_curves.csv', index=False)
    A2.to_csv(pre + 'ae2_rows.csv', index=False)
    A2k.to_csv(pre + 'ae2_tau0_check.csv', index=False)
    A5.to_csv(pre + 'ae5_rows.csv', index=False)
    A5x.to_csv(pre + 'ae5_check3_rows.csv', index=False)
    A6.to_csv(pre + 'ae6_rows.csv', index=False)
    DZ.to_csv(pre + 'ae1_z0_D_units.csv', index=False)
    lines = summarize(D, A2c, A2, A2k, A5, A5c, A5x, A6, DZ, CB, smoke, ctrs)
    open(pre + 'summary.txt', 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    log('完成')


# ---------------------------------------------------------------- 汇总
def s9(v):
    return v.drop([c for c in DROP if c in v.index])


def summarize(D, A2c, A2, A2k, A5, A5c, A5x, A6, DZ, CB, smoke, ctrs):
    lines = []
    P = lines.append

    def ci(v, nd=2):
        v = pd.Series(v).dropna()
        return f'9 国 {B.f95(B.t_summary(s9(v)), nd)} | 11 国 {B.f95(B.t_summary(v), nd)}'

    P('确认运行登记前的不训练分析：AE 线 AE2、AE5、AE6 —— 事后分析，C 级、描述性（G s0/s1 与检验 3 测试集均已多次复用；'
      '本批数字不进摘要）。' + ('【冒烟：AE2/AE6 为 AT、RO；AE5 为 AT_s0_r0、RO_s1_r1】' if smoke else ''))
    P(f'规格 PRECONF_ANALYSES_SPEC_2026-09-29.md sha256[:16]={sha16(SPEC)}（登记 {SPEC_HASH}）；'
      f'ae_mechanism.py sha256[:16]={sha16(os.path.abspath(__file__))}；step0B2.py {sha16(os.path.join(PARTB2, "step0B2.py"))}；'
      f'step0B.py {sha16(B2.STEP0B_PY)}（均只 import，未改）；check3_priors.py {sha16(os.path.join(B.L, "check3_priors.py"))}。')
    P(f'τ 网格 {TAUS[0]}..{TAUS[-1]} 步长 0.05（{len(TAUS)} 点）；窗口 W_w = [锚点 − w, 锚点] 步长 0.1，w ∈ {WIDTHS}（w = 4 为主）；'
      f'共同可达 <{MIN_COV:.0%} 记缺失。格式：国家层均值 [95% t CI]，>0 国家数/国家数；9 国 = 去掉 RO、PT。单位：BA、OA 为 pp，'
      'L1 与 M 为原始比例。')
    P('AE−S2 类对比本身就是两特征之差，没有"两特征国家值平均（both）"一说；这类量以 AE−S2 为主，ae、s2 的水平分列。')
    P('')

    # ================================================================ 对账
    P('== 对账 ==')
    ok_all = True
    # (1) 检验 3 各模型 τ=0 与已有报告
    ca = pd.read_csv(C3_ARMS)
    ca = ca[ca.split == 'random'].rename(columns={'feat': 'clf'})
    cc = pd.read_csv(C3_CAP).rename(columns={'feat': 'clf'})
    K = A2k.copy()
    K['arm'] = K.model.map(C3_ARM)
    ref_es = K[K.setting == 'es'].merge(ca[['target', 'clf', 'arm', 'oa', 'ba_ge30']], on=['target', 'clf', 'arm'],
                                        suffixes=('', '_ref'))
    cc_ = cc.rename(columns={'cap': 'setting'})
    ref_cap = K.merge(cc_[['target', 'clf', 'setting', 'arm', 'oa', 'ba_ge30']],
                      on=['target', 'clf', 'setting', 'arm'], suffixes=('', '_ref'))
    for name, J, n_exp in (('check3_arms.csv（es，random）', ref_es, int((K.setting == 'es').sum())),
                           ('check3_capacity_arms.csv（es 与 c300）', ref_cap, len(K))):
        do, db = (J.oa - J.oa_ref).abs().max(), (J.ba_ge30 - J.ba_ge30_ref).abs().max()
        ok = len(J) == n_exp and do < 1e-9 and db < 1e-9
        ok_all &= ok
        P(f'(1) AE2 检验 3 within/loco τ=0 的 OA、BA 与 {name}（W0 / L0 行）：{len(J)}/{n_exp} 行，最大 |Δ| OA {do:.2e}，'
          f'BA {db:.2e} → {"通过" if ok else "不通过"}')
    N = pd.read_csv(NUMBERS).set_index('id')
    mx, nrow = 0.0, 0
    for st in SETTINGS:
        for met in ('oa', 'ba_ge30'):
            for f in FEATS:
                pc = pd.Series(json.loads(N.loc[f'H1.total.{met}.{st}.{f}', 'per_country']))
                kk = K[(K.setting == st) & (K.clf == f)]
                k_ = kk.pivot(index='target', columns='model', values=met)
                dom = kk[kk.model == 'dom'].set_index('target').oa_wnat if met == 'oa' else k_['dom']
                dd = (dom - k_['xb'] - pc.reindex(k_.index)).abs()
                mx = max(mx, dd.max())
                nrow += len(dd)
    ok = mx < 5.1e-4
    ok_all &= ok
    P(f'(2) AE2 τ=0 的国内 − 跨境（BA：W0 − L0；OA：Wnat − L0，NUMBERS 的 OA 链终点是不加权本国模型）与 NUMBERS.csv H1.total.{{oa,ba_ge30}}.{{es,c300}}.{{ae,s2}} 的逐国值（3 位小数）：'
      f'{nrow} 个值，最大 |Δ| {mx:.1e} → {"通过" if ok else "不通过"}；检验 3 的 y_test 与数据集标签 D.y[test] 逐一相同（断言通过）')
    # (2b) 检验 3 训练类数
    nu = {mk: K[K.model == mk].n_untrained for mk in MODELS3}
    P(f'    检验 3 概率中恒为 0 的类（训练集中没有）：within 每个 (国家, 特征, 设置) {int(nu["dom"].min())}–{int(nu["dom"].max())} 列，'
      f'loco {int(nu["xb"].min())}–{int(nu["xb"].max())} 列（精确 log 为 −inf，任何 τ 都不会预测；π̂ 对应项为 0，偏移取 0）')
    # (3) Z0、Z0EM 曲线与 curves_B2.csv 逐位
    CN = A5c.copy()
    CN['ti'] = np.rint(CN.tau * 100).astype(int)
    CB2 = CB[CB.arm.isin(['Z0', 'Z0EM'])].copy()
    CB2['ti'] = np.rint(CB2.tau * 100).astype(int)
    JC = CN.merge(CB2, on=['target', 'split', 'rep', 'unit', 'clf', 'arm', 'ti'], suffixes=('', '_B2'), how='inner')
    same = np.array_equal(JC.oa.to_numpy(), JC.oa_B2.to_numpy()) and np.array_equal(JC.ba_ge30.to_numpy(),
                                                                                    JC.ba_ge30_B2.to_numpy())
    ok = same and len(JC) == len(CN)
    ok_all &= ok
    P(f'(3) AE5 重算的 LightGBM Z0、Z0EM（EM 修正）τ 曲线与 partB2/curves_B2.csv：{len(JC)}/{len(CN)} 行对上，OA、BA '
      f'{"逐位相同" if same else "不同"}（最大 |Δ| OA {(JC.oa - JC.oa_B2).abs().max():.2e}，BA '
      f'{(JC.ba_ge30 - JC.ba_ge30_B2).abs().max():.2e}）→ {"通过" if ok else "不通过"}')
    # (4) EM 迭代、收敛，π_train L1 与 B2 fa_rows
    FB = pd.read_csv(FA_B2, float_precision='round_trip')
    FB = FB[FB.grid == 'new']
    em = FB[FB.arm == 'Z0EM'][['unit', 'clf', 'em_iter', 'em_conv', 'em_fallback']]
    J = A5.merge(em, on=['unit', 'clf'], suffixes=('', '_B2'))
    pk = FB[FB.arm == 'PK200'][['unit', 'clf', 'prior_L1_train']]
    J2 = A5.merge(pk, on=['unit', 'clf'])
    ok = (len(J) == len(A5) and (J.em_iter == J.em_iter_B2).all() and (J.em_conv == J.em_conv_B2.astype(bool)).all()
          and (J.em_fallback == J.em_fallback_B2).all() and len(J2) == len(A5)
          and (J2.L1_train - J2.prior_L1_train).abs().max() < 1e-12)
    ok_all &= ok
    P(f'(4) AE5 EM 迭代数、收敛标记、回退行数与 fa_rows_B2.csv（Z0EM）逐行相同 {len(J)}/{len(A5)}；‖π_train − π_test‖₁ 与 '
      f'fa_rows_B2 的 prior_L1_train 最大 |Δ| {(J2.L1_train - J2.prior_L1_train).abs().max():.1e} → {"通过" if ok else "不通过"}；'
      f'EM 对数似然单调不减 {int(A5.em_ll_monotone.sum())}/{len(A5)}')
    # (5) G foreign_only 的 P1 臂 EM
    GR = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(B.R_G, 'rows_t1_*.csv'))], ignore_index=True)
    g0 = GR[GR.method == 'foreign_only'][['target', 'split', 'rep', 'clf', 'prior_L1']]
    J3 = A5.merge(g0, on=['target', 'split', 'rep', 'clf'])
    d3 = (J3.L1_raw_em_pool - J3.prior_L1).abs().max()
    ok = len(J3) == len(A5) and d3 < 1e-12
    ok_all &= ok
    P(f'(5) 次要口径（G 的 P1 臂：未校准 P0test_i300 上的 EM）相对 π_pool 的 L1 与 G rows（foreign_only）prior_L1：'
      f'{len(J3)}/{len(A5)} 行，最大 |Δ| {d3:.1e} → {"通过" if ok else "不通过"}')
    # (6) 检验 3 EM 与 check3_arms
    X = A5x[A5x.setting == 'es']
    ref = ca.pivot_table(index=['target', 'clf'], columns='arm', values=['oa', 'ba_ge30'])
    mx6 = 0.0
    for a in ('L0', 'PEM', 'PO'):
        for met, col in (('oa', f'oa_{a}'), ('ba_ge30', f'ba_{a}')):
            r = ref[(met, a)].reindex(pd.MultiIndex.from_frame(X[['target', 'clf']]))
            mx6 = max(mx6, np.nanmax(np.abs(X[col].to_numpy() - r.to_numpy())))
    Xc = A5x[A5x.setting == 'c300']
    refc = cc[cc.cap == 'c300'].pivot_table(index=['target', 'clf'], columns='arm', values=['oa', 'ba_ge30'])
    mx6c = 0.0
    for a in ('L0', 'PO'):
        for met, col in (('oa', f'oa_{a}'), ('ba_ge30', f'ba_{a}')):
            r = refc[(met, a)].reindex(pd.MultiIndex.from_frame(Xc[['target', 'clf']]))
            mx6c = max(mx6c, np.nanmax(np.abs(Xc[col].to_numpy() - r.to_numpy())))
    ok = mx6 < 1e-9 and mx6c < 1e-9
    ok_all &= ok
    P(f'(6) 检验 3 EM 复算：es 下 L0、PEM、PO 的 OA、BA 与 check3_arms.csv 最大 |Δ| {mx6:.2e}；c300 下 L0、PO 与 '
      f'check3_capacity_arms.csv 最大 |Δ| {mx6c:.2e}（c300 的 PEM 为本脚本新算，检验 3 未报）→ {"通过" if ok else "不通过"}')
    if not smoke:
        for f in FEATS:
            x = X[X.clf == f].set_index('target')
            rec11 = (x.oa_PEM - x.oa_L0).sum() / (x.oa_PO - x.oa_L0).sum()
            x9 = x.loc[C9]
            rec9 = (x9.oa_PEM - x9.oa_L0).sum() / (x9.oa_PO - x9.oa_L0).sum()
            v11 = float(N.loc[f'H5.pem_recovery.{f}', 'value'])
            v9 = float(N.loc[f'H5.pem_recovery.{f}', 'noROPT'])
            ok = abs(100 * rec11 - v11) < 1e-9 and abs(100 * rec9 - v9) < 1e-9
            ok_all &= ok
            P(f'    检验 3 EM 合并恢复率（es）{f}：11 国 {100 * rec11:.4f}%（NUMBERS H5.pem_recovery {v11:.4f}%），'
              f'9 国 {100 * rec9:.4f}%（{v9:.4f}%）→ {"通过" if ok else "不通过"}')
    # (7) AE1 Z0 D 复算与 AE–S2 整合评估
    MU = pd.read_csv(MATCHED_UNITS)
    dz4 = DZ[DZ.width == 4].merge(MU[['unit', 'W4_min_d', 'W4_min_cov']], on='unit')
    d7 = (dz4.D - dz4.W4_min_d).abs().max()
    # 整合评估的窗口写法是 np.round(np.arange(锚点 − 4, 锚点 + 1e-9, 0.1), 6)（o 值四舍五入到 1e-6），本脚本用 锚点 + OFFS；
    # 按整合评估的写法重算一遍，应逐位相同
    z = CB[CB.arm == 'Z0']
    d7x = 0.0
    for unit, g in z.groupby('unit'):
        cv = {f: (g[g.clf == f].sort_values('tau').oa.to_numpy(), g[g.clf == f].sort_values('tau').ba_ge30.to_numpy())
              for f in FEATS}
        anc = min(cv['ae'][0][I_T0], cv['s2'][0][I_T0])
        grid = np.round(np.arange(anc - 4, anc + 1e-9, 0.1), 6)
        ba_a, ba_s = B.envelope([cv['ae']], grid)[0], B.envelope([cv['s2']], grid)[0]
        okk = np.isfinite(ba_a) & np.isfinite(ba_s)
        dx = float(np.mean(ba_a[okk] - ba_s[okk])) if okk.mean() >= MIN_COV else np.nan
        d7x = max(d7x, abs(dx - MU.set_index('unit').W4_min_d[unit]))
    ok = len(dz4) == DZ.unit.nunique() and d7 < 1e-5 and d7x < 1e-12 and (dz4['cov'] - dz4.W4_min_cov).abs().max() < 1e-12
    ok_all &= ok
    P(f'(7) AE1 的 Z0 D（由 curves_B2.csv 复算，W4，锚点 = min(OA★_ae, OA★_s2)）与 AE–S2 整合评估 matched_oa_units.csv 的 '
      f'W4_min_d：{len(dz4)} 个单元；按整合评估的窗口写法（o 四舍五入到 1e-6）重算最大 |Δ| {d7x:.1e}；本脚本窗口（锚点 + OFFS，'
      f'同 step0B）与之最大 |Δ| D {d7:.2e}（仅窗口 o 值舍入不同），共同可达率最大 |Δ| '
      f'{(dz4["cov"] - dz4.W4_min_cov).abs().max():.1e} → {"通过" if ok else "不通过"}')
    # (8) AE6 自检
    A6e = A6[A6.eligible]
    P(f'(8) AE6：可比 (国家, 特征, 类) {int((A6e.cls != "all").sum())} 个 + 类无关 {int((A6e.cls == "all").sum())} 个；'
      f'M ∈ [{A6e.M.min():.3f}, {A6e.M.max():.3f}]；第 10、11 近邻距离并列的目标样本合计 {int(A6e.n_tie_k.sum())}，'
      f'有零距离近邻的目标样本合计 {int(A6e.n_zero_dist.sum())}；两特征的可比类集合相同：'
      f'{A6e[A6e.clf == "ae"].groupby("target").cls.apply(set).equals(A6e[A6e.clf == "s2"].groupby("target").cls.apply(set))}')
    P(f'对账结论：{"全部通过" if ok_all else "有不通过项，见上"}')
    P('')

    # ================================================================ AE2
    P('######## AE2 国内与跨境：同 OA 下 AE−S2 前沿差（检验 3，随机 70/30、无缓冲；LightGBM）########')
    P('D_dom = within（W0）两特征曲线在 [min(OA★_ae, OA★_s2) − 4, min(...)] 上的 BA_ae − BA_s2 平均；D_xb 同理用 loco（L0），'
      '锚点取 loco 自己的 τ=0 OA。每国 1 个单元。')
    A2w = {}
    for st in SETTINGS:
        tag = '（主）' if st == 'c300' else '（次）'
        P(f'==== 设置 {st}{tag} ====')
        for w in WIDTHS:
            x = A2[(A2.setting == st) & (A2.width == w)]
            Wd = x.pivot(index='target', columns='model', values='D').reindex(ctrs)
            A2w[(st, w)] = Wd
            if w != 4:
                continue
            P(f'-- W{w} --')
            P(f'  D_dom            {ci(Wd["dom"])}')
            P(f'  D_xb             {ci(Wd["xb"])}')
            P(f'  D_xb − D_dom     {ci(Wd["xb"] - Wd["dom"])}')
            for mk, nm in (('dom', '国内'), ('xb', '跨境')):
                xx = x[x.model == mk].set_index('target').reindex(ctrs)
                P(f'  {nm}：共同可达率 最小 {xx["cov"].min():.2f} / 中位 {xx["cov"].median():.2f}；缺失国家 '
                  f'{",".join(xx.index[xx.D.isna()]) or "—"}；AE 全窗口占优 {int((xx.ae_dom == True).sum())}/{len(xx)}，'  # noqa: E712
                  f'S2 全窗口占优 {int((xx.s2_dom == True).sum())}/{len(xx)}；锚点中位 {xx.anchor.median():.1f}')  # noqa: E712
            P('  -- ae、s2 分列：窗口共同可达点上的 BA 前沿水平（mean BA over o）--')
            for mk, nm in (('dom', '国内'), ('xb', '跨境')):
                xx = x[x.model == mk].set_index('target').reindex(ctrs)
                P(f'  {nm} ae 水平       {ci(xx.lev_ae)}')
                P(f'  {nm} s2 水平       {ci(xx.lev_s2)}')
        P('  -- 对照：τ = 0 的 AE − S2 --')
        x = A2[(A2.setting == st) & (A2.width == 4)]
        for met, nm in (('ba0', 'BA'), ('oa0', 'OA')):
            Wm = {mk: (x[x.model == mk].set_index('target')[f'{met}_ae'] - x[x.model == mk].set_index('target')[f'{met}_s2'])
                  .reindex(ctrs) for mk in MODELS3}
            P(f'  τ=0 {nm} AE−S2 国内           {ci(Wm["dom"])}')
            P(f'  τ=0 {nm} AE−S2 跨境           {ci(Wm["xb"])}')
            P(f'  τ=0 {nm} AE−S2 跨境 − 国内    {ci(Wm["xb"] - Wm["dom"])}')
        for mk, nm in (('dom', '国内'), ('xb', '跨境')):
            xx = x[x.model == mk].set_index('target').reindex(ctrs)
            P(f'  ae、s2 分列 τ=0 BA {nm}：ae {ci(xx.ba0_ae)}；s2 {ci(xx.ba0_s2)}')
            P(f'  ae、s2 分列 τ=0 OA {nm}：ae {ci(xx.oa0_ae)}；s2 {ci(xx.oa0_s2)}')
        P('  -- 窗口敏感性（附加，非规格要求）：W2 / W8 --')
        for w in (2, 8):
            Wd = A2w[(st, w)]
            P(f'  W{w} D_dom {ci(Wd["dom"])}')
            P(f'  W{w} D_xb  {ci(Wd["xb"])}')
            P(f'  W{w} D_xb − D_dom {ci(Wd["xb"] - Wd["dom"])}')
        P('')
    P('-- AE2 逐国（c300，W4；pp）：D_dom、D_xb、D_xb − D_dom | τ=0 BA 的 AE−S2 国内、跨境 | 锚点 国内、跨境 | 共同可达率 国内、跨境 --')
    x = A2[(A2.setting == 'c300') & (A2.width == 4)]
    for t in ctrs:
        d_, x_ = x[(x.target == t) & (x.model == 'dom')].iloc[0], x[(x.target == t) & (x.model == 'xb')].iloc[0]
        P(f'  {t}  {d_.D:+6.2f} {x_.D:+6.2f} {x_.D - d_.D:+6.2f} | {d_.ba0_ae - d_.ba0_s2:+6.2f} {x_.ba0_ae - x_.ba0_s2:+6.2f} | '
          f'{d_.anchor:5.1f} {x_.anchor:5.1f} | {d_["cov"]:.2f} {x_["cov"]:.2f}')
    P('')

    # ================================================================ AE5
    P('######## AE5 零标签 EM 的先验误差（G 的 P_T，LightGBM；π̂_EM = em_prior(P_T 测试行, π_train)）########')
    P('L1_em = ‖π̂_EM − π_test‖₁；参照：L1_train = ‖π_train − π_test‖₁（不修正时模型内含的先验），L1_mean = ‖P_T 测试行平均预测 − '
      'π_test‖₁（τ 族所用 π̂）。国家值 = 4 单元平均（EM 只依赖 P_T 测试行与 π_train，同一划分两个重复相同）。')
    A5k = A5.copy()
    cv5 = {col: {f: A5k[A5k.clf == f].groupby('target')[col].mean().reindex(ctrs) for f in FEATS}
           for col in ('L1_em', 'L1_train', 'L1_mean', 'L1_raw_em', 'oa_t0_Z0', 'oa_t0_EM', 'ba_t0_Z0', 'ba_t0_EM',
                       'oa_t0_PKo')}
    for col, nm in (('L1_em', 'L1_em（主）'), ('L1_train', 'L1_train'), ('L1_mean', 'L1_mean'),
                    ('L1_raw_em', 'L1_raw_em（次：未校准 P0test_i300 上的 EM，= G 的 P1 臂）')):
        P(f'-- {nm} --')
        for f in FEATS:
            P(f'  {f}            {ci(cv5[col][f], 3)}')
        P(f'  AE−S2         {ci(cv5[col]["ae"] - cv5[col]["s2"], 3)}')
    P('-- EM 相对不修正的改进 L1_train − L1_em（>0 = EM 把先验估得更接近真实）--')
    for f in FEATS:
        P(f'  {f}            {ci(cv5["L1_train"][f] - cv5["L1_em"][f], 3)}')
    P(f'  AE−S2         {ci((cv5["L1_train"]["ae"] - cv5["L1_em"]["ae"]) - (cv5["L1_train"]["s2"] - cv5["L1_em"]["s2"]), 3)}')
    P('-- 描述：G P_T 上 EM 修正在 τ=0 的 OA / BA 变化（Z0EM − Z0），与测试真实先验修正（PKo − Z0）的 OA 变化 --')
    for f in FEATS:
        P(f'  {f} OA EM−Z0   {ci(cv5["oa_t0_EM"][f] - cv5["oa_t0_Z0"][f])}')
        P(f'  {f} BA EM−Z0   {ci(cv5["ba_t0_EM"][f] - cv5["ba_t0_Z0"][f])}')
        P(f'  {f} OA PKo−Z0  {ci(cv5["oa_t0_PKo"][f] - cv5["oa_t0_Z0"][f])}')
    em_ = A5k.groupby('clf').agg(conv=('em_conv', 'mean'), it=('em_iter', 'median'))
    P(f'  EM 收敛比例 ae {em_.loc["ae", "conv"]:.0%}、s2 {em_.loc["s2", "conv"]:.0%}；迭代数中位数 ae {em_.loc["ae", "it"]:.0f}、'
      f's2 {em_.loc["s2", "it"]:.0f}（未收敛 = 200 次迭代上限）')
    # 按类的有符号误差
    P('-- 按类：π̂_EM − π_test 的平均（11 国，先按国家平均；+ = 高估）| 平均 |误差| ae、s2 --')
    for c in range(C):
        e = {f: (A5k[A5k.clf == f][f'pi_em_{c}'] - A5k[A5k.clf == f][f'pi_test_{c}']).groupby(A5k[A5k.clf == f].target)
             .mean().reindex(ctrs) for f in FEATS}
        P(f'  {D.classes[c]:17s} ae {e["ae"].mean():+.4f}  s2 {e["s2"].mean():+.4f} | |·| ae {e["ae"].abs().mean():.4f}  '
          f's2 {e["s2"].abs().mean():.4f}')
    # 检验 3 关联
    P('-- 关联（描述）：检验 3 的 EM（L0 的 pool 预测上 em_prior，训练类上均匀先验；恢复率 = (PEM − L0)/(PO − L0)，OA）--')
    for st in SETTINGS:
        X = A5x[A5x.setting == st]
        cx = {col: {f: X[X.clf == f].set_index('target')[col].reindex(ctrs) for f in FEATS}
              for col in ('L1_em', 'L1_train', 'rec_oa', 'oa_PEM', 'oa_L0', 'oa_PO')}
        tag = '（检验 3 原口径）' if st == 'es' else '（本脚本新算，描述）'
        P(f'  [{st}{tag}]')
        for f in FEATS:
            P(f'    L1_em {f}       {ci(cx["L1_em"][f], 3)}')
        P(f'    L1_em AE−S2    {ci(cx["L1_em"]["ae"] - cx["L1_em"]["s2"], 3)}')
        P(f'    L1_train（均匀）AE−S2 {ci(cx["L1_train"]["ae"] - cx["L1_train"]["s2"], 3)}')
        gain = {f: cx['oa_PEM'][f] - cx['oa_L0'][f] for f in FEATS}
        P(f'    OA PEM−L0 AE−S2 {ci(gain["ae"] - gain["s2"])}')
        for f in FEATS:
            x = X[X.clf == f].set_index('target').reindex(ctrs)
            r11 = (x.oa_PEM - x.oa_L0).sum() / (x.oa_PO - x.oa_L0).sum()
            x9 = x.loc[[c for c in C9 if c in x.index]]
            r9 = (x9.oa_PEM - x9.oa_L0).sum() / (x9.oa_PO - x9.oa_L0).sum() if len(x9) else np.nan
            P(f'    合并恢复率 {f}：11 国 {100 * r11:.1f}%，9 国 {100 * r9:.1f}%')
    P('-- 逐国：G P_T 的 L1_em ae / s2 / AE−S2 | 检验 3（es）L1_em ae / s2 / AE−S2 | 检验 3（es）OA 恢复率 ae / s2 | '
      '检验 3 PO−L0 ae / s2（pp，恢复率的分母）--')
    X = A5x[A5x.setting == 'es']
    for t in ctrs:
        xa, xs = X[(X.target == t) & (X.clf == 'ae')].iloc[0], X[(X.target == t) & (X.clf == 's2')].iloc[0]
        ga, gs = cv5['L1_em']['ae'][t], cv5['L1_em']['s2'][t]
        P(f'  {t}  {ga:.3f} {gs:.3f} {ga - gs:+.3f} | {xa.L1_em:.3f} {xs.L1_em:.3f} {xa.L1_em - xs.L1_em:+.3f} | '
          f'{100 * xa.rec_oa:6.1f}% {100 * xs.rec_oa:6.1f}% | {xa.oa_PO - xa.oa_L0:+6.2f} {xs.oa_PO - xs.oa_L0:+6.2f}')
    dG = cv5['L1_em']['ae'] - cv5['L1_em']['s2']
    d3 = (X[X.clf == 'ae'].set_index('target').L1_em - X[X.clf == 's2'].set_index('target').L1_em).reindex(ctrs)
    dr = (X[X.clf == 'ae'].set_index('target').rec_oa - X[X.clf == 's2'].set_index('target').rec_oa).reindex(ctrs)
    dg = ((X[X.clf == 'ae'].set_index('target').oa_PEM - X[X.clf == 'ae'].set_index('target').oa_L0) -
          (X[X.clf == 's2'].set_index('target').oa_PEM - X[X.clf == 's2'].set_index('target').oa_L0)).reindex(ctrs)
    if len(ctrs) >= 3:
        for nm, v in (('检验 3 L1_em 的 AE−S2', d3), ('检验 3 恢复率的 AE−S2', dr), ('检验 3 OA PEM−L0 的 AE−S2', dg)):
            r = stats.spearmanr(dG, v, nan_policy='omit')
            r9 = stats.spearmanr(s9(dG), s9(v), nan_policy='omit')
            P(f'  Spearman（描述）：G L1_em 的 AE−S2 vs {nm}：11 国 ρ = {r[0]:+.2f}（p = {r[1]:.3f}）；9 国 ρ = {r9[0]:+.2f}')
    P('')

    # ================================================================ AE6
    P('######## AE6 类条件分离：混合指数 M（stage0 划分 0；G_f 候选行 vs GF_f 外国池；32 维联合 PCA；10 近邻）########')
    P(f'M_c：每类两边都 ≥ {AE6_MIN} 时各抽 min({AE6_NC}, n)；M_all：两边各抽 {AE6_NALL}。M = 目标样本 10 近邻中外国样本比例的平均；'
      '两边样本量相等时完全混合 ≈ 0.5，完全分离 → 0。E_mix = n_F/(n_T + n_F − 1) 为完全混合时的期望（样本量不等时 ≠ 0.5）；'
      'M_norm = M / E_mix 另列（方案外，描述）。')
    A6e = A6[A6.eligible]
    Mc = A6e[A6e.cls != 'all']
    Ma = A6e[A6e.cls == 'all']
    cm = {f: Mc[Mc.clf == f].groupby('target').M.median().reindex(ctrs) for f in FEATS}
    cmn = {f: Mc[Mc.clf == f].groupby('target').M_norm.median().reindex(ctrs) for f in FEATS}
    cmp_ = {f: Mc[Mc.clf == f].groupby('target').M_pair.median().reindex(ctrs) for f in FEATS}
    ca_ = {f: Ma[Ma.clf == f].set_index('target').M.reindex(ctrs) for f in FEATS}
    ncls = Mc[Mc.clf == 'ae'].groupby('target').size().reindex(ctrs)
    P(f'-- 每国可比类数：{ncls.to_dict()}；不等样本量（n_T ≠ n_F）的 (国家, 类) {int((Mc[Mc.clf == "ae"].n_sT != Mc[Mc.clf == "ae"].n_sF).sum())}'
      f'/{int((Mc.clf == "ae").sum())} --')
    for nm, dct in (('median_c M_c（主）', cm), ('M_all（主）', ca_), ('median_c M_norm（方案外）', cmn),
                    ('median_c M_c，两特征同一批地块（方案外）', cmp_)):
        P(f'-- {nm} --')
        for f in FEATS:
            P(f'  {f}         {ci(dct[f], 3)}')
        P(f'  AE−S2      {ci(dct["ae"] - dct["s2"], 3)}')
    # 与 AE1 Z0 D 的 Spearman
    dz = DZ[DZ.width == 4]
    Dc = dz.groupby('target').D.mean().reindex(ctrs)
    Dc0 = dz[dz.split == 0].groupby('target').D.mean().reindex(ctrs)
    P('-- 与 AE1 的 Z0 D（LightGBM，W4，锚点 min(OA★)，国家值 = 4 单元平均）的 Spearman（n = 11，描述）--')
    P(f'  AE1 Z0 D：{ci(Dc)}')
    if len(ctrs) >= 3:
        for nm, v in (('median_c M_c 的 AE−S2', cm['ae'] - cm['s2']), ('M_all 的 AE−S2', ca_['ae'] - ca_['s2']),
                      ('median_c M_norm 的 AE−S2', cmn['ae'] - cmn['s2']),
                      ('median_c M_c 的 AE−S2（同一批地块）', cmp_['ae'] - cmp_['s2']),
                      ('median_c M_c（ae 水平）', cm['ae']), ('median_c M_c（s2 水平）', cm['s2'])):
            r = stats.spearmanr(Dc, v, nan_policy='omit')
            r0 = stats.spearmanr(Dc0, v, nan_policy='omit')
            r9 = stats.spearmanr(s9(Dc), s9(v), nan_policy='omit')
            P(f'  D vs {nm:28s}：ρ = {r[0]:+.2f}（p = {r[1]:.3f}，n = {int(pd.concat([Dc, v], axis=1).dropna().shape[0])}）；'
              f'只用划分 0 的 D：ρ = {r0[0]:+.2f}；9 国 ρ = {r9[0]:+.2f}')
    P('-- 逐国：median M_c ae / s2 / AE−S2 | M_all ae / s2 / AE−S2 | 可比类数 | AE1 Z0 D --')
    for t in ctrs:
        P(f'  {t}  {cm["ae"][t]:.3f} {cm["s2"][t]:.3f} {cm["ae"][t] - cm["s2"][t]:+.3f} | {ca_["ae"][t]:.3f} {ca_["s2"][t]:.3f} '
          f'{ca_["ae"][t] - ca_["s2"][t]:+.3f} | {int(ncls[t]) if pd.notna(ncls[t]) else 0:2d} | {Dc[t]:+.2f}')
    P('-- 按类（描述）：M_c 的 AE−S2，在该类可比的国家上平均（国家数）；ae、s2 水平 --')
    W6 = Mc.pivot_table(index=['target', 'cls_name'], columns='clf', values='M').dropna()
    for cn, g in W6.groupby(level='cls_name'):
        dd = g['ae'] - g['s2']
        P(f'  {cn:17s} n={len(g):2d}  AE−S2 {dd.mean():+.3f}（>0 {int((dd > 0).sum())}/{len(g)}）  ae {g["ae"].mean():.3f}  '
          f's2 {g["s2"].mean():.3f}')
    P('')

    # ================================================================ 读法
    P('######## 读法（规格事先写定；AE2 适用 c300、W4；9 国与 11 国分别判定）########')
    for st in SETTINGS:
        Wd = A2w[(st, 4)]
        for drop in (True, False):
            v = (Wd['xb'] - Wd['dom']).dropna()
            v = s9(v) if drop else v
            s = B.t_summary(v)
            sd, sx = B.t_summary(s9(Wd['dom']) if drop else Wd['dom']), B.t_summary(s9(Wd['xb']) if drop else Wd['xb'])
            if not np.isfinite(s['lo95']):
                rd = '无法判定（国家数不足）'
            elif s['lo95'] > 0:
                rd = 'D_xb − D_dom 的 95% CI 下界 > 0：AE 的前沿优势在跨境时放大'
            elif s['hi95'] < 0:
                rd = 'D_xb − D_dom 的 95% CI 上界 < 0：AE 的前沿优势在跨境时缩小'
            else:
                rd = 'CI 覆盖 0：没有检出放大，按规格写成"AE 在国内与跨境都抬高前沿"'
                prem = (sd['lo95'] > 0, sx['lo95'] > 0)
                rd += (f'（前提核对：D_dom 下界 {"> 0" if prem[0] else "≤ 0"}，D_xb 下界 {"> 0" if prem[1] else "≤ 0"}'
                       + ('；两者都 > 0，措辞有数据支持）' if all(prem) else '；并非两者都显著 > 0，措辞需按实际 CI 收紧）'))
            P(f'  [{st}{"（主）" if st == "c300" else "（次）"} {"9 国" if drop else "11 国"}] D_xb − D_dom = {B.f95(s)}；'
              f'D_dom = {B.f95(sd)}；D_xb = {B.f95(sx)} → {rd}')
    P('  AE5、AE6 规格未设判定读法，只作描述（见上）。')
    if smoke:
        P('  （冒烟只有 2 个国家 / 2 个单元，上面的判定无意义，只验证代码路径。）')
    return lines


if __name__ == '__main__':
    main()
