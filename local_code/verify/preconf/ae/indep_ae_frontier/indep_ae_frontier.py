"""
确认运行登记前的不训练分析 —— ae_frontier 流的独立复核（AE1 部分量 + AE4 的 τ=0 分组 OA）。
规格：label_expansion/PRECONF_ANALYSES_SPEC_2026-09-29.md（sha256[:16]=d0906bd1417839d2，不改）。
事后分析，C 级、描述性；不训练任何分类器，只读已存概率、G 的 stage0 与选样记录、GData 缓存中的 y 与 area。
测试标签只用于评估。未读同事主实现（verify/preconf/ae/ae_frontier_*）。

本流复核的量：
  AE1  D = 同 OA 下 AE−S2 的前沿差（W4 = [锚点−4, 锚点]，步长 0.1，41 点）：
       - LightGBM：Z0 与 RT200 两臂，曲线直接读补遗 B2 的 curves_B2.csv（另从已存概率独立重算曲线做逐位对账）；
         锚点 = min(OA★_ae, OA★_s2)，OA★ = 该特征 Z0 在 τ=0 的 OA（两臂共用同一锚点）。
       - LR、MLP：k=0 已存概率（guidance_G_ml/probs/{lr,mlp}/{国家}_s{划分}_{f}_k0.npz 的 'P'）在 τ ∈ [−2, 2]
         步长 0.05 上重算 τ 族；锚点 = 本学习器 k=0 在 τ=0 时两特征 OA 的较小者。
       逐点 BA_ae(o) − BA_s2(o)（两特征各用自己的单曲线上包络，step0B.envelope），只在两者共同可达的 o 上平均；
       共同可达比例 < 50% 记缺失。另报 W2 / W8 与锚点取 OA★_ae / OA★_s2 的版本（描述）。
  AE4  LightGBM Z0 在 τ=0：测试地块按 面积 / (π·30² m²) 分 5 组 <0.5、[0.5,1)、[1,2)、[2,4)、≥4；
       每组：占测试地块比例、组内 OA 的 AE−S2（主）；另报组内 BA 的 AE−S2（只计该组 ≥30 个地块的类，<2 类记缺失）。
汇总：国家值 = 该国单元平均；11 国与 9 国（去 RO、PT）；均值、95% t CI（df = n−1）、>0 的国家数。

复用（import，不改）：step0B.tau_preds / Scorer / envelope / log_or_neginf / norm64（曲线与包络）；本流新增的
窗口比较、共同可达、国家汇总、面积分组全部在本文件中自写。
运行：/Users/perng_mini/AlphaCrops_local/.venv/bin/python -B indep_ae_frontier.py
"""
import os
import sys
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_v] = '2'
sys.dont_write_bytecode = True  # 不在项目目录写 __pycache__
import hashlib
import json
import time

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
LOCAL = '/Users/perng_mini/AlphaCrops_local'
PARTB = os.path.join(LOCAL, 'verify', 'step0', 'partB')
PARTB2 = os.path.join(LOCAL, 'verify', 'step0', 'partB2')
sys.path.insert(0, PARTB)
import step0B as B  # noqa: E402  导入时 chdir 到项目目录（只读）并导入 guidance_G

G = B.G
SPEC = os.path.join(B.L, 'PRECONF_ANALYSES_SPEC_2026-09-29.md')
SPEC_HASH = 'd0906bd1417839d2'
R_G = os.path.join(LOCAL, 'results', 'guidance_G')
R_ML = os.path.join(LOCAL, 'results', 'guidance_G_ml')
CACHE = os.path.join(LOCAL, 'data', 'cache_guidance_G')
DATASET = os.path.join(LOCAL, 'data', 'dataset_v2.parquet')

C = 10
FEATS = ['ae', 's2']
COUNTRIES = 'AT,BE,DE,EE,LT,LV,PT,RO,SE,SI,SK'.split(',')
DROP = ['RO', 'PT']
C9 = [c for c in COUNTRIES if c not in DROP]
TAUS = np.round(np.arange(-2.0, 2.0 + 1e-9, 0.05), 2)   # 81 点
assert len(TAUS) == 81
I_T0 = int(np.flatnonzero(TAUS == 0.0)[0])
I_T05 = int(np.flatnonzero(TAUS == 0.5)[0])
IDX_OLD = np.flatnonzero(TAUS >= -0.5 - 1e-9)                # 第 0 步 B 旧网格（−0.5..2.0，51 点）= 新网格子集，只用于对账
assert len(IDX_OLD) == 51
MIN_COV = 0.5
MIN_N = 30
assert G.MIN_N == MIN_N
KERNEL_M2 = np.pi * 30.0 ** 2                       # ≈ 2,827 m²
AREA_EDGES = np.array([0.5, 1.0, 2.0, 4.0])          # 面积 / 核面积 的分组界；左闭右开
AREA_LABELS = ['<0.5', '0.5-1', '1-2', '2-4', '>=4']


def sha16(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()[:16]


def offs(w):
    """窗口相对锚点的偏移：[−w, 0] 步长 0.1（与 step0B.OFFS 同一构造）。"""
    o = np.round(np.arange(-float(w), 0.0 + 1e-9, 0.1), 1)
    assert len(o) == int(round(10 * w)) + 1
    return o


assert np.array_equal(offs(4), B.OFFS)


# ---------------------------------------------------------------- 曲线（复用 step0B 的 τ 族与打分）
def curve_from_P(P, yT, big):
    """P：(n, C) 测试集概率。Z = 精确 log（0 → −inf）；π̂ = 该臂测试集平均预测；τ 网格 −2..2 步长 0.05。"""
    Z = B.log_or_neginf(B.norm64(P))
    Pa = np.exp(Z - Z.max(1, keepdims=True))
    Pa /= Pa.sum(1, keepdims=True)
    pihat = Pa.mean(0)
    saved = B.TAUS
    B.TAUS = TAUS
    try:
        PH = B.tau_preds(Z, pihat)
    finally:
        B.TAUS = saved
    oa, ba = B.Scorer(yT, big).curve(PH)
    return oa, ba


# ---------------------------------------------------------------- 本流新增：同 OA 窗口比较（自写）
def window_compare(cv_ae, cv_s2, anchor, w):
    """两特征各自单曲线上包络（step0B.envelope，单曲线情形），在 [anchor−w, anchor] 逐点做 BA_ae − BA_s2，
    只在共同可达点上平均；共同可达比例 < 50% → D 缺失。另返回占优标记（共同可达点上全 >、全 ≥）。"""
    grid = anchor + offs(w)
    ba_a = B.envelope([cv_ae], grid)[0]
    ba_s = B.envelope([cv_s2], grid)[0]
    ok = np.isfinite(ba_a) & np.isfinite(ba_s)
    cov = float(ok.mean())
    out = dict(cov=cov, reach_ae=float(np.isfinite(ba_a).mean()), reach_s2=float(np.isfinite(ba_s).mean()))
    if cov >= MIN_COV:
        d = ba_a[ok] - ba_s[ok]
        out.update(D=float(d.mean()), ae_dom=bool((d > 0).all()), s2_dom=bool((d < 0).all()),
                   ae_dom_ge=bool((d >= 0).all()), s2_dom_ge=bool((d <= 0).all()))
    else:
        out.update(D=np.nan, ae_dom=np.nan, s2_dom=np.nan, ae_dom_ge=np.nan, s2_dom_ge=np.nan)
    return out


# ---------------------------------------------------------------- 本流新增：汇总（自写）
def tsum(v):
    v = pd.Series(v, dtype=float).dropna()
    n = len(v)
    if n < 2:
        return dict(n=n, mean=np.nan, lo=np.nan, hi=np.nan, npos=int((v > 0).sum()), sd=np.nan)
    m, sd = float(v.mean()), float(v.std(ddof=1))
    h = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
    return dict(n=n, mean=m, lo=m - h, hi=m + h, npos=int((v > 0).sum()), sd=sd)


def fmt(s):
    if s['n'] < 2:
        return f'n={s["n"]}（不足 2 国）'
    return f'{s["mean"]:+.2f} [{s["lo"]:+.2f}, {s["hi"]:+.2f}]，>0 {s["npos"]}/{s["n"]}'


def cvals(U, col, key='target'):
    """国家值 = 该国可用单元的平均。"""
    return U.groupby(key)[col].mean().reindex(COUNTRIES)


def area_group(area_ha):
    r = np.asarray(area_ha, dtype=np.float64) * 1e4 / KERNEL_M2
    return np.searchsorted(AREA_EDGES, r, side='right'), r  # <0.5→0，[0.5,1)→1，…，≥4→4


# ---------------------------------------------------------------- 主程序
def main():
    t0 = time.time()
    logs = []

    def P(s=''):
        print(s, flush=True)
        logs.append(s)

    assert sha16(SPEC) == SPEC_HASH, '规格哈希与登记值不一致'
    meta = json.load(open(os.path.join(CACHE, 'meta.json')))
    st = os.stat(DATASET)
    assert meta['source'] == f'{DATASET}|{st.st_size}|{int(st.st_mtime)}', 'GData 缓存与数据集不一致'
    y = np.load(os.path.join(CACHE, 'y.npy'))
    area = np.load(os.path.join(CACHE, 'area.npy'))
    ctry = np.load(os.path.join(CACHE, 'ctry.npy'))

    cur_b2 = pd.read_csv(os.path.join(PARTB2, 'curves_B2.csv'), float_precision='round_trip')
    cur_b2['ti'] = np.rint(cur_b2.tau * 100).astype(int)
    cur_b = pd.read_csv(os.path.join(PARTB, 'curves_B.csv'), float_precision='round_trip')
    cur_b['ti'] = np.rint(cur_b.tau * 100).astype(int)
    GR = B.load_G_rows()
    MLR = pd.concat([pd.read_csv(os.path.join(R_ML, f'rows_{lrn}_{t}_s{s}_r{r}.csv')).assign(learner=lrn)
                     for lrn in ('lr', 'mlp') for t in COUNTRIES for s in (0, 1) for r in (0, 1)], ignore_index=True)
    MLR = MLR[(MLR.method == 'foreign_only') & (MLR.cond == 'none') & (MLR.k == 0)]

    def b2_curve(unit, f, arm):
        g = cur_b2[(cur_b2.unit == unit) & (cur_b2.clf == f) & (cur_b2.arm == arm)].sort_values('ti')
        assert np.array_equal(g.ti.to_numpy(), np.rint(TAUS * 100).astype(int)), (unit, f, arm)
        return g.oa.to_numpy(), g.ba_ge30.to_numpy()

    CHK = []           # 对账记录
    U1 = []            # AE1 单元行
    U4 = []            # AE4 单元 × 组行
    ML_CURVES = []     # LR/MLP k=0 新网格曲线
    for tgt in COUNTRIES:
        for split in (0, 1):
            S = np.load(os.path.join(R_G, 'stage0', f'{tgt}_s{split}.npz'))
            n_cand = len(S['cand'])
            test = S['test']
            assert np.array_equal(S['allT'][n_cand:], test)
            assert (ctry[test] == tgt).all()
            yT = y[test]
            nT = len(yT)
            big = np.flatnonzero(np.bincount(yT, minlength=C) >= MIN_N)

            # ---- LightGBM Z0：独立重算曲线（从 P_T 测试行），与 curves_B2 对账
            PT = {f: S[f'P_T_{f}'][n_cand:] for f in FEATS}
            cz = {f: curve_from_P(PT[f], yT, big) for f in FEATS}
            # ---- LR / MLP k=0 曲线
            cml = {}
            for lrn in ('lr', 'mlp'):
                for f in FEATS:
                    Pk = np.load(os.path.join(R_ML, 'probs', lrn, f'{tgt}_s{split}_{f}_k0.npz'))['P']
                    assert Pk.shape == (nT, C), (lrn, tgt, split, f, Pk.shape)
                    cml[(lrn, f)] = curve_from_P(Pk, yT, big)
                    oa, ba = cml[(lrn, f)]
                    ML_CURVES += [dict(target=tgt, split=split, learner=lrn, clf=f, tau=TAUS[i], oa=oa[i], ba_ge30=ba[i])
                                  for i in range(len(TAUS))]
                    n_zero = int((np.asarray(Pk) == 0).sum())
                    # τ=0 与 rows_{lrn}（foreign_only，k=0；两个 rep 的记录应相同）
                    rr = MLR[(MLR.learner == lrn) & (MLR.target == tgt) & (MLR.split == split) & (MLR.clf == f)]
                    assert len(rr) == 2, (lrn, tgt, split, f, len(rr))
                    CHK.append(dict(check=f'{lrn}_k0_tau0_vs_rows', target=tgt, split=split, clf=f,
                                    d_oa=float(np.abs(rr.oa.to_numpy() - oa[I_T0]).max()),
                                    d_ba=float(np.abs(rr.ba_ge30.to_numpy() - ba[I_T0]).max()), n_zero_prob=n_zero))
                    # 旧网格重叠 τ 与 partB curves_B.csv 的 K0lr / K0mlp
                    for rep in (0, 1):
                        g = cur_b[(cur_b.unit == f'{tgt}_s{split}_r{rep}') & (cur_b.clf == f) &
                                  (cur_b.arm == f'K0{lrn}')].sort_values('ti')
                        idx = np.searchsorted(np.rint(TAUS * 100).astype(int), g.ti.to_numpy())
                        CHK.append(dict(check=f'{lrn}_k0_curve_vs_curves_B', target=tgt, split=split, clf=f, rep=rep,
                                        n=len(g), d_oa=float(np.abs(g.oa.to_numpy() - oa[idx]).max()),
                                        d_ba=float(np.abs(g.ba_ge30.to_numpy() - ba[idx]).max())))

            # ---- AE4：LightGBM Z0 在 τ=0，按面积分组（(国家, 划分) 层；两个 rep 的 Z0 相同）
            grp, ratio = area_group(area[test])
            ph0 = {f: B.norm64(PT[f]).argmax(1) for f in FEATS}
            for f in FEATS:  # τ=0 的 argmax 与曲线 τ=0 列一致
                assert abs(100 * (ph0[f] == yT).mean() - cz[f][0][I_T0]) < 1e-9
            for gi, gl in enumerate(AREA_LABELS):
                m = grp == gi
                row = dict(target=tgt, split=split, group=gl, n=int(m.sum()), frac=float(m.mean()))
                for f in FEATS:
                    if m.any():
                        row[f'oa_{f}'] = 100 * float((ph0[f][m] == yT[m]).mean())
                        cls = np.flatnonzero(np.bincount(yT[m], minlength=C) >= MIN_N)
                        row[f'ncls_{f}'] = len(cls)
                        row[f'ba_{f}'] = (100 * float(np.mean([(ph0[f][m & (yT == c)] == c).mean() for c in cls]))
                                          if len(cls) >= 2 else np.nan)
                    else:
                        row[f'oa_{f}'] = row[f'ba_{f}'] = np.nan
                        row[f'ncls_{f}'] = 0
                row['d_oa'] = row['oa_ae'] - row['oa_s2']
                row['d_ba'] = row['ba_ae'] - row['ba_s2']
                U4.append(row)
            # 分组比例与总体 OA 的恒等式：Σ frac_g·OA_g = 总体 OA
            for f in FEATS:
                tot = sum(r['frac'] * r[f'oa_{f}'] for r in U4[-5:] if r['n'] > 0)
                CHK.append(dict(check='AE4_group_oa_identity', target=tgt, split=split, clf=f,
                                d_oa=abs(tot - cz[f][0][I_T0])))

            for rep in (0, 1):
                unit = f'{tgt}_s{split}_r{rep}'
                # Z0 曲线逐位对账（curves_B2 的 Z0）
                cb2 = {f: b2_curve(unit, f, 'Z0') for f in FEATS}
                for f in FEATS:
                    CHK.append(dict(check='Z0_curve_recompute_vs_B2', target=tgt, split=split, rep=rep, clf=f,
                                    bitwise=bool(np.array_equal(cz[f][0], cb2[f][0]) and np.array_equal(cz[f][1], cb2[f][1])),
                                    d_oa=float(np.abs(cz[f][0] - cb2[f][0]).max()),
                                    d_ba=float(np.abs(cz[f][1] - cb2[f][1]).max())))
                # RT200：读 curves_B2；另从已存概率独立重算做逐位对账
                crt = {f: b2_curve(unit, f, 'RT200') for f in FEATS}
                sel_hashes = {}
                for f in FEATS:
                    g = GR[(GR.method == 'srs') & (GR.target == tgt) & (GR.split == split) & (GR.rep == rep) &
                           (GR.clf == f) & (GR.k == 200)]
                    assert len(g) == 1
                    sel_hashes[f] = g.iloc[0].sel_hash
                    Pr = np.load(os.path.join(R_G, 'probs', f'{unit}_{f}_k200_{g.iloc[0].sel_hash}.npz'))['i300']
                    oa_r, ba_r = curve_from_P(Pr, yT, big)
                    CHK.append(dict(check='RT200_curve_recompute_vs_B2', target=tgt, split=split, rep=rep, clf=f,
                                    bitwise=bool(np.array_equal(oa_r, crt[f][0]) and np.array_equal(ba_r, crt[f][1])),
                                    d_oa=float(np.abs(oa_r - crt[f][0]).max()), d_ba=float(np.abs(ba_r - crt[f][1]).max()),
                                    d_oa_rows=abs(oa_r[I_T0] - g.iloc[0].oa), d_ba_rows=abs(ba_r[I_T0] - g.iloc[0].ba_ge30)))
                CHK.append(dict(check='RT200_same_sel_both_feats', target=tgt, split=split, rep=rep,
                                bitwise=bool(sel_hashes['ae'] == sel_hashes['s2'])))

                # ---- AE1 单元行
                oas_ae, oas_s2 = cb2['ae'][0][I_T0], cb2['s2'][0][I_T0]
                anchors = {'min': min(oas_ae, oas_s2), 'ae': oas_ae, 's2': oas_s2}
                for arm, cv in (('Z0', cb2), ('RT200', crt)):
                    base = dict(target=tgt, split=split, rep=rep, unit=unit, learner='lgbm', arm=arm,
                                OAstar_ae=oas_ae, OAstar_s2=oas_s2,
                                d_oa_t0=cv['ae'][0][I_T0] - cv['s2'][0][I_T0], d_ba_t0=cv['ae'][1][I_T0] - cv['s2'][1][I_T0],
                                d_oa_t05=cv['ae'][0][I_T05] - cv['s2'][0][I_T05],
                                d_ba_t05=cv['ae'][1][I_T05] - cv['s2'][1][I_T05])
                    for an, a in anchors.items():
                        for w in (2, 4, 8):
                            if an != 'min' and w != 4:
                                continue
                            r = window_compare(cv['ae'], cv['s2'], a, w)
                            base.update({f'{k}_W{w}_{an}': v for k, v in r.items()})
                    # 对账用（非规格口径）：旧网格子集；以及本臂自身 τ=0 OA 的锚点（整合评估 matched_oa_lr_mlp.py 的做法）
                    old = {f: (cv[f][0][IDX_OLD], cv[f][1][IDX_OLD]) for f in FEATS}
                    own = min(cv['ae'][0][I_T0], cv['s2'][0][I_T0])
                    base['D_W4_min_oldgrid'] = window_compare(old['ae'], old['s2'], anchors['min'], 4)['D']
                    base['D_W4_own_newgrid'] = window_compare(cv['ae'], cv['s2'], own, 4)['D']
                    base['D_W4_own_oldgrid'] = window_compare(old['ae'], old['s2'], own, 4)['D']
                    U1.append(base)
                for lrn in ('lr', 'mlp'):
                    ca, cs = cml[(lrn, 'ae')], cml[(lrn, 's2')]
                    oa_a, oa_s = ca[0][I_T0], cs[0][I_T0]
                    base = dict(target=tgt, split=split, rep=rep, unit=unit, learner=lrn, arm='K0',
                                OAstar_ae=oa_a, OAstar_s2=oa_s, d_oa_t0=oa_a - oa_s, d_ba_t0=ca[1][I_T0] - cs[1][I_T0],
                                d_oa_t05=ca[0][I_T05] - cs[0][I_T05], d_ba_t05=ca[1][I_T05] - cs[1][I_T05])
                    for an, a in {'min': min(oa_a, oa_s), 'ae': oa_a, 's2': oa_s}.items():
                        for w in (2, 4, 8):
                            if an != 'min' and w != 4:
                                continue
                            r = window_compare(ca, cs, a, w)
                            base.update({f'{k}_W{w}_{an}': v for k, v in r.items()})
                    old = {f: (cml[(lrn, f)][0][IDX_OLD], cml[(lrn, f)][1][IDX_OLD]) for f in FEATS}
                    base['D_W4_min_oldgrid'] = window_compare(old['ae'], old['s2'], min(oa_a, oa_s), 4)['D']
                    base['D_W4_own_newgrid'] = base['D_W4_min']
                    base['D_W4_own_oldgrid'] = base['D_W4_min_oldgrid']
                    U1.append(base)
            P(f'[{time.time() - t0:.0f}s] {tgt} s{split} 完成（测试 {nT:,}，≥30 类 {len(big)}）')

    U1, U4, CHK, MLC = map(pd.DataFrame, (U1, U4, CHK, ML_CURVES))
    U1.to_csv(os.path.join(HERE, 'ae1_units.csv'), index=False)
    U4.to_csv(os.path.join(HERE, 'ae4_units_tau0.csv'), index=False)
    CHK.to_csv(os.path.join(HERE, 'checks.csv'), index=False)
    MLC.to_csv(os.path.join(HERE, 'curves_lr_mlp_k0.csv'), index=False)
    # 国家值表（便于与主实现逐国对账）
    CT = []
    for (lrn, arm), x in U1.groupby(['learner', 'arm']):
        for col in ('D_W4_min', 'D_W2_min', 'D_W8_min', 'D_W4_ae', 'D_W4_s2', 'd_ba_t0', 'd_oa_t0', 'd_ba_t05', 'd_oa_t05'):
            cv = cvals(x, col)
            CT += [dict(learner=lrn, arm=arm, quantity=col, target=t, value=cv[t]) for t in COUNTRIES]
    pd.DataFrame(CT).to_csv(os.path.join(HERE, 'ae1_country.csv'), index=False)
    C4 = U4.groupby(['target', 'group'])[['n', 'frac', 'oa_ae', 'oa_s2', 'd_oa', 'ba_ae', 'ba_s2', 'd_ba']].mean().reset_index()
    C4.to_csv(os.path.join(HERE, 'ae4_country_tau0.csv'), index=False)

    # ================================================================ 报告
    L = []
    Q = L.append
    Q('ae_frontier 流独立复核（AE1：LightGBM Z0 / RT200、LR / MLP k=0 的 D；AE4：τ=0 分组 OA 的 AE−S2）—— 事后分析，C 级、描述性。')
    Q(f'规格 PRECONF_ANALYSES_SPEC_2026-09-29.md sha256[:16]={sha16(SPEC)}（登记 {SPEC_HASH}）；'
      f'step0B.py sha256[:16]={sha16(os.path.join(PARTB, "step0B.py"))}（只 import）；'
      f'curves_B2.csv sha256[:16]={sha16(os.path.join(PARTB2, "curves_B2.csv"))}；'
      f'本脚本 sha256[:16]={sha16(os.path.abspath(__file__))}。')
    Q('口径：τ ∈ [−2, 2] 步长 0.05；π̂ = 该臂测试集平均预测；精确 log；单曲线上包络（相邻 τ 线性插值，多段取 BA 最大）；'
      '窗口 [锚点−w, 锚点] 步长 0.1；D = 共同可达 o 上 BA_ae − BA_s2 的平均，共同可达 <50% 记缺失；'
      '国家值 = 单元平均；95% t CI；9 国 = 去 RO、PT。')
    Q('')
    Q('== 对账 ==')
    for ck, g in CHK.groupby('check', sort=False):
        txt = f'  {ck:34s} {len(g):3d} 行'
        if 'bitwise' in g and g.bitwise.notna().any():
            txt += f'；逐位相同 {int(g.bitwise.fillna(False).astype(bool).sum())}/{int(g.bitwise.notna().sum())}'
        for c in ('d_oa', 'd_ba', 'd_oa_rows', 'd_ba_rows'):
            if c in g and g[c].notna().any():
                txt += f'；max|{c}| {g[c].max():.2e}'
        if 'n_zero_prob' in g and g.n_zero_prob.notna().any():
            txt += f'；概率恰为 0 的元素合计 {int(g.n_zero_prob.sum())}'
        Q(txt)
    # 与 AE–S2 整合评估已有 Z0 D（matched_oa_units.csv 的 W4_min_d）对账
    MO = pd.read_csv(os.path.join(LOCAL, 'verify', 'storyline', 'ae_s2_integration_20260929', 'ae_s2_evidence',
                                  'matched_oa_units.csv'))
    z = U1[(U1.learner == 'lgbm') & (U1.arm == 'Z0')].merge(MO[['unit', 'W4_min_d', 'W4_min_cov']], on='unit')
    Q(f'  Z0 D(W4,min) vs 整合评估 matched_oa_units.csv W4_min_d：{len(z)} 单元，max|Δ| {np.nanmax(np.abs(z.D_W4_min - z.W4_min_d)):.2e}，'
      f'缺失一致 {int((z.D_W4_min.isna() == z.W4_min_d.isna()).sum())}/{len(z)}；共同可达率 max|Δ| {np.abs(z.cov_W4_min - z.W4_min_cov).max():.2e}'
      '（整合评估的网格为 round(arange(锚点−4, 锚点, 0.1), 6)，与本流 锚点 + OFFS 在 1e-6 级不同，允许微小差异）')
    Q('  与整合评估 matched_oa_lr_mlp_out.txt（旧网格 τ∈[−0.5,2]、锚点 = 各臂自身 τ=0 OA 的两特征较小者）并列（D W4；9 国 | 11 国）：')
    for lrn, arm in (('lgbm', 'Z0'), ('lgbm', 'RT200'), ('lr', 'K0'), ('mlp', 'K0')):
        x = U1[(U1.learner == lrn) & (U1.arm == arm)]
        txt = []
        for col, lab in (('D_W4_min', '规格口径'), ('D_W4_min_oldgrid', '旧网格·Z0锚点'), ('D_W4_own_newgrid', '新网格·自身锚点'),
                         ('D_W4_own_oldgrid', '旧网格·自身锚点')):
            cv = cvals(x, col)
            txt.append(f'{lab} {tsum(cv.loc[C9])["mean"]:+.2f} | {tsum(cv)["mean"]:+.2f}')
        Q(f'    {lrn:4s} {arm:5s} ' + '；'.join(txt))
        cv = cvals(x, 'D_W4_own_oldgrid')
        Q('         旧网格·自身锚点 国家值：' + '，'.join(f'{t} {cv[t]:+.2f}' for t in COUNTRIES))
    Q('')

    Q('######## AE1：同 OA 下 AE−S2 前沿差 D（pp；国家值 = 单元平均）########')
    for lrn, arm in (('lgbm', 'Z0'), ('lgbm', 'RT200'), ('lr', 'K0'), ('mlp', 'K0')):
        x = U1[(U1.learner == lrn) & (U1.arm == arm)]
        Q(f'-- {lrn} {arm}（单元 {len(x)}；锚点 = {"LightGBM Z0" if lrn == "lgbm" else lrn + " k=0"} 在 τ=0 的两特征 OA 较小者）--')
        for col, lab in (('D_W4_min', 'D  W4 锚点 min（主）'), ('D_W2_min', 'D  W2 锚点 min'), ('D_W8_min', 'D  W8 锚点 min'),
                         ('D_W4_ae', 'D  W4 锚点 OA★_ae'), ('D_W4_s2', 'D  W4 锚点 OA★_s2'),
                         ('d_ba_t0', 'τ=0   BA AE−S2'), ('d_oa_t0', 'τ=0   OA AE−S2'),
                         ('d_ba_t05', 'τ=0.5 BA AE−S2'), ('d_oa_t05', 'τ=0.5 OA AE−S2')):
            cv = cvals(x, col)
            Q(f'  {lab:20s} 9 国 {fmt(tsum(cv.loc[C9]))} | 11 国 {fmt(tsum(cv))}')
        nmiss = int(x.D_W4_min.isna().sum())
        cov = x.cov_W4_min
        Q(f'  W4 min：缺失单元 {nmiss}/{len(x)}；共同可达率 <100% 的单元 {int((cov < 1).sum())}，最小 {cov.min():.2f}；'
          f'AE 全窗口占优（共同可达点全 >）{int(x.ae_dom_W4_min.fillna(False).astype(bool).sum())}/{len(x)}，'
          f'S2 全窗口占优 {int(x.s2_dom_W4_min.fillna(False).astype(bool).sum())}/{len(x)}'
          f'（≥ 口径：{int(x.ae_dom_ge_W4_min.fillna(False).astype(bool).sum())} / {int(x.s2_dom_ge_W4_min.fillna(False).astype(bool).sum())}）')
        cv = cvals(x, 'D_W4_min')
        Q('  国家值 D(W4,min)：' + '，'.join(f'{t} {cv[t]:+.2f}' for t in COUNTRIES))
    Q('')

    Q('######## AE4：LightGBM Z0 在 τ=0 按面积分组（面积 / (π·30² m²)）；pp ########')
    Q('  组      占测试地块比例（11 国平均） | 组内 OA AE−S2：9 国 | 11 国 | 组内 BA AE−S2（≥30 地块的类，<2 类缺失）：9 国 | 11 国')
    for gl in AREA_LABELS:
        x = U4[U4.group == gl]
        fr = cvals(x, 'frac')
        co, cb = cvals(x, 'd_oa'), cvals(x, 'd_ba')
        Q(f'  {gl:6s}  {fr.mean():.3f}（9 国 {fr.loc[C9].mean():.3f}；最小 n {int(x.n.min())}） | '
          f'{fmt(tsum(co.loc[C9]))} | {fmt(tsum(co))} | {fmt(tsum(cb.loc[C9]))} | {fmt(tsum(cb))}')
    Q('  国家值：组内 OA AE−S2（行 = 国家；列 = 面积组）')
    tab = U4.groupby(['target', 'group'])['d_oa'].mean().unstack('group').reindex(index=COUNTRIES, columns=AREA_LABELS)
    for t in COUNTRIES:
        Q(f'    {t}  ' + ' '.join(f'{tab.loc[t, g]:+7.2f}' for g in AREA_LABELS))
    Q('  国家值：各组占测试地块比例')
    tf = U4.groupby(['target', 'group'])['frac'].mean().unstack('group').reindex(index=COUNTRIES, columns=AREA_LABELS)
    for t in COUNTRIES:
        Q(f'    {t}  ' + ' '.join(f'{tf.loc[t, g]:7.3f}' for g in AREA_LABELS))
    tn = U4.groupby(['target', 'group'])['n'].min().unstack('group').reindex(index=COUNTRIES, columns=AREA_LABELS)
    Q('  各国两划分中该组测试地块数的较小者：')
    for t in COUNTRIES:
        Q(f'    {t}  ' + ' '.join(f'{int(tn.loc[t, g]):7d}' for g in AREA_LABELS))
    Q(f'  （注：OA 规格未设组内最小地块数，按字面 n≥1 即计算；n<30 的 (国家,划分,组) 共 {int((U4.n < 30).sum())} 个，'
      f'n=0 的 {int((U4.n == 0).sum())} 个。）')
    Q('')
    Q(f'总耗时 {time.time() - t0:.0f}s')
    open(os.path.join(HERE, 'summary_indep.txt'), 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
