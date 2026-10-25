"""
G 实验分析（ANALYSIS_PLAN_guidance.md 第 7–8 节）。

用法：
  python analyze_G.py                 # 正式分析（层级 1 与 2 全部完成后才解读）
  python analyze_G.py --noise         # 指标换成随机数，只用于在不看结果的情况下测试流程
  python analyze_G.py --allow-partial # 允许单元不全（只用于流程测试；报告首行会标明）
输出：RESULTS_DIR/guidance_G/report_G[_NOISE].txt 与 summary_G[_NOISE].csv
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

from guidance_G import E1, FAMILIES, OUT, COUNTRIES_G

NOISE = '--noise' in sys.argv
PARTIAL = '--allow-partial' in sys.argv
SESOI = 0.3
OA_NI = -0.5
RISK = -0.5
CHECK2 = ['PT', 'SE', 'SK', 'LV']
DEV = ['PT', 'BE', 'SI']
UNIT = ['target', 'split', 'rep']
KEY = UNIT + ['method', 'clf', 'k']
METRICS = ['ba_ge30', 'oa', 'f1_ge30', 'ba_P1', 'oa_P1', 'ba_P2', 'oa_P2', 'ba_far', 'oa_far',
           'ba_i200', 'ba_i250', 'oa_i200', 'oa_i250']


class Tee:
    def __init__(self, path):
        self.f = open(path, 'w')

    def write(self, s):
        sys.__stdout__.write(s)
        self.f.write(s)

    def flush(self):
        self.f.flush()


def load_rows(out=None):
    fs = glob.glob(os.path.join(out or OUT, 'rows_t*_*.csv'))
    R = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    if NOISE:
        rng = np.random.default_rng(0)
        for m in METRICS:
            if m in R:
                R[m] = rng.normal(60, 5, len(R))
    return R


# ------------------------------------------------------------------ 估计量
def deltas(R, cond_a, cond_b, methods, metric='ba_ge30', reps=None, ks=(50, 200), clfs=('s2', 'ae')):
    """同一 (单元, 方法, 分类器, k) 上 cond_a − cond_b 的配对差。默认只取主分析的 k 与两种分类器，
    层级 3 的融合分类器（clf='fus'）与 k=1000 的行不会混入。"""
    sub = R[R.k.isin(ks) & R.clf.isin(clfs) & R.method.isin(methods)]
    a = sub[sub.cond == cond_a].set_index(KEY)[metric]
    b = sub[sub.cond == cond_b].set_index(KEY)[metric]
    d = (a - b).dropna().rename('d').reset_index()
    if reps is not None:
        d = d[d.rep.isin(reps)]
    return d


def country_values(d, col='d'):
    """先在 (方法, 分类器, k) 内对单元取平均，再对这些格子等权平均 → 每国一个值。"""
    return d.groupby(['target', 'method', 'clf', 'k'])[col].mean().groupby('target').mean()


def t_summary(v):
    """国家值的单样本 t 推断（df = n − 1）。"""
    v = np.asarray(v, dtype=float)
    n = len(v)
    m, se = v.mean(), v.std(ddof=1) / np.sqrt(n)
    q95, q90 = stats.t.ppf(0.975, n - 1), stats.t.ppf(0.95, n - 1)
    p = 2 * stats.t.sf(abs(m / se), n - 1) if se > 0 else np.nan
    return dict(n=n, mean=m, se=se, lo95=m - q95 * se, hi95=m + q95 * se, lo90=m - q90 * se, hi90=m + q90 * se,
                p=p, n_pos=int((v > 0).sum()))


def verdict(s, sesoi=SESOI, leq=None):
    """方案第 8 节的判定表（θ_E1，ba_ge30，P0），与表中措辞逐字对齐。leq：不确定时附上的标签当量文字。"""
    if s['lo95'] > 0:
        return 'AE 更好' + ('（统计上略好，但优势小于 0.3 pp）' if s['hi90'] < sesoi else '')
    if s['hi95'] < 0:
        return 'S2 更好'
    if s['lo90'] > -sesoi and s['hi90'] < sesoi:
        return '等价（±0.3 pp，TOST）'
    return f'不确定：效应被限定在 [{s["lo95"]:.2f}, {s["hi95"]:.2f}] pp' + (f'，约合 {leq} 个随机标签' if leq else '')


def unit_level(d, col='d'):
    """单元值 = 该单元内各 (方法, 分类器, k) 格子的平均。"""
    return d.groupby(UNIT)[col].mean()


def reml_tau2(y, v):
    """随机效应元分析的 REML τ²（v 为已知的国家内方差）。"""
    from scipy.optimize import minimize_scalar
    y, v = np.asarray(y, float), np.asarray(v, float)

    def nll(t2):
        w = 1 / (v + t2)
        mu = (w * y).sum() / w.sum()
        return 0.5 * (np.log(v + t2).sum() + np.log(w.sum()) + (w * (y - mu) ** 2).sum())
    hi = max(10 * np.var(y, ddof=1), 1e-6)
    r = minimize_scalar(nll, bounds=(0, hi), method='bounded', options={'xatol': 1e-10})
    return float(r.x) if nll(r.x) < nll(0.0) else 0.0


def prediction_interval(y, v, level=0.80):
    """R0：μ̂（REML 权重）± t_{(1+level)/2, n−2}·√(τ̂² + SE_HKSJ²)。"""
    y, v = np.asarray(y, float), np.asarray(v, float)
    n = len(y)
    t2 = reml_tau2(y, v)
    w = 1 / (v + t2)
    mu = (w * y).sum() / w.sum()
    se = np.sqrt((w * (y - mu) ** 2).sum() / ((n - 1) * w.sum()))
    q = stats.t.ppf((1 + level) / 2, n - 2)
    h = q * np.sqrt(t2 + se ** 2)
    return dict(mu=mu, tau=np.sqrt(t2), se_hksj=se, pi_lo=mu - h, pi_hi=mu + h)


def within_var(d, col='d'):
    """每国国家值的方差：单元值的方差 / 单元数，跨国取合并值（方案：国家内方差取合并值）。"""
    u = unit_level(d, col).groupby('target')
    v = (u.var(ddof=1) / u.size()).dropna()
    return float(v.mean())


def webb_wcb(v, B=9999, seed=0):
    """国家均值的 wild cluster bootstrap（Webb 六点权重，施加 H0: μ = 0 的受限版），返回双侧 p。"""
    v = np.asarray(v, float)
    n = len(v)
    t0 = v.mean() / (v.std(ddof=1) / np.sqrt(n))
    W = np.random.default_rng(seed).choice(np.r_[-np.sqrt(1.5), -1, -np.sqrt(.5), np.sqrt(.5), 1, np.sqrt(1.5)],
                                           size=(B, n))
    ys = W * v[None, :]
    ts = ys.mean(1) / (ys.std(1, ddof=1) / np.sqrt(n))
    return float((np.abs(ts) >= abs(t0)).mean())


def shrink_methods(d):
    """逐方法部分合并：方法国家均值的均值与 SE，跨方法 REML τ²，收缩 = μ + B_j(m_j − μ)。"""
    rows = []
    for m, g in d.groupby('method'):
        s = t_summary(country_values(g))
        rows.append(dict(method=m, raw=s['mean'], se=s['se']))
    M = pd.DataFrame(rows)
    if len(M) < 2:
        return M
    t2 = reml_tau2(M.raw, M.se ** 2)
    w = 1 / (M.se ** 2 + t2)
    mu = (w * M.raw).sum() / w.sum()
    Bj = t2 / (t2 + M.se ** 2)
    M['shrunk'] = mu + Bj * (M.raw - mu)
    M['shrunk_se'] = np.sqrt(Bj * M.se ** 2 + (1 - Bj) ** 2 / w.sum())
    M['tau_methods'] = np.sqrt(t2)
    return M


# ------------------------------------------------------------------ 可靠性
def vs_srs(R, cond, methods, metric='ba_ge30', ks=(50, 200), clfs=('s2', 'ae')):
    """E1 方法在某指导下相对 SRS 的差（同一单元、分类器、k）。"""
    R = R[R.k.isin(ks) & R.clf.isin(clfs)]
    a = R[(R.cond == cond) & R.method.isin(methods)].set_index(KEY)[metric]
    s = R[R.method == 'srs'].set_index(UNIT + ['clf', 'k'])[metric]
    d = a.reset_index()
    d['d'] = a.to_numpy() - s.reindex(pd.MultiIndex.from_frame(d[UNIT + ['clf', 'k']])).to_numpy()
    return d.dropna(subset=['d'])


def risk_by_country(R, cond, methods):
    d = vs_srs(R, cond, methods)
    return d.assign(bad=(d.d < RISK).astype(float)).groupby('target').bad.mean(), d


def log_sd_ratio(R, methods):
    """R2：每国、每 (方法, 分类器, k) 上单元间 SD 的 log(AE/S2)，先在国内平均。"""
    out = {}
    for cond in ('AE', 'S2'):
        d = vs_srs(R, cond, methods)
        out[cond] = d.groupby(['target', 'method', 'clf', 'k']).d.std(ddof=1)
    lr = np.log(out['AE'] / out['S2']).replace([np.inf, -np.inf], np.nan).dropna()
    return lr.groupby('target').mean()


# ------------------------------------------------------------------ 标签当量
def srs_curve(R, target, clf):
    """SRS 学习曲线（k = 0/50/100/200/400，单元平均），PAV 单调化后线性插值。"""
    from sklearn.isotonic import IsotonicRegression
    g = R[(R.target == target) & (R.clf == clf) & R.method.isin(['srs', 'foreign_only'])]
    c = g.groupby('k').ba_ge30.mean().sort_index()
    c = c[c.index.isin([0, 50, 100, 200, 400])]
    yk = IsotonicRegression().fit_transform(c.index.to_numpy(float), c.to_numpy())
    return c.index.to_numpy(float), yk


def label_equiv(ks, f, k, delta):
    """Δ 在预算 k 处相当于多少个随机标签：f⁻¹(f(k) + Δ) − k。超出曲线范围时返回 ±inf。"""
    fk = np.interp(k, ks, f)
    tgt = fk + delta
    if tgt > f.max():
        return np.inf
    if tgt < f.min():
        return -np.inf
    # f 单调不减；平台段取最小的 k
    j = np.searchsorted(f, tgt, side='left')
    if j == 0:
        return ks[0] - k
    x0, x1, y0, y1 = ks[j - 1], ks[j], f[j - 1], f[j]
    kk = x1 if y1 == y0 else x0 + (tgt - y0) * (x1 - x0) / (y1 - y0)
    return kk - k


def f_inv(ks, f, y):
    """单调不减分段线性曲线的广义逆 inf{x: f(x) ≥ y}；超出范围返回 ±inf。"""
    if y > f.max():
        return np.inf
    if y < f.min():
        return -np.inf
    j = int(np.searchsorted(f, y, side='left'))
    if j == 0:
        return float(ks[0])
    x0, x1, y0, y1 = ks[j - 1], ks[j], f[j - 1], f[j]  # 此时 y0 < y ≤ y1
    return float(x0 + (y - y0) * (x1 - x0) / (y1 - y0))


def label_equiv_anchored(ks, f, k, delta):
    """rev1 B4：锚定定义 f⁻¹(f(k)+Δ) − f⁻¹(f(k))，Δ=0 时恒为 0（原定义在 PAV 平台处不为 0）。"""
    fk = float(np.interp(k, ks, f))
    return f_inv(ks, f, fk + delta) - f_inv(ks, f, fk)


def label_equiv_detail(R, deltas_by_name, fn):
    rows = []
    for t in sorted(R.target.unique()):
        for clf in ('s2', 'ae'):
            ks, f = srs_curve(R, t, clf)
            if len(ks) < 3:
                continue
            for k in (50, 200):
                for name, dv in deltas_by_name.items():
                    rows.append(dict(target=t, clf=clf, k=k, what=name, delta=dv, labels=fn(ks, f, k, dv)))
    return pd.DataFrame(rows)


def censored_summary(L):
    """每格：中位数（±inf 按删失值参与排序）、+inf/−inf 个数、有限值的四分位。"""
    def one(x):
        x = np.asarray(x, float)
        fin = x[np.isfinite(x)]
        q = np.percentile(fin, [25, 75]) if len(fin) else [np.nan, np.nan]
        return pd.Series(dict(median=np.median(x), n_pinf=int((x == np.inf).sum()), n_ninf=int((x == -np.inf).sum()),
                              n=len(x), q25_finite=q[0], q75_finite=q[1]))
    return L.groupby(['what', 'k']).labels.apply(one).unstack()


def mean_curve_equiv(R, deltas_by_name):
    """敏感性：跨国平均的 SRS 曲线（每种分类器），PAV 后锚定换算。"""
    from sklearn.isotonic import IsotonicRegression
    out = {}
    for clf in ('s2', 'ae'):
        g = R[(R.clf == clf) & R.method.isin(['srs', 'foreign_only']) & R.k.isin([0, 50, 100, 200, 400])]
        c = g.groupby(['target', 'k']).ba_ge30.mean().groupby('k').mean().sort_index()
        ks = c.index.to_numpy(float)
        f = IsotonicRegression().fit_transform(ks, c.to_numpy())
        for k in (50, 200):
            for name, dv in deltas_by_name.items():
                out[(clf, k, name)] = label_equiv_anchored(ks, f, k, dv)
    return pd.Series(out).unstack(1)


def label_equiv_table(R, deltas_by_name):
    rows = []
    for t in sorted(R.target.unique()):
        for clf in ('s2', 'ae'):
            ks, f = srs_curve(R, t, clf)
            if len(ks) < 3:
                continue
            for k in (50, 200):
                for name, dv in deltas_by_name.items():
                    rows.append(dict(target=t, clf=clf, k=k, what=name, delta=dv, labels=label_equiv(ks, f, k, dv)))
    L = pd.DataFrame(rows)
    return L.groupby(['what', 'k']).labels.median().unstack('k') if len(L) else L


# ------------------------------------------------------------------ rev1 B3：补充报告的工具
def heterogeneity(d):
    """国家异质性：单元值的单因素 F 检验（国家为因素），以及把国家内方差当已知的 Cochran Q（偏激进）。"""
    u = unit_level(d).reset_index()
    groups = [g.d.to_numpy() for _, g in u.groupby('target')]
    F, pF = stats.f_oneway(*groups)
    y = np.array([g.mean() for g in groups])
    v = np.array([g.var(ddof=1) / len(g) for g in groups])
    w = 1 / v
    Q = float((w * (y - (w * y).sum() / w.sum()) ** 2).sum())
    k = len(y)
    return dict(F=F, df1=k - 1, df2=len(u) - k, pF=pF, Q=Q, pQ=stats.chi2.sf(Q, k - 1), I2=max(0.0, (Q - (k - 1)) / Q))


def loo(v):
    """留一国：均值范围与各自的判定。"""
    out = []
    for c in v.index:
        s = t_summary(v.drop(c))
        out.append(dict(drop=c, mean=s['mean'], lo90=s['lo90'], hi90=s['hi90'], verdict=verdict(s)))
    return pd.DataFrame(out)


def maxT_signflip(V):
    """V：国家 × 对比 的国家值矩阵。按国家做精确符号翻转（2^n），给出单步 maxT 调整后的双侧 p。"""
    import itertools
    X = V.to_numpy(float)
    n = X.shape[0]

    def tstat(A):
        return A.mean(0) / (A.std(0, ddof=1) / np.sqrt(n))
    t0 = np.abs(tstat(X))
    mx = np.array([np.abs(tstat(X * np.array(sg)[:, None])).max() for sg in itertools.product([1, -1], repeat=n)])
    return pd.Series([(mx >= t - 1e-12).mean() for t in t0], index=V.columns)


def holm(p):
    p = pd.Series(p)
    o = p.sort_values()
    m = len(p)
    vals = np.maximum.accumulate([(m - i) * pv for i, pv in enumerate(o.to_numpy())])
    return pd.Series(np.minimum(vals, 1), index=o.index).reindex(p.index)


def friedman(R, cond_a, cond_b, methods, metric='ba_ge30', reps=(0,)):
    """方法间差异：国家 × 方法 的国家值矩阵上的 Friedman 检验。"""
    M = pd.concat({m: country_values(deltas(R, cond_a, cond_b, [m], metric, reps=list(reps))) for m in methods}, axis=1)
    M = M.dropna()
    return stats.friedmanchisquare(*[M[m] for m in methods]).pvalue, M


def guidance_quality(R):
    """ORACLE、AE、S2 指导的预测质量（重复 0 的候选池）：准确率、平衡精度、NLL。事后描述量。"""
    import guidance_G as G
    D = G.GData()
    rows = []
    for (t, s), _ in R[R.cond == 'ORACLE'].groupby(['target', 'split']):
        S = dict(np.load(os.path.join(OUT, 'stage0', f'{t}_s{s}.npz')))
        pp = G.pool_positions(S, s, 0)
        y = D.y[S['allT'][pp]]
        O = np.load(os.path.join(OUT, 'oracle', f'{t}_s{s}.npz'))
        for name, P in (('ORACLE', O['P_T']), ('AE', S['P_T_ae'][pp]), ('S2', S['P_T_s2'][pp])):
            P = np.asarray(P, float)
            ph = P.argmax(1)
            rec = [(ph[y == c] == c).mean() for c in np.unique(y) if (y == c).sum() >= MIN_N_POOL]
            rows.append(dict(target=t, split=s, guide=name, acc=100 * (ph == y).mean(), ba=100 * np.mean(rec),
                             nll=-np.log(np.clip(P[np.arange(len(y)), y], 1e-300, None)).mean()))
    return pd.DataFrame(rows).groupby('guide')[['acc', 'ba', 'nll']].mean()


MIN_N_POOL = 30


# ------------------------------------------------------------------ 主程序
def fmt(s, unit=' pp'):
    return (f'{s["mean"]:+.3f}{unit}，95% CI [{s["lo95"]:+.3f}, {s["hi95"]:+.3f}]，90% CI [{s["lo90"]:+.3f}, '
            f'{s["hi90"]:+.3f}]，p={s["p"]:.3f}，>0 的国家 {s["n_pos"]}/{s["n"]}')


def complete_cond(R, cond, n=22):
    return R[R.cond == cond].groupby(UNIT).ngroups == n


def main():
    tag = '_NOISE' if NOISE else ''
    sys.stdout = Tee(os.path.join(OUT, f'report_G{tag}.txt'))
    if NOISE:
        print('!!! 噪声模式：指标为随机数，结果无意义，只用于测试流程 !!!')
    R = load_rows()
    n_units = R[R.cond == 'AE'].groupby(UNIT).ngroups
    complete = n_units == 44 and set(R.target) == set(COUNTRIES_G)
    if not complete:
        msg = f'单元不全（AE 条件 {n_units}/44 个单元，{R.target.nunique()} 国）'
        if not PARTIAL:
            raise SystemExit(msg + '；正式分析要求层级 1 全部完成（流程测试请加 --allow-partial）')
        print(f'!!! {msg}：只用于流程测试，不作结论 !!!')
    print(f'行数 {len(R)}；国家 {R.target.nunique()}；条件 {sorted(R.cond.unique())}')
    print('冻结方案 ANALYSIS_PLAN_guidance.md（ff55edf89528dd07）；偏离披露与事后补充见 ANALYSIS_PLAN_guidance_rev1.md。'
          '\n标"事后"的各节均为看过层级 1+2 结果后追加，只作探索性描述。')
    summ = []

    # ---------- H1：θ_E1
    print('\n' + '=' * 20 + ' H1：θ_E1（只用几何的通路，ba_ge30，P0；4 方法 × 2 分类器 × 2 档 k × 4 单元）' + '=' * 20)
    dE1 = deltas(R, 'AE', 'S2', E1)
    vE1 = country_values(dE1)
    sE1 = t_summary(vE1)
    q = stats.t.ppf(0.95, sE1['n'] - 1)
    p_tost = max(stats.t.sf((sE1['mean'] + SESOI) / sE1['se'], sE1['n'] - 1),
                 stats.t.sf((SESOI - sE1['mean']) / sE1['se'], sE1['n'] - 1))
    print('θ_E1 =', fmt(sE1))
    print(f'判定（方案第 8 节）：{verdict(sE1)}；TOST p = {p_tost:.4f}；能通过 TOST 的最小界值 = '
          f'{max(abs(sE1["mean"] - q * sE1["se"]), abs(sE1["mean"] + q * sE1["se"])):.3f} pp')
    print('各国：', vE1.round(3).to_dict())
    summ.append(dict(what='theta_E1', **sE1))

    # ---------- R0：80% 预测区间
    pi = prediction_interval(vE1.to_numpy(), np.full(len(vE1), within_var(dE1)))
    print(f'R0：μ̂={pi["mu"]:+.3f}，τ̂={pi["tau"]:.3f}，SE_HKSJ={pi["se_hksj"]:.3f}，'
          f'80% 预测区间 [{pi["pi_lo"]:+.3f}, {pi["pi_hi"]:+.3f}] pp（新国家中 AE−S2 的分布；国家内方差取合并值）')
    summ.append(dict(what='R0_PI80', mean=pi['mu'], lo95=pi['pi_lo'], hi95=pi['pi_hi'], se=pi['se_hksj']))

    # ---------- 门控 H2 → H3 → H4
    print('\n' + '=' * 20 + ' 门控（H1 判定方向后依次检验）' + '=' * 20)
    sOA = t_summary(country_values(deltas(R, 'AE', 'S2', E1, 'oa')))
    comp = {m: FAMILIES[m] for m in ('geom', 'pred', 'hyb')}
    path_vals = {}
    for name, ms in comp.items():
        path_vals[name] = country_values(deltas(R, 'AE', 'S2', ms, reps=[0] if name != 'geom' else None))
    vE2 = pd.concat(path_vals, axis=1).mean(1)
    sE2 = t_summary(vE2.dropna())
    rAE, _ = risk_by_country(R, 'AE', E1)
    rS2, _ = risk_by_country(R, 'S2', E1)
    dr = (rAE - rS2).dropna()
    sR1 = t_summary(dr)
    w_p = stats.wilcoxon(dr).pvalue if (dr != 0).any() else np.nan
    w_pn = stats.wilcoxon(dr, method='approx').pvalue if (dr != 0).any() else np.nan
    sign = 1 if sE1['lo95'] > 0 else (-1 if sE1['hi95'] < 0 else 0)
    print(f'H2（OA 非劣效，界值 {OA_NI}）：Δ_OA =', fmt(sOA))
    print('H3（θ_E2 = 三条通路等权）：', fmt(sE2))
    print(f'H4（R1：方法 − SRS < {RISK} pp 的单元比例，AE − S2）：', fmt(sR1, ''),
          f'；Wilcoxon p={w_p:.3f}（scipy 默认；|差|有并列），正态近似 p={w_pn:.3f}')
    print(f'   R1 比例均值：AE {rAE.mean():.3f}，S2 {rS2.mean():.3f}')
    if sign == 0:
        print('H1 未判出方向 → 门控不继续（H2–H4 只作描述）')
    else:
        h2 = (sOA['lo95'] > OA_NI) if sign > 0 else (sOA['hi95'] < -OA_NI)
        h3 = (sE2['lo95'] > 0) if sign > 0 else (sE2['hi95'] < 0)
        h4 = (sR1['hi95'] < 0) if sign > 0 else (sR1['lo95'] > 0)
        chain = []
        for name, ok in (('H2', h2), ('H3', h3), ('H4', h4)):
            chain.append(f'{name} {"成立" if ok else "不成立"}')
            if not ok:
                break
        print('门控：', ' → '.join(chain))
        if sR1['lo95'] > 0 and sign > 0:
            print('R1 显著更差 → 表述为"平均更好但不更可靠"')
    summ += [dict(what='H2_oa', **sOA), dict(what='H3_E2', **sE2), dict(what='H4_R1', **sR1)]

    # ---------- 次要
    print('\n' + '=' * 20 + ' 次要结果（预注册，不校正；族错误率校正见下文事后部分）' + '=' * 20)
    g = dE1.pivot_table(index=UNIT + ['method', 'k'], columns='clf', values='d').dropna()
    g['gamma'] = (g['ae'] - g['s2']) / 2
    vG = g.reset_index().groupby(['target', 'method', 'k']).gamma.mean().groupby('target').mean()
    sG = t_summary(vG)
    print('γ（同源交互）：', fmt(sG))
    sec = {'γ': vG}
    for clf in ('s2', 'ae'):
        v = country_values(dE1[dE1.clf == clf])
        sec[f'E1|clf={clf}'] = v
        print(f'  Δ_E1 | 分类器 {clf}：', fmt(t_summary(v)))
    for k in (50, 200):
        v = country_values(dE1[dE1.k == k])
        sec[f'E1|k={k}'] = v
        print(f'  Δ_E1 | k={k}：', fmt(t_summary(v)))
    for name, v in path_vals.items():
        print(f'  通路 {name}：', fmt(t_summary(v.dropna())))
        summ.append(dict(what=f'path_{name}', **t_summary(v.dropna())))
        if name != 'geom':
            sec[f'通路 {name}'] = v
    summ.append(dict(what='gamma', **sG))
    print('\n逐方法（部分合并收缩；只引用收缩值，不下显著性结论）：')
    allm = sum(FAMILIES.values(), [])
    print(shrink_methods(deltas(R, 'AE', 'S2', allm)).round(3).to_string(index=False))

    # 机制：P2 拆分、E_far（rev1 B5 的术语）
    for lab, met in (('Δ(P2) 先验校正后的剩余部分', 'ba_P2'), ('Δ(P1)', 'ba_P1'), ('E_far', 'ba_far')):
        v = country_values(deltas(R, 'AE', 'S2', E1, met))
        sec[lab.split(' ')[0]] = v
        print(f'  {lab}：', fmt(t_summary(v)))
    dd = deltas(R, 'AE', 'S2', E1, 'ba_ge30').set_index(KEY).d - deltas(R, 'AE', 'S2', E1, 'ba_P2').set_index(KEY).d
    print('  Δ(P0) − Δ(P2) 全局先验校正可去除的部分：', fmt(t_summary(country_values(dd.rename('d').reset_index()))))
    print('  注：E_far 的类集合取整个测试集中 ≥30 个地块的类（rev1 A5）。')

    # XY 与 S2XY（层级 2，重复 0，lpm2_nogeo 与 kcenter）
    if {'XY', 'S2XY'} <= set(R.cond):
        ms = ['lpm2_nogeo', 'kcenter']
        print('\nXY 与 S2XY（lpm2_nogeo、kcenter，重复 0 的 22 个单元）：')
        for a, b in (('AE', 'S2'), ('AE', 'S2XY'), ('AE', 'XY'), ('S2XY', 'S2')):
            v = country_values(deltas(R, a, b, ms, reps=[0]))
            s = t_summary(v)
            print(f'  {a} − {b}：', fmt(s))
            summ.append(dict(what=f'{a}_minus_{b}', **s))
            if (a, b) != ('AE', 'S2'):
                sec[f'{a}−{b}'] = v
    # ORACLE 提升空间（预注册，原实现）
    three = ['lpm2_nogeo', 'typiclust', 'margin']
    if 'ORACLE' in set(R.cond):
        vH = country_values(deltas(R, 'ORACLE', 'S2', three, reps=[0]))
        sH = t_summary(vH)
        sA = t_summary(country_values(deltas(R, 'AE', 'S2', three, reps=[0])))
        sec['H_ORACLE'] = vH
        print('\nORACLE 提升空间 H = θ_ORACLE − θ_S2（lpm2_nogeo、typiclust、margin；预注册，原实现）：', fmt(sH))
        print('  同一方法上 θ_AE − θ_S2：', fmt(sA))
        print(f'  ρ = {sA["mean"] / sH["mean"]:.2f}' if sH['lo95'] > 0 else '  H 的 95% CI 下界 ≤ 0 → 不报告 ρ')
        summ += [dict(what='oracle_H', **sH), dict(what='oracle_AE_same_methods', **sA)]

    # ---------- 可靠性描述
    print('\n' + '=' * 20 + ' 可靠性（描述）' + '=' * 20)
    lr = log_sd_ratio(R, E1)
    print('R2 log(SD_AE / SD_S2)（SD 为同一国家、方法、分类器、k 内各单元"方法 − SRS"的标准差）：', fmt(t_summary(lr), ''))
    for cond in ('AE', 'S2'):
        d = vs_srs(R, cond, E1)
        print(f'R3 {cond}：方法 − SRS 的均值 {d.d.mean():+.3f}，P(方法 < SRS) = {(d.d < 0).mean():.3f}，'
              f'P(< {RISK}) = {(d.d < RISK).mean():.3f}；国家层面：', fmt(t_summary(country_values(d))))

    # ---------- 标签当量
    names = {'θ_E1': sE1['mean'], 'L95': sE1['lo95'], 'U95': sE1['hi95'], '+0.3': SESOI, '−0.3': -SESOI}
    print('\n' + '=' * 20 + ' 标签当量（预注册定义 f⁻¹(f(k)+Δ)−k；各国 × 分类器的中位数；inf = 超出曲线范围）' + '=' * 20)
    L = label_equiv_table(R, names)
    print(L.to_string() if len(L) else '（无 SRS 学习曲线）')
    print('\n[事后，rev1 B4] 锚定定义 f⁻¹(f(k)+Δ) − f⁻¹(f(k))，删失汇总（22 条曲线 = 11 国 × 2 分类器）：')
    La = label_equiv_detail(R, names, label_equiv_anchored)
    print(censored_summary(La).round(1).to_string())
    print('\n[事后] 跨国平均 SRS 曲线（PAV）上的锚定换算：')
    print(mean_curve_equiv(R, names).round(1).to_string())
    ks_gain = R[R.method.isin(['srs', 'foreign_only'])].groupby(['target', 'clf', 'k']).ba_ge30.mean().unstack('k')
    print(f'\nSRS 学习曲线增益（22 条曲线）：k=0→50 中位数 {(ks_gain[50] - ks_gain[0]).median():+.2f} pp；'
          f'50→200 {(ks_gain[200] - ks_gain[50]).median():+.2f}；50→400 中位数 {(ks_gain[400] - ks_gain[50]).median():+.2f}，'
          f'均值 {(ks_gain[400] - ks_gain[50]).mean():+.2f} pp；不单调的曲线 '
          f'{int((ks_gain[[0, 50, 100, 200, 400]].diff(axis=1).iloc[:, 1:] < 0).any(axis=1).sum())}/{len(ks_gain)}')

    # ---------- 敏感性
    print('\n' + '=' * 20 + ' 敏感性' + '=' * 20)
    for lab, drop in (('去掉检验 2 的 4 国', CHECK2), ('去掉 RO', ['RO']), ('去掉开发国', DEV)):
        v = vE1.drop([c for c in drop if c in vE1.index])
        print(f'  {lab}（{len(v)} 国）：', fmt(t_summary(v)))
    for it in (200, 250):
        s = t_summary(country_values(deltas(R, 'AE', 'S2', E1, f'ba_i{it}')))
        print(f'  第 {it} 轮：', fmt(s))
        if np.sign(s['mean']) != np.sign(sE1['mean']) and it == 250:
            print('  !!! 250 与 300 轮符号相反 → 结论标注为"依赖训练容量"')
    print(f'  wild cluster bootstrap（Webb）p = {webb_wcb(vE1.to_numpy()):.3f}')
    u = unit_level(dE1).groupby('target')
    print(f'  平衡混合模型（单元值 ~ 1 + 国家随机截距，REML；各国单元数相同）：μ̂={vE1.mean():+.3f}，SE={sE1["se"]:.3f}（与国家层面 t 检验相同）')
    vc = (u.var(ddof=1) / u.size()).reindex(vE1.index)
    if vc.notna().all():
        pm = prediction_interval(vE1.to_numpy(), vc.to_numpy())
        print(f'  逆方差加权随机效应元分析（国家内方差各国自取；PT、RO、SE 权重较低）：μ̂={pm["mu"]:+.3f}，'
              f'SE_HKSJ={pm["se_hksj"]:.3f}，τ̂={pm["tau"]:.3f}，80% PI [{pm["pi_lo"]:+.3f}, {pm["pi_hi"]:+.3f}]')

    # ================================================================ 事后补充（rev1 B3）
    print('\n' + '#' * 20 + ' 事后补充（rev1 B3；看过结果后追加，只作探索性描述）' + '#' * 20)
    h = heterogeneity(dE1)
    print(f'国家异质性（θ_E1）：单因素 F({h["df1"]},{h["df2"]}) = {h["F"]:.2f}，p = {h["pF"]:.3f}；'
          f'Cochran Q = {h["Q"]:.1f}，p = {h["pQ"]:.3f}，I² = {h["I2"]:.2f}（把国家内方差当已知，偏激进）')
    lo = loo(vE1)
    print(f'留一国：均值 {lo["mean"].min():+.3f} 至 {lo["mean"].max():+.3f}（去 {lo.loc[lo["mean"].idxmin(), "drop"]} / '
          f'去 {lo.loc[lo["mean"].idxmax(), "drop"]}）；判定 {lo.verdict.str.split("：").str[0].value_counts().to_dict()}')
    for k in (50, 200):
        s = t_summary(country_values(dE1[dE1.k == k]))
        print(f'按 k 的 TOST（非预注册）k={k}：90% CI [{s["lo90"]:+.3f}, {s["hi90"]:+.3f}] → {verdict(s)}')
    s = t_summary(country_values(dE1[(dE1.k == 200) & (dE1.target != 'PT')]))
    print(f'  k=200 去掉 PT：{s["mean"]:+.3f}，90% CI [{s["lo90"]:+.3f}, {s["hi90"]:+.3f}] → {verdict(s)}')
    print('逐方法 TOST（E1，只作描述）：')
    for m in E1:
        s = t_summary(country_values(dE1[dE1.method == m]))
        print(f'  {m:<11}{s["mean"]:+.3f}，90% CI [{s["lo90"]:+.3f}, {s["hi90"]:+.3f}] → {verdict(s)}')
    print('其他指标上的 θ_E1：')
    for met in ('oa', 'f1_ge30', 'ba_P1', 'ba_P2', 'ba_far', 'oa_P2'):
        s = t_summary(country_values(deltas(R, 'AE', 'S2', E1, met)))
        print(f'  {met:<8}{s["mean"]:+.3f}，90% CI [{s["lo90"]:+.3f}, {s["hi90"]:+.3f}] → {verdict(s)}')

    # 次要结果的族错误率
    V = pd.concat(sec, axis=1).dropna()
    praw = pd.Series({c: t_summary(V[c])['p'] for c in V.columns})
    adj = pd.DataFrame(dict(p_raw=praw, holm=holm(praw), maxT_signflip=maxT_signflip(V)))
    print(f'\n次要结果的族错误率（{len(V.columns)} 项；Holm 与精确符号翻转 maxT）：')
    print(adj.round(3).to_string())

    # 预测通路的多指标表
    print('\n预测通路（margin、predstrat_srs；重复 0 的 22 个单元）的多指标表：')
    for met in ('ba_ge30', 'oa', 'f1_ge30', 'ba_P1', 'ba_P2', 'oa_P2', 'ba_far'):
        print(f'  {met:<8}', fmt(t_summary(country_values(deltas(R, 'AE', 'S2', FAMILIES['pred'], met, reps=[0])))))
    lp = loo(path_vals['pred'].dropna())
    print(f'  留一国：均值 {lp["mean"].min():+.3f} 至 {lp["mean"].max():+.3f}；'
          f'稳健位置：中位数 {path_vals["pred"].median():+.3f}，20% 截尾均值 {stats.trim_mean(path_vals["pred"], 0.2):+.3f}')
    for clf in ('s2', 'ae'):
        for k in (50, 200):
            dp = deltas(R, 'AE', 'S2', FAMILIES['pred'], reps=[0])
            s = t_summary(country_values(dp[(dp.clf == clf) & (dp.k == k)]))
            print(f'  clf={clf} k={k}：{s["mean"]:+.3f} [{s["lo95"]:+.3f}, {s["hi95"]:+.3f}]，{s["n_pos"]}/11')

    # ORACLE 逐方法与指导质量
    if 'ORACLE' in set(R.cond):
        print('\nORACLE 逐方法（重复 0；ORACLE − S2 与 ORACLE − SRS）：')
        for m in three:
            parts = []
            for met in ('ba_ge30', 'oa', 'ba_P2'):
                s = t_summary(country_values(deltas(R, 'ORACLE', 'S2', [m], met, reps=[0])))
                parts.append(f'{met} {s["mean"]:+.3f} [{s["lo95"]:+.3f}, {s["hi95"]:+.3f}] {s["n_pos"]}/11')
            dv = vs_srs(R[R.rep == 0], 'ORACLE', [m])
            s = t_summary(country_values(dv))
            print(f'  {m:<11}' + '；'.join(parts) + f'；相对 SRS {s["mean"]:+.3f}')
        for met in ('ba_ge30', 'oa'):
            pf, _ = friedman(R, 'ORACLE', 'S2', three, met)
            print(f'  方法间差异 Friedman（{met}）p = {pf:.4f}')
        if not NOISE:
            print('指导质量（重复 0 的候选池）：')
            print(guidance_quality(R).round(3).to_string())
    if 'ORACLE2' in set(R.cond):
        two = ['lpm2_nogeo', 'typiclust']
        done = complete_cond(R, 'ORACLE2')
        print(f'\n[rev1 B1] ORACLE2（修正几何；lpm2_nogeo、typiclust）{"" if done else "（单元未齐，只作流程检查）"}：')
        v2 = country_values(deltas(R, 'ORACLE2', 'S2', two, reps=[0]))
        v1 = country_values(deltas(R, 'ORACLE', 'S2', two, reps=[0]))
        vm = country_values(deltas(R, 'ORACLE', 'S2', ['margin'], reps=[0]))
        print('  ORACLE2 − S2（两个几何方法）：', fmt(t_summary(v2)))
        print('  ORACLE  − S2（同两方法，原实现）：', fmt(t_summary(v1)))
        for m in two:
            print(f'    {m}：ORACLE2 − S2', fmt(t_summary(country_values(deltas(R, 'ORACLE2', 'S2', [m], reps=[0])))))
        dH = pd.concat([deltas(R, 'ORACLE2', 'S2', two, reps=[0]), deltas(R, 'ORACLE', 'S2', ['margin'], reps=[0])])
        sH2 = t_summary(country_values(dH))
        sA = t_summary(country_values(deltas(R, 'AE', 'S2', three, reps=[0])))
        print('  H_ORACLE2（lpm2_nogeo、typiclust 用修正几何，margin 不变）：', fmt(sH2))
        print(f'  ρ = {sA["mean"] / sH2["mean"]:.2f}' if sH2['lo95'] > 0 else '  H_ORACLE2 的 95% CI 下界 ≤ 0 → 不报告 ρ')

    # 层级 3：RAWZ 与 NATIVE
    for cond_a, cond_b, ms, lab in (('RAWZ_AE', 'RAWZ_S2', E1, 'RAWZ（阶段 A 的原始 z + PCA）'),
                                    ('NATIVE_AE', 'NATIVE_S2', ['lpm2_nogeo', 'typiclust'], 'NATIVE（原生几何）')):
        if cond_a in set(R.cond):
            done = complete_cond(R, cond_a) and complete_cond(R, cond_b)
            print(f'\n[rev1 B2] 层级 3 {lab}，k=50，重复 0{"" if done else "（单元未齐，只作流程检查）"}：')
            print(f'  {cond_a} − {cond_b}：', fmt(t_summary(country_values(deltas(R, cond_a, cond_b, ms, reps=[0], ks=(50,))))))
            print('  同一子集上的主分析 AE − S2：',
                  fmt(t_summary(country_values(deltas(R, 'AE', 'S2', ms, reps=[0], ks=(50,))))))
            print(f'  {cond_a} − AE（预处理对 AE 自身的影响）：',
                  fmt(t_summary(country_values(deltas(R, cond_a, 'AE', ms, reps=[0], ks=(50,))))))
            print(f'  {cond_b} − S2：', fmt(t_summary(country_values(deltas(R, cond_b, 'S2', ms, reps=[0], ks=(50,))))))

    # 事后描述（第二轮审计要求补出处）
    print('\n[事后] 各方法相对 SRS（国家层面；E1 用 44 个单元，其余方法用重复 0）：')
    for m in sum(FAMILIES.values(), []):
        parts = []
        for cond in ('AE', 'S2'):
            dv = vs_srs(R if m in E1 else R[R.rep == 0], cond, [m])
            s = t_summary(country_values(dv))
            parts.append(f'{cond} {s["mean"]:+.2f} [{s["lo95"]:+.2f}, {s["hi95"]:+.2f}] {s["n_pos"]}/11')
        print(f'  {m:<14}' + '；'.join(parts))
    # 同一（单元, 分类器, k）格内：4 个几何方法 × 2 种指导的双因素平方和分解
    sub = R[R.cond.isin(['AE', 'S2']) & R.method.isin(E1) & R.k.isin([50, 200]) & R.clf.isin(['s2', 'ae'])]
    ssm = ssg = sst = 0.0
    for _, c in sub.groupby(UNIT + ['clf', 'k']):
        x = c.pivot_table(index='method', columns='cond', values='ba_ge30')
        mu = x.to_numpy().mean()
        ssm += 2 * ((x.mean(1) - mu) ** 2).sum()
        ssg += len(x) * ((x.mean(0) - mu) ** 2).sum()
        sst += ((x.to_numpy() - mu) ** 2).sum()
    print(f'[事后] 同一（单元, 分类器, k）格内的方差份额：方法 {ssm / sst:.1%}，指导 {ssg / sst:.1%}，交互 {1 - (ssm + ssg) / sst:.1%}')
    if 'ORACLE2' in set(R.cond):
        for m in ('lpm2_nogeo', 'typiclust'):
            print(f'[事后] ORACLE2 − ORACLE（{m}）：',
                  fmt(t_summary(country_values(deltas(R, 'ORACLE2', 'ORACLE', [m], reps=[0])))))
    qp = os.path.join(OUT, 'guidance_quality_G.csv')
    if os.path.exists(qp) and not NOISE:
        Q = pd.read_csv(qp).pivot_table(index='target', columns='feat', values='knn10_ba')
        gap = (Q['ae'] - Q['s2']).reindex(vE1.index)
        r, pr = stats.pearsonr(gap, vE1)
        rho, prho = stats.spearmanr(gap, vE1)
        z, se = np.arctanh(r), 1 / np.sqrt(len(gap) - 3)
        loo_r = {c: stats.pearsonr(gap.drop(c), vE1.drop(c))[0] for c in gap.index}
        keep = ~gap.index.isin(['SE', 'PT'])
        print(f'[事后，未登记] 10 近邻 BA 的 AE 优势（{gap.min():+.1f} 至 {gap.max():+.1f} pp，11/11 国为正）与 θ_E1 的相关：'
              f'Pearson r={r:+.2f}（95% CI [{np.tanh(z - 1.96 * se):+.2f}, {np.tanh(z + 1.96 * se):+.2f}]，p={pr:.3f}），'
              f'Spearman ρ={rho:+.2f}（p={prho:.3f}）；留一国 r {min(loo_r.values()):+.2f}（去 {min(loo_r, key=loo_r.get)}）'
              f'至 {max(loo_r.values()):+.2f}（去 {max(loo_r, key=loo_r.get)}）；去 PT {loo_r["PT"]:+.2f}；'
              f'去 SE 与 PT {stats.pearsonr(gap[keep], vE1[keep])[0]:+.2f}')
    se_ = t_summary(country_values(deltas(R, 'AE', 'S2', FAMILIES['pred'], 'ba_far', reps=[0])))
    print(f'[事后，四位小数] 预测通路 E_far：{se_["mean"]:+.4f}，95% CI [{se_["lo95"]:+.4f}, {se_["hi95"]:+.4f}]，p={se_["p"]:.4f}')

    pd.DataFrame(summ).to_csv(os.path.join(OUT, f'summary_G{tag}.csv'), index=False)


if __name__ == '__main__':
    main()
