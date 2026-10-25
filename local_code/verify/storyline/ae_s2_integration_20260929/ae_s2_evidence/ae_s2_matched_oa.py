"""
只读描述性重算（不训练）：
(1) 同 OA 下比较 AE 与 S2 的零标签外国模型（第 0 步 B2 的 Z0 τ 曲线，LightGBM，44 单元）：
    Δ_feat 在 τ=0 是固定工作点上的比较；这里在两特征都可达的 OA 窗口内比较 BA 前沿。
(2) 跨境损失（检验 3 本国模型 − L0）的 AE − S2 配对差，由 NUMBERS.csv 的逐国值重算。
运行：/Users/perng_mini/AlphaCrops_local/.venv/bin/python -B ae_s2_matched_oa.py
"""
import json
import os

os.environ.setdefault('OMP_NUM_THREADS', '2')
import numpy as np
import pandas as pd
from scipy import stats

OUT = os.path.dirname(os.path.abspath(__file__))
L = '/Users/perng_mini/AlphaCrops_local'
C = ['AT', 'BE', 'DE', 'EE', 'LT', 'LV', 'PT', 'RO', 'SE', 'SI', 'SK']
C9 = [c for c in C if c not in ('RO', 'PT')]
EPS = 1e-9
log = []


def P(*a):
    s = ' '.join(str(x) for x in a)
    print(s)
    log.append(s)


def tsum(v):
    v = pd.Series(v).dropna()
    n = len(v); m = v.mean(); se = v.std(ddof=1) / np.sqrt(n)
    h = stats.t.ppf(0.975, n - 1) * se
    p = 2 * stats.t.sf(abs(m / se), n - 1) if se > 0 else np.nan
    return dict(n=n, mean=m, lo95=m - h, hi95=m + h, p=p, npos=int((v > 0).sum()))


def fmt(s):
    return f"{s['mean']:+.2f} [{s['lo95']:+.2f}, {s['hi95']:+.2f}]，p={s['p']:.3f}，>0 {s['npos']}/{s['n']}"


def ba_at(oa, ba, grid):
    """单条 τ 曲线上 OA = o 的所有交点取 BA 最大（线性插值）；不可达为 NaN。与 step0B.envelope 单曲线情形相同。"""
    o0, o1, b0, b1 = oa[:-1], oa[1:], ba[:-1], ba[1:]
    lo, hi = np.minimum(o0, o1) - EPS, np.maximum(o0, o1) + EPS
    d = o1 - o0
    flat = np.abs(d) < EPS
    inside = (grid[:, None] >= lo[None]) & (grid[:, None] <= hi[None])
    lm = np.clip((grid[:, None] - o0[None]) / np.where(flat, 1.0, d)[None], 0.0, 1.0)
    lm = np.where(flat[None], (b1 > b0).astype(float)[None], lm)
    val = np.where(inside, b0[None] + lm * (b1 - b0)[None], -np.inf)
    best = val.max(1)
    best[~np.isfinite(best)] = np.nan
    return best


cur = pd.read_csv(f'{L}/verify/step0/partB2/curves_B2.csv')
z = cur[cur.arm == 'Z0'].sort_values(['unit', 'clf', 'tau'])
rows = []
for unit, g in z.groupby('unit'):
    a = g[g.clf == 'ae']; s = g[g.clf == 's2']
    t0a = a[np.isclose(a.tau, 0)].iloc[0]; t0s = s[np.isclose(s.tau, 0)].iloc[0]
    oa_star_min = min(t0a.oa, t0s.oa)
    res = dict(unit=unit, target=g.target.iloc[0], split=g.split.iloc[0], rep=g.rep.iloc[0],
               oa0_ae=t0a.oa, oa0_s2=t0s.oa, ba0_ae=t0a.ba_ge30, ba0_s2=t0s.ba_ge30)
    for name, lo_off, anchor in (('W4_min', 4, oa_star_min), ('W8_min', 8, oa_star_min),
                                 ('W4_s2', 4, t0s.oa), ('W4_ae', 4, t0a.oa)):
        grid = np.round(np.arange(anchor - lo_off, anchor + 1e-9, 0.1), 6)
        ba_a = ba_at(a.oa.to_numpy(), a.ba_ge30.to_numpy(), grid)
        ba_s = ba_at(s.oa.to_numpy(), s.ba_ge30.to_numpy(), grid)
        ok = np.isfinite(ba_a) & np.isfinite(ba_s)
        cov = ok.mean()
        res[f'{name}_cov'] = cov
        res[f'{name}_d'] = float(np.mean(ba_a[ok] - ba_s[ok])) if cov >= 0.5 else np.nan
        res[f'{name}_ae_dominates'] = bool((ba_a[ok] >= ba_s[ok]).all()) if cov >= 0.5 else np.nan
        res[f'{name}_s2_dominates'] = bool((ba_s[ok] >= ba_a[ok]).all()) if cov >= 0.5 else np.nan
    rows.append(res)
U = pd.DataFrame(rows)
U['d_tau0_ba'] = U.ba0_ae - U.ba0_s2
U['d_tau0_oa'] = U.oa0_ae - U.oa0_s2
U.to_csv(f'{OUT}/matched_oa_units.csv', index=False)

P('# 同 OA 下 AE 与 S2 零标签外国模型（Z0，LightGBM，温度校准 P_T 底座，τ ∈ [−2, 2] 步长 0.05）的 BA 比较')
P('窗口：W4_min = [min(OA★_AE, OA★_S2) − 4, min(...)]；W8_min 为 8 pp 宽；W4_s2 / W4_ae 以单一特征的 τ=0 OA 为锚。覆盖 <50% 记缺失。')
Cn = U.groupby('target')[['d_tau0_ba', 'd_tau0_oa', 'W4_min_d', 'W8_min_d', 'W4_s2_d', 'W4_ae_d',
                          'W4_min_cov', 'W4_s2_cov']].mean().loc[C]
P(Cn.round(2).to_string())
for col in ['d_tau0_ba', 'd_tau0_oa', 'W4_min_d', 'W8_min_d', 'W4_s2_d', 'W4_ae_d']:
    P(f'  {col:10s} 11 国 {fmt(tsum(Cn[col]))} | 9 国 {fmt(tsum(Cn.loc[C9, col]))}')
dom = U.groupby('target')[['W4_min_ae_dominates', 'W4_min_s2_dominates']].mean().loc[C]
P('  单元中 AE 前沿在 W4_min 全窗口不低于 S2 的比例（逐国）：', dom.W4_min_ae_dominates.round(2).to_dict())
P('  单元中 S2 前沿在 W4_min 全窗口不低于 AE 的比例（逐国）：', dom.W4_min_s2_dominates.round(2).to_dict())
P(f"  合计：AE 全窗口占优 {int(U.W4_min_ae_dominates.sum())}/44，S2 全窗口占优 {int(U.W4_min_s2_dominates.sum())}/44")
r = stats.spearmanr(Cn.d_tau0_ba, Cn.W4_min_d)
P(f'  τ=0 的 ΔBA 与同 OA 的 ΔBA 逐国 Spearman ρ={r[0]:+.2f}')

# ---------------------------------------------------------------- 跨境损失 AE − S2（检验 3，NUMBERS.csv 逐国值）
NUM = '/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops/label_expansion/storyline/numbers_table/NUMBERS.csv'
N = pd.read_csv(NUM).set_index('id')
P('\n# 跨境总损失（本国模型 − L0）的 AE − S2（负 = AE 跨境损失更小；检验 3 随机 70/30 无缓冲，LightGBM；由 NUMBERS.csv per_country 重算）')
for setting in ('es', 'c300'):
    for met in ('oa', 'ba_ge30'):
        a = pd.Series(json.loads(N.loc[f'H1.total.{met}.{setting}.ae', 'per_country']))
        s = pd.Series(json.loads(N.loc[f'H1.total.{met}.{setting}.s2', 'per_country']))
        d = (a - s).loc[C]
        P(f'  {setting:4s} {met:7s} AE {a.mean():.2f} vs S2 {s.mean():.2f}；AE−S2 11 国 {fmt(tsum(d))} | 9 国 {fmt(tsum(d.loc[C9]))}')
        P(f'       逐国：{d.round(2).to_dict()}')
# 缓冲格网（G 划分 0）：tgt300 − k0 的差距，AE − S2
ub = pd.read_csv(f'{L}/verify/model_diag/fit/table_upper_bound.csv').pivot(index='tgt', columns='feat')
for met in ('ba', 'oa'):
    d = (ub[(f'gap_{met}', 'ae')] - ub[(f'gap_{met}', 's2')]).loc[C]
    P(f'  G 缓冲格网划分 0：tgt300 − k0 差距 {met}：AE {ub[(f"gap_{met}", "ae")].mean():.2f} vs S2 {ub[(f"gap_{met}", "s2")].mean():.2f}；AE−S2 11 国 {fmt(tsum(d))} | 9 国 {fmt(tsum(d.loc[C9]))}')

with open(f'{OUT}/ae_s2_matched_oa_report.txt', 'w') as f:
    f.write('\n'.join(log) + '\n')
