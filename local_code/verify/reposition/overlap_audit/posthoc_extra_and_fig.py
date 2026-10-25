"""
posthoc_distance.py 的补充：(1) 控制类别构成距离；(2) 每个源国/目标国内部的斜率符号；(3) 作图。
只读；不训练模型。
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '2')
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import binomtest

OUT = os.path.dirname(os.path.abspath(__file__))
RES = '/Users/perng_mini/AlphaCrops_local/results'
x = pd.read_csv(os.path.join(OUT, 'pairwise_with_distance.csv'))
g0 = pd.read_csv(os.path.join(OUT, 'multisource_k0_by_target.csv'))
dd = pd.read_csv(os.path.join(RES, 'domain_distance.csv'))
js = {}
for r in dd.itertuples():
    js[(r.c1, r.c2)] = js[(r.c2, r.c1)] = r.class_js
x['class_js'] = [js[(s, t)] for s, t in zip(x.source, x.target)]


def ols(y, X):
    X = np.column_stack([np.ones(len(y)), X])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return b


out = {}
# (1) 控制类别构成距离（需要标签，只作混杂检查）
b = ols(x.d_ae_s2.to_numpy(), np.column_stack([x.dlat, x.class_js]))
out['d_ae_s2 ~ dlat + class_js'] = dict(dlat_slope=round(b[1], 3), class_js_coef=round(b[2], 3))
b = ols(x.d_ae_s2.to_numpy(), np.column_stack([x.dlat, x.class_js, x.climate_diff]))
out['d_ae_s2 ~ dlat + class_js + climate_diff'] = dict(dlat_slope=round(b[1], 3), class_js_coef=round(b[2], 3),
                                                       climate_diff_coef=round(b[3], 3))
# (2) 每个源国（固定源，10 个目标）/ 每个目标国（固定目标，10 个源）内部的斜率符号
for role in ['source', 'target']:
    sl = {}
    for c, g in x.groupby(role):
        sl[c] = round(ols(g.d_ae_s2.to_numpy(), g.dlat.to_numpy())[1], 3)
    neg = sum(v < 0 for v in sl.values())
    out[f'within_{role}_slopes'] = dict(slopes=sl, n_negative=neg, n=len(sl),
                                        sign_test_p=round(binomtest(neg, len(sl)).pvalue, 4))
json.dump(out, open(os.path.join(OUT, 'posthoc_extra.json'), 'w'), ensure_ascii=False, indent=1, default=int)
print(json.dumps(out, ensure_ascii=False, indent=1, default=int))

# (3) 图
BLUE, ORANGE, GRAY = '#2a78d6', '#eb6834', '#8a8983'
INK, INK2 = '#0b0b0b', '#52514e'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': '#c9c8c2', 'axes.labelcolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.spines.top': False,
                     'axes.spines.right': False, 'font.family': ['Arial Unicode MS', 'DejaVu Sans']})
fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.2))

# a: 各特征 BA 与 |Δlat|
xs = np.linspace(0, 18.5, 50)
for col, c, lab in [('ae', BLUE, 'AE'), ('s2', ORANGE, 'S2')]:
    ax[0].scatter(x.dlat, x[col], s=14, color=c, alpha=.45, lw=0)
    bb = ols(x[col].to_numpy(), x.dlat.to_numpy())
    ax[0].plot(xs, bb[0] + bb[1] * xs, color=c, lw=2)
    ax[0].text(18.6, bb[0] + bb[1] * 18.5, f'{lab}  {bb[1]:+.2f} pp/°', color=INK, va='center', fontsize=8.5)
ax[0].set_xlim(0, 24)
ax[0].set_xlabel('|Δ纬度|（源-目标国质心，°）')
ax[0].set_ylabel('单源迁移平衡精度 BA（%）')
ax[0].set_title('a  单源两两迁移：精度随纬度差下降', loc='left', fontsize=10, color=INK)

# b: AE−S2 与 |Δlat|，分箱均值 ± SD，Cheng 模式的换算参考线
ax[1].axhline(0, color='#c9c8c2', lw=1)
ax[1].scatter(x.dlat, x.d_ae_s2, s=14, color=BLUE, alpha=.4, lw=0)
bb = ols(x.d_ae_s2.to_numpy(), x.dlat.to_numpy())
ax[1].plot(xs, bb[0] + bb[1] * xs, color=BLUE, lw=2)
ax[1].text(18.6, bb[0] + bb[1] * 18.5, f'本数据\n{bb[1]:+.2f} pp/°', color=INK, va='center', fontsize=8.5)
bins = [0, 3, 6, 9, 12, 20]
x['bin'] = pd.cut(x.dlat, bins, right=False)
bt = x.groupby('bin', observed=True).agg(m=('d_ae_s2', 'mean'), sd=('d_ae_s2', 'std'),
                                          xm=('dlat', 'mean'), n=('d_ae_s2', 'size'))
ax[1].errorbar(bt.xm, bt.m, yerr=bt.sd, fmt='o', ms=6, color=INK, ecolor=INK2, elinewidth=1, capsize=3,
               markeredgecolor='white', markeredgewidth=1.5, zorder=5)
for r in bt.itertuples():
    ax[1].text(r.xm, 12.4, f'n={r.n}', ha='center', fontsize=7.5, color=INK2)
# Cheng Fig.6b（RF）：ΔOA = −0.12 + 0.09/1000 °C·d；按 1° ≈ 150 °C·d 换算（仅量级参照）
ax[1].plot(xs, -12 + 1.35 * xs, color=GRAY, lw=1.5, ls='--')
ax[1].text(18.6, -12 + 1.35 * 18.5, 'Cheng 模式换算\n+1.35 pp/°\n（ΔOA，6 类）', color=INK2, va='center', fontsize=8)
ax[1].set_xlim(0, 24); ax[1].set_ylim(-16, 14)
ax[1].set_xlabel('|Δ纬度|（°）')
ax[1].set_ylabel('AE − S2（pp）')
ax[1].set_title('b  AE 相对 S2 的差随纬度差变化', loc='left', fontsize=10, color=INK)

# c: 多源 k=0（G foreign_only）逐目标国
ax[2].axhline(0, color='#c9c8c2', lw=1)
ax[2].scatter(g0.mean_dlat_to_pool, g0.d_ba, s=40, color=BLUE, lw=1.5, edgecolor='white', zorder=4, label='ba_ge30')
ax[2].scatter(g0.mean_dlat_to_pool, g0.d_oa, s=40, color=ORANGE, marker='s', lw=1.5, edgecolor='white', zorder=4, label='OA')
for r in g0.itertuples():
    ax[2].text(r.mean_dlat_to_pool + 0.12, r.d_ba, r.target, fontsize=7.5, color=INK2, va='center')
    if r.target in ('PT', 'RO', 'SE'):
        ax[2].text(r.mean_dlat_to_pool - 0.12, r.d_oa, r.target + ' (OA)', fontsize=7.5, color=INK2, va='center', ha='right')
ax[2].set_xlabel('目标国到其余 10 国的平均 |Δ纬度|（°）')
ax[2].set_ylabel('AE − S2（pp），k=0 多源外国池')
ax[2].legend(frameon=False, fontsize=8, loc='lower left')
ax[2].set_title('c  多源训练（G，k=0）逐目标国', loc='left', fontsize=10, color=INK)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'fig_posthoc_distance.png'), dpi=200)
print('saved')

# (4) 用 Cheng 2026 Fig. 2 发表的区域年均 GDD（只有 4 国与我们重合；区域范围不同：
#     Cheng 的 DE 是勃兰登堡，我们是 NRW+LS；SE 为 SE12/21/22/23；AT 为 AT1/AT2；PT 为 PT16/PT18）
gdd = {'AT': 3532, 'DE': 3901, 'PT': 6050, 'SE': 3057}
s4 = x[x.source.isin(gdd) & x.target.isin(gdd)].copy()
s4['dgdd'] = [abs(gdd[a] - gdd[b]) for a, b in zip(s4.source, s4.target)]
s4 = s4.sort_values('dgdd')
s4[['source', 'target', 'dlat', 'dgdd', 'ae', 's2', 'd_ae_s2']].round(3).to_csv(
    os.path.join(OUT, 'cheng_gdd_4country_pairs.csv'), index=False)
b4 = ols(s4.d_ae_s2.to_numpy(), (s4.dgdd / 1000).to_numpy())
print('4 国 12 对：AE−S2 对 |ΔGDD| 斜率 pp/1000 °C·d =', round(b4[1], 2),
      '；|ΔGDD|>2000 的对中 AE<S2：', int((s4[s4.dgdd > 2000].d_ae_s2 < 0).sum()), '/', int((s4.dgdd > 2000).sum()))
