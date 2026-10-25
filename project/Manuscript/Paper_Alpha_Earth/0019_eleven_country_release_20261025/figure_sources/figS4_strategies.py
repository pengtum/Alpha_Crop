"""Fig. S4 选样策略对比随机（1.5 栏，120 × 70 mm）。

核心句：在预注册的选样指导实验中，8 种几何/不确定性选样方法相对随机抽样的 BA 差都落在约 ±1% 内或跨 0；
无论用 AE 还是 S2 作指导特征，Holm 校正后都没有方法显著优于随机。
数据：source_data/strategy_vs_srs.csv（guidance ∈ {AE, S2}；AE+S2 不画）。
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory

import nstyle as ns

ns.setup()
MM = ns.MM
C = ns.C

# ------------------------------------------------------------ 数据
s = pd.read_csv(ns.src('strategy_vs_srs.csv'))
s = s[s.guidance.isin(['AE', 'S2'])].set_index(['method', 'guidance'])
N_COUNTRIES = 11

METHODS = [
    ('predstrat_srs', 'prediction-stratified'),
    ('kcenter', 'k-center'),
    ('kherding', 'kernel herding'),
    ('lpm2_nogeo', 'LPM (feature space)'),
    ('typiclust', 'TypiClust'),
    ('margin', 'margin'),
    ('gist_margin', 'GIST-margin'),
    ('badge_c', 'BADGE'),
]
GUIDE = [('AE', C['AE'], -0.23), ('S2', C['S2'], +0.23)]   # 指导特征：颜色与上下错开量

# ------------------------------------------------------------ 版式（mm）
FIG_W, FIG_H = 120.0, 70.0
AX_L, AX_R, AX_B, AX_T = 23.0, 111.0, 8.0, 68.0
fig = plt.figure(figsize=(FIG_W * MM, FIG_H * MM))
ax = fig.add_axes([AX_L / FIG_W, AX_B / FIG_H, (AX_R - AX_L) / FIG_W, (AX_T - AX_B) / FIG_H])

XMIN, XMAX = -2.25, 2.5
ax.set_xlim(XMIN, XMAX)
ax.set_ylim(len(METHODS) - 0.45, -0.95)

# ±1% 背景带与 0 参考线
ax.axvspan(-1, 1, color=C['BAND'], lw=0, zorder=0)
ns.zero_line(ax, axis='x')
ax.text(-0.95, -0.62, '±1%', ha='left', va='center', fontsize=ns.FS['small'], color=C['MUTED'])

tr = blended_transform_factory(ax.transAxes, ax.transData)
for i, (m, lab) in enumerate(METHODS):
    for g, col, dy in GUIDE:
        r = s.loc[(m, g)]
        y = i + dy
        ax.plot([r.lo95, r.hi95], [y, y], color=col, lw=1.0, solid_capstyle='butt', zorder=2)
        ax.plot(r['mean'], y, 'o', ms=3.6, mfc=col, mec='white', mew=0.5, zorder=3)
        # 右侧：均值 > 0 的国家数 / 11
        ax.text(1.03, y, f'{int(r.n_pos)}/{N_COUNTRIES}', transform=tr, ha='left', va='center',
                fontsize=ns.FS['small'], color=col, clip_on=False)

# 右侧列标题
ax.text(1.03, -0.5, 'countries\n> 0', transform=tr, ha='left', va='bottom', linespacing=1.15, fontsize=ns.FS['small'],
        color=C['MUTED'], clip_on=False)

# 指导特征的直接标注：放在第一行左侧空白处，纵向位置与各自的点对齐
for g, col, dy in GUIDE:
    ax.text(XMIN + 0.08, 0 + dy, f'{g}-guided', ha='left', va='center', fontsize=ns.FS['small'], color=col)

ax.set_yticks(np.arange(len(METHODS)))
ax.set_yticklabels([lab for _, lab in METHODS])
ax.tick_params(axis='y', length=0, pad=3)
ax.spines['left'].set_visible(False)
ax.set_xticks([-2, -1, 0, 1, 2])
ax.set_xticklabels(['−2', '−1', '0', '1', '2'])
ax.set_xlabel('ΔBA (%)', labelpad=2)

ns.save(fig, 'figS4_strategies')
