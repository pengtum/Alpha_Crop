"""Fig. S3 AE 作为选源距离（单栏，89 × 60 mm）。

核心句：以 −距离 与单源迁移 BA 的 Spearman ρ 衡量，AE 嵌入上的 MMD 并不比 S2 上的 MMD
更能预测哪个外国源迁移得好（Δρ = +0.021，区间跨 0，6/11 国更好）；简单的地理距离与类别构成差异同样有效。
数据：source_data/source_selection_rho.csv（均值与 95% bootstrap CI，原样）；source_selection_rho_per_target.csv（逐目标国 ρ，只作散点）。
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import nstyle as ns

ns.setup()
MM = ns.MM
C = ns.C

# ------------------------------------------------------------ 数据
rho = pd.read_csv(ns.src('source_selection_rho.csv')).set_index('descriptor')
pt = pd.read_csv(ns.src('source_selection_rho_per_target.csv'))
wide = pt.pivot(index='target', columns='descriptor', values='rho')

# 行：显示名、颜色（AE → AE，S2 → S2，其余 → REF_Z0）
ROWS = [
    ('mmd_ae', 'MMD (AE)', C['AE']),
    ('mmd_s2', 'MMD (S2)', C['S2']),
    ('c2st_ae', 'C2ST (AE)', C['AE']),
    ('c2st_s2', 'C2ST (S2)', C['S2']),
    ('geo_km', 'geographic distance', C['REF_Z0']),
    ('climate_diff', 'climate difference', C['REF_Z0']),
    ('class_js', 'class-composition JS', C['REF_Z0']),
]
# 行的纵坐标：AE/S2 成对，组间留小空隙
Y = np.array([0, 1, 2.35, 3.35, 4.7, 5.7, 6.7])

PRIMARY = ['AT', 'BE', 'DE', 'EE', 'LT', 'LV', 'SE', 'SI', 'SK']
EXTRA = ['RO', 'PT']
# 逐国散点的纵向偏移：固定到国家（同一国家在各行位置相同）；RO、PT 放在带的上沿，便于标注
OFF = dict(zip(PRIMARY, np.linspace(-0.12, 0.26, len(PRIMARY))))
OFF.update(RO=-0.27, PT=-0.27)

# Δρ（AE-MMD − S2-MMD）：点估计与 n_pos 由逐目标国值核对；区间取自源分析报告（见图例文件）
d_mmd = (wide['mmd_ae'] - wide['mmd_s2'])
_dl = pd.read_csv(ns.src('source_selection_delta.csv')).iloc[0]   # Δρ 与 95% bootstrap CI（分析报告原样解析）
DRHO, DRHO_LO, DRHO_HI = float(_dl.d_rho), float(_dl.lo95), float(_dl.hi95)
assert abs(d_mmd.mean() - DRHO) < 5e-4 and int((d_mmd > 0).sum()) == 6

# ------------------------------------------------------------ 版式（mm）
FIG_W, FIG_H = 89.0, 60.0
AX_L, AX_R, AX_B, AX_T = 24.0, 75.0, 8.0, 59.5
fig = plt.figure(figsize=(FIG_W * MM, FIG_H * MM))
ax = fig.add_axes([AX_L / FIG_W, AX_B / FIG_H, (AX_R - AX_L) / FIG_W, (AX_T - AX_B) / FIG_H])

XMIN, XMAX = -0.85, 1.0
ax.set_xlim(XMIN, XMAX)
ax.set_ylim(Y[-1] + 0.55, -0.75)
ns.zero_line(ax, axis='x')

for (key, lab, col), y in zip(ROWS, Y):
    # 逐目标国 ρ：主集实心淡点，RO/PT 空心
    for cc in PRIMARY + EXTRA:
        v = wide.loc[cc, key]
        if np.isnan(v):
            continue
        if cc in PRIMARY:
            ax.plot(v, y + OFF[cc], 'o', ms=2.1, mew=0, color=col, alpha=0.38, zorder=2)
        else:
            ax.plot(v, y + OFF[cc], 'o', ms=2.1, mfc='white', mec=col, mew=0.5, alpha=0.9, zorder=2)
    # 目标国平均 ρ 与 95% bootstrap CI
    r = rho.loc[key]
    ax.plot([r.lo95, r.hi95], [y, y], color=col, lw=1.1, solid_capstyle='butt', zorder=3)
    ax.plot(r.rho, y, 'o', ms=4.2, mfc=col, mec='white', mew=0.6, zorder=4)

# RO、PT 只在第一行标注国家代码
for cc in EXTRA:
    v = wide.loc[cc, 'mmd_ae']
    ax.text(v, Y[0] + OFF[cc] - 0.17, cc, ha='center', va='bottom', fontsize=ns.FS['small'],
            color=C['MUTED'])

# y 轴：显示名，着色与点一致
ax.set_yticks(Y)
ax.set_yticklabels([r[1] for r in ROWS])
for tl, (_, _, col) in zip(ax.get_yticklabels(), ROWS):
    tl.set_color(col)
ax.tick_params(axis='y', length=0, pad=3)
ax.spines['left'].set_visible(False)

ax.set_xticks([-0.5, 0, 0.5, 1.0])
ax.set_xticklabels(['−0.5', '0', '0.5', '1.0'])
ax.set_xlabel('Spearman ρ (−distance vs transfer BA)', labelpad=2)

# AE-MMD 与 S2-MMD 的配对差：右侧括号 + 注释
bx = 0.95
ax.plot([bx - 0.03, bx, bx, bx - 0.03], [Y[0], Y[0], Y[1], Y[1]], color=C['TEXT'], lw=0.5,
        clip_on=False, zorder=3)
ax.text(bx + 0.07, (Y[0] + Y[1]) / 2,
        f'MMD, AE − S2:\nΔρ = {ns.fmt_pp(DRHO, nd=3)}\n[{ns.fmt_pp(DRHO_LO, signed=False, nd=3)}, '
        f'{ns.fmt_pp(DRHO_HI, signed=False, nd=3)}]\nAE higher in 6/11',
        ha='left', va='center', fontsize=ns.FS['small'], color=C['TEXT'], linespacing=1.25,
        clip_on=False)

# 样本说明：右侧空白处，与下方三行对齐
ax.text(bx + 0.07, Y[-1] + 0.25, 'n = 11 targets\n× 10 sources\n= 110 pairs', ha='left', va='bottom',
        fontsize=ns.FS['small'], color=C['MUTED'], linespacing=1.25, clip_on=False)

ns.save(fig, 'figS3_source_selection')
