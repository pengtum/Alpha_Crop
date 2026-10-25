"""Fig. S2 AE 优势来自哪里（C 级，事后描述；双栏 183 × 62 mm）。

a｜按作物类：同 OA 点的 AE − S2 召回差（11 国口径），按 d11 排序的横向森林图。
b｜按地块大小：同 OA 点的 AE − S2 BA（11 国口径），组为 30 m 核面积的倍数；x 轴下方标各组地块占比。

只读 source_data/：ae_by_class.csv、ae_by_size.csv。
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import nstyle as ns

ns.setup()
C, FS, MM = ns.C, ns.FS, ns.MM

cls = pd.read_csv(ns.src('ae_by_class.csv'))
size = pd.read_csv(ns.src('ae_by_size.csv'))

# 类名显示（图中英文）
NAMES = dict(grassland='Grassland', maize='Maize', orchards='Orchards', other_crops='Other crops',
             potatoes='Potatoes', root_crops='Root crops', strawberries='Strawberries',
             tree_wood_forest='Tree/wood/forest', vegetables='Vegetables', winter_cereals='Winter cereals')


def f2(v, signed=True):
    return ns.fmt_pp(v, signed=signed, nd=2)


FW, FH = 183, 62
fig = plt.figure(figsize=(FW * MM, FH * MM))


def add_ax(x0, y0, w, h):
    return fig.add_axes([x0 / FW, y0 / FH, w / FW, h / FH])


def plabel(ax, letter, x_mm, y_mm):
    """面板标号：按 mm 位置换算为轴坐标后调用 ns.panel。"""
    p = ax.get_position()
    ns.panel(ax, letter, x=(x_mm / FW - p.x0) / p.width, y=(y_mm / FH - p.y0) / p.height)


AX_Y0, AX_H = 15.5, 40.0
LAB_Y = 58.0

# ================================================================== a｜按作物类
ax = add_ax(25, AX_Y0, 72, AX_H)
d = cls.sort_values('d11', ascending=False).reset_index(drop=True)
y = np.arange(len(d))
for yi, r in zip(y, d.itertuples()):
    ax.plot([r.lo11, r.hi11], [yi, yi], color=C['AE'], lw=0.9, solid_capstyle='butt', zorder=2)
    ax.plot([r.d11], [yi], 'o', ms=3.4, color=C['AE'], mec=C['AE'], zorder=3)
    # 点估计写在点上方；白底只遮零线
    ax.annotate(f2(r.d11), (r.d11, yi), xytext=(0, 1.8), textcoords='offset points', ha='center', va='bottom',
                fontsize=FS['small'], color=C['AE'], zorder=5,
                bbox=dict(boxstyle='square,pad=0.05', fc='white', ec='none'))
    ax.text(1.015, yi, f'{r.pos11}/{r.n11}', transform=ax.get_yaxis_transform(), ha='left', va='center',
            fontsize=FS['small'], clip_on=False)
ax.text(1.015, -0.95, '> 0', transform=ax.get_yaxis_transform(), ha='left', va='bottom', fontsize=FS['small'],
        color=C['MUTED'], clip_on=False)
ns.zero_line(ax, axis='x')
ax.set_yticks(y)
ax.set_yticklabels([f'{NAMES[c]} ({n})' for c, n in zip(d.cls, d.n11)])
ax.set_ylim(len(d) - 0.45, -0.95)
ax.set_xlim(-12, 30)
ax.set_xticks([-10, 0, 10, 20, 30])
ax.tick_params(axis='y', length=0, pad=2)
ax.spines['left'].set_visible(False)
ax.set_xlabel('AE − S2 recall at matched OA (pp)')
ax.text(-0.02, 1.0, 'Class (countries)', transform=ax.transAxes, ha='right', va='bottom', fontsize=FS['small'],
        color=C['MUTED'])
plabel(ax, 'a', 1.0, LAB_Y)

# ================================================================== b｜按地块大小
ax = add_ax(123, AX_Y0, 57, AX_H)
x = np.arange(len(size))
for xi, r in zip(x, size.itertuples()):
    ax.plot([xi, xi], [r.lo11, r.hi11], color=C['AE'], lw=0.9, solid_capstyle='butt', zorder=2)
    ax.plot([xi], [r.d11], 'o', ms=3.6, color=C['AE'], mec=C['AE'], zorder=3)
    ax.text(xi + 0.12, r.d11, f2(r.d11), ha='left', va='center', fontsize=FS['small'], color=C['AE'], zorder=5,
            bbox=dict(boxstyle='square,pad=0.05', fc='white', ec='none'))
    ax.text(xi, r.hi11 + 0.35, f'{r.pos11}/{r.n11}', ha='center', va='bottom', fontsize=FS['small'],
            color=C['TEXT'])
ns.zero_line(ax, axis='y')
ax.set_xticks(x)
ax.set_xticklabels(size.size_group)
ax.set_xlim(-0.5, len(size) - 0.3)
ax.set_ylim(-6, 11)
ax.set_yticks([-5, 0, 5, 10])
ax.set_ylabel('AE − S2 BA at matched OA (pp)')
# x 轴下方另起一行：各组地块占比（11 国）
# 用 offset points 定位在刻度标签下方，避免与轴标签重叠
for xi, sh in zip(x, size.share11):
    ax.annotate(f'{sh * 100:.1f}%', (xi, 0), xycoords=('data', 'axes fraction'), xytext=(0, -13.5),
                textcoords='offset points', ha='center', va='top', fontsize=FS['small'], color=C['MUTED'])
ax.annotate('share of parcels', (-0.5, 0), xycoords=('data', 'axes fraction'), xytext=(-3, -13.5),
            textcoords='offset points', ha='right', va='top', fontsize=FS['small'], color=C['MUTED'])
ax.set_xlabel('Parcel area (multiples of the kernel area π·30² m² ≈ 0.28 ha)', labelpad=10)
ax.text(0.0, 1.0, '11 countries', transform=ax.transAxes, ha='left', va='bottom', fontsize=FS['small'],
        color=C['MUTED'])
plabel(ax, 'b', 111.0, LAB_Y)

ns.save(fig, 'figS2_ae_drivers')
print('saved')
