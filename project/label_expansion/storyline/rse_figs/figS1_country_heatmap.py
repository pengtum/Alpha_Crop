"""Fig. S1 逐国结果热图（双栏，约 183 × 100 mm）。

核心句：逐国看，各项确认性结论在 9 个主集国家中方向一致；RO 在所有加标签配置下为负，
AE − S2 的优势在加标签后仍保留。
数据：source_data/confirm_country_values.csv（只读，不重算）。
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb
from matplotlib.patches import Rectangle
from matplotlib.colorbar import ColorbarBase

import nstyle as ns

ns.setup()
MM = ns.MM
C = ns.C

# ------------------------------------------------------------ 数据
d = pd.read_csv(ns.src('confirm_country_values.csv'))

PRIMARY = ['AT', 'BE', 'DE', 'EE', 'LT', 'LV', 'SE', 'SI', 'SK']   # 9 国主集（按字母）
EXTRA = ['RO', 'PT']                                               # 不在主集中
COUNTRIES = PRIMARY + EXTRA

# 分组 → [(learner, feat, name, 行标签)]；行顺序严格按契约
GROUPS = [
    ('Scenario A (random labels)', [
        ('lgb', 'both', 'dFA_RT50', '+50 random target labels'),
        ('lgb', 'both', 'dFA_RT100', '+100 random target labels'),
        ('lgb', 'both', 'dFA_RT200', '+200 random target labels'),
        ('lgb', 'both', 'dFA_RT400', '+400 random target labels'),
        ('lgb', 'both', 'dFA_TGT', 'target-only model (~30k labels)'),
    ]),
    ('Scenario B (existing survey)', [
        ('lgb', 'both', 'dFA_B_S0', 'existing survey (100 clustered labels)'),
        ('lgb', 'both', 'dFA_B_SRS100', 'survey + 100, spread nationally (random)'),
        ('lgb', 'both', 'dFA_B_SRS300', 'survey + 300, spread nationally (random)'),
        ('lgb', 'both', 'dFA_B_EXT100', 'survey + 100, extend surveyed area'),
        ('lgb', 'both', 'dFA_B_EXT300', 'survey + 300, extend surveyed area'),
        ('lgb', 'both', 'dFA_B_BA100', 'survey + 100, Betti allocation'),
        ('lgb', 'both', 'dFA_B_BA300', 'survey + 300, Betti allocation'),
    ]),
    ('Placement contrasts', [
        ('lgb', 'both', 'B1_SRS-EXT100', 'Spread − extend, +100'),
        ('lgb', 'both', 'B1_SRS-EXT300', 'Spread − extend, +300'),
        ('lgb', 'both', 'B1_BA-SRS100', 'Betti − spread, +100'),
        ('lgb', 'both', 'B1_BA-SRS300', 'Betti − spread, +300'),
    ]),
    ('AE − S2', [
        ('lgb', 'ae-s2', 'D_Z0', 'zero labels'),
        ('lgb', 'ae-s2', 'D_RT200', '+200 random target labels'),
        ('lgb', 'ae-s2', 'D_B_SRS100', 'survey + 100, spread nationally'),
    ]),
    ('Random forest', [
        ('rf', 'both', 'dFA_RT200', '+200 random target labels'),
        ('rf', 'ae-s2', 'D_Z0', 'AE − S2, zero labels'),
    ]),
]


def row_values(learner, feat, name):
    s = d[(d.learner == learner) & (d.feat == feat) & (d.name == name)].set_index('country').value
    assert set(COUNTRIES) <= set(s.index), (learner, feat, name)
    return s.reindex(COUNTRIES).to_numpy(float)


def fmt1(v):
    """1 位小数，负号用 U+2212；四舍五入为 0 的值不带符号。"""
    s = f'{v:.1f}'
    if s in ('-0.0', '0.0'):
        return '0.0'
    return s.replace('-', '−')


# ------------------------------------------------------------ 颜色：DOWN — #F7F7F7 — LABEL，截断 ±8
VMAX = 8.0
cmap = LinearSegmentedColormap.from_list('div', [C['DOWN'], '#F7F7F7', C['LABEL']], N=256)
norm = Normalize(-VMAX, VMAX, clip=True)


def text_colour(v):
    r, g, b = to_rgb(cmap(norm(v)))
    return 'white' if 0.299 * r + 0.587 * g + 0.114 * b < 0.5 else C['TEXT']


# ------------------------------------------------------------ 版式（全部以 mm 计，保证最终尺寸）
FIG_W, FIG_H = 183.0, 100.0
X_LAB = 1.0            # 分组标题左端
X_HM = 55.0            # 热图左端
CW = 8.9               # 单元格宽
GAP_COL = 0.45         # 主集与 RO/PT 之间的空列（以单元格宽计）
RH = 3.2               # 行高
GH = 1.35              # 组标题占用的行高倍数
TOP = 9.5              # 顶部列标题区高度

col_x = np.r_[np.arange(len(PRIMARY)), len(PRIMARY) + GAP_COL + np.arange(len(EXTRA))]
n_units = sum(len(g[1]) for g in GROUPS) + GH * len(GROUPS)
body_h = n_units * RH
hm_w = (col_x[-1] + 1) * CW

fig = plt.figure(figsize=((FIG_W - 1.2) * MM, FIG_H * MM))   # 导出时 pad 约加 1 mm，画布收窄以保证成图 ≤ 183 mm
# 用一个覆盖整张图的坐标轴，单位为 mm，y 向下增长
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, FIG_W)
ax.set_ylim(FIG_H, 0)
ax.axis('off')

# 列标题：国家代码 + 跨列说明
y_code = TOP - 1.2
for j, cc in enumerate(COUNTRIES):
    xc = X_HM + (col_x[j] + 0.5) * CW
    ax.text(xc, y_code, cc, ha='center', va='bottom', fontsize=ns.FS['tick'],
            color=C['TEXT'] if cc in PRIMARY else C['MUTED'])
y_span = 3.2
x0p, x1p = X_HM + 0.15 * CW, X_HM + (col_x[len(PRIMARY) - 1] + 0.85) * CW
ax.plot([x0p, x1p], [y_span + 0.9] * 2, color=C['MUTED'], lw=0.5, solid_capstyle='butt')
ax.text((x0p + x1p) / 2, y_span + 0.2, 'Primary set (9 countries)', ha='center', va='bottom',
        fontsize=ns.FS['small'], color=C['MUTED'])
x0e, x1e = X_HM + (col_x[len(PRIMARY)] + 0.15) * CW, X_HM + (col_x[-1] + 0.85) * CW
ax.plot([x0e, x1e], [y_span + 0.9] * 2, color=C['MUTED'], lw=0.5, solid_capstyle='butt')
ax.text((x0e + x1e) / 2, y_span + 0.2, 'Not in primary set', ha='center', va='bottom',
        fontsize=ns.FS['small'], color=C['MUTED'])

# 热图主体
y = TOP
printed = []            # 记录图中每个数字，供图例文件核对
clipped = []
for gname, rows in GROUPS:
    # 组标题：放在该组第一行上方的空隙里
    ax.text(X_LAB, y + GH * RH - 0.55, gname, ha='left', va='bottom', fontsize=ns.FS['label'],
            fontweight='bold', color=C['TEXT'])
    y += GH * RH
    for learner, feat, name, lab in rows:
        vals = row_values(learner, feat, name)
        yc = y + RH / 2
        ax.text(X_HM - 1.4, yc, lab, ha='right', va='center', fontsize=ns.FS['tick'], color=C['TEXT'])
        for j, v in enumerate(vals):
            x0 = X_HM + col_x[j] * CW
            ax.add_patch(Rectangle((x0, y), CW, RH, facecolor=cmap(norm(v)), edgecolor='white',
                                   lw=0.6, zorder=1))
            tc = text_colour(v)
            s = fmt1(v)
            ax.text(x0 + CW / 2, yc + 0.08, s, ha='center', va='center', fontsize=ns.FS['small'],
                    color=tc, zorder=3)
            printed.append((name, learner, feat, COUNTRIES[j], s))
            # 超出色域（|v| > 8）的格子：右上角小三角，正向 ▲、负向 ▼
            if abs(v) > VMAX:
                mk = '^' if v > 0 else 'v'
                ax.plot(x0 + CW - 1.05, y + 0.95, marker=mk, ms=2.1, mew=0, color=tc, zorder=3)
                clipped.append((name, learner, feat, COUNTRIES[j], s))
        y += RH

body_bottom = y

# ------------------------------------------------------------ 色标（竖向，右侧，与 Scenario A 组顶端对齐）
cb_x = X_HM + hm_w + 5.0
cb_top = TOP + GH * RH
cb_h = 38.0
cax = fig.add_axes([cb_x / FIG_W, 1 - (cb_top + cb_h) / FIG_H, 2.2 / FIG_W, cb_h / FIG_H])
cb = ColorbarBase(cax, cmap=cmap, norm=norm, orientation='vertical', extend='both', extendfrac=0.06,
                  ticks=[-8, -4, 0, 4, 8])
cb.ax.set_yticklabels(['−8', '−4', '0', '4', '8'])
cb.outline.set_linewidth(0.4)
cb.ax.tick_params(width=0.5, length=2, labelsize=ns.FS['tick'], pad=1.5)
cb.set_label('Difference in BA at\nmatched OA (pp)', fontsize=ns.FS['label'], labelpad=3)

# 色标下方说明截断符号
note_y = cb_top + cb_h + 5.5
ax.plot(cb_x + 1.1, note_y, marker='^', ms=2.6, mew=0, color=C['LABEL'])
ax.plot(cb_x + 1.1, note_y + 3.0, marker='v', ms=2.6, mew=0, color=C['DOWN'])
ax.text(cb_x + 3.0, note_y, 'beyond +8', ha='left', va='center', fontsize=ns.FS['small'], color=C['TEXT'])
ax.text(cb_x + 3.0, note_y + 3.0, 'beyond −8', ha='left', va='center', fontsize=ns.FS['small'], color=C['TEXT'])

ns.save(fig, 'figS1_country_heatmap')

# ------------------------------------------------------------ 供核对：打印超色域格子与版面尺寸
print('body bottom (mm):', round(body_bottom, 1), ' heatmap right (mm):', round(X_HM + hm_w, 1))
print('clipped cells:', clipped)
print('n printed numbers:', len(printed))
