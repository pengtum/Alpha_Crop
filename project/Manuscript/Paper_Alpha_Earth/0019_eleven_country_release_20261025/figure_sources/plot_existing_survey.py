"""Fig. 4 RQ2：已有调查的价值与补采位置（双栏 183 × 64 mm，三个面板）。

核心句：已有的 100 个集中样本约值 100 个随机标签；在原区继续补采，价值递减；
在全国补采（随机或 Betti 分配）则与零起点随机设计打平，而 Betti 分配并不比随机好。

只读 source_data/confirm_estimates.csv（learner lgb、feat both、set 9；均值与 95% t CI）。
"""
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import nstyle as ns

ns.setup()
C, FS, MM = ns.C, ns.FS, ns.MM

# ------------------------------------------------------------------ 数据
est = pd.read_csv(ns.src('confirm_estimates.csv'))


def E(est_id, feat='both', set_=9):
    """取 confirm_estimates 中唯一一行（LightGBM 主分析；est_id + feat + set）。"""
    r = est[(est.est_id == est_id) & (est.feat == feat) & (est['set'] == set_) & (est.learner == 'lgb')]
    assert len(r) == 1, (est_id, feat, set_, len(r))
    return r.iloc[0]


def f2(v, signed=True):
    return ns.fmt_pp(v, signed=signed, nd=2)


def npos(r):
    return f'{int(r.n_pos)}/{int(r.n)}'


# ------------------------------------------------------------------ 版式（全部以 mm 定位）
FW, FH = 183, 64
fig = plt.figure(figsize=(FW * MM, FH * MM))
AX_Y0, AX_H = 9.5, 46.5          # 三个面板共用轴底与轴高（mm）
LAB_Y = 60.5                     # 面板标号基线（mm）


def add_ax(x0, y0, w, h):
    return fig.add_axes([x0 / FW, y0 / FH, w / FW, h / FH])


def plabel(ax, letter, x_mm, y_mm=LAB_Y):
    """面板标号：按 mm 位置换算成轴坐标后调用 ns.panel，保证三个标号齐平。"""
    p = ax.get_position()
    ns.panel(ax, letter, x=(x_mm / FW - p.x0) / p.width, y=(y_mm / FH - p.y0) / p.height)


def side_text(ax, y, s, x_mm_off=1.4, **kw):
    """在轴右侧外（按 mm 偏移）写一列文字，y 为数据坐标。"""
    p = ax.get_position()
    kw.setdefault('fontsize', FS['small'])
    kw.setdefault('va', 'center')
    kw.setdefault('ha', 'left')
    ax.text(1 + x_mm_off / (p.width * FW), y, s, transform=ax.get_yaxis_transform(), clip_on=False, **kw)


def ci_row(ax, y, m, lo, hi, color, ms=3.4, lw=0.9):
    ax.plot([lo, hi], [y, y], color=color, lw=lw, solid_capstyle='butt', zorder=3)
    ax.plot([m], [y], 'o', ms=ms, mfc=color, mec=color, mew=0.6, zorder=4)


def val_above(ax, x, y, s, color, dy=2.0):
    # 数值远离零线，保持原来的估计与区间。
    dx = (8 if x >= 0 else -8) if abs(x) < .25 else 0
    ha = ('left' if dx > 0 else 'right') if dx else 'center'
    ax.annotate(s, (x, y), xytext=(dx, dy), textcoords='offset points', ha=ha, va='bottom',
                fontsize=FS['small'], color=color, zorder=5)


def forest_frame(ax, ymax, xlabel, zero_to=None):
    """森林图公共样式：去左轴线、y 刻度不画短线、顶部写国家数与 "> 0" 列头。
    zero_to 给定时，零参考线只画到该 y（数据坐标），为其下方的带内说明留出位置。"""
    if zero_to is None:
        ns.zero_line(ax, axis='x')
    else:
        # 与 ns.zero_line 同样式，只是截短
        ax.plot([0, 0], [-0.9, zero_to], color=C['MUTED'], lw=0.5, ls=(0, (2, 2)), zorder=0)
    ax.set_ylim(ymax, -0.9)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0, pad=2)
    ax.set_xlabel(xlabel)
    side_text(ax, -0.9, '> 0', color=C['MUTED'], va='bottom')
    ax.text(1.0, 1.0, '9 countries', transform=ax.transAxes, ha='right', va='bottom',
            fontsize=FS['small'], color=C['MUTED'])


# ================================================================== a｜总标签数—增益曲线（主面板）
ax = add_ax(11.5, AX_Y0, 50, AX_H)
Z = dict(x=0, m=0.0, lo=0.0, hi=0.0)       # 零标签迁移锚点：按定义 ΔBA = 0
S0 = E('X2_B_S0')
fam = [  # (族, 颜色, x 错开量, [(总标签数, est_id 或 None)])
    ('random', C['LABEL'], -9, [(0, None), (50, 'A1_RT50'), (100, 'A1_RT100'), (200, 'A1_RT200'),
                                 (400, 'A1_RT400')]),
    ('conc', C['CONC'], -3, [(0, None), (100, 'X2_B_S0'), (200, 'X2_B_EXT100'), (400, 'X2_B_EXT300')]),
    ('spread', C['SPREAD'], 3, [(100, 'X2_B_S0'), (200, 'X2_B_SRS100'), (400, 'X2_B_SRS300')]),
    ('betti', C['BETTI'], 9, [(100, 'X2_B_S0'), (200, 'X2_B_BA100'), (400, 'X2_B_BA300')]),
]
for name, col, dx, pts in fam:
    xs, ms_ = [], []
    for k, eid in pts:
        if eid is None:                     # 原点不错开：所有族共用同一个零标签锚点
            xs.append(0.0)
            ms_.append(0.0)
            continue
        r = E(eid)
        # S0 点属于集中族（CONC）；spread/Betti 两族的线从 S0 点的实际位置出发，不重复画点
        x = k + (-3 if eid == 'X2_B_S0' else dx)
        xs.append(x)
        ms_.append(r['mean'])
        if eid == 'X2_B_S0' and name != 'conc':
            continue
        ax.plot([x, x], [r.lo95, r.hi95], color=col, lw=0.6, solid_capstyle='butt', zorder=2)
        ax.plot(x, r['mean'], 'o', ms=3.2, mfc=col, mec='white', mew=0.4, zorder=4)
    ax.plot(xs, ms_, color=col, lw=1.0, zorder=3)
ax.plot(0, 0, 'o', ms=3.2, mfc=C['REF_Z0'], mec='white', mew=0.4, zorder=5)
ns.zero_line(ax, axis='y')
ax.set_xlim(-15, 430)
ax.set_ylim(-0.75, 5.4)
ax.set_xticks([0, 100, 200, 300, 400])
ax.set_yticks([0, 1, 2, 3, 4, 5])
ax.set_xlabel('Target labels in total')
ax.set_ylabel('ΔBA (%)')
# 直接标注：原点与已有调查点
ax.annotate('zero-label transfer', (0, 0), xytext=(4, -3.2), textcoords='offset points', ha='left', va='top',
            fontsize=FS['small'], color=C['REF_Z0'])
ax.annotate('existing survey\n(100 clustered labels)', (97, S0['mean']), xytext=(118, 0.45),
            textcoords='data', ha='left', va='center', fontsize=FS['small'], color=C['CONC'],
            arrowprops=dict(arrowstyle='-', lw=0.5, color=C['CONC'], shrinkA=0.5, shrinkB=2.2))
handles = [Line2D([], [], color=c, lw=1.0, marker='o', ms=3.2, mfc=c, mec='white', mew=0.4) for c in
           (C['LABEL'], C['CONC'], C['SPREAD'], C['BETTI'])]
labels = ['random labels from zero', 'existing survey + extend surveyed area',
          'existing survey + spread nationally', 'existing survey + regional allocation']
ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.0, 1.03), fontsize=FS['small'],
          handlelength=1.8, handletextpad=0.5, labelspacing=0.3, borderaxespad=0.2)
plabel(ax, 'a', 0.8)

# ================================================================== b｜新标签补在哪里（森林图）
ax = add_ax(91.5, AX_Y0, 24, AX_H)
rows_b = [('Spread − extend, +100', 'B1_SRS-EXT100', C['SPREAD'], 0.0),
          ('Spread − extend, +300', 'B1_SRS-EXT300', C['SPREAD'], 1.0),
          ('Regional allocation\n− spread, +100', 'B1_BA-SRS100', C['BETTI'], 2.5),
          ('Regional allocation\n− spread, +300', 'B1_BA-SRS300', C['BETTI'], 3.5)]
ax.axvspan(-1, 1, color=C['BAND'], lw=0, zorder=0)
for lab, eid, col, y in rows_b:
    r = E(eid)
    ci_row(ax, y, r['mean'], r.lo95, r.hi95, col)
    val_above(ax, r['mean'], y, f2(r['mean']), col)
    side_text(ax, y, npos(r))
forest_frame(ax, 5.3, 'ΔBA (%)', zero_to=4.1)
# 带内说明：写在带的底部，零参考线在其上方截止
ax.text(0, 5.15, '±1%\nsmall-effect band', ha='center', va='bottom', fontsize=FS['small'],
        color=C['MUTED'], linespacing=1.3, zorder=5)
ax.set_xlim(-1.45, 1.45)
ax.set_xticks([-1, 0, 1])
ax.set_yticks([y for *_, y in rows_b])
ax.set_yticklabels([l for l, *_ in rows_b])
for t, (_, _, col, _) in zip(ax.get_yticklabels(), rows_b):
    t.set_color(col)
plabel(ax, 'b', 67.5)

# ================================================================== c｜同总数下与零起点随机设计比较（森林图）
CX0, CW = 149.5, 28.0            # c 轴左缘与宽度（mm）；加宽到 28 mm，使导出宽度接近 183 mm 双栏
ax = add_ax(CX0, AX_Y0, CW, AX_H)
groups_c = [('vs 100 random labels', [('Survey (100)', 'X1_S0-RT100', C['CONC'])]),
            ('vs 200 random labels', [('Survey + spread 100', 'X1_SRS100-RT200', C['SPREAD']),
                                      ('Survey + extend 100', 'X1_EXT100-RT200', C['CONC']),
                                      ('Survey + regional\nallocation 100', 'X1_BA100-RT200', C['BETTI'])]),
            ('vs 400 random labels', [('Survey + spread 300', 'X1_SRS300-RT400', C['SPREAD']),
                                      ('Survey + extend 300', 'X1_EXT300-RT400', C['CONC']),
                                      ('Survey + regional\nallocation 300', 'X1_BA300-RT400', C['BETTI'])])]
yt, ylab, ycol = [], [], []
y = 0.0
p = ax.get_position()
for head, rows in groups_c:
    # 组标题：左端与行标签列左缘对齐，写在该组第一行上方
    ax.text(-(CX0 - 125.5) / CW, y - 0.05, head, transform=ax.get_yaxis_transform(), ha='left',
            va='bottom', fontsize=FS['small'], color=C['MUTED'], style='italic', clip_on=False)
    y += 0.8
    for lab, eid, col in rows:
        r = E(eid)
        ci_row(ax, y, r['mean'], r.lo95, r.hi95, col)
        val_above(ax, r['mean'], y, f2(r['mean']), col)
        side_text(ax, y, npos(r))
        yt.append(y)
        ylab.append(lab)
        ycol.append(col)
        y += 1.0
    y += 0.1
forest_frame(ax, y - 0.4, 'ΔBA (%)')
ax.set_xlim(-1.3, 0.6)
ax.set_xticks([-1, -0.5, 0, 0.5])
ax.set_xticklabels(['−1', '−0.5', '0', '0.5'])
ax.set_yticks(yt)
ax.set_yticklabels(ylab)
for t, col in zip(ax.get_yticklabels(), ycol):
    t.set_color(col)
plabel(ax, 'c', 124.5)

ns.save(fig, 'fig4_existing_survey')
print('saved fig4_existing_survey')
