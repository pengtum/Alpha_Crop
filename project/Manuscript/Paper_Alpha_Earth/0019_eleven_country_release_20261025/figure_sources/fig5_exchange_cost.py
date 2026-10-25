"""Fig. 5 RQ2：交换率与成本规则（双栏 183 × 62 mm，三个面板；c 为三个并排小图）。

核心句：集中样本在总数 200 时约值 0.83 个随机标签，在总数 400 时约值 0.72 个。
只有当全国分散采一个样本的成本超过集中采样的约 1.2–1.4 倍时，扩大原调查区才更划算。

只读 source_data/confirm_x3_equiv.csv（k_eq、交换率、集中 − 随机，90% 国家层 bootstrap 区间）
与 source_data/confirm_x3_cost.csv（kind == 'budget'，成本效益点估计）。
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import nstyle as ns

ns.setup()
C, FS, MM = ns.C, ns.FS, ns.MM

# ------------------------------------------------------------------ 数据
eq = pd.read_csv(ns.src('confirm_x3_equiv.csv'))
cost = pd.read_csv(ns.src('confirm_x3_cost.csv'))


def Q(kind, item):
    """取 confirm_x3_equiv 中唯一一行（kind + item）。"""
    r = eq[(eq.kind == kind) & (eq['item'].astype(str) == str(item))]
    assert len(r) == 1, (kind, item, len(r))
    return r.iloc[0]


def f2(v, signed=False):
    return ns.fmt_pp(v, signed=signed, nd=2)


# ------------------------------------------------------------------ 版式（全部以 mm 定位）
FW, FH = 183, 62
fig = plt.figure(figsize=(FW * MM, FH * MM))
AX_Y0, AX_H = 9.5, 39.0          # 三个面板共用轴底与轴高（mm）
LAB_Y = 51.5                     # 面板标号基线（mm）


def add_ax(x0, y0, w, h, **kw):
    return fig.add_axes([x0 / FW, y0 / FH, w / FW, h / FH], **kw)


def plabel(ax, letter, x_mm, y_mm=LAB_Y):
    """面板标号：按 mm 位置换算成轴坐标后调用 ns.panel，保证三个标号齐平。"""
    p = ax.get_position()
    ns.panel(ax, letter, x=(x_mm / FW - p.x0) / p.width, y=(y_mm / FH - p.y0) / p.height)


FAM_COL = dict(S0=C['CONC'], EXT=C['CONC'], SRS=C['SPREAD'], BA=C['BETTI'])

# ------------------------------------------------------------------ 全图共用图例条（放在最上方）
handles = [Line2D([], [], color=C['LABEL'], lw=1.0, marker='o', ms=3.0, mfc=C['LABEL'], mec='white', mew=0.4),
           Line2D([], [], color=C['CONC'], lw=1.0, marker='o', ms=3.0, mfc=C['CONC'], mec='white', mew=0.4),
           Line2D([], [], color=C['SPREAD'], lw=1.0, marker='o', ms=3.0, mfc=C['SPREAD'], mec='white', mew=0.4),
           Line2D([], [], color=C['BETTI'], lw=1.0, marker='o', ms=3.0, mfc=C['BETTI'], mec='white', mew=0.4),
           Line2D([], [], ls='none', marker='*', ms=5.5, mfc='white', mec='black', mew=0.5)]
labels = ['random labels', 'concentrated: existing survey, + extend surveyed area',
          'existing survey + spread nationally', 'existing survey + regional allocation', 'best option at that budget']
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=5, fontsize=FS['small'],
           handlelength=1.8, handletextpad=0.4, columnspacing=1.2, borderaxespad=0.1)

# ================================================================== a｜各配置每个标签值多少个随机标签（交换率 = k_eq ÷ n）
# 横轴是按总标签数分组的类别轴：组内左右错开只为排版，不表示数值（避免在定量轴上错开造成误读）
ax = add_ax(12.5, AX_Y0, 37.5, AX_H)
CAP = 1.25                       # 无上界（inf）时误差棒画到 CAP 并加向上箭头
cfg = [('B_S0', 0.0, 'S0'),
       ('B_EXT100', 1.0 - 0.22, 'EXT'), ('B_SRS100', 1.0, 'SRS'), ('B_BA100', 1.0 + 0.22, 'BA'),
       ('B_EXT300', 2.0 - 0.22, 'EXT'), ('B_SRS300', 2.0, 'SRS'), ('B_BA300', 2.0 + 0.22, 'BA')]
ax.axhline(1.0, color=C['LABEL'], lw=0.7, ls=(0, (3, 2)), zorder=1)
ax.text(0.22, 1.0 + 0.012, 'one random label', ha='left', va='bottom', fontsize=FS['small'], color=C['LABEL'])
for item, x, fam_ in cfg:
    r = Q('rate', item)
    col = FAM_COL[fam_]
    lo, hi, pt = r.lo90, r.hi90, r.point
    ax.plot([x, x], [lo, min(hi, CAP)], color=col, lw=0.9, solid_capstyle='butt', zorder=2)
    if np.isinf(hi):
        ax.annotate('', xy=(x, CAP + 0.06), xytext=(x, CAP), zorder=2, annotation_clip=False,
                    arrowprops=dict(arrowstyle='->,head_length=0.25,head_width=0.15', lw=0.8, color=col,
                                    shrinkA=0, shrinkB=0))
    if np.isinf(pt):
        # 点估计本身超出曲线范围（k_eq > 400）：在 CAP 处画向上三角并标 "> 1.00"
        ax.plot(x, CAP, '^', ms=3.8, mfc=col, mec='white', mew=0.4, zorder=4, clip_on=False)
        ax.text(x + 0.07, CAP - 0.035, '> 1.00', ha='left', va='center', fontsize=FS['small'], color=col)
    else:
        ax.plot(x, pt, 'o', ms=3.6, mfc=col, mec='white', mew=0.4, zorder=4)
        if fam_ in ('EXT', 'S0'):
            ax.text(x - 0.09, pt, f2(pt), ha='right', va='center', fontsize=FS['small'], color=col)
ax.set_xlim(-0.45, 2.5)
ax.set_ylim(0.6, 1.33)
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(['100', '200', '400'])   # 100 = 只有已有调查（S0）
ax.set_yticks([0.6, 0.8, 1.0, 1.2])
ax.set_xlabel('Target labels in total')
ax.set_ylabel('Worth of one label (random labels)')
plabel(ax, 'a', 0.8)

# ================================================================== b｜集中样本的交换率与盈亏平衡成本比
ax = add_ax(69.5, AX_Y0, 45, AX_H)
xs = [100, 200, 400]
d = [Q('conc_diff', k) for k in xs]
ns.zero_line(ax, axis='y')
ax.plot(xs, [r.point for r in d], color=C['CONC'], lw=0.8, zorder=2)
for k, r in zip(xs, d):
    ax.plot([k, k], [r.lo90, r.hi90], color=C['CONC'], lw=0.8, solid_capstyle='butt', zorder=3)
    ax.plot(k, r.point, 'o', ms=3.4, mfc=C['CONC'], mec='white', mew=0.4, zorder=4)
ax.text(445, 0.02, 'random labels, same total', ha='right', va='bottom', fontsize=FS['small'],
        color=C['MUTED'])
# 点旁标注：交换率（conc_rate）及其派生量 break-even = 1 / rate
rt = {k: Q('conc_rate', k) for k in (200, 400)}
be = {k: 1.0 / rt[k].point for k in (200, 400)}
ax.text(112, d[0].point + 0.23, 'not distinguishable\n(rate not reported)', ha='left', va='bottom',
        fontsize=FS['small'], color=C['TEXT'], linespacing=1.05)
ax.text(213, d[1].point + 0.035,
        f'rate {f2(rt[200].point)} [{f2(rt[200].lo90)}, {f2(rt[200].hi90)}]\n'
        f'→ break-even cost ratio {f2(be[200])}',
        ha='left', va='bottom', fontsize=FS['small'], color=C['TEXT'], linespacing=1.05)
ax.text(385, d[2].point - 0.07,
        f'rate {f2(rt[400].point)} [{f2(rt[400].lo90)}, {f2(rt[400].hi90)}]\n→ {f2(be[400])}',
        ha='right', va='top', multialignment='left', fontsize=FS['small'], color=C['TEXT'], linespacing=1.05)
ax.set_xlim(60, 450)
ax.set_ylim(-1.15, 0.48)
ax.set_xticks(xs)
ax.set_yticks([-1, -0.5, 0])
ax.set_yticklabels(['−1', '−0.5', '0'])
ax.set_xlabel('Target labels in total')
ax.set_ylabel('ΔBA (%)')
plabel(ax, 'b', 55.5)

# ================================================================== c｜成本效益（c = 1、2、3 三个小图，共用 y 轴）
bud = cost[cost.kind == 'budget'].copy()
LINES = [('random', C['LABEL']), ('S0+SRS', C['SPREAD']), ('S0+BA', C['BETTI']), ('concentrated', C['CONC'])]
X0, W, GAP = 131.5, 15.5, 2.2
axes_c = []
for j, c in enumerate((1, 2, 3)):
    ax = add_ax(X0 + j * (W + GAP), AX_Y0, W, AX_H, sharey=axes_c[0] if axes_c else None)
    axes_c.append(ax)
    sub = bud[bud.c == c]
    for name, col in LINES:        # 集中族最后画：并列最优（均为已有调查本身）时星号显示集中色
        s = sub[sub.name == name].sort_values('cost')
        ax.plot(s.cost, s.value, color=col, lw=0.9, marker='o', ms=2.4, mfc=col, mec='white', mew=0.3,
                zorder=3)
        b = s[s.best.astype(str) == 'True']
        ax.plot(b.cost, b.value, ls='none', marker='*', ms=6.0, mfc=col, mec='black', mew=0.45, zorder=5)
    ax.set_xscale('log', base=2)
    ax.set_xlim(80, 500)
    ax.set_xticks([100, 200, 400])
    ax.set_xticklabels(['100', '200', '400'])
    ax.minorticks_off()
    ax.set_title(f'c = {c}', fontsize=FS['label'], pad=2.5)
    if j > 0:
        ax.tick_params(axis='y', labelleft=False)
axes_c[0].set_ylim(0, 4.2)
axes_c[0].set_yticks([0, 1, 2, 3, 4])
axes_c[0].set_ylabel('ΔBA (%)')
axes_c[1].set_xlabel('Budget (in costs of one concentrated label)')
fig.text((X0 + 1.5 * W + GAP) / FW + 0.0, (AX_Y0 + AX_H + 4.3) / FH,
         '(c = cost of a spread label ÷ cost of a concentrated label)', ha='center', va='bottom',
         fontsize=FS['small'], color=C['MUTED'])
plabel(axes_c[0], 'c', 119.5)

ns.save(fig, 'fig5_exchange_cost')
print('saved fig5_exchange_cost')
