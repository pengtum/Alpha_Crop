"""Fig. 7 给实践者的决策图（双栏 183 mm；示意图）。

核心句：先换表征，再决定标签——没有调查时随机采几百个；已有调查时保留它，新标签按成本比决定
分散还是扩区；用同 OA 的 BA 评估。

版式：三列（1 表征 → 2 标签 → 3 评估），标题框之间用粗箭头连接；每列下挂 2 个内容框（两行对齐）。
框内数字全部从 source_data/ 读取，不手写；不重算统计量（break-even = 1/rate 为派生量，图例说明）。
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import nstyle as ns

ns.setup()
C, FS, MM = ns.C, ns.FS, ns.MM


def tint(c, a):
    """颜色与白色按比例混合（不透明），a = 0.12 即 12% 浓度的填色。"""
    r, g, b = mpl.colors.to_rgb(c)
    return np.array([1 - a + a * r, 1 - a + a * g, 1 - a + a * b])


# ------------------------------------------------------------------ 读数（全部来自 source_data）
est = pd.read_csv(ns.src('confirm_estimates.csv'))
x3 = pd.read_csv(ns.src('confirm_x3_equiv.csv'))
a2 = pd.read_csv(ns.src('confirm_a2_ratio.csv'))


def e(est_id, feat, s, col='mean'):
    r = est[(est.est_id == est_id) & (est.feat == feat) & (est.set == s) & (est.learner == 'lgb')]
    assert len(r) == 1, (est_id, feat, s)
    return float(r[col].iloc[0])


pp = ns.fmt_pp
H2_11 = e('H2', 'ae-s2', 11)                        # AE − S2，零标签，11 国（确认性）
N_H2 = int(e('H2', 'ae-s2', 11, 'n'))
RT200 = e('A1_RT200', 'both', 9)                    # +200 随机标签，9 国
RT400 = e('A1_RT400', 'both', 9)                    # +400 随机标签，9 国
RATIO400 = float(a2[(a2.k == 400) & (a2.set == 9)].ratio.iloc[0])        # 400 个标签补上 TGT 缺口的比例
KEQ_S0 = x3[(x3.kind == 'keq') & (x3.item == 'B_S0')]                    # 100 个集中样本约值几个随机标签
KEQ_S0 = f'{float(KEQ_S0.point.iloc[0]):.0f}'
# 已有调查之后的盈亏平衡成本比（派生量）：新标签预算相同（以集中标签单价计）时，扩区 = k1 个集中标签，
# 分散 = k1/c 个分散标签；在 S0 → S0+SRS100 → S0+SRS300 族内按成本线性插值（与成本表相同），求两者相等的 c
_cfg = pd.read_csv(ns.src('confirm_x3_cost.csv'))
_v = _cfg[(_cfg.kind == 'config') & (_cfg.c == 1)].set_index('name').value
BE_LO = (_v['B_SRS100'] - _v['B_S0']) / (_v['B_EXT100'] - _v['B_S0'])                       # 新增 100 个
BE_HI = 300 / (100 + 200 * (_v['B_EXT300'] - _v['B_SRS100']) / (_v['B_SRS300'] - _v['B_SRS100']))  # 新增 300 个
H2_9 = e('H2', 'ae-s2', 9)                          # AE − S2，零标签，9 国（与 400 标签同口径比较）
M1_BA = e('M1_RT200-Z0_BA', 'both', 9)              # 固定 τ = 0 时 +200 标签的 BA 差
H1 = e('H1', 'both', 9)                             # 同 OA 下 +200 标签的 BA 差
assert abs(H1 - RT200) < 1e-9
assert RT400 < H2_9                                 # 同为 9 国：AE 增益的均值高于 400 个标签

# ------------------------------------------------------------------ 版式常数（mm）
FS_BODY, FS_LEAD, FS_HEAD = 6.5, 6.5, 7.0
LH = 3.25           # 正文行高
PADX, PADY = 2.0, 2.5
GAPX = 0.6          # 同一行两段文字之间的空隙（约一个空格宽）
GAP_ARROW = 0.9     # Arial 的 → 字形几乎占满字宽，其后多留一点（视觉补偿）
HEAD_H, ROW_GAP, HEAD_GAP, MARGIN = 7.0, 3.2, 3.6, 0.4
BOX_H = 3 * LH + 2 * PADY                       # 每个内容框 3 行
# 三列位置（左边 x、宽 w）；右边止于 181.9 mm，加上导出时 0.5 mm 的白边后总宽 ≤ 183 mm
COLS = [(0.3, 49.8), (56.2, 72.2), (134.6, 47.3)]

# ------------------------------------------------------------------ 画布（mm 坐标）
FW = 181.98          # 画布宽；导出时两侧各加 0.51 mm 白边，总宽 183 mm
FH = MARGIN + HEAD_H + HEAD_GAP + BOX_H + ROW_GAP + BOX_H + MARGIN
fig = plt.figure(figsize=(FW * MM, FH * MM))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, FW)
ax.set_ylim(0, FH)
ax.set_axis_off()
fig.set_dpi(600)                       # 在高分辨率下量文字宽度，减少字形微调带来的误差
renderer = fig.canvas.get_renderer()
inv = ax.transData.inverted()

HEAD_TOP = FH - MARGIN
ROW1_TOP = HEAD_TOP - HEAD_H - HEAD_GAP


def draw_line(x, y, runs, x_max):
    """画一行富文本（runs = [(文字, 样式, 颜色)]，样式 None/'bold'/'ital'），段间留一个空格宽；检查是否超出 x_max。"""
    for j, (txt, style, col) in enumerate(runs):
        if j > 0:
            x += GAP_ARROW if runs[j - 1][0].endswith('→') else GAPX
        bold = style == 'bold'
        t = ax.text(x, y, txt, fontsize=FS_LEAD if bold else FS_BODY, fontweight='bold' if bold else 'normal',
                    fontstyle='italic' if style == 'ital' else 'normal', color=col or C['TEXT'],
                    ha='left', va='center')
        x = inv.transform((t.get_window_extent(renderer=renderer).x1, 0))[0]
    if x > x_max + 1e-6:
        print(f'  [warn] line overflows by {x - x_max:.2f} mm: {"".join(r[0] for r in runs)}')


RADIUS = 1.6        # 圆角半径（mm）


def rr_vertices(x, y, w, h, r, step=0.3):
    """圆角矩形的稠密顶点（逆时针、闭合），圆角每段约 step mm，直边每段约 step mm，用于逐段着色的渐变边线。"""
    pts = []
    for cx, cy, a0 in ((x + w - r, y + r, -90), (x + w - r, y + h - r, 0), (x + r, y + h - r, 90), (x + r, y + r, 180)):
        a = np.radians(np.linspace(a0, a0 + 90, 13))
        pts.append(np.c_[cx + r * np.cos(a), cy + r * np.sin(a)])
    pts.append(pts[0][:1])
    poly = np.concatenate(pts)
    return np.concatenate([np.linspace(p0, p1, max(2, int(np.hypot(*(p1 - p0)) / step) + 1))[:-1]
                           for p0, p1 in zip(poly[:-1], poly[1:])] + [poly[-1:]])


def rounded(x, y, w, h, fc, ec, ls='-', lw=0.7, z=2):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={RADIUS}', facecolor=fc, edgecolor=ec,
                       linewidth=lw, linestyle=ls, zorder=z)
    ax.add_patch(p)
    return p


def content_box(x, y_top, w, lines, color, color2=None, ls='-', fill_a=0.12):
    """内容框：12% 填色 + 同色边框；color2 给出时做左→右双色渐变（填色与边线都渐变）。返回框底 y。"""
    h = len(lines) * LH + 2 * PADY
    y = y_top - h
    if color2 is None:
        rounded(x, y, w, h, tint(color, fill_a), color, ls=ls)
    else:
        # 渐变：填色用一张横向渐变小图并裁到圆角框内；边线把圆角框路径加密后按 x 逐段着色（无条带接缝）
        c1, c2 = np.array(mpl.colors.to_rgb(color)), np.array(mpl.colors.to_rgb(color2))
        f = np.linspace(0, 1, 512)[:, None]
        grad = np.stack([(1 - 0.12) + 0.12 * ((1 - f) * c1 + f * c2)] * 2, axis=0)   # 2 × 512 × 3
        frame = rounded(x, y, w, h, 'none', 'none', z=1)
        im = ax.imshow(grad, extent=(x, x + w, y, y + h), aspect='auto', interpolation='bilinear', zorder=2)
        im.set_clip_path(frame)
        verts = rr_vertices(x, y, w, h, RADIUS)
        segs = np.stack([verts[:-1], verts[1:]], axis=1)
        fx = ((segs[:, :, 0].mean(axis=1) - x) / w).clip(0, 1)[:, None]
        ax.add_collection(LineCollection(segs, colors=(1 - fx) * c1 + fx * c2, linewidths=0.7,
                                         capstyle='round', joinstyle='round', zorder=3))
        ax.set_xlim(0, FW)
        ax.set_ylim(0, FH)
    for i, runs in enumerate(lines):
        draw_line(x + PADX, y_top - PADY - LH * (i + 0.5), runs, x + w - PADX)
    return y


# ------------------------------------------------------------------ 标题框与粗箭头
heads = ['1 · Representation', '2 · Labels', '3 · Evaluate']
for (x, w), txt in zip(COLS, heads):
    rounded(x, HEAD_TOP - HEAD_H, w, HEAD_H, C['REF_TGT'], 'none')
    ax.text(x + PADX + 0.3, HEAD_TOP - HEAD_H / 2, txt, fontsize=FS_HEAD, fontweight='bold', color='white',
            ha='left', va='center', zorder=4)
for (xa, wa), (xb, _) in zip(COLS[:-1], COLS[1:]):
    ax.add_patch(FancyArrowPatch((xa + wa + 0.9, HEAD_TOP - HEAD_H / 2), (xb - 0.9, HEAD_TOP - HEAD_H / 2),
                                 arrowstyle='simple,head_length=0.6,head_width=1.0,tail_width=0.45',
                                 mutation_scale=10, facecolor='#B3B3B3', edgecolor='none', shrinkA=0, shrinkB=0))

T, M = C['TEXT'], C['MUTED']

# ------------------------------------------------------------------ 列 1：表征
x, w = COLS[0]
content_box(x, ROW1_TOP, w, [
    [('Use AE embeddings, not S2 statistics', 'bold', C['AE'])],
    [(f'{pp(H2_9)} pp BA at matched OA (9 countries),', None, T)],
    [(f'above 400 labels ({pp(RT400)}); all {N_H2}: {pp(H2_11)} pp', None, T)],
], C['AE'])

# ------------------------------------------------------------------ 列 2：标签（分叉成两个框）
x, w = COLS[1]
IND = 5.0                                      # 分叉连线占用的左缩进
yA = content_box(x + IND, ROW1_TOP, w - IND, [
    [('No survey →', 'bold', C['LABEL']), ('k', 'ital', T), (f'random labels: {pp(RT200)} pp for 200,', None, T)],
    [(f'{pp(RT400)} pp for 400 (≤ {RATIO400 * 100:.0f}% of the target-only gap)', None, T)],
    [('No sampling design clearly beat random', None, T)],
], C['LABEL'])
ROW2_TOP = yA - ROW_GAP
yB = content_box(x + IND, ROW2_TOP, w - IND, [
    [('Existing survey →', 'bold', C['CONC']), (f'keep it (100 clustered ≈ {KEQ_S0} random)', None, T)],
    [('Add new labels nationally unless a spread label costs >', None, T)],
    [(f'{BE_LO:.1f}–{BE_HI:.1f}× an extension label; Betti allocation ≈ random', None, T)],
], C['CONC'], color2=C['SPREAD'])
# 分叉连线：从标题框底部下行，分两支进入两个框
xs = x + 2.2
yA_mid = (ROW1_TOP + yA) / 2
yB_mid = (ROW2_TOP + yB) / 2
ax.plot([xs, xs], [HEAD_TOP - HEAD_H, yB_mid], color=C['REF_Z0'], lw=0.6, solid_capstyle='butt', zorder=1)
for ym in (yA_mid, yB_mid):
    ax.add_patch(FancyArrowPatch((xs, ym), (x + IND - 0.1, ym), arrowstyle='-|>', mutation_scale=5,
                                 linewidth=0.6, color=C['REF_Z0'], shrinkA=0, shrinkB=0, zorder=1))

# 列 1 第二行：AE 优势最大的作物与地块（来自补充分析，数字读自 source_data）
_cls = pd.read_csv(ns.src('ae_by_class.csv')).set_index('cls')
_siz = pd.read_csv(ns.src('ae_by_size.csv')).set_index('size_group')
x, w = COLS[0]
content_box(x, ROW2_TOP, w, [
    [('Largest AE gains', 'bold', C['AE'])],
    [(f"orchards {pp(_cls.loc['orchards', 'd11'])}, root crops {pp(_cls.loc['root_crops', 'd11'])} pp recall;", None, T)],
    [(f"large parcels {pp(_siz.loc['≥4', 'd11'])} pp BA", None, T)],
], C['AE'])

# ------------------------------------------------------------------ 列 3：评估
x, w = COLS[2]
content_box(x, ROW1_TOP, w, [
    [('Compare BA at matched OA', 'bold', C['REF_Z0'])],
    [('Fixed-operating-point readings hide most', None, T)],
    [(f'of the label gain ({pp(M1_BA)} vs {pp(H1)} pp)', None, T)],
], C['REF_Z0'])
content_box(x, ROW2_TOP, w, [
    [('Not supported', 'bold', M)],
    [('AE as a guide for choosing', None, T)],
    [('which parcels to label', None, T)],
], M, ls=(0, (3, 1.5)))

ns.save(fig, 'fig7_decision')
print('saved fig7_decision')
