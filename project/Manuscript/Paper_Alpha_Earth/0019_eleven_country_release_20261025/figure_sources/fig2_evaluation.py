"""Fig. 2 为什么要在同 OA 下比较 BA（fig2_evaluation）。

核心句：加标签和换表征都会移动工作点（OA 与 BA 相互交换），固定在 τ = 0 读数会把两根杠杆的 BA 回报读小；
在同 OA 窗口内比较 BA，才能把前沿提升与沿前沿的移动分开。

a｜示例单元（BE_F1，LightGBM，AE 特征）：零标签、+200 随机标签（第 1 次抽样）与只用目标国的 τ 族 OA–BA 曲线；
b｜同一单元：零标签下 AE 与 S2 的 τ 族曲线；
c｜确认运行：两根杠杆的 τ = 0 读数与同 OA 读数（国家为单位，95% t CI）。
只读 source_data/ 下的 CSV，不重算统计量；箭头位置只是几何标注，印出的数字全部来自源文件。
"""
import os

os.environ.setdefault('OMP_NUM_THREADS', '2')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator

import nstyle as ns

ns.setup()
C, FS, MM = ns.C, ns.FS, ns.MM
fmt = ns.fmt_pp

# ------------------------------------------------------------------ 读数据
cur = pd.read_csv(ns.src('example_unit_curves.csv'))
fa = pd.read_csv(ns.src('example_unit_fa.csv'))
dif = pd.read_csv(ns.src('example_unit_diffs.csv'))
est = pd.read_csv(ns.src('confirm_estimates.csv'))

cur = cur[(cur.unit == 'BE_F1') & (cur.learner == 'lgb')]


def curve(feat, arm, r=0):
    """取一条 τ 族曲线，按 τ 排序。"""
    s = cur[(cur.feat == feat) & (cur.arm == arm) & (cur.r == r)].sort_values('tau')
    return s.tau.to_numpy(), s.OA.to_numpy(), s.BA.to_numpy()


def t0(feat, arm, r=0):
    """τ = 0 的工作点 (OA, BA)。"""
    tau, oa, ba = curve(feat, arm, r)
    i = int(np.argmin(np.abs(tau)))
    assert abs(tau[i]) < 1e-9
    return oa[i], ba[i]


def frontier_ba(feat, arm, oa_q, r=0):
    """在前沿分支（OA 最大处之后、OA 随 τ 递减的一段）上按 OA 线性插值 BA，只用于摆放箭头。"""
    tau, oa, ba = curve(feat, arm, r)
    j = int(np.argmax(oa))
    o, b = oa[j:], ba[j:]
    return float(np.interp(oa_q, o[::-1], b[::-1]))


def dval(name, feat, r=0):
    s = dif[(dif.name == name) & (dif.feat == feat) & (dif.r == r)]
    assert len(s) == 1, (name, feat, r)
    return float(s.value.iloc[0])


def e(est_id, feat, st):
    s = est[(est.est_id == est_id) & (est.feat == feat) & (est.set == st)]
    assert len(s) == 1, (est_id, feat, st)
    return s.iloc[0]


# 示例单元的锚点与差值
oa_star_ae = float(fa[(fa.feat == 'ae') & (fa.arm == 'Z0')].OA_star.iloc[0])      # 84.51
oa_star_s2 = float(fa[(fa.feat == 's2') & (fa.arm == 'Z0')].OA_star.iloc[0])      # 81.73
d_rt200 = dval('dFA_RT200', 'ae')                                                  # +3.18
d_z0 = dval('D_Z0', 'ae-s2')                                                       # +6.38
oa_z0, ba_z0 = t0('ae', 'Z0')
oa_rt, ba_rt = t0('ae', 'RT200')
oa_tg, ba_tg = t0('ae', 'TGT')
oa_s2, ba_s2 = t0('s2', 'Z0')
t0_dOA, t0_dBA = oa_rt - oa_z0, ba_rt - ba_z0                                      # +1.66、+1.11
# 与分析输出对账（t0 行）
assert abs(t0_dBA - dval('t0_RT200-Z0_BA', 'ae')) < 1e-9 and abs(t0_dOA - dval('t0_RT200-Z0_OA', 'ae')) < 1e-9

# ------------------------------------------------------------------ 版式（毫米）
W, H = 183, 62
fig = plt.figure(figsize=(W * MM, H * MM))


def ax_mm(l, b, w, h):
    return fig.add_axes([l / W, b / H, w / W, h / H])


BOT, AH = 9.0, 46.5                     # 轴底与轴高（a、b、c 共用）
TOP_LBL = BOT + AH + 3.2                # 面板标号的基线高度


def plabel(ax, lab, left_mm):
    """面板标号放在各面板区域的左上角（毫米定位，保证 a、b、c 同一水平线）。"""
    l, b, w, h = ax.get_position().bounds
    ns.panel(ax, lab, x=(left_mm / W - l) / w, y=(TOP_LBL / H - b) / h)


XL, YL = (70, 92), (35, 77)


def frame_oa_ba(ax):
    ax.set_xlim(*XL)
    ax.set_ylim(*YL)
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.set_xlabel('OA (%)', labelpad=1.5)


def window(ax, lo, hi):
    """同 OA 窗口：浅灰竖带；标签放在轴顶之上，避开曲线。"""
    ax.axvspan(lo, hi, color=C['BAND'], lw=0, zorder=0)
    ax.text((lo + hi) / 2, YL[1] + 0.6, 'matched-OA window', ha='center', va='bottom', fontsize=FS['small'],
            color=C['MUTED'], clip_on=False)


def gap_arrow(ax, x, y0, y1, text, color, tx, ty, ha):
    """两条前沿之间的双向箭头；数值（窗口内逐点差的平均）放在 (tx, ty)。"""
    ax.annotate('', xy=(x, y1), xytext=(x, y0),
                arrowprops=dict(arrowstyle='<|-|>', lw=0.6, color=color, shrinkA=0, shrinkB=0,
                                mutation_scale=4.5), zorder=5)
    ax.text(tx, ty, text, ha=ha, va='center', fontsize=FS['tick'], color=color, fontweight='bold')


def t0_label(ax, xy, xytext):
    """标出 τ = 0 工作点（短引线）。"""
    ax.annotate('τ = 0', xy=xy, xytext=xytext, fontsize=FS['small'], color=C['TEXT'], ha='left', va='bottom',
                arrowprops=dict(arrowstyle='-', lw=0.4, color=C['MUTED'], shrinkA=0.5, shrinkB=2.2))


# ================================================================== a｜标签怎样移动前沿
axa = ax_mm(10.5, BOT, 44.5, AH)
frame_oa_ba(axa)
axa.set_ylabel('BA (%)', labelpad=1.5)
w_lo, w_hi = oa_star_ae - 4, oa_star_ae
window(axa, w_lo, w_hi)

for feat, arm, col, lw, z in (('ae', 'TGT', C['REF_TGT'], 0.8, 2), ('ae', 'Z0', C['REF_Z0'], 1.0, 3),
                              ('ae', 'RT200', C['LABEL'], 1.2, 4)):
    _, oa, ba = curve(feat, arm)
    axa.plot(oa, ba, color=col, lw=lw, zorder=z, solid_capstyle='round')
for (o, b), col in (((oa_z0, ba_z0), C['REF_Z0']), ((oa_rt, ba_rt), C['LABEL']), ((oa_tg, ba_tg), C['REF_TGT'])):
    axa.plot(o, b, 'o', ms=3.2, mfc=col, mec='white', mew=0.5, zorder=6)

# 同 OA 读数：窗口中点处两条前沿之间的箭头；印出的是窗口内逐点差的平均
xm = (w_lo + w_hi) / 2
y0, y1 = frontier_ba('ae', 'Z0', xm), frontier_ba('ae', 'RT200', xm)
gap_arrow(axa, xm, y0, y1, fmt(d_rt200) + '%', C['LABEL'], tx=xm, ty=y1 + 2.0, ha='center')

# τ = 0 读数：标出工作点，并在右下角写两点的 OA 差与 BA 差
t0_label(axa, (oa_rt, ba_rt), (86.55, 56.2))
axa.text(91.7, 36.0, f'at τ = 0:\nΔOA {fmt(t0_dOA)}%\nΔBA {fmt(t0_dBA)}%', fontsize=FS['small'],
         color=C['LABEL'], ha='right', va='bottom', linespacing=1.2)

# 直接标注曲线
axa.text(71.0, 51.6, 'zero-label transfer', fontsize=FS['small'], color=C['REF_Z0'], ha='left', va='top')
axa.text(71.0, 58.3, '+200 random\ntarget labels', fontsize=FS['small'], color=C['LABEL'], ha='left',
         va='bottom', linespacing=1.1)
axa.text(91.7, 76.6, 'target-only\n(~30k labels)', fontsize=FS['small'], color=C['REF_TGT'], ha='right', va='top',
         linespacing=1.1)
plabel(axa, 'a', 0.0)

# ================================================================== b｜表征怎样移动前沿
axb = ax_mm(62.0, BOT, 39.0, AH)
frame_oa_ba(axb)
axb.tick_params(labelleft=False)
w2_lo, w2_hi = min(oa_star_ae, oa_star_s2) - 4, min(oa_star_ae, oa_star_s2)
window(axb, w2_lo, w2_hi)
for feat, col, lw, z in (('s2', C['S2'], 1.0, 3), ('ae', C['AE'], 1.2, 4)):
    _, oa, ba = curve(feat, 'Z0')
    axb.plot(oa, ba, color=col, lw=lw, zorder=z, solid_capstyle='round')
for (o, b), col in (((oa_z0, ba_z0), C['AE']), ((oa_s2, ba_s2), C['S2'])):
    axb.plot(o, b, 'o', ms=3.2, mfc=col, mec='white', mew=0.5, zorder=6)
xm2 = (w2_lo + w2_hi) / 2
y0, y1 = frontier_ba('s2', 'Z0', xm2), frontier_ba('ae', 'Z0', xm2)
gap_arrow(axb, xm2, y0, y1, fmt(d_z0) + '%', C['AE'], tx=xm2 + 0.4, ty=48.4, ha='left')
t0_label(axb, (oa_z0, ba_z0), (86.4, 55.0))
axb.text(71.0, 57.0, 'AE embeddings', fontsize=FS['small'], color=C['AE'], ha='left', va='bottom')
axb.text(71.0, 49.3, 'S2 statistics', fontsize=FS['small'], color=C['S2'], ha='left', va='top')
axb.text(91.7, 76.6, 'illustrative unit\n(Belgium)\nzero-label transfer', fontsize=FS['small'], color=C['MUTED'],
         ha='right', va='top', linespacing=1.1)
plabel(axb, 'b', 58.0)

# ================================================================== c｜两种读法对比（森林图）
axc = ax_mm(131.0, BOT, 24.0, AH)
groups = [
    ('+200 random labels (vs zero-label), 9 countries', C['LABEL'], [
        ('ΔBA at τ = 0', 'M1_RT200-Z0_BA', 'both', 9, False),
        ('ΔOA at τ = 0', 'M1_RT200-Z0_OA', 'both', 9, False),
        ('ΔBA at matched OA', 'H1', 'both', 9, True)]),
    ('AE instead of S2 (zero labels), 11 countries', C['AE'], [
        ('ΔBA at τ = 0', 'M1_Z0_AE-S2_BA', 'ae-s2', 11, False),
        ('ΔOA at τ = 0', 'M1_Z0_AE-S2_OA', 'ae-s2', 11, False),
        ('ΔBA at matched OA', 'H2', 'ae-s2', 11, True)]),
]
XLC = (-1.5, 9.5)
y = 8.6
rows, heads = [], []
for title, col, items in groups:
    heads.append((y, title, col))
    y -= 1.0
    ys = []
    for lab, eid, feat, st, main in items:
        rows.append((y, lab, e(eid, feat, st), col, main))
        ys.append(y)
        y -= 1.0
    # 0 参考线只画在本组的行范围内，避免穿过组标题
    axc.plot([0, 0], [min(ys) - 0.5, max(ys) + 0.5], color=C['MUTED'], lw=0.5, ls=(0, (2, 2)), zorder=0)
    y -= 0.55
axc.set_ylim(y + 0.55 + 0.45, 9.2)
axc.set_xlim(*XLC)
axc.xaxis.set_major_locator(MultipleLocator(2))
axc.set_yticks([])
axc.spines['left'].set_visible(False)
axc.set_xlabel('ΔBA or ΔOA (%)', labelpad=1.5)

mm_x = (XLC[1] - XLC[0]) / 24.0          # 每毫米对应的数据单位
X_LBL = XLC[0] - 1.2 * mm_x               # 行标签右端
X_HEAD = XLC[0] - 21.5 * mm_x             # 组标题左端
X_TXT = XLC[1] + 1.2 * mm_x               # 数值列左端
X_POS = XLC[1] + 22.8 * mm_x              # n_pos/n 列左端
for yy, title, col in heads:
    axc.text(X_HEAD, yy, title, ha='left', va='center', fontsize=FS['tick'], color=col, fontweight='bold',
             clip_on=False)
for yy, lab, r, col, main in rows:
    axc.plot([r['lo95'], r['hi95']], [yy, yy], color=col, lw=0.9 if main else 0.7, solid_capstyle='butt',
             zorder=3)
    axc.plot(r['mean'], yy, 'o', ms=4.2 if main else 3.2, mfc=col if main else 'white', mec=col,
             mew=0.8, zorder=4)
    axc.text(X_LBL, yy, lab, ha='right', va='center', fontsize=FS['tick'], clip_on=False,
             fontweight='bold' if main else 'normal')
    axc.text(X_TXT, yy, f"{fmt(r['mean'])} [{fmt(r['lo95'], signed=False)}, {fmt(r['hi95'], signed=False)}]",
             ha='left', va='center', fontsize=FS['small'], clip_on=False, color=C['TEXT'])
    axc.text(X_POS, yy, f"{int(r['n_pos'])}/{int(r['n'])}", ha='left', va='center',
             fontsize=FS['small'], clip_on=False, color=C['MUTED'])
axc.text(X_TXT, 9.35, 'mean [95% CI]', ha='left', va='bottom', fontsize=FS['small'], color=C['MUTED'],
         clip_on=False)
axc.text(X_POS, 9.35, '> 0', ha='left', va='bottom', fontsize=FS['small'], color=C['MUTED'], clip_on=False)
plabel(axc, 'c', 106.0)

ns.save(fig, 'fig2_evaluation')
print('saved')
