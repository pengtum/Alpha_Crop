"""Fig. 3 RQ1：从零起步，标签能买到什么（fig3_labels_zero）。

核心句：多源迁移之后，50–400 个随机目标标签让同 OA 下的 BA 提高 0.8–3.7 pp，9 个主集国家全部为正；
这是真实的增益，但最多只补上本国全量训练缺口的四分之一。RF 给出同样的答案。

a｜剂量曲线：逐国细线（9 国主集 + RO/PT 灰色虚线）与 9 国均值（95% t CI）；
b｜随机标签补上"只用目标国训练"缺口的比例；
c｜换学习器检验：+200 标签时 LightGBM 与 RF 的逐国值。
只读 source_data/ 下的 CSV，不重算统计量。
"""
import os

os.environ.setdefault('OMP_NUM_THREADS', '2')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, MultipleLocator

import nstyle as ns

ns.setup()
C, FS, MM = ns.C, ns.FS, ns.MM
fmt = ns.fmt_pp

# ------------------------------------------------------------------ 读数据
est = pd.read_csv(ns.src('confirm_estimates.csv'))
cv = pd.read_csv(ns.src('confirm_country_values.csv'))
a2 = pd.read_csv(ns.src('confirm_a2_ratio.csv'))
cc = pd.read_csv(ns.src('country_counts.csv'))

PRIMARY = cc[cc.primary9].country.tolist()          # 9 国主集
EXTRA = cc[~cc.primary9].country.tolist()           # RO、PT
KS = [50, 100, 200, 400]


def e(est_id, feat, st, learner=None):
    s = est[(est.est_id == est_id) & (est.feat == feat) & (est.set == st)]
    if learner is not None:
        s = s[s.learner == learner]
    assert len(s) == 1, (est_id, feat, st)
    return s.iloc[0]


def ci_txt(r, unit=''):
    """"+2.34 pp [1.54, 3.14]" 形式（负号为 U+2212）。"""
    return f"{fmt(r['mean'])}{unit} [{fmt(r['lo95'], signed=False)}, {fmt(r['hi95'], signed=False)}]"


# 逐国值：lgb、both，dFA_RT*；k = 0 处为 0
v = cv[(cv.learner == 'lgb') & (cv.feat == 'both') & (cv.name.isin([f'dFA_RT{k}' for k in KS]))]
dose = v.pivot_table(index='country', columns='name', values='value')[[f'dFA_RT{k}' for k in KS]]
assert set(dose.index) == set(PRIMARY + EXTRA) and dose.notna().all().all()
m9 = {k: e(f'A1_RT{k}', 'both', 9) for k in KS}
m11_400 = e('A1_RT400', 'both', 11)
h1 = e('H1', 'both', 9)
assert abs(h1['mean'] - m9[200]['mean']) < 1e-12            # H1 与 A1_RT200 同一量
tgt = e('A2_TGT', 'both', 9)
rf200 = e('A3_RF_RT200', 'both', 9)
ratio = a2[a2.set == 9].set_index('k')
assert all(abs(ratio.loc[k, 'num'] - m9[k]['mean']) < 1e-9 for k in KS)
assert abs(ratio.loc[50, 'den'] - tgt['mean']) < 1e-9

# LightGBM 与 RF 的逐国值（dFA_RT200、both，11 国）
sc = cv[(cv.feat == 'both') & (cv.name == 'dFA_RT200')].pivot_table(index='country', columns='learner',
                                                                    values='value')
assert len(sc) == 11

# ------------------------------------------------------------------ 版式（毫米）
W, H = 183, 62
fig = plt.figure(figsize=(W * MM, H * MM))


def ax_mm(l, b, w, h):
    return fig.add_axes([l / W, b / H, w / W, h / H])


BOT, AH = 9.0, 46.5
TOP_LBL = BOT + AH + 3.2


def plabel(ax, lab, left_mm):
    l, b, w, h = ax.get_position().bounds
    ns.panel(ax, lab, x=(left_mm / W - l) / w, y=(TOP_LBL / H - b) / h)


GREY = C['MUTED']
DASH = (0, (3, 1.6))

# ================================================================== a｜剂量曲线（主面板）
axa = ax_mm(10.5, BOT, 53.0, AH)
XA = (-12, 432)
axa.set_xlim(*XA)
axa.set_ylim(-7.2, 7.2)
axa.xaxis.set_major_locator(FixedLocator([0] + KS))
axa.yaxis.set_major_locator(MultipleLocator(2))
axa.set_xlabel('Random target labels (k)', labelpad=1.5)
axa.set_ylabel('ΔBA (%)', labelpad=1.5)
ns.zero_line(axa, 'y')
xk = np.array([0] + KS)
for c in PRIMARY:
    axa.plot(xk, np.r_[0, dose.loc[c].to_numpy()], color=C['LABEL'], alpha=0.25, lw=0.6, zorder=2)
for c in EXTRA:
    yv = np.r_[0, dose.loc[c].to_numpy()]
    axa.plot(xk, yv, color=GREY, lw=0.6, ls=DASH, zorder=2)
    axa.text(xk[-1] + 9, yv[-1], c, ha='left', va='center', fontsize=FS['small'], color=GREY)

# 9 国均值 + 95% t CI
mu = np.r_[0, [m9[k]['mean'] for k in KS]]
lo = np.r_[0, [m9[k]['lo95'] for k in KS]]
hi = np.r_[0, [m9[k]['hi95'] for k in KS]]
axa.plot(xk, mu, color=C['LABEL'], lw=1.4, zorder=4, solid_capstyle='round')
axa.vlines(xk[1:], lo[1:], hi[1:], color=C['LABEL'], lw=0.9, zorder=4)
axa.plot(xk, mu, 'o', ms=3.4, mfc=C['LABEL'], mec='white', mew=0.5, zorder=5)

# 左下角的图例键：粗线为 9 国均值，细线为单个国家（RO、PT 已在线末端直接标注）
axa.legend(handles=[Line2D([], [], color=C['LABEL'], lw=1.4, marker='o', ms=3.0, mec='white', mew=0.4),
                    Line2D([], [], color=C['LABEL'], lw=0.6, alpha=0.35)],
           labels=['9-country mean', 'single countries'], loc='lower left',
           bbox_to_anchor=(0.0, 0.0), fontsize=FS['small'], handlelength=1.8, handletextpad=0.5,
           borderaxespad=0.2, labelspacing=0.3)

# k = 200 的确认性检验标注（文字向左错开、引线斜向指到 CI 上端，避免引线与 CI 竖线共线被误读成更长的 CI）
axa.annotate(f"{ci_txt(h1, '%')}, {int(h1['n_pos'])}/{int(h1['n'])}\n(confirmatory, Holm)",
             xy=(197, hi[3] + 0.18), xytext=(118, 5.55), ha='center', va='bottom', fontsize=FS['small'],
             color=C['LABEL'], linespacing=1.15,
             arrowprops=dict(arrowstyle='-', lw=0.4, color=C['MUTED'], shrinkA=1.0, shrinkB=0,
                             relpos=(0.78, 0.0)))

# k = 400 的 11 国均值（灰色，略向右错开）
x11 = 413
axa.vlines(x11, m11_400['lo95'], m11_400['hi95'], color=GREY, lw=0.6, zorder=3)
axa.plot(x11, m11_400['mean'], 'D', ms=2.6, mfc='white', mec=GREY, mew=0.7, zorder=5)
axa.text(x11 + 6, m11_400['lo95'] - 0.35, f"11 countries:\n{ci_txt(m11_400)}", ha='right', va='top',
         fontsize=FS['small'], color=GREY, linespacing=1.15)
plabel(axa, 'a', 0.0)

# ================================================================== b｜补上缺口的比例
axb = ax_mm(90.0, BOT, 32.0, AH)
XB = (0, 20)
axb.set_xlim(*XB)
axb.xaxis.set_major_locator(MultipleLocator(5))
axb.set_xlabel('ΔBA (%)', labelpad=1.5)
Y_TGT, Y_K = 5.1, {50: 3.4, 100: 2.4, 200: 1.4, 400: 0.4}
BH = 0.6
axb.set_ylim(-0.3, 6.0)
# 只用目标国训练：浅填色条 + 95% CI
axb.barh(Y_TGT, tgt['mean'], height=BH, color='#D4D4D4', lw=0, zorder=2)
axb.hlines(Y_TGT, tgt['lo95'], tgt['hi95'], color=C['REF_TGT'], lw=0.7, zorder=3)
axb.text(0.45, Y_TGT, ci_txt(tgt), ha='left', va='center', fontsize=FS['small'], color=C['REF_TGT'], zorder=4)
# 随机标签
X_PCT = 6.2
for k in KS:
    r = m9[k]
    axb.barh(Y_K[k], r['mean'], height=BH, color=C['LABEL'], lw=0, zorder=2)
    axb.hlines(Y_K[k], r['lo95'], r['hi95'], color=C['REF_TGT'], lw=0.7, zorder=3)
    axb.text(X_PCT, Y_K[k], f"{ratio.loc[k, 'ratio'] * 100:.0f}%", ha='left', va='center', fontsize=FS['tick'],
             color=C['LABEL'], fontweight='bold')
axb.text(X_PCT, Y_K[50] + 0.62, 'share of target-only gain', ha='left', va='bottom', fontsize=FS['small'],
         color=GREY)
axb.set_yticks([Y_TGT] + [Y_K[k] for k in KS])
axb.set_yticklabels(['target-only model\n(~30k labels)'] + [f'+{k} random labels' for k in KS],
                    linespacing=1.1)
axb.tick_params(axis='y', length=0, pad=2)
axb.spines['left'].set_visible(False)
plabel(axb, 'b', 68.0)

# ================================================================== c｜换学习器检验
CW = 39.5                                           # c 轴宽（毫米）；右移使整图铺满 183 mm
axc = ax_mm(141.5, BOT, CW, AH)
XC = (-6.5, 5.0)
span = (XC[1] - XC[0]) * AH / CW                     # 等比例：y 跨度按轴的高宽比换算
YC = (7.4 - span, 7.4)                               # 上端固定在 7.4，给角落文字留出与 SE 标签之间的间隙
axc.set_xlim(*XC)
axc.set_ylim(*YC)
axc.set_aspect('equal', adjustable='box')
axc.xaxis.set_major_locator(MultipleLocator(2))
axc.yaxis.set_major_locator(MultipleLocator(2))
axc.plot([XC[0], XC[1]], [XC[0], XC[1]], color=GREY, lw=0.6, ls=(0, (3, 2)), zorder=1)
axc.text(-3.25, -4.3, 'y = x', ha='center', va='center', fontsize=FS['small'], color=GREY, rotation=45,
         rotation_mode='anchor')
ns.zero_line(axc, 'y')
# 竖向 0 参考线只画到角落文字下方
axc.plot([0, 0], [YC[0], 4.2], color=GREY, lw=0.5, ls=(0, (2, 2)), zorder=0)
# 轴标签沿用全文术语 "ΔBA at matched OA"
axc.set_xlabel('LightGBM ΔBA (%)', labelpad=1.5)
axc.set_ylabel('Random forest ΔBA (%)', labelpad=1.5)
# 国家代码的摆放（dx, dy 以数据单位计；ha, va）
OFF = dict(AT=(-0.22, 0.0, 'right', 'center'), SI=(-0.22, 0.0, 'right', 'center'),
           SE=(0.0, 0.3, 'center', 'bottom'), SK=(0.25, 0.0, 'left', 'center'),
           BE=(0.24, 0.0, 'left', 'center'), LT=(-0.05, -0.3, 'center', 'top'),
           DE=(0.24, 0.0, 'left', 'center'), EE=(0.05, -0.3, 'center', 'top'),
           LV=(-0.12, 0.28, 'right', 'bottom'), PT=(0.4, 0.0, 'left', 'center'),
           RO=(0.4, 0.0, 'left', 'center'))          # 空心点半径更大，代码再右移一点，避免贴住标记
for c, row in sc.iterrows():
    main = c in PRIMARY
    axc.plot(row['lgb'], row['rf'], 'o', ms=3.4, mfc=C['LABEL'] if main else 'white',
             mec='white' if main else GREY, mew=0.5 if main else 0.8, zorder=4)
    dx, dy, ha, va = OFF[c]
    axc.text(row['lgb'] + dx, row['rf'] + dy, c, ha=ha, va=va, fontsize=FS['small'],
             color=C['TEXT'] if main else GREY, zorder=5)
# 角落：两种学习器的 9 国均值
yt = YC[1] - 0.3
# 小标题写明是 9 国均值（图中画的是 11 国的点，避免读成"本面板只有 9 国"）
axc.text(XC[0] + 0.3, yt, '+200 random labels; 9-country mean', ha='left', va='top', fontsize=FS['small'],
         color=GREY)
for i, (name, r) in enumerate((('LightGBM', h1), ('RF', rf200))):
    yy = yt - 1.05 * (i + 1)
    axc.text(XC[0] + 0.3, yy, name, ha='left', va='top', fontsize=FS['small'], color=C['LABEL'],
             fontweight='bold')
    axc.text(XC[0] + 3.35, yy, f"{ci_txt(r)}, {int(r['n_pos'])}/{int(r['n'])}", ha='left', va='top',
             fontsize=FS['small'], color=C['LABEL'])
plabel(axc, 'c', 129.0)

ns.save(fig, 'fig3_labels_zero')
print('saved')
