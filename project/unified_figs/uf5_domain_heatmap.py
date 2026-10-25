"""UF5 逐域热图 uf5_domain_heatmap_{task}（三个任务同一段代码循环生成）。

核心：把各任务层估计背后的逐域值摊开，看方向是否在域间一致。
行分 4 组（随机标签、AE 替换 S2、交互、τ = 0 读数），列为主集域；作物在主集后空一列放非主集域（us.others：RO、PT）。
数据只读 source_data/{task}_domain_values.csv（经 us.dom）与 tasks_meta.json；不重算统计量。

版式：整张图用一个以 mm 为单位的坐标轴（y 向下），行高、字号、行标签区、色标与组标题位置三任务完全相同。
成图宽度按第二轮统一规则固定：主集域数 ≥ us.CI_MIN_N 的任务（作物）为 us.W2，其余（LCZ、LUCAS）为 us.W15；
单元格宽 = 热图可用宽 / 有效列数（数据驱动）。
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb
from matplotlib.patches import Rectangle

import ustyle as us

us.setup()
C, FS, MM, T, D = us.C, us.FS, us.MM, us.TERM, us.DESIGN

# ------------------------------------------------------------ 行定义：(name, feat, 行标签)
# 第三轮补充：REF 行名一律用 META['ref_short']（长形式只进图例文字）；组标题已写 "AE instead of S2" 时，
# 组内零标签行写 "0 labels"，单独出现（τ = 0 读数组）时写 TERM['ae0']
REF = '__ref__'


def rk(k):
    return T['rk'].format(k=k)


GROUPS = [
    ('Random labels', 'gain over zero-label transfer (mean of AE and S2)',
     [(f'dFA_RT{k}', 'both', rk(k)) for k in us.KS] + [('dFA_REF', 'both', REF)]),
    ('AE instead of S2', 'AE − S2 with the same labels', [
        ('D_Z0', 'ae-s2', '0 labels'),
        ('D_RT200', 'ae-s2', rk(200)),
        ('D_RT400', 'ae-s2', rk(400)),
    ]),
    ('Interaction', T['inter'].replace(' (pp)', ''), [
        ('dFA_RT200', 'inter', rk(200)),
        ('dFA_RT400', 'inter', rk(400)),
        ('dFA_REF', 'inter', REF),
    ]),
    ('Readings at τ = 0', 'no OA matching', [
        ('t0_RT200-Z0_BA', 'both', rk(200)),
        ('t0_Z0_AE-S2_BA', 'ae-s2', T['ae0']),
    ]),
]

# ------------------------------------------------------------ 颜色：DOWN — #F7F7F7 — LABEL，截断 ±8
VMAX = 8.0
CMAP = LinearSegmentedColormap.from_list('uf5_div', [C['DOWN'], '#F7F7F7', C['LABEL']], N=256)
NORM = Normalize(-VMAX, VMAX, clip=True)

# ------------------------------------------------------------ 版式常数（mm）：三任务相同
FIG_H = 85.0
X_LAB = 1.0            # 组标题左端
X_HM = 50.0            # 热图左端（行标签右对齐于 X_HM − 1.4）
RIGHT = 22.0           # 热图右侧留给色标的宽度
GAP_COL = 0.45         # 主集与非主集之间的空列（以单元格宽计）
RH = 3.9               # 行高
GH = 1.5               # 组标题占用的行高倍数
TOP = 10.0             # 顶部列标题区高度
PAD_MM = 2 * 0.02 * 25.4   # us.save 导出时 bbox 两侧各加 pad 0.02 in，画布相应收窄，成图宽度恰为目标栏宽


def fig_width(task):
    """第二轮统一规则 7：n < CI_MIN_N 的任务固定 1.5 栏，否则双栏（mm，已扣除导出 pad）。"""
    w = us.W2 if len(us.primary(task)) >= us.CI_MIN_N else us.W15
    return w / MM - PAD_MM


def fmt1(v):
    """1 位小数（负号 U+2212，经 us.fmt_pp）；四舍五入为 0 的值不带符号。"""
    s = us.fmt_pp(v, signed=False, nd=1)
    return '0.0' if s == '−0.0' else s


def text_colour(v):
    r, g, b = to_rgb(CMAP(NORM(v)))
    return 'white' if 0.299 * r + 0.587 * g + 0.114 * b < 0.5 else C['TEXT']


def draw(task):
    m = us.META[task]
    prim, rest = list(us.primary(task)), us.others(task)
    cols = prim + rest
    col_x = np.r_[np.arange(len(prim)), (len(prim) + GAP_COL + np.arange(len(rest))) if rest else []]
    n_eff = col_x[-1] + 1
    fig_w = fig_width(task)
    cw = (fig_w - X_HM - RIGHT) / n_eff
    hm_w = n_eff * cw

    fig = plt.figure(figsize=(fig_w * MM, FIG_H * MM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, fig_w)
    ax.set_ylim(FIG_H, 0)
    ax.axis('off')

    # ---- 列标题：域短码 us.dlabel（非主集灰色）+ 跨列说明
    y_code = TOP - 1.2
    for j, d in enumerate(cols):
        ax.text(X_HM + (col_x[j] + 0.5) * cw, y_code, us.dlabel(d), ha='center', va='bottom',
                fontsize=FS['tick'], color=C['TEXT'] if d in prim else C['MUTED'])
    y_span = 3.2

    def span(j0, j1, txt):
        x0, x1 = X_HM + (col_x[j0] + 0.15) * cw, X_HM + (col_x[j1] + 0.85) * cw
        ax.plot([x0, x1], [y_span + 0.9] * 2, color=C['MUTED'], lw=0.5, solid_capstyle='butt')
        ax.text((x0 + x1) / 2, y_span + 0.2, txt, ha='center', va='bottom', fontsize=FS['small'], color=C['MUTED'])

    span(0, len(prim) - 1, f"{T['primary'].capitalize()} ({len(prim)} {m['domains']})")
    if rest:
        span(len(prim), len(cols) - 1, f"Not in {T['primary']}")

    # ---- 热图主体
    y = TOP
    printed, clipped = [], []
    for gname, gdesc, rows in GROUPS:
        yb = y + GH * RH - 0.55
        t = ax.text(X_LAB, yb, gname, ha='left', va='bottom', fontsize=FS['label'], fontweight='bold',
                    color=C['TEXT'])
        ax.annotate(gdesc, xy=(1, 0), xycoords=t, xytext=(4, 0), textcoords='offset points',
                    ha='left', va='bottom', fontsize=FS['small'], color=C['MUTED'])
        y += GH * RH
        for name, feat, lab in rows:
            lab = m['ref_short'] if lab == REF else lab
            s_ = us.dom(task, name, feat)
            assert set(cols) <= set(s_.index), (task, name, feat)
            vals = s_.reindex(cols).to_numpy(float)
            yc = y + RH / 2
            ax.text(X_HM - 1.4, yc, lab, ha='right', va='center', fontsize=FS['tick'], color=C['TEXT'])
            for j, v in enumerate(vals):
                x0 = X_HM + col_x[j] * cw
                ax.add_patch(Rectangle((x0, y), cw, RH, facecolor=CMAP(NORM(v)), edgecolor='white', lw=0.6,
                                       zorder=1))
                tc = text_colour(v)
                s = fmt1(v)
                ax.text(x0 + cw / 2, yc + 0.08, s, ha='center', va='center', fontsize=FS['small'], color=tc,
                        zorder=3)
                printed.append((gname, name, feat, lab, cols[j], v, s))
                if abs(v) > VMAX:      # 超出色域：右上角小三角，正 ▲ 负 ▼
                    ax.plot(x0 + cw - 1.05, y + 0.95, marker='^' if v > 0 else 'v', ms=2.1, mew=0, color=tc,
                            zorder=3)
                    clipped.append((name, feat, cols[j], s))
            y += RH
    body_bottom = y

    # ---- 色标（竖向，右侧，顶端与第一组首行对齐）
    cb_x = X_HM + hm_w + 5.0
    cb_top = TOP + GH * RH
    cb_h = 36.0
    cax = fig.add_axes([cb_x / fig_w, 1 - (cb_top + cb_h) / FIG_H, 2.2 / fig_w, cb_h / FIG_H])
    cb = fig.colorbar(ScalarMappable(norm=NORM, cmap=CMAP), cax=cax, orientation='vertical', extend='both',
                      extendfrac=0.06, ticks=[-8, -4, 0, 4, 8])
    cb.ax.set_yticklabels(['−8', '−4', '0', '4', '8'])
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(width=0.5, length=2, labelsize=FS['tick'], pad=1.5)
    cb.set_label('Difference in BA (pp)', fontsize=FS['label'], labelpad=3)

    note_y = cb_top + cb_h + 5.5
    ax.plot(cb_x + 1.1, note_y, marker='^', ms=2.6, mew=0, color=C['LABEL'])
    ax.plot(cb_x + 1.1, note_y + 3.0, marker='v', ms=2.6, mew=0, color=C['DOWN'])
    ax.text(cb_x + 3.0, note_y, 'beyond +8', ha='left', va='center', fontsize=FS['small'], color=C['TEXT'])
    ax.text(cb_x + 3.0, note_y + 3.0, 'beyond −8', ha='left', va='center', fontsize=FS['small'], color=C['TEXT'])

    us.save(fig, f'uf5_domain_heatmap_{task}')
    print(f'== {task}: 画布 {fig_w:.1f} × {FIG_H:.1f} mm（成图约 {fig_w + PAD_MM:.1f} mm 宽），'
          f'单元格宽 {cw:.2f} mm，主体底部 {body_bottom:.1f} mm')
    print('   超色域格子:', clipped)
    return printed, clipped, prim, rest


# ------------------------------------------------------------ 图例文字（英文，投稿用）
NUM = {1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six', 7: 'seven', 8: 'eight', 9: 'nine'}


def legend_text(task, prim, rest, clipped):
    m = us.META[task]
    ur = us.read(task, 'unit_rows')
    n_units = int(ur.groupby('domain').unit.nunique().max())       # 作物每国两个半区，其余 1
    n_draws = int(ur[ur.name == 'dFA_RT200'].r.nunique())          # 随机标签臂的抽样次数
    dw, dws = m['domain'], m['domains']
    n_prim, n_all = len(prim), len(prim) + len(rest)
    ref_desc = m['ref'].replace(' (', ', ').replace(')', '')     # 去掉嵌套括号
    n_cells = n_all * sum(len(g[2]) for g in GROUPS)
    ks = ', '.join(str(k) for k in us.KS)

    if n_units > 1:
        avg = (f"Each AE or S2 {dw} value averages the {NUM[n_units]} spatial halves of the {dw}'s test design "
               f"and, for random-label arms, {NUM[n_draws]} random draws per half; zero-label transfer and the "
               f"reference arm are single runs per half.")
    else:
        avg = (f"Each AE or S2 {dw} value averages {NUM[n_draws]} random draws for random-label arms; zero-label "
               f"transfer and the reference arm are single runs.")

    # 列：短码与全名对照（us.dlabel ↔ us.dname）
    codes = [d for d in prim + rest if us.dlabel(d) != us.dname(task, d)]
    code_txt = (' Column codes: ' + '; '.join(f'{us.dlabel(d)}, {us.dname(task, d)}' for d in prim + rest) + '.'
                if codes else '')
    if rest:
        ar = us.read(task, 'arm_rows')
        st = ar[(ar.arm == 'Z0') & ar.domain.isin(rest)].groupby(['domain', 'feat']).OA_star.mean()
        cols_txt = (f"Columns: the {n_prim} {dws} of the {T['primary']} and, after a gap, "
                    f"{' and '.join(rest)} (grey codes), which are not in the {T['primary']} because their "
                    f"zero-label OA★ is only about {st.min():.0f}–{st.max():.0f}%; n = {n_all} {dws}, one value "
                    f"each.{code_txt}")
    else:
        names = ', '.join(us.dname(task, d) for d in prim)
        cols_txt = (f"Columns: the {NUM.get(n_prim, n_prim)} target {dws}" + ('' if codes else f" ({names})")
                    + f"; n = {n_prim}, one value each.{code_txt}")

    rule = (f"No intervals are shown here. In the other figures, task-level means over the {T['primary']} carry "
            f"95% t confidence intervals (90% for the interaction) when the {T['primary']} has at least "
            f"{us.CI_MIN_N} domains, and the single domain values are shown otherwise.")
    if n_prim >= us.CI_MIN_N:
        rule += f" Here the {T['primary']} has {n_prim} {dws}."
    else:
        rule += (f" Here the {T['primary']} has {NUM.get(n_prim, n_prim)} {dws}, so the other figures show "
                 f"the single {dw} values instead of intervals.")

    clip_txt = f" Here, {len(clipped)} of the {n_cells} cells lie beyond ±8 pp." if clipped else ''

    txt = f"""
{m['long']}: per-{dw} values behind the task-level estimates.

Each cell is the value for one target {dw} (column) of one quantity (row), in percentage points (pp). Unless stated otherwise, BA is read at matched OA, separately for each target unit and feature set: prior-shift adjustments follow the {D['tau']}, and BA is averaged over the {D['window']}. Classifier: LightGBM. Features: {D['ae_desc']} or {D['s2_desc']}; {m['support']}. {m['pool']}. {m['test']}.

Rows, by group:
• Random labels: ΔBA at matched OA from adding k random target labels to the source pool (k = {ks}; target labels weighted {D['target_weight']}) and from the reference arm ({ref_desc}), each relative to zero-label transfer with the same features. Each {dw} value is the mean of its AE and S2 values. The reference-arm rows, here and under Interaction, are labelled "{m['ref_short']}".
• AE instead of S2: difference in BA at matched OA between AE and S2 models trained on the same labels, with the window anchored at min(OA★_AE, OA★_S2): with 0 target labels (zero-label transfer) and with +200 or +400 random target labels.
• Interaction: AE − S2 difference in label gain, i.e. the label gain (ΔBA at matched OA relative to zero-label transfer with the same features) obtained with AE minus that obtained with S2, for +200 and +400 random target labels and for the reference arm.
• Readings at τ = 0: the same contrasts read without prior-shift adjustment (τ = 0) and without OA matching: the change in BA from +200 random target labels (mean of AE and S2) and the AE − S2 difference in BA with 0 target labels.

{avg}

{cols_txt} {rule}

Colour: diverging scale from red (negative) through near-white (zero) to blue (positive), truncated at ±8 pp. A small ▲ (▼) in the upper-right corner of a cell marks a value above +8 pp (below −8 pp), whose colour is therefore saturated; the printed number is the actual value.{clip_txt} Printed values are rounded to 1 decimal place; values that round to 0.0 are printed without a sign.
"""
    return txt


# ------------------------------------------------------------ 主程序：三个任务同一段代码
if __name__ == '__main__':
    for task in us.TASKS:
        printed, clipped, prim, rest = draw(task)
        us.write_legend(f'uf5_domain_heatmap_{task}', legend_text(task, prim, rest, clipped))
        # 供核对：按图中行列顺序打印每个数字
        df = pd.DataFrame(printed, columns=['group', 'name', 'feat', 'label', 'domain', 'value', 'shown'])
        for (g, n, f), r in df.groupby(['group', 'name', 'feat'], sort=False):
            print(f'   {g:18s} {n:16s} {f:6s}', ' '.join(f'{d}={s}' for d, s in zip(r.domain, r.shown)))
