"""UF6 AE 优势来自哪里（按类、按样本面积；事后描述）。三个任务由同一段代码循环生成。

尺寸：crop、lucas 为双栏 183 × 62 mm（a 按类 + b 按面积）；lcz 没有面积数据，为 1.5 栏 us.W15 × 70 mm，只有 a。
a｜按类：class_summary 的主集均值；各域值 = class_rows 按 (domain, cls_name) 对 d_mid 求平均（作物两个半区）。
   us.forest(cols=True) 绘制（列头基线 = 轴顶），右上角角注 us.forest_note（固定在列头上方一行）。
b｜按面积：size_summary 的主集均值；各域值 = size_rows 按 (domain, grp_lab) 对 d_mid 求平均。
   竖向版本的 forest（局部 vforest，规则与样式同 us.forest）；x 轴下方的数值行与 forest 的数值列同一写法
   （mean、[95% CI]、> 0），另加各组样本占比；角注同样用 us.forest_note。
a、b 都是跨域汇总（契约第三轮）：作物的 RO、PT 不画，只在图例中说明不在主集、未计入。

只读 source_data/：{task}_class_summary、{task}_class_rows、{task}_size_summary、{task}_size_rows、tasks_meta.json。
运行：cd unified_figs && PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 ~/AlphaCrops_local/.venv/bin/python uf6_ae_drivers.py
"""
import os
import re

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import ustyle as us

us.setup()
C, FS, MM, META, TERM, DESIGN = us.C, us.FS, us.MM, us.META, us.TERM, us.DESIGN
FIG = 'uf6_ae_drivers'

# 类名显示：下划线换空格、首字母大写；个别特例
CLS_DISPLAY = {'tree_wood_forest': 'tree/wood/forest'}
# 面积分组的后果说明（图例用）：取样支撑本身取自 META['support']，这里只写它对面积分组意味着什么
AREA_CONSEQ = {
    'crop': 'parcels smaller than one unit may therefore include pixels of neighbouring fields',
    'lucas': 'the unit is kept only to make the size groups comparable across tasks',
}
TOP_PAD = 6.5          # 轴顶到图顶（mm），放列头、角注与面板标号
A_END = 101.0          # 有面板 b 时，面板 a（含右侧数值列）的右缘（mm）
B_X0, B_X1 = 114.0, 182.0   # 面板 b 的轴左右缘（mm）；每组约 13 mm，容得下 "[−5.56, −1.14]"
ROW_PT = 6.5           # 面板 b x 轴下方数值行的行距（pt）


def cls_label(c):
    s = CLS_DISPLAY.get(c, c.replace('_', ' '))
    return s[0].upper() + s[1:]


def has_size(task):
    return os.path.exists(us.src(f'{task}_size_summary.csv'))


def domain_table(task, what, key):
    """summary 与逐域值（行 = 组，列 = 域）；并核对主集均值、域数、>0 计数与 summary 一致。"""
    s = us.read(task, f'{what}_summary')
    r = us.read(task, f'{what}_rows')
    dv = r.groupby(['domain', key]).d_mid.mean().unstack('domain')
    prim = us.primary(task)
    for x in s.itertuples():
        v = dv.loc[getattr(x, key)].reindex(prim) if getattr(x, key) in dv.index else None
        v = np.array([], float) if v is None else v.values[np.isfinite(v.values)]
        assert len(v) == x.n, (task, what, getattr(x, key))
        if x.n:
            assert abs(v.mean() - x.mean) < 1e-9 and int((v > 0).sum()) == x.n_pos, (task, what, getattr(x, key))
    return s, dv


def est_of(x):
    """summary 行 → us.forest 所需的估计 dict（summary 只有 95% 区间，90% 补 NaN）。"""
    return dict(mean=x.mean, lo95=x.lo95, hi95=x.hi95, lo90=np.nan, hi90=np.nan, n=int(x.n), n_pos=int(x.n_pos))


def mean_ci(e):
    """数值列的文字（与 us.forest 一致，取自 us.ci_text 去掉计数部分）：('+7.84', '[4.53, 11.15]') 或 ('+0.88', '')。"""
    s = us.ci_text(e).rsplit(', ', 1)[0]
    m, _, ci = s.partition(' ')
    return m, ci


def text_mm(fig, s, size):
    """文字宽度（mm），用于按最长行标签与数值列确定轴的左右缘。"""
    t = fig.text(0, 0, s, fontsize=size)
    w = t.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi * 25.4
    t.remove()
    return w


def nice_lim(vals, pad=0.06):
    v = np.asarray(vals, float)
    v = v[np.isfinite(v)]
    lo, hi = min(v.min(), 0.0), max(v.max(), 0.0)
    span = hi - lo
    return lo - pad * span, hi + pad * span


def drawn_vals(s, dv, key, prim):
    """图上实际画出的所有值（均值、n ≥ 5 的区间、n < 5 的主集域值），用于定坐标范围。"""
    vals = [0.0]
    for x in s.itertuples():
        if x.n == 0:
            continue
        vals.append(x.mean)
        if x.n >= us.CI_MIN_N and np.isfinite(x.lo95):
            vals += [x.lo95, x.hi95]
        else:
            vals += list(dv.loc[getattr(x, key)].reindex(prim).values)
    return vals


def vforest(ax, x, e, dv, color, ms=3.2):
    """竖向点估计（与 us.forest 同一规则与样式）：域数 ≥ CI_MIN_N 画 95% t CI；否则画各域的空心点与细连线，
    各域点画在均值之上。"""
    if e['n'] >= us.CI_MIN_N and np.isfinite(e['lo95']):
        ax.plot([x, x], [e['lo95'], e['hi95']], color=color, lw=0.9, solid_capstyle='butt')
    else:
        dv = np.asarray(dv, float)
        dv = dv[np.isfinite(dv)]
        ax.plot([x, x], [dv.min(), dv.max()], color=color, lw=0.5, alpha=0.5)
        ax.plot(np.full(len(dv), x), dv, 'o', ms=ms * 0.6, color=color, mfc='white', mew=0.6, zorder=4)
    ax.plot(x, e['mean'], 'o', ms=ms, color=color, mfc=color, mew=0.8, zorder=3)


# ================================================================== 绘图（三个任务同一段代码）
def draw(task):
    m, prim = META[task], us.primary(task)
    size = has_size(task)
    FW, FH = (us.W2 / MM, 62.0) if size else (us.W15 / MM, 70.0)
    ss = sdv = None
    if size:
        ss, sdv = domain_table(task, 'size', 'grp_lab')
        ss = ss.sort_values('order').reset_index(drop=True)
        b_rows = [('mean', C['AE'], [mean_ci(est_of(x))[0] if x.n else '–' for x in ss.itertuples()])]
        if (ss.n >= us.CI_MIN_N).any():                     # 与 forest 列头 "mean [95% CI]" 对应，区间单列一行
            b_rows.append(('[95% CI]', C['AE'], [mean_ci(est_of(x))[1] if x.n else '' for x in ss.itertuples()]))
        b_rows += [('> 0', C['MUTED'], [f'{int(k)}/{int(n)}' if n else '–' for k, n in zip(ss.n_pos, ss.n)]),
                   (f'share of {m["samples_short"]}', C['MUTED'], [f'{v * 100:.1f}%' for v in ss.share])]
        # 轴底（mm）：x 刻度标签 + 数值行 + 轴标签
        Y0 = 16.5 + (len(b_rows) - 3) * ROW_PT / 72 * 25.4
    else:
        Y0 = 10.0
    AX_H = FH - Y0 - TOP_PAD
    fig = plt.figure(figsize=(FW * MM, FH * MM))

    def add_ax(x0, w):
        return fig.add_axes([x0 / FW, Y0 / FH, w / FW, AX_H / FH])

    def plabel(ax, letter, x_mm):
        p = ax.get_position()
        us.panel(ax, letter, x=(x_mm / FW - p.x0) / p.width, y=((FH - 3.2) / FH - p.y0) / p.height)

    # ------------------------------------------------------------ a｜按类（us.forest）
    cs, cdv = domain_table(task, 'class', 'cls_name')
    cs = cs.sort_values('mean', ascending=False).reset_index(drop=True)
    labels = [f'{cls_label(c)} ({int(n)})' for c, n in zip(cs.cls_name, cs.n)]
    # 只在 1 个域中评估的类：均值就是该域的值，不再叠画域点（否则实心点上叠白心小圈，像另一种符号）。
    # 跨域汇总面板：不带 others（作物 RO、PT 不画）
    rows = [dict(label=lab, e=est_of(x), color=C['AE'],
                 dv=cdv.loc[x.cls_name].reindex(prim).values if x.n > 1 else np.array([], float))
            for lab, x in zip(labels, cs.itertuples())]
    # 右侧数值列的宽度按实际文字定：列间隔 G1、G2（mm）
    G1, G2 = 2.0, 2.5
    hdr = 'mean' + (' [95% CI]' if rows[0]['e']['n'] >= us.CI_MIN_N else '')
    w_mean = max([text_mm(fig, ' '.join(t for t in mean_ci(r['e']) if t), FS['small']) for r in rows]
                 + [text_mm(fig, hdr, FS['small'])])
    w_cnt = max([text_mm(fig, f"{r['e']['n_pos']}/{r['e']['n']}", FS['small']) for r in rows]
                + [text_mm(fig, '> 0', FS['small'])])
    a_x0 = 1.0 + max(text_mm(fig, s, FS['tick']) for s in labels) + 2.0
    a_x1 = (A_END if size else FW - 1.0) - (G1 + w_mean + G2 + w_cnt)
    aw = a_x1 - a_x0
    ax = add_ax(a_x0, aw)
    us.forest(ax, rows, level=95, xlabel='AE − S2 recall at matched OA (pp)', cols=True,
              col_x=(1 + G1 / aw, 1 + (G1 + w_mean + G2) / aw))
    us.forest_note(ax, task, 95)
    # 横轴范围：只按图上实际画出的量（均值、n ≥ 5 的区间、n < 5 的主集域值）
    ax.set_xlim(*nice_lim(drawn_vals(cs, cdv, 'cls_name', prim)))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=7, steps=[1, 2, 2.5, 5, 10]))
    # 行标签的列头（与 forest 数值列头同一基线 = 轴顶）：括号内数字 = 评估到该类的域数
    ax.annotate(f'Class ({m["domains"]})', (0, 1.0), xycoords='axes fraction', xytext=(-3.5, 0),
                textcoords='offset points', ha='right', va='bottom', fontsize=FS['small'], color=C['MUTED'],
                annotation_clip=False)
    plabel(ax, 'a', 1.0)

    # ------------------------------------------------------------ b｜按样本面积（crop、lucas；竖向 forest）
    if size:
        bx = add_ax(B_X0, B_X1 - B_X0)
        xs = np.arange(len(ss))
        XL = (-0.6, len(ss) - 0.4)
        bx.set_xlim(*XL)
        bx.set_ylim(*nice_lim(drawn_vals(ss, sdv, 'grp_lab', prim), pad=0.08))
        bx.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
        us.zero_line(bx, axis='y')
        for xi, x in zip(xs, ss.itertuples()):
            if x.n == 0:                                  # 该组没有样本（LUCAS 没有 ≥ 2 单位的多边形）
                bx.text(xi, 0, 'none', ha='center', va='center', fontsize=FS['small'], color=C['MUTED'],
                        bbox=dict(boxstyle='square,pad=0.1', fc='white', ec='none'), zorder=4)
                continue
            vforest(bx, xi, est_of(x), sdv.loc[x.grp_lab].reindex(prim).values, C['AE'])
        bx.set_xticks(xs)
        bx.set_xticklabels(ss.grp_lab)
        bx.tick_params(axis='x', length=0)
        # x 轴下方的数值行（与 forest 数值列同一写法），行头在轴左侧
        for i, (head, col, txt) in enumerate(b_rows):
            dy = -12.0 - i * ROW_PT
            for xi, t in zip(xs, txt):
                bx.annotate(t, (xi, 0), xycoords=('data', 'axes fraction'), xytext=(0, dy),
                            textcoords='offset points', ha='center', va='top', fontsize=FS['small'], color=col)
            bx.annotate(head, (XL[0], 0), xycoords=('data', 'axes fraction'), xytext=(-2, dy),
                        textcoords='offset points', ha='right', va='top', fontsize=FS['small'], color=C['MUTED'],
                        annotation_clip=False)
        bx.set_xlabel(f'{m["sample"].capitalize()} area (multiples of π·30² m² ≈ 0.28 ha)',
                      labelpad=9.0 + (len(b_rows) - 1) * ROW_PT + 1.0)
        bx.set_ylabel(TERM['d_ba'])
        us.forest_note(bx, task, 95)
        plabel(bx, 'b', A_END + 3.0)

    us.save(fig, f'{FIG}_{task}')
    us.write_legend(f'{FIG}_{task}', legend(task, cs, ss))
    return cs, ss


# ================================================================== 图例文字（英文，投稿用）
def legend(task, cs, ss):
    m, prim, oth = META[task], us.primary(task), us.others(task)
    D, Ds, smp = m['domain'], m['domains'], m['samples_short']
    n_dom = len(prim)
    what = 'class or size group' if ss is not None else 'class'
    k_units = us.read(task, 'class_rows').groupby('domain').unit.nunique().max()
    unit_txt = (f'Values from the {k_units} spatial halves of each {D} (the test units) are averaged, and the {D} is '
                f'the unit of analysis. ' if k_units > 1 else f'Each {D} is one test unit and the unit of analysis. ')
    other_n = re.search(r'other \d+ \w+', m['pool']).group(0)
    pool = f'{"foreign " if m["pool"].lower().startswith("foreign") else ""}source pool ({other_n})'
    any_ci = bool((cs.n >= us.CI_MIN_N).any() or (ss is not None and (ss.n >= us.CI_MIN_N).any()))
    if any_ci:
        stat = (f'Filled points, means across {Ds}. Where a {what} is evaluated in at least {us.CI_MIN_N} {Ds}, '
                f'bars are 95% t confidence intervals across {Ds}; where it is evaluated in fewer, no interval is '
                f'drawn and open circles show the individual {D} values, with a thin line spanning their range. ')
    else:
        stat = (f'With {n_dom} {Ds}, no confidence interval is drawn: filled points are means across the {Ds} in '
                f'which the {what} is evaluated, open circles are the individual {D} values and the thin line spans '
                f'their range. ')
    if (cs.n == 1).any():
        stat += f'A class evaluated in only one {D} is shown by that {D}\'s value alone (filled point). '
    stat += (f'"> 0" gives the number of {Ds} with AE − S2 > 0 over the number of {Ds} in which the {what} is '
             f'evaluated.')
    if oth:                                          # 跨域汇总面板不画非主集域，只说明未计入
        names = ' and '.join(f'{us.dname(task, d)} ({us.dlabel(d)})' for d in oth)
        stat += f' {names} are not in the {TERM["primary"]} and are not included.'
    head = f'Where the AE advantage comes from — {m["title"]}. Post hoc and descriptive.\n\n'
    body = (f'{m["long"]}; {TERM["primary"]} of {n_dom} target {Ds} '
            f'({", ".join(us.dname(task, d) for d in prim)}). '
            f'{"Both panels break" if ss is not None else "The panel breaks"} down the "{TERM["ae0"]}" comparison. '
            f'{TERM["zero"].capitalize()} is a model trained only on the {pool}; built on {DESIGN["ae_desc"]}, '
            f'it is compared with the same model built on {DESIGN["s2_desc"]} at the same overall accuracy (OA). '
            f'Decision rules follow the '
            f'{DESIGN["tau"]}; each feature set traces an OA–BA curve. Matched-OA comparisons use the '
            f'{DESIGN["window"]}. Here OA★ is taken per feature set, the window is anchored at '
            f'min(OA★_AE, OA★_S2), and the matched operating point is its middle, o_mid = min(OA★_AE, OA★_S2) − 2. '
            f'For each feature set, the '
            f'point at OA = o_mid is located on the envelope of its OA–BA curve by linear interpolation, and class '
            f'recalls are interpolated with the same weights. Only the classes that enter the unit\'s balanced '
            f'accuracy (BA) are counted. {unit_txt}'
            f'All differences are in percentage points (pp). This analysis is descriptive only: no test was '
            f'pre-specified and no inference is drawn. {stat}')
    col = '"mean [95% CI]"' if any_ci else '"mean"'
    pa = (f'\n\na, Per class: AE − S2 difference in class recall at o_mid, sorted by the mean. The number in '
          f'parentheses after each class is the number of {Ds} in which the class is evaluated '
          f'({int(cs.n.min())}–{int(cs.n.max())}). The {col} column gives the mean across those {Ds}'
          f'{" and, where the class is evaluated in at least " + str(us.CI_MIN_N) + " " + Ds + ", its 95% CI" if any_ci else ""}'
          f'; "> 0" as above.')
    pb = ''
    if ss is not None:
        empty = ss[ss.n == 0].grp_lab.tolist()
        part = ss[(ss.n > 0) & (ss.n < n_dom)]
        pb = (f'\n\nb, Per {m["sample"]}-area group: AE − S2 difference in BA at o_mid, computed within groups of '
              f'{m["sample"]} area. Area is given in multiples of π·30² m² ≈ 0.28 ha, the area of a 30-m-radius '
              f'disc. Feature support: {m["support"]}; {AREA_CONSEQ[task]}. Groups: '
              f'{", ".join(ss.grp_lab[:-1])} and {ss.grp_lab.iloc[-1]} units. Within a group, BA is taken over the '
              f'classes with at least 30 test {smp} in that group; a group with fewer than two such classes is '
              f'missing for that unit. The rows below the axis give, per group, the mean'
              f'{", its 95% CI" if (ss.n >= us.CI_MIN_N).any() else ""} and "> 0" as in a, and the share of test '
              f'{smp} in the group, averaged over the {n_dom} {Ds}.')
        if len(part):
            pb += (' ' + '; '.join(f'The {g} group is evaluated in {int(n)} {Ds}' for g, n in zip(part.grp_lab, part.n))
                   + f', all other {"non-empty " if empty else ""}groups in {n_dom}.')
        if empty:
            pb += f' No test {m["sample"]} falls in the {" or ".join(empty)} group ("none"; share 0.0%).'
    return head + body + pa + pb


if __name__ == '__main__':
    for t in us.TASKS:
        cs, ss = draw(t)
        print(t, 'saved', len(cs), 'classes', '' if ss is None else f'{len(ss)} size groups')
