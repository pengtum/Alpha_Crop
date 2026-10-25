"""UF4 互补性 uf4_complementarity_{task}（三个任务各一张，双栏 183 × 62 mm）。

a｜标签对两种特征的回报：AE、S2 各自相对本特征零标签迁移的剂量曲线（us.dose，show_domains=False，偏移 −4 / +4，
   x 范围 us.DOSE_XLIM）；域数 < 5 时另用细线画各域（契约允许的辅助代码：两种特征的单域线若都在末端标域名会相互
   挤压，故不标）。按契约第三轮，a 不画 us.ae_band：y 轴是各特征相对自身零标签模型的增益，D_Z0 已在 b 中。
b｜AE instead of S2 在加标签后：D_Z0、D_RT200、D_RT400（us.forest，cols=True，AE 色大点；组标题 "AE instead of S2"
   写在面板标题里，行名为 "0 labels"、"+200 random labels"、"+400 random labels"）。
c｜交互：dFA_RT50…RT400 与 dFA_REF 的 inter（us.forest，level = 90，cols=True；REF 行名用 META ref_short）。
b、c 的右上角角注用 us.forest_note；右侧数值列由 us.forest 写出。列位置以 mm 计、按三个任务共同的最长文本固定
（COL_MM，参照 uf3），三张图因此同宽。
作物的 RO、PT 不在主集，本图只画任务层估计，故 b、c 用 show_others=False（图例写明）。

三个任务由同一段代码循环生成；任务间只有数据驱动的差别（坐标范围、域数决定画区间还是画各域点）。
只读 source_data/（经 ustyle），不重算统计量。
"""
import os

os.environ.setdefault('OMP_NUM_THREADS', '2')

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

import ustyle as us  # noqa: E402

us.setup()
C, FS, MM, META, TERM, DESIGN = us.C, us.FS, us.MM, us.META, us.TERM, us.DESIGN
FW, FH = 183, 62                       # 图幅（mm）
AX_Y0, AX_H = 10.0, 37.5               # 三个面板共用的轴底与轴高（mm）
LAB_Y = 55.0                           # 面板标号与面板标题的基线（mm）
# 各面板的轴位置（左缘 x0、宽 w，mm）；b、c 左侧留给行标签，右侧留给 forest 的两列数值
PA = dict(x0=10.5, w=36.0, lab_x=1.0)
PB = dict(x0=72.0, w=21.0, lab_x=49.5)
PC = dict(x0=141.5, w=17.5, lab_x=116.0)
COL_GAP0, COL_GAP1 = 1.6, 1.8          # 轴右缘 → 均值列、均值列 → "> 0" 列的间距（mm）


# ================================================================== 局部辅助函数
def add_ax(fig, p):
    return fig.add_axes([p['x0'] / FW, AX_Y0 / FH, p['w'] / FW, AX_H / FH])


def plabel(fig, ax, letter, x_mm, title):
    """面板标号与面板标题按 mm 定位，保证同一行齐平；标题紧跟标号。"""
    pos = ax.get_position()
    us.panel(ax, letter, x=(x_mm / FW - pos.x0) / pos.width, y=(LAB_Y / FH - pos.y0) / pos.height)
    fig.text((x_mm + 3.6) / FW, LAB_Y / FH, title, fontsize=FS['small'], color=C['MUTED'], ha='left', va='bottom')


def text_w_mm(fig, s, fs):
    """文字在图中的宽度（mm），用于定 forest 数值列的位置。"""
    t = fig.text(0, 0, s, fontsize=fs)
    w = t.get_window_extent(renderer=fig.canvas.get_renderer()).width / fig.dpi / MM
    t.remove()
    return w


def col_mm(rows_fn, level):
    """forest 两列距轴右缘的毫米数（三个任务共用，参照 uf3 的 COL_MM）：均值列距轴右缘 COL_GAP0；"> 0" 列在
    三个任务中最宽的均值文字（含列头）之后再空 COL_GAP1。均值列文字与 us.forest 写出的一致
    （us.ci_text 去掉计数部分）。"""
    fig = plt.figure()
    txt = []
    for t in us.TASKS:
        rows = rows_fn(t)
        txt += [us.ci_text(r['e'], level=level).rsplit(', ', 1)[0] for r in rows]
        txt.append('mean' + (f' [{level}% CI]' if rows[0]['e']['n'] >= us.CI_MIN_N else ''))
    w = max(text_w_mm(fig, s, FS['small']) for s in txt)
    plt.close(fig)
    return COL_GAP0, COL_GAP0 + w + COL_GAP1


def col_x(p, panel):
    """COL_MM（mm）换算成 us.forest 的 col_x（轴坐标）。"""
    return tuple(1 + c / p['w'] for c in COL_MM[panel])


def nice_axis(vals, nbins=5, pad=0.07):
    """由数据范围（含 0）给出留白后的轴范围，以及落在该范围内的整齐刻度（步长 1、2、5 × 10^n）。"""
    v = np.asarray([x for x in vals if np.isfinite(x)] + [0.0], float)
    span = v.max() - v.min()
    lim = (v.min() - pad * span, v.max() + pad * span)
    ticks = MaxNLocator(nbins=nbins, steps=[1, 2, 5, 10]).tick_values(*lim)
    ticks = ticks[(ticks >= lim[0] - 1e-9) & (ticks <= lim[1] + 1e-9)]
    return ticks, lim


def tick_labels(ticks):
    """刻度标签：整数不带小数，负号 U+2212。"""
    nd = 0 if np.allclose(ticks, np.round(ticks)) else 1
    return [f'{t:.{nd}f}'.replace('-', '−') if abs(t) > 1e-12 else '0' for t in ticks]


def set_x(ax, rows, level, nbins):
    """forest 的 x 范围与刻度：域数 ≥ CI_MIN_N 用区间端点，否则用各域值。"""
    vals = []
    for r in rows:
        e = r['e']
        vals.append(e['mean'])
        vals += [e[f'lo{level}'], e[f'hi{level}']] if e['n'] >= us.CI_MIN_N else list(np.asarray(r['dv'], float))
    xt, xlim = nice_axis(vals, nbins=nbins)
    ax.set_xlim(*xlim)
    ax.set_xticks(xt)
    ax.set_xticklabels(tick_labels(xt))


def dose_domains(ax, task, feat, color, offset):
    """域数 < CI_MIN_N 时，逐域细线（同色、同偏移），k = 0 处为 0。返回所画的值，供定 y 范围。"""
    ks = np.array((0,) + us.KS, float) + offset
    out = []
    for d in us.primary(task):
        v = [0.0] + [us.dom(task, f'dFA_RT{k}', feat).get(d, np.nan) for k in us.KS]
        ax.plot(ks, v, color=color, lw=0.6, alpha=0.4, zorder=1, solid_capstyle='round')
        out += v
    return out


def end_labels(ax, ys, names, colors, x, min_gap_frac=0.085):
    """曲线右端直接标注特征名；两端点太近时按 y 轴比例对称推开，保持上下次序。"""
    y0, y1 = ax.get_ylim()
    gap = min_gap_frac * (y1 - y0)
    order = np.argsort(ys)                    # 低 → 高
    lo_i, hi_i = order[0], order[1]
    ylo, yhi = ys[lo_i], ys[hi_i]
    if yhi - ylo < gap:
        mid = 0.5 * (ylo + yhi)
        ylo, yhi = mid - gap / 2, mid + gap / 2
    pos = {lo_i: ylo, hi_i: yhi}
    for i, (n, c) in enumerate(zip(names, colors)):
        ax.text(x, pos[i], n, color=c, fontsize=FS['base'], fontweight='bold', ha='left', va='center')


def rows_b(task):
    """b 的行：组含义（AE instead of S2）在面板标题里，行名只写标签数；AE 行一律大点（ms 4.6）。"""
    return [us.row(task, 'D_Z0', 'ae-s2', '0 labels', C['AE'], ms=4.6),
            us.row(task, 'D_RT200', 'ae-s2', TERM['rk'].format(k=200), C['AE'], ms=4.6),
            us.row(task, 'D_RT400', 'ae-s2', TERM['rk'].format(k=400), C['AE'], ms=4.6)]


def rows_c(task):
    rows = [us.row(task, f'dFA_RT{k}', 'inter', TERM['rk'].format(k=k), C['LABEL']) for k in us.KS]
    return rows + [us.row(task, 'dFA_REF', 'inter', META[task]['ref_short'], us.REF_COLOR)]


# b、c 数值列的位置（mm，距轴右缘）：三个任务共同的最长文本决定，三张图相同
COL_MM = dict(b=col_mm(rows_b, 95), c=col_mm(rows_c, 90))


# ================================================================== 每个任务一张图
def build(task):
    fig = plt.figure(figsize=(FW * MM, FH * MM))
    n = int(us.est(task, 'dFA_RT50', 'ae')['n'])
    doms = META[task]['domains']

    # ---------------------------------------------------------- a｜标签对两种特征的回报
    ax = add_ax(fig, PA)
    yvals = []
    for feat, col, off in (('s2', C['S2'], +4.0), ('ae', C['AE'], -4.0)):
        if n < us.CI_MIN_N:
            yvals += dose_domains(ax, task, feat, col, off)
        yvals += list(us.dose(ax, task, feat=feat, color=col, show_domains=False, offset=off))
        if n >= us.CI_MIN_N:
            yvals += [us.est(task, f'dFA_RT{k}', feat)[b] for k in us.KS for b in ('lo95', 'hi95')]
    us.zero_line(ax, axis='y')
    yt, ylim = nice_axis(yvals, nbins=6, pad=0.06)
    ax.set_ylim(*ylim)
    ax.set_yticks(yt)
    ax.set_yticklabels(tick_labels(yt))
    ax.set_xlim(us.DOSE_XLIM)
    ax.set_xlabel(TERM['axis_k'])
    ax.set_ylabel(TERM['dba'])
    end_labels(ax, [us.est(task, 'dFA_RT400', f)['mean'] for f in ('ae', 's2')], ['AE', 'S2'],
               [C['AE'], C['S2']], x=420)
    stat = f'{n} {doms}; mean and 95% CI' if n >= us.CI_MIN_N else f'{n} {doms}; mean and single {doms}'
    ax.text(0.0, 1.03, f'vs {TERM["zero"]} with the same features\n{stat}', transform=ax.transAxes,
            ha='left', va='bottom', fontsize=FS['small'], color=C['MUTED'], linespacing=1.2)
    plabel(fig, ax, 'a', PA['lab_x'], 'Label gain per feature set')

    # ---------------------------------------------------------- b｜AE instead of S2，加标签后
    ax = add_ax(fig, PB)
    rb = rows_b(task)
    us.forest(ax, rb, level=95, xlabel=TERM['d_ba'], cols=True, col_x=col_x(PB, 'b'), show_others=False)
    set_x(ax, rb, 95, nbins=5)
    us.forest_note(ax, task, level=95)
    plabel(fig, ax, 'b', PB['lab_x'], 'AE instead of S2')

    # ---------------------------------------------------------- c｜交互（AE − S2 的标签增益差，90% CI）
    ax = add_ax(fig, PC)
    rc = rows_c(task)
    us.forest(ax, rc, level=90, xlabel=TERM['inter'].replace(' in label', '\nin label'), cols=True,
              col_x=col_x(PC, 'c'), show_others=False)
    set_x(ax, rc, 90, nbins=5)
    us.forest_note(ax, task, level=90)
    plabel(fig, ax, 'c', PC['lab_x'], 'Interaction')

    name = f'uf4_complementarity_{task}'
    us.save(fig, name)
    us.write_legend(name, legend_text(task))
    print('saved', name)


# ================================================================== 图例文字（英文，投稿用）
def above(v):
    return 'above 0' if v > 0 else ('below 0' if v < 0 else 'at 0')


def legend_text(task):
    m = META[task]
    n = int(us.est(task, 'D_Z0', 'ae-s2')['n'])
    big = n >= us.CI_MIN_N
    dom, doms = m['domain'], m['domains']
    # source pool：去掉 META 中的前缀与嵌套括号
    detail = m['pool'].split(': ', 1)[1]
    if ' × ' not in detail:
        detail = detail.replace(' (', ', ').rstrip(')')
    pool = ('a foreign source pool' if m['pool'].startswith('Foreign') else 'a source pool') + f' ({detail})'
    test = m['test'].split(': ', 1)[1].replace(', ', ' with a ')
    oth = us.others(task)
    if oth:
        # 非主集域：全名取自 us.dname；OA★ 范围为各域零标签迁移 OA★ 的单元均值（AE、S2 两种特征）
        ar = us.read(task, 'arm_rows')
        oa = ar[(ar.arm == 'Z0') & ar.domain.isin(oth)].groupby(['domain', 'feat']).OA_star.mean()
        names = ' and '.join(us.dname(task, d) for d in oth)
        set_note = (f"Estimates use the {TERM['primary']} of {n} {doms} (all {len(us.all_domains(task))} target "
                    f"{doms} except {names}, whose OA★ of {TERM['zero']} is only about "
                    f"{oa.min():.0f}–{oa.max():.0f}%); {names} are not shown in this figure.")
    else:
        set_note = f'Estimates use all {n} target {doms} ({TERM["primary"]}).'
    ref_note = ('the target-only model, trained on the target country alone (~30k labels) without the source pool'
                if task == 'crop' else f"the model trained on the source pool plus {m['ref']}")
    if big:
        stat_a = (f'Thick lines and points, means across the {n} {doms}; bars, 95% t confidence intervals '
                  f'(df = {n - 1}).')
        stat_b = f'Points, means across the {n} {doms}; bars, 95% t confidence intervals (df = {n - 1}).'
        stat_c = f'Points, means across the {n} {doms}; bars, 90% t confidence intervals (df = {n - 1}).'
        cols = ('Columns at the right of b and c give the mean with its confidence interval (95% in b, 90% in c) '
                f'and the number of {doms} with a value > 0.')
        rule = ''
    else:
        stat_a = (f'Thick lines and points, means across the {n} {doms}; thin lines, single {doms}.')
        stat_b = (f'Filled circles, means across the {n} {doms}; open circles, single {doms}, joined by a thin '
                  f'line spanning their range.')
        stat_c = stat_b
        cols = f'Columns at the right of b and c give the mean and the number of {doms} with a value > 0.'
        rule = (f' With fewer than {us.CI_MIN_N} {doms}, t intervals are not informative and are not drawn; '
                f'single-{dom} values are shown instead.')

    # b、c 的数字（只陈述数字与符号，取自源数据）
    e0, e400 = us.est(task, 'D_Z0', 'ae-s2'), us.est(task, 'D_RT400', 'ae-s2')
    s0, s4 = above(e0['mean']), above(e400['mean'])
    if s0 == s4:
        num_b = (f"The mean is {s0} both with 0 labels ({us.ci_text(e0)}) and with "
                 f"{TERM['rk'].format(k=400)} ({us.ci_text(e400)}).")
    else:
        num_b = (f"The mean is {s0} with 0 labels ({us.ci_text(e0)}) and {s4} with "
                 f"{TERM['rk'].format(k=400)} ({us.ci_text(e400)}).")
    ek = [us.est(task, f'dFA_RT{k}', 'inter') for k in us.KS]
    eref = us.est(task, 'dFA_REF', 'inter')
    mk = [e['mean'] for e in ek]
    pos = sorted({int(e['n_pos']) for e in ek})
    cnt = (f'{pos[0]}/{n} {doms} > 0 at each k' if len(pos) == 1
           else f'{pos[0]}/{n} to {pos[-1]}/{n} {doms} > 0')
    rng = f'{us.fmt_pp(min(mk))} to {us.fmt_pp(max(mk))} for +50 to +400 random labels ({cnt})'
    signs = {above(v) for v in mk + [eref['mean']]}
    lead = f'All five means are {signs.pop()}: ' if len(signs) == 1 else 'Means range from '
    num_c = f"{lead}{rng}; reference arm, {us.ci_text(eref, level=90)}."

    return f"""
Complementarity of random target labels and the AE representation — {m['long']}.

LightGBM models were trained on {pool} plus k random target {m['samples_short']} (k = 0, 50, 100, 200 or 400; sample weight {DESIGN['target_weight']}) and evaluated on {test}. Features are {DESIGN['ae_desc']} or {DESIGN['s2_desc']}; sampling support: {m['support']}. BA at matched OA: for every model, the {DESIGN['tau']} traces a BA–OA frontier. Frontier BA is averaged over the {DESIGN['window']}; differences between two models are taken on this average. Each {dom} value is the mean over {m['units_note']} (draws apply to arms with random target labels); the {dom} is the unit of inference (n = {n} {doms}). {set_note}{rule} {cols} Units: percentage points (pp).

a, Gain from k random target labels for each feature set (AE, teal; S2, grey), each relative to {TERM['zero']} with the same features; OA★ is the τ = 0 OA of that zero-label model. The y axis is a gain over each feature set's own zero-label model, not an absolute accuracy level; the AE − S2 difference with 0 labels is shown in b. AE and S2 points are offset horizontally by −4 and +4 labels for legibility. {stat_a}

b, AE instead of S2 with the same target labels: AE − S2 difference in BA at matched OA with 0 labels, {TERM['rk'].format(k=200)} and {TERM['rk'].format(k=400)}. The window anchor is min(OA★_AE, OA★_S2), where OA★ is the τ = 0 OA of {TERM['zero']} with each feature set. {stat_b} {num_b}

c, AE − S2 difference in label gain (interaction): the {dom} value of a gain with AE features minus the same gain with S2 features, for +50 to +400 random labels (the gains in a) and for the reference arm, {ref_note}. A value of 0 means that labels add the same BA with either feature set. {stat_c} {num_c}
"""


if __name__ == '__main__':
    for t in us.TASKS:
        build(t)
