"""UF3 标签与表征同尺度（uf3_labels_representation_{task}；双栏 183 × 62 mm，三个任务各一张）。

a｜剂量曲线（us.dose）+ AE 参考线（us.ae_band，默认参数；n < 5 时只画均值线，单域线末端标域名）；
   x 范围 us.DOSE_XLIM；y 上限按图例键与 AE 两行标注的实测高度留白；
   在 k = 200 处标注 us.ci_text(dFA_RT200, both)，放在该点右下方的空白带（避开各曲线、CI 竖线与其他文字），短引线。
b｜两根杠杆同尺度（us.forest(cols=True) + us.forest_note）：+50…+400 随机标签、参照臂 REF、
   AE instead of S2（0 标签，ms = 4.6）。
c｜逐域对比（us.domain_dots(grey_markers=False) 画两次）：D_Z0（AE 实心）与 dFA_RT400（both，LABEL 空心），
   按 D_Z0 排序，灰色短线连接；非主集域只把行标签改灰。

三个任务由同一段代码循环生成，版式、尺寸、颜色、文字规则相同；任务间只有数据驱动的差别
（域数 < 5 时不画区间、改画各域值；作物另有 RO、PT 两个非主集域）。
只读 source_data/，不重算统计量。
"""
import os

os.environ.setdefault('OMP_NUM_THREADS', '2')

import matplotlib.pyplot as plt
import matplotlib.text as mtext
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

import ustyle as us

us.setup()
C, FS, MM, T = us.C, us.FS, us.MM, us.TERM
NAME = 'uf3_labels_representation'
GREY = C['MUTED']

# ------------------------------------------------------------------ 版式（毫米；三个任务完全相同）
W, H = 183, 62
BOT, AH = 9.0, 41.5                      # 轴底与轴高（三个面板相同）
LAB_Y = BOT + AH + 6.6                   # 面板标号基线
AX = dict(a=(10.5, 47.0), b=(88.5, 28.0), c=(155.0, 27.5))    # 各面板轴的左缘与宽
LAB_X = dict(a=0.0, b=67.5, c=142.5)                            # 面板标号左缘
COL_MM = (2.0, 22.0)                     # b 右侧两列（均值 [CI]、> 0 计数）距轴右缘的毫米数；按作物最长的 CI 文本定


def ax_mm(fig, l, w):
    return fig.add_axes([l / W, BOT / H, w / W, AH / H])


def plabel(ax, lab):
    """面板标号：按 mm 位置换算成轴坐标，保证三个标号齐平。"""
    p = ax.get_position()
    us.panel(ax, lab, x=(LAB_X[lab] / W - p.x0) / p.width, y=(LAB_Y / H - p.y0) / p.height)


def wrap_paren(s):
    """把 "xxx (yyy)" 折成两行，行标签更窄。"""
    return s.replace(' (', '\n(', 1)


def nice_lim(vals, lo_pad=0.08, hi_pad=0.08):
    v = np.asarray(vals, float)
    v = v[np.isfinite(v)]
    s = v.max() - v.min()
    return v.min() - lo_pad * s, v.max() + hi_pad * s


def forest_rows(task):
    """b 的行（与 UF0 第二行相同）：+50…+400 随机标签、REF、AE instead of S2（0 标签，放大点）。"""
    m = us.META[task]
    rows = [us.row(task, f'dFA_RT{k}', 'both', T['rk'].format(k=k), C['LABEL']) for k in us.KS]
    rows.append(us.row(task, 'dFA_REF', 'both', wrap_paren(m['ref_short']), us.REF_COLOR))
    rows.append(us.row(task, 'D_Z0', 'ae-s2', wrap_paren(T['ae0']), C['AE'], ms=4.6))
    return rows


def row_span(r):
    """forest 一行实际画出的横向范围（CI 或各域值，加均值），用于定 x 范围。"""
    e = r['e']
    if e['n'] >= us.CI_MIN_N and np.isfinite(e['lo95']):
        return [e['lo95'], e['hi95'], e['mean']]
    dv = np.asarray(r['dv'], float)
    return list(dv[np.isfinite(dv)]) + [e['mean']]


KS0 = np.r_[0.0, us.KS]                  # 剂量曲线的 x（k = 0 为零标签迁移，值固定为 0）


def dose_curves(task, ci_ok):
    """a 中画出的全部折线（零线、各域线、均值线；n ≥ 5 时加 CI 上下端），每条为 KS0 上的 y 值。"""
    cs = [np.zeros(len(KS0))]
    for d in us.all_domains(task):
        cs.append(np.r_[0.0, [us.dom(task, f'dFA_RT{k}', 'both').get(d, np.nan) for k in us.KS]])
    es = [us.est(task, f'dFA_RT{k}', 'both') for k in us.KS]
    cs.append(np.r_[0.0, [e['mean'] for e in es]])
    if ci_ok:
        cs += [np.r_[0.0, [e['lo95'] for e in es]], np.r_[0.0, [e['hi95'] for e in es]]]
    return [c for c in cs if np.all(np.isfinite(c))]


def data_box(ax, a, r):
    """文字（或图例）在数据坐标中的范围 (x0, y0, x1, y1)。"""
    (x0, y0), (x1, y1) = ax.transData.inverted().transform(a.get_window_extent(r).get_points())
    return x0, y0, x1, y1


def k200_note(ax, task, ci_ok, dx_mm=1.5):
    """a 中 k = 200 的任务层估计（us.ci_text）：文字左上角放在该点右侧 dx_mm 处，自上而下找第一条
    不压任何曲线（零线、各域线、均值线、CI 竖线）与其他文字（单域线末端域名、AE 标注）的水平空白带；
    短引线从文字左上角指到 CI 下端（无 CI 时指到均值点）。文字总在该点下方，引线不会穿过上方的 AE 虚线。"""
    e = us.est(task, 'dFA_RT200', 'both')
    tip = e['lo95'] if ci_ok else e['mean']
    r = ax.figure.canvas.get_renderer()
    others = [data_box(ax, a, r) for a in ax.texts]          # 已有文字（先于本标注量取）
    t = ax.annotate(us.ci_text(e), xy=(200, tip), xytext=(200, tip), ha='left', va='top',
                    fontsize=FS['small'], color=C['LABEL'], zorder=5,
                    arrowprops=dict(arrowstyle='-', lw=0.4, color=GREY, shrinkA=0.6, shrinkB=1.6,
                                    relpos=(0.0, 1.0)))
    x0, y0, x1, y1 = data_box(ax, t, r)
    w, h = x1 - x0, y1 - y0
    p = ax.get_position()
    xl = ax.get_xlim()
    xa = min(200 + dx_mm * (xl[1] - xl[0]) / (p.width * W), xl[1] - w)    # 不超出轴右缘
    # 文字横向范围内的占用区间：折线（KS0 之间线性）取点值；CI 竖线与其他文字取上下端
    xs = np.linspace(xa, min(xa + w, KS0[-1]), 60)
    occ = [(v, v) for c in dose_curves(task, ci_ok) for v in np.interp(xs, KS0, c)]
    if ci_ok:
        occ += [(us.est(task, f'dFA_RT{k}', 'both')['lo95'], us.est(task, f'dFA_RT{k}', 'both')['hi95'])
                for k in us.KS if xa <= k <= xa + w]
    occ += [(b[1], b[3]) for b in others if b[0] < xa + w and b[2] > xa]
    lo_, hi_ = np.array(occ).T
    pad = 0.35 * h
    top = tip - pad
    while np.any((hi_ > top - h - pad) & (lo_ < top + pad)):  # 带内有线或文字 → 下移
        top -= 0.05 * h
    assert top - h > ax.get_ylim()[0], (task, 'k = 200 标注超出下缘')
    t.set_position((xa, top))
    return t


def dose_ylim(ax, task, ci_ok, e_ae, leg):
    """a 的 y 范围：下方 8% 留白；上限取三者最大——全部值 + 6% 留白；图例键（左上）下缘高于其横向范围内的
    全部值；AE 均值线上方两行标注（us.ae_band）不出轴顶。图例与标注的高度按渲染实测（以轴高的比例计，
    与 y 范围无关）。"""
    r = ax.figure.canvas.get_renderer()
    curves = dose_curves(task, ci_ok)
    ys = np.concatenate(curves + [[e_ae['mean']]] + ([[e_ae['lo95'], e_ae['hi95']]] if ci_ok else []))
    lo = ys.min() - 0.08 * np.ptp(ys)
    # 图例键：右缘以左的全部折线值（含 n ≥ 5 时贯穿全宽的 AE CI 带）须低于图例下缘
    lb = ax.transAxes.inverted().transform(leg.get_window_extent(r).get_points())
    xl = ax.get_xlim()
    xr = xl[0] + lb[1, 0] * (xl[1] - xl[0])
    xs = np.linspace(0, min(xr, KS0[-1]), 60)
    yl = np.concatenate([np.interp(xs, KS0, c) for c in curves] + [[e_ae['mean']]] +
                        ([[e_ae['hi95']]] if ci_ok else []))
    # AE 标注：上缘距 AE 均值线的高度（偏移 + 两行文字）
    ann = [a for a in ax.texts if a.get_text().startswith(T['ae0'].split(' (')[0])][0]
    ah = (ann.get_window_extent(r).y1 - ax.transData.transform((0, e_ae['mean']))[1]) / ax.bbox.height
    top = max(ys.max() + 0.06 * (ys.max() - lo),
              lo + (yl.max() - lo) / (lb[0, 1] - 0.03),
              lo + (e_ae['mean'] - lo) / (1 - ah - 0.02))
    ax.set_ylim(lo, top)


# ================================================================== 逐任务出图（同一段代码）
def make(task):
    m = us.META[task]
    n = int(us.est(task, 'dFA_RT50', 'both')['n'])
    dom1, doms = m['domain'], m['domains']
    ci_ok = n >= us.CI_MIN_N
    fig = plt.figure(figsize=(W * MM, H * MM))

    # -------------------------------------------------------------- a｜剂量曲线 + AE 参考线
    ax = ax_mm(fig, *AX['a'])
    us.zero_line(ax, 'y')
    us.dose(ax, task)
    e_ae = us.ae_band(ax, task)
    ax.set_xlim(us.DOSE_XLIM)
    ax.set_ylabel(T['dba'], labelpad=1.5)
    ax.set_xlabel(T['axis_k'], labelpad=1.5)
    # 左上角图例键：粗线为主集均值，细线为单个域
    leg = ax.legend(handles=[Line2D([], [], color=C['LABEL'], lw=1.4, marker='o', ms=3.0),
                             Line2D([], [], color=C['LABEL'], lw=0.6, alpha=0.35)],
                    labels=[f'{n}-{dom1} mean' + (', 95% CI' if ci_ok else ''), f'single {doms}'],
                    loc='upper left', bbox_to_anchor=(0.0, 1.0), fontsize=FS['small'], handlelength=1.8,
                    handletextpad=0.5, borderaxespad=0.2, labelspacing=0.3)
    dose_ylim(ax, task, ci_ok, e_ae, leg)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10], integer=True))
    k200_note(ax, task, ci_ok)
    plabel(ax, 'a')

    # -------------------------------------------------------------- b｜两根杠杆同尺度
    ax = ax_mm(fig, *AX['b'])
    rows = forest_rows(task)
    wb = AX['b'][1]
    us.forest(ax, rows, xlabel=T['dba'], cols=True, col_x=tuple(1 + c / wb for c in COL_MM), show_others=False)
    ax.set_xlim(*nice_lim(np.concatenate([row_span(r) for r in rows] + [[0.0]]), 0.10, 0.10))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 5, 10]))
    ax.tick_params(axis='y', pad=2)
    ax.xaxis.labelpad = 1.5
    us.forest_note(ax, task)
    plabel(ax, 'b')

    # -------------------------------------------------------------- c｜逐域对比（哑铃）
    ax = ax_mm(fig, *AX['c'])
    a = us.dom(task, 'D_Z0', 'ae-s2').reindex(us.all_domains(task))
    r = us.dom(task, 'dFA_RT400', 'both').reindex(us.all_domains(task))
    order = list(a.sort_values().index)               # y = 0 在底部 → D_Z0 最大的域在顶部
    ax.hlines(range(len(order)), a[order], r[order], color=C['GRID'], lw=0.7, zorder=1)
    # 空心圈（外径 4.0 + 0.8 pt）先画、实心点（外径 2.8 + 0.8 pt）后画：两值重合时仍能看到外圈；
    # 点保持系列颜色（grey_markers=False），非主集域只把行标签改灰
    us.domain_dots(ax, task, 'dFA_RT400', 'both', C['LABEL'], order=order, filled=False, ms=4.0,
                   grey_markers=False)
    us.domain_dots(ax, task, 'D_Z0', 'ae-s2', C['AE'], order=order, ms=2.8, grey_markers=False)
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.tick_params(axis='y', pad=2)
    ax.set_xlim(*nice_lim(np.r_[a.values, r.values, 0.0], 0.10, 0.10))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 5, 10]))
    ax.set_xlabel(T['dba'], labelpad=1.5)
    pc = ax.get_position()
    ax.legend(handles=[Line2D([], [], ls='none', marker='o', ms=2.8, mfc=C['AE'], mec=C['AE'], mew=0.8),
                       Line2D([], [], ls='none', marker='o', ms=4.0, mfc='white', mec=C['LABEL'], mew=0.8)],
              labels=[T['ae0'], T['rk'].format(k=400)], loc='lower left',
              bbox_to_anchor=(((LAB_X['c'] + 2.0) / W - pc.x0) / pc.width, 1.0), fontsize=FS['small'],
              handlelength=1.0, handletextpad=0.4, borderaxespad=0.15, labelspacing=0.25, borderpad=0.0)
    plabel(ax, 'c')

    name = f'{NAME}_{task}'
    us.save(fig, name)
    us.write_legend(name, legend_text(task))
    print('saved', name)


# ================================================================== 图例文字（英文，投稿用）
def dnames(task, ds):
    """图例中的域名：全名（us.dname），与图中短码不同时在括号中给出短码。"""
    out = []
    for d in ds:
        full, code = us.dname(task, d), us.dlabel(d)
        out.append(full if full == code else f'{full} ({code})')
    return out


def join_and(xs):
    return xs[0] if len(xs) == 1 else ', '.join(xs[:-1]) + ' and ' + xs[-1]


def legend_text(task):
    """图例文字：全部由 META、us.DESIGN、us.TERM 与源数据生成，三个任务同一模板。"""
    m, D = us.META[task], us.DESIGN
    prim, alld, extra = us.primary(task), us.all_domains(task), us.others(task)
    n, N = len(prim), len(alld)
    dom1, doms = m['domain'], m['domains']
    ci_ok = int(us.est(task, 'dFA_RT50', 'both')['n']) >= us.CI_MIN_N
    e_ae = us.est(task, 'D_Z0', 'ae-s2')
    pool_name, pool = m['pool'].split(': ', 1)
    pool_name = pool_name.lower()                                            # "foreign source pool" / "source pool"
    pool = pool.replace(' (', ', ').rstrip(')')                              # 避免括号嵌套
    test = m['test'][len('Test: '):].replace(', ', ' with a ')
    ref = m['ref']
    if 'target-only' in ref:
        ref_desc = f'the {ref}, trained on the target {dom1} alone without the {pool_name}'
    else:
        ref_desc = f'the {pool_name} plus {ref} of the target {dom1}'
    ex = join_and(dnames(task, extra)) if extra else ''
    if extra:
        set_note = (f'Estimates use the {T["primary"]} of {n} {doms}: {join_and(dnames(task, prim))}. {ex}, the '
                    f'other target {doms}, are outside the {T["primary"]}: they appear in a (grey dashed lines) and '
                    f'c (grey codes) but enter no mean.')
    else:
        set_note = f'Estimates use the {T["primary"]}, which comprises all {n} target {doms}: {join_and(dnames(task, prim))}.'
    if ci_ok:
        rule = f'Means are shown with 95% t confidence intervals (df = {n - 1}).'
        a_lines = f'Thin lines, single {"primary-set " if extra else ""}{doms}'
        a_mean = f'thick line and points, {n}-{dom1} mean with 95% CI bars'
        a_ae = 'teal band, its 95% CI'
        a_ann = (f'mean [95% CI], number of {doms} > 0 / number of {doms}; the leader points to the lower end of '
                 f'the CI bar')
        b_pts = 'Points, means; bars, 95% CI.'
        b_cols = f'the mean [95% CI] and the number of {doms} with a value > 0 over the number of {doms}'
    else:
        rule = (f'With fewer than {us.CI_MIN_N} {doms}, t intervals are not informative and are not drawn; '
                f'single-{dom1} values are shown instead.')
        a_lines = f'Thin lines, single {doms}, named at the line end'
        a_mean = f'thick line and points, {n}-{dom1} mean'
        a_ae = f'only the mean is drawn here, and single-{dom1} values are shown in c'
        a_ann = f'mean, number of {doms} > 0 / number of {doms}; the leader points to the mean'
        b_pts = (f'Filled circles, means across the {n} {doms}; open circles, single {doms}, joined by a thin line '
                 f'spanning their range.')
        b_cols = f'the mean and the number of {doms} with a value > 0 over the number of {doms}'
    a_extra = f'; grey dashed lines with codes, {ex}' if extra else ''
    c_extra = f' {ex}, shown with grey codes, are outside the {T["primary"]}.' if extra else ''
    Dom1 = dom1[0].upper() + dom1[1:]
    ks = ', '.join(str(k) for k in (0,) + us.KS[:-1]) + f' or {us.KS[-1]}'
    return f"""
Labels and representation on one scale — {m['long']}.

LightGBM models were trained on a {pool_name} ({pool}) plus k random target labels ({m['samples_short']} drawn at random from the target {dom1}; k = {ks}; sample weight {D['target_weight']}) and evaluated on {test}. Features are {D['ae_desc']} or {D['s2_desc']}; sampling support, {m['support']}. ΔBA at matched OA: every model is read through the prior-shift {D['tau']}; the family traces a BA–OA frontier, and frontier BA is differenced between two models and averaged over the {D['window']}. Label gains (+k random labels and the reference arm) are differences from {T['zero']} with the same features and are averaged over the AE and S2 feature sets. {T['ae0']} is the AE − S2 difference between the two {T['zero']} models, read in the window anchored at min(OA★_AE, OA★_S2). Each {dom1} value is the mean over {m['units_note']} (draws apply to arms with random target labels); the {dom1} is the unit of inference (n = {n} {doms}). {set_note} {rule} Units: percentage points (pp).

a, Dose–response: gain in BA at matched OA from k random target labels (k = 0 is {T['zero']}, fixed at 0). {a_lines}; {a_mean}{a_extra}. Teal dashed line, mean gain from {T['ae0']}, {us.fmt_pp(e_ae['mean'])} pp, drawn on the same scale; {a_ae}. Annotation, the {n}-{dom1} gain from 200 random labels: {a_ann}.

b, The two levers on one scale ({n} {doms}). Rows, top to bottom: gains from +50 to +400 random target labels (blue); gain from the reference arm, {ref_desc} (black); and {T['ae0']}, in teal with a larger point. {b_pts} Columns at right give {b_cols}.

c, {Dom1} by {dom1} (all {N} target {doms}): filled teal points, {T['ae0']}; open blue circles, +400 random target labels, averaged over the AE and S2 gains. A grey segment joins the two values of each {dom1}; rows are sorted by the teal value.{c_extra}

AE, AlphaEarth embeddings; BA, balanced accuracy; OA, overall accuracy; S2, Sentinel-2.
"""


if __name__ == '__main__':
    for t in us.TASKS:
        make(t)
