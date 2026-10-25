"""UF0 跨任务总览（uf0_overview；双栏 183 × 约 110 mm，两行三列，一列一个任务）。

核心句：在三个任务中，迁移之后换用 AE 带来的 BA 增益都大于 400 个随机目标标签。

第一行 a–c：剂量曲线（us.dose）+ AE 参考线（us.ae_band，默认 xmax 与 x_dom），x 范围 us.DOSE_XLIM；
            每列一个任务，各用自己的 y 范围、相同刻度风格。域数 < 5 时 ae_band 只画均值线（AE 逐域值见 e、f），
            us.dose 在单域线末端标域名。
第二行 d–f：两根杠杆同尺度（us.forest，cols=True 右侧数值列 + us.forest_note 角注）：
            +50…+400 随机标签、参照臂 REF、AE instead of S2（0 标签，行级 ms = 4.6）；三列共用同一 x 尺度。
            本面板是任务层汇总，不是逐域面板：作物的 RO、PT 只在 a 中画（灰色虚线），d 中 show_others=False。
三个任务由同一段代码循环生成；任务间只有数据驱动的差别（域数 < 5 时不画区间、改画各域值）。
只读 source_data/，不重算统计量。
"""
import os
import re

os.environ.setdefault('OMP_NUM_THREADS', '2')

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, MultipleLocator

import ustyle as us

us.setup()
C, FS, MM, TERM, D = us.C, us.FS, us.MM, us.TERM, us.DESIGN
NAME = 'uf0_overview'

# ------------------------------------------------------------------ 版式（毫米）
W, H = 183, 112
fig = plt.figure(figsize=(W * MM, H * MM))
MEAS = plt.figure(dpi=600)                          # 量字宽用（600 dpi 下的字宽接近矢量输出；低 dpi 的字形微调会低估）
RND = MEAS.canvas.get_renderer()


def ax_mm(l, b, w, h):
    return fig.add_axes([l / W, b / H, w / W, h / H])


def plabel(ax, lab, x_mm, y_mm):
    """面板标号：按 mm 位置换算成轴坐标，保证同一行标号齐平、同一列标号对齐。"""
    p = ax.get_position()
    us.panel(ax, lab, x=(x_mm / W - p.x0) / p.width, y=(y_mm / H - p.y0) / p.height)


def text_mm(s, fs):
    """文字在图中的宽度（mm），用于给行标签与右侧数值列留位。"""
    t = MEAS.text(0, 0, s, fontsize=fs)
    w = t.get_window_extent(RND).width / MEAS.dpi / MM
    t.remove()
    return w


def wrap(s, max_mm, fs=FS['tick']):
    """按词折行，使每行宽度 ≤ max_mm（行标签更窄，给右侧数值列留出空间）。"""
    lines = []
    for wd in s.split(' '):
        if lines and text_mm(lines[-1] + ' ' + wd, fs) <= max_mm:
            lines[-1] += ' ' + wd
        else:
            lines.append(wd)
    return '\n'.join(lines)


COLW = 61.0
COL = (0.0, 61.0, 122.0)                            # 三列起点（mm）
R1_B, R1_H, R1_L, R1_W = 67.0, 36.0, 9.0, 41.0      # 第一行：轴底、轴高、列内左偏移、轴宽
R2_B, R2_H = 9.0, 39.0                              # 第二行：轴底、轴高（轴宽与左偏移由文字宽度决定，见下）
LAB1, LAB2 = R1_B + R1_H + 5.0, R2_B + R2_H + 5.0    # 面板标号基线
LEG_H, ANN_H = 6.0, 6.0                             # 第一行：左上角图例键、AE 两行标注所需的顶部空间（mm）
LAB_MAX = 14.0                                      # 行标签每行最大宽（mm），超出即折行
C0, CGAP, COL_END = 1.2, 1.5, 1.6                   # 数值列与轴右缘、两列之间、与下一列的间距（mm）
NUMS = []                                           # 图中印出的数字及来源（写入报告）


# ------------------------------------------------------------------ 局部辅助
def line_no(task, name, feat, set_='primary'):
    """estimates 表中该行的行号（表头为第 1 行），供报告核对。"""
    e = us.read(task, 'estimates')
    i = e.index[(e.name == name) & (e.feat == feat) & (e.set == set_)]
    assert len(i) == 1
    return int(i[0]) + 2


def forest_rows(task):
    """第二行的行：+50…+400 随机标签（LABEL）、REF（REF_COLOR）、AE instead of S2（AE，0 标签，大点）。"""
    m = us.META[task]
    rows = [us.row(task, f'dFA_RT{k}', 'both', wrap(TERM['rk'].format(k=k), LAB_MAX), C['LABEL'],
                   key=(f'dFA_RT{k}', 'both')) for k in us.KS]
    rows.append(us.row(task, 'dFA_REF', 'both', wrap(m['ref_short'], LAB_MAX), us.REF_COLOR, key=('dFA_REF', 'both')))
    rows.append(us.row(task, 'D_Z0', 'ae-s2', wrap(TERM['ae0'], LAB_MAX), C['AE'], ms=4.6, key=('D_Z0', 'ae-s2')))
    return rows


def span(r):
    """一行实际画出的横向范围（CI 或各域值，加均值）。"""
    e = r['e']
    if e['n'] >= us.CI_MIN_N and np.isfinite(e['lo95']):
        return [e['lo95'], e['hi95'], e['mean']]
    dv = np.asarray(r['dv'], float)
    return list(dv[np.isfinite(dv)]) + [e['mean']]


def col_texts(rows):
    """右侧两列的文字（与 us.forest 一致：均值列取 us.ci_text 去掉计数部分；计数列为 n_pos/n）及列头。"""
    n = rows[0]['e']['n']
    mean_col = [us.ci_text(r['e']).rsplit(', ', 1)[0] for r in rows]
    cnt_col = [f"{int(r['e']['n_pos'])}/{int(r['e']['n'])}" for r in rows]
    hdr = 'mean' + (' [95% CI]' if n >= us.CI_MIN_N else '')
    return mean_col + [hdr], cnt_col + ['> 0']


# 第二行版式：行标签宽、右侧两列宽由实际文字决定；三列轴宽相同（共用 x 尺度）
ROWS = {t: forest_rows(t) for t in us.TASKS}
LAB_W = max(text_mm(s, FS['tick']) for t in us.TASKS for r in ROWS[t] for s in r['label'].split('\n'))
MEAN_W = {t: max(text_mm(s, FS['small']) for s in col_texts(ROWS[t])[0]) for t in us.TASKS}
CNT_W = {t: max(text_mm(s, FS['small']) for s in col_texts(ROWS[t])[1]) for t in us.TASKS}
R2_L = LAB_W + 1.0                                  # 轴左缘距列起点（行标签 + 刻度间距）
R2_W = COLW - R2_L - max(C0 + MEAN_W[t] + CGAP + CNT_W[t] for t in us.TASKS) - COL_END
plt.close(MEAS)

# 第二行三列共用 x 尺度：由三个任务画出的全部值决定，取 5 的倍数
allv = np.concatenate([span(r) for t in us.TASKS for r in ROWS[t]])
XF = (5 * np.floor(allv.min() / 5 - 0.08), 5 * np.ceil(allv.max() / 5 + 0.02))

# ================================================================== 三个任务同一段代码
for j, task in enumerate(us.TASKS):
    m = us.META[task]
    x0 = COL[j]
    n = int(us.est(task, 'dFA_RT50', 'both')['n'])
    dom_word = m['domains'] if n != 1 else m['domain']

    # -------------------------------------------------------------- 第一行：剂量曲线 + AE 参考线
    ax = ax_mm(x0 + R1_L, R1_B, R1_W, R1_H)
    us.zero_line(ax, 'y')
    us.dose(ax, task)
    e_ae = us.ae_band(ax, task)
    NUMS.append(f"{'abc'[j]}: \"AE instead of S2 (0 labels) {us.fmt_pp(e_ae['mean'])}\" ← {task}_estimates.csv "
                f"line {line_no(task, 'D_Z0', 'ae-s2')} (name D_Z0, feat ae-s2, set primary, mean {e_ae['mean']:.4f})")
    ax.set_xlim(us.DOSE_XLIM)
    # y 范围：本任务画出的全部值（各域曲线、均值与 CI、AE 线与 CI 带；n < 5 时 ae_band 只画均值线）。
    # yl 为左上角图例键下方（k ≤ 200）的值，加上贯穿全宽的 AE 线与 CI 带
    ys, yl = [0.0], [0.0, e_ae['mean']]
    for d in us.all_domains(task):
        v = [us.dom(task, f'dFA_RT{k}', 'both').get(d, np.nan) for k in us.KS]
        ys += v
        yl += v[:3]
    for k in us.KS:
        ek = us.est(task, f'dFA_RT{k}', 'both')
        v = [ek['mean']] + ([ek['lo95'], ek['hi95']] if ek['n'] >= us.CI_MIN_N else [])
        ys += v
        yl += v if k <= 200 else []
    if e_ae['n'] >= us.CI_MIN_N:
        yl += [e_ae['lo95'], e_ae['hi95']]
    ys, yl = [np.asarray(a, float)[np.isfinite(a)] for a in (ys + yl, yl)]
    # 下方留 8% 白；上方取三者最大：全部值 + 6% 白；图例键下方的值 + LEG_H；AE 线 + 其两行标注 ANN_H
    lo = ys.min() - 0.08 * (ys.max() - ys.min())
    ax.set_ylim(lo, max(ys.max() + 0.06 * (ys.max() - lo), lo + (yl.max() - lo) / (1 - LEG_H / R1_H),
                        lo + (e_ae['mean'] - lo) / (1 - ANN_H / R1_H)))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10], integer=True))
    ax.xaxis.labelpad = 1.5
    if j == 0:
        ax.set_ylabel(TERM['dba'], labelpad=1.5)
    ax.set_title(m['title'], fontsize=FS['title'], pad=3)
    # 左上角图例键：粗线为主集均值（n ≥ 5 附 95% CI），细线为单个域（三列同一写法，文字由数据决定）
    ax.legend(handles=[Line2D([], [], color=C['LABEL'], lw=1.4, marker='o', ms=3.0),
                       Line2D([], [], color=C['LABEL'], lw=0.6, alpha=0.35)],
              labels=[f'{n}-{m["domain"]} mean' + (', 95% CI' if n >= us.CI_MIN_N else ''), f'single {dom_word}'],
              loc='upper left', bbox_to_anchor=(0.0, 1.0), fontsize=FS['small'], handlelength=1.8,
              handletextpad=0.5, borderaxespad=0.2, labelspacing=0.3)
    plabel(ax, 'abc'[j], x0, LAB1)

    # -------------------------------------------------------------- 第二行：两根杠杆同尺度
    ax = ax_mm(x0 + R2_L, R2_B, R2_W, R2_H)
    rows = ROWS[task]
    col_x = (1 + C0 / R2_W, 1 + (C0 + MEAN_W[task] + CGAP) / R2_W)
    us.forest(ax, rows, cols=True, col_x=col_x, show_others=False)
    for t in ax.get_yticklabels():                  # 折行标签行距收紧，行与行之间留出空隙
        t.set_linespacing(1.0)
    ax.set_xlim(*XF)
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.tick_params(axis='y', pad=2)
    ax.xaxis.labelpad = 1.5
    us.forest_note(ax, task)
    for r, s0, s1 in zip(rows, *[c[:-1] for c in col_texts(rows)]):
        nm, ft = r['key']
        NUMS.append(f"{'def'[j]}: \"{s0}\" \"{s1}\" ({r['label'].replace(chr(10), ' ')}) ← {task}_estimates.csv "
                    f"line {line_no(task, nm, ft)} (name {nm}, feat {ft}, set primary, mean {r['e']['mean']:.4f}, "
                    f"lo95 {r['e']['lo95']:.4f}, hi95 {r['e']['hi95']:.4f}, n {int(r['e']['n'])}, "
                    f"n_pos {int(r['e']['n_pos'])})")
    plabel(ax, 'def'[j], x0, LAB2)

us.save(fig, NAME)

# ================================================================== 图例文字（英文，投稿用）
prim_crop = us.primary('crop')
extra = us.others('crop')
arm = us.read('crop', 'arm_rows')
oa_x = arm[(arm.arm == 'Z0') & arm.domain.isin(extra)].groupby(['domain', 'feat']).OA_star.mean()
n_crop = len(prim_crop)


def names(task, ds, codes=False):
    """域名列表（us.dname），英文连词；codes=True 时附短码。"""
    s = [us.dname(task, d)
         + (f' ({us.dlabel(d)})' if codes else '') for d in ds]
    return s[0] if len(s) == 1 else ', '.join(s[:-1]) + ' and ' + s[-1]


def n_source(task):
    """每个目标域的源池大小：作物为其余目标国；LCZ、LUCAS 为 role = source 的域数。"""
    t = us.read(task, 'domains')
    return len(us.all_domains(task)) - 1 if task == 'crop' else int((t.role == 'source').sum())


def task_sentence(task):
    """每个任务一句：数据源、类别数、样本单位与域、源池、测试划分、特征取样支撑（全部取自 META）。"""
    m = us.META[task]
    src = re.search(r'\((.*)\)$', m['long']).group(1)
    low = lambda s: s[0].lower() + s[1:]
    return (f"{m['title']}: {src}; {m['n_classes']} classes; samples are {m['samples']} and domains are "
            f"{m['domains']}; {low(m['pool'])}; {low(m['test'])}; {m['support']}.")


lat = us.read('lucas', 'domains').set_index('domain').lat
luc_ns = sorted(us.primary('lucas'), key=lambda d: lat[d])           # 自南向北
smp = [us.META[t]['samples_short'] for t in us.TASKS]
legend = f"""
Random target labels versus the AE representation across three mapping tasks.

The three tasks were analysed with the same pipeline. In each task, a target domain is mapped by a LightGBM model trained on a source pool of other domains plus k random target labels (labelled {smp[0]}, {smp[1]} or {smp[2]}; k = 0, 50, 100, 200 or 400; target labels weighted {D['target_weight']}); k = 0 is {TERM['zero']}. Features are {D['ae_desc']} or {D['s2_desc']}. {task_sentence('crop')} {task_sentence('lcz')} {task_sentence('lucas')}

Metric: ΔBA at matched OA. For every model, decisions follow the {D['tau']}, which traces a BA–OA frontier. BA is averaged over the {D['window']} with the same features, and differenced against {TERM['zero']}. Label gains (random labels and reference arm) are the mean of the AE and S2 values of each domain. AE instead of S2 (0 labels) is the AE − S2 difference in {TERM['zero']}, with the window anchored at min(OA★_AE, OA★_S2).

Domains and inference. The domain is the unit of inference. Crop type: {len(us.all_domains('crop'))} target countries, each mapped from a foreign source pool of the other {n_source('crop')}; means use the {TERM['primary']} of {n_crop} countries ({names('crop', prim_crop)}); {names('crop', extra, codes=True)} are outside the {TERM['primary']} because their zero-label OA★ is only about {oa_x.min():.0f}–{oa_x.max():.0f}% (grey dashed lines in a; not shown in d). Each country value averages {us.META['crop']['units_note'].replace(' per country', '')}. Urban form (LCZ): {len(us.primary('lcz'))} target cities ({names('lcz', us.primary('lcz'))}), one per region, each mapped from a source pool of {n_source('lcz')} other cities. Land cover (LUCAS): {len(us.primary('lucas'))} target countries from south to north, {names('lucas', luc_ns, codes=True)}, each mapped from a source pool of {n_source('lucas')} other countries. For LCZ and LUCAS, each domain value averages 3 random draws of the target labels. Zero-label transfer and reference models are single runs.

Reference arm. Crop type: a {us.META['crop']['ref']}, trained on labels of the target country alone. LCZ: the source pool plus {us.META['lcz']['ref']}. LUCAS: the source pool plus {us.META['lucas']['ref']}. Row labels in d–f give the approximate (crop type) or largest (LCZ, LUCAS) number of reference labels per domain.

Statistics display (all panels). When a task has at least {us.CI_MIN_N} primary domains (crop type, n = {n_crop}), means are shown with 95% t confidence intervals (df = n − 1). With fewer than {us.CI_MIN_N} domains (LCZ and LUCAS, n = 3), t intervals are not informative and are not drawn; the single-domain values are shown instead (label gains: thin lines in b and c, open circles in e and f; AE instead of S2: open circles in e and f only).

a–c, Dose–response of ΔBA at matched OA on the number k of random target labels (k = 0 is {TERM['zero']}, at 0 by definition); one column per task, each with its own y range. Thick line with points, mean over the {TERM['primary']} (crop type: bars, 95% CI); thin lines, single primary domains (LCZ and LUCAS: labelled at k = 400 with the city name or country code); grey dashed lines (crop type only), {names('crop', extra)}. Teal dashed line and number, mean gain from AE instead of S2 (0 labels), drawn on the same scale as the label gains; crop type: shaded band, its 95% CI; LCZ and LUCAS: mean line only, with the single-domain values shown in e and f.

d–f, The two levers on one scale (d–f share the x axis). Rows: gain from +50, +100, +200 and +400 random labels and from the reference arm (black), each relative to {TERM['zero']}; and the gain from AE instead of S2 (0 labels), shown as a larger teal point. Points, means over the {TERM['primary']}; crop type: bars, 95% CI; LCZ and LUCAS: small open circles, single domains, joined by a thin line spanning their range. Right-hand columns: mean (crop type: with 95% CI) and the number of primary domains with a value > 0.

BA, balanced accuracy; OA, overall accuracy; pp, percentage points.
"""
us.write_legend(NAME, legend)
print('saved')
print(f'layout: R2_L {R2_L:.1f} mm, R2_W {R2_W:.1f} mm, MEAN_W {MEAN_W}, CNT_W {CNT_W}, XF {XF}')
print('\n'.join(NUMS))
