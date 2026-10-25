"""Fig. 1 研究设计（双栏 183 mm；示意图为主的组合图）。

核心句：两个平行场景共用同一个迁移协议与同一个评估——外国池来自 10 国；目标国要么零标签起步，
要么已有一次集中的局部调查；评估在带缓冲的空间测试格上按同 OA 比较 BA。

面板：
a 研究区地图（EPSG:3035），11 个目标国，RO/PT 用斜线区分，地图下方列出各国地块数（千）；
b 迁移与评估流程示意（4 个圆角框，竖排）；
c 示例目标单元（比利时，一半格）的 2×2 小图：场景 A 随机、场景 B 已有调查、原区扩采、Betti 分配。
只读 source_data/；不重算任何统计量。
"""
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import patheffects as pe
from matplotlib.collections import PatchCollection
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch, Rectangle

import nstyle as ns

ns.setup()

C, FS, MM = ns.C, ns.FS, ns.MM
FILL_STUDY = '#B4C0E4'      # 11 个目标国填色
FILL_OTHER = '#EEEEEE'      # 其他国家
CAND = '#D8D8D8'            # 候选地块点
TESTC = '#4D4D4D'           # 测试格填色（alpha 0.18）
TESTP = '#ADADAD'           # 测试地块点（比候选点略深）


def tint(hex_c, a):
    """把颜色与白色按比例混合，得到不透明的浅色（打印时比 alpha 叠加更稳定）。"""
    r, g, b = mpl.colors.to_rgb(hex_c)
    return (1 - a + a * r, 1 - a + a * g, 1 - a + a * b)


# ------------------------------------------------------------------ 画布（以 mm 为单位布局）
FW, FH = 183.0, 150.0
fig = plt.figure(figsize=(FW * MM, FH * MM))
fig.set_dpi(600)                       # 在高分辨率下量文字宽度，减少字形微调带来的误差
renderer = fig.canvas.get_renderer()


def add_ax(x, y, w, h):
    """按 mm 在画布上放置坐标轴（x、y 为左下角）。"""
    return fig.add_axes([x / FW, y / FH, w / FW, h / FH])


def panel_at(ax, label, x_mm, y_mm):
    """把面板标号放到画布上固定的 mm 位置（仍调用 ns.panel，保证字号与字重一致）。"""
    bb = ax.get_position()
    ns.panel(ax, label, x=(x_mm / FW - bb.x0) / bb.width, y=(y_mm / FH - bb.y0) / bb.height)


TOP = 146          # 面板标号底边的统一高度（mm）
ROW0 = 141        # 第一行内容（小标题、框）的上沿

# 三列的横向位置（mm）
AX0, AW = 0.5, 54.0     # a
BX0, BW = 100.0, 80.0    # b
CX0 = 0.5             # c
R_EDGE = 181.9          # 内容右边界；加上导出时 0.5 mm 白边后总宽 ≤ 183 mm

# 面板 c 的几何先算好：a、b 的底边与 c 的图例底边对齐
cells = pd.read_csv(ns.src('example_unit_cells.csv'))
xmin, xmax = cells.cx.min() * 10 - 2, (cells.cx.max() + 1) * 10 + 2
ymin, ymax = cells.cy.min() * 10 - 2, (cells.cy.max() + 1) * 10 + 2
HGAP, title_h, KEY_H = 3.5, 7.0, 9.0
cw = (R_EDGE - CX0 - 3*HGAP) / 4
ch = cw * (ymax - ymin) / (xmax - xmin)
row_top = [20 + ch + title_h]
BOTTOM = 75.0

# =================================================================== a｜研究区地图
cc = pd.read_csv(ns.src('country_counts.csv'))
g = gpd.read_file(ns.src('countries.geojson')).to_crs(3035)
main9 = set(cc.loc[cc.primary9, 'country'])
side2 = set(cc.loc[~cc.primary9, 'country'])          # RO、PT

# 范围：以 11 国外包框为基准外扩约 120 km（约合经度 −10…31、纬度 35…70）
bx0, by0, bx1, by1 = g[g.CNTR_CODE.isin(main9 | side2)].total_bounds
x0, x1, y0, y1 = bx0 - 1.2e5, bx1 + 1.2e5, by0 - 1.0e5, by1 + 0.8e5
map_h = AW * (y1 - y0) / (x1 - x0)

# 地图上方：角注（两行）
fig.text((AX0 + 3.6) / FW, ROW0 / FH, '11 target countries; each is mapped\nfrom a pool of the other 10',
         fontsize=FS['small'], ha='left', va='top', color=C['TEXT'], linespacing=1.2)
ax_a = add_ax(AX0, ROW0 - 6.6 - map_h, AW, map_h)
ax_a.set_xlim(x0, x1)
ax_a.set_ylim(y0, y1)
ax_a.set_aspect('equal')
ax_a.set_axis_off()

g[~g.CNTR_CODE.isin(main9 | side2)].plot(ax=ax_a, facecolor=FILL_OTHER, edgecolor='white', linewidth=0.3)
g[g.CNTR_CODE.isin(main9 - {'DE'})].plot(ax=ax_a, facecolor=FILL_STUDY, edgecolor='white', linewidth=0.3)
# 德国只用了两个州：全国画浅色，两州画主集色
g[g.CNTR_CODE == 'DE'].plot(ax=ax_a, facecolor=tint(FILL_STUDY, 0.35), edgecolor='white', linewidth=0.3)
gpd.read_file(ns.src('study_regions_de.geojson')).to_crs(3035).plot(ax=ax_a, facecolor=FILL_STUDY,
                                                                      edgecolor='white', linewidth=0.3)
# RO、PT：更浅的底色 + 同色斜线；边线另画白色
HATCH = '//////'
g_side = g[g.CNTR_CODE.isin(side2)]
# PT 全国有标签：整国斜线；RO 只有南部边境地带有标签：全国浅色，覆盖区斜线
g[g.CNTR_CODE == 'PT'].plot(ax=ax_a, facecolor=tint(FILL_STUDY, 0.3), edgecolor=FILL_STUDY, hatch=HATCH, linewidth=0)
g[g.CNTR_CODE == 'RO'].plot(ax=ax_a, facecolor=tint(FILL_STUDY, 0.15), edgecolor='none', linewidth=0)
gpd.read_file(ns.src('study_regions_ro.geojson')).to_crs(3035).plot(ax=ax_a, facecolor=tint(FILL_STUDY, 0.3),
                                                                      edgecolor=FILL_STUDY, hatch=HATCH, linewidth=0)
g_side.boundary.plot(ax=ax_a, color='white', linewidth=0.3)

# 国家代码标在质心（SE 用代表点；少数国家微调以免压边）
# 白色光晕单独画成底层描边（只有 Stroke、没有 Normal，导出为白色路径），其上再画不带 path effect 的文字：
# 带 path effect 的文字在 SVG/PDF 中会被转成轮廓路径，不再是可编辑的 <text>
halo = [pe.Stroke(linewidth=1.3, foreground='white')]
nudge = {'SI': (0.2e5, -0.9e5), 'SE': (0.2e5, -1.2e5), 'BE': (-1.1e5, 0.3e5), 'PT': (0.1e5, 0)}
for _, r in g[g.CNTR_CODE.isin(main9 | side2)].iterrows():
    p = r.geometry.representative_point() if r.CNTR_CODE == 'SE' else r.geometry.centroid
    dx, dy = nudge.get(r.CNTR_CODE, (0, 0))
    kw = dict(fontsize=7.5, ha='center', va='center')
    # 国家代码仅保留可编辑的文字层。
    ax_a.text(p.x + dx, p.y + dy, r.CNTR_CODE, color=C['TEXT'] if r.CNTR_CODE in main9 else C['MUTED'],
              zorder=6, **kw)

for collection in ax_a.collections:collection.set_rasterized(True)
panel_at(ax_a, 'a', AX0 - 0.5, TOP)

# 地图下方：图例（主集 / RO、PT）
y_leg = ROW0 - 6.6 - map_h - 1.2
ax_leg = add_ax(AX0, y_leg - 5.4, AW, 5.4)
ax_leg.set_axis_off()
ax_leg.legend(handles=[Patch(facecolor=FILL_STUDY, edgecolor='none', label='primary set (9 countries)'),
                       Patch(facecolor=tint(FILL_STUDY, 0.3), edgecolor=FILL_STUDY, hatch=HATCH, linewidth=0,
                             label='RO, PT (reported separately)')],
              loc='upper left', bbox_to_anchor=(0.06, 1.0), fontsize=FS['small'], handlelength=1.3,
              handleheight=0.9, borderaxespad=0, labelspacing=0.3, handletextpad=0.5)

for handle in ax_leg.get_legend().legend_handles:handle.set_rasterized(True)

# 各国地块数（千），4 列 × 3 行
cc_sorted = pd.concat([cc[cc.primary9].sort_values('country'), cc[~cc.primary9].sort_values('country')])
y_tab = 132.0     # 表的最后一行与面板 c 图例底行对齐
fig.text(61.0 / FW, y_tab / FH, 'Parcels\n(thousands)', fontsize=FS['small'],
         ha='left', va='top', color=C['TEXT'])
col_w = 10.8
for i, (_, r) in enumerate(cc_sorted.iterrows()):
    row_, col_ = i, 0
    xx = 61.0
    yy = y_tab - 9.0 - row_ * 3.7
    colr = C['TEXT'] if r.primary9 else C['MUTED']
    fig.text(xx / FW, yy / FH, r.country, fontsize=FS['small'], ha='left', va='top', color=colr)
    fig.text((xx + 24.0) / FW, yy / FH, f'{r.n_parcels / 1000:.0f}', fontsize=FS['small'], ha='right',
             va='top', color=colr)

# =================================================================== b｜迁移与评估流程
B_TOP, B_BOT = ROW0, BOTTOM + 0.3
ax_b = add_ax(BX0, B_BOT, BW, B_TOP - B_BOT)
ax_b.set_xlim(0, BW)
ax_b.set_ylim(0, B_TOP - B_BOT)
ax_b.set_axis_off()
HB = B_TOP - B_BOT

LH = 3.5                                # 6 pt 文字的行高（mm）
PAD = 2.1                               # 框内上下留白（mm）
GAPX = 0.55                             # 同一行中两段文字之间的空隙（约一个空格宽，mm）
boxes = [
    [[('Foreign pool', 'bold')], [('10 countries × 5,000 parcels', None)]],
    [[('+ Target labels', 'bold'), ('(weight 50)', None)],
     [('Scenario A', 'A'), ('— none →', None), ('k', 'ital'), ('random', None)],
     [('Scenario B', 'B'), ('— existing survey +', None), ('k', 'ital'), ('1', 'sub'), ('new', None)]],
    [[('Classifier', 'bold')], [('LightGBM (main) · Random forest (check)', None)]],
    [[('Test:', 'bold'), ('10-km cells of the target', None)],
     [('country, 1-km buffer', None)],
     [('Metric:', 'bold'), ('BA at matched OA', None)]],
]
heights = [len(b) * LH + 2 * PAD for b in boxes]
gap = (HB - sum(heights)) / (len(boxes) - 1)           # 4 个框铺满面板高度，与面板 c 底边对齐
y_cur = HB
rects = []
for h in heights:
    rects.append((y_cur - h, h))
    y_cur -= h + gap


def draw_runs(ax, x, y, runs, fs=7.5):
    """在一行里依次画若干段文字（可加粗/着色/斜体/下标），段与段之间留一个空格宽的空隙。
    下标用 5.5 pt 并下移，紧贴前一段（保证最小字号 ≥ 5.5 pt，不用 mathtext 的 70% 缩小下标）。"""
    inv = ax.transData.inverted()
    for j, (txt, style) in enumerate(runs):
        if j > 0 and style != 'sub':
            x += GAPX
        kw = dict(fontsize=fs, ha='left', va='center', color=C['TEXT'])
        yy = y
        if style in ('bold', 'A', 'B'):
            kw['fontweight'] = 'bold'
        if style in ('A', 'B'):
            kw['color'] = C['LABEL'] if style == 'A' else C['CONC']
        if style == 'ital':
            kw['fontstyle'] = 'italic'
        if style == 'sub':
            kw['fontsize'] = FS['small']
            yy = y - 0.55
        t = ax.text(x, yy, txt, **kw)
        x = inv.transform((t.get_window_extent(renderer=renderer).x1, 0))[0]


for b, (yb, h) in zip(boxes, rects):
    ax_b.add_patch(FancyBboxPatch((0.3, yb), BW - 0.6, h, boxstyle='round,pad=0,rounding_size=1.4',
                                  facecolor='white', edgecolor=C['REF_Z0'], linewidth=0.6))
    for i, runs in enumerate(b):
        yl = yb + h - PAD - LH * (i + 0.5)
        x_start = 2.0
        if runs[0][1] in ('A', 'B'):          # 场景 A、B 行前加小色块
            col = C['LABEL'] if runs[0][1] == 'A' else C['CONC']
            ax_b.add_patch(Rectangle((2.0, yl - 0.8), 1.6, 1.6, facecolor=col, edgecolor='none'))
            x_start = 4.3
        draw_runs(ax_b, x_start, yl, runs)

for (ya, _), (yb_, hb_) in zip(rects[:-1], rects[1:]):
    ax_b.add_patch(FancyArrowPatch((BW / 2, ya - 0.4), (BW / 2, yb_ + hb_ + 0.4), arrowstyle='-|>',
                                   mutation_scale=6, linewidth=0.6, color=C['REF_Z0'], shrinkA=0, shrinkB=0))
panel_at(ax_b, 'b', BX0 - 2.5, TOP)

# =================================================================== c｜示例目标单元（比利时）
lay = pd.read_csv(ns.src('example_unit_layout.csv'))
tcells = pd.read_csv(ns.src('example_unit_test_cells.csv'))

cand = lay[lay.role == 'candidate']
test = lay[lay.role == 'test']
s0 = lay[lay.role == 'S0']
S0_PALE = tint(C['CONC'], 0.45)


def base_map(ax):
    """底图：候选点 + 测试地块点 + 测试格 + 20 km 比例尺（右上角空白处）。点均为抽样显示。"""
    ax.scatter(cand.x_km, cand.y_km, s=0.3, c=CAND, linewidths=0, rasterized=True, zorder=1)
    ax.scatter(test.x_km, test.y_km, s=0.3, c=TESTP, linewidths=0, rasterized=True, zorder=1)
    ax.add_collection(PatchCollection([Rectangle((x * 10, y * 10), 10, 10) for x, y in zip(tcells.cx, tcells.cy)],
                                      facecolor=TESTC, alpha=0.18, edgecolor='none', zorder=2))
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    xb, yb = xmax - 12 - 20, 5641.0           # 右上角（荷兰一侧）空白处；标签不出轴右界
    ax.plot([xb, xb + 20], [yb, yb], color=C['TEXT'], lw=0.8, solid_capstyle='butt', zorder=6)
    ax.text(xb + 10, yb + 2.0, '20 km', fontsize=FS['small'], ha='center', va='bottom', zorder=6)


def survey_regions(ax):
    reg = cells[cells.S0_region]
    ax.add_collection(PatchCollection([Rectangle((x * 10, y * 10), 10, 10) for x, y in zip(reg.cx, reg.cy)],
                                      facecolor=C['CONC'], alpha=0.10, edgecolor='none', zorder=2))


specs = [
    ('Scenario A\nRandom labels', C['LABEL'], 'SRS100', C['LABEL'], False),
    ('Scenario B\nExisting survey', C['CONC'], None, None, True),
    ('Scenario B\nExtend surveyed area', C['CONC'], 'EXT100', C['CONC'], True),
    ('Scenario B\nRegional allocation', C['BETTI'], 'BA100', C['BETTI'], True),
]
axes_c = []
for i, (title, tcol, role, pcol, has_s0) in enumerate(specs):
    r_, c_ = 0, i
    ax = add_ax(CX0 + c_ * (cw + HGAP), row_top[r_] - title_h - ch, cw, ch)
    base_map(ax)
    if has_s0:
        survey_regions(ax)
        faint = role is not None
        ax.scatter(s0.x_km, s0.y_km, s=2.0 if faint else 4, c=[S0_PALE if faint else C['CONC']],
                   linewidths=0, zorder=4)
    if role is not None:
        pts = lay[lay.role == role]
        ax.scatter(pts.x_km, pts.y_km, s=4, c=pcol, linewidths=0, zorder=5)
    ax.text(0.0, 1.0 + 0.6 / ch, title, transform=ax.transAxes, fontsize=FS['label'], color=tcol,
            ha='left', va='bottom')
    axes_c.append(ax)

# 小图 2 左下角（法国一侧的空白）写 S0 的构造
fig.text(.5, 16.5/FH, 'Existing survey: 100 labels in 10 cells, 10 parcels per cell, across 2 of 8 regions',
               fontsize=FS['small'], ha='center', va='top', color=C['CONC'], linespacing=1.3)
panel_at(axes_c[0], 'c', CX0 - 0.5, row_top[0]+6)

# 面板 c 下方的共用图例
dot = lambda col, ms=2.3: Line2D([], [], marker='o', ls='none', ms=ms, mfc=col, mec='none')
ax_key = add_ax(CX0, 3, R_EDGE - CX0, KEY_H)
ax_key.set_axis_off()
ax_key.legend(handles=[dot(CAND), (Patch(facecolor=TESTC, alpha=0.18, edgecolor='none'), dot(TESTP, 1.6)),
                       Patch(facecolor=C['CONC'], alpha=0.10, edgecolor='none'),
                       (dot(C['CONC']), dot(S0_PALE)), (dot(C['LABEL']), dot(C['CONC']), dot(C['BETTI']))],
              labels=['candidate parcels', 'test cells (10 km) and parcels', 'surveyed regions',
                      'existing survey (100)', 'new labels (100)'],
              handler_map={tuple: HandlerTuple(ndivide=None, pad=0.1)},
              loc='upper left', bbox_to_anchor=(0.0, 1.0), ncol=3, fontsize=FS['small'], handlelength=1.6,
              handleheight=0.9, borderaxespad=0, labelspacing=0.35, handletextpad=0.4, columnspacing=1.0)

ns.save(fig, 'fig1_design')
print('saved fig1_design')
