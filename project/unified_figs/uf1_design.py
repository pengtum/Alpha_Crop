"""UF1 研究设计图（双栏 183 × 75 mm；三个任务各一张：uf1_design_crop / _lcz / _lucas）。

同一段代码对 us.TASKS 循环生成，版式（三列网格、字号、颜色、图例位置规则）完全相同；任务间只有数据驱动的差别：
a｜研究区地图：crop 与 lucas 为欧洲（EPSG:3035），lcz 为世界（Equal Earth，EPSG:8857）；
b｜迁移与评估流程（4 个圆角框，竖排，文字来自 us.META）；
c｜示例目标单元的布局：测试格、候选样本、测试样本、200 个随机目标标签（第 1 次抽样）；
   lcz 另有缓冲带样本；示例图形状窄高（lucas 的瑞典）时，图例放到右侧，否则放在下方。
只读 source_data/，不重算统计量。
"""
import glob
import os
import re

os.environ.setdefault('OMP_NUM_THREADS', '2')

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patheffects as pe
from matplotlib.collections import PatchCollection
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch, Rectangle
from pyproj import Transformer
from shapely.affinity import translate
from shapely.geometry import MultiPolygon, box
from shapely.ops import unary_union

import ustyle as us

us.setup()
C, FS, MM = us.C, us.FS, us.MM

# ------------------------------------------------------------------ 颜色（契约 UF1）
LAND = '#EEEEEE'            # 陆地底色
FILL_TGT = '#B4C0E4'        # 目标域国家
FILL_SRC = '#E1E6F3'        # 只作源池的国家（lucas）
CAND = '#D8D8D8'            # 候选样本点
TESTP = '#9A9A9A'           # 测试样本点
TESTC = '#4D4D4D'           # 测试格（alpha 0.15）
TESTC_A = 0.15
HATCH = '//////'
SCALE_KM = {'crop': 50, 'lcz': 5, 'lucas': 50}
NOTE = {    # 角落小字（契约原文）；数字在下面与 domains 表核对
    'crop': '11 target countries;\neach mapped from a pool of the other 10',
    'lcz': '3 target cities (one per region);\neach mapped from the other 29',
    'lucas': '3 target countries (south, central-west, north);\neach mapped from the other 25',
}
REGION_WORD = {'IT': 'south', 'UK': 'central-west', 'SE': 'north'}      # lucas 目标国的区位（与角注同义）
DE_STATES = {'DE9': 'Lower Saxony', 'DEA': 'North Rhine-Westphalia'}   # NUTS-1 代码 → 州名（图例文字用）


def tint(hex_c, a):
    """颜色与白色按比例混合，得到不透明浅色（同 RSE 图 1）。"""
    r, g, b = mpl.colors.to_rgb(hex_c)
    return (1 - a + a * r, 1 - a + a * g, 1 - a + a * b)


# ------------------------------------------------------------------ 底图（三个任务共用一次读取）
def polys(g):
    """每行只保留面要素，合成一个 (Multi)Polygon（裁切后可能出现线、点或几何集合），一行一国。"""
    def keep(geom):
        parts = [p for p in getattr(geom, 'geoms', [geom]) if p.geom_type in ('Polygon', 'MultiPolygon')]
        parts = [q for p in parts for q in getattr(p, 'geoms', [p])]
        return MultiPolygon(parts) if parts else None
    g = g.copy()
    g['geometry'] = g.geometry.apply(keep)
    return g[g.geometry.notna() & ~g.geometry.is_empty]


def wrap_lon(geom):
    """LSIB 中跨日界线的面（俄罗斯东部、斐济）经度写成 < −180：整体移回，再在 ±180 处切开，
    否则在 Equal Earth 下会拉出横贯全图的色带。"""
    out = []
    for p in geom.geoms:
        if p.bounds[0] < -180:
            p = translate(p, xoff=360)
        for lo, hi, dx in ((-180, 180, 0), (180, 540, -360)):
            q = p.intersection(box(lo, -90, hi, 90))
            if not q.is_empty:
                out.append(translate(q, xoff=dx))
    return unary_union(out)


WORLD = gpd.read_file(us.src('world_lsib.geojson'))
WORLD = WORLD[WORLD.name != 'Antarctica']
# LSIB 把大国拆成几块（加拿大、俄罗斯、中国等），块边并不严格重合：按名称合并后再做 0.02° 的闭运算，
# 否则块间会画出白色虚线
WORLD = polys(WORLD)
WORLD['geometry'] = WORLD.geometry.make_valid()
WORLD = polys(WORLD)
WORLD['geometry'] = WORLD.geometry.apply(wrap_lon)
WORLD = polys(WORLD.dissolve(by='name').reset_index())
WORLD['geometry'] = WORLD.geometry.buffer(0.02).buffer(-0.02)
WORLD = polys(WORLD)

# 欧洲：先在经纬度下裁一个比视窗大的框再投影，避免远处几何在 LAEA 下变形（裁切边须落在视窗外，否则白色边线可见）
EU = polys(gpd.clip(WORLD, box(-40, 20, 90, 85))).to_crs(3035)
EU['geometry'] = EU.geometry.simplify(1500)
EU = EU[~EU.geometry.is_empty]
# 视窗：经度约 −11…35、纬度约 34…71。用 4 个锚点（葡萄牙/爱尔兰西岸、塞浦路斯东端、克里特南岸、芬兰北端）在 3035 下的
# 外包框再外扩 120 km；crop 与 lucas 用同一视窗
_tr = Transformer.from_crs(4326, 3035, always_xy=True)
_ax_, _ay_ = _tr.transform([-10.5, 34.6, 24.0, 27.0], [39.0, 35.0, 34.8, 70.1])
EU_EXT = (min(_ax_) - 1.2e5, max(_ax_) + 1.2e5, min(_ay_) - 1.2e5, max(_ay_) + 1.2e5)

WEE = WORLD.to_crs(8857)
WEE['geometry'] = WEE.geometry.simplify(15000)
WEE = WEE[~WEE.geometry.is_empty]
_te = Transformer.from_crs(4326, 8857, always_xy=True)
_wx, _ = _te.transform([-180, 180], [0, 0])
_, _wy = _te.transform([0, 0], [-56.5, 84.0])
WEE_EXT = (_wx[0], _wx[1], _wy[0], _wy[1])


# ------------------------------------------------------------------ 数据整理
def domains(task):
    """统一 domains 表：domain、name、n、role（lcz 的 city 列改名为 domain，名称用 us.dname）。"""
    d = us.read(task, 'domains')
    if 'city' in d.columns:
        d = d.rename(columns={'city': 'domain'})
        d['name'] = d.domain.map(lambda x: us.dname(task, x))
    return d


def pool_parts(task):
    """源池的（标题, 正文），取自 META pool（"Source pool" / 作物 "Foreign source pool"，措辞已在 META 中统一）。"""
    h, b = us.META[task]['pool'].split(': ', 1)
    assert 'source pool' in h.lower(), (task, h)
    return h, b


def check_note(task, d):
    """角注里的域数必须与数据一致。"""
    tgt = d[d.role.str.startswith('target')]
    src = d[d.role == 'source']
    pool = len(tgt) - 1 if src.empty else len(src)       # crop：其余 10 个目标国组成源池
    assert f'{len(tgt)} target' in NOTE[task] and f'other {pool}' in NOTE[task], (task, len(tgt), pool)
    # 流程框里的源池样本数（来自 us.META）须等于 domains 表中源域样本数之和
    if len(src):
        assert f'other {len(src)} ' in us.META[task]['pool'] and f'{int(src.n.sum()):,}' in us.META[task]['pool'], task
    if task == 'lucas':                                   # 角注的区位词与图例文字的区位词一致
        assert all(w in NOTE[task] for w in REGION_WORD.values()) and set(REGION_WORD) == set(tgt.domain), task


# ------------------------------------------------------------------ 画布与 mm 布局
FW, FH = 183.0, 75.0
TOP = FH - 3.6              # 面板标号底边
ROW0 = TOP - 1.6            # 内容上沿
BOTTOM = 0.5                # 内容下沿
AX0, AW = 0.5, 58.0         # a
BX0, BW = 63.5, 44.0        # b
CX0, R_EDGE = 112.5, 181.9  # c
LH_SMALL = 5.5 * 1.25 / 72 * 25.4      # 5.5 pt 文字的行高（mm）


class Canvas:
    def __init__(self):
        self.fig = plt.figure(figsize=(FW * MM, FH * MM))
        self.fig.set_dpi(600)                       # 高分辨率下量字宽
        self.r = self.fig.canvas.get_renderer()

    def ax(self, x, y, w, h):
        return self.fig.add_axes([x / FW, y / FH, w / FW, h / FH])

    def panel(self, ax, label, x_mm, y_mm=TOP):
        bb = ax.get_position()
        us.panel(ax, label, x=(x_mm / FW - bb.x0) / bb.width, y=(y_mm / FH - bb.y0) / bb.height)

    def width_mm(self, s, **kw):
        t = self.fig.text(0, 0, s, **kw)
        w = t.get_window_extent(renderer=self.r).width / self.fig.dpi * 25.4
        t.remove()
        return w

    def text(self, x, y, s, **kw):
        return self.fig.text(x / FW, y / FH, s, **kw)


def map_legend(ax, handles, labels, ncol=1):
    ax.set_axis_off()
    ax.legend(handles=handles, labels=labels, loc='upper left', bbox_to_anchor=(0.0, 1.0), ncol=ncol,
              fontsize=FS['small'], handlelength=1.4, handleheight=0.9, borderaxespad=0, labelspacing=0.35,
              handletextpad=0.5, columnspacing=1.2,
              handler_map={tuple: HandlerTuple(ndivide=None, pad=0.15)})


HALO = [pe.Stroke(linewidth=1.3, foreground='white')]


def halo_text(ax, x, y, s, color, **kw):
    """白色光晕单独画在底层（只有 Stroke），上面再画不带 path effect 的文字，SVG/PDF 中仍是可编辑文字（同 RSE 图 1）。"""
    ax.text(x, y, s, color='white', path_effects=HALO, zorder=7, **kw)
    ax.text(x, y, s, color=color, zorder=8, **kw)


# =================================================================== a｜研究区地图
def draw_europe(ax, task, d):
    """crop 与 lucas：欧洲 3035。目标国 FILL_TGT；非主集目标国斜线；有研究区文件的国家叠加实际研究区。"""
    prim = set(us.primary(task))
    tgt = d[d.role.str.startswith('target')]
    src = d[d.role == 'source']
    regions = {os.path.basename(f).split('_')[-1].split('.')[0].upper(): gpd.read_file(f).to_crs(3035)
               for f in sorted(glob.glob(us.src(f'{task}_study_regions_*.geojson')))}
    g = EU                                     # 已是一国一行
    by_name = lambda names: g[g.name.isin(list(names))]

    g.plot(ax=ax, facecolor=LAND, edgecolor='white', linewidth=0.3)
    if len(src):
        by_name(src.name).plot(ax=ax, facecolor=FILL_SRC, edgecolor='white', linewidth=0.3)
    hatched = []
    for _, r in tgt.iterrows():
        gg = by_name([r['name']])
        assert len(gg) == 1, r['name']
        if r.domain in prim:
            if r.domain in regions:            # 只用了部分地区：全国浅色，研究区主色（RSE 图 1 的 DE）
                gg.plot(ax=ax, facecolor=tint(FILL_TGT, 0.35), edgecolor='white', linewidth=0.3)
                regions[r.domain].plot(ax=ax, facecolor=FILL_TGT, edgecolor='white', linewidth=0.3)
            else:
                gg.plot(ax=ax, facecolor=FILL_TGT, edgecolor='white', linewidth=0.3)
        else:                                  # 非主集：同色斜线（RSE 图 1 的 RO、PT）
            hatched.append(gg)
            if r.domain in regions:
                gg.plot(ax=ax, facecolor=tint(FILL_TGT, 0.15), edgecolor='none', linewidth=0)
                regions[r.domain].plot(ax=ax, facecolor=tint(FILL_TGT, 0.3), edgecolor=FILL_TGT, hatch=HATCH,
                                       linewidth=0)
            else:
                gg.plot(ax=ax, facecolor=tint(FILL_TGT, 0.3), edgecolor=FILL_TGT, hatch=HATCH, linewidth=0)
    for gg in hatched:
        gg.boundary.plot(ax=ax, color='white', linewidth=0.3)

    # 目标国代码：质心（SE 用代表点）；少数国家微调以免压边（同 RSE 图 1）
    nudge = {'SI': (0.2e5, -0.9e5), 'SE': (0.2e5, -1.2e5), 'BE': (-1.1e5, 0.3e5), 'PT': (0.1e5, 0)}
    for _, r in tgt.iterrows():
        geom = by_name([r['name']]).geometry.iloc[0]
        p = geom.representative_point() if r.domain == 'SE' else geom.centroid
        dx, dy = nudge.get(r.domain, (0, 0))
        halo_text(ax, p.x + dx, p.y + dy, us.dlabel(r.domain), C['TEXT'] if r.domain in prim else C['MUTED'],
                  fontsize=6, ha='center', va='center')
    x0, x1, y0, y1 = EU_EXT
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)

    if len(src):
        handles = [Patch(facecolor=FILL_TGT, edgecolor='none'), Patch(facecolor=FILL_SRC, edgecolor='none')]
        labels = [f'target {us.META[task]["domain"]} ({len(tgt)})',
                  f'source-only {us.META[task]["domain"]} ({len(src)})']
    else:
        handles = [Patch(facecolor=FILL_TGT, edgecolor='none'),
                   Patch(facecolor=tint(FILL_TGT, 0.3), edgecolor=FILL_TGT, hatch=HATCH, linewidth=0)]
        labels = [f'{us.TERM["primary"]} ({len(prim)} {us.META[task]["domains"]})',
                  f'{", ".join(us.others(task))} (not in {us.TERM["primary"]})']
    return handles, labels


# 世界图上目标城市名放在放大框之外（Amsterdam 在框左，Qingdao 在框上），不压框线
CITY_OFF = {'amsterdam': (-9.5, 3.0, 'right'), 'qingdao': (0.0, 12.0, 'center'), 'melbourne': (-4.0, -2.5, 'right')}
CITY_OFF_INSET = {'amsterdam': (0.0, 5.5, 'center'), 'qingdao': (4.0, 3.0, 'left')}
# lcz：世界图压扁后下方有空位，用两个区域放大图（源城市密集处，经纬度框）显示目标城市与近邻源城市
INSETS = {'lcz': [('Europe', (-12.0, 35.0, 34.0, 58.0)), ('East Asia', (106.0, 144.0, 18.0, 44.0))]}
INSET_EDGE = '#9A9A9A'


def city_points(d):
    return gpd.GeoDataFrame(d, geometry=gpd.points_from_xy(d.lon, d.lat), crs=4326).to_crs(8857)


def draw_cities(ax, pts, offsets=CITY_OFF):
    """源城市 S2 色小点，目标城市 LABEL 色大点并标名（名字放在海面一侧；轴外的目标不标）。"""
    src = pts[pts.role == 'source']
    tgt = pts[pts.role == 'target']
    ax.scatter(src.geometry.x, src.geometry.y, s=6, color=C['S2'], linewidths=0, zorder=4)
    ax.scatter(tgt.geometry.x, tgt.geometry.y, s=20, color=C['LABEL'], edgecolors='white', linewidths=0.5, zorder=5)
    for _, r in tgt.iterrows():
        dx, dy, ha = offsets.get(r.domain, (4.0, 0.0, 'left'))
        for kw in (dict(color='white', path_effects=HALO, zorder=7), dict(color=C['TEXT'], zorder=8)):
            ax.annotate(r['name'], (r.geometry.x, r.geometry.y), xytext=(dx, dy), textcoords='offset points',
                        fontsize=6, ha=ha, va='center', annotation_clip=True, **kw)
    return src, tgt


def draw_world(ax, task, d):
    """lcz：世界 Equal Earth。"""
    WEE.plot(ax=ax, facecolor=LAND, edgecolor='white', linewidth=0.3)
    src, tgt = draw_cities(ax, city_points(d))
    x0, x1, y0, y1 = WEE_EXT
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    dot = lambda col, ms: Line2D([], [], marker='o', ls='none', ms=ms, mfc=col, mec='none')
    return ([dot(C['LABEL'], 3.6), dot(C['S2'], 2.2)],
            [f'target {us.META[task]["domain"]} ({len(tgt)})', f'source {us.META[task]["domain"]} ({len(src)})'])


def draw_inset(ax, d, title, ll):
    """区域放大图：同一投影与点样式；数据范围按轴框比例外扩（adjustable='datalim'），返回最终范围。"""
    lo0, lo1, la0, la1 = ll
    lon = np.r_[np.linspace(lo0, lo1, 50), np.full(50, lo1), np.linspace(lo1, lo0, 50), np.full(50, lo0)]
    lat = np.r_[np.full(50, la0), np.linspace(la0, la1, 50), np.full(50, la1), np.linspace(la1, la0, 50)]
    x, y = _te.transform(lon, lat)
    WEE.cx[x.min() - 2e6:x.max() + 2e6, y.min() - 2e6:y.max() + 2e6].plot(
        ax=ax, facecolor=LAND, edgecolor='white', linewidth=0.3)
    draw_cities(ax, city_points(d), CITY_OFF_INSET)
    # 按轴框长宽比把经纬度框外扩成同比例的数据范围（等比例，不留白）
    bb = ax.get_position(original=True)            # geopandas 绘图后活动位置已被改动，取原始轴框
    box_ar = (bb.width * FW) / (bb.height * FH)
    cx_, cy_, dx, dy = (x.min() + x.max()) / 2, (y.min() + y.max()) / 2, x.max() - x.min(), y.max() - y.min()
    dx, dy = (dy * box_ar, dy) if dx / dy < box_ar else (dx, dx / box_ar)
    ax.set_xlim(cx_ - dx / 2, cx_ + dx / 2)
    ax.set_ylim(cy_ - dy / 2, cy_ + dy / 2)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color(INSET_EDGE)
        s.set_linewidth(0.4)
    ax.text(0.0, 1.0, title, transform=ax.transAxes, fontsize=FS['small'], ha='left', va='bottom', color=C['TEXT'])
    return ax.get_xlim() + ax.get_ylim()


def panel_a(cv, task, d):
    ext = WEE_EXT if task == 'lcz' else EU_EXT
    ar = (ext[1] - ext[0]) / (ext[3] - ext[2])
    cv.text(AX0 + 1.0, ROW0, NOTE[task], fontsize=FS['small'], ha='left', va='top', color=C['TEXT'],
            linespacing=1.25)
    map_top = ROW0 - 2 * LH_SMALL - 1.4
    leg_h = 2 * LH_SMALL + 0.6
    leg_top = BOTTOM + leg_h                 # 图例固定在面板底部（三个任务同一位置）
    h_av = map_top - leg_top - 1.0
    w = min(AW, h_av * ar)
    h = w / ar
    ax = cv.ax(AX0, map_top - h, w, h)
    ax.set_aspect('equal')
    ax.set_axis_off()
    handles, labels = (draw_world if task == 'lcz' else draw_europe)(ax, task, d)
    insets = INSETS.get(task, [])
    if insets:
        gap, title_h = 3.0, 2.6
        ins_top = map_top - h - 1.2 - title_h
        ih = ins_top - leg_top - 1.0
        iw = (AW - gap * (len(insets) - 1)) / len(insets)
        for i, (title, ll) in enumerate(insets):
            axi = cv.ax(AX0 + i * (iw + gap), ins_top - ih, iw, ih)
            x0, x1, y0, y1 = draw_inset(axi, d, title, ll)
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor='none', edgecolor=INSET_EDGE,
                                   linewidth=0.4, zorder=3))
    ax_leg = cv.ax(AX0 + 1.0, BOTTOM, AW - 1.0, leg_h)
    map_legend(ax_leg, handles, labels)
    cv.panel(ax, 'a', AX0 - 0.5)
    return dict(map_w=w, map_h=h)


# =================================================================== b｜流程示意
NL = None
STY = {'r': {}, 'b': dict(fontweight='bold'), 'i': dict(fontstyle='italic'),
       'bl': dict(fontweight='bold', color=C['LABEL']), 'bli': dict(fontweight='bold', fontstyle='italic', color=C['LABEL']),
       'ae': dict(fontweight='bold', color=C['AE']), 's2': dict(fontweight='bold', color=C['S2'])}
NBSP = '\u00a0'


GLUE = 'glue'      # 第三个元素为 GLUE 的片段与前一个词不断开


def flow_boxes(task):
    """4 个框的文字（契约 UF1-b）。每段为 (文字, 样式[, GLUE])；NL 强制换行；括号内与数字前后用不换行空格。
    设计常量（目标标签权重、k 范围、特征维数）取自 us.DESIGN 与 us.KS。"""
    m, D = us.META[task], us.DESIGN
    # 框 3 的简写须与图例文字所用的 ae_desc、s2_desc 一致
    assert f"{D['ae_dim']} dimensions" in D['ae_desc'] and f"{D['s2_n']} " in D['s2_desc'], D
    pool_h, pool_b = pool_parts(task)
    test_h, test_b = m['test'].split(': ', 1)

    def keep(s):
        out, depth = [], 0
        head, sep, tail = s.rpartition(', ')            # 最后一个逗号之后的短语不断开（如 "1-km buffer"）
        s = head + sep + tail.replace(' ', NBSP) if sep else s
        for ch in s.replace(' × ', f'{NBSP}×{NBSP}'):
            depth += (ch == '(') - (ch == ')')
            out.append(NBSP if (ch == ' ' and depth > 0) else ch)
        return ''.join(out)

    return [
        [(pool_h, 'b'), NL, (keep(pool_b[0].upper() + pool_b[1:]), 'r')],
        [('+', 'bl'), ('k', 'bli'), ('random target labels', 'bl'), NL, (f"(weight {D['target_weight']}),", 'r'),
         ('k', 'i'), (f'={NBSP}{us.KS[0]}–{us.KS[-1]}', 'r', GLUE), NL, ('Reference:', 'b'), (keep(m['ref']), 'r')],
        [('Classifier:', 'b'), ('LightGBM', 'r'), NL, ('Features:', 'b'), ('AE', 'ae'), (f"({D['ae_dim']}-d) or", 'r'),
         ('S2', 's2'), (keep(f"({D['s2_n']} statistics)"), 'r')],
        [(test_h + ':', 'b'), (keep(test_b), 'r'), NL, ('Metric:', 'b'), ('BA at matched OA', 'r')],
    ]


def wrap_runs(cv, tokens, width, fs):
    """按"不可断单元"贪心换行。单元 = 一个词，或被 GLUE 粘在一起的几个词。
    返回 lines：每行是 [(文字, 样式)]，同一样式的相邻词已合并；以及空格宽（mm）。"""
    sp = cv.width_mm('a a', fontsize=fs) - cv.width_mm('aa', fontsize=fs)
    wd = lambda word, st: cv.width_mm(word, fontsize=fs, **STY[st])
    units = []                 # 每个单元：list of (word, style)；NL 原样保留
    for tok in tokens:
        if tok is NL:
            units.append(NL)
            continue
        txt, st = tok[0], tok[1]
        for i, word in enumerate(txt.split(' ')):
            if i == 0 and len(tok) > 2 and tok[2] == GLUE and units and units[-1] is not NL:
                units[-1].append((word, st))
            else:
                units.append([(word, st)])
    lines, x = [[]], 0.0
    for u in units:
        if u is NL:
            lines.append([])
            x = 0.0
            continue
        uw = sum(wd(w_, s_) for w_, s_ in u) + sp * (len(u) - 1)
        need = uw + (sp if lines[-1] else 0.0)
        if lines[-1] and x + need > width:
            lines.append([])
            x, need = 0.0, uw
        lines[-1].extend(u)
        x += need
    merged = []
    for ln in lines:
        out = []
        for word, st in ln:
            if out and out[-1][1] == st:
                out[-1] = (out[-1][0] + ' ' + word, st)
            else:
                out.append((word, st))
        merged.append(out)
    return merged, sp


def panel_b(cv, task):
    fs, LH, PAD, PADX = 6.0, 2.9, 1.9, 2.0
    hb = ROW0 - BOTTOM - 0.3
    ax = cv.ax(BX0, BOTTOM + 0.3, BW, hb)
    ax.set_xlim(0, BW)
    ax.set_ylim(0, hb)
    ax.set_axis_off()
    boxes = [wrap_runs(cv, b, BW - 0.6 - 2 * PADX, fs) for b in flow_boxes(task)]
    heights = [len(lines) * LH + 2 * PAD for lines, _ in boxes]
    gap = (hb - sum(heights)) / (len(boxes) - 1)
    y_cur, rects = hb, []
    for h in heights:
        rects.append((y_cur - h, h))
        y_cur -= h + gap
    for (lines, sp), (yb, h) in zip(boxes, rects):
        ax.add_patch(FancyBboxPatch((0.3, yb), BW - 0.6, h, boxstyle='round,pad=0,rounding_size=1.4',
                                    facecolor='white', edgecolor=C['REF_Z0'], linewidth=0.6))
        for i, ln in enumerate(lines):
            yl = yb + h - PAD - LH * (i + 0.5)
            x = 0.3 + PADX
            for j, (txt, st) in enumerate(ln):
                if j > 0:
                    x += sp
                ax.text(x, yl, txt, fontsize=fs, ha='left', va='center', **{'color': C['TEXT'], **STY[st]})
                x += cv.width_mm(txt, fontsize=fs, **STY[st])
    for (ya, _), (yb_, hb_) in zip(rects[:-1], rects[1:]):
        ax.add_patch(FancyArrowPatch((BW / 2, ya - 0.4), (BW / 2, yb_ + hb_ + 0.4), arrowstyle='-|>',
                                     mutation_scale=6, linewidth=0.6, color=C['REF_Z0'], shrinkA=0, shrinkB=0))
    cv.panel(ax, 'b', BX0 - 2.5)


# =================================================================== c｜示例目标单元布局
K_EX = 200                                             # 示例布局画出的随机标签数（RT200）
RT_KEY = us.TERM['rk'].format(k=K_EX).lstrip('+')      # "200 random labels"：图例键与标题共用


def example_unit(task):
    """示例单元与其所属域（作物的评估单元是国家的一个空间半区，如 BE_F1 → BE）。"""
    unit = us.meta_json()[task]['example_unit']
    return unit, (unit.split('_')[0] if task == 'crop' else unit)


def example_title(task):
    unit, code = example_unit(task)
    return f'Example target: {us.dname(task, code, unit)}', f'{RT_KEY} (draw 1)', code


def scale_bar_pos(pts, ext, L, h_km):
    """比例尺放在 4 个角中点最少的一个（长度 L km；h_km 为文字高度对应的 km）。"""
    x0, x1, y0, y1 = ext
    mx, my = 0.03 * (x1 - x0), 0.03 * (y1 - y0)
    cands = [(x1 - mx - L, y1 - my - h_km), (x1 - mx - L, y0 + my), (x0 + mx, y1 - my - h_km), (x0 + mx, y0 + my)]
    best, nb = None, None
    for bx, by in cands:
        n = ((pts.x_km > bx - mx) & (pts.x_km < bx + L + mx) & (pts.y_km > by - my) & (pts.y_km < by + h_km + my)).sum()
        if nb is None or n < nb:
            best, nb = (bx, by), n
    return best


def scale_bar(ax, bx, by, L, km_per_mm):
    """比例尺：长 L km 的横线，文字居中在线上方（数据坐标，可画在轴外；不加底色）。"""
    ax.plot([bx, bx + L], [by, by], color=C['TEXT'], lw=0.8, solid_capstyle='butt', zorder=6, clip_on=False)
    ax.text(bx + L / 2, by + 0.6 * km_per_mm, f'{L} km', fontsize=FS['small'], ha='center', va='bottom', zorder=6,
            clip_on=False)


def draw_layout(ax, lay, cells, ext, w, h, L, frame=False, bar=True):
    """一张布局图：候选点、测试点、（缓冲带点）、测试格、200 个随机标签；bar=True 时比例尺画在点最少的角。
    w、h 为轴框 mm。"""
    size, off = float(cells.size_km.iloc[0]), float(cells.off_km.iloc[0])
    cx0, cy0 = off + cells.cx * size, off + cells.cy * size
    inside = lambda df: df[(df.x_km > ext[0]) & (df.x_km < ext[1]) & (df.y_km > ext[2]) & (df.y_km < ext[3])]
    # 点大小按显示密度：直径 ≈ 0.35 × 平均点距，限制在 0.2–0.6 mm
    small = inside(lay[lay.role.isin(['candidate', 'test', 'buffer'])])
    d_mm = float(np.clip(0.35 * np.sqrt(w * h / max(len(small), 1)), 0.2, 0.6))
    s_small = (d_mm / 25.4 * 72) ** 2
    pt = lambda r: lay[lay.role == r]
    ax.scatter(pt('candidate').x_km, pt('candidate').y_km, s=s_small, c=CAND, linewidths=0, rasterized=True, zorder=1)
    if (lay.role == 'buffer').any():     # 缓冲带样本：空心，表示既不作候选也不作测试
        ax.scatter(pt('buffer').x_km, pt('buffer').y_km, s=s_small, facecolors='white', edgecolors=CAND,
                   linewidths=0.4, rasterized=True, zorder=1)
    ax.scatter(pt('test').x_km, pt('test').y_km, s=s_small, c=TESTP, linewidths=0, rasterized=True, zorder=1)
    ax.add_collection(PatchCollection([Rectangle((x, y), size, size) for x, y in zip(cx0, cy0)],
                                      facecolor=TESTC, alpha=TESTC_A, edgecolor='none', zorder=2))
    ax.scatter(pt('RT200').x_km, pt('RT200').y_km, s=4, c=C['LABEL'], linewidths=0, zorder=5)
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for sp_ in ax.spines.values():
        sp_.set_visible(frame)
        sp_.set_color(INSET_EDGE)
        sp_.set_linewidth(0.4)
    if bar:
        km_per_mm = (ext[1] - ext[0]) / w
        bx, by = scale_bar_pos(inside(lay), ext, L, 3.0 * km_per_mm)
        scale_bar(ax, bx, by, L, km_per_mm)
    return s_small


def panel_c(cv, task):
    lay = us.read(task, 'example_layout')
    cells = us.read(task, 'example_test_cells')
    size, off = float(cells.size_km.iloc[0]), float(cells.off_km.iloc[0])
    t1, t2, code = example_title(task)
    assert (lay.role == 'RT200').sum() == K_EX, task
    cv.text(CX0, ROW0, t1, fontsize=FS['label'], ha='left', va='top', color=C['TEXT'])
    cv.text(CX0, ROW0 - 6.5 * 1.25 / 72 * 25.4, t2, fontsize=FS['label'], ha='left', va='top', color=C['LABEL'])
    title_h = 2 * 6.5 * 1.25 / 72 * 25.4 + 1.2

    # 范围：样本点与测试格的外包框，外扩半个格
    cx0, cy0 = off + cells.cx * size, off + cells.cy * size
    pad = 0.5 * size
    ext = (min(lay.x_km.min(), cx0.min()) - pad, max(lay.x_km.max(), (cx0 + size).max()) + pad,
           min(lay.y_km.min(), cy0.min()) - pad, max(lay.y_km.max(), (cy0 + size).max()) + pad)
    ar = (ext[1] - ext[0]) / (ext[3] - ext[2])
    roles = [r for r in ('RT200', 'candidate', 'test', 'buffer') if (lay.role == r).any()]
    leg_h = len(roles) * LH_SMALL * 1.12 + 0.4
    cw = R_EDGE - CX0
    top = ROW0 - title_h
    # 图例位置规则：若把图放满整列高度后右侧还剩 ≥ 30 mm，图例放右侧；否则放下方
    h_full = top - BOTTOM
    legend_right = cw - h_full * ar >= 30.0
    if legend_right:
        h = h_full
        w = h * ar
    else:
        h_av = top - (BOTTOM + leg_h + 1.0)
        w = min(cw, h_av * ar)
        h = w / ar
    ax = cv.ax(CX0, top - h, w, h)
    L = SCALE_KM[task]
    s_small = draw_layout(ax, lay, cells, ext, w, h, L)

    # 放大图规则：测试格在主图上 < 1 mm（看不清格子）且右侧有空位时，在右侧加一个放大窗口
    zoom = None
    if legend_right and size * w / (ext[1] - ext[0]) < 1.0:
        x0r = CX0 + w + 4.0
        wr = R_EDGE - x0r
        lab_h = 2.6
        bar_h = 3.4                                                      # 窗外比例尺：文字 + 横线 + 间隙（mm）
        hz = top - lab_h - (BOTTOM + leg_h + 1.2 + bar_h)                # 放大窗口填满比例尺以上的高度
        Z = max(5 * size, round(wr / 3.0 * size / 50.0) * 50.0)          # 窗口宽（km）：放大后每格约 3 mm
        Zy = Z * hz / wr
        ccx, ccy = cx0 + size / 2, cy0 + size / 2
        cnt = [((abs(ccx - x) <= Z / 2) & (abs(ccy - y) <= Zy / 2)).sum() for x, y in zip(ccx, ccy)]
        i = int(np.argmax(cnt))                                          # 测试格最密的窗口
        zext = (ccx.iloc[i] - Z / 2, ccx.iloc[i] + Z / 2, ccy.iloc[i] - Zy / 2, ccy.iloc[i] + Zy / 2)
        axz = cv.ax(x0r, top - lab_h - hz, wr, hz)
        draw_layout(axz, lay, cells, zext, wr, hz, L, frame=True, bar=False)
        # 比例尺画在窗口外下方、右缘对齐（窗内点密，放外面不压点，也不需底色）
        kpm = Z / wr
        scale_bar(axz, zext[1] - L, zext[2] - bar_h * kpm, L, kpm)
        axz.text(0.0, 1.0, 'Enlarged view', transform=axz.transAxes, fontsize=FS['small'], ha='left', va='bottom')
        ax.add_patch(Rectangle((zext[0], zext[2]), Z, Zy, facecolor='none', edgecolor=INSET_EDGE, linewidth=0.5,
                               zorder=7))
        zoom = dict(Z=Z, Zy=Zy, n_cells=int(cnt[i]))

    # 图例
    m = us.META[task]
    dot = lambda col, ms=2.3, mfc=None: Line2D([], [], marker='o', ls='none', ms=ms, mfc=mfc or col, mec=col,
                                                mew=0.5 if mfc else 0)
    sm = m['samples_short']                # 图例键用短词：parcels / patches / polygons（全称写在图例文字里）
    H = {'RT200': (dot(C['LABEL']), RT_KEY),
         'candidate': (dot(CAND), f'candidate {sm}'),
         'test': ((Patch(facecolor=TESTC, alpha=TESTC_A, edgecolor='none'), dot(TESTP, 1.6)),
                  f'test cells ({size:g} km) and test {sm}'),
         'buffer': (dot(CAND, 1.8, 'white'), f'buffer {sm} (unused)')}
    if legend_right:
        ax_leg = cv.ax(CX0 + w + 4.0, BOTTOM, R_EDGE - CX0 - w - 4.0, leg_h)
    else:
        ax_leg = cv.ax(CX0, top - h - 1.0 - leg_h, cw, leg_h)
    map_legend(ax_leg, [H[r][0] for r in roles], [H[r][1] for r in roles])
    cv.panel(ax, 'c', CX0 - 2.5)
    n_role = lay.role.value_counts().to_dict()
    return dict(code=code, n_role=n_role, size=size, ext=ext, w=w, h=h, legend_right=legend_right, s_small=s_small,
                n_cells=len(cells), L=L, zoom=zoom)


# =================================================================== 图例文字
def fmt_int(v):
    return f'{int(v):,}'


def items(pairs):
    """"Name (CODE), n; …" 形式的逐域列表（不嵌套括号）。"""
    return '; '.join(f"{nm}, {fmt_int(n)}" for nm, n in pairs)


def ref_range(task):
    """META ref 中候选标签数的范围 "n = a–b"（作物无此范围，返回 None）。"""
    r = re.search(r'n = ([\d,]+)–([\d,]+)', us.META[task]['ref'])
    return tuple(int(x.replace(',', '')) for x in r.groups()) if r else None


def legend_text(task, d, info):
    m, D, T = us.META[task], us.DESIGN, us.TERM
    ss = m['samples_short']
    prim = us.primary(task)
    tgt = d[d.role.str.startswith('target')]
    src = d[d.role == 'source']
    side = us.others(task)
    nr = info['n_role']
    dn = d.set_index('domain').n
    nm = lambda c: us.dname(task, c)
    head = f"Study design: {m['long'][0].lower() + m['long'][1:]}."
    if task == 'crop':
        regs = sorted(os.path.basename(f).split('_')[-1].split('.')[0].upper()
                      for f in glob.glob(us.src(f'{task}_study_regions_*.geojson')))
        de = gpd.read_file(us.src('crop_study_regions_de.geojson'))
        states = ' and '.join(DE_STATES.get(x, x) for x in de.land)
        a = (f"a, Study area (ETRS89-LAEA, EPSG:3035). Each of the {len(tgt)} countries served in turn as the target "
             f"domain and was mapped from a foreign source pool of the other {len(tgt) - 1} countries; two-letter "
             f"codes mark the targets. Solid fill, the {T['primary']} of {len(prim)} countries; hatched, "
             f"{' and '.join(nm(c) for c in side)}, reported separately. In {' and '.join(nm(c) for c in regs)} only "
             f"the labelled areas are drawn at full tone ({nm('DE')}: {states}; {nm('RO')}: an area in the south of "
             f"the country); the rest of each country is drawn pale. Parcels per country: "
             + items((f"{nm(c)} ({c})", dn[c]) for c in sorted(tgt.domain)) + '.')
    elif task == 'lcz':
        ins = ' and '.join(t for t, _ in INSETS.get(task, []))
        a = (f"a, The {len(d)} cities of the data set (Equal Earth projection, EPSG:8857); the boxed regions "
             f"({ins}) are enlarged below. Large blue dots, the {len(tgt)} target cities "
             f"({', '.join(nm(c) for c in tgt.domain)}; one per region); small grey dots, the {len(src)} source "
             f"cities. Each target city was mapped from the source pool of the other {len(src)} cities "
             f"({fmt_int(src.n.sum())} {ss}). Patches per target city: "
             + items((nm(c), dn[c]) for c in tgt.domain) + '.')
    else:
        order = {'south': 0, 'central-west': 1, 'north': 2}
        tg = [f"{nm(c)} ({c}, {REGION_WORD[c]})" for c in sorted(tgt.domain, key=lambda c: order[REGION_WORD[c]])]
        a = (f"a, Study area (ETRS89-LAEA, EPSG:3035). Dark fill, the {len(tgt)} target countries: "
             f"{', '.join(tg[:-1])} and {tg[-1]}; light fill, the {len(src)} source-only countries. Each target was "
             f"mapped from the source pool of the other {len(src)} countries ({fmt_int(src.n.sum())} survey "
             f"polygons). Survey polygons per target country: "
             + items((f"{nm(c)} ({c})", dn[c]) for c in tgt.domain) + '.')
    pool_h, pool_b = pool_parts(task)
    b = (f"b, Transfer and evaluation protocol. {pool_h}: {pool_b}. The model is trained on the source pool plus "
         f"k random labels from the target domain (k = {', '.join(str(k) for k in us.KS)}; each target label "
         f"carries sample weight {D['target_weight']}); k = 0 is {T['zero']}. Reference arm: {m['ref']}. "
         f"Classifier: LightGBM on either {D['ae_desc']} or {D['s2_desc']}; {m['support']}. {m['test']}. Metric: balanced "
         f"accuracy (BA) compared at matched overall accuracy (OA).")

    # c：候选池 = 候选点 + 已抽中的 200 个标签；布局点少于该域样本总数时，灰点是显示用的随机子样本
    unit, code = example_unit(task)
    disp = sum(nr.values()) < dn[code]
    shown = lambda r: (f"a random display subsample of the {'other ' if r == 'candidate' else ''}{r} {ss} "
                       f"(n = {fmt_int(nr[r])} shown)")
    test_txt = shown('test') if disp else f"test {ss} (n = {fmt_int(nr['test'])})"
    sub = [f"darker grey dots, {test_txt}"]
    if 'buffer' in nr:
        sub.append(f"open circles, {ss} in the buffer around test cells, used neither for training nor for "
                   f"testing (n = {fmt_int(nr['buffer'])})")
    rt = f"blue dots, the first random draw of {fmt_int(nr['RT200'])} labels (k = {K_EX})"
    if disp:
        pool = (f"Candidate pool from which target labels are drawn: {rt}; light grey dots, {shown('candidate')}. "
                f"{nm(code)} has {fmt_int(dn[code])} {ss} in all.")
    else:
        n_pool = nr['candidate'] + nr['RT200']
        rr = ref_range(task)                         # 示例单元的候选池须落在 REF 臂的候选标签数范围内
        assert rr is None or rr[0] <= n_pool <= rr[1], (task, n_pool, rr)
        pool = (f"Candidate pool from which target labels are drawn (n = {fmt_int(n_pool)} {ss}): {rt}; light "
                f"grey dots, the other {fmt_int(nr['candidate'])} candidate {ss}.")
    zoom = ''
    if info['zoom']:
        z = info['zoom']
        zoom = (f" Right, enlarged view of a {z['Z']:.0f} km × {z['Zy']:.0f} km window (outlined in the main map) "
                f"placed where test cells are densest.")
    c = (f"c, Example target domain: {us.dname(task, code, unit)}. Shaded squares, test cells ({info['size']:g} km; "
         f"{info['n_cells']} cells); " + '; '.join(sub) + f". {pool}{zoom} Scale bar{'s' if zoom else ''}, "
         f"{info['L']} km{' (that of the enlarged view is drawn below it)' if zoom else ''}.")
    dom_txt = f"{len(prim)} target {m['domains'] if len(prim) > 1 else m['domain']}"
    units = (f"Evaluation units: {m['units_note']}. Domains: {dom_txt}"
             + (f" in the {T['primary']}; {' and '.join(side)} reported separately." if side else '.'))
    stats = ("This figure shows no estimates. In all result figures, means are shown with 95% t confidence "
             f"intervals only when at least {us.CI_MIN_N} domains are available; with fewer domains the value of "
             "each domain is plotted instead.")
    return '\n\n'.join([head, a, b, c, units + ' ' + stats])


# =================================================================== 主循环
for task in us.TASKS:
    d = domains(task)
    check_note(task, d)
    cv = Canvas()
    panel_a(cv, task, d)
    panel_b(cv, task)
    info = panel_c(cv, task)
    name = f'uf1_design_{task}'
    us.save(cv.fig, name)
    us.write_legend(name, legend_text(task, d, info))
    print(f'saved {name}: c legend_right={info["legend_right"]}, map {info["w"]:.1f}×{info["h"]:.1f} mm, '
          f's_small={info["s_small"]:.2f}, roles={info["n_role"]}')
