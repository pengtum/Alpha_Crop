"""真实地块标签与质心邻域示意；继承论文配色，不生成模拟观测值。"""
from pathlib import Path
import sys
import json
import numpy as np
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Patch
from PIL import Image
from shapely.geometry import box

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path.insert(0, str(ROOT / 'label_expansion/storyline/rse_figs'))
from nstyle import setup, C, MM, FONT
SKILL = ROOT.parent / 'nature-skills-main/skills/nature-figure/scripts'
sys.path.insert(0, str(SKILL))
from audit_panel_alignment import require_matplotlib_panel_alignment

setup()
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': [FONT, 'Arial', 'Helvetica'],
                     'svg.fonttype': 'none', 'pdf.fonttype': 42,
                     'font.size': 6.5, 'savefig.dpi': 600})
SOURCE = HERE / 'source_data'
OUT = HERE / 'out'
META = json.loads((SOURCE / 'metadata.json').read_text())
PARCELS = gpd.read_file(SOURCE / 'parcels.geojson').to_crs(31370)
EXAMPLE = PARCELS[PARCELS.in_selected_record]
RGB = np.asarray(Image.open(SOURCE / 'orthophoto_2021.png').convert('RGB'))
X, Y = META['centroid_lambert72']
EXTENT = META['extent_lambert72']
COLORS = {'grassland': '#8FAE9B', 'maize': '#7E9FC0', 'winter_cereals': '#D4BC7F',
          'potatoes': '#B2A3C4', 'root_crops': '#C6A18B', 'vegetables': '#B98489',
          'orchards': '#83A6A4', 'other_crops': '#BBBBBB',
          'tree_wood_forest': '#6C8A77', 'strawberries': '#A87892',
          'fallow': '#D8D8D8', 'unknown': '#DDDDDD'}


def text(fig, x, y, s, width, height, **kw):
    return fig.text(x / width, y / height, s, ha='left', va='center', **kw)


def title(ax, letter, s):
    ax.text(0, 1.035, letter, transform=ax.transAxes, fontsize=8,
            fontweight='bold', ha='left', va='bottom')
    ax.text(0.07, 1.035, s, transform=ax.transAxes, fontsize=7,
            ha='left', va='bottom')


def map_image(ax, extent, aerial=True):
    if aerial:
        ax.imshow(RGB, extent=[EXTENT[0], EXTENT[2], EXTENT[1], EXTENT[3]],
                  origin='upper', interpolation='nearest', zorder=0)
    ax.set_xlim(extent[0], extent[2])
    ax.set_ylim(extent[1], extent[3])
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.45)
        sp.set_color('#BDBDBD')


def scale(ax, length, white=False):
    lo, hi = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    x = lo + 0.065 * (hi - lo)
    y = y0 + 0.075 * (y1 - y0)
    color = 'white' if white else C['TEXT']
    ax.plot([x, x + length], [y, y], lw=1.6, color=color, solid_capstyle='butt')
    ax.text(x + length / 2, y + 0.027 * (y1-y0), f'{length:g} m',
            ha='center', va='bottom', color=color, fontsize=6)


def export(fig, name, axes):
    fig.canvas.draw()
    require_matplotlib_panel_alignment(
        fig, axes=axes, panel_ids=list('abc')[:len(axes)],
        row_groups=[list('abc')[:len(axes)]], require_panel_labels=True,
        json_out=OUT / f'{name}.alignment.json',
        overlay_svg=OUT / f'{name}.alignment.svg', strict=True,
        tolerance_pt=1.5, gutter_tolerance_pt=1.5)
    fig.savefig(OUT / f'{name}.pdf', facecolor='white')
    fig.savefig(OUT / f'{name}.svg', facecolor='white')
    fig.savefig(OUT / f'{name}.png', dpi=300, facecolor='white')
    fig.savefig(OUT / f'{name}.tiff', dpi=600, facecolor='white',
                pil_kwargs={'compression': 'tiff_lzw'})
    plt.close(fig)


def parcel_labels():
    w, h = 183, 128
    fig = plt.figure(figsize=(w*MM, h*MM))
    axes = [fig.add_axes([x/w, 32/h, 84/w, 84/h]) for x in [4, 95]]
    a, b = axes
    shown = PARCELS.copy()
    shown.geometry = shown.geometry.intersection(box(*EXTENT))
    map_image(a, EXTENT)
    shown.boundary.plot(ax=a, color='white', linewidth=0.35, alpha=0.8)
    EXAMPLE.boundary.plot(ax=a, color='white', linewidth=2)
    EXAMPLE.boundary.plot(ax=a, color=C['LABEL'], linewidth=1.1)
    map_image(a, EXTENT)
    # 图像重复设置后，边界仍位于较高图层；不调整原影像亮度。
    a.text(X, Y, 'P1', color='white', weight='bold', fontsize=7,
           ha='center', va='center')
    scale(a, 100, white=True)
    title(a, 'a', 'Fields and recorded parcel boundaries')
    shown.plot(ax=b, color=[COLORS.get(v, '#DDDDDD') for v in shown.analysis_class],
                 edgecolor='white', linewidth=0.45)
    EXAMPLE.boundary.plot(ax=b, color=C['LABEL'], linewidth=1.2)
    map_image(b, EXTENT, aerial=False)
    b.text(X, Y, 'P1', weight='bold', fontsize=7, ha='center', va='center')
    scale(b, 100)
    title(b, 'b', 'Crop labels attached to the same parcels')
    # 显示全部窗口内类别，不将示例区域用作类别频率估计。
    classes = [k for k in COLORS if k in set(PARCELS.analysis_class)]
    classes += sorted(set(PARCELS.analysis_class) - set(classes))
    handles = [Patch(facecolor=COLORS.get(k, '#DDDDDD'), edgecolor='none',
                     label=k.replace('_', ' ').capitalize()) for k in classes]
    fig.legend(handles=handles, loc='center', bbox_to_anchor=(0.5, 26/h),
               ncol=min(3, len(handles)), fontsize=6, handlelength=1,
               handletextpad=0.45, columnspacing=1.3, labelspacing=0.6)
    yhead, yval = 17, 11
    for x, head, value in [
        (4, 'Example parcel', f"P1  |  {META['area_ha']:.2f} ha"),
        (37, 'Farmer declaration (Dutch)', META['source_record']['GWSNAM_H']),
        (91, 'Harmonised crop identity', 'Winter common soft wheat'),
        (150, 'Analysis label', 'Winter cereals')]:
        text(fig, x, yhead, head, w, h, fontsize=5.5, color=C['MUTED'])
        text(fig, x, yval, value, w, h, fontsize=6.5)
    text(fig, 4, 3.5, 'Tielt area, Belgium, 2021  |  Labels: EuroCrops / Flemish crop declarations  |  Orthophoto: Digitaal Vlaanderen',
         w, h, fontsize=5.5, color=C['MUTED'])
    export(fig, 'fig_parcel_ground_reference', axes)


def arrow(ax, start, end, color=C['MUTED']):
    ax.annotate('', xy=end, xytext=start,
                arrowprops={'arrowstyle': '-|>', 'lw': 0.65, 'color': color,
                            'shrinkA': 0, 'shrinkB': 0, 'mutation_scale': 7})


def vector_glyph(ax, y, n, name, color):
    ax.text(0.04, y, name, color=color, fontsize=7, va='center')
    # 空白单元只表示维度，不编码或捏造特征值。
    for i in [0, 1, 2, 4, 5, 6]:
        for j in range(2):
            ax.add_patch(Rectangle((0.32+i*0.078, y-0.09-j*0.085), 0.066, 0.064,
                                   facecolor='none', edgecolor=color, lw=0.5))
    for j in range(2):
        ax.text(0.32+3*0.078+0.033, y-0.058-j*0.085, '…',
                fontsize=6.5, ha='center', va='center', color=color)
    ax.text(0.205, y-0.058, 'Mean', fontsize=5.5, ha='right', va='center')
    ax.text(0.205, y-0.143, 'SD', fontsize=5.5, ha='right', va='center')
    ax.text(0.595, y-0.218, f'{n} channels × 2 statistics = {n*2}',
            fontsize=6.5, color=color, ha='center', va='center')


def centroid_features():
    w, h = 183, 87
    fig = plt.figure(figsize=(w*MM, h*MM))
    axes = [fig.add_axes([x/w, 22/h, 54/w, 54/h]) for x in [4, 65, 126]]
    a, b, c = axes
    bounds = EXAMPLE.total_bounds
    side = max(bounds[2]-bounds[0], bounds[3]-bounds[1]) + 80
    extent = [X-side/2, Y-side/2, X+side/2, Y+side/2]
    map_image(a, extent)
    EXAMPLE.boundary.plot(ax=a, color='white', linewidth=2.2)
    EXAMPLE.boundary.plot(ax=a, color=C['LABEL'], linewidth=1.25)
    a.add_patch(Circle((X, Y), 30, fill=False, edgecolor='white', lw=0.9,
                       linestyle=(0, (3, 2)), zorder=6))
    a.scatter([X], [Y], s=13, color='white', edgecolor=C['LABEL'], lw=0.7, zorder=7)
    map_image(a, extent)
    title(a, 'a', 'Parcel and saved centroid')
    scale(a, 50, white=True)
    # 名义 10 m 网格示意，不冒充真实 Earth Engine 网格配准。
    b.set_xlim(-45, 45)
    b.set_ylim(-45, 45)
    b.set_aspect('equal')
    for i in range(-3, 4):
        for j in range(-3, 4):
            inside = i*i + j*j <= 9
            b.add_patch(Rectangle((i*10-5, j*10-5), 10, 10,
                                  facecolor='#D9E6F2' if inside else '#F7F7F7',
                                  edgecolor='#CACACA', lw=0.35))
    b.add_patch(Circle((0, 0), 30, fill=False, edgecolor=C['LABEL'],
                       lw=0.85, linestyle=(0, (3, 2))))
    b.scatter([0], [0], color=C['LABEL'], s=15, zorder=8)
    arrow(b, (0, 0), (21.2, 21.2), C['LABEL'])
    b.text(-42, 40, 'Radius: 3 pixels (nominally 30 m)', fontsize=5.5,
           ha='left', va='center', color=C['LABEL'])
    b.plot([-35, -25], [-40, -40], color=C['TEXT'], lw=0.65)
    b.text(-21, -40, '10 m requested scale', fontsize=5.5, ha='left', va='center')
    b.set_xticks([])
    b.set_yticks([])
    for sp in b.spines.values():
        sp.set_visible(False)
    title(b, 'b', 'Circular neighbourhood')
    c.set_xlim(0, 1)
    c.set_ylim(0, 1)
    c.axis('off')
    title(c, 'c', 'Per-parcel feature vectors')
    vector_glyph(c, 0.89, 64, 'AE', C['AE'])
    vector_glyph(c, 0.48, 84, 'S2', C['S2'])
    c.text(0.5, 0.07, 'Paired with the same parcel label', ha='center', fontsize=6.5,
           color=C['LABEL'])
    text(fig, 4, 16.5, 'P1: Winter cereals', w, h, fontsize=6.5, color=C['LABEL'])
    text(fig, 65, 16.5, 'Mean and SD within each channel', w, h, fontsize=6)
    text(fig, 126, 16.5, 'AE: annual embedding coordinates', w, h, fontsize=5.5)
    text(fig, 126, 11.5, 'S2: temporal summaries before spatial reduction', w, h, fontsize=5.5)
    text(fig, 4, 5, 'Grid geometry is schematic; the neighbourhood is not clipped to the parcel boundary.',
         w, h, fontsize=5.5, color=C['MUTED'])
    export(fig, 'fig_centroid_feature_extraction', axes)


if __name__ == '__main__':
    parcel_labels()
    centroid_features()
    print('Two figure bundles exported.')
