"""RSE 故事线配图的共享样式（Nature 风格；所有图都 import 本模块，保证字体、字号、颜色语义与导出一致）。

颜色语义全文固定（颜色跟随实体，不跟随排名）：
- 标签杠杆（蓝系）：随机标签/场景 A = LABEL；已有调查 + 全国随机补采 = SPREAD；已有调查 + Betti 分配 = BETTI
- 集中样本（已有调查本身、原区扩采）= CONC
- 表征杠杆：AE = AE；S2 = S2
- 参照：零标签迁移 Z0 = REF_Z0；只用目标国训练的上界 TGT = REF_TGT
- 绿/红只用于增益/损失的方向提示
"""
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager

MM = 1 / 25.4
W1, W15, W2 = 89 * MM, 120 * MM, 183 * MM          # 单栏、1.5 栏、双栏（英寸）

C = dict(
    LABEL='#0F4D92',       # 随机标签（场景 A）
    SPREAD='#3775BA',      # 已有调查 + 全国随机补采（S0 + SRS）
    BETTI='#7884B4',       # 已有调查 + Betti 分配（S0 + BA）
    CONC='#9A4D8E',        # 集中样本（S0、S0 + EXT）
    AE='#42949E',          # AlphaEarth
    S2='#767676',          # Sentinel-2 物候统计
    REF_Z0='#4D4D4D',      # 零标签迁移
    REF_TGT='#272727',     # 只用目标国（约 3 万标签）
    GRID='#CFCECE',
    LIGHT='#E6E6E6',
    TEXT='#272727',
    MUTED='#767676',
    UP='#2E9E44',
    DOWN='#B64342',
    BAND='#EFEFEF',
)

_fonts = {f.name for f in font_manager.fontManager.ttflist}
FONT = 'Arial' if 'Arial' in _fonts else ('Helvetica' if 'Helvetica' in _fonts else 'DejaVu Sans')

FS = dict(base=8, tick=7.5, label=8, title=9, panel=10, small=7.5)


def setup():
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': [FONT, 'Helvetica', 'DejaVu Sans'],
        'svg.fonttype': 'none',
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'font.size': FS['base'],
        'axes.labelsize': FS['label'],
        'axes.titlesize': FS['title'],
        'axes.titleweight': 'normal',
        'xtick.labelsize': FS['tick'],
        'ytick.labelsize': FS['tick'],
        'legend.fontsize': FS['tick'],
        'legend.frameon': False,
        'axes.spines.right': False,
        'axes.spines.top': False,
        'axes.linewidth': 0.6,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'xtick.major.size': 2.5,
        'ytick.major.size': 2.5,
        'xtick.minor.size': 1.5,
        'ytick.minor.size': 1.5,
        'lines.linewidth': 1.0,
        'lines.markersize': 3.5,
        'axes.edgecolor': C['TEXT'],
        'axes.labelcolor': C['TEXT'],
        'xtick.color': C['TEXT'],
        'ytick.color': C['TEXT'],
        'text.color': C['TEXT'],
        'figure.dpi': 150,
        'savefig.dpi': 600,
        'axes.unicode_minus': True,
        'hatch.linewidth': 0.5,
    })


def panel(ax, label, x=-0.14, y=1.04):
    """面板标号：小写粗体 8 pt，放在轴左上角外侧。"""
    ax.text(x, y, label, transform=ax.transAxes, fontsize=FS['panel'], fontweight='bold',
            ha='left', va='bottom', color='black')


def zero_line(ax, axis='y', v=0.0):
    (ax.axhline if axis == 'y' else ax.axvline)(v, color=C['MUTED'], lw=0.5, ls=(0, (2, 2)), zorder=0)


def save(fig, name, out_dir=None):
    """导出 SVG（可编辑文字）、PDF（TrueType）、TIFF（600 dpi）与 PNG（300 dpi，供 HTML 使用）。"""
    out_dir = out_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out')
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, name)
    from layout_readability import finalize
    finalize(fig, name, base)
    fig.savefig(base + '.svg', bbox_inches='tight', pad_inches=0.02)
    fig.savefig(base + '.pdf', bbox_inches='tight', pad_inches=0.02)
    fig.savefig(base + '.tiff', dpi=600, bbox_inches='tight', pad_inches=0.02, pil_kwargs={'compression': 'tiff_lzw'})
    fig.savefig(base + '.png', dpi=300, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    return base


def src(name):
    return '/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops/Manuscript/Paper_Alpha_Earth/0004_representation_labels_rse/figure_sources/regional_allocation/source_data' + "/" + name


def fmt_pp(v, signed=True, nd=2):
    s = f'{v:+.{nd}f}' if signed else f'{v:.{nd}f}'
    return s.replace('-', '−')

"""只调整已有图形对象的字号与布局，保留数据、区间和图元。"""
from pathlib import Path
import sys
import matplotlib.pyplot as plt
from matplotlib.text import Text
from matplotlib.lines import Line2D
MM = 1/25.4
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT.parent/'nature-skills-main/skills/nature-figure/scripts'))
from audit_panel_alignment import require_matplotlib_panel_alignment

def finalize(fig,name,base):
    for t in fig.findobj(Text):
        if t.get_visible() and t.get_text().strip():t.set_fontsize(max(7.5,t.get_fontsize()))
    aa=fig.axes
    ids=[]; rows=[]; cols=[]; exemptions=[]
    def size(w,h):fig.set_size_inches(w*MM,h*MM,forward=True)
    def pos(ax,x,y,w,h):
        W,H=fig.get_size_inches()/MM
        ax.set_position([x/W,y/H,w/W,h/H])
    def label(ax,l):
        for t in list(ax.texts):
            if t.get_text() in list('abcdef') and t.get_fontweight()=='bold':t.remove()
        ax.annotate(l,(0,1),xycoords='axes fraction',xytext=(0,8),textcoords='offset points',fontsize=10,fontweight='bold',va='bottom')
    if name=='fig2_evaluation':
        size(183,126)
        a,b,c=aa
        pos(a,14,70,75,46.5);pos(b,104,70,75,46.5);pos(c,61,12,67,40)
        for t in c.texts:
            tx=t.get_text()
            x,y=t.get_position()
            if tx in ['a','b','c']:continue
            if 'countries' in tx:t.set_position((-10.5,y));t.set_fontsize(8)
            elif tx.startswith('Δ'):t.set_position((-1.8,y))
            elif '[' in tx or tx=='mean [95% CI]':t.set_position((10.1,y))
            elif '/' in tx or tx=='> 0':t.set_position((16.4,y))
        for ax,l in zip(aa,'abc'):label(ax,l)
        ids=list('abc');rows=[['a','b']]
    elif name=='fig6_representation':
        size(183,192)
        a,b,c,d,e,f=aa
        coords=[(37,140,46,40),(113,140,61,40),(37,77,46,37),(113,77,61,37),(37,14,46,43),(114,14,34,43)]
        for ax,co in zip(aa,coords):pos(ax,*co)
        for t in list(fig.texts):t.remove()
        # 六面板按原字母顺序排成两列三行。
        for ax,l in zip(aa,'abcdef'):label(ax,l)
        for ax in (b,c):ax.set_xlabel('ΔBA (%)')
        e.set_xlabel('ΔBA interaction (%)');f.set_xlabel('ΔBA (%)')
        # 两行汇总统计兼作图例，保留原图的均值、置信区间和国家计数。
        for t in b.texts:
            if t.get_text().startswith('LightGBM '):t.set_position((.06,-.31))
            elif t.get_text().startswith('RF '):t.set_position((.06,-.42))
        for i,ln in enumerate([ln for ln in b.lines if ln.get_transform()==b.transAxes]):
            ln.set_ydata([-.31 if i==0 else -.42])
        for t in c.texts:
            if t.get_text()=='11 countries':t.set_position((.99,1.03))
        for ax in (a,c,e):
            for t in ax.texts:
                if hasattr(t,'xy') and t.get_text() not in list('abcdef') and t.xy[0] is not None:
                    x,y=t.xy
                    if isinstance(x,(int,float)) and abs(x)<1.0:
                        t.set_position((8 if x>=0 else -8,2.4));t.set_ha('left' if x>=0 else 'right')
        f.text(.22,1.14,'11 countries; 90% CI',transform=f.transAxes,fontsize=7.5,color='#767676')
        ids=list('abcdef');rows=[['a','b'],['c','d'],['e','f']]
    elif name=='fig4_existing_survey':
        size(183,132)
        a,b,c=aa
        pos(a,15,76,68,44);pos(b,125,76,48,44);pos(c,70,13,103,43)
        for t in c.texts:
            if t.get_text().startswith('vs '):t.set_position((-.56,t.get_position()[1]))
        for ax,l in zip(aa,'abc'):label(ax,l)
        ids=list('abc');rows=[['a','b']]
    elif name=='fig5_exchange_cost':
        size(183,142)
        a,b,c,d,e=aa
        pos(a,15,84,64,41);pos(b,109,84,65,41)
        for ax,x in zip([c,d,e],[15,72,129]):pos(ax,x,18,45,43)
        for leg in list(fig.legends):leg.remove()
        from nstyle import C
        hs=[Line2D([],[],color=C[k],marker='o',ms=3, label=s) for k,s in [('LABEL','Random labels'),('CONC','Survey / local extension'),('SPREAD','Survey + national spread'),('BETTI','Survey + regional allocation')]]
        hs.append(Line2D([],[],color='black',ls='',marker='*',label='Best at each budget'))
        fig.legend(handles=hs,loc='upper center',bbox_to_anchor=(.5,1.01),ncol=3,fontsize=7.5,columnspacing=1.2,handlelength=1.4)
        for t in list(fig.texts):t.remove()
        fig.text(.52,71/142,'c = cost per spread label / cost per concentrated label',ha='center',fontsize=7.5,color='#767676')
        for ax,l in zip([a,b,c],'abc'):label(ax,l)
        d.set_xlabel('Budget (cost units of one concentrated label)')
        ids=['a','b','c1','c2','c3'];rows=[['a','b'],['c1','c2','c3']]
    elif name=='fig3_labels_zero':
        # 第一、三幅图放大；中间条形图用更短标签释放绘图区。
        size(183,111)
        a,b,c=aa
        pos(a,30,61,56,40);pos(b,30,12,56,34);pos(c,107.5,33,75,68)
        c.set_ylim(-3,7.4);c.set_aspect('equal')
        c.set_xlabel('LightGBM ΔBA (%)');c.set_ylabel('Random forest ΔBA (%)')
        a.set_xticks([0,100,200,400])
        a.get_legend().set_bbox_to_anchor((0,-.01))
        a.get_legend()._loc=3
        for t in a.texts:
            if t.get_text().startswith('11 countries:'):t.set_position((320,-.4));t.set_va('top')
        # 图例放在两条附加国家曲线之间的空白区域。
        a.set_ylim(-8.5,7.2);a.get_legend().set_bbox_to_anchor((.02,.01))
        for t in c.texts:
            if t.get_text()=='y = x':t.set_position((-2.5,-1.5))
            if t.get_text()=='SK':t.set_position((4.17,3.35));t.set_ha('center');t.set_va('bottom')
        for t in list(b.texts):
            if t.get_text().startswith('share of'):t.set_text('Gap closed');t.set_position((6.2,4.05))
            if t.get_text().startswith('+16.'):
                t.set_position((.4,5.8));t.set_va('bottom');t.set_fontsize(7.5)
        b.set_yticklabels(['Target-only\n(~30k labels)']+[f'+{k} labels' for k in [50,100,200,400]])
        for t in c.texts:
            tx=t.get_text()
            if tx.startswith('+200 random labels;'):t.set_text('+200 labels; 9-country means')
        for ax,l in zip(aa,'abc'):label(ax,l)
        ids=list('abc');cols=[['a','b']]
    elif name=='fig_zero_label_accuracy':
        size(183,84)
        for ax,x in zip(aa,[12,73,134]):pos(ax,x,16,45,57)
        for ax,l in zip(aa,'abc'):label(ax,l)
        for t in list(fig.texts):t.remove()
        aa[0].set_xlabel('Overall accuracy (%)');aa[1].set_xlabel('Balanced accuracy (%)');aa[2].set_xlabel('ΔBA (%)')
        ids=list('abc');rows=[ids]
    elif name=='fig_zero_label_confusion':
        size(183,113)
        pos(aa[0],29,39,61,61);pos(aa[1],119,39,61,61);pos(aa[2],48,7,88,1.8)
        for t in list(fig.texts):t.remove()
        for ax,l in zip(aa[:2],'ab'):
            label(ax,l)
            for t in ax.get_xticklabels():t.set_rotation(90);t.set_ha('center');t.set_va('top');t.set_rotation_mode('default')
        ids=['a','b'];rows=[ids];aa=aa[:2]
    elif name=='figS4_strategies':
        size(150,78)
        pos(aa[0],33,12,103,60)
        for t in aa[0].texts:
            if t.get_text().startswith('countries'):t.set_text('> 0')
        ids=['a'];rows=[]
    elif name=='fig1_design':
        # 补图的四幅目标区地图保留原有范围，字体增大；总图紧裁边。
        ids=['map','protocol']+[f'local{i+1}' for i in range(4)]
        aa=[fig.axes[i] for i in [0,2,3,4,5,6]]
        rows=[['local1','local2','local3','local4']]
    if not ids:ids=[str(i) for i in range(len(aa))]
    # 实际裁切轴外折线，避免 PDF 的隐藏路径干扰碰撞检查。
    from shapely.geometry import LineString, box
    import numpy as np
    for ax in fig.axes:
        for ln in ax.lines:
            if ln.get_transform()!=ax.transData or not ln.get_clip_on():continue
            xy=ln.get_xydata()
            if len(xy)<2 or not np.isfinite(xy).all():continue
            lim=box(*[min(ax.get_xlim()),min(ax.get_ylim()),max(ax.get_xlim()),max(ax.get_ylim())])
            if lim.covers(LineString(xy)):continue
            cut=LineString(xy).intersection(lim)
            parts=[cut] if cut.geom_type=='LineString' else list(cut.geoms) if hasattr(cut,'geoms') else []
            coords=[]
            for part in parts:
                if part.geom_type=='LineString':coords.extend(list(part.coords)+[(float('nan'),float('nan'))])
            if coords:ln.set_data(np.array(coords).T)
    for t in fig.findobj(Text):
        if '\n' in t.get_text():t.set_linespacing(1.3)
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,axes=aa,panel_ids=ids,row_groups=rows,column_groups=cols,
        strict=True,exemptions=exemptions,json_out=base+'.alignment.json',overlay_svg=base+'.alignment.svg')

"""Fig. 6 RQ3：换表征与加标签（双栏 183 × 115 mm，两行六个面板）。

核心句：零标签时把 S2 物候统计换成 AE 嵌入，同 OA 下 BA 提高约 5.4 pp，超过 400 个随机标签的增益；
优势在加标签后仍在、与标签可叠加、换 RF 也成立；但 AE 作选样指导时并不比 S2 更好。

只读 source_data/：confirm_estimates.csv、confirm_country_values.csv、guidance_theta.csv。
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import nstyle as ns

ns.setup()
C, FS, MM = ns.C, ns.FS, ns.MM

# ------------------------------------------------------------------ 数据
est = pd.read_csv(ns.src('confirm_estimates.csv'))
cv = pd.read_csv(ns.src('confirm_country_values.csv'))
gth = pd.read_csv(ns.src('guidance_theta.csv'))


def E(est_id, feat, set_):
    """取 confirm_estimates 中唯一一行（est_id + feat + set）。"""
    r = est[(est.est_id == est_id) & (est.feat == feat) & (est['set'] == set_)]
    assert len(r) == 1, (est_id, feat, set_, len(r))
    return r.iloc[0]


def f2(v, signed=True):
    return ns.fmt_pp(v, signed=signed, nd=2)


def ci(r, lvl=95):
    return f'[{f2(r[f"lo{lvl}"], False)}, {f2(r[f"hi{lvl}"], False)}]'


def npos(r):
    return f'{int(r.n_pos)}/{int(r.n)}'


# ------------------------------------------------------------------ 版式（全部以 mm 定位）
FW, FH = 183, 115
fig = plt.figure(figsize=(FW * MM, FH * MM))


def add_ax(x0, y0, w, h):
    return fig.add_axes([x0 / FW, y0 / FH, w / FW, h / FH])


def plabel(ax, letter, x_mm, y_mm):
    """面板标号：按 mm 位置换算成轴坐标后调用 ns.panel，保证同一行标号齐平。"""
    p = ax.get_position()
    xa = (x_mm / FW - p.x0) / p.width
    ya = (y_mm / FH - p.y0) / p.height
    ns.panel(ax, letter, x=xa, y=ya)


def side_text(ax, y, s, x_mm_off=1.2, **kw):
    """在轴右侧外（按 mm 偏移）写一列文字，y 为数据坐标。"""
    p = ax.get_position()
    xa = 1 + x_mm_off / (p.width * FW)
    kw.setdefault('fontsize', FS['small'])
    kw.setdefault('va', 'center')
    kw.setdefault('ha', 'left')
    ax.text(xa, y, s, transform=ax.get_yaxis_transform(), clip_on=False, **kw)


def ci_row(ax, y, m, lo, hi, color, marker='o', mfc=None, ms=3.6, lw=0.9, z=3):
    ax.plot([lo, hi], [y, y], color=color, lw=lw, solid_capstyle='butt', zorder=z)
    ax.plot([m], [y], marker=marker, ms=ms, mfc=color if mfc is None else mfc, mec=color, mew=0.8,
            ls='none', zorder=z + 1)


def val_above(ax, x, y, s, color=C['TEXT'], dy=2.2, fs=None):
    # 白底小框只遮住其下的零参考线，不遮数据
    ax.annotate(s, (x, y), xytext=(0, dy), textcoords='offset points', ha='center', va='bottom',
                fontsize=fs or FS['small'], color=color, zorder=5,
                bbox=dict(boxstyle='square,pad=0.05', fc='white', ec='none'))


def vline0(ax):
    ns.zero_line(ax, axis='x')


ROW1_Y0, ROW1_H = 71.5, 34.0      # 第一行轴底与高
ROW2_Y0, ROW2_H = 9.0, 36.0       # 第二行轴底与高
ROW1_LAB, ROW2_LAB = 109.5, 49.5  # 面板标号基线（mm）

# ================================================================== a｜两根杠杆放在同一尺度（主面板）
ax = add_ax(24, ROW1_Y0, 40, ROW1_H)
rows_a = [('+50 random labels', E('A1_RT50', 'both', 9)),
          ('+100 random labels', E('A1_RT100', 'both', 9)),
          ('+200 random labels', E('A1_RT200', 'both', 9)),
          ('+400 random labels', E('A1_RT400', 'both', 9))]
h2_9, h2_11 = E('H2', 'ae-s2', 9), E('H2', 'ae-s2', 11)
ya = [0, 1, 2, 3]
for y, (lab, r) in zip(ya, rows_a):
    ci_row(ax, y, r['mean'], r.lo95, r.hi95, C['LABEL'])
    val_above(ax, r['mean'], y, f2(r['mean']), color=C['LABEL'])
    side_text(ax, y, npos(r), color=C['TEXT'])
# AE 行（9 国，主）与 11 国灰色附行
y_ae, y_ae11 = 4.35, 5.35
ci_row(ax, y_ae, h2_9['mean'], h2_9.lo95, h2_9.hi95, C['AE'], ms=4.6, lw=1.2)
val_above(ax, h2_9['mean'], y_ae, f'{f2(h2_9["mean"])} {ci(h2_9)}', color=C['AE'], dy=2.8)
side_text(ax, y_ae, npos(h2_9), color=C['TEXT'])
ci_row(ax, y_ae11, h2_11['mean'], h2_11.lo95, h2_11.hi95, C['MUTED'], mfc='white', ms=3.4, lw=0.7)
val_above(ax, h2_11['mean'], y_ae11, f'{f2(h2_11["mean"])} {ci(h2_11)}', color=C['MUTED'])
side_text(ax, y_ae11, npos(h2_11), color=C['MUTED'])
vline0(ax)
ax.set_ylim(y_ae11 + 0.6, -0.75)
ax.set_xlim(-1, 11)
ax.set_xticks([0, 2, 4, 6, 8, 10])
ax.set_yticks(ya + [y_ae, y_ae11])
ax.set_yticklabels([l for l, _ in rows_a] + ['AE instead of S2\n(zero labels)', '11 countries\n(confirmatory)'])
for t, c in zip(ax.get_yticklabels(), [C['LABEL']] * 4 + [C['AE'], C['MUTED']]):
    t.set_color(c)
ax.get_yticklabels()[-1].set_fontsize(FS['small'])
ax.tick_params(axis='y', length=0, pad=2)
ax.spines['left'].set_visible(False)
ax.set_xlabel('ΔBA (%)')
side_text(ax, -0.75, '> 0', color=C['MUTED'], va='bottom')
ax.text(0.99, 1.0, '9 countries', transform=ax.transAxes, ha='right', va='bottom', fontsize=FS['small'],
        color=C['MUTED'])
plabel(ax, 'a', 1.0, ROW1_LAB)

# ================================================================== b｜逐国 AE − S2（LightGBM 与 RF）
ax = add_ax(80, ROW1_Y0, 37, ROW1_H)
d = cv[(cv.name == 'D_Z0') & (cv.feat == 'ae-s2')].pivot(index='country', columns='learner', values='value')
d = d.sort_values('lgb', ascending=False)
yb = np.arange(len(d))
for y, (cc, r) in zip(yb, d.iterrows()):
    ax.plot([r.lgb, r.rf], [y, y], color=C['GRID'], lw=0.7, zorder=1)
    ax.plot(r.lgb, y, 'o', ms=3.4, mfc=C['AE'], mec=C['AE'], mew=0.8, zorder=3)
    ax.plot(r.rf, y, 'o', ms=3.4, mfc='white', mec=C['AE'], mew=0.8, zorder=2)
vline0(ax)
ax.set_yticks(yb)
ax.set_yticklabels(d.index)
for t in ax.get_yticklabels():
    if t.get_text() in ('RO', 'PT'):
        t.set_color(C['MUTED'])
ax.set_ylim(len(d) - 0.4, -0.6)
ax.set_xlim(-5, 16)
ax.set_xticks([-4, 0, 4, 8, 12, 16])
ax.tick_params(axis='y', length=0, pad=2)
ax.spines['left'].set_visible(False)
ax.set_xlabel('ΔBA (%)')
ax.text(0.99, 1.0, 'zero labels; 11 countries', transform=ax.transAxes, ha='right', va='bottom',
        fontsize=FS['small'], color=C['MUTED'])
# 底部均值（兼作图例：实心 = LightGBM，空心 = RF）
lg11, rf11 = E('H2', 'ae-s2', 11), E('E1_RF_D_Z0', 'ae-s2', 11)
for i, (name, r, mfc) in enumerate([('LightGBM', lg11, C['AE']), ('RF', rf11, 'white')]):
    yy = -0.335 - i * 0.085
    ax.plot([0.02], [yy], 'o', ms=3.4, mfc=mfc, mec=C['AE'], mew=0.8, transform=ax.transAxes, clip_on=False)
    ax.text(0.06, yy, f'{name} {f2(r["mean"])} {ci(r)}, {npos(r)}', transform=ax.transAxes, ha='left',
            va='center', fontsize=FS['small'])
plabel(ax, 'b', 73.0, ROW1_LAB)

# ================================================================== c｜加标签后仍在（11 国）
ax = add_ax(143, ROW1_Y0, 30, ROW1_H)
rows_c = [('zero labels', E('H2', 'ae-s2', 11)),
          ('+200 random labels', E('E1_D_RT200', 'ae-s2', 11)),
          ('survey + 100 spread', E('E1_D_S0+SRS100', 'ae-s2', 11))]
yc = np.arange(len(rows_c))
for y, (lab, r) in zip(yc, rows_c):
    ci_row(ax, y, r['mean'], r.lo95, r.hi95, C['AE'], ms=3.8)
    val_above(ax, r['mean'], y, f2(r['mean']), color=C['AE'])
    side_text(ax, y, npos(r))
side_text(ax, -0.75, '> 0', color=C['MUTED'], va='bottom')
vline0(ax)
ax.set_yticks(yc)
ax.set_yticklabels([l for l, _ in rows_c])
ax.set_ylim(len(rows_c) - 0.4, -0.75)
ax.set_xlim(-1, 10)
ax.set_xticks([0, 2, 4, 6, 8, 10])
ax.tick_params(axis='y', length=0, pad=2)
ax.spines['left'].set_visible(False)
ax.set_xlabel('ΔBA (%)')
ax.text(0.99, 1.0, '11 countries', transform=ax.transAxes, ha='right', va='bottom', fontsize=FS['small'],
        color=C['MUTED'])
plabel(ax, 'c', 122.0, ROW1_LAB)

# ================================================================== d｜标签对两种特征的回报相同
ax = add_ax(11, ROW2_Y0, 43, ROW2_H)
ks = [50, 100, 200, 400]
for feat, col, dx in [('s2', C['S2'], 6), ('ae', C['AE'], -6)]:
    rr = [E(f'A1_RT{k}', feat, 9) for k in ks]
    xs = [0] + [k + dx for k in ks]
    ys = [0] + [r['mean'] for r in rr]
    ax.plot(xs, ys, color=col, lw=1.1, zorder=2)
    for k, r in zip(ks, rr):
        ax.plot([k + dx] * 2, [r.lo95, r.hi95], color=col, lw=0.8, zorder=2)
    ax.plot(xs[1:], ys[1:], 'o', ms=3.4, color=col, mec='white', mew=0.4, zorder=3)
ax.plot([0], [0], 'o', ms=3.4, color=C['REF_Z0'], mec='white', mew=0.4, zorder=4)
ax.text(135, 2.55, 'AE', color=C['AE'], fontsize=FS['base'], ha='right', va='bottom', fontweight='bold')
ax.text(165, 1.25, 'S2', color=C['S2'], fontsize=FS['base'], ha='left', va='top', fontweight='bold')
ns.zero_line(ax, axis='y')
ax.set_xlim(-15, 420)
ax.set_xticks([0, 50, 100, 200, 400])
ax.set_ylim(-0.4, 5.6)
ax.set_yticks([0, 1, 2, 3, 4, 5])
ax.set_xlabel('Random target labels (k)')
ax.set_ylabel('ΔBA (%)')
ax.text(0.02, 1.0, 'gain over the zero-label model\nwith the same features; 9 countries', transform=ax.transAxes,
        ha='left', va='top', fontsize=FS['small'], color=C['MUTED'], linespacing=1.15)
plabel(ax, 'd', 1.0, ROW2_LAB)

# ================================================================== e｜交互量接近 0（90% CI）
ax = add_ax(87, ROW2_Y0, 31, ROW2_H)
rows_e = [('label gain (+200)', E('E2_RT200', 'inter', 9), 0.0, False),
          ('11 countries', E('E2_RT200', 'inter', 11), 0.85, True),
          ('spread − extend (+100)', E('E2_B1_SRS-EXT100', 'inter', 9), 2.0, False),
          ('spread − extend (+300)', E('E2_B1_SRS-EXT300', 'inter', 9), 3.0, False),
          ('Regional allocation\n− spread (+100)', E('E2_B1_BA-SRS100', 'inter', 9), 4.0, False),
          ('Regional allocation\n− spread (+300)', E('E2_B1_BA-SRS300', 'inter', 9), 5.0, False)]
for lab, r, y, grey in rows_e:
    col = C['MUTED'] if grey else C['AE']
    ci_row(ax, y, r['mean'], r.lo90, r.hi90, col, mfc='white' if grey else None, ms=3.2 if grey else 3.6,
           lw=0.7 if grey else 0.9)
    val_above(ax, r['mean'], y, f2(r['mean']), color=col)
    side_text(ax, y, npos(r), color=col if grey else C['TEXT'])
side_text(ax, -0.75, '> 0', color=C['MUTED'], va='bottom')
vline0(ax)
ax.set_yticks([r[2] for r in rows_e])
ax.set_yticklabels([r[0] for r in rows_e])
for t, (_, _, _, grey) in zip(ax.get_yticklabels(), rows_e):
    if grey:
        t.set_color(C['MUTED'])
        t.set_fontsize(FS['small'])
ax.set_ylim(5.6, -0.75)
ax.set_xlim(-2, 1.5)
ax.set_xticks([-2, -1, 0, 1])
ax.tick_params(axis='y', length=0, pad=2)
ax.spines['left'].set_visible(False)
ax.set_xlabel('ΔBA interaction (%)')
ax.text(0.99, 1.0, '9 countries; 90% CI', transform=ax.transAxes, ha='right', va='bottom',
        fontsize=FS['small'], color=C['MUTED'])
plabel(ax, 'e', 62.0, ROW2_LAB)

# ================================================================== f｜AE 作选样指导（另一次预注册实验）
ax = add_ax(141, ROW2_Y0, 22, ROW2_H)
g = gth[gth.metric == 'ba_ge30'].set_index('learner')
rows_f = [('LightGBM', 'lgb', 'o', None), ('RF', 'rf', 'o', 'white'), ('MLP', 'mlp', 's', None),
          ('LR', 'lr', '^', None)]
# 等价带与零线只画到数据区上沿，把上方留给带的文字标注
Y_TOP = -0.55
ax.fill_betweenx([Y_TOP, len(rows_f)], -0.3, 0.3, color=C['BAND'], lw=0, zorder=0)
ax.plot([0, 0], [Y_TOP, len(rows_f)], color=C['MUTED'], lw=0.5, ls=(0, (2, 2)), zorder=1)
ax.text(0, Y_TOP - 0.1, 'equivalence margin\n±0.3%', ha='center', va='bottom', fontsize=FS['small'],
        color=C['MUTED'], linespacing=1.1)
for y, (lab, key, mk, mfc) in enumerate(rows_f):
    r = g.loc[key]
    ci_row(ax, y, r['mean'], r.lo90, r.hi90, C['AE'], marker=mk, mfc=mfc, ms=3.8 if mk == '^' else 3.6)
    p = r.tost_p
    if p < 0.05:
        s = f'equivalent\n(TOST p = {p:.3f})' if key == 'lgb' else f'equivalent\n(p = {p:.3f})'
        side_text(ax, y, s, color=C['TEXT'], linespacing=1.1)
    else:
        side_text(ax, y, 'inconclusive', color=C['MUTED'])
ax.set_yticks(range(len(rows_f)))
ax.set_yticklabels([r[0] for r in rows_f])
ax.set_ylim(len(rows_f) - 0.45, -1.35)
ax.set_xlim(-0.5, 0.5)
ax.set_xticks([-0.4, 0, 0.4])
ax.set_xticklabels(['−0.4', '0', '0.4'])
ax.tick_params(axis='y', length=0, pad=2)
ax.spines['left'].set_visible(False)
ax.set_xlabel('ΔBA (%)')
plabel(ax, 'f', 129.0, ROW2_LAB)
# 副标题首行与面板标号顶端对齐
fig.text(134.0 / FW, (ROW2_LAB + 2.9) / FH, 'separate pre-specified experiment;\n11 countries; 90% CI',
         ha='left', va='top', fontsize=FS['small'], color=C['MUTED'], linespacing=1.15)

ns.save(fig, 'fig6_representation')
print('saved')
