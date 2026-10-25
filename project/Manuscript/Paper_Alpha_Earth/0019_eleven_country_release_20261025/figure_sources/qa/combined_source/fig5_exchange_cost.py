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

"""Fig. 5 RQ2：交换率与成本规则（双栏 183 × 62 mm，三个面板；c 为三个并排小图）。

核心句：集中样本在总数 200 时约值 0.83 个随机标签，在总数 400 时约值 0.72 个。
只有当全国分散采一个样本的成本超过集中采样的约 1.2–1.4 倍时，扩大原调查区才更划算。

只读 source_data/confirm_x3_equiv.csv（k_eq、交换率、集中 − 随机，90% 国家层 bootstrap 区间）
与 source_data/confirm_x3_cost.csv（kind == 'budget'，成本效益点估计）。
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import nstyle as ns

ns.setup()
C, FS, MM = ns.C, ns.FS, ns.MM

# ------------------------------------------------------------------ 数据
eq = pd.read_csv(ns.src('confirm_x3_equiv.csv'))
cost = pd.read_csv(ns.src('confirm_x3_cost.csv'))


def Q(kind, item):
    """取 confirm_x3_equiv 中唯一一行（kind + item）。"""
    r = eq[(eq.kind == kind) & (eq['item'].astype(str) == str(item))]
    assert len(r) == 1, (kind, item, len(r))
    return r.iloc[0]


def f2(v, signed=False):
    return ns.fmt_pp(v, signed=signed, nd=2)


# ------------------------------------------------------------------ 版式（全部以 mm 定位）
FW, FH = 183, 62
fig = plt.figure(figsize=(FW * MM, FH * MM))
AX_Y0, AX_H = 9.5, 39.0          # 三个面板共用轴底与轴高（mm）
LAB_Y = 51.5                     # 面板标号基线（mm）


def add_ax(x0, y0, w, h, **kw):
    return fig.add_axes([x0 / FW, y0 / FH, w / FW, h / FH], **kw)


def plabel(ax, letter, x_mm, y_mm=LAB_Y):
    """面板标号：按 mm 位置换算成轴坐标后调用 ns.panel，保证三个标号齐平。"""
    p = ax.get_position()
    ns.panel(ax, letter, x=(x_mm / FW - p.x0) / p.width, y=(y_mm / FH - p.y0) / p.height)


FAM_COL = dict(S0=C['CONC'], EXT=C['CONC'], SRS=C['SPREAD'], BA=C['BETTI'])

# ------------------------------------------------------------------ 全图共用图例条（放在最上方）
handles = [Line2D([], [], color=C['LABEL'], lw=1.0, marker='o', ms=3.0, mfc=C['LABEL'], mec='white', mew=0.4),
           Line2D([], [], color=C['CONC'], lw=1.0, marker='o', ms=3.0, mfc=C['CONC'], mec='white', mew=0.4),
           Line2D([], [], color=C['SPREAD'], lw=1.0, marker='o', ms=3.0, mfc=C['SPREAD'], mec='white', mew=0.4),
           Line2D([], [], color=C['BETTI'], lw=1.0, marker='o', ms=3.0, mfc=C['BETTI'], mec='white', mew=0.4),
           Line2D([], [], ls='none', marker='*', ms=5.5, mfc='white', mec='black', mew=0.5)]
labels = ['random labels', 'concentrated: existing survey, + extend surveyed area',
          'existing survey + spread nationally', 'existing survey + regional allocation', 'best option at that budget']
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=5, fontsize=FS['small'],
           handlelength=1.8, handletextpad=0.4, columnspacing=1.2, borderaxespad=0.1)

# ================================================================== a｜各配置每个标签值多少个随机标签（交换率 = k_eq ÷ n）
# 横轴是按总标签数分组的类别轴：组内左右错开只为排版，不表示数值（避免在定量轴上错开造成误读）
ax = add_ax(12.5, AX_Y0, 37.5, AX_H)
CAP = 1.25                       # 无上界（inf）时误差棒画到 CAP 并加向上箭头
cfg = [('B_S0', 0.0, 'S0'),
       ('B_EXT100', 1.0 - 0.22, 'EXT'), ('B_SRS100', 1.0, 'SRS'), ('B_BA100', 1.0 + 0.22, 'BA'),
       ('B_EXT300', 2.0 - 0.22, 'EXT'), ('B_SRS300', 2.0, 'SRS'), ('B_BA300', 2.0 + 0.22, 'BA')]
ax.axhline(1.0, color=C['LABEL'], lw=0.7, ls=(0, (3, 2)), zorder=1)
ax.text(0.22, 1.0 + 0.012, 'one random label', ha='left', va='bottom', fontsize=FS['small'], color=C['LABEL'])
for item, x, fam_ in cfg:
    r = Q('rate', item)
    col = FAM_COL[fam_]
    lo, hi, pt = r.lo90, r.hi90, r.point
    ax.plot([x, x], [lo, min(hi, CAP)], color=col, lw=0.9, solid_capstyle='butt', zorder=2)
    if np.isinf(hi):
        ax.annotate('', xy=(x, CAP + 0.06), xytext=(x, CAP), zorder=2, annotation_clip=False,
                    arrowprops=dict(arrowstyle='->,head_length=0.25,head_width=0.15', lw=0.8, color=col,
                                    shrinkA=0, shrinkB=0))
    if np.isinf(pt):
        # 点估计本身超出曲线范围（k_eq > 400）：在 CAP 处画向上三角并标 "> 1.00"
        ax.plot(x, CAP, '^', ms=3.8, mfc=col, mec='white', mew=0.4, zorder=4, clip_on=False)
        ax.text(x + 0.07, CAP - 0.035, '> 1.00', ha='left', va='center', fontsize=FS['small'], color=col)
    else:
        ax.plot(x, pt, 'o', ms=3.6, mfc=col, mec='white', mew=0.4, zorder=4)
        if fam_ in ('EXT', 'S0'):
            ax.text(x - 0.09, pt, f2(pt), ha='right', va='center', fontsize=FS['small'], color=col)
ax.set_xlim(-0.45, 2.5)
ax.set_ylim(0.6, 1.33)
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(['100', '200', '400'])   # 100 = 只有已有调查（S0）
ax.set_yticks([0.6, 0.8, 1.0, 1.2])
ax.set_xlabel('Target labels in total')
ax.set_ylabel('Worth of one label (random labels)')
plabel(ax, 'a', 0.8)

# ================================================================== b｜集中样本的交换率与盈亏平衡成本比
ax = add_ax(69.5, AX_Y0, 45, AX_H)
xs = [100, 200, 400]
d = [Q('conc_diff', k) for k in xs]
ns.zero_line(ax, axis='y')
ax.plot(xs, [r.point for r in d], color=C['CONC'], lw=0.8, zorder=2)
for k, r in zip(xs, d):
    ax.plot([k, k], [r.lo90, r.hi90], color=C['CONC'], lw=0.8, solid_capstyle='butt', zorder=3)
    ax.plot(k, r.point, 'o', ms=3.4, mfc=C['CONC'], mec='white', mew=0.4, zorder=4)
ax.text(445, 0.02, 'random labels, same total', ha='right', va='bottom', fontsize=FS['small'],
        color=C['MUTED'])
# 点旁标注：交换率（conc_rate）及其派生量 break-even = 1 / rate
rt = {k: Q('conc_rate', k) for k in (200, 400)}
be = {k: 1.0 / rt[k].point for k in (200, 400)}
ax.text(112, d[0].point + 0.23, 'not distinguishable\n(rate not reported)', ha='left', va='bottom',
        fontsize=FS['small'], color=C['TEXT'], linespacing=1.05)
ax.text(213, d[1].point + 0.035,
        f'rate {f2(rt[200].point)} [{f2(rt[200].lo90)}, {f2(rt[200].hi90)}]\n'
        f'→ break-even cost ratio {f2(be[200])}',
        ha='left', va='bottom', fontsize=FS['small'], color=C['TEXT'], linespacing=1.05)
ax.text(385, d[2].point - 0.07,
        f'rate {f2(rt[400].point)} [{f2(rt[400].lo90)}, {f2(rt[400].hi90)}]\n→ {f2(be[400])}',
        ha='right', va='top', multialignment='left', fontsize=FS['small'], color=C['TEXT'], linespacing=1.05)
ax.set_xlim(60, 450)
ax.set_ylim(-1.15, 0.48)
ax.set_xticks(xs)
ax.set_yticks([-1, -0.5, 0])
ax.set_yticklabels(['−1', '−0.5', '0'])
ax.set_xlabel('Target labels in total')
ax.set_ylabel('ΔBA (%)')
plabel(ax, 'b', 55.5)

# ================================================================== c｜成本效益（c = 1、2、3 三个小图，共用 y 轴）
bud = cost[cost.kind == 'budget'].copy()
LINES = [('random', C['LABEL']), ('S0+SRS', C['SPREAD']), ('S0+BA', C['BETTI']), ('concentrated', C['CONC'])]
X0, W, GAP = 131.5, 15.5, 2.2
axes_c = []
for j, c in enumerate((1, 2, 3)):
    ax = add_ax(X0 + j * (W + GAP), AX_Y0, W, AX_H, sharey=axes_c[0] if axes_c else None)
    axes_c.append(ax)
    sub = bud[bud.c == c]
    for name, col in LINES:        # 集中族最后画：并列最优（均为已有调查本身）时星号显示集中色
        s = sub[sub.name == name].sort_values('cost')
        ax.plot(s.cost, s.value, color=col, lw=0.9, marker='o', ms=2.4, mfc=col, mec='white', mew=0.3,
                zorder=3)
        b = s[s.best.astype(str) == 'True']
        ax.plot(b.cost, b.value, ls='none', marker='*', ms=6.0, mfc=col, mec='black', mew=0.45, zorder=5)
    ax.set_xscale('log', base=2)
    ax.set_xlim(80, 500)
    ax.set_xticks([100, 200, 400])
    ax.set_xticklabels(['100', '200', '400'])
    ax.minorticks_off()
    ax.set_title(f'c = {c}', fontsize=FS['label'], pad=2.5)
    if j > 0:
        ax.tick_params(axis='y', labelleft=False)
axes_c[0].set_ylim(0, 4.2)
axes_c[0].set_yticks([0, 1, 2, 3, 4])
axes_c[0].set_ylabel('ΔBA (%)')
axes_c[1].set_xlabel('Budget (in costs of one concentrated label)')
fig.text((X0 + 1.5 * W + GAP) / FW + 0.0, (AX_Y0 + AX_H + 4.3) / FH,
         '(c = cost of a spread label ÷ cost of a concentrated label)', ha='center', va='bottom',
         fontsize=FS['small'], color=C['MUTED'])
plabel(axes_c[0], 'c', 119.5)

ns.save(fig, 'fig5_exchange_cost')
print('saved fig5_exchange_cost')
