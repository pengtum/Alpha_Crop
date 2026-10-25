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

"""Fig. 4 RQ2：已有调查的价值与补采位置（双栏 183 × 64 mm，三个面板）。

核心句：已有的 100 个集中样本约值 100 个随机标签；在原区继续补采，价值递减；
在全国补采（随机或 Betti 分配）则与零起点随机设计打平，而 Betti 分配并不比随机好。

只读 source_data/confirm_estimates.csv（learner lgb、feat both、set 9；均值与 95% t CI）。
"""
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import nstyle as ns

ns.setup()
C, FS, MM = ns.C, ns.FS, ns.MM

# ------------------------------------------------------------------ 数据
est = pd.read_csv(ns.src('confirm_estimates.csv'))


def E(est_id, feat='both', set_=9):
    """取 confirm_estimates 中唯一一行（LightGBM 主分析；est_id + feat + set）。"""
    r = est[(est.est_id == est_id) & (est.feat == feat) & (est['set'] == set_) & (est.learner == 'lgb')]
    assert len(r) == 1, (est_id, feat, set_, len(r))
    return r.iloc[0]


def f2(v, signed=True):
    return ns.fmt_pp(v, signed=signed, nd=2)


def npos(r):
    return f'{int(r.n_pos)}/{int(r.n)}'


# ------------------------------------------------------------------ 版式（全部以 mm 定位）
FW, FH = 183, 64
fig = plt.figure(figsize=(FW * MM, FH * MM))
AX_Y0, AX_H = 9.5, 46.5          # 三个面板共用轴底与轴高（mm）
LAB_Y = 60.5                     # 面板标号基线（mm）


def add_ax(x0, y0, w, h):
    return fig.add_axes([x0 / FW, y0 / FH, w / FW, h / FH])


def plabel(ax, letter, x_mm, y_mm=LAB_Y):
    """面板标号：按 mm 位置换算成轴坐标后调用 ns.panel，保证三个标号齐平。"""
    p = ax.get_position()
    ns.panel(ax, letter, x=(x_mm / FW - p.x0) / p.width, y=(y_mm / FH - p.y0) / p.height)


def side_text(ax, y, s, x_mm_off=1.4, **kw):
    """在轴右侧外（按 mm 偏移）写一列文字，y 为数据坐标。"""
    p = ax.get_position()
    kw.setdefault('fontsize', FS['small'])
    kw.setdefault('va', 'center')
    kw.setdefault('ha', 'left')
    ax.text(1 + x_mm_off / (p.width * FW), y, s, transform=ax.get_yaxis_transform(), clip_on=False, **kw)


def ci_row(ax, y, m, lo, hi, color, ms=3.4, lw=0.9):
    ax.plot([lo, hi], [y, y], color=color, lw=lw, solid_capstyle='butt', zorder=3)
    ax.plot([m], [y], 'o', ms=ms, mfc=color, mec=color, mew=0.6, zorder=4)


def val_above(ax, x, y, s, color, dy=2.0):
    # 数值远离零线，保持原来的估计与区间。
    dx = (8 if x >= 0 else -8) if abs(x) < .25 else 0
    ha = ('left' if dx > 0 else 'right') if dx else 'center'
    ax.annotate(s, (x, y), xytext=(dx, dy), textcoords='offset points', ha=ha, va='bottom',
                fontsize=FS['small'], color=color, zorder=5)


def forest_frame(ax, ymax, xlabel, zero_to=None):
    """森林图公共样式：去左轴线、y 刻度不画短线、顶部写国家数与 "> 0" 列头。
    zero_to 给定时，零参考线只画到该 y（数据坐标），为其下方的带内说明留出位置。"""
    if zero_to is None:
        ns.zero_line(ax, axis='x')
    else:
        # 与 ns.zero_line 同样式，只是截短
        ax.plot([0, 0], [-0.9, zero_to], color=C['MUTED'], lw=0.5, ls=(0, (2, 2)), zorder=0)
    ax.set_ylim(ymax, -0.9)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0, pad=2)
    ax.set_xlabel(xlabel)
    side_text(ax, -0.9, '> 0', color=C['MUTED'], va='bottom')
    ax.text(1.0, 1.0, '9 countries', transform=ax.transAxes, ha='right', va='bottom',
            fontsize=FS['small'], color=C['MUTED'])


# ================================================================== a｜总标签数—增益曲线（主面板）
ax = add_ax(11.5, AX_Y0, 50, AX_H)
Z = dict(x=0, m=0.0, lo=0.0, hi=0.0)       # 零标签迁移锚点：按定义 ΔBA = 0
S0 = E('X2_B_S0')
fam = [  # (族, 颜色, x 错开量, [(总标签数, est_id 或 None)])
    ('random', C['LABEL'], -9, [(0, None), (50, 'A1_RT50'), (100, 'A1_RT100'), (200, 'A1_RT200'),
                                 (400, 'A1_RT400')]),
    ('conc', C['CONC'], -3, [(0, None), (100, 'X2_B_S0'), (200, 'X2_B_EXT100'), (400, 'X2_B_EXT300')]),
    ('spread', C['SPREAD'], 3, [(100, 'X2_B_S0'), (200, 'X2_B_SRS100'), (400, 'X2_B_SRS300')]),
    ('betti', C['BETTI'], 9, [(100, 'X2_B_S0'), (200, 'X2_B_BA100'), (400, 'X2_B_BA300')]),
]
for name, col, dx, pts in fam:
    xs, ms_ = [], []
    for k, eid in pts:
        if eid is None:                     # 原点不错开：所有族共用同一个零标签锚点
            xs.append(0.0)
            ms_.append(0.0)
            continue
        r = E(eid)
        # S0 点属于集中族（CONC）；spread/Betti 两族的线从 S0 点的实际位置出发，不重复画点
        x = k + (-3 if eid == 'X2_B_S0' else dx)
        xs.append(x)
        ms_.append(r['mean'])
        if eid == 'X2_B_S0' and name != 'conc':
            continue
        ax.plot([x, x], [r.lo95, r.hi95], color=col, lw=0.6, solid_capstyle='butt', zorder=2)
        ax.plot(x, r['mean'], 'o', ms=3.2, mfc=col, mec='white', mew=0.4, zorder=4)
    ax.plot(xs, ms_, color=col, lw=1.0, zorder=3)
ax.plot(0, 0, 'o', ms=3.2, mfc=C['REF_Z0'], mec='white', mew=0.4, zorder=5)
ns.zero_line(ax, axis='y')
ax.set_xlim(-15, 430)
ax.set_ylim(-0.75, 5.4)
ax.set_xticks([0, 100, 200, 300, 400])
ax.set_yticks([0, 1, 2, 3, 4, 5])
ax.set_xlabel('Target labels in total')
ax.set_ylabel('ΔBA (%)')
# 直接标注：原点与已有调查点
ax.annotate('zero-label transfer', (0, 0), xytext=(4, -3.2), textcoords='offset points', ha='left', va='top',
            fontsize=FS['small'], color=C['REF_Z0'])
ax.annotate('existing survey\n(100 clustered labels)', (97, S0['mean']), xytext=(118, 0.45),
            textcoords='data', ha='left', va='center', fontsize=FS['small'], color=C['CONC'],
            arrowprops=dict(arrowstyle='-', lw=0.5, color=C['CONC'], shrinkA=0.5, shrinkB=2.2))
handles = [Line2D([], [], color=c, lw=1.0, marker='o', ms=3.2, mfc=c, mec='white', mew=0.4) for c in
           (C['LABEL'], C['CONC'], C['SPREAD'], C['BETTI'])]
labels = ['random labels from zero', 'existing survey + extend surveyed area',
          'existing survey + spread nationally', 'existing survey + regional allocation']
ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.0, 1.03), fontsize=FS['small'],
          handlelength=1.8, handletextpad=0.5, labelspacing=0.3, borderaxespad=0.2)
plabel(ax, 'a', 0.8)

# ================================================================== b｜新标签补在哪里（森林图）
ax = add_ax(91.5, AX_Y0, 24, AX_H)
rows_b = [('Spread − extend, +100', 'B1_SRS-EXT100', C['SPREAD'], 0.0),
          ('Spread − extend, +300', 'B1_SRS-EXT300', C['SPREAD'], 1.0),
          ('Regional allocation\n− spread, +100', 'B1_BA-SRS100', C['BETTI'], 2.5),
          ('Regional allocation\n− spread, +300', 'B1_BA-SRS300', C['BETTI'], 3.5)]
ax.axvspan(-1, 1, color=C['BAND'], lw=0, zorder=0)
for lab, eid, col, y in rows_b:
    r = E(eid)
    ci_row(ax, y, r['mean'], r.lo95, r.hi95, col)
    val_above(ax, r['mean'], y, f2(r['mean']), col)
    side_text(ax, y, npos(r))
forest_frame(ax, 5.3, 'ΔBA (%)', zero_to=4.1)
# 带内说明：写在带的底部，零参考线在其上方截止
ax.text(0, 5.15, '±1%\nsmall-effect band', ha='center', va='bottom', fontsize=FS['small'],
        color=C['MUTED'], linespacing=1.3, zorder=5)
ax.set_xlim(-1.45, 1.45)
ax.set_xticks([-1, 0, 1])
ax.set_yticks([y for *_, y in rows_b])
ax.set_yticklabels([l for l, *_ in rows_b])
for t, (_, _, col, _) in zip(ax.get_yticklabels(), rows_b):
    t.set_color(col)
plabel(ax, 'b', 67.5)

# ================================================================== c｜同总数下与零起点随机设计比较（森林图）
CX0, CW = 149.5, 28.0            # c 轴左缘与宽度（mm）；加宽到 28 mm，使导出宽度接近 183 mm 双栏
ax = add_ax(CX0, AX_Y0, CW, AX_H)
groups_c = [('vs 100 random labels', [('Survey (100)', 'X1_S0-RT100', C['CONC'])]),
            ('vs 200 random labels', [('Survey + spread 100', 'X1_SRS100-RT200', C['SPREAD']),
                                      ('Survey + extend 100', 'X1_EXT100-RT200', C['CONC']),
                                      ('Survey + regional\nallocation 100', 'X1_BA100-RT200', C['BETTI'])]),
            ('vs 400 random labels', [('Survey + spread 300', 'X1_SRS300-RT400', C['SPREAD']),
                                      ('Survey + extend 300', 'X1_EXT300-RT400', C['CONC']),
                                      ('Survey + regional\nallocation 300', 'X1_BA300-RT400', C['BETTI'])])]
yt, ylab, ycol = [], [], []
y = 0.0
p = ax.get_position()
for head, rows in groups_c:
    # 组标题：左端与行标签列左缘对齐，写在该组第一行上方
    ax.text(-(CX0 - 125.5) / CW, y - 0.05, head, transform=ax.get_yaxis_transform(), ha='left',
            va='bottom', fontsize=FS['small'], color=C['MUTED'], style='italic', clip_on=False)
    y += 0.8
    for lab, eid, col in rows:
        r = E(eid)
        ci_row(ax, y, r['mean'], r.lo95, r.hi95, col)
        val_above(ax, r['mean'], y, f2(r['mean']), col)
        side_text(ax, y, npos(r))
        yt.append(y)
        ylab.append(lab)
        ycol.append(col)
        y += 1.0
    y += 0.1
forest_frame(ax, y - 0.4, 'ΔBA (%)')
ax.set_xlim(-1.3, 0.6)
ax.set_xticks([-1, -0.5, 0, 0.5])
ax.set_xticklabels(['−1', '−0.5', '0', '0.5'])
ax.set_yticks(yt)
ax.set_yticklabels(ylab)
for t, col in zip(ax.get_yticklabels(), ycol):
    t.set_color(col)
plabel(ax, 'c', 124.5)

ns.save(fig, 'fig4_existing_survey')
print('saved fig4_existing_survey')
