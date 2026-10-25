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

"""Fig. 2 为什么要在同 OA 下比较 BA（fig2_evaluation）。

核心句：加标签和换表征都会移动工作点（OA 与 BA 相互交换），固定在 τ = 0 读数会把两根杠杆的 BA 回报读小；
在同 OA 窗口内比较 BA，才能把前沿提升与沿前沿的移动分开。

a｜示例单元（BE_F1，LightGBM，AE 特征）：零标签、+200 随机标签（第 1 次抽样）与只用目标国的 τ 族 OA–BA 曲线；
b｜同一单元：零标签下 AE 与 S2 的 τ 族曲线；
c｜确认运行：两根杠杆的 τ = 0 读数与同 OA 读数（国家为单位，95% t CI）。
只读 source_data/ 下的 CSV，不重算统计量；箭头位置只是几何标注，印出的数字全部来自源文件。
"""
import os

os.environ.setdefault('OMP_NUM_THREADS', '2')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator

import nstyle as ns

ns.setup()
C, FS, MM = ns.C, ns.FS, ns.MM
fmt = ns.fmt_pp

# ------------------------------------------------------------------ 读数据
cur = pd.read_csv(ns.src('example_unit_curves.csv'))
fa = pd.read_csv(ns.src('example_unit_fa.csv'))
dif = pd.read_csv(ns.src('example_unit_diffs.csv'))
est = pd.read_csv(ns.src('confirm_estimates.csv'))

cur = cur[(cur.unit == 'BE_F1') & (cur.learner == 'lgb')]


def curve(feat, arm, r=0):
    """取一条 τ 族曲线，按 τ 排序。"""
    s = cur[(cur.feat == feat) & (cur.arm == arm) & (cur.r == r)].sort_values('tau')
    return s.tau.to_numpy(), s.OA.to_numpy(), s.BA.to_numpy()


def t0(feat, arm, r=0):
    """τ = 0 的工作点 (OA, BA)。"""
    tau, oa, ba = curve(feat, arm, r)
    i = int(np.argmin(np.abs(tau)))
    assert abs(tau[i]) < 1e-9
    return oa[i], ba[i]


def frontier_ba(feat, arm, oa_q, r=0):
    """在前沿分支（OA 最大处之后、OA 随 τ 递减的一段）上按 OA 线性插值 BA，只用于摆放箭头。"""
    tau, oa, ba = curve(feat, arm, r)
    j = int(np.argmax(oa))
    o, b = oa[j:], ba[j:]
    return float(np.interp(oa_q, o[::-1], b[::-1]))


def dval(name, feat, r=0):
    s = dif[(dif.name == name) & (dif.feat == feat) & (dif.r == r)]
    assert len(s) == 1, (name, feat, r)
    return float(s.value.iloc[0])


def e(est_id, feat, st):
    s = est[(est.est_id == est_id) & (est.feat == feat) & (est.set == st)]
    assert len(s) == 1, (est_id, feat, st)
    return s.iloc[0]


# 示例单元的锚点与差值
oa_star_ae = float(fa[(fa.feat == 'ae') & (fa.arm == 'Z0')].OA_star.iloc[0])      # 84.51
oa_star_s2 = float(fa[(fa.feat == 's2') & (fa.arm == 'Z0')].OA_star.iloc[0])      # 81.73
d_rt200 = dval('dFA_RT200', 'ae')                                                  # +3.18
d_z0 = dval('D_Z0', 'ae-s2')                                                       # +6.38
oa_z0, ba_z0 = t0('ae', 'Z0')
oa_rt, ba_rt = t0('ae', 'RT200')
oa_tg, ba_tg = t0('ae', 'TGT')
oa_s2, ba_s2 = t0('s2', 'Z0')
t0_dOA, t0_dBA = oa_rt - oa_z0, ba_rt - ba_z0                                      # +1.66、+1.11
# 与分析输出对账（t0 行）
assert abs(t0_dBA - dval('t0_RT200-Z0_BA', 'ae')) < 1e-9 and abs(t0_dOA - dval('t0_RT200-Z0_OA', 'ae')) < 1e-9

# ------------------------------------------------------------------ 版式（毫米）
W, H = 183, 62
fig = plt.figure(figsize=(W * MM, H * MM))


def ax_mm(l, b, w, h):
    return fig.add_axes([l / W, b / H, w / W, h / H])


BOT, AH = 9.0, 46.5                     # 轴底与轴高（a、b、c 共用）
TOP_LBL = BOT + AH + 3.2                # 面板标号的基线高度


def plabel(ax, lab, left_mm):
    """面板标号放在各面板区域的左上角（毫米定位，保证 a、b、c 同一水平线）。"""
    l, b, w, h = ax.get_position().bounds
    ns.panel(ax, lab, x=(left_mm / W - l) / w, y=(TOP_LBL / H - b) / h)


XL, YL = (70, 92), (35, 77)


def frame_oa_ba(ax):
    ax.set_xlim(*XL)
    ax.set_ylim(*YL)
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.set_xlabel('OA (%)', labelpad=1.5)


def window(ax, lo, hi):
    """同 OA 窗口：浅灰竖带；标签放在轴顶之上，避开曲线。"""
    ax.axvspan(lo, hi, color=C['BAND'], lw=0, zorder=0)
    ax.text((lo + hi) / 2, YL[1] + 0.6, 'matched-OA window', ha='center', va='bottom', fontsize=FS['small'],
            color=C['MUTED'], clip_on=False)


def gap_arrow(ax, x, y0, y1, text, color, tx, ty, ha):
    """两条前沿之间的双向箭头；数值（窗口内逐点差的平均）放在 (tx, ty)。"""
    ax.annotate('', xy=(x, y1), xytext=(x, y0),
                arrowprops=dict(arrowstyle='<|-|>', lw=0.6, color=color, shrinkA=0, shrinkB=0,
                                mutation_scale=4.5), zorder=5)
    ax.text(tx, ty, text, ha=ha, va='center', fontsize=FS['tick'], color=color, fontweight='bold')


def t0_label(ax, xy, xytext):
    """标出 τ = 0 工作点（短引线）。"""
    ax.annotate('τ = 0', xy=xy, xytext=xytext, fontsize=FS['small'], color=C['TEXT'], ha='left', va='bottom',
                arrowprops=dict(arrowstyle='-', lw=0.4, color=C['MUTED'], shrinkA=0.5, shrinkB=2.2))


# ================================================================== a｜标签怎样移动前沿
axa = ax_mm(10.5, BOT, 44.5, AH)
frame_oa_ba(axa)
axa.set_ylabel('BA (%)', labelpad=1.5)
w_lo, w_hi = oa_star_ae - 4, oa_star_ae
window(axa, w_lo, w_hi)

for feat, arm, col, lw, z in (('ae', 'TGT', C['REF_TGT'], 0.8, 2), ('ae', 'Z0', C['REF_Z0'], 1.0, 3),
                              ('ae', 'RT200', C['LABEL'], 1.2, 4)):
    _, oa, ba = curve(feat, arm)
    axa.plot(oa, ba, color=col, lw=lw, zorder=z, solid_capstyle='round')
for (o, b), col in (((oa_z0, ba_z0), C['REF_Z0']), ((oa_rt, ba_rt), C['LABEL']), ((oa_tg, ba_tg), C['REF_TGT'])):
    axa.plot(o, b, 'o', ms=3.2, mfc=col, mec='white', mew=0.5, zorder=6)

# 同 OA 读数：窗口中点处两条前沿之间的箭头；印出的是窗口内逐点差的平均
xm = (w_lo + w_hi) / 2
y0, y1 = frontier_ba('ae', 'Z0', xm), frontier_ba('ae', 'RT200', xm)
gap_arrow(axa, xm, y0, y1, fmt(d_rt200) + '%', C['LABEL'], tx=xm, ty=y1 + 2.0, ha='center')

# τ = 0 读数：标出工作点，并在右下角写两点的 OA 差与 BA 差
t0_label(axa, (oa_rt, ba_rt), (86.55, 56.2))
axa.text(91.7, 36.0, f'at τ = 0:\nΔOA {fmt(t0_dOA)}%\nΔBA {fmt(t0_dBA)}%', fontsize=FS['small'],
         color=C['LABEL'], ha='right', va='bottom', linespacing=1.2)

# 直接标注曲线
axa.text(71.0, 51.6, 'zero-label transfer', fontsize=FS['small'], color=C['REF_Z0'], ha='left', va='top')
axa.text(71.0, 58.3, '+200 random\ntarget labels', fontsize=FS['small'], color=C['LABEL'], ha='left',
         va='bottom', linespacing=1.1)
axa.text(91.7, 76.6, 'target-only\n(~30k labels)', fontsize=FS['small'], color=C['REF_TGT'], ha='right', va='top',
         linespacing=1.1)
plabel(axa, 'a', 0.0)

# ================================================================== b｜表征怎样移动前沿
axb = ax_mm(62.0, BOT, 39.0, AH)
frame_oa_ba(axb)
axb.tick_params(labelleft=False)
w2_lo, w2_hi = min(oa_star_ae, oa_star_s2) - 4, min(oa_star_ae, oa_star_s2)
window(axb, w2_lo, w2_hi)
for feat, col, lw, z in (('s2', C['S2'], 1.0, 3), ('ae', C['AE'], 1.2, 4)):
    _, oa, ba = curve(feat, 'Z0')
    axb.plot(oa, ba, color=col, lw=lw, zorder=z, solid_capstyle='round')
for (o, b), col in (((oa_z0, ba_z0), C['AE']), ((oa_s2, ba_s2), C['S2'])):
    axb.plot(o, b, 'o', ms=3.2, mfc=col, mec='white', mew=0.5, zorder=6)
xm2 = (w2_lo + w2_hi) / 2
y0, y1 = frontier_ba('s2', 'Z0', xm2), frontier_ba('ae', 'Z0', xm2)
gap_arrow(axb, xm2, y0, y1, fmt(d_z0) + '%', C['AE'], tx=xm2 + 0.4, ty=48.4, ha='left')
t0_label(axb, (oa_z0, ba_z0), (86.4, 55.0))
axb.text(71.0, 57.0, 'AE embeddings', fontsize=FS['small'], color=C['AE'], ha='left', va='bottom')
axb.text(71.0, 49.3, 'S2 statistics', fontsize=FS['small'], color=C['S2'], ha='left', va='top')
axb.text(91.7, 76.6, 'illustrative unit\n(Belgium)\nzero-label transfer', fontsize=FS['small'], color=C['MUTED'],
         ha='right', va='top', linespacing=1.1)
plabel(axb, 'b', 58.0)

# ================================================================== c｜两种读法对比（森林图）
axc = ax_mm(131.0, BOT, 24.0, AH)
groups = [
    ('+200 random labels (vs zero-label), 9 countries', C['LABEL'], [
        ('ΔBA at τ = 0', 'M1_RT200-Z0_BA', 'both', 9, False),
        ('ΔOA at τ = 0', 'M1_RT200-Z0_OA', 'both', 9, False),
        ('ΔBA at matched OA', 'H1', 'both', 9, True)]),
    ('AE instead of S2 (zero labels), 11 countries', C['AE'], [
        ('ΔBA at τ = 0', 'M1_Z0_AE-S2_BA', 'ae-s2', 11, False),
        ('ΔOA at τ = 0', 'M1_Z0_AE-S2_OA', 'ae-s2', 11, False),
        ('ΔBA at matched OA', 'H2', 'ae-s2', 11, True)]),
]
XLC = (-1.5, 9.5)
y = 8.6
rows, heads = [], []
for title, col, items in groups:
    heads.append((y, title, col))
    y -= 1.0
    ys = []
    for lab, eid, feat, st, main in items:
        rows.append((y, lab, e(eid, feat, st), col, main))
        ys.append(y)
        y -= 1.0
    # 0 参考线只画在本组的行范围内，避免穿过组标题
    axc.plot([0, 0], [min(ys) - 0.5, max(ys) + 0.5], color=C['MUTED'], lw=0.5, ls=(0, (2, 2)), zorder=0)
    y -= 0.55
axc.set_ylim(y + 0.55 + 0.45, 9.2)
axc.set_xlim(*XLC)
axc.xaxis.set_major_locator(MultipleLocator(2))
axc.set_yticks([])
axc.spines['left'].set_visible(False)
axc.set_xlabel('ΔBA or ΔOA (%)', labelpad=1.5)

mm_x = (XLC[1] - XLC[0]) / 24.0          # 每毫米对应的数据单位
X_LBL = XLC[0] - 1.2 * mm_x               # 行标签右端
X_HEAD = XLC[0] - 21.5 * mm_x             # 组标题左端
X_TXT = XLC[1] + 1.2 * mm_x               # 数值列左端
X_POS = XLC[1] + 22.8 * mm_x              # n_pos/n 列左端
for yy, title, col in heads:
    axc.text(X_HEAD, yy, title, ha='left', va='center', fontsize=FS['tick'], color=col, fontweight='bold',
             clip_on=False)
for yy, lab, r, col, main in rows:
    axc.plot([r['lo95'], r['hi95']], [yy, yy], color=col, lw=0.9 if main else 0.7, solid_capstyle='butt',
             zorder=3)
    axc.plot(r['mean'], yy, 'o', ms=4.2 if main else 3.2, mfc=col if main else 'white', mec=col,
             mew=0.8, zorder=4)
    axc.text(X_LBL, yy, lab, ha='right', va='center', fontsize=FS['tick'], clip_on=False,
             fontweight='bold' if main else 'normal')
    axc.text(X_TXT, yy, f"{fmt(r['mean'])} [{fmt(r['lo95'], signed=False)}, {fmt(r['hi95'], signed=False)}]",
             ha='left', va='center', fontsize=FS['small'], clip_on=False, color=C['TEXT'])
    axc.text(X_POS, yy, f"{int(r['n_pos'])}/{int(r['n'])}", ha='left', va='center',
             fontsize=FS['small'], clip_on=False, color=C['MUTED'])
axc.text(X_TXT, 9.35, 'mean [95% CI]', ha='left', va='bottom', fontsize=FS['small'], color=C['MUTED'],
         clip_on=False)
axc.text(X_POS, 9.35, '> 0', ha='left', va='bottom', fontsize=FS['small'], color=C['MUTED'], clip_on=False)
plabel(axc, 'c', 106.0)

ns.save(fig, 'fig2_evaluation')
print('saved')
