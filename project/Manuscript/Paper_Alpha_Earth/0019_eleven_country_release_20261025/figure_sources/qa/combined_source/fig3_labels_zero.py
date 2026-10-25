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

"""Fig. 3 RQ1：从零起步，标签能买到什么（fig3_labels_zero）。

核心句：多源迁移之后，50–400 个随机目标标签让同 OA 下的 BA 提高 0.8–3.7 pp，9 个主集国家全部为正；
这是真实的增益，但最多只补上本国全量训练缺口的四分之一。RF 给出同样的答案。

a｜剂量曲线：逐国细线（9 国主集 + RO/PT 灰色虚线）与 9 国均值（95% t CI）；
b｜随机标签补上"只用目标国训练"缺口的比例；
c｜换学习器检验：+200 标签时 LightGBM 与 RF 的逐国值。
只读 source_data/ 下的 CSV，不重算统计量。
"""
import os

os.environ.setdefault('OMP_NUM_THREADS', '2')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, MultipleLocator

import nstyle as ns

ns.setup()
C, FS, MM = ns.C, ns.FS, ns.MM
fmt = ns.fmt_pp

# ------------------------------------------------------------------ 读数据
est = pd.read_csv(ns.src('confirm_estimates.csv'))
cv = pd.read_csv(ns.src('confirm_country_values.csv'))
a2 = pd.read_csv(ns.src('confirm_a2_ratio.csv'))
cc = pd.read_csv(ns.src('country_counts.csv'))

PRIMARY = cc[cc.primary9].country.tolist()          # 9 国主集
EXTRA = cc[~cc.primary9].country.tolist()           # RO、PT
KS = [50, 100, 200, 400]


def e(est_id, feat, st, learner=None):
    s = est[(est.est_id == est_id) & (est.feat == feat) & (est.set == st)]
    if learner is not None:
        s = s[s.learner == learner]
    assert len(s) == 1, (est_id, feat, st)
    return s.iloc[0]


def ci_txt(r, unit=''):
    """"+2.34 pp [1.54, 3.14]" 形式（负号为 U+2212）。"""
    return f"{fmt(r['mean'])}{unit} [{fmt(r['lo95'], signed=False)}, {fmt(r['hi95'], signed=False)}]"


# 逐国值：lgb、both，dFA_RT*；k = 0 处为 0
v = cv[(cv.learner == 'lgb') & (cv.feat == 'both') & (cv.name.isin([f'dFA_RT{k}' for k in KS]))]
dose = v.pivot_table(index='country', columns='name', values='value')[[f'dFA_RT{k}' for k in KS]]
assert set(dose.index) == set(PRIMARY + EXTRA) and dose.notna().all().all()
m9 = {k: e(f'A1_RT{k}', 'both', 9) for k in KS}
m11_400 = e('A1_RT400', 'both', 11)
h1 = e('H1', 'both', 9)
assert abs(h1['mean'] - m9[200]['mean']) < 1e-12            # H1 与 A1_RT200 同一量
tgt = e('A2_TGT', 'both', 9)
rf200 = e('A3_RF_RT200', 'both', 9)
ratio = a2[a2.set == 9].set_index('k')
assert all(abs(ratio.loc[k, 'num'] - m9[k]['mean']) < 1e-9 for k in KS)
assert abs(ratio.loc[50, 'den'] - tgt['mean']) < 1e-9

# LightGBM 与 RF 的逐国值（dFA_RT200、both，11 国）
sc = cv[(cv.feat == 'both') & (cv.name == 'dFA_RT200')].pivot_table(index='country', columns='learner',
                                                                    values='value')
assert len(sc) == 11

# ------------------------------------------------------------------ 版式（毫米）
W, H = 183, 62
fig = plt.figure(figsize=(W * MM, H * MM))


def ax_mm(l, b, w, h):
    return fig.add_axes([l / W, b / H, w / W, h / H])


BOT, AH = 9.0, 46.5
TOP_LBL = BOT + AH + 3.2


def plabel(ax, lab, left_mm):
    l, b, w, h = ax.get_position().bounds
    ns.panel(ax, lab, x=(left_mm / W - l) / w, y=(TOP_LBL / H - b) / h)


GREY = C['MUTED']
DASH = (0, (3, 1.6))

# ================================================================== a｜剂量曲线（主面板）
axa = ax_mm(10.5, BOT, 53.0, AH)
XA = (-12, 432)
axa.set_xlim(*XA)
axa.set_ylim(-7.2, 7.2)
axa.xaxis.set_major_locator(FixedLocator([0] + KS))
axa.yaxis.set_major_locator(MultipleLocator(2))
axa.set_xlabel('Random target labels (k)', labelpad=1.5)
axa.set_ylabel('ΔBA (%)', labelpad=1.5)
ns.zero_line(axa, 'y')
xk = np.array([0] + KS)
for c in PRIMARY:
    axa.plot(xk, np.r_[0, dose.loc[c].to_numpy()], color=C['LABEL'], alpha=0.25, lw=0.6, zorder=2)
for c in EXTRA:
    yv = np.r_[0, dose.loc[c].to_numpy()]
    axa.plot(xk, yv, color=GREY, lw=0.6, ls=DASH, zorder=2)
    axa.text(xk[-1] + 9, yv[-1], c, ha='left', va='center', fontsize=FS['small'], color=GREY)

# 9 国均值 + 95% t CI
mu = np.r_[0, [m9[k]['mean'] for k in KS]]
lo = np.r_[0, [m9[k]['lo95'] for k in KS]]
hi = np.r_[0, [m9[k]['hi95'] for k in KS]]
axa.plot(xk, mu, color=C['LABEL'], lw=1.4, zorder=4, solid_capstyle='round')
axa.vlines(xk[1:], lo[1:], hi[1:], color=C['LABEL'], lw=0.9, zorder=4)
axa.plot(xk, mu, 'o', ms=3.4, mfc=C['LABEL'], mec='white', mew=0.5, zorder=5)

# 左下角的图例键：粗线为 9 国均值，细线为单个国家（RO、PT 已在线末端直接标注）
axa.legend(handles=[Line2D([], [], color=C['LABEL'], lw=1.4, marker='o', ms=3.0, mec='white', mew=0.4),
                    Line2D([], [], color=C['LABEL'], lw=0.6, alpha=0.35)],
           labels=['9-country mean', 'single countries'], loc='lower left',
           bbox_to_anchor=(0.0, 0.0), fontsize=FS['small'], handlelength=1.8, handletextpad=0.5,
           borderaxespad=0.2, labelspacing=0.3)

# k = 200 的确认性检验标注（文字向左错开、引线斜向指到 CI 上端，避免引线与 CI 竖线共线被误读成更长的 CI）
axa.annotate(f"{ci_txt(h1, '%')}, {int(h1['n_pos'])}/{int(h1['n'])}\n(confirmatory, Holm)",
             xy=(197, hi[3] + 0.18), xytext=(118, 5.55), ha='center', va='bottom', fontsize=FS['small'],
             color=C['LABEL'], linespacing=1.15,
             arrowprops=dict(arrowstyle='-', lw=0.4, color=C['MUTED'], shrinkA=1.0, shrinkB=0,
                             relpos=(0.78, 0.0)))

# k = 400 的 11 国均值（灰色，略向右错开）
x11 = 413
axa.vlines(x11, m11_400['lo95'], m11_400['hi95'], color=GREY, lw=0.6, zorder=3)
axa.plot(x11, m11_400['mean'], 'D', ms=2.6, mfc='white', mec=GREY, mew=0.7, zorder=5)
axa.text(x11 + 6, m11_400['lo95'] - 0.35, f"11 countries:\n{ci_txt(m11_400)}", ha='right', va='top',
         fontsize=FS['small'], color=GREY, linespacing=1.15)
plabel(axa, 'a', 0.0)

# ================================================================== b｜补上缺口的比例
axb = ax_mm(90.0, BOT, 32.0, AH)
XB = (0, 20)
axb.set_xlim(*XB)
axb.xaxis.set_major_locator(MultipleLocator(5))
axb.set_xlabel('ΔBA (%)', labelpad=1.5)
Y_TGT, Y_K = 5.1, {50: 3.4, 100: 2.4, 200: 1.4, 400: 0.4}
BH = 0.6
axb.set_ylim(-0.3, 6.0)
# 只用目标国训练：浅填色条 + 95% CI
axb.barh(Y_TGT, tgt['mean'], height=BH, color='#D4D4D4', lw=0, zorder=2)
axb.hlines(Y_TGT, tgt['lo95'], tgt['hi95'], color=C['REF_TGT'], lw=0.7, zorder=3)
axb.text(0.45, Y_TGT, ci_txt(tgt), ha='left', va='center', fontsize=FS['small'], color=C['REF_TGT'], zorder=4)
# 随机标签
X_PCT = 6.2
for k in KS:
    r = m9[k]
    axb.barh(Y_K[k], r['mean'], height=BH, color=C['LABEL'], lw=0, zorder=2)
    axb.hlines(Y_K[k], r['lo95'], r['hi95'], color=C['REF_TGT'], lw=0.7, zorder=3)
    axb.text(X_PCT, Y_K[k], f"{ratio.loc[k, 'ratio'] * 100:.0f}%", ha='left', va='center', fontsize=FS['tick'],
             color=C['LABEL'], fontweight='bold')
axb.text(X_PCT, Y_K[50] + 0.62, 'share of target-only gain', ha='left', va='bottom', fontsize=FS['small'],
         color=GREY)
axb.set_yticks([Y_TGT] + [Y_K[k] for k in KS])
axb.set_yticklabels(['target-only model\n(~30k labels)'] + [f'+{k} random labels' for k in KS],
                    linespacing=1.1)
axb.tick_params(axis='y', length=0, pad=2)
axb.spines['left'].set_visible(False)
plabel(axb, 'b', 68.0)

# ================================================================== c｜换学习器检验
CW = 39.5                                           # c 轴宽（毫米）；右移使整图铺满 183 mm
axc = ax_mm(141.5, BOT, CW, AH)
XC = (-6.5, 5.0)
span = (XC[1] - XC[0]) * AH / CW                     # 等比例：y 跨度按轴的高宽比换算
YC = (7.4 - span, 7.4)                               # 上端固定在 7.4，给角落文字留出与 SE 标签之间的间隙
axc.set_xlim(*XC)
axc.set_ylim(*YC)
axc.set_aspect('equal', adjustable='box')
axc.xaxis.set_major_locator(MultipleLocator(2))
axc.yaxis.set_major_locator(MultipleLocator(2))
axc.plot([XC[0], XC[1]], [XC[0], XC[1]], color=GREY, lw=0.6, ls=(0, (3, 2)), zorder=1)
axc.text(-3.25, -4.3, 'y = x', ha='center', va='center', fontsize=FS['small'], color=GREY, rotation=45,
         rotation_mode='anchor')
ns.zero_line(axc, 'y')
# 竖向 0 参考线只画到角落文字下方
axc.plot([0, 0], [YC[0], 4.2], color=GREY, lw=0.5, ls=(0, (2, 2)), zorder=0)
# 轴标签沿用全文术语 "ΔBA at matched OA"
axc.set_xlabel('LightGBM ΔBA (%)', labelpad=1.5)
axc.set_ylabel('Random forest ΔBA (%)', labelpad=1.5)
# 国家代码的摆放（dx, dy 以数据单位计；ha, va）
OFF = dict(AT=(-0.22, 0.0, 'right', 'center'), SI=(-0.22, 0.0, 'right', 'center'),
           SE=(0.0, 0.3, 'center', 'bottom'), SK=(0.25, 0.0, 'left', 'center'),
           BE=(0.24, 0.0, 'left', 'center'), LT=(-0.05, -0.3, 'center', 'top'),
           DE=(0.24, 0.0, 'left', 'center'), EE=(0.05, -0.3, 'center', 'top'),
           LV=(-0.12, 0.28, 'right', 'bottom'), PT=(0.4, 0.0, 'left', 'center'),
           RO=(0.4, 0.0, 'left', 'center'))          # 空心点半径更大，代码再右移一点，避免贴住标记
for c, row in sc.iterrows():
    main = c in PRIMARY
    axc.plot(row['lgb'], row['rf'], 'o', ms=3.4, mfc=C['LABEL'] if main else 'white',
             mec='white' if main else GREY, mew=0.5 if main else 0.8, zorder=4)
    dx, dy, ha, va = OFF[c]
    axc.text(row['lgb'] + dx, row['rf'] + dy, c, ha=ha, va=va, fontsize=FS['small'],
             color=C['TEXT'] if main else GREY, zorder=5)
# 角落：两种学习器的 9 国均值
yt = YC[1] - 0.3
# 小标题写明是 9 国均值（图中画的是 11 国的点，避免读成"本面板只有 9 国"）
axc.text(XC[0] + 0.3, yt, '+200 random labels; 9-country mean', ha='left', va='top', fontsize=FS['small'],
         color=GREY)
for i, (name, r) in enumerate((('LightGBM', h1), ('RF', rf200))):
    yy = yt - 1.05 * (i + 1)
    axc.text(XC[0] + 0.3, yy, name, ha='left', va='top', fontsize=FS['small'], color=C['LABEL'],
             fontweight='bold')
    axc.text(XC[0] + 3.35, yy, f"{ci_txt(r)}, {int(r['n_pos'])}/{int(r['n'])}", ha='left', va='top',
             fontsize=FS['small'], color=C['LABEL'])
plabel(axc, 'c', 129.0)

ns.save(fig, 'fig3_labels_zero')
print('saved')
