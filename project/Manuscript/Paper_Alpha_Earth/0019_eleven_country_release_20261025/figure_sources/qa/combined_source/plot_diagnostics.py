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

"""沿用共享样式绘制四组描述性图件；只读整理后的源数据。"""
import os
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[k] = '2'
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import nstyle as ns
ns.setup()
assert ns.FONT == 'Arial', '必须使用与现有图一致的 Arial'
SRC, OUT = Path('/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops/label_expansion/storyline/rse_figs/revision_20261007/source_data'), HERE / 'out'
OUT.mkdir(exist_ok=True)
C = ns.C
CC = ['AT','BE','DE','EE','LT','LV','PT','RO','SE','SI','SK']
CL = ['grassland','maize','orchards','other_crops','potatoes','root_crops','strawberries','tree_wood_forest','vegetables','winter_cereals']
LABELS = ['Grassland','Maize','Orchards','Other crops','Potatoes','Root crops','Strawberries','Tree/wood/forest','Vegetables','Winter cereals']
BLUE = LinearSegmentedColormap.from_list('rse_blue', ['#F4F7FA', '#A7C4DD', C['LABEL']])
DIVERGE = LinearSegmentedColormap.from_list('rse_difference', ['#A66D43','#F7F7F7', C['AE']])
QA = []

def read(n):
    return pd.read_csv(SRC / n)

def canvas(w,h):
    f=plt.figure(figsize=(w*ns.MM,h*ns.MM))
    def ax(x,y,ww,hh): return f.add_axes([x/w,y/h,ww/w,hh/h])
    return f,ax

def panel(f,letter,x,y,w,h):
    f.text(x/w,y/h,letter,fontsize=8,fontweight='bold',va='bottom')

def save(fig,name,legend):
    from matplotlib.text import Text
    fig.canvas.draw()
    # 自动检查所有非空文字的字号与有限坐标；导出后再目视检查版式。
    sizes=[t.get_fontsize() for t in fig.findobj(Text) if t.get_text().strip() and t.get_visible()]
    assert min(sizes)>=5.5,(name,min(sizes))
    QA.append(dict(figure=name,font=ns.FONT,min_font_pt=min(sizes)))
    ns.save(fig,name,out_dir=str(OUT))
    (OUT/f'{name}_legend.txt').write_text(legend+'\n')

def heat(ax,values,rows,cols,cmap=BLUE,vmin=0,vmax=100,fmt='.0f',norm=None):
    arr=np.asarray(values,dtype=float)
    cmap=cmap.copy();cmap.set_bad('#E3E3E3')
    im=ax.imshow(np.ma.masked_invalid(arr),cmap=cmap,vmin=vmin if norm is None else None,
                 vmax=vmax if norm is None else None,norm=norm,aspect='auto',interpolation='nearest')
    ax.set_xticks(range(len(cols)),cols,rotation=55,ha='right',rotation_mode='anchor')
    ax.set_yticks(range(len(rows)),rows)
    ax.tick_params(length=0,pad=2)
    for sp in ax.spines.values():sp.set_visible(False)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            val=arr[i,j]
            if not np.isfinite(val): txt='—'; color=C['MUTED']
            else:
                if fmt=='share':
                    txt='0' if val==0 else ('<0.1' if val<.1 else f'{val:.1f}')
                else:
                    txt=format(val,fmt).replace('-','−')
                    if fmt=='+.0f' and round(val)==0:txt='0'
                rgba=im.cmap(im.norm(val));lum=.2126*rgba[0]+.7152*rgba[1]+.0722*rgba[2]
                color='white' if lum<.52 else C['TEXT']
            ax.text(j,i,txt,ha='center',va='center',fontsize=5.5,color=color)
    return im

def dataset():
    f,ax=canvas(183,163)
    g=gpd.read_file(HERE.parent/'source_data/countries.geojson').to_crs(3035)
    density=read('spatial_density.csv')
    a=ax(1,64,66,88)
    g.plot(ax=a,facecolor='#EEEEEE',edgecolor='white',linewidth=.3)
    # 方格表示全部地块计数，不作抽样，不把全国边界当作标签覆盖。
    im=a.scatter(density.x,density.y,c=density.n,s=1.0,marker='s',cmap=BLUE,
                 norm=LogNorm(1,density.n.max()),linewidths=0,rasterized=True)
    a.set_xlim(density.x.min()-140000,density.x.max()+140000)
    a.set_ylim(density.y.min()-140000,density.y.max()+140000)
    a.set_aspect('equal');a.axis('off')
    # 国家代码靠近实际标签覆盖，不以全国质心替代研究范围。
    offsets={'BE':(-140000,-30000),'DE':(100000,70000),'AT':(-60000,60000),'SI':(-140000,-60000),
             'EE':(190000,50000),'LV':(190000,0),'LT':(190000,-30000),'PT':(-170000,0)}
    for r in read('country_label_positions.csv').itertuples():
        code=r.country;dx,dy=offsets.get(code,(0,0));xx,yy=r.x+dx,r.y+dy
        a.text(xx,yy,code,fontsize=6,ha='center',color=C['TEXT'],
               bbox=dict(facecolor='white',edgecolor='none',alpha=.75,pad=.2))
    cb=f.colorbar(im,cax=ax(7,59,51,1.6),orientation='horizontal')
    cb.set_ticks([1,10,100,1000],labels=['1','10','100','1,000'])
    cb.set_label('Parcels per 10-km cell',labelpad=1)
    panel(f,'a',1,155,183,163)
    shares=read('class_shares.csv').set_index('country').loc[CC,CL]
    b=ax(91,86,87,65)
    # 百分比用连续蓝色，缺席类单独显示，避免把零与极少量混为一谈。
    z=shares.to_numpy(); masked=z.copy();masked[z==0]=np.nan
    im=heat(b,masked,CC,LABELS,vmin=0,vmax=80,fmt='share')
    b.set_ylabel('Country',labelpad=2)
    cb=f.colorbar(im,cax=ax(94,60,70,1.6),orientation='horizontal')
    cb.set_label('Share of parcels within each country (%)',labelpad=1)
    f.text(91/183,52/163,'Grey cells: no retained labels for that class',fontsize=5.5)
    panel(f,'b',77,155,183,163)
    count=read('class_counts.csv').set_index('country')[CL].sum()
    c=ax(17,16,158,25)
    c.bar(np.arange(10),count,color=C['LABEL'],width=.65)
    c.set_yscale('log');c.set_ylim(100,1e6)
    c.set_yticks([1e2,1e3,1e4,1e5,1e6],labels=['100','1,000','10,000','100,000','1,000,000'])
    c.set_xticks(range(10),LABELS,rotation=25,ha='right',rotation_mode='anchor')
    c.set_ylabel('Retained parcels\n(log scale)',labelpad=2)
    for j,n in enumerate(count):c.text(j,n*1.3,f'{n:,}',fontsize=5.5,ha='center')
    panel(f,'c',1,43,183,163)
    save(f,'fig_dataset_imbalance',
        'Spatial and label support in the paired-feature dataset. (a) All retained parcel centroids aggregated '
        'into 10-km cells in EPSG:3035; logarithmic colour scale shows parcel counts, not agricultural area or sampling density per unit agricultural land. '
        '(b) Within-country class shares; grey cells indicate absent labels in this dataset, not proven absence of the crop. '
        '(c) Pooled parcel counts, logarithmic y-axis. Source tables: spatial_density, class_shares and class_counts; '
        'all rows. The dataset contains 611,519 parcels. RO labels are coarse land-cover labels; DE covers two states. '
        'Descriptive analysis; no confidence intervals or hypothesis tests.')

def baseline():
    d=read('zero_label_countries.csv')
    f,ax=canvas(183,91)
    for j,metric in enumerate(['OA','BA']):
        a=ax(12+62*j,19,46,63)
        p=d.pivot(index='country',columns='feat',values=metric).loc[CC]
        for i,cc in enumerate(CC):
            a.plot([p.loc[cc,'s2'],p.loc[cc,'ae']],[i,i],color='#B9B9B9',lw=.8,zorder=1)
            for feat in ['ae','s2']:
                a.plot(p.loc[cc,feat],i,'o',ms=3.4,color=C[feat.upper()],
                       mfc='white' if cc in ['RO','PT'] else C[feat.upper()],mew=.8)
        a.set_yticks(range(11),CC);a.set_ylim(10.6,-.6);a.set_xlim(0,100)
        a.set_xlabel(f'{metric} at default decision rule (%)')
        panel(f,chr(97+j),1+62*j,85,183,91)
    a=ax(138,19,41,63)
    e=pd.read_csv(Path('/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops/label_expansion/storyline/rse_figs/source_data/confirm_country_values.csv'))
    e=e[(e.learner=='lgb')&(e.feat=='ae-s2')&(e.name=='D_Z0')].set_index('country').loc[CC]
    ns.zero_line(a,'x')
    for i,(cc,r) in enumerate(e.iterrows()):
        a.plot(r.value,i,'o',color=C['AE'],ms=3.4,mfc='white' if cc in ['RO','PT'] else C['AE'])
    a.set_yticks(range(11),CC);a.set_ylim(10.6,-.6)
    a.set_xlabel('ΔBA (%)')
    panel(f,'c',127,85,183,91)
    f.legend(handles=[Line2D([],[],marker='o',color=C['AE'],lw=0,label='AE'),
                      Line2D([],[],marker='o',color=C['S2'],lw=0,label='S2'),
                      Line2D([],[],marker='o',color=C['MUTED'],mfc='white',lw=0,label='RO / PT')],
             loc='lower center',bbox_to_anchor=(.5,.01),ncol=3)
    save(f,'fig_zero_label_accuracy',
        'Zero-target-label multi-source transfer with LightGBM. (a) Overall accuracy; (b) balanced accuracy at '
        'the default argmax decision rule (tau=0). Each point is the equal mean of two disjoint spatial evaluation halves. '
        'BA uses the frozen eligible-class set in each half, identical for AE and S2. Source: zero_label_countries, '
        'country × feature rows; each metric reconciles to the frozen analysis within 1e-9 pp. '
        '(c) AE minus S2 BA at matched OA, from the existing confirm_country_values table, lgb / ae-s2 / D_Z0 rows. '
        'Open markers identify RO and PT. No target labels enter training. Points are descriptive country estimates; no error bars.')

def confusion_figure(data,country=None):
    f,ax=canvas(183,116)
    for j,feat in enumerate(['ae','s2']):
        a=ax(27+89*j,40,61,61)
        sub=data[data.feat==feat]
        if country is None:
            m=sub.pivot(index='true',columns='predicted',values='mean').reindex(index=CL,columns=CL)
            n=sub.groupby('true')['count'].first().reindex(CL)
            rows=[f'{l} ({int(k)})' for l,k in zip(LABELS,n)]
        else:
            sub=sub[sub.country==country]
            m=sub.pivot(index='true',columns='predicted',values='row_pct').reindex(index=CL,columns=CL)
            n=sub.groupby('true').support.first().reindex(CL)
            rows=[f'{l} ({int(k):,})' for l,k in zip(LABELS,n)]
        im=heat(a,m,rows,LABELS)
        a.set_xlabel('Predicted class',labelpad=2)
        a.set_title(feat.upper()+(f' · {country}' if country else ''),fontsize=7,color=C[feat.upper()])
        panel(f,chr(97+j),1+89*j,106,183,116)
    cb=f.colorbar(im,cax=ax(52,7,80,1.8),orientation='horizontal')
    cb.set_label('Share of each true class predicted into each class (%)',labelpad=1)
    return f

def confusion():
    d=read('confusion_country_equal.csv')
    f=confusion_figure(d)
    save(f,'fig_zero_label_confusion',
        'Row-normalised confusion matrices for zero-target-label multi-source transfer at tau=0. '
        'For each true class, rows are first averaged over eligible halves within each country, then equally over countries. '
        'Eligibility follows the frozen BA class sets; unsupported half-class combinations are omitted, not set to zero. '
        'Numbers in row labels give the contributing number of countries, which varies by class. '
        'Cell annotations are percentages rounded to the nearest integer; diagonal entries are class recalls. '
        'Identical rows and test parcels are used for AE and S2. Source: confusion_country_equal, '
        'feature × true × predicted rows. These descriptive matrices do not represent a globally pooled OA or the '
        'country-averaged BA estimand. Per-country matrices with sample counts are provided separately.')
    return
    raw=read('confusion_countries.csv')
    with PdfPages(OUT/'confusion_by_country.pdf') as pdf:
        for cc in CC:
            f=confusion_figure(raw,cc)
            f.text(.5,.965,'True-class sample counts in parentheses; grey rows have no test labels',ha='center',fontsize=5.5)
            pdf.savefig(f,bbox_inches='tight',pad_inches=.02)
            f.savefig(OUT/f'confusion_{cc}.png',dpi=300,bbox_inches='tight',pad_inches=.02)
            plt.close(f)
    (OUT/'confusion_by_country_legend.txt').write_text(
        'One page per target country, AE and S2 on the same zero-label test parcels. Counts from two disjoint '
        'spatial halves are pooled within each country, then each true-class row is normalised. '
        'All observed classes are displayed, including rare classes excluded from formal BA. Parentheses give '
        'true-class test counts; grey rows mean no observations. Read small-support rows cautiously. '
        'Source: confusion_countries, country × feature × true × predicted rows. No interval estimates.\n')

def pairwise():
    d=read('pairwise_transfer.csv')
    mats={feat:d[d.feature==feat].pivot(index='source',columns='target',values='BA').reindex(index=CC,columns=CC) for feat in ['ae','s2']}
    f,ax=canvas(183,88)
    for j,(title,m) in enumerate([('AE',mats['ae']),('S2',mats['s2']),('AE − S2',mats['ae']-mats['s2'])]):
        a=ax(12+61*j,27,46,46)
        norm=TwoSlopeNorm(vmin=-15,vcenter=0,vmax=15) if j==2 else None
        im=heat(a,m,CC,CC,cmap=DIVERGE if j==2 else BLUE,fmt='+.0f' if j==2 else '.0f',norm=norm)
        a.set_xticklabels(CC,rotation=0);a.set_xlabel('Target country',labelpad=2)
        if j==0:a.set_ylabel('Source country',labelpad=2)
        a.set_title(title,fontsize=7,color=C.get(title,C['TEXT']))
        panel(f,chr(97+j),1+61*j,79,183,88)
        cb=f.colorbar(im,cax=ax(15+61*j,11,40,1.8),orientation='horizontal')
        cb.set_label('ΔBA (%)' if j==2 else 'Balanced accuracy (%)',labelpad=1)
    save(f,'fig_pairwise_transfer',
        'Single-source cross-country transfer on the repaired paired-feature dataset. Rows are training countries; '
        'columns are target countries. (a) AE BA; (b) S2 BA; (c) paired AE minus S2 BA. '
        'All 110 directed cross-country pairs per feature are shown; diagonal domestic tests are masked because '
        'they use a different split. BA includes every class observed in the target country, including classes '
        'absent from the source. Each source used a stratified 70/10/20 train/early-stopping/domestic-test split; '
        'cross-country evaluation uses all retained target parcels, a single seed and no spatial target holdout. '
        'This earlier descriptive experiment is distinct from the formal ten-source buffered evaluation; absolute '
        'scores must not be subtracted across protocols. Source: pairwise_transfer, all source × target × feature rows. '
        'Cells are rounded to integer percentages/pp; no uncertainty intervals or confirmatory inference.')

if __name__=='__main__':
    baseline();confusion()
    (OUT/'style_qa.json').write_text(json.dumps(QA,indent=2))
