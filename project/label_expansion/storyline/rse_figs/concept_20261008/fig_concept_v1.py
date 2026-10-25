"""研究问题概念图：迁移情境、表征与标签的配对设计、标签节省假说。

图件契约：示意图主导，183×132 mm。直接复用共享样式；不改动已有图件。
田块和曲线均为概念示意，不是空间数据或拟合结果；没有虚构定量结果。
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[k]='2'
from pathlib import Path
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch
from matplotlib.text import Text

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import nstyle as ns
ns.setup()
assert ns.FONT=='Arial'
C=ns.C
W,H=183,132
fig=plt.figure(figsize=(W*ns.MM,H*ns.MM))
ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,W),ylim=(0,H));ax.axis('off')

def txt(x,y,s,size=6.5,color=None,ha='left',va='center',weight='normal',**kw):
    return ax.text(x,y,s,fontsize=size,color=color or C['TEXT'],ha=ha,va=va,fontweight=weight,**kw)

def arrow(x1,y1,x2,y2,color=None,style='-|>',lw=.9,**kw):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle=style,mutation_scale=7,
        linewidth=lw,color=color or C['MUTED'],shrinkA=0,shrinkB=0,**kw))

def box(x,y,w,h,edge,fill='white',lw=.7,r=1.2):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={r}',
        linewidth=lw,edgecolor=edge,facecolor=fill))

def fields(x,y,w,h,labelled=True,variant=0):
    # 田块是抽象图标，颜色仅表示源池中不同的作物标签。
    palette=['#B4C0E4','#D8C1D5','#D9D9D9']
    mixtures=[[0,0,0,0,1,2],[1,1,1,0,0,2],[0,0,0,0,0,2],[2,2,1,1,1,0]]
    for i in range(3):
        for j in range(2):
            color=palette[mixtures[variant%4][i+3*j]] if labelled else '#F4F5F6'
            ax.add_patch(Rectangle((x+i*w/3,y+j*h/2),w/3-.45,h/2-.45,
                facecolor=color,edgecolor='white' if labelled else '#C1C4C7',linewidth=.4))
            if not labelled:txt(x+(i+.5)*w/3-.2,y+(j+.5)*h/2-.2,'?',size=6,color=C['MUTED'],ha='center')

# a｜跨国家迁移：源国标签在目标国训练之前已经存在。
txt(2,127,'a',size=8,weight='bold')
txt(8,127,'Transfer to a country with scarce local labels',size=7,weight='bold')
txt(3,117,'Labelled source countries',weight='bold')
for j in range(2):
    for i in range(5):fields(3+i*8.5,95+j*8.5,7.4,6.4,True,i+j)
txt(23,90,'10 countries · 50,000 parcels',size=5.5,ha='center')
arrow(48,104,75,104,color=C['REF_Z0'],lw=1.1)
txt(61.5,111,'Train abroad',size=6,ha='center')
txt(61.5,98,'Transfer',size=6,ha='center')
fields(80,95,24,17,False)
txt(92,117,'Target country',weight='bold',ha='center')
txt(92,90,'No labels / an existing survey',size=5.5,ha='center')
ax.plot([117,117],[91,119],color='#D8D8D8',lw=.6)
txt(124,117,'What changes across borders?',size=6.5,weight='bold')
for yy,t in [(110,'Crop conditions and phenology'),(103,'Class proportions and imbalance'),(96,'Which classes have labels')]:
    ax.add_patch(Rectangle((124,yy-.65),1.2,1.2,facecolor=C['MUTED'],edgecolor='none'))
    txt(128,yy,t,size=6)
ax.plot([2,181],[84,84],color='#D8D8D8',lw=.6)

# b｜共同迁移任务下的两个干预维度，不把标签选择混同于表示改善。
txt(2,78,'b',size=8,weight='bold')
txt(8,78,'Improve the representation or collect labels?',size=7,weight='bold')
txt(37,69,'No local labels',size=6,ha='center')
txt(73,69,'Add local labels',size=6,ha='center')
for y,feature,color,fill in [(48,'S2',C['S2'],'#F5F5F5'),(23,'AE',C['AE'],'#EFF7F7')]:
    box(24,y,26,17,color,fill)
    box(60,y,26,17,color,fill)
    txt(37,y+10.7,feature,size=8,color=color,ha='center',weight='bold')
    txt(73,y+10.7,feature,size=8,color=color,ha='center',weight='bold')
    txt(37,y+4.4,'0 local labels',size=6,ha='center')
    txt(73,y+4.4,'+ k local labels',size=6,ha='center')
    arrow(51.5,y+8.5,58.5,y+8.5,color=C['LABEL'])
arrow(37,46,37,42,color=C['AE']);arrow(73,46,73,42,color=C['AE'])
txt(18,56.5,'S2 statistics',size=6,color=C['S2'],ha='right')
txt(18,31.5,'AE embeddings',size=6,color=C['AE'],ha='right')
txt(55,43.5,'Richer representation',size=5.5,color=C['AE'],ha='center',
    bbox=dict(facecolor='white',edgecolor='none',pad=.5))
txt(55,18,'Same foreign labels · learner · test parcels',size=5.5,ha='center')
txt(3,11,'If a survey exists: extend locally or spread nationally?',size=5.5,color=C['CONC'])

# c｜标签节省的概念定义：用问号与明确注记防止被误读为实验结论。
ax.plot([97,97],[12,79],color='#D8D8D8',lw=.6)
txt(102,78,'c',size=8,weight='bold')
txt(108,78,'Can richer features reduce label needs?',size=7,weight='bold')
left,bottom,cw,ch=113,29,62,38
arrow(left,bottom,left+cw+1,bottom,color=C['TEXT'],lw=.6)
arrow(left,bottom,left,bottom+ch+1,color=C['TEXT'],lw=.6)
xx=np.linspace(0,1,160)
ys=.12+.68*(1-np.exp(-2.2*xx))
ya=.39+.49*(1-np.exp(-2.2*xx))
ax.plot(left+cw*xx,bottom+ch*ys,color=C['S2'],lw=1.2,ls=(0,(3,1.7)))
ax.plot(left+cw*xx,bottom+ch*ya,color=C['AE'],lw=1.4,ls=(0,(3,1.7)))
target=.64
xa=-np.log(1-(target-.39)/.49)/2.2
xs=-np.log(1-(target-.12)/.68)/2.2
yt=bottom+ch*target
ax.plot([left,left+cw],[yt,yt],color='#A7A7A7',lw=.6,ls=(0,(1.5,2)))
for x,col in [(xa,C['AE']),(xs,C['S2'])]:
    xp=left+cw*x
    ax.plot([xp,xp],[bottom,yt],color=col,lw=.65,ls=(0,(1.5,2)))
    ax.plot(xp,yt,'o',ms=3,mfc='white',mec=col,mew=.85)
txt(116,yt+3.2,'Common accuracy target',size=5.5)
txt(177,bottom+ch*ya[-1],'AE',size=6,color=C['AE'])
txt(177,bottom+ch*ys[-1]-1.2,'S2',size=6,color=C['S2'])
txt(106,48,'BA at matched OA',size=6,rotation=90,ha='center')
txt(left,bottom-3.5,'0',size=5.5,ha='center')
arrow(left+cw*xa,25,left+cw*xs,25,color=C['LABEL'],style='<->',lw=.85)
txt((2*left+cw*(xa+xs))/2,21.5,'Fewer labels?',size=6,color=C['LABEL'],ha='center')
txt(145,16,'Number of local labels',size=6,ha='center')
txt(145,10.5,'Hypothesis schematic · not measured curves',size=5.5,color=C['MUTED'],ha='center')

ax.plot([2,181],[7,7],color='#D8D8D8',lw=.6)
txt(91.5,3,'Test on spatially held-out parcels · Compare balanced accuracy (BA) at matched overall accuracy (OA)',
    size=5.5,ha='center')

OUT=HERE/'out';OUT.mkdir(exist_ok=True)
fig.canvas.draw()
texts=[t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]
assert min(t.get_fontsize() for t in texts)>=5.5
ns.save(fig,'fig_concept',out_dir=str(OUT))
(OUT/'fig_concept_legend.txt').write_text(
    'Research concept: can a richer representation reduce local label requirements after cross-country transfer? '
    '(a) For each target country, a source pool of 50,000 labelled parcels from the other ten countries is used '
    'to train a crop classifier. Target countries differ in crop conditions, class proportions and label support. '
    'The target initially has no local training labels or an existing clustered survey. '
    '(b) Paired comparisons cross the feature representation (Sentinel-2 statistics, S2; AlphaEarth embeddings, AE) '
    'with the addition of local labels, holding the foreign pool, learner and test parcels fixed. Existing surveys '
    'also motivate comparisons of concentrated and nationally spread additions. '
    '(c) The label-saving hypothesis: a higher accuracy curve could reach a common accuracy target with fewer '
    'local labels. Dashed curves and field icons are conceptual illustrations, not fitted results, geographic '
    'boundaries or estimates of a label-saving quantity. All empirical comparisons use spatial tests and BA '
    'at matched OA; the applicability of a common target depends on attainable accuracy.\n')
(OUT/'figure_contract.json').write_text(json.dumps(dict(archetype='schematic-led composite',
    purpose='Research questions, not empirical results',font=ns.FONT,width_mm=W,height_mm=H,
    source_design='10 source countries; 5,000 labelled parcels per country',
    synthetic_elements=['abstract field icons','explicitly labelled hypothesis curves'],
    minimum_font_pt=5.5),indent=2))
print('Concept figure exported in SVG, PDF, TIFF and PNG.')
