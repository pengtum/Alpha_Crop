"""图1：真实不均衡证据、两种迁移起点，以及表征与标签的研究假说。"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='2'
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,FancyBboxPatch,FancyArrowPatch
from matplotlib.text import Text
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import nstyle as ns
ns.setup();C=ns.C
W,H=182,166
fig=plt.figure(figsize=(W*ns.MM,H*ns.MM))
ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,W),ylim=(0,H));ax.axis('off')
def txt(x,y,s,size=6.5,color=None,ha='left',weight='normal',**kw):
    return ax.text(x,y,s,fontsize=size,color=color or C['TEXT'],ha=ha,va='center',fontweight=weight,**kw)
def arrow(x,y,xx,yy,color=None,style='-|>',lw=.8):
    ax.add_patch(FancyArrowPatch((x,y),(xx,yy),arrowstyle=style,mutation_scale=7,linewidth=lw,color=color or C['MUTED'],shrinkA=0,shrinkB=0))
def box(x,y,w,h,color,fill='white',lw=.7):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=1.2',edgecolor=color,facecolor=fill,linewidth=lw))
def line(y):ax.plot([2,180],[y,y],color=C['GRID'],lw=.6)
def panel(x,y,l,title):txt(x,y,l,8,weight='bold');txt(x+6,y,title,7,weight='bold')
# 地图沿用原图的真实 BE_F1 地块、测试网格和调查区域；补采位置来自冻结选择。
from matplotlib.collections import PatchCollection
source=HERE.parent/'source_data'
lay=pd.read_csv(source/'example_unit_layout.csv')
cells=pd.read_csv(source/'example_unit_cells.csv')
tcells=pd.read_csv(source/'example_unit_test_cells.csv')
rt=pd.read_csv(HERE/'example_rt100.csv')
meta=json.loads((HERE/'map_metadata.json').read_text())
xlim=(cells.cx.min()*10-6,(cells.cx.max()+1)*10+6)
ylim=(cells.cy.min()*10-6,(cells.cy.max()+1)*10+6)
def parcels(x,y,w=24,h=18,existing=False,add=None):
    m=fig.add_axes([x/W,y/H,w/W,h/H],zorder=2)
    m.set_facecolor('white')
    for role,col in [('candidate','#D8D8D8'),('test','#ADADAD')]:
        pts=lay[lay.role==role]
        m.scatter(pts.x_km,pts.y_km,s=.3,c=col,linewidths=0,rasterized=True,zorder=1)
    m.add_collection(PatchCollection([Rectangle((r.cx*10,r.cy*10),10,10) for r in tcells.itertuples()],facecolor='#4D4D4D',alpha=.18,edgecolor='none',zorder=2))
    if existing:
        reg=cells[cells.S0_region]
        m.add_collection(PatchCollection([Rectangle((r.cx*10,r.cy*10),10,10) for r in reg.itertuples()],facecolor=C['CONC'],alpha=.1,edgecolor='none',zorder=2))
        pts=lay[lay.role=='S0']
        col=tuple(.55+.45*v for v in __import__('matplotlib').colors.to_rgb(C['CONC'])) if add else C['CONC']
        m.scatter(pts.x_km,pts.y_km,s=1.2 if add else 2.5,c=[col],linewidths=0,zorder=4)
    if add:
        roles={'spread':'SRS100','local':'EXT100','betti':'BA100'}
        col={'random':C['LABEL'],'spread':C['SPREAD'],'local':C['CONC'],'betti':C['BETTI']}[add]
        pts=rt if add=='random' else lay[lay.role==roles[add]]
        m.scatter(pts.x_km,pts.y_km,s=2.5,c=col,marker='o',linewidths=0,zorder=5)
    m.set(xlim=xlim,ylim=ylim);m.set_aspect('equal');m.axis('off')
    # 复用图3右上角的比例尺位置，全部地图保持相同范围。
    xb=xlim[1]-32;yb=5641.
    m.plot([xb,xb+20],[yb,yb],color=C['TEXT'],lw=.7)
    m.text(xb+10,yb+3,'20 km',fontsize=5.5,ha='center',va='bottom')
# a：直接从冻结数据读取，选取三种具有不同占优类别的国家。
counts=pd.read_csv(HERE.parent/'revision_20261007/source_data/class_counts.csv',index_col=0)
panel(2,161,'a','One transfer problem: uneven labels, shifting crop mixtures')
txt(3,152,'611,519 parcels · 11 countries · 10 classes',6,weight='bold')
cols=['#819E87','#B7A1B5','#D2B889','#DFE2E4']
for y,co,name in [(144,'SE','Sweden'),(136.5,'PT','Portugal'),(129,'RO','Romania')]:
    row=counts.loc[co];vals=[row.grassland,row.orchards,row.other_crops,row.sum()-row.grassland-row.orchards-row.other_crops]
    txt(3,y,name,6)
    start=23
    for v,col in zip(vals,cols):
        ww=59*v/row.sum();ax.add_patch(Rectangle((start,y-2.3),ww,4.6,facecolor=col,edgecolor='white',lw=.3))
        if ww>12:txt(start+ww/2,y,f'{100*v/row.sum():.0f}%',5.5,ha='center')
        start+=ww
for x,col,label in zip([3,25,46,67],cols,['Grassland','Orchards','Other crops','Remaining']):
    ax.add_patch(Rectangle((x,121),1.8,1.8,facecolor=col,edgecolor='none'));txt(x+2.8,121.9,label,5.5)
ax.plot([89,89],[122,154],color=C['GRID'],lw=.6)
txt(96,152,'Dominant classes can hide minority-class failure',6.5,weight='bold')
txt(96,143.5,'44–77%',12,color=C['LABEL'],weight='bold');txt(127,143.5,'largest-class share within a country',5.5)
txt(96,134.5,'421 : 1',12,color=C['LABEL'],weight='bold');txt(127,134.5,'grassland : strawberries, pooled',5.5)
txt(96,125.5,'11 / 110',12,color=C['LABEL'],weight='bold');txt(127,125.5,'country–class cells with no labels',5.5)
# 两个路径共用同一来源池；明确训练标签，而非无验证标签。
box(3,106,176,10,C['GRID'],'#F7F8F9')
txt(91,111,'Foreign training pool: 10 source countries × 5,000 labels  →  one target country',6.5,ha='center',weight='bold')
arrow(46,106,46,101);arrow(137,106,137,101)
# b：没有本地训练标签。
box(2,45,86,55,'#B8CADA','white')
panel(5,95,'b','Scenario A | Start with no local labels')
parcels(6,64,31,23)
parcels(50,64,31,23,add='random')
arrow(38,77,50,77,C['LABEL'])
txt(22,89,'Zero-label transfer',6,ha='center',weight='bold')
txt(66,89,'Collect local labels',6,ha='center',weight='bold')
txt(22,62,'0 target training labels',5.5,ha='center')
txt(66,62,'+ 50 / 100 / 200 / 400',5.5,ha='center',color=C['LABEL'])
txt(45,55,'Random sampling across the target country',6,ha='center')
txt(45,49,'How much can labels add after transfer?',6,ha='center',weight='bold')
# c：已有调查作为起点，三个方案都保留原始调查。
box(94,45,86,55,'#D5BBD1','white')
panel(97,95,'c','Scenario B | Start with an existing survey')
parcels(98,75,29,18,existing=True)
txt(131,85,'100 existing labels',6,color=C['CONC'],weight='bold')
txt(131,79,'Spatially clustered survey',5.5)
arrow(111,75,111,72,C['CONC']);ax.plot([111,168],[72,72],color=C['CONC'],lw=.6)
for x,kind,title in [(98,'local','Extend locally'),(125,'spread','Spread nationally'),(152,'betti','Betti allocation')]:
    arrow(x+12,72,x+12,70,C['CONC'])
    parcels(x,49.5,26,20,existing=True,add=kind)
    txt(x+13,47.5,title,5.5,ha='center')
txt(145,74,'+ 100 or 300 new labels',5.5,ha='center',color=C['CONC'])
# 公共图例：颜色遵循论文采样语义；点是旧样本，加号为新增。
ax.plot(6,40,'o',color=C['CONC'],ms=3);ax.plot(8,40,'o',color='#D2AFCC',ms=3);txt(11,40,'Existing labels',5.5)
ax.plot(39,40,'o',color=C['LABEL'],ms=3);txt(42,40,'New labels (colour = sampling strategy)',5.5)
txt(179,40,'Belgium · BE_F1 · maps show 100 new labels',5.5,ha='right',color=C['MUTED'])
ax.add_patch(Rectangle((6,35.5),2,1.8,facecolor='#D8D8D8',edgecolor='none'));txt(10,36.4,'Parcel locations',5.5)
ax.add_patch(Rectangle((39,35.5),2,1.8,facecolor='#BBBBBB',edgecolor='none'));txt(43,36.4,'10-km test cells',5.5)
ax.add_patch(Rectangle((76,35.5),2,1.8,facecolor='#F1E3EF',edgecolor='none'));txt(80,36.4,'Surveyed regions',5.5)
line(33.5)
# d：共同表征对照与假说，曲线不宣称结果。
panel(2,29.5,'d','In both scenarios: can richer representations reduce local label needs?')
box(4,14,33,10,C['S2'],'#F4F4F4');txt(20.5,20.5,'S2 statistics',6.5,ha='center',weight='bold',color=C['S2']);txt(20.5,16.4,'168 features',5.5,ha='center')
box(49,14,33,10,C['AE'],'#EFF7F7');txt(65.5,20.5,'AE embeddings',6.5,ha='center',weight='bold',color=C['AE']);txt(65.5,16.4,'128 features',5.5,ha='center')
txt(43,19,'vs',6,ha='center')
txt(43,9,'Same foreign pool, local labels, learner and test parcels',5.5,ha='center')
# 小曲线仅表达假说；无伪造精度或标签数量。
left,bottom,cw,ch=110,11,57,14
arrow(left,bottom,left+cw+2,bottom,C['TEXT'],lw=.55);arrow(left,bottom,left,bottom+ch+1,C['TEXT'],lw=.55)
xx=np.linspace(0,1,100);ys=.1+.7*(1-np.exp(-2.2*xx));ya=.38+.51*(1-np.exp(-2.2*xx))
for yy,col in [(ys,C['S2']),(ya,C['AE'])]:ax.plot(left+cw*xx,bottom+ch*yy,color=col,lw=1.1,ls=(0,(3,1.5)))
target=.62;xa=-np.log(1-(target-.38)/.51)/2.2;xs=-np.log(1-(target-.1)/.7)/2.2
ax.plot([left,left+cw],[bottom+ch*target]*2,color=C['GRID'],lw=.55,ls=':')
for x,col in [(xa,C['AE']),(xs,C['S2'])]:ax.plot([left+cw*x]*2,[bottom,bottom+ch*target],color=col,lw=.6,ls=':')
arrow(left+cw*xa,8.5,left+cw*xs,8.5,C['LABEL'],'<->');txt(140,5,'Fewer local labels?',5.5,ha='center',color=C['LABEL'])
txt(169,23,'AE',5.5,color=C['AE']);txt(169,19,'S2',5.5,color=C['S2'])
txt(105,18,'Accuracy',5.5,ha='center',rotation=90)
txt(43,3,'Spatial test · Compare BA at matched OA',5.5,ha='center',weight='bold')
txt(179,1.5,'Hypothesis curves, not measured results',5.5,ha='right',color=C['MUTED'])
OUT=HERE/'out';OUT.mkdir(exist_ok=True)
fig.canvas.draw();texts=[t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]
assert min(t.get_fontsize() for t in texts)>=5.5
ns.save(fig,'fig_concept',str(OUT))
legend='Research concept: richer representations, fewer local labels? (a) Observed label imbalance and shifts in crop composition. Bars show the retained label shares in three illustrative countries; summary statistics use all eleven countries. Remaining combines the seven classes other than grassland, orchards and other crops. Missing retained labels do not imply crop absence. For each target, 50,000 foreign labels are sampled from the other ten countries. (b) Scenario A starts with no target training labels and adds 50, 100, 200 or 400 random target labels. (c) Scenario B retains an existing clustered survey of 100 labels and adds 100 or 300 labels by local extension, nationwide random sampling or Betti allocation. (d) Both scenarios compare Sentinel-2 statistics (S2) with AlphaEarth embeddings (AE) under matched training and test conditions. The hypothesis is that richer representations could reach a common accuracy target with fewer local labels. Maps use actual parcel locations and frozen sampling selections for Belgium (BE_F1, repeat 0); the illustrated additions contain 100 labels. Grey cells indicate spatial test cells and pale purple cells indicate surveyed regions. Existing labels are faded when new labels are overlaid; all sampling markers are solid circles, matching current Figure 3. Background parcel points are subsampled for display. All maps use the same extent and local kilometre coordinates as the original design figure. Only dashed accuracy curves are schematic, not fitted results or estimates of labels saved. Empirical comparisons use spatially held-out parcels and BA at matched OA.'
(OUT/'fig_concept_legend.txt').write_text(legend+'\n')
(OUT/'figure_contract.json').write_text(json.dumps(dict(width_mm=W,height_mm=H,font=ns.FONT,min_font_pt=5.5,empirical='Panel a: frozen class_counts.csv',schematic='Panel d: hypothesis curves',map_data='BE_F1 repeat 0: actual parcels and frozen selections; 100 new labels shown',scenarios={'A':'0 + 50/100/200/400','B':'100 + 100/300; EXT/SRS/Betti'},style='shared nstyle.py'),indent=2))
print('Exported revised Figure 1.')
