"""图1：抽象跨国迁移任务、两种采样起点、表征与样本量的关系。"""
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
W,H=182,169
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
# a：国家边界真实，点的位置及类别组成仅为概念示意；地图分别缩放。
import geopandas as gpd
from shapely.geometry import Point
geo=gpd.read_file(source/'countries.geojson').to_crs(3035)
panel(2,165,'a','Cross-country transfer under unequal class support')
cropcols=['#A7B5CF','#C5AABD','#D5D6D8']
rng=np.random.default_rng(42)
def domain(x,y,w,h,country,mixture=None):
    shape=geo.loc[geo.CNTR_CODE==country,'geometry'].iloc[0]
    lo_x,lo_y,hi_x,hi_y=shape.bounds
    pts=[]
    while len(pts)<36:
        q=rng.uniform([lo_x,lo_y],[hi_x,hi_y])
        if shape.contains(Point(*q)):pts.append(q)
    pts=np.array(pts)
    m=fig.add_axes([x/W,y/H,w/W,h/H])
    gpd.GeoSeries([shape],crs=3035).plot(ax=m,facecolor='white',edgecolor='#A9A9A9',linewidth=.55)
    if mixture is None:
        m.scatter(pts[:,0],pts[:,1],s=3,facecolors='white',edgecolors='#A2A2A2',linewidths=.45)
    else:
        idx=np.repeat(np.arange(3),mixture);rng.shuffle(idx)
        m.scatter(pts[:,0],pts[:,1],s=3,c=np.array(cropcols)[idx],linewidths=0)
    m.set_aspect('equal');m.axis('off')
for x,mix,code,label in [(5,[28,6,2],'AT','Austria'),(30,[8,27,1],'LT','Lithuania'),(55,[31,5,0],'SE','Sweden')]:
    domain(x,142,19,17,code,mix);txt(x+9.5,139,label,5.5,ha='center')
txt(40,160.5,'Source countries (examples)',5.5,ha='center')
arrow(82,150,124,150,C['REF_Z0'])
txt(103,154,'Transfer a crop classifier',6,ha='center')
txt(103,145.5,'Geographic and class-distribution shift',5.5,ha='center',color=C['MUTED'])
domain(143,143,24,15,'BE')
txt(155,139,'Belgium (target example)',6,ha='center')
txt(40,132.5,'Unequal class frequencies; some classes unrepresented',5.5,ha='center')
txt(148,132.5,'No labels or a limited local survey',5.5,ha='center')
for x,col,label in zip([6,24,42],cropcols,['Class 1','Class 2','Class 3']):
    ax.plot(x,128.5,'o',ms=2.5,color=col);txt(x+2,128.5,label,5.5)
txt(178,128.5,'Real country outlines; illustrative class composition',5.5,ha='right',color=C['MUTED'])
line(125)
# b：地图保持目前图3的点云、测试格和采样点样式，不加演示文稿式卡片。
panel(2,120.5,'b','Two starting conditions for local label collection')
txt(5,114,'A  No local training labels',6.5,color=C['LABEL'])
txt(97,114,'B  An existing clustered survey',6.5,color=C['CONC'])
ax.plot([91,91],[68,115],color=C['GRID'],lw=.5)
parcels(5,85,33,26)
parcels(49,85,33,26,add='random')
arrow(39,98,48,98,C['LABEL'])
txt(21.5,81,'0 local labels',5.5,ha='center')
txt(65.5,81,'+ 50 / 100 / 200 / 400',5.5,ha='center',color=C['LABEL'])
txt(44,74,'Random sampling across the target country',6,ha='center')
parcels(98,96,28,17,existing=True)
txt(131,105,'100 existing labels',6,color=C['CONC'])
txt(131,100,'Retained in every addition',5.5)
arrow(111,96,111,92,C['CONC']);ax.plot([111,169],[92,92],color=C['CONC'],lw=.6)
for x,kind,title in [(98,'local','Local extension'),(125,'spread','National random'),(152,'betti','Betti allocation')]:
    arrow(x+12,92,x+12,90,C['CONC'])
    parcels(x,70,26,20,existing=True,add=kind)
    txt(x+13,67.5,title,5.5,ha='center')
txt(145,94,'+ 100 or 300 new labels',5.5,ha='center',color=C['CONC'])
ax.plot(6,61,'o',color=C['CONC'],ms=3);ax.plot(8,61,'o',color='#D2AFCC',ms=3);txt(11,61,'Existing labels',5.5)
ax.plot(39,61,'o',color=C['LABEL'],ms=3);txt(42,61,'New labels',5.5)
ax.add_patch(Rectangle((66,60),2,1.8,facecolor='#BBBBBB',edgecolor='none'));txt(70,61,'Test cells',5.5)
ax.add_patch(Rectangle((92,60),2,1.8,facecolor='#F1E3EF',edgecolor='none'));txt(96,61,'Surveyed regions',5.5)
txt(179,57,'Maps: Belgium example; 100 new labels shown per addition',5.5,ha='right',color=C['MUTED'])
line(54)
# c：同一学习曲线上呈现纵向的表征收益和横向的标签需求，不把假说包装成结果。
panel(2,49.5,'c','Representation and local sample size')
ax.plot([5,12],[41,41],color=C['S2'],lw=1.1);txt(15,41,'Sentinel-2 statistics (S2)',6.5)
ax.plot([5,12],[35,35],color=C['AE'],lw=1.3);txt(15,35,'AlphaEarth embeddings (AE)',6.5)
txt(5,26,'At fixed sample size',6,weight='bold')
txt(5,21.5,'Does AE improve balanced accuracy?',6)
txt(5,13.5,'At a common accuracy target',6,weight='bold')
txt(5,9,'Can AE require fewer local labels?',6)
left,bottom,cw,ch=98,12,69,31
arrow(left,bottom,left+cw+2,bottom,C['TEXT'],lw=.6)
arrow(left,bottom,left,bottom+ch+1,C['TEXT'],lw=.6)
xx=np.linspace(0,1,200)
ys=.10+.76*(1-np.exp(-2.4*xx));ya=.38+.53*(1-np.exp(-2.4*xx))
for yy,col,lw in [(ys,C['S2'],1.1),(ya,C['AE'],1.3)]:
    ax.plot(left+cw*xx,bottom+ch*yy,color=col,lw=lw)
target=.66
xa=-np.log(1-(target-.38)/.53)/2.4;xs=-np.log(1-(target-.10)/.76)/2.4
yt=bottom+ch*target
ax.plot([left,left+cw*xs],[yt,yt],color='#A9A9A9',lw=.65,ls=(0,(2,2)))
txt(left+2,yt+2.7,'Common accuracy target',5.5,color=C['MUTED'])
for x,col,label in [(xa,C['AE'],'n(AE)'),(xs,C['S2'],'n(S2)')]:
    xp=left+cw*x;ax.plot([xp,xp],[bottom,yt],color=col,lw=.65,ls=(0,(2,2)))
    ax.plot(xp,yt,'o',ms=3,mfc='white',mec=col,mew=.8)
    txt(xp,9,label,5.5,color=col,ha='center')
arrow(left+cw*xa,6,left+cw*xs,6,C['TEXT'],'<->',lw=.65)
x=.12;y1=.10+.76*(1-np.exp(-2.4*x));y2=.38+.53*(1-np.exp(-2.4*x))
arrow(left+cw*x,bottom+ch*y1,left+cw*x,bottom+ch*y2,C['TEXT'],'<->',lw=.7)
txt(left+cw*x+2,bottom+ch*(y1+y2)/2,'ΔBA',5.5)
txt(169,bottom+ch*ya[-1]+.8,'AE',6,color=C['AE'])
txt(169,bottom+ch*ys[-1]-.8,'S2',6,color=C['S2'])
txt(91,28,'BA at matched OA',6,ha='center',rotation=90)
txt(left,bottom-2.7,'0',5.5,ha='center')
txt(135,2,'Number of local training labels, n',6,ha='center')
txt(5,2,'Conceptual curves; hypothesis, not fitted results',5.5,color=C['MUTED'])
OUT=HERE/'out';OUT.mkdir(exist_ok=True)
fig.canvas.draw();texts=[t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]
assert min(t.get_fontsize() for t in texts)>=5.5
ns.save(fig,'fig_concept',str(OUT))
legend='Research framework for representation and label requirements in cross-country crop mapping. (a) Real country outlines (Austria, Lithuania and Sweden as source examples; Belgium as a target example) illustrate the cross-country transfer setting. Coloured points illustrate unequal class support and changes in class composition. A classifier is transferred to a target with no or limited local training labels. Boundaries use EPSG:3035 and are independently scaled for legibility, not shown at a common geographic scale. Point positions and class counts are schematic and do not represent observed parcel locations or national crop statistics. The depicted source countries are examples, not the complete foreign training pool. (b) Two local sampling settings: starting without labels and adding 50–400 random labels, or retaining an existing clustered survey of 100 labels and adding 100 or 300 through local extension, national random sampling or Betti allocation. Maps reproduce the actual Belgium BE_F1 repeat-0 parcel and sampling records, showing additions of 100 labels. Grey cells are spatial test cells and pale purple cells are surveyed regions; existing labels are faded when additions are shown. (c) The representation–sample-size hypothesis has two readings: a vertical accuracy difference at fixed label count and a horizontal difference in label requirements at a common accuracy target. Both representations use matched foreign labels, local labels, learners and spatial tests. Curves are conceptual and do not estimate empirical learning curves or labels saved; comparisons at a common target require that both methods attain that target. BA is balanced accuracy and OA is overall accuracy.'
(OUT/'fig_concept_legend.txt').write_text(legend+'\n')
(OUT/'figure_contract.json').write_text(json.dumps(dict(width_mm=W,height_mm=H,font=ns.FONT,min_font_pt=5.5,empirical='Panel b: Belgium BE_F1 repeat 0 sampling maps',schematic='Panel a points and class composition; panel c hypothesis curves',scenarios={'A':'0 + 50/100/200/400','B':'100 + 100/300; EXT/SRS/Betti'},style='shared nstyle.py; sampling maps follow current Figure 3'),indent=2))
print('Exported three-panel research concept.')
