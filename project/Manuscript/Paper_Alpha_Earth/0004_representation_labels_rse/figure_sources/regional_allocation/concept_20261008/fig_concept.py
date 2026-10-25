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
# a：真实国家轮廓与源国/目标国角色矩阵，明确11轮留一国迁移。
import geopandas as gpd
geo=gpd.read_file(source/'countries.geojson').to_crs(3035)
countries=['AT','BE','DE','EE','LT','LV','PT','RO','SE','SI','SK']
panel(2,165,'a','Each country serves as the target in turn')
source_color='#B4C0E4';target_color=C['LABEL']
xs=np.linspace(36,175,11)
for x,co in zip(xs,countries):
    m=fig.add_axes([(x-5.5)/W,148/H,11/W,10/H])
    geo[geo.CNTR_CODE==co].plot(ax=m,facecolor='#EEEEEE',edgecolor='#999999',linewidth=.45)
    m.set_aspect('equal');m.axis('off')
    txt(x,145.5,co,5.5,ha='center')
for y,target,label in [(140,'AT','Target: AT'),(135,'BE','Target: BE'),(127.5,'SK','Target: SK')]:
    txt(3,y,label,6)
    ax.plot([29,180],[y,y],color='#EFEFEF',lw=.5,zorder=0)
    for x,co in zip(xs,countries):
        ax.add_patch(Rectangle((x-2.1,y-1.35),4.2,2.7,facecolor=target_color if co==target else source_color,edgecolor='none'))
        if co==target:txt(x,y,'T',5.5,color='white',ha='center')
ax.plot([16]*3,[130.8,131.5,132.2],'.',color=C['MUTED'],ms=1.4)
txt(109,131.5,'… rotate through the remaining countries …',5.5,ha='center',color=C['MUTED'])
ax.add_patch(Rectangle((80,160.5),2.5,1.8,facecolor=source_color,edgecolor='none'));txt(84,161.4,'10 source countries',5.5)
ax.add_patch(Rectangle((119,160.5),2.5,1.8,facecolor=target_color,edgecolor='none'));txt(123,161.4,'1 target country',5.5)
txt(178,161.4,'11 rotations',5.5,ha='right')
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
for x,kind,title in [(98,'local','Local extension'),(125,'spread','National random'),(152,'betti','Regional allocation')]:
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
# c：用同一组样本的两种概念特征空间与嵌套标签子集表示两条干预轴。
# 淡色点为类别组成示意，黑色轮廓标记已有标签；非实际降维或聚类结果。
panel(2,49.5,'c','Representation × local labels')
cropcols=['#8DA3C4','#B696B3','#C4B89D']
rng=np.random.default_rng(47)
noise=rng.normal(size=(3,18,2))*.62
centres_s=np.array([[-.5,.1],[.45,.35],[0,-.5]])
centres_a=np.array([[-1.6,.85],[1.5,.75],[0,-1.5]])
# 每列增加标签，保持点的坐标与类别、以及旧标签完全不变。
small_ids={0,18,36};large_ids=small_ids|{3,7,12,20,25,31,39,43,49}
def feature_cloud(x,y,w,h,rich,nlabels):
    centres=centres_a if rich else centres_s
    pts=(noise+centres[:,None,:]).reshape(-1,2)
    xx=x+(pts[:,0]+3.5)/7*w;yy=y+(pts[:,1]+3.5)/7*h
    classes=np.repeat(np.arange(3),18)
    for cl,col in enumerate(cropcols):
        z=classes==cl
        ax.scatter(xx[z],yy[z],s=4,c=col,alpha=.35,linewidths=0,zorder=2)
    selected=sorted(small_ids if nlabels==3 else large_ids)
    ax.scatter(xx[selected],yy[selected],s=9,c=np.array(cropcols)[classes[selected]],edgecolors='#333333',linewidths=.45,zorder=3)
    ax.plot([x,x+w],[y,y],color='#C8C8C8',lw=.45)
    ax.plot([x,x],[y,y+h],color='#C8C8C8',lw=.45)
for y,rich,label,col in [(29,False,'S2',C['S2']),(10,True,'AE',C['AE'])]:
    txt(5,y+7,label,6.5,color=col)
    feature_cloud(20,y,24,15,rich,3)
    feature_cloud(54,y,24,15,rich,12)
arrow(12.5,36,12.5,18,C['AE'],lw=.7)
arrow(31,6,67,6,C['TEXT'],lw=.65)
txt(49,6,'n',6,ha='center',bbox=dict(facecolor='white',edgecolor='none',pad=1))
# 标签量用空心圈数直接编码，避免文字说明。
for x,num in [(31,3),(67,12)]:
    for i in range(num):
        ax.plot(x+(i-(num-1)/2)*1.0,7.8,'o',ms=1.7,mfc='white',mec='#333333',mew=.45)
ax.plot(21,1.8,'o',ms=2.5,mfc=cropcols[0],mec='#333333',mew=.45)
txt(24,1.8,'Labelled',5.5)
for i,col in enumerate(cropcols):ax.plot(47+i*2,1.8,'o',ms=2.2,color=col,alpha=.6)
txt(54,1.8,'Classes (schematic)',5.5)
arrow(82,27,90,27,C['MUTED'],lw=.7)
# 右侧横纵两个差值分别对应左侧矩阵的两条轴。
left,bottom,cw,ch=101,12,66,31
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
for x,col,label in [(xa,C['AE'],'n(AE)'),(xs,C['S2'],'n(S2)')]:
    xp=left+cw*x;ax.plot([xp,xp],[bottom,yt],color=col,lw=.65,ls=(0,(2,2)))
    ax.plot(xp,yt,'o',ms=3,mfc='white',mec=col,mew=.8)
    txt(xp,9,label,5.5,color=col,ha='center')
arrow(left+cw*xa,6,left+cw*xs,6,C['TEXT'],'<->',lw=.65)
txt(left+cw*(xa+xs)/2,6,'Δn',5.5,ha='center',bbox=dict(facecolor='white',edgecolor='none',pad=1))
x=.12;y1=.10+.76*(1-np.exp(-2.4*x));y2=.38+.53*(1-np.exp(-2.4*x))
arrow(left+cw*x,bottom+ch*y1,left+cw*x,bottom+ch*y2,C['TEXT'],'<->',lw=.7)
txt(left+cw*x+2,bottom+ch*(y1+y2)/2,'ΔBA',5.5)
txt(169,bottom+ch*ya[-1]+.8,'AE',6,color=C['AE'])
txt(169,bottom+ch*ys[-1]-.8,'S2',6,color=C['S2'])
txt(94,28,'BA at matched OA',6,ha='center',rotation=90)
txt(left,bottom-2.7,'0',5.5,ha='center')
txt(168,9,'n',6,ha='center')
txt(178,1.8,'Conceptual feature spaces and curves',5.5,ha='right',color=C['MUTED'])
OUT=HERE/'out';OUT.mkdir(exist_ok=True)
fig.canvas.draw();texts=[t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]
assert min(t.get_fontsize() for t in texts)>=5.5
ns.save(fig,'fig_concept',str(OUT))
legend='Research framework for representation and label requirements in cross-country crop mapping. (a) Leave-one-country-out transfer: each of eleven countries serves as the target in turn, with the other ten supplying the foreign training pool (5,000 labels per source country). Columns identify countries using actual outlines and ISO codes. Each row is one target-country configuration; three rows are shown, with the intervening rotations indicated by an ellipsis. Dark blue T cells identify the target and pale blue cells the ten sources. Outlines are independently scaled and show national boundaries, not the extent of retained labels. (b) Two local sampling settings: starting without labels and adding 50–400 random labels, or retaining an existing clustered survey of 100 labels and adding 100 or 300 through local extension, national random sampling or regional allocation. Maps reproduce actual Belgium BE_F1 repeat-0 parcel and sampling records, showing additions of 100 labels. Grey cells are spatial test cells and pale purple cells are surveyed regions; existing labels are faded when additions are shown. (c) Conceptual feature spaces illustrate the same parcels under two representations (rows) and increasing local label counts (columns). Class colours and point positions are synthetic, not an empirical embedding projection; better class separation is a hypothesis, not an established property of AE. Black-outlined points indicate labelled parcels. The larger labelled subset includes the smaller one, and coordinates do not change when labels are added. The representation–sample-size hypothesis has two readings: a vertical accuracy difference at fixed label count and a horizontal difference in label requirements at a common accuracy target. Both representations use matched foreign labels, local labels, learners and spatial tests. Curves are conceptual and do not estimate empirical learning curves or labels saved; comparisons at a common target require that both methods attain that target. BA is balanced accuracy and OA is overall accuracy.'
(OUT/'fig_concept_legend.txt').write_text(legend+'\n')
(OUT/'figure_contract.json').write_text(json.dumps(dict(width_mm=W,height_mm=H,font=ns.FONT,min_font_pt=5.5,empirical='Panel b: Belgium BE_F1 repeat 0 sampling maps',schematic='Panel a source/target role matrix; panel c synthetic feature spaces and hypothesis curves',scenarios={'A':'0 + 50/100/200/400','B':'100 + 100/300; EXT/SRS/regional allocation'},style='shared nstyle.py; sampling maps follow current Figure 3'),indent=2))
print('Exported three-panel research concept.')
