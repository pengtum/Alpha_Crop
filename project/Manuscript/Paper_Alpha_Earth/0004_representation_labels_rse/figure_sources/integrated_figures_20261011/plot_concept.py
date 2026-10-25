"""图2：配对特征预处理、跨国轮换与两种采样起点。"""
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
OLD=HERE.parent/'regional_allocation/concept_20261008'
sys.path.insert(0,str(OLD.parent))
import nstyle as ns
ns.setup();C=ns.C
width_mm = 183
W,H=width_mm,199.5
BASE_H=169
SY=.85
plt.rcParams.update({'font.family':'Arial','svg.fonttype':'none','pdf.fonttype':42})
fig=plt.figure(figsize=(W*ns.MM,H*ns.MM))
ax=fig.add_axes([0,0,1,BASE_H*SY/H]);ax.set(xlim=(0,W),ylim=(0,BASE_H));ax.axis('off')
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
source=OLD.parent/'source_data'
lay=pd.read_csv(source/'example_unit_layout.csv')
cells=pd.read_csv(source/'example_unit_cells.csv')
tcells=pd.read_csv(source/'example_unit_test_cells.csv')
rt=pd.read_csv(OLD/'example_rt100.csv')
meta=json.loads((OLD/'map_metadata.json').read_text())
xlim=(cells.cx.min()*10-6,(cells.cx.max()+1)*10+6)
ylim=(cells.cy.min()*10-6,(cells.cy.max()+1)*10+6)
def parcels(x,y,w=24,h=18,existing=False,add=None):
    m=fig.add_axes([x/W,y*SY/H,w/W,h*SY/H],zorder=2)
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
    m.text(xb+10,yb+3,'20 km',fontsize=6,ha='center',va='bottom')
# a：真实国家轮廓与源国/目标国角色矩阵，明确11轮留一国迁移。
import geopandas as gpd
geo=gpd.read_file(source/'countries.geojson').to_crs(3035)
countries=['AT','BE','DE','EE','LT','LV','PT','RO','SE','SI','SK']
panel(2,165,'b','Each country serves as the target in turn')
source_color='#B4C0E4';target_color=C['LABEL']
xs=np.linspace(36,175,11)
for x,co in zip(xs,countries):
    m=fig.add_axes([(x-5.5)/W,148*SY/H,11/W,10*SY/H])
    geo[geo.CNTR_CODE==co].plot(ax=m,facecolor='#EEEEEE',edgecolor='#999999',linewidth=.45)
    m.set_aspect('equal');m.axis('off')
    txt(x,146,co,6,ha='center')
for y,target,label in [(140,'AT','Target: AT'),(135,'BE','Target: BE'),(127.5,'SK','Target: SK')]:
    txt(3,y,label,6)
    for x,co in zip(xs,countries):
        ax.add_patch(Rectangle((x-2.1,y-1.35),4.2,2.7,facecolor=target_color if co==target else source_color,edgecolor='none'))
        if co==target:txt(x,y,'T',6,color='white',ha='center')
ax.plot([16]*3,[130.8,131.5,132.2],'.',color=C['MUTED'],ms=1.4)
txt(109,131.5,'… rotate through the remaining countries …',6,ha='center',color=C['MUTED'])
ax.add_patch(Rectangle((80,160.5),2.5,1.8,facecolor=source_color,edgecolor='none'));txt(84,161.4,'10 source countries',6)
ax.add_patch(Rectangle((119,160.5),2.5,1.8,facecolor=target_color,edgecolor='none'));txt(123,161.4,'1 target country',6)
txt(178,161.4,'11 rotations',6,ha='right')
line(125)
# b：地图保持目前图3的点云、测试格和采样点样式，不加演示文稿式卡片。
panel(2,120.5,'c','Two starting conditions for local label collection')
txt(5,114,'A  No local training labels',6.5,color=C['LABEL'])
txt(97,114,'B  An existing clustered survey',6.5,color=C['CONC'])
ax.plot([91,91],[68,115],color=C['GRID'],lw=.5)
parcels(5,85,33,26)
parcels(49,85,33,26,add='random')
arrow(39,98,48,98,C['LABEL'])
txt(21.5,81,'0 local labels',6,ha='center')
txt(66,81,'+ 50 / 100 / 200 / 400',6,ha='center',color=C['LABEL'])
txt(44,74,'Random sampling across the target country',6,ha='center')
parcels(98,96,28,17,existing=True)
txt(131,105,'100 existing labels',6,color=C['CONC'])
txt(131,100,'Retained in every addition',6)
arrow(111,96,111,92,C['CONC']);ax.plot([111,169],[92,92],color=C['CONC'],lw=.6)
for x,kind,title in [(98,'local','Local extension'),(125,'spread','National random'),(152,'betti','Regional allocation')]:
    arrow(x+12,92,x+12,90,C['CONC'])
    parcels(x,70,26,20,existing=True,add=kind)
    txt(x+13,67.5,title,6,ha='center')
txt(145,94,'+ 100 or 300 new labels',6,ha='center',color=C['CONC'])
ax.plot(6,61,'o',color=C['CONC'],ms=3);ax.plot(8,61,'o',color='#D2AFCC',ms=3);txt(11,61,'Existing labels',6)
ax.plot(39,61,'o',color=C['LABEL'],ms=3);txt(42,61,'New labels',6)
ax.add_patch(Rectangle((66,60),2,1.8,facecolor='#BBBBBB',edgecolor='none'));txt(70,61,'Test cells',6)
ax.add_patch(Rectangle((92,60),2,1.8,facecolor='#F1E3EF',edgecolor='none'));txt(96,61,'Surveyed regions',6)
txt(179,57,'Maps: Belgium example; 100 new labels shown per addition',6,ha='right',color=C['MUTED'])
line(54)
# c：用同一组样本的两种概念特征空间与嵌套标签子集表示两条干预轴。
# 淡色点为类别组成示意，黑色轮廓标记已有标签；非实际降维或聚类结果。
panel(2,49.5,'d','Representation × local labels')
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
arrow(31,6,46,6,C['TEXT'],style='-',lw=.65)
arrow(52,6,67,6,C['TEXT'],lw=.65)
txt(49,6,'n',6,ha='center',bbox=dict(facecolor='white',edgecolor='none',pad=1))
# 标签量用空心圈数直接编码，避免文字说明。
for x,num in [(31,3),(67,12)]:
    for i in range(num):
        ax.plot(x+(i-(num-1)/2)*1.0,7.8,'o',ms=1.7,mfc='white',mec='#333333',mew=.45)
ax.plot(21,1.8,'o',ms=2.5,mfc=cropcols[0],mec='#333333',mew=.45)
txt(24,1.8,'Labelled',6)
for i,col in enumerate(cropcols):ax.plot(47+i*2,1.8,'o',ms=2.2,color=col,alpha=.6)
txt(54,1.8,'Classes (schematic)',6)
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
    txt(xp,9,label,6,color=col,ha='center')
mid=left+cw*(xa+xs)/2
arrow(left+cw*xa,6,mid-2.8,6,C['TEXT'],'<-',lw=.65)
arrow(mid+2.8,6,left+cw*xs,6,C['TEXT'],'->',lw=.65)
txt(left+cw*(xa+xs)/2,6,'Δn',6,ha='center',bbox=dict(facecolor='white',edgecolor='none',pad=1))
x=.12;y1=.10+.76*(1-np.exp(-2.4*x));y2=.38+.53*(1-np.exp(-2.4*x))
arrow(left+cw*x,bottom+ch*y1,left+cw*x,bottom+ch*y2,C['TEXT'],'<->',lw=.7)
txt(left+cw*x+2,bottom+ch*(y1+y2)/2+2,'ΔBA',6)
txt(169,bottom+ch*ya[-1]+.8,'AE',6,color=C['AE'])
txt(169,bottom+ch*ys[-1]-.8,'S2',6,color=C['S2'])
txt(94,28,'BA at matched OA',6,ha='center',rotation=90)
txt(left,bottom-2.7,'0',6,ha='center')
txt(168,9,'n',6,ha='center')
txt(178,1.8,'Conceptual feature spaces and curves',6,ha='right',color=C['MUTED'])

# 新增的 a：同一已保存质心与真实地块，配合明确标记的名义像元网格。
sys.path.insert(0,str(HERE.parent/'parcel_reference_20261011'))
import plot_figures as pf
from matplotlib.patches import Circle
from shapely.geometry import box as geom_box
ROOT=HERE.parents[4]
sys.path.insert(0,str(ROOT.parent/'nature-skills-main/skills/nature-figure/scripts'))
from audit_panel_alignment import require_matplotlib_panel_alignment

def ft(x,y,s,size=6,**kw):
    return fig.text(x/W,y/H,s,fontsize=size,ha=kw.pop('ha','left'),va='center',**kw)

ft(2,197,'a',8,fontweight='bold')
ft(8,197,'From ground reference to paired satellite features',7,fontweight='bold')
aa=fig.add_axes([4/W,150/H,39/W,39/H])
bb=fig.add_axes([57/W,150/H,39/W,39/H])
cc=fig.add_axes([111/W,150/H,70/W,39/H])
bounds=pf.EXAMPLE.total_bounds
side=max(bounds[2]-bounds[0],bounds[3]-bounds[1])+80
extent=[pf.X-side/2,pf.Y-side/2,pf.X+side/2,pf.Y+side/2]
pf.map_image(aa,extent)
pf.EXAMPLE.boundary.plot(ax=aa,color='white',linewidth=2)
pf.EXAMPLE.boundary.plot(ax=aa,color=C['LABEL'],linewidth=1.1)
aa.add_patch(Circle((pf.X,pf.Y),30,fill=False,edgecolor='white',lw=.8,linestyle=(0,(3,2)),zorder=6))
aa.scatter([pf.X],[pf.Y],s=13,color='white',edgecolor=C['LABEL'],lw=.7,zorder=7)
pf.map_image(aa,extent)
pf.scale(aa,50,white=True)
bb.set(xlim=(-45,45),ylim=(-45,45));bb.set_aspect('equal');bb.axis('off')
for i in range(-3,4):
    for j in range(-3,4):
        bb.add_patch(Rectangle((i*10-5,j*10-5),10,10,facecolor='#D9E6F2' if i*i+j*j<=9 else '#F7F7F7',edgecolor='#CACACA',lw=.35))
bb.add_patch(Circle((0,0),30,fill=False,edgecolor=C['LABEL'],lw=.8,linestyle=(0,(3,2))))
bb.scatter([0],[0],color=C['LABEL'],s=13,zorder=8)
pf.arrow(bb,(0,0),(21.2,21.2),C['LABEL'])
bb.text(-42,40,'3-pixel radius',fontsize=6,color=C['LABEL'],va='center')
bb.plot([-35,-25],[-40,-40],color=C['TEXT'],lw=.65)
bb.text(-21,-40,'10 m (nominal)',fontsize=6,va='center')
cc.set(xlim=(0,1),ylim=(0,1));cc.axis('off')
for y,name,n,col,label in [(.91,'AE',64,C['AE'],'annual coordinates'),(.47,'S2',84,C['S2'],'temporal summaries')]:
    cc.text(.02,y,name,fontsize=7,color=col,va='center')
    cc.text(.14,y,f'{n} {label}',fontsize=6,va='center')
    for j,stat in enumerate(['Mean','SD']):
        yy=y-.13-j*.10
        cc.text(.19,yy,stat,fontsize=6,ha='right',va='center')
        for k in [0,1,2,4,5]:
            cc.add_patch(Rectangle((.24+k*.075,yy-.028),.064,.056,facecolor='none',edgecolor=col,lw=.5))
        cc.text(.24+3*.075+.032,yy,'…',fontsize=6,ha='center',va='center',color=col)
    cc.text(.76,y-.18,f'= {n*2}',fontsize=7,color=col,va='center')
cc.text(.02,.07,'Same parcel identity and crop label',fontsize=6,color=C['LABEL'])
ft(4,192,'Parcel and stored centroid')
ft(57,192,'Circular neighbourhood')
ft(111,192,'Per-parcel feature vectors')
ft(4,147,'P1: Winter cereals',color=C['LABEL'])
ft(57,147,'Schematic grid; nominal 30-m radius',color=C['MUTED'])
for left,right in [(45,54),(98,108)]:
    fig.add_artist(FancyArrowPatch((left/W,170/H),(right/W,170/H),transform=fig.transFigure,
                    arrowstyle='-|>',mutation_scale=7,lw=.7,color=C['MUTED']))
fig.add_artist(plt.Line2D([2/W,180/W],[144/H,144/H],transform=fig.transFigure,color=C['GRID'],lw=.6))

OUT=HERE/'out'
fig.canvas.draw()
# 所有最终输出文字至少 6 pt，在当前 160-mm 稿件中仍超过 5 pt。
for t in fig.findobj(Text):
    if t.get_text().strip() and t.get_visible() and t.get_fontsize()<6:
        t.set_fontsize(6)
fig.canvas.draw()
require_matplotlib_panel_alignment(fig,axes=[aa,bb,cc],panel_ids=['parcel','kernel','features'],
    row_groups=[['parcel','kernel'],['kernel','features']],strict=True,
    json_out=OUT/'fig_concept.alignment.json',overlay_svg=OUT/'fig_concept.alignment.svg')
fig.savefig(OUT/'fig_concept.pdf',facecolor='white')
fig.savefig(OUT/'fig_concept.svg',facecolor='white')
fig.savefig(OUT/'fig_concept.png',dpi=300,facecolor='white')
fig.savefig(OUT/'fig_concept.tiff',dpi=600,facecolor='white',pil_kwargs={'compression':'tiff_lzw'})
plt.close(fig)
print('Integrated concept figure exported.')
