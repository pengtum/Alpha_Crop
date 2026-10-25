"""将真实地块示例与总体覆盖、类别支持合并；仅重排原有源数据。"""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap
from matplotlib.patches import Patch
from shapely.geometry import box
from shapely.ops import unary_union

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / 'Manuscript/Paper_Alpha_Earth/0004_representation_labels_rse/figure_sources/parcel_reference_20261011'))
import plot_figures as pf
sys.path.insert(0, str(ROOT.parent / 'nature-skills-main/skills/nature-figure/scripts'))
from audit_panel_alignment import require_matplotlib_panel_alignment

C = pf.C
OUT = HERE / 'out'
SRC = ROOT / 'label_expansion/storyline/rse_figs/revision_20261007/source_data'
CC = ['AT','BE','DE','EE','LT','LV','PT','RO','SE','SI','SK']
CL = ['grassland','maize','orchards','other_crops','potatoes','root_crops','strawberries','tree_wood_forest','vegetables','winter_cereals']
LABELS = ['Grassland','Maize','Orchards','Other crops','Potatoes','Root crops','Strawberries','Tree/wood/forest','Vegetables','Winter cereals']
BLUE = LinearSegmentedColormap.from_list('rse_blue', ['#F4F7FA', '#A7C4DD', C['LABEL']])
width_mm = 183
height_mm = 158
plt.rcParams.update({'font.size':8,'xtick.labelsize':7.5,'ytick.labelsize':7.5})
plt.rcParams.update({'font.family':'Arial','svg.fonttype':'none','pdf.fonttype':42})
fig = plt.figure(figsize=(width_mm/25.4, height_mm/25.4))

def axes(x,y,w,h):
    return fig.add_axes([x/width_mm,y/height_mm,w/width_mm,h/height_mm])

def txt(x,y,s,**kw):
    return fig.text(x/width_mm,y/height_mm,s,fontsize=kw.pop('fontsize',7.5),
                    ha=kw.pop('ha','left'),va='center',**kw)

def title(ax, letter, label):
    ax.annotate(letter, (0, 1), xycoords='axes fraction', xytext=(0, 5),
                textcoords='offset points', fontsize=9.0, fontweight='bold', va='bottom')
    ax.annotate(label, (0, 1), xycoords='axes fraction', xytext=(15, 5),
                textcoords='offset points', fontsize=8.0, va='bottom')

# a：全体样本，严格保留 10-km 聚合数值与原来的对数配色。
a=axes(4,96,54,54)
geo=gpd.read_file(ROOT/'label_expansion/storyline/rse_figs/source_data/countries.geojson').to_crs(3035)
density=pd.read_csv(SRC/'spatial_density.csv')
assert density.n.sum()==611519
map_ymin, map_ymax = density.y.min()-140000, density.y.max()+140000
map_side = map_ymax-map_ymin
map_xcentre = (density.x.min()+density.x.max())/2
# 只扩展背景范围，使地图与右侧两个面板等宽等高；投影纵横比及全部数据不变。
map_extent=[map_xcentre-map_side/2,map_ymin,map_xcentre+map_side/2,map_ymax]
geo.geometry=geo.geometry.intersection(box(*map_extent))
geo.plot(ax=a,facecolor='#EEEEEE',edgecolor='none',zorder=0)
im=a.scatter(density.x,density.y,c=density.n,s=1,marker='s',cmap=BLUE,
             norm=LogNorm(1,density.n.max()),linewidths=0,rasterized=True)
a.set_xlim(map_extent[0],map_extent[2])
a.set_ylim(map_extent[1],map_extent[3])
a.set_aspect('equal');a.axis('off')
offsets={'BE':(-140000,-30000),'DE':(100000,70000),'AT':(-60000,60000),'SI':(-140000,-60000),
         'EE':(190000,50000),'LV':(190000,0),'LT':(190000,-30000),'PT':(-170000,0)}
country_texts=[]
for r in pd.read_csv(SRC/'country_label_positions.csv').itertuples():
    dx,dy=offsets.get(r.country,(0,0))
    country_texts.append(a.text(r.x+dx,r.y+dy,r.country,fontsize=7.5,ha='center',color=C['TEXT'],
           bbox=dict(facecolor='white',edgecolor='none',pad=.3)))
# 字号确定后，将本来被标签白底遮住的背景边界实质裁除。
fig.canvas.draw()
renderer=fig.canvas.get_renderer()
mask=unary_union([box(*t.get_window_extent(renderer).expanded(1.25,1.3).transformed(a.transData.inverted()).extents) for t in country_texts])
gpd.GeoSeries(geo.boundary.difference(mask),crs=geo.crs).plot(ax=a,color='white',linewidth=.3,zorder=.5)
title(a,'a','Geographic coverage')
cb=fig.colorbar(im,cax=axes(7,90,48,1.6),orientation='horizontal')
cb.set_ticks([1,10,100,1000],labels=['1','10','100','1,000'])
cb.set_label('Parcels per 10-km cell',labelpad=1,fontsize=7.5)

# b/c：同一真实范围内的全部原始地块；先裁切几何，避免 PDF 中隐藏笔画干扰审计。
b,c=[axes(x,96,54,54) for x in [65,126]]
shown=pf.PARCELS.copy()
shown.geometry=shown.geometry.intersection(box(*pf.EXTENT))
pf.map_image(b,pf.EXTENT)
shown.boundary.plot(ax=b,color='white',linewidth=.3,alpha=.8)
pf.EXAMPLE.boundary.plot(ax=b,color='white',linewidth=2)
pf.EXAMPLE.boundary.plot(ax=b,color=C['LABEL'],linewidth=1.1)
b.text(pf.X,pf.Y,'P1',color='white',fontweight='bold',fontsize=8.0,ha='center',va='center')
pf.map_image(b,pf.EXTENT)
title(b,'b','Recorded fields')
shown.plot(ax=c,color=[pf.COLORS.get(v,'#DDDDDD') for v in shown.analysis_class],edgecolor='white',linewidth=.4)
pf.EXAMPLE.boundary.plot(ax=c,color=C['LABEL'],linewidth=1.1)
pf.map_image(c,pf.EXTENT,aerial=False)
c.text(pf.X,pf.Y,'P1',fontweight='bold',fontsize=8.0,ha='center',va='center')
title(c,'c','Crop labels')
for x in [68,129]:
    length=100/(pf.EXTENT[2]-pf.EXTENT[0])*54
    fig.add_artist(plt.Line2D([x/width_mm,(x+length)/width_mm],[93/height_mm]*2,
                   transform=fig.transFigure,color=C['TEXT'],lw=1))
    txt(x+length+2,93,'100 m')
classes=[k for k in pf.COLORS if k in set(pf.PARCELS.analysis_class)]
handles=[Patch(facecolor=pf.COLORS[k],label=k.replace('_',' ').capitalize()) for k in classes]
fig.legend(handles=handles,loc='center',bbox_to_anchor=(122.5/width_mm,87.5/height_mm),
           ncol=3,fontsize=7.5,handlelength=.9,handletextpad=.4,columnspacing=1,labelspacing=.6)
txt(65,80.5,'P1  |  2.17 ha  |  Tielt area, Belgium, 2021',color=C['LABEL'])
txt(65,76.5,'Declared / harmonised: Winter wheat  →  Analysis: Winter cereals')

# d/e：转置国家比例矩阵，使相同作物行与总体数量一一对应。
shares=pd.read_csv(SRC/'class_shares.csv').set_index('country').loc[CC,CL]
counts=pd.read_csv(SRC/'class_counts.csv').set_index('country').loc[CC,CL].sum()
assert int(counts.sum())==611519
d=axes(29,20,100,48)
z=shares.to_numpy().T
cm=BLUE.copy();cm.set_bad('#E3E3E3')
im=d.imshow(np.ma.masked_equal(z,0),cmap=cm,vmin=0,vmax=80,aspect='auto',interpolation='nearest')
d.set_xticks(range(11),CC);d.set_yticks(range(10),LABELS)
d.tick_params(length=0,pad=2)
for sp in d.spines.values():sp.set_visible(False)
for i in range(10):
    for j in range(11):
        val=z[i,j]
        s='0' if val==0 else ('<0.1' if val<.1 else f'{val:.1f}')
        rgba=im.cmap(im.norm(val))
        lum=.2126*rgba[0]+.7152*rgba[1]+.0722*rgba[2]
        d.text(j,i,s,ha='center',va='center',fontsize=7.5,
               color='white' if val>0 and lum<.52 else C['TEXT'])
title(d,'d','Class composition by country (%)')
e=axes(139,20,41,48)
e.barh(np.arange(10),counts,color=C['LABEL'],height=.55)
e.set_xscale('log');e.set_xlim(100,1e7);e.set_ylim(9.5,-.5)
e.set_xticks([1e2,1e4,1e6],labels=['100','10,000','1,000,000'])
e.set_yticks([]);e.spines['left'].set_visible(False)
e.set_xlabel('Parcels (log scale)',fontsize=7.5,labelpad=2)
for i,n in enumerate(counts):
    e.text(n*1.18,i,f'{n:,}',fontsize=7.5,ha='left',va='center')
title(e,'e','Pooled counts')
cb=fig.colorbar(im,cax=axes(36,8,72,1.5),orientation='horizontal')
cb.set_ticks([0,20,40,60,80]);cb.set_label('Share within country (%)',fontsize=7.5,labelpad=1)
txt(116,6,'Grey cells: no retained labels',fontsize=7.5)

fig.canvas.draw()
require_matplotlib_panel_alignment(fig,axes=[a,b,c,d,e],panel_ids=['a','b','c','d','e'],
    row_groups=[['a','b','c'],['d','e']],require_panel_labels=True,strict=True,
    json_out=OUT/'fig_dataset_imbalance.alignment.json',overlay_svg=OUT/'fig_dataset_imbalance.alignment.svg')
fig.savefig(OUT/'fig_dataset_imbalance.pdf',facecolor='white')
fig.savefig(OUT/'fig_dataset_imbalance.svg',facecolor='white')
fig.savefig(OUT/'fig_dataset_imbalance.png',dpi=300,facecolor='white')
fig.savefig(OUT/'fig_dataset_imbalance.tiff',dpi=600,facecolor='white',pil_kwargs={'compression':'tiff_lzw'})
plt.close(fig)
print('Integrated dataset figure exported.')
