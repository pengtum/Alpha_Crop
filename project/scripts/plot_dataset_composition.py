"""Figure 1 — what the benchmark actually contains.

Core conclusion defended by this figure:
    The benchmark is balanced by country but sharply imbalanced by class and
    unevenly covered — every country contributes ~50,000 parcels, yet class
    abundance spans 370-fold, only two of ten classes occur in all eleven
    countries, and median parcel size differs 16-fold across countries. Every
    border therefore changes the shape of the label space, not just its
    spectral appearance.

Panel map
    a (hero)  country x class parcel-count matrix, log colour, absent cells
              hatched, with country totals on top and class totals at right.
    b         within-country class composition (100%), annotated with the
              effective number of classes exp(H) — this is the prior shift a
              transferred model meets, which panel a's absolute counts hide.
    c         parcel-area distribution per country (log) — a covariate that
              differs 16-fold and is not derivable from a or b.
    d         class coverage: in how many countries a class clears 1 / 10 / 100
              parcels — explains the 'absent' and 'n<10' cells in the
              leave-one-country-out maps.

Source data: results/dataset_summary/source_*.csv
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from protocol_labels import stamp, stamp_multi, NAME
import os
import numpy as np, pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import Patch, Rectangle
from matplotlib.cm import ScalarMappable

mpl.rcParams.update({
    'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],
    'svg.fonttype':'none','pdf.fonttype':42,'ps.fonttype':42,'font.size':7,
    'axes.spines.right':False,'axes.spines.top':False,'axes.linewidth':0.7,
    'xtick.major.width':0.7,'ytick.major.width':0.7,
    'xtick.major.size':2.4,'ytick.major.size':2.4,'legend.frameon':False,
})

CTR=['AT','BE','DE','EE','LT','LV','PT','RO','SE','SI','SK']
LBL={'grassland':'Grassland','other_crops':'Other crops','winter_cereals':'Winter cereals',
     'maize':'Maize','tree_wood_forest':'Tree / wood / forest','orchards':'Orchards',
     'vegetables':'Vegetables','potatoes':'Potatoes','root_crops':'Root crops',
     'strawberries':'Strawberries'}
COL={'grassland':'#8BCF8B','other_crops':'#CFCECE','winter_cereals':'#E3B23C',
     'maize':'#E07B39','tree_wood_forest':'#2E6B45','orchards':'#9A4D8E',
     'vegetables':'#42949E','potatoes':'#B08968','root_crops':'#3775BA',
     'strawberries':'#B64342'}
INK='#272727'; MID='#767676'; LITE='#CFCECE'; BLUE='#0F4D92'

ae=pd.read_parquet('results/dataset_summary/ae_meta.parquet')
s2=pd.read_parquet('results/dataset_summary/s2_meta.parquet')
ORD=ae.label_tier_b.value_counts().index.tolist()          # abundant -> rare
M=(ae.pivot_table(index='label_tier_b',columns='country',values='parcel_id',aggfunc='count')
     .reindex(index=ORD,columns=CTR).fillna(0).astype(int))
CT=M.sum(axis=0); CL=M.sum(axis=1)

os.makedirs('results/dataset_summary',exist_ok=True)
M.to_csv('results/dataset_summary/source_country_x_class_counts.csv')

def k(v):
    if v==0: return ''
    if v>=10000: return f'{v/1000:.0f}k'
    if v>=1000:  return f'{v/1000:.1f}k'
    return f'{v:,}'

fig=plt.figure(figsize=(7.20,7.95))
outer=GridSpec(2,1,height_ratios=[1.00,0.70],hspace=0.50,figure=fig,
               left=0.075,right=0.983,top=0.955,bottom=0.118)
top=GridSpecFromSubplotSpec(1,2,subplot_spec=outer[0],width_ratios=[1.46,1.0],wspace=0.42)
bot=GridSpecFromSubplotSpec(1,2,subplot_spec=outer[1],width_ratios=[1.0,0.92],wspace=0.36)

# ---------------------------------------------------------------- a  hero
ga=GridSpecFromSubplotSpec(2,2,subplot_spec=top[0],height_ratios=[0.155,1.0],
                           width_ratios=[1.0,0.155],hspace=0.055,wspace=0.045)
axT=fig.add_subplot(ga[0,0]); axH=fig.add_subplot(ga[1,0]); axR=fig.add_subplot(ga[1,1])

Mm=np.ma.masked_where(M.values==0,M.values)
cmap=plt.cm.YlGnBu.copy(); cmap.set_bad('white')
im=axH.imshow(Mm,cmap=cmap,norm=LogNorm(vmin=1,vmax=M.values.max()),aspect='auto')
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        v=M.values[i,j]
        if v==0:
            axH.add_patch(Rectangle((j-.5,i-.5),1,1,facecolor='white',
                                    edgecolor='#B0B0B0',hatch='////',lw=0.5,zorder=2))
        else:
            lum=np.log10(max(v,1))/np.log10(M.values.max())
            axH.text(j,i,k(v),ha='center',va='center',fontsize=5.0,zorder=3,
                     color='white' if lum>0.62 else INK)
axH.set_xticks(range(11)); axH.set_xticklabels(CTR,fontsize=6.6)
axH.set_yticks(range(len(ORD))); axH.set_yticklabels([LBL[c] for c in ORD],fontsize=6.6)
axH.set_xticks(np.arange(-.5,11,1),minor=True); axH.set_yticks(np.arange(-.5,len(ORD),1),minor=True)
axH.grid(which='minor',color='white',lw=0.8); axH.tick_params(which='minor',length=0)
for s in axH.spines.values(): s.set_visible(False)
axH.set_xlabel('Country',fontsize=7,labelpad=1.5)

axT.bar(range(11),CT.values/1000,color=LITE,edgecolor=MID,lw=0.5,width=0.78)
for j,v in enumerate(CT.values): axT.text(j,v/1000+3,f'{v/1000:.0f}',ha='center',fontsize=5.2,color=MID)
axT.set_xlim(-.5,10.5); axT.set_ylim(0,128); axT.set_xticks([]); axT.set_yticks([0,50,100])
axT.tick_params(labelsize=5.6); axT.set_ylabel('k parcels',fontsize=5.9,labelpad=1)
axT.spines['left'].set_visible(True)
axT.set_title('a   Parcels per country and class — balanced by country, not by class',
              loc='left',fontsize=8,fontweight='bold',pad=3.5)

axR.barh(range(len(ORD)),CL.values,color=[COL[c] for c in ORD],edgecolor='none',height=0.78)
axR.set_xscale('log'); axR.set_xlim(500,7e5); axR.invert_yaxis()
axR.set_ylim(len(ORD)-.5,-.5); axR.set_yticks([]); axR.set_xticks([1e3,1e5])
axR.set_xticklabels(['10$^3$','10$^5$'],fontsize=5.6)
axR.tick_params(axis='x',pad=1); axR.set_xlabel('total',fontsize=5.9,labelpad=1)
for i,v in enumerate(CL.values):
    axR.text(v*1.35,i,k(v),va='center',fontsize=5.2,color=INK)

cax=fig.add_axes([0.078,0.462,0.112,0.010])
cb=fig.colorbar(im,cax=cax,orientation='horizontal')
cb.set_label('parcels per cell (log)',fontsize=5.8,labelpad=1.5); cb.ax.tick_params(labelsize=5.4)
fig.legend(handles=[Patch(facecolor='white',edgecolor='#B0B0B0',hatch='////',lw=0.5,
                          label='class absent from that country')],
           loc='lower left',bbox_to_anchor=(0.212,0.4575),fontsize=6.0)

# ---------------------------------------------------------------- b  composition
axb=fig.add_subplot(top[1])
frac=(M/M.sum(axis=0)).T[ORD]
left=np.zeros(11)
for c in ORD:
    axb.barh(range(11),frac[c].values*100,left=left,color=COL[c],edgecolor='white',
             lw=0.35,height=0.78,label=LBL[c])
    left+=frac[c].values*100
p=frac.values.copy(); p[p<=0]=np.nan
H=np.nansum(-p*np.log(p),axis=1); eff=np.exp(H)
for i,e in enumerate(eff): axb.text(101.5,i,f'{e:.1f}',va='center',fontsize=5.8,color=MID)
axb.text(101.5,-0.72,'exp($H$)',fontsize=5.8,color=MID,ha='left')
axb.set_yticks(range(11)); axb.set_yticklabels(CTR,fontsize=6.6); axb.invert_yaxis()
axb.set_xlim(0,100); axb.set_xlabel('Share of the country’s parcels (%)',fontsize=7,labelpad=1.5)
axb.spines['left'].set_visible(False); axb.tick_params(axis='y',length=0)
axb.set_title('b   Class composition differs at every border',
              loc='left',fontsize=8,fontweight='bold',pad=25)
axb.text(0,1.015,'Colours as in the class-total bars of a. exp($H$): effective number of classes — Romania reaches 2.5\n'
         'with only 3 of the 10 classes present, Austria 4.5 with all 10.',
         transform=axb.transAxes,fontsize=5.9,color=MID,style='italic',linespacing=1.4)

# ---------------------------------------------------------------- c  parcel area
axc=fig.add_subplot(bot[0])
med=ae.groupby('country').area_ha.median().sort_values()
order=med.index.tolist()
data=[ae.loc[ae.country==c,'area_ha'].values for c in order]
bp=axc.boxplot(data,vert=False,showfliers=False,widths=0.62,patch_artist=True,
               medianprops=dict(color=INK,lw=1.0),whiskerprops=dict(color=MID,lw=0.6),
               capprops=dict(color=MID,lw=0.6),boxprops=dict(lw=0.5,edgecolor=MID))
for i,b in enumerate(bp['boxes']):
    b.set_facecolor(BLUE if order[i] in ('PT','RO') else LITE)
    b.set_alpha(1.0 if order[i] in ('PT','RO') else 0.9)
axc.set_xscale('log'); axc.set_yticks(range(1,12)); axc.set_yticklabels(order,fontsize=6.6)
axc.set_xlabel('Parcel area (ha, log scale)',fontsize=7,labelpad=1.5)
axc.tick_params(axis='y',length=0); axc.spines['left'].set_visible(False)
axc.set_xlim(0.02,300)
for i,c in enumerate(order):
    axc.text(215,i+1,f'{med[c]:.2f}',va='center',ha='right',fontsize=5.6,color=MID)
axc.text(215,11.72,'median (ha)',ha='right',fontsize=5.8,color=MID)
axc.annotate('',xy=(0.229,0.42),xytext=(3.806,0.42),
             arrowprops=dict(arrowstyle='<|-|>',color=BLUE,lw=0.7,
                             shrinkA=0,shrinkB=0,mutation_scale=7))
axc.text(0.93,0.62,'16$\\times$',fontsize=6.4,color=BLUE,ha='center',va='bottom',fontweight='bold')
axc.set_ylim(0.2,12.1)
axc.set_title('c   Parcel size differs 16-fold across countries',
              loc='left',fontsize=8,fontweight='bold',pad=25)
axc.text(0.0,1.015,'Countries ordered by median. Box: IQR; whiskers 1.5$\\times$IQR; outliers omitted.\n'
         'Parcel size is itself a feature (log area) in every model reported here.',
         transform=axc.transAxes,fontsize=5.9,color=MID,style='italic',linespacing=1.4)

# ---------------------------------------------------------------- d  coverage
axd=fig.add_subplot(bot[1])
thr=[(1,'≥ 1 parcel',LITE,4.2),(10,'≥ 10 parcels','#7FA8D0',3.4),(100,'≥ 100 parcels',BLUE,2.4)]
y=np.arange(len(ORD))
for t,lab,c,ms in thr:
    n=(M>=t).sum(axis=1).reindex(ORD).values
    axd.plot(n,y,'o',ms=ms*1.55,color=c,mec='white',mew=0.4,label=lab,zorder=3-thr.index((t,lab,c,ms)))
for i,cl in enumerate(ORD):
    n1=(M.loc[cl]>=1).sum(); n3=(M.loc[cl]>=100).sum()
    if n1>n3: axd.plot([n3,n1],[i,i],color=LITE,lw=1.0,zorder=1)
axd.axvline(11,color=MID,lw=0.6,ls=(0,(2,2)))
axd.text(10.85,9.72,'all 11',fontsize=5.9,color=MID,ha='right')
axd.set_yticks(y); axd.set_yticklabels([LBL[c] for c in ORD],fontsize=6.6); axd.invert_yaxis()
axd.set_xlim(0,12.6); axd.set_xticks([0,3,6,9,11]); axd.tick_params(axis='y',length=0)
axd.spines['left'].set_visible(False)
axd.set_xlabel('Number of countries in which the class occurs',fontsize=7,labelpad=1.5)
axd.set_title('d   Only two classes occur everywhere',loc='left',fontsize=8,fontweight='bold',pad=3.5)
axd.legend(loc='upper left',fontsize=6.0,handletextpad=0.3,borderaxespad=0.3,
           labelspacing=0.32)
axd.grid(axis='x',color='#EDEDED',lw=0.6,zorder=0); axd.set_axisbelow(True)

# ---------------------------------------------------------------- footer
fig.text(0.075,0.010,
    'EuroCrops 2021, eleven countries with strictly year-aligned declarations, Tier-B taxonomy (10 classes). '
    'AlphaEarth feature set: 644,949 parcels; Sentinel-2 feature set: 628,123; the 623,197 parcels present in both\n'
    'are used wherever the two representations are compared parcel-by-parcel. Country totals are capped at ~50,000 parcels by design (Belgium 94,995 and Germany 99,988 retain\n'
    'larger samples). Class abundance spans 370-fold, from 309,509 grassland parcels to 836 strawberry parcels. Source data: results/dataset_summary/.',
    fontsize=5.9,color=MID,style='italic',va='bottom')

stamp(fig,'dataset',x=0.075,y=0.068,fontsize=5.9)
for ext in ['pdf','png','svg']:
    fig.savefig(f'Manuscript/Paper_Alpha_Earth/0001_draft_v1/figures/fig_dataset.{ext}',
                dpi=600,bbox_inches='tight')
fig.savefig('SLIDES_FIGURES/S00_dataset_composition.pdf',bbox_inches='tight')
fig.savefig('SLIDES_FIGURES/S00_dataset_composition.png',dpi=300,bbox_inches='tight')
plt.close(fig)
print('saved fig_dataset / S00_dataset_composition')
