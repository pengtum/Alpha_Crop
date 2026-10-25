"""Coordinate-ablation decomposition figure (11-country set, 168 LightGBM fits).
Values from results/experiments/exp7_uncertainty_ablation/RESULTS.md."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from protocol_labels import stamp, stamp_multi, NAME
import numpy as np, matplotlib, os
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42,'font.family':'Arial','font.size':8,
    'axes.spines.right':False,'axes.spines.top':False,'axes.linewidth':0.7,
    'savefig.bbox':'tight','savefig.pad_inches':0.06,'figure.dpi':200})
OUT=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'Manuscript','Paper_Alpha_Earth','0001_draft_v1','figures')
# ID / OOD balanced accuracy (%) per feature set and arm
D={'ae':    {'pure':(63.99,41.27),'coords':(65.23,40.82),'area':(65.44,41.83),'both':(66.64,41.12)},
   's2':    {'pure':(61.31,39.74),'coords':(64.40,38.23),'area':(62.79,39.21),'both':(65.72,38.08)},
   'fusion':{'pure':(65.31,41.87),'coords':(66.29,41.66),'area':(66.60,42.86),'both':(67.51,41.49)}}
FT=['ae','s2','fusion']; FN={'ae':'AlphaEarth','s2':'Sentinel-2','fusion':'Fusion'}
FC={'ae':'#3B6CA8','s2':'#E08A3C','fusion':'#4F9D5B'}
AC={'coords':'#B0483B','area':'#3F7F5F','both':'#8A8A8A'}
AN={'coords':'lon, lat','area':'log(parcel area)','both':'both'}
MK={'coords':'o','area':'s','both':'^'}
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(7.6,3.1),gridspec_kw={'width_ratios':[1.12,1]})
x=np.arange(3); w=0.26
for i,arm in enumerate(['coords','area','both']):
    v=[D[f][arm][1]-D[f]['pure'][1] for f in FT]
    ax1.bar(x+(i-1)*w,v,w,label=AN[arm],color=AC[arm])
ax1.axhline(0,color='#333',lw=0.7); ax1.set_xticks(x)
ax1.set_xticklabels([FN[f] for f in FT],fontsize=7.5)
ax1.set_ylabel('Change in cross-border accuracy (pp)')
ax1.legend(frameon=False,fontsize=7,loc='lower left')
ax1.set_title('a  Out-of-distribution effect',loc='left',fontsize=8.5,fontweight='bold')
for f in FT:
    for arm in ['coords','area','both']:
        di=D[f][arm][0]-D[f]['pure'][0]; do=D[f][arm][1]-D[f]['pure'][1]
        ax2.scatter(di,do,marker=MK[arm],s=48,color=FC[f],edgecolor='white',linewidth=0.6,zorder=3)
ax2.axhline(0,color='#333',lw=0.7); ax2.axvline(0,color='#333',lw=0.7)
ax2.set_xlabel('Change in in-distribution accuracy (pp)')
ax2.set_ylabel('Change in cross-border accuracy (pp)')
ax2.text(0.97,0.06,'helps in distribution,\nhurts transfer',transform=ax2.transAxes,ha='right',
         fontsize=6.6,color='#B0483B',style='italic',linespacing=1.3)
ax2.text(0.97,0.94,'helps both',transform=ax2.transAxes,ha='right',va='top',
         fontsize=6.6,color='#3F7F5F',style='italic')
h=[Line2D([],[],marker=MK[a],color='#555',ls='',ms=6,label=AN[a]) for a in ['coords','area','both']]+\
  [Line2D([],[],marker='o',color=FC[f],ls='',ms=6,label=FN[f]) for f in FT]
ax2.legend(handles=h,frameon=False,fontsize=6.3,loc='lower left',ncol=2,
           handletextpad=0.3,columnspacing=0.7)
ax2.set_title('b  Attribution',loc='left',fontsize=8.5,fontweight='bold')
for ext in ['pdf','png']:
    stamp_multi(fig,['pooled_cv','loco'],x=0.0,y=0.0,fontsize=7.4,panel_letters=False,
        extra='A matched pair from the same training-breadth row of the protocol 2×2, so the ID/OOD contrast is not confounded with training-set size.')
    fig.savefig(os.path.join(OUT,f'fig_ablation_decomposition.{ext}'))
print('saved fig_ablation_decomposition.{pdf,png} ->',OUT)
