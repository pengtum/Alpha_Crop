"""Study-overview schematic: one research question, two families of experiments,
and the operational consequence that follows. Academic register, no theatrical labels."""
import os, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42,'font.family':'Arial',
    'savefig.bbox':'tight','savefig.pad_inches':0.12,'figure.dpi':180})
OUT=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'Manuscript','Paper_Alpha_Earth','0001_draft_v1','figures')
BLUE,GREEN,GREY,RQC='#1F6FB4','#2E7D32','#5A5A5A','#1F3A93'
def box(ax,x,y,w,h,fc,ec,lw=1.5,r=0.7):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0.02,rounding_size={r}',
        facecolor=fc,edgecolor=ec,linewidth=lw))
def arr(ax,x1,y1,x2,y2,c=GREY,lw=1.7,ms=14):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',color=c,linewidth=lw,mutation_scale=ms))
fig,ax=plt.subplots(figsize=(13,6.2)); ax.set_xlim(0,100); ax.set_ylim(0,64)
ax.set_aspect('equal'); ax.axis('off')
box(ax,3,53,94,8.5,'#EAF0F8',RQC,lw=1.6)
ax.text(50,59.4,'Research question',ha='center',va='center',fontsize=10,color=RQC,fontweight='bold',style='italic')
ax.text(50,55.8,"Does AlphaEarth's frozen embedding encode transferable crop semantics, or a geographic\n"
    "confound that helps in-distribution accuracy but limits cross-border transfer?",
    ha='center',va='center',fontsize=10.2,color='#1A1A1A',linespacing=1.4)
CY,CH=24,24
cards=[dict(x=6,w=39,c=BLUE,fc='#E7F1FB',title='Cross-distribution classification',
   sub='pooled cross-validation  →  leave-one-country-out\n→  pairwise country-to-country (11×11)',
   find="AlphaEarth's +0.8 pp in-distribution edge does not\n"
        "survive transfer (42.7 vs 43.2%; Wilcoxon p = 0.41).\n"
        "The pairwise matrix maps hubs and asymmetry."),
 dict(x=55,w=39,c=GREEN,fc='#E9F5EA',title='Representation diagnostics',
   sub='CKA  ·  country-predictability probe  ·  UMAP\n·  coordinate-ablation intervention',
   find="Organised by geography, not crop: country alignment\n"
        "1.5× Sentinel-2's (crop alignment equal); country\n"
        "recoverable from features alone at 94.4% vs 88.8%.")]
for c in cards:
    cx=c['x']+c['w']/2
    box(ax,c['x'],CY,c['w'],CH,c['fc'],c['c'],lw=1.6)
    ax.text(cx,CY+CH-2.0,c['title'],ha='center',va='top',fontsize=11.5,color=c['c'],fontweight='bold')
    ax.text(cx,CY+CH-6.4,c['sub'],ha='center',va='top',fontsize=8.4,color='#444',linespacing=1.5)
    ax.text(cx,CY+1.2,c['find'],ha='center',va='bottom',fontsize=8.6,color='#1A1A1A',linespacing=1.5)
    arr(ax,cx,53,cx,CY+CH+0.4,c='#9AA0A6',lw=1.1,ms=11)
arr(ax,45.4,CY+CH/2,54.4,CY+CH/2,c=GREY,lw=1.9)
ax.text(49.9,CY+CH/2+1.8,'why?',ha='center',fontsize=8.6,color=GREY,style='italic')
box(ax,3,4,94,15,'#FBFAF7','#B4AB99',lw=1.2,r=0.5)
ax.text(50,16.3,'Operational consequence',ha='center',va='center',fontsize=9.8,color='#5A5040',fontweight='bold')
ax.text(50,10.0,
 "A frozen encoder cannot be debiased, and the geographic signal is entangled with genuine biophysical variation, so the only lever is\n"
 "data-side. Decomposing the intervention localises the penalty to explicit location: coordinates raise in-distribution accuracy while\n"
 "lowering cross-border accuracy, whereas parcel geometry raises both. Complementary spectral features carry cross-border signal the\n"
 "embedding does not — how far that can be pushed under an annotation budget is developed in a companion paper.",
 ha='center',va='center',fontsize=8.6,color='#3A3226',linespacing=1.5)
for ext in ['pdf','png']:
    fig.savefig(os.path.join(OUT,f'fig_narrative_arc.{ext}'))
print('saved fig_narrative_arc')
