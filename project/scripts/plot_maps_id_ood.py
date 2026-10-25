"""Four spatial views of transferability, on OSM boundaries, one figure per feature set.

Every panel states the quantity it maps and how it is computed, because three of
the four are derived quantities:

  a  Within country      5-fold cross-validation inside each country: train and
                         test are both that country's parcels.
  b  Cross-border        Leave-one-country-out: train on the other ten countries,
                         test on this one, which never appears in training.
  c  Cost of the border  b - a, signed. Negative means accuracy is lost. Sweden
                         under Sentinel-2 is positive, so this must stay signed.
  d  One neighbour vs    (best single source country) - b. Positive means one
     all ten             well-chosen neighbour beats pooling all ten.

Panels a and b share one colour scale so the drop between them reads as colour,
not only as numbers.
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from protocol_labels import stamp, stamp_multi, NAME
import json, numpy as np, geopandas as gpd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.cm import ScalarMappable
plt.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42,'font.family':'Arial','font.size':9,
    'savefig.bbox':'tight','savefig.pad_inches':0.08,'figure.dpi':215})
FIG='Manuscript/Paper_Alpha_Earth/0001_draft_v1/figures'; SL='SLIDES_FIGURES'
G=gpd.read_file('data/boundaries/osm_countries.gpkg')
ISO3={'AT':'AUT','BE':'BEL','DE':'DEU','EE':'EST','LT':'LTU','LV':'LVA',
      'PT':'PRT','RO':'ROU','SE':'SWE','SI':'SVN','SK':'SVK'}
C=list(ISO3)
LC={'AT':(14.3,47.6),'BE':(4.6,50.6),'DE':(10.4,51.2),'EE':(25.6,58.7),'LT':(23.9,55.3),
    'LV':(24.8,56.9),'PT':(-8.2,39.6),'RO':(25.0,45.9),'SE':(15.6,62.5),'SI':(14.9,46.1),'SK':(19.6,48.7)}
geo=lambda i: G[G.ISO3==ISO3[i]]
idp=json.load(open('results/experiments/exp0_per_country_cv/tier_b_nocoords_2021.json'))
loco=json.load(open('results/experiments/exp2_loco/tier_b_nocoords_2021.json'))
e2b=json.load(open('results/experiments/exp2b_country_transfer/tier_b_nocoords_2021.json'))
FEATS=[('ae','AlphaEarth'),('s2','Sentinel-2'),('fusion','Fusion (AlphaEarth + Sentinel-2)')]
GREY='#555'
X0=[0.020,0.265,0.510,0.755]; W=0.225; YB=0.135; H=0.735


def panel(fig,i,tag,title,sub,vals,cmap,norm,ann,cblabel,cbticks=None,fs=6.6):
    ax=fig.add_axes([X0[i],YB,W,H])
    G.plot(ax=ax,color='#F2F2F0',edgecolor='#D8D8D5',linewidth=0.3)
    for c in C:
        v=vals.get(c)
        if v is None: continue
        geo(c).plot(ax=ax,color=cmap(norm(v)),edgecolor='white',linewidth=0.55)
        x,y=LC[c]
        ax.annotate(f'{c}\n{ann[c]}',(x,y),ha='center',va='center',fontsize=fs,color='black',
                    path_effects=[pe.withStroke(linewidth=1.7,foreground='white')])
    ax.set_xlim(-11,31); ax.set_ylim(35,71.5); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_aspect(1/np.cos(np.deg2rad(53)))
    ax.set_title(f'{tag}  {title}',loc='left',fontsize=11,fontweight='bold',pad=24)
    ax.text(0,1.017,sub,transform=ax.transAxes,ha='left',va='bottom',
            fontsize=7.8,color=GREY,style='italic',linespacing=1.35)
    cax=fig.add_axes([X0[i]+0.036,0.072,W*0.68,0.019])
    cb=fig.colorbar(ScalarMappable(cmap=cmap,norm=norm),cax=cax,orientation='horizontal')
    cb.set_label(cblabel,fontsize=8.2); cb.ax.tick_params(labelsize=7.2)
    if cbticks is not None: cb.set_ticks(cbticks)
    return ax


for FT,FNAME in FEATS:
    ID={c:idp[FT][c]['balanced_accuracy']['mean']*100 for c in C}
    OOD={c:loco[FT]['per_country'][c]['balanced_accuracy']*100 for c in C}
    DGAP={c:OOD[c]-ID[c] for c in C}                       # signed: negative = lost
    tm=e2b[FT]['transfer_matrix']
    BEST={t:max(((s,tm[s][t]['balanced_accuracy']*100) for s in C if t in tm.get(s,{})),
                key=lambda x:x[1]) for t in C}
    DLT={t:BEST[t][1]-OOD[t] for t in C}

    fig=plt.figure(figsize=(14.0,6.0))
    nAB=Normalize(10,80)
    panel(fig,0,'a','Within country',
          'train and test are parcels of the\nsame country (5-fold CV)',
          ID,plt.cm.YlGnBu,nAB,{c:f'{ID[c]:.0f}' for c in C},
          'Balanced accuracy (%) — same scale as b')
    panel(fig,1,'b','Cross-border',
          'trained on the other ten countries,\ntested on this one (leave-one-country-out)',
          OOD,plt.cm.YlGnBu,nAB,{c:f'{OOD[c]:.0f}' for c in C},
          'Balanced accuracy (%) — same scale as a')
    nC=TwoSlopeNorm(vcenter=0,vmin=-45,vmax=10)
    panel(fig,2,'c','What the border costs',
          'c = b − a: change in accuracy when the\ntest country leaves the training set',
          DGAP,plt.cm.RdBu,nC,{c:f'{DGAP[c]:+.0f}' for c in C},
          'Change in balanced accuracy (pp)',cbticks=[-45,-30,-15,0,10])
    nD=TwoSlopeNorm(vcenter=0,vmin=-24,vmax=24)
    panel(fig,3,'d','One neighbour vs. all ten',
          'd = best single source − b: is one well-chosen\nneighbour better (+) than pooling all ten?',
          DLT,plt.cm.RdBu_r,nD,{t:f'{DLT[t]:+.0f} ←{BEST[t][0]}' for t in C},
          'Best single source − all ten (pp)',cbticks=[-24,-12,0,12,24],fs=6.1)

    fig.suptitle(f'Where transfer succeeds and what it costs — {FNAME}',
                 fontsize=13.5,fontweight='bold',x=0.020,ha='left',y=0.995)
    stamp_multi(fig,['within_country_cv','loco','loco','pairwise'],x=0.020,y=0.018,fontsize=7.4,
        extra='\nc is derived from a and b; d compares the pairwise protocol against b. Balanced accuracy over the 10 Tier-B classes, LightGBM, no coordinate features. In d, "←XX" names '
              'the single source\ncountry that transfers best to that target out of the ten candidates in the 11×11 pairwise matrix, and the number is how much better (+) or worse (−) that one source '
              'is than training on all ten together.')
    for e in ['pdf','png']:
        fig.savefig(f'{FIG}/fig_spatial_id_ood_{FT}.{e}')
        fig.savefig(f'{SL}/S12_spatial_ID_OOD_gap_{FT}.{e}')
    plt.close(fig); print(f'saved S12_spatial_ID_OOD_gap_{FT}')
