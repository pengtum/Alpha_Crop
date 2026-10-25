"""All spatial figures redrawn on OSM-derived country boundaries (geoBoundaries gbOpen,
ADM0). Replaces the coarse Natural Earth 110m outlines used previously."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from protocol_labels import stamp, stamp_multi, NAME
import json, numpy as np, geopandas as gpd, matplotlib, os
matplotlib.use('Agg'); import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import Normalize; from matplotlib.cm import ScalarMappable
from matplotlib.patches import Patch
plt.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42,'font.family':'Arial','font.size':9,
    'savefig.bbox':'tight','savefig.pad_inches':0.06,'figure.dpi':220})
FIG='Manuscript/Paper_Alpha_Earth/0001_draft_v1/figures'; SL='SLIDES_FIGURES'
G=gpd.read_file('data/boundaries/osm_countries.gpkg')
ISO3={'AT':'AUT','BE':'BEL','DE':'DEU','EE':'EST','LT':'LTU','LV':'LVA',
      'PT':'PRT','RO':'ROU','SE':'SWE','SI':'SVN','SK':'SVK'}
C=list(ISO3)
LC={'AT':(14.3,47.6),'BE':(4.6,50.6),'DE':(10.4,51.2),'EE':(25.6,58.7),'LT':(23.9,55.3),
    'LV':(24.8,56.9),'PT':(-8.2,39.6),'RO':(25.0,45.9),'SE':(15.6,62.5),'SI':(14.9,46.1),'SK':(19.6,48.7)}
ZONE={'BE':'Atlantic','DE':'Atlantic','AT':'Continental','SK':'Continental',
      'PT':'Mediterranean','SI':'Mediterranean','EE':'Nordic-Baltic','LT':'Nordic-Baltic',
      'LV':'Nordic-Baltic','SE':'Nordic-Baltic','RO':'Southeast'}
ZC={'Atlantic':'#6BAED6','Continental':'#74C476','Mediterranean':'#FD8D3C',
    'Nordic-Baltic':'#9E9AC8','Southeast':'#E6550D'}
geo=lambda i: G[G.ISO3==ISO3[i]]
def base(ax):
    G.plot(ax=ax,color='#F2F2F0',edgecolor='#D8D8D5',linewidth=0.35)
def finish(ax):
    ax.set_xlim(-11,31); ax.set_ylim(35,71.5); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_aspect(1/np.cos(np.deg2rad(53)))   # simple equal-ish aspect for mid-Europe

# ---- Map 1: study area ----
fig,ax=plt.subplots(figsize=(5.2,5.8)); base(ax)
for i in C:
    geo(i).plot(ax=ax,color=ZC[ZONE[i]],edgecolor='white',linewidth=0.6)
    x,y=LC[i]; ax.annotate(i,(x,y),ha='center',va='center',fontsize=8.5,color='white',
        fontweight='bold',path_effects=[pe.withStroke(linewidth=2.2,foreground='#333')])
ax.legend(handles=[Patch(facecolor=v,edgecolor='white',label=k) for k,v in ZC.items()],
          loc='upper left',frameon=True,facecolor='white',edgecolor='#CCC',framealpha=0.92,
          fontsize=7.5,title='Climate zone',title_fontsize=8,borderpad=0.6)
finish(ax)
stamp(fig,'dataset',x=0.0,y=0.0,fontsize=7,
      extra='Eleven countries with strictly year-aligned 2021 declarations, grouped into five climate zones.')
for e in ['pdf','png']:
    fig.savefig(f'{FIG}/study_area_map.{e}'); fig.savefig(f'{SL}/S03_study_area.{e}')
plt.close(fig)

# ---- Maps 2&3: hub + generalization ----
e2b=json.load(open('results/experiments/exp2b_country_transfer/tier_b_nocoords_2021.json'))
tm=e2b['fusion']['transfer_matrix']
de={t:tm['DE'][t]['balanced_accuracy']*100 for t in tm['DE']}
mo={s:np.mean([tm[s][t]['balanced_accuracy'] for t in tm[s]])*100 for s in C}
def choro(vals,lab,fn,src=None,vmin=None,vmax=None,note=''):
    fig,ax=plt.subplots(figsize=(5.2,5.8)); base(ax)
    cmap=plt.cm.YlGnBu; nm=Normalize(vmin=vmin,vmax=vmax)
    for i in C:
        if src and i==src:
            geo(i).plot(ax=ax,color='#3A3A3A',edgecolor='white',linewidth=0.7)
            x,y=LC[i]; ax.annotate(f'{i}\n(source)',(x,y),ha='center',va='center',
                fontsize=7.5,color='white',fontweight='bold'); continue
        v=vals.get(i)
        if v is None: continue
        geo(i).plot(ax=ax,color=cmap(nm(v)),edgecolor='white',linewidth=0.6)
        x,y=LC[i]; ax.annotate(f'{i}\n{v:.0f}',(x,y),ha='center',va='center',fontsize=7.5,
            color='black',path_effects=[pe.withStroke(linewidth=1.8,foreground='white')])
    sm=ScalarMappable(cmap=cmap,norm=nm); sm.set_array([])
    cb=fig.colorbar(sm,ax=ax,shrink=0.48,aspect=20,pad=0.01)
    cb.set_label(lab,fontsize=8); cb.ax.tick_params(labelsize=7)
    finish(ax)
    stamp(fig,'pairwise',x=0.0,y=0.0,fontsize=7,extra=note)
    for e in ['pdf','png']: fig.savefig(f'{FIG}/{fn}.{e}')
    plt.close(fig)
choro(de,'Transfer accuracy from Germany (%)','map_hub_germany',src='DE',vmin=10,vmax=55,
      note='Germany is the single source country; each other country is a target. Fusion features.')
choro(mo,'Mean outgoing transfer accuracy (%)','map_generalization_power',vmin=15,vmax=40,
      note='Each country is coloured by its mean accuracy as a source over the ten targets. Fusion features.')
import shutil
for a,b in [('map_hub_germany','S07_map_hub_Germany'),('map_generalization_power','S06_map_generalization')]:
    for e in ['pdf','png']: shutil.copy(f'{FIG}/{a}.{e}',f'{SL}/{b}.{e}')
print('redrawn on OSM boundaries: study_area, map_hub_germany, map_generalization_power')
