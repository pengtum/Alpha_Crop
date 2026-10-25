"""Cross-border recall mapped per crop class.

Primary output is a single grid: one ROW per class, one COLUMN per feature set,
so the three representations are read side by side within each class and the
classes stack vertically. Per-class stand-alone figures (same content, one class
each) are also written for slide use.

Absence of a class in a country is drawn separately from presence-but-failure,
because the two call for different remedies.
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from protocol_labels import stamp, stamp_multi, NAME
import json, numpy as np, geopandas as gpd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import Normalize; from matplotlib.cm import ScalarMappable
from matplotlib.patches import Patch
plt.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42,'font.family':'Arial','font.size':9,
    'savefig.bbox':'tight','savefig.pad_inches':0.08,'figure.dpi':150})

G=gpd.read_file('data/boundaries/osm_countries.gpkg')
ISO3={'AT':'AUT','BE':'BEL','DE':'DEU','EE':'EST','LT':'LTU','LV':'LVA',
      'PT':'PRT','RO':'ROU','SE':'SWE','SI':'SVN','SK':'SVK'}
C=list(ISO3)
LC={'AT':(14.3,47.6),'BE':(4.6,50.6),'DE':(10.4,51.2),'EE':(25.6,58.7),'LT':(23.9,55.3),
    'LV':(24.8,56.9),'PT':(-8.2,39.6),'RO':(25.0,45.9),'SE':(15.6,62.5),'SI':(14.9,46.1),'SK':(19.6,48.7)}
geo=lambda i: G[G.ISO3==ISO3[i]]
loco=json.load(open('results/experiments/exp2_loco/tier_b_nocoords_2021.json'))

FEATS=[('ae','AlphaEarth'),('s2','Sentinel-2'),('fusion','Fusion (AlphaEarth + Sentinel-2)')]
CLS=[('maize','Maize','transfers well across borders'),
     ('winter_cereals','Winter cereals','transfers well across borders'),
     ('potatoes','Potatoes','AlphaEarth advantage: +11.9 pp, $p$ = 0.039'),
     ('other_crops','Other crops','abundant but semantically heterogeneous'),
     ('tree_wood_forest','Tree / wood / forest','largest collapse; Sentinel-2 advantage: +6.2 pp, $p$ = 0.031'),
     ('strawberries','Strawberries','fails almost everywhere under every feature set')]
cmap=plt.cm.RdYlGn; nm=Normalize(0,90)
LEG=[Patch(facecolor='white',edgecolor='#BBB',hatch='///',label='class absent from that country'),
     Patch(facecolor='#E8E8E8',edgecolor='#BBB',label='fewer than 10 parcels')]


def common_countries(cl):
    """Countries where every feature set has >=10 target parcels, so the per-panel
    means within a row are computed on the same set and are directly comparable."""
    out=[]
    for i in C:
        v=[loco[f]['per_country'][i]['per_class'].get(cl) for f,_ in FEATS]
        if all(x and int(x['support'])>=10 for x in v): out.append(i)
    return out


def panel(ax,feat,cl,common,lab_fs=6.3):
    G.plot(ax=ax,color='#F4F4F2',edgecolor='#DCDCD9',linewidth=0.3)
    pc=loco[feat]['per_country']; tot=0; rec=[]
    for i in C:
        v=pc[i]['per_class'].get(cl); sup=int(v['support']) if v else 0; tot+=sup
        g=geo(i)
        if sup==0:
            g.plot(ax=ax,color='#FFFFFF',edgecolor='#BBB',linewidth=0.6,hatch='///')
            lab=f'{i}\nabsent'
        elif sup<10:
            g.plot(ax=ax,color='#E8E8E8',edgecolor='#BBB',linewidth=0.6)
            lab=f'{i}\nn={sup}'
        else:
            r=v['recall']*100
            if i in common: rec.append(r)
            g.plot(ax=ax,color=cmap(nm(r)),edgecolor='white',linewidth=0.6)
            lab=f'{i}\n{r:.0f}%'
        x,y=LC[i]
        ax.annotate(lab,(x,y),ha='center',va='center',fontsize=lab_fs,color='black',
                    path_effects=[pe.withStroke(linewidth=1.6,foreground='white')])
    ax.set_xlim(-11,31); ax.set_ylim(35,71.5); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_aspect(1/np.cos(np.deg2rad(53)))
    return tot,(float(np.mean(rec)) if rec else float('nan'))


# ---------------------------------------------------------------- grid figure
NR=len(CLS)
fig,axes=plt.subplots(NR,3,figsize=(13.4,6.3*NR))
fig.subplots_adjust(left=0.035,right=0.868,top=0.958,bottom=0.026,hspace=0.24,wspace=0.02)
for r,(cl,name,strap) in enumerate(CLS):
    common=common_countries(cl)
    for ax,(feat,fname) in zip(axes[r],FEATS):
        tot,mrec=panel(ax,feat,cl,common,lab_fs=5.9)
        ax.set_title(f'{fname}\n$n$ = {tot:,} parcels · mean recall {mrec:.1f}%',
                     loc='left',fontsize=8.6,fontweight='bold',color='#222',pad=4)
    # row banner, left-aligned over the whole row
    bb=axes[r,0].get_position()
    axes[r,0].text(0,1.175,f'{name} — {strap}',transform=axes[r,0].transAxes,
                   ha='left',va='bottom',fontsize=12,fontweight='bold',color='#111')
    axes[r,0].annotate('',xy=(3.42,1.165),xytext=(0,1.165),xycoords='axes fraction',
                       arrowprops=dict(arrowstyle='-',color='#111',lw=1.0))
fig.suptitle(f'Cross-border recall by crop class and feature set — {NAME["loco"]}',
             fontsize=15,fontweight='bold',x=0.035,ha='left',y=0.994)
sm=ScalarMappable(cmap=cmap,norm=nm); sm.set_array([])
cax=fig.add_axes([0.885,0.775,0.013,0.125])
cb=fig.colorbar(sm,cax=cax)
cb.set_label('Cross-border recall (%)',fontsize=9.5)
cb.ax.tick_params(labelsize=8.5)
fig.legend(handles=LEG,loc='lower center',ncol=2,frameon=False,fontsize=10,
           bbox_to_anchor=(0.45,0.006))
stamp(fig,'loco',x=0.5,y=-0.001,ha='center',fontsize=8.6,
      extra='(10% of the training parcels are held out for validation, also drawn from the training countries.) Each map colours a country by the recall it then\n'
            'achieves for that class. Within a row, mean recall is averaged over the countries in which all three feature sets have at least 10 parcels of the class,\n'
            'so the three panels of a row are directly comparable.')
for e in ['pdf','png']:
    fig.savefig(f'SLIDES_FIGURES/S16_class_grid.{e}')
    fig.savefig(f'Manuscript/Paper_Alpha_Earth/0001_draft_v1/figures/fig_class_grid.{e}')
plt.close(fig); print('saved S16_class_grid')

# ------------------------------------------- stand-alone per-class (slide use)
for cl,name,strap in CLS:
    fig,axes=plt.subplots(1,3,figsize=(15.6,6.6))
    common=common_countries(cl)
    for ax,(feat,fname) in zip(axes,FEATS):
        tot,mrec=panel(ax,feat,cl,common)
        ax.set_title(f'{fname}\n$n$ = {tot:,} parcels · mean recall {mrec:.1f}%',
                     loc='left',fontsize=9.5,fontweight='bold')
    fig.suptitle(f'{name} — cross-border recall, {NAME["loco"]}\n{strap}',
                 fontsize=12.5,fontweight='bold',y=1.015)
    sm=ScalarMappable(cmap=cmap,norm=nm); sm.set_array([])
    cb=fig.colorbar(sm,ax=axes,shrink=0.62,aspect=24,pad=0.012)
    cb.set_label('Cross-border recall (%)',fontsize=9)
    fig.legend(handles=LEG,loc='lower center',ncol=2,frameon=False,fontsize=9,
               bbox_to_anchor=(0.45,0.035))
    stamp(fig,'loco',x=0.45,y=-0.005,ha='center',fontsize=8.2,
          extra=f'Each country is coloured by the recall it then achieves for this class. Mean recall is averaged over the {len(common)} countries in which all three\n'
                'feature sets have at least 10 parcels of this class, so the three panels are directly comparable.')
    for e in ['pdf','png']:
        fig.savefig(f'SLIDES_FIGURES/S16_class_{cl}.{e}')
        fig.savefig(f'Manuscript/Paper_Alpha_Earth/0001_draft_v1/figures/fig_class_{cl}.{e}')
    plt.close(fig); print(f'saved S16_class_{cl}')
