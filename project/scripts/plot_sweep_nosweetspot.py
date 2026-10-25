"""Cross-border accuracy vs number of removed country-predictive dimensions, and the
trade-off it implies: for Sentinel-2 there is an operating point where geography is
removed, the representation survives, and transfer improves. For AlphaEarth there is none."""
import json, numpy as np, matplotlib, os
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42,'font.family':'Arial','font.size':8,
    'axes.spines.right':False,'axes.spines.top':False,'axes.linewidth':0.7,
    'savefig.bbox':'tight','savefig.pad_inches':0.06,'figure.dpi':200})
B='results/experiments/exp8_debias_test'
OUT=os.path.join('Manuscript','Paper_Alpha_Earth','0001_draft_v1','figures')
sw=json.load(open(f'{B}/sweep.json')); lk=json.load(open(f'{B}/country_leakage_vs_K.json'))
T={}
for v in sw.values():
    if v['mode']=='top': T.setdefault(v['feat'],{})[v['K']]=(v['id']*100,v['ood']*100)
L={}
for v in lk.values():
    L.setdefault({'AlphaEarth':'AE','Sentinel-2':'S2'}[v['feat']],{})[v['K']]=v['country_ba']
C={'AE':'#3B6CA8','S2':'#E08A3C'}; N={'AE':'AlphaEarth','S2':'Sentinel-2'}
fig,ax=plt.subplots(1,3,figsize=(11.2,3.2))
# a: dOOD vs K
for f in ['AE','S2']:
    ks=sorted(T[f]); b=T[f][0][1]
    ax[0].plot(ks,[T[f][k][1]-b for k in ks],'-o',color=C[f],ms=3.2,lw=1.5,label=N[f])
ax[0].axhline(0,color='#333',lw=0.7); ax[0].set_xlabel('$K$ country-predictive dimensions removed')
ax[0].set_ylabel('Change in cross-border accuracy (pp)'); ax[0].legend(frameon=False,fontsize=7)
ax[0].set_title('a  Does removing them help transfer?',loc='left',fontsize=8.5,fontweight='bold')
ax[0].set_ylim(-20,4.2)
ax[0].annotate('Sentinel-2 peaks at $K\\approx$10 (+1.07 pp)',(10,1.07),xytext=(24,-4.5),
    fontsize=6.4,color='#9A5A18',style='italic',arrowprops=dict(arrowstyle='->',color='#9A5A18',lw=0.7))
ax[0].annotate('AlphaEarth: never positive',(24,-1.4),xytext=(26,-12),
    fontsize=6.4,color='#2A4A72',style='italic',arrowprops=dict(arrowstyle='->',color='#2A4A72',lw=0.7))
# b: country predictability vs K
for f in ['AE','S2']:
    ks=sorted(L[f]); ax[1].plot(ks,[L[f][k] for k in ks],'-o',color=C[f],ms=3.2,lw=1.5,label=N[f])
ax[1].set_xlabel('$K$ dimensions removed'); ax[1].set_ylabel('Country predictability (%)')
ax[1].set_title('b  Was the geography actually removed?',loc='left',fontsize=8.5,fontweight='bold')
ax[1].legend(frameon=False,fontsize=7)
# c: no sweet spot -- geography removed (x) vs transfer gain (y); hollow = representation damaged
for f in ['AE','S2']:
    ks=[k for k in sorted(T[f]) if k in L[f]]
    x=[L[f][0]-L[f][k] for k in ks]; y=[T[f][k][1]-T[f][0][1] for k in ks]
    dmg=[(T[f][k][0]-T[f][0][0])<-2 for k in ks]
    for xi,yi,dm,k in zip(x,y,dmg,ks):
        ax[2].scatter(xi,yi,s=40,facecolor='none' if dm else C[f],edgecolor=C[f],
                      linewidth=1.2,zorder=3)
        if k in (10,20,56,64): ax[2].annotate(f'$K$={k}',(xi,yi),xytext=(3,-7),
            textcoords='offset points',fontsize=6,color=C[f])
    ax[2].plot(x,y,'-',color=C[f],lw=1.0,alpha=0.5,label=N[f])
ax[2].axhline(0,color='#333',lw=0.7)
ax[2].axhspan(0,2.4,color='#3F7F5F',alpha=0.06)
ax[2].text(30,1.55,'geography removed\nAND transfer improved',fontsize=6.4,color='#2E6B45',style='italic')
ax[2].set_xlabel('Geography removed (pp drop in country predictability)')
ax[2].set_ylabel('Change in cross-border accuracy (pp)')
ax[2].set_title('c  Is there an operating point?',loc='left',fontsize=8.5,fontweight='bold')
ax[2].legend(frameon=False,fontsize=7,loc='lower left')
ax[2].text(0.98,0.42,'hollow = representation\ndamaged ($\\Delta$ID < $-$2 pp)',transform=ax[2].transAxes,
    ha='right',fontsize=6.2,color='#555',style='italic')
plt.tight_layout()
for e in ['pdf','png']: fig.savefig(os.path.join(OUT,f'fig_debias_sweep.{e}'))
print('saved fig_debias_sweep')
