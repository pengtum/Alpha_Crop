"""Slide-ready figures following the presentation narrative: larger type, fewer
elements, no misleading annotations. Adds the country-ranking figure and the
feature-set-independence result that the slides asked for."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from protocol_labels import stamp, stamp_multi, NAME
import json, numpy as np, matplotlib, os
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.stats import pearsonr
plt.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42,'font.family':'Arial','font.size':12,
    'axes.spines.right':False,'axes.spines.top':False,'axes.linewidth':1.0,
    'savefig.bbox':'tight','savefig.pad_inches':0.12,'figure.dpi':200})
OUT='SLIDES_FIGURES'
C=['AT','BE','DE','EE','LT','LV','PT','RO','SE','SI','SK']
e2=json.load(open('results/experiments/exp2_loco/tier_b_nocoords_2021.json'))
e2b=json.load(open('results/experiments/exp2b_country_transfer/tier_b_nocoords_2021.json'))
CL={'ae':'#3B6CA8','s2':'#E08A3C','fusion':'#4F9D5B'}
NM={'ae':'AlphaEarth','s2':'Sentinel-2','fusion':'Fusion'}

# ---------- S01: country ranking, as source and as target ----------
out={}
for f in ['ae','s2','fusion']:
    tm=e2b[f]['transfer_matrix']
    out[f]={'src':{s:np.mean([tm[s][t]['balanced_accuracy'] for t in tm[s]])*100 for s in C},
            'tgt':{t:np.mean([tm[s][t]['balanced_accuracy'] for s in C if t in tm.get(s,{})])*100 for t in C}}
fig,ax=plt.subplots(1,2,figsize=(13.5,4.6))
for j,(k,lab) in enumerate([('src','As SOURCE — mean outgoing transfer'),('tgt','As TARGET — mean incoming transfer')]):
    order=sorted(C,key=lambda c:-out['fusion'][k][c])
    x=np.arange(len(C)); w=0.26
    for i,f in enumerate(['ae','s2','fusion']):
        ax[j].bar(x+(i-1)*w,[out[f][k][c] for c in order],w,color=CL[f],label=NM[f])
    ax[j].set_xticks(x); ax[j].set_xticklabels(order,fontsize=12)
    ax[j].set_ylabel('Balanced accuracy (%)'); ax[j].set_title(lab,fontsize=13,fontweight='bold')
    ax[j].set_ylim(0,50)
    if j==0: ax[j].legend(frameon=False,fontsize=11)
stamp(fig,'pairwise',x=0.0,y=0.0,fontsize=9,
      extra='Each bar averages one country\u2019s row (as source) or column (as target) of the 11\u00d711 matrix.')
fig.savefig(f'{OUT}/S01_country_ranking.pdf'); fig.savefig(f'{OUT}/S01_country_ranking.png'); plt.close(fig)
for k,lab in [('src','source'),('tgt','target')]:
    o=sorted(C,key=lambda c:-out['fusion'][k][c])
    print(f'S01 {lab} ranking :', ' > '.join('%s %.0f'%(c,out['fusion'][k][c]) for c in o))

# ---------- S02: the difficulty pattern is feature-set independent ----------
ae=np.array([e2['ae']['per_country'][c]['balanced_accuracy'] for c in C])*100
s2=np.array([e2['s2']['per_country'][c]['balanced_accuracy'] for c in C])*100
r,p=pearsonr(ae,s2)
fig,ax=plt.subplots(figsize=(6.2,5.4))
ax.scatter(s2,ae,s=110,color='#3B6CA8',zorder=3,edgecolor='white',linewidth=1.2)
OFF={'AT':(7,4),'SE':(7,-11),'SI':(-24,-10),'LT':(7,3),'BE':(7,-4),'DE':(7,2),'SK':(7,-6),
     'EE':(7,-2),'LV':(7,-4),'PT':(7,-4),'RO':(7,-4)}
for i,c in enumerate(C):
    ax.annotate(c,(s2[i],ae[i]),xytext=OFF.get(c,(7,-4)),textcoords='offset points',fontsize=11)
lim=[5,68]; ax.plot(lim,lim,'--',color='#888',lw=1.2)
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel('Sentinel-2 — held-out accuracy (%)')
ax.set_ylabel('AlphaEarth — held-out accuracy (%)')
ax.set_title('Which countries are hard is a property of geography,\nnot of the feature set',fontsize=13,fontweight='bold')
ax.text(0.05,0.93,f'$r$ = {r:.2f}  ($p$ = {p:.4f})',transform=ax.transAxes,fontsize=13,color='#1F3A93')
stamp(fig,'loco',x=0.0,y=0.0,fontsize=9)
fig.savefig(f'{OUT}/S02_pattern_featureset_independent.pdf'); fig.savefig(f'{OUT}/S02_pattern_featureset_independent.png'); plt.close(fig)
print(f'S02 AE vs S2 per-country r={r:.3f} p={p:.4f}')
