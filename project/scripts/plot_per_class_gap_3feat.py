"""What the border costs, per class, for each feature set separately.
The earlier version of this panel showed Fusion only."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from protocol_labels import stamp, stamp_multi, NAME
import json, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42,'font.family':'Arial','font.size':9,
    'axes.spines.right':False,'axes.spines.top':False,'axes.linewidth':0.8,
    'savefig.bbox':'tight','savefig.pad_inches':0.08,'figure.dpi':210})
C=['AT','BE','DE','EE','LT','LV','PT','RO','SE','SI','SK']
F=['ae','s2','fusion']; NM={'ae':'AlphaEarth','s2':'Sentinel-2','fusion':'Fusion'}
CL={'ae':'#3B6CA8','s2':'#E08A3C','fusion':'#4F9D5B'}
loco=json.load(open('results/experiments/exp2_loco/tier_b_nocoords_2021.json'))
cvs={f:pd.read_csv(f'results/experiments/exp0_per_country_cv/per_class_{f}_tier_b.csv') for f in F}
rows=[]
for f in F:
    for c in C:
        for cl,v in loco[f]['per_country'][c]['per_class'].items():
            if v['support']>=10: rows.append(dict(feat=f,country=c,cls=cl,ood=v['recall']*100))
D=pd.DataFrame(rows)
I=pd.concat([cvs[f].rename(columns={'class':'cls'}).assign(feat=f)[['feat','country','cls','recall','support']] for f in F])
I=I[I.support>=10]; I['idv']=I.recall*100
M=D.merge(I[['feat','country','cls','idv']],on=['feat','country','cls']); M['gap']=M.idv-M.ood
T=M.groupby(['cls','feat']).agg(ID=('idv','mean'),OOD=('ood','mean'),GAP=('gap','mean')).reset_index()
ORD=T[T.feat=='fusion'].sort_values('OOD',ascending=False).cls.tolist()
L=lambda c:c.replace('_',' ')
fig,ax=plt.subplots(1,2,figsize=(13.4,5.6),gridspec_kw={'width_ratios':[1.25,1]})
# a: ID -> OOD dumbbells, three feature sets offset
OFF={'ae':+0.26,'s2':0.0,'fusion':-0.26}
for i,cl in enumerate(ORD):
    for f in F:
        r=T[(T.cls==cl)&(T.feat==f)]
        if not len(r): continue
        a,o=float(r.ID.iloc[0]),float(r.OOD.iloc[0]); yy=i+OFF[f]
        ax[0].hlines(yy,o,a,color=CL[f],lw=1.6,alpha=.45,zorder=1)
        ax[0].scatter(a,yy,s=26,facecolor='white',edgecolor=CL[f],linewidth=1.4,zorder=3)
        ax[0].scatter(o,yy,s=30,color=CL[f],zorder=3)
for f in F:
    ax[0].scatter([],[],s=30,color=CL[f],label=NM[f])
ax[0].scatter([],[],s=26,facecolor='white',edgecolor='#555',linewidth=1.4,label='within country (open)')
ax[0].scatter([],[],s=30,color='#555',label='cross-border (filled)')
ax[0].set_yticks(range(len(ORD))); ax[0].set_yticklabels([L(c) for c in ORD],fontsize=9); ax[0].invert_yaxis()
ax[0].set_xlabel('Recall (%)'); ax[0].legend(frameon=False,fontsize=7.5,loc='lower right',ncol=1)
ax[0].set_title('a  Within-country → cross-border, by class and feature set',loc='left',fontsize=10,fontweight='bold')
ax[0].grid(axis='y',color='#EEE',lw=0.7)
# b: the gap as grouped bars
y=np.arange(len(ORD)); w=0.26
for k,f in enumerate(F):
    v=[float(T[(T.cls==c)&(T.feat==f)].GAP.iloc[0]) if len(T[(T.cls==c)&(T.feat==f)]) else np.nan for c in ORD]
    ax[1].barh(y+(1-k)*w,v,w,color=CL[f],label=NM[f])
ax[1].axvline(0,color='#333',lw=0.7)
ax[1].set_yticks(y); ax[1].set_yticklabels([L(c) for c in ORD],fontsize=9); ax[1].invert_yaxis()
ax[1].set_xlabel('Recall lost when crossing the border (pp)')
ax[1].legend(frameon=False,fontsize=8,loc='lower right')
ax[1].set_title('b  What the border costs',loc='left',fontsize=10,fontweight='bold')
plt.tight_layout()
stamp_multi(fig,['within_country_cv','loco'],x=0.0,y=0.0,fontsize=7.8,panel_letters=False,
    extra='\nThe gap plotted here is within-country recall minus cross-border recall, for the same country and class. This pairs two cells on the DIAGONAL of the protocol 2×2: it changes both the training breadth (1 country vs 10) and the distribution shift at once. It is the choice a practitioner\nactually faces — label this country, or rely on ten others — but it is not a clean measurement of either effect alone, and it is NOT the headline 24–26 pp ID→OOD gap,\nwhich contrasts pooled cross-validation with leave-one-country-out at fixed training breadth.')
for e in ['pdf','png']:
    fig.savefig(f'SLIDES_FIGURES/S15_per_class_gap_by_featureset.{e}')
    fig.savefig(f'Manuscript/Paper_Alpha_Earth/0001_draft_v1/figures/fig_per_class_gap.{e}')
print('saved S15_per_class_gap_by_featureset')
