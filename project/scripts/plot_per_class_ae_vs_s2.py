"""Per-class comparison of the two representations. Overall they are indistinguishable,
but two classes differ in opposite directions, and AlphaEarth fails more completely."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from protocol_labels import stamp, stamp_multi, NAME
import json, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
plt.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42,'font.family':'Arial','font.size':9,
    'axes.spines.right':False,'axes.spines.top':False,'axes.linewidth':0.8,
    'savefig.bbox':'tight','savefig.pad_inches':0.08,'figure.dpi':210})
C=['AT','BE','DE','EE','LT','LV','PT','RO','SE','SI','SK']
loco=json.load(open('results/experiments/exp2_loco/tier_b_nocoords_2021.json'))
rows=[]
for f in ['ae','s2']:
    for c in C:
        for cl,v in loco[f]['per_country'][c]['per_class'].items():
            rows.append(dict(feat=f,country=c,cls=cl,r=v['recall']*100,sup=v['support']))
D=pd.DataFrame(rows)
D.loc[D.sup<10,'r']=np.nan          # single-parcel classes are not interpretable
ORD=D[D.feat=='s2'].groupby('cls').r.mean().sort_values(ascending=False).index.tolist()
L=lambda c:c.replace('_',' ')
piv=lambda f: D[D.feat==f].pivot_table(index='cls',columns='country',values='r').reindex(ORD)[C]
A,S=piv('ae'),piv('s2'); DIF=A-S
fig=plt.figure(figsize=(15.6,7.6))
gs=fig.add_gridspec(2,3,height_ratios=[1,0.92],hspace=0.45,wspace=0.28)
def hm(ax,P,title,cmap,vmin,vmax,cl):
    im=ax.imshow(P.values,cmap=cmap,vmin=vmin,vmax=vmax,aspect='auto')
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            v=P.values[i,j]
            ax.text(j,i,'·' if np.isnan(v) else f'{v:.0f}',ha='center',va='center',fontsize=6.2,
                    color='#999' if np.isnan(v) else ('#222' if vmin+0.2*(vmax-vmin)<v<vmax-0.2*(vmax-vmin) else 'white'))
    ax.set_xticks(range(len(C))); ax.set_xticklabels(C,fontsize=7.5)
    ax.set_yticks(range(len(ORD))); ax.set_yticklabels([L(c) for c in ORD],fontsize=7.5)
    cb=fig.colorbar(im,ax=ax,shrink=0.62,pad=0.02); cb.set_label(cl,fontsize=7.5); cb.ax.tick_params(labelsize=6.5)
    ax.set_title(title,loc='left',fontsize=10,fontweight='bold')
hm(fig.add_subplot(gs[0,0]),A,'a  AlphaEarth','RdYlGn',0,90,'Cross-border recall (%)')
hm(fig.add_subplot(gs[0,1]),S,'b  Sentinel-2','RdYlGn',0,90,'Cross-border recall (%)')
hm(fig.add_subplot(gs[0,2]),DIF,'c  AlphaEarth − Sentinel-2','RdBu_r',-40,40,'Difference (pp)')
# d: per-class paired comparison with p-values
ax=fig.add_subplot(gs[1,0:2])
P=D.pivot_table(index=['cls','country'],columns='feat',values='r').dropna()
stat=[]
for cl in ORD:
    g=P.loc[cl] if cl in P.index.get_level_values(0) else None
    if g is None or len(g)<5: stat.append((cl,np.nan,np.nan,np.nan,np.nan)); continue
    p=wilcoxon(g['ae'],g['s2']).pvalue
    stat.append((cl,g['ae'].mean(),g['s2'].mean(),(g['ae']-g['s2']).mean(),p))
y=np.arange(len(ORD))
for i,(cl,a,s,d,p) in enumerate(stat):
    if np.isnan(a): continue
    ax.hlines(i,min(a,s),max(a,s),color='#CCC',lw=2,zorder=1)
    ax.scatter(a,i,s=46,color='#3B6CA8',zorder=3)
    ax.scatter(s,i,s=46,color='#E08A3C',zorder=3)
    if p<0.05: ax.text(max(a,s)+2.5,i,f'$p$={p:.3f} *',fontsize=7.5,va='center',color='#B0483B',fontweight='bold')
ax.scatter([],[],s=46,color='#3B6CA8',label='AlphaEarth'); ax.scatter([],[],s=46,color='#E08A3C',label='Sentinel-2')
ax.set_yticks(y); ax.set_yticklabels([L(c) for c in ORD],fontsize=8); ax.invert_yaxis()
ax.set_xlabel('Cross-border recall, mean over held-out countries (%)'); ax.set_xlim(-4,92)
ax.legend(frameon=False,fontsize=8,loc='lower right')
ax.set_title('d  Overall indistinguishable — but two classes differ, in opposite directions',loc='left',fontsize=10,fontweight='bold')
# e: complete-failure rate
ax=fig.add_subplot(gs[1,2])
v=[(P['ae']<1).mean()*100,(P['s2']<1).mean()*100,(P['ae']<10).mean()*100,(P['s2']<10).mean()*100]
ax.bar([0,1],v[:2],color=['#3B6CA8','#E08A3C'],width=.62)
ax.bar([2.5,3.5],v[2:],color=['#3B6CA8','#E08A3C'],width=.62)
for x,q in zip([0,1,2.5,3.5],v): ax.text(x,q+0.5,f'{q:.0f}%',ha='center',fontsize=8)
ax.set_xticks([0,1,2.5,3.5]); ax.set_xticklabels(['AE','S2','AE','S2'],fontsize=8.5)
ax.set_ylim(0,26); ax.set_ylabel('Share of country×class cases (%)')
ax.text(0.5,-4.2,'recall < 1%',ha='center',fontsize=7.5,color='#444')
ax.text(3.0,-4.2,'recall < 10%',ha='center',fontsize=7.5,color='#444')
ax.set_title('e  AlphaEarth fails more completely',loc='left',fontsize=10,fontweight='bold')
stamp(fig,'loco',x=0.0,y=-0.005,fontsize=7.8,
      extra='Cells are per-class recall in the held-out country; cells with fewer than 10 parcels of the class are excluded.')
for e in ['pdf','png']:
    fig.savefig(f'SLIDES_FIGURES/S14_per_class_AE_vs_S2.{e}')
    fig.savefig(f'Manuscript/Paper_Alpha_Earth/0001_draft_v1/figures/fig_per_class_ae_vs_s2.{e}')
print('saved S14_per_class_AE_vs_S2')
