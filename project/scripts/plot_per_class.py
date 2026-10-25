"""Per-class heterogeneity: a single balanced-accuracy number hides that some classes
transfer across borders and others collapse entirely."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from protocol_labels import stamp, stamp_multi, NAME
import json, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42,'font.family':'Arial','font.size':9,
    'axes.spines.right':False,'axes.spines.top':False,'axes.linewidth':0.8,
    'savefig.bbox':'tight','savefig.pad_inches':0.08,'figure.dpi':210})
C=['AT','BE','DE','EE','LT','LV','PT','RO','SE','SI','SK']
loco=json.load(open('results/experiments/exp2_loco/tier_b_nocoords_2021.json'))
cvs={f:pd.read_csv(f'results/experiments/exp0_per_country_cv/per_class_{f}_tier_b.csv') for f in ['ae','s2','fusion']}
rows=[]
for f in ['ae','s2','fusion']:
    for c in C:
        for cl,v in loco[f]['per_country'][c]['per_class'].items():
            if v['support']>0: rows.append(dict(feat=f,country=c,cls=cl,ood=v['recall']*100,support=v['support']))
D=pd.DataFrame(rows)
I=pd.concat([cvs[f].rename(columns={'class':'cls','feature_type':'feat'}).assign(feat=f)[['feat','country','cls','recall','support']]
             for f in cvs]).query('support>0')
I['idv']=I.recall*100
M=D.merge(I[['feat','country','cls','idv']],on=['feat','country','cls'],how='inner'); M['gap']=M.idv-M.ood
ORD=(D[D.feat=='fusion'].groupby('cls').ood.mean().sort_values(ascending=False).index.tolist())
LBL={c:c.replace('_',' ') for c in ORD}
fig=plt.figure(figsize=(14.5,7.4))
gs=fig.add_gridspec(2,3,width_ratios=[1.35,1,1],height_ratios=[1,1],hspace=0.42,wspace=0.30)
# a: class x country heatmap (fusion)
ax=fig.add_subplot(gs[:,0])
P=D[D.feat=='fusion'].pivot_table(index='cls',columns='country',values='ood').reindex(ORD)[C]
im=ax.imshow(P.values,cmap='RdYlGn',vmin=0,vmax=90,aspect='auto')
for i in range(P.shape[0]):
    for j in range(P.shape[1]):
        v=P.values[i,j]
        ax.text(j,i,'–' if np.isnan(v) else f'{v:.0f}',ha='center',va='center',fontsize=6.6,
                color='#222' if (np.isnan(v) or 15<v<75) else 'white')
ax.set_xticks(range(len(C))); ax.set_xticklabels(C,fontsize=8)
ax.set_yticks(range(len(ORD))); ax.set_yticklabels([LBL[c] for c in ORD],fontsize=8)
ax.set_xlabel('Held-out country'); cb=fig.colorbar(im,ax=ax,shrink=0.5,pad=0.02)
cb.set_label('Cross-border recall (%)',fontsize=8); cb.ax.tick_params(labelsize=7)
ax.set_title('a  Cross-border recall by class and country (Fusion)',loc='left',fontsize=10,fontweight='bold')
# b: ID vs OOD per class
ax=fig.add_subplot(gs[0,1])
g=M[M.feat=='fusion'].groupby('cls').agg(ID=('idv','mean'),OOD=('ood','mean')).reindex(ORD).dropna()
y=np.arange(len(g))
ax.hlines(y,g.OOD,g.ID,color='#BBB',lw=2,zorder=1)
ax.scatter(g.ID,y,s=42,color='#3B6CA8',zorder=3,label='within country')
ax.scatter(g.OOD,y,s=42,color='#B0483B',zorder=3,label='cross-border')
ax.set_yticks(y); ax.set_yticklabels([LBL[c] for c in g.index],fontsize=8); ax.invert_yaxis()
ax.set_xlabel('Recall (%)'); ax.legend(frameon=False,fontsize=7.5,loc='lower right')
ax.set_title('b  What the border costs, per class (Fusion)',loc='left',fontsize=10,fontweight='bold')
# c: complete-failure rate AE vs S2
ax=fig.add_subplot(gs[0,2])
p=M.pivot_table(index=['country','cls'],columns='feat',values='ood').dropna()
bars=[(p['ae']<1).mean()*100,(p['s2']<1).mean()*100,(p['ae']<10).mean()*100,(p['s2']<10).mean()*100]
ax.bar([0,1],bars[:2],color=['#3B6CA8','#E08A3C'],width=.65)
ax.bar([2.6,3.6],bars[2:],color=['#3B6CA8','#E08A3C'],width=.65)
for x,v in zip([0,1,2.6,3.6],bars): ax.text(x,v+0.6,f'{v:.0f}%',ha='center',fontsize=8)
ax.set_xticks([0,1,2.6,3.6]); ax.set_xticklabels(['AE','S2','AE','S2'],fontsize=8.5)
ax.text(0.5,-4.8,'recall < 1%\n(class vanishes)',ha='center',fontsize=7.5,color='#444')
ax.text(3.1,-4.8,'recall < 10%',ha='center',fontsize=7.5,color='#444')
ax.set_ylabel('Share of country×class cases (%)'); ax.set_ylim(0,28)
ax.set_title('c  AlphaEarth fails more completely',loc='left',fontsize=10,fontweight='bold')
ax.text(0.5,24,'6/6 discordant cases\nfavour S2 ($p$=0.03)',ha='center',fontsize=7,color='#B0483B',style='italic')
# d: recall vs support
ax=fig.add_subplot(gs[1,1:])
sub=D[D.feat=='fusion']
ax.scatter(sub.support,sub.ood,s=20,color='#4F9D5B',alpha=.55,edgecolor='none')
ax.set_xscale('log'); ax.set_xlabel('Class support in the held-out country (parcels, log)')
ax.set_ylabel('Cross-border recall (%)')
for cl in ['strawberries','grassland','maize','other_crops']:
    q=sub[sub.cls==cl]
    ax.annotate(LBL[cl],(q.support.median(),q.ood.median()),fontsize=7.5,color='#2E6B45',
                xytext=(6,4),textcoords='offset points')
from scipy.stats import spearmanr
r=spearmanr(np.log10(sub.support+1),sub.ood).statistic
ax.text(0.02,0.95,f'Spearman $\\rho$ = {r:.2f}',transform=ax.transAxes,fontsize=9,color='#2E6B45')
ax.set_title('d  Rare classes do not transfer — but abundance is not sufficient (Fusion)',loc='left',fontsize=10,fontweight='bold')
stamp_multi(fig,['within_country_cv','loco'],x=0.0,y=0.0,fontsize=7.8,panel_letters=False,
    extra='\nPanels a, c and d show leave-one-country-out recall only; panel b contrasts the two. This pairs two cells on the DIAGONAL of the protocol 2×2: it changes both the training breadth (1 country vs 10) and the distribution shift at once. It is the choice a practitioner\nactually faces — label this country, or rely on ten others — but it is not a clean measurement of either effect alone, and it is NOT the headline 24–26 pp ID→OOD gap,\nwhich contrasts pooled cross-validation with leave-one-country-out at fixed training breadth.')
for e in ['pdf','png']:
    fig.savefig(f'SLIDES_FIGURES/S13_per_class_heterogeneity.{e}')
    fig.savefig(f'Manuscript/Paper_Alpha_Earth/0001_draft_v1/figures/fig_per_class.{e}')
print('saved S13_per_class_heterogeneity')
