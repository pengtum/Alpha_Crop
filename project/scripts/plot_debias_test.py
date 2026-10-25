"""Can the geographic signal be removed by deleting the dimensions that carry it?
Drop the top-K country-predictive features (SHAP-ranked) and re-train the crop
classifier. Random-drop controls isolate 'losing K features' from 'losing these K'."""
import numpy as np, matplotlib, os
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42,'font.family':'Arial','font.size':8,
    'axes.spines.right':False,'axes.spines.top':False,'axes.linewidth':0.7,
    'savefig.bbox':'tight','savefig.pad_inches':0.06,'figure.dpi':200})
OUT=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'Manuscript','Paper_Alpha_Earth','0001_draft_v1','figures')
D={'AE':{0:(63.97,41.96),10:(63.60,41.83),20:(62.96,41.02),40:(59.60,38.90)},
   'S2':{0:(60.75,38.78),10:(60.42,39.86),20:(59.60,39.50),40:(58.04,38.26)}}
R={'AE':{20:(63.82,41.42),40:(63.79,41.14)},'S2':{20:(60.89,38.87),40:(60.36,38.82)}}
C={'AE':'#3B6CA8','S2':'#E08A3C'}; N={'AE':'AlphaEarth','S2':'Sentinel-2'}
Ks=[0,10,20,40]
fig,(ax2,ax1)=plt.subplots(1,2,figsize=(7.4,3.0))
for j,(ax,i,lab) in enumerate([(ax2,1,'cross-border (leave-one-country-out)'),(ax1,0,'in-distribution')]):
    for f in ['AE','S2']:
        b=D[f][0][i]
        ax.plot(Ks,[D[f][k][i]-b for k in Ks],'-o',color=C[f],ms=4.5,lw=1.6,
                label=f'{N[f]} — drop top-$K$ geographic')
        rk=[0,20,40]; ax.plot(rk,[0]+[R[f][k][i]-b for k in [20,40]],'--s',color=C[f],ms=3.5,
                lw=1.1,alpha=0.55,label=f'{N[f]} — drop $K$ at random')
    ax.axhline(0,color='#333',lw=0.7)
    ax.set_xlabel('$K$ dimensions removed'); ax.set_xticks(Ks)
    ax.set_ylabel(f'Change in {"cross-border" if i==1 else "in-distribution"}\nbalanced accuracy (pp)')
    ax.set_title(('a  '+lab) if j==0 else ('b  '+lab),loc='left',fontsize=8.5,fontweight='bold')
ax2.set_ylim(-3.6,1.7)
ax2.legend(frameon=False,fontsize=6.3,loc='lower left')
ax2.annotate('removing them\nhelps transfer',(10,1.08),xytext=(13,0.35),fontsize=6.5,
    color='#9A5A18',style='italic',ha='left')
ax2.annotate('never helps',(20,-0.94),xytext=(24,-1.9),fontsize=6.5,color='#2A4A72',
    style='italic',ha='left')
for ext in ['pdf','png']: fig.savefig(os.path.join(OUT,f'fig_debias_test.{ext}'))
print('saved fig_debias_test')
