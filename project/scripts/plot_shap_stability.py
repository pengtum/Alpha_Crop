"""Explanation drift predicts transfer failure, and does so equally for a learned
embedding and engineered features. Drift is computed without target labels."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from protocol_labels import stamp, stamp_multi, NAME
import json, numpy as np, pandas as pd, matplotlib, os
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.stats import pearsonr, wilcoxon
plt.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42,'font.family':'Arial','font.size':8,
    'axes.spines.right':False,'axes.spines.top':False,'axes.linewidth':0.7,
    'savefig.bbox':'tight','savefig.pad_inches':0.06,'figure.dpi':200})
OUT=os.path.join('Manuscript','Paper_Alpha_Earth','0001_draft_v1','figures')
d=json.load(open('results/experiments/exp9_shap_stability/stability.json'))
C=['AT','BE','DE','EE','LT','LV','PT','RO','SE','SI','SK']
rows=[]
for f in ['AE','S2']:
    for s in C:
        b=d[f'{f}|{s}']['baseline_within_country']['spearman']
        for t,v in d[f'{f}|{s}']['targets'].items():
            rows.append(dict(feat=f,src=s,tgt=t,drift=1-v['spearman'],ba=v['ba'],base=1-b))
D=pd.DataFrame(rows)
CL={'AE':'#3B6CA8','S2':'#E08A3C'}; NM={'AE':'AlphaEarth','S2':'Sentinel-2'}
fig,ax=plt.subplots(1,3,figsize=(11.0,3.2))
# a: drift vs transfer accuracy
for f in ['AE','S2']:
    x=D[D.feat==f]; r,p=pearsonr(x.drift,x.ba)
    ax[0].scatter(x.drift,x.ba,s=13,color=CL[f],alpha=0.55,edgecolor='none',
                  label=f'{NM[f]}  $r$={r:+.2f}')
    z=np.polyfit(x.drift,x.ba,1); xs=np.linspace(x.drift.min(),x.drift.max(),50)
    ax[0].plot(xs,np.polyval(z,xs),color=CL[f],lw=1.6)
bl=D.base.mean()
ax[0].axvline(bl,color='#666',ls=':',lw=1.0)
ax[0].text(bl,ax[0].get_ylim()[1]*0.97,' within-country\n baseline',fontsize=6.2,color='#555',
           va='top',style='italic')
ax[0].set_xlabel('Explanation drift  $1-\\rho$(source, target)')
ax[0].set_ylabel('Transfer accuracy (%)'); ax[0].legend(frameon=False,fontsize=7)
ax[0].set_title('a  Drift predicts transfer failure',loc='left',fontsize=8.5,fontweight='bold')
# b: within-target-country (centred) -- does it survive fixed effects
for f in ['AE','S2']:
    x=D[D.feat==f].copy()
    x['dc']=x.drift-x.groupby('tgt').drift.transform('mean')
    x['bc']=x.ba   -x.groupby('tgt').ba.transform('mean')
    r,p=pearsonr(x.dc,x.bc)
    ax[1].scatter(x.dc,x.bc,s=13,color=CL[f],alpha=0.55,edgecolor='none',
                  label=f'{NM[f]}  $r$={r:+.2f}')
    z=np.polyfit(x.dc,x.bc,1); xs=np.linspace(x.dc.min(),x.dc.max(),50)
    ax[1].plot(xs,np.polyval(z,xs),color=CL[f],lw=1.6)
ax[1].axhline(0,color='#333',lw=0.6); ax[1].axvline(0,color='#333',lw=0.6)
ax[1].set_xlabel('Drift, centred within target country')
ax[1].set_ylabel('Transfer accuracy, centred (pp)'); ax[1].legend(frameon=False,fontsize=7)
ax[1].set_title('b  Survives target-country fixed effects',loc='left',fontsize=8.5,fontweight='bold')
# c: AE vs S2 paired drift
pa=D[D.feat=='AE'].set_index(['src','tgt']).drift
ps=D[D.feat=='S2'].set_index(['src','tgt']).drift
j=pd.concat([pa.rename('AE'),ps.rename('S2')],axis=1).dropna()
w,p=wilcoxon(j.AE,j.S2)
ax[2].scatter(j.S2,j.AE,s=15,color='#5A6B7A',alpha=0.6,edgecolor='none')
lim=[0,max(j.AE.max(),j.S2.max())*1.05]
ax[2].plot(lim,lim,'--',color='#333',lw=0.8)
ax[2].set_xlim(lim); ax[2].set_ylim(lim)
ax[2].set_xlabel('Sentinel-2 drift'); ax[2].set_ylabel('AlphaEarth drift')
ax[2].set_title('c  No difference between representations',loc='left',fontsize=8.5,fontweight='bold')
ax[2].text(0.04,0.93,f'paired Wilcoxon $p$ = {p:.2f}\n$n$ = {len(j)} source–target pairs\nmean {j.AE.mean():.3f} vs {j.S2.mean():.3f}',
    transform=ax[2].transAxes,fontsize=6.6,va='top',color='#333')
plt.tight_layout()
stamp_multi(fig,['within_country_cv','pairwise'],x=0.0,y=0.0,fontsize=7.4,panel_letters=False,
    extra='A matched pair from the 1-country training row of the protocol 2×2: one single-source model is explained on held-out source parcels and on each other country.')
for e in ['pdf','png']: fig.savefig(os.path.join(OUT,f'fig_explanation_drift.{e}'))
print('saved fig_explanation_drift')
