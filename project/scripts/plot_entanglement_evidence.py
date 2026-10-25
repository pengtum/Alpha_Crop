"""Four-panel evidence that AlphaEarth's geographic signal is redundantly encoded and
inseparable from its crop signal, while Sentinel-2's is localised and removable."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from protocol_labels import stamp, stamp_multi, NAME
import json, numpy as np, matplotlib, os
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42,'font.family':'Arial','font.size':8,
    'axes.spines.right':False,'axes.spines.top':False,'axes.linewidth':0.7,
    'savefig.bbox':'tight','savefig.pad_inches':0.06,'figure.dpi':200})
B='results/experiments/exp8_debias_test'
OUT=os.path.join('Manuscript','Paper_Alpha_Earth','0001_draft_v1','figures')
arm=json.load(open(f'{B}/crop_ranked_arm.json'))
leak=json.load(open(f'{B}/country_leakage_vs_K.json'))
lc=json.load(open(f'{B}/probe_learning_curve.json'))
R={}
for v in arm.values(): R.setdefault(v['feat'],{}).setdefault(v['K'],{})[v['rank']]=v['crop_id_ba']
L={}
for v in leak.values(): L.setdefault(v['feat'],{})[v['K']]=v['country_ba']
C={'AlphaEarth':'#3B6CA8','Sentinel-2':'#E08A3C'}
RC={'random':'#9AA0A6','country':'#B0483B','crop':'#3F7F5F'}
RL={'random':'random $K$','country':'top-$K$ country-predictive','crop':'top-$K$ crop-predictive'}
fig,ax=plt.subplots(2,2,figsize=(7.6,6.0))
# a,b: three reference lines per feature set
for j,f in enumerate(['AlphaEarth','Sentinel-2']):
    A=ax[0,j]; b=R[f][0]['crop']; Ks=sorted(k for k in R[f] if k>0)
    for rank in ['random','country','crop']:
        v=[R[f][k][rank]-b for k in Ks if rank in R[f][k]]
        kk=[k for k in Ks if rank in R[f][k]]
        A.plot(kk,v,'-o',color=RC[rank],ms=3.2,lw=1.5,label=RL[rank])
    A.axhline(0,color='#333',lw=0.6); A.set_xlabel('$K$ dimensions removed')
    A.set_ylabel('Change in crop accuracy (pp)'); A.set_ylim(-33,2)
    A.set_title(f'{"ab"[j]}  {f}',loc='left',fontsize=8.5,fontweight='bold')
    if j==0: A.legend(frameon=False,fontsize=6.6,loc='lower left')
# c: country predictability vs K
A=ax[1,0]
for f in ['AlphaEarth','Sentinel-2']:
    Ks=sorted(L[f]); A.plot(Ks,[L[f][k] for k in Ks],'-o',color=C[f],ms=3.2,lw=1.6,label=f)
A.set_xlabel('$K$ country-predictive dimensions removed')
A.set_ylabel('Country predictability (%)'); A.legend(frameon=False,fontsize=7)
A.set_title('c  Is the geography actually removed?',loc='left',fontsize=8.5,fontweight='bold')
A.annotate('AlphaEarth barely drops:\ngeography is redundantly encoded',(24,91.4),
    xytext=(4,68),fontsize=6.5,color='#2A4A72',style='italic',
    arrowprops=dict(arrowstyle='->',color='#2A4A72',lw=0.7))
# d: entanglement index
A=ax[1,1]
for f in ['AlphaEarth','Sentinel-2']:
    b=R[f][0]['crop']; xs=[];ys=[]
    for k in sorted(R[f]):
        if k<8: continue
        r=R[f][k]
        if not all(x in r for x in ['crop','country','random']): continue
        dr=r['random']-b; dk=r['country']-b; dc=r['crop']-b
        if abs(dc-dr)<1e-6: continue
        xs.append(k); ys.append((dk-dr)/(dc-dr))
    A.plot(xs,ys,'-o',color=C[f],ms=3.2,lw=1.6,label=f)
A.axhline(1,color='#777',ls='--',lw=0.8); A.axhline(0,color='#777',ls='--',lw=0.8)
A.text(50,1.06,'fully entangled',fontsize=6.3,color='#555',style='italic')
A.text(50,0.04,'separable',fontsize=6.3,color='#555',style='italic')
A.set_xlabel('$K$ dimensions removed'); A.set_ylabel('Entanglement index')
A.set_ylim(-0.15,1.7); A.legend(frameon=False,fontsize=7,loc='center right')
A.set_title('d  Do the geographic dimensions carry the crop signal?',loc='left',fontsize=8.5,fontweight='bold')
plt.tight_layout()
stamp(fig,'probe',x=0.0,y=0.0,fontsize=7.4,
      extra='Crop-classification accuracy is not measured in this figure.')
for e in ['pdf','png']: fig.savefig(os.path.join(OUT,f'fig_entanglement_evidence.{e}'))
print('saved fig_entanglement_evidence')
