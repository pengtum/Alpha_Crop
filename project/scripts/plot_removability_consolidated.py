"""Single corrected figure replacing three overlapping ones. Transfer effect is shown
as the per-country median with the inter-quartile range, because the mean is distorted
by one country (Slovakia) and neither feature set improves significantly."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from protocol_labels import stamp, stamp_multi, NAME
import json, numpy as np, matplotlib, os
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
plt.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42,'font.family':'Arial','font.size':9,
    'axes.spines.right':False,'axes.spines.top':False,'axes.linewidth':0.8,
    'savefig.bbox':'tight','savefig.pad_inches':0.08,'figure.dpi':200})
B='results/experiments/exp8_debias_test'
arm=json.load(open(f'{B}/crop_ranked_arm.json')); leak=json.load(open(f'{B}/country_leakage_vs_K.json'))
sw=json.load(open(f'{B}/sweep.json'))
C=['AT','BE','DE','EE','LT','LV','PT','RO','SE','SI','SK']
R={}
AB={'AlphaEarth':'AE','Sentinel-2':'S2','AE':'AE','S2':'S2'}
for v in arm.values(): R.setdefault(AB[v['feat']],{}).setdefault(v['K'],{})[v['rank']]=v['crop_id_ba']
L={}
for v in leak.values(): L.setdefault(AB[v['feat']],{})[v['K']]=v['country_ba']
T={}
for v in sw.values():
    if v['mode']=='top': T.setdefault(v['feat'],{})[v['K']]=v['per_country']
CL={'AE':'#3B6CA8','S2':'#E08A3C'}; NM={'AE':'AlphaEarth','S2':'Sentinel-2'}
RC={'random':'#9AA0A6','country':'#B0483B','crop':'#3F7F5F'}
RL={'random':'random $K$','country':'top-$K$ country-predictive','crop':'top-$K$ crop-predictive'}
fig,ax=plt.subplots(2,3,figsize=(12.6,6.6))
# a,b: three reference arms
for j,f in enumerate(['AE','S2']):
    A=ax[0,j]; b=R[f][0]['crop']; Ks=sorted(k for k in R[f] if k>0)
    for rk in ['random','country','crop']:
        kk=[k for k in Ks if rk in R[f][k]]
        A.plot(kk,[R[f][k][rk]-b for k in kk],'-o',color=RC[rk],ms=3,lw=1.5,label=RL[rk])
    A.axhline(0,color='#333',lw=0.6); A.set_ylim(-33,2)
    A.set_xlabel('$K$ dimensions removed'); A.set_ylabel('Change in crop accuracy (pp)')
    A.set_title(f'{"ab"[j]}  {NM[f]}',loc='left',fontsize=10,fontweight='bold')
    if j==0: A.legend(frameon=False,fontsize=7.5,loc='lower left')
# c: entanglement index
A=ax[0,2]
for f in ['AE','S2']:
    b=R[f][0]['crop']; xs=[];ys=[]
    for k in sorted(R[f]):
        if k<8: continue
        r=R[f][k]
        if not all(z in r for z in ['crop','country','random']): continue
        dr=r['random']-b; dk=r['country']-b; dc=r['crop']-b
        if abs(dc-dr)<1e-6: continue
        xs.append(k); ys.append((dk-dr)/(dc-dr))
    A.plot(xs,ys,'-o',color=CL[f],ms=3,lw=1.6,label=NM[f])
A.axhline(1,color='#777',ls='--',lw=0.8); A.axhline(0,color='#777',ls='--',lw=0.8)
A.text(50,1.06,'fully entangled',fontsize=7,color='#555',style='italic')
A.text(50,0.05,'separable',fontsize=7,color='#555',style='italic')
A.set_ylim(-0.15,1.7); A.set_xlabel('$K$ dimensions removed'); A.set_ylabel('Entanglement index')
A.legend(frameon=False,fontsize=8,loc='center right')
A.set_title('c  Are the geographic dimensions also the crop dimensions?',loc='left',fontsize=10,fontweight='bold')
# d: country predictability (redundancy) -- shown ONCE
A=ax[1,0]
for f in ['AE','S2']:
    ks=sorted(L[f]); A.plot(ks,[L[f][k] for k in ks],'-o',color=CL[f],ms=3,lw=1.6,label=NM[f])
A.set_xlabel('$K$ country-predictive dimensions removed'); A.set_ylabel('Country predictability (%)')
A.legend(frameon=False,fontsize=8)
A.set_title('d  Is the geography actually removed?',loc='left',fontsize=10,fontweight='bold')
A.annotate('AlphaEarth barely drops:\ngeography is redundantly encoded',(24,91.4),xytext=(3,66),
    fontsize=7,color='#2A4A72',style='italic',arrowprops=dict(arrowstyle='->',color='#2A4A72',lw=0.7))
# e: transfer effect -- MEDIAN with IQR (mean is Slovakia-driven)
A=ax[1,1]
for f in ['AE','S2']:
    ks=sorted(T[f]); med=[];lo=[];hi=[]
    for k in ks:
        d=np.array([(T[f][k][c]-T[f][0][c])*100 for c in C])
        med.append(np.median(d)); lo.append(np.percentile(d,25)); hi.append(np.percentile(d,75))
    A.plot(ks,med,'-o',color=CL[f],ms=3,lw=1.6,label=NM[f])
    A.fill_between(ks,lo,hi,color=CL[f],alpha=0.16,lw=0)
A.axhline(0,color='#333',lw=0.7)
A.set_xlabel('$K$ country-predictive dimensions removed')
A.set_ylabel('Change in cross-border accuracy (pp)\nmedian over 11 countries, IQR shaded')
A.legend(frameon=False,fontsize=8,loc='lower left')
A.set_title('e  Does removing them help transfer?  No.',loc='left',fontsize=10,fontweight='bold')
A.text(0.97,0.95,'no $K$ is significant for either\nfeature set (Wilcoxon, $n$=11)',transform=A.transAxes,
       ha='right',va='top',fontsize=7,color='#555',style='italic')
# f: why the mean misleads -- per-country at K=10
A=ax[1,2]
k=10
for i,f in enumerate(['AE','S2']):
    d=np.array([(T[f][k][c]-T[f][0][c])*100 for c in C])
    A.scatter(np.full(11,i)+np.random.RandomState(0).uniform(-.09,.09,11),d,s=34,
              color=CL[f],alpha=.8,edgecolor='white',linewidth=.6,zorder=3)
    A.hlines(np.median(d),i-.22,i+.22,color='#333',lw=2,zorder=4)
    A.hlines(d.mean(),i-.22,i+.22,color='#B0483B',lw=1.6,ls='--',zorder=4)
    j=int(np.argmax(d))
    if d[j]>5: A.annotate(C[j],(i,d[j]),xytext=(9,-3),textcoords='offset points',fontsize=8,color='#B0483B')
A.axhline(0,color='#333',lw=0.6); A.set_xticks([0,1]); A.set_xticklabels(['AlphaEarth','Sentinel-2'])
A.set_ylabel('Change at $K$=10 (pp)')
A.set_title('f  Why the mean misleads',loc='left',fontsize=10,fontweight='bold')
A.text(0.03,0.96,'— median      - - mean',transform=A.transAxes,fontsize=7.5,va='top',color='#444')
plt.tight_layout()
stamp(fig,'probe',x=0.0,y=0.0,fontsize=7.4,
      extra='Crop-classification accuracy is not measured in this figure.')
for e in ['pdf','png']:
    fig.savefig(f'SLIDES_FIGURES/S08_removability_CORRECTED.{e}')
    fig.savefig(f'Manuscript/Paper_Alpha_Earth/0001_draft_v1/figures/fig_removability.{e}')
print('saved S08_removability_CORRECTED')
