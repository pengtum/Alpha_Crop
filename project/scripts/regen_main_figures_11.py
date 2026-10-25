"""Regenerate main-text figures from the 11-country 2021-aligned results (Nature style:
no embedded chatty titles, sentence-case labels, restrained palette). Outputs to /tmp
for review; installer step copies approved PDFs into the manuscript figures/ folder."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from protocol_labels import stamp, stamp_multi
import json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42,'font.family':'Arial','font.size':8,
    'axes.spines.right':False,'axes.spines.top':False,'axes.linewidth':0.7,
    'savefig.bbox':'tight','savefig.pad_inches':0.06,'figure.dpi':150})
R='results/experiments'
e1=json.load(open(f'{R}/exp1_id_baseline/tier_b_nocoords_2021.json'))
e2=json.load(open(f'{R}/exp2_loco/tier_b_nocoords_2021.json'))
e2b=json.load(open(f'{R}/exp2b_country_transfer/tier_b_nocoords_2021.json'))
FEAT=['ae','s2','fusion']; FNAME={'ae':'AlphaEarth','s2':'Sentinel-2','fusion':'Fusion'}
COL={'ae':'#3B6CA8','s2':'#E08A3C','fusion':'#4F9D5B'}
ZONE={'PT':'med','SI':'med','BE':'atl','DE':'cont','AT':'cont','SK':'cont',
      'SE':'nord','EE':'nord','LV':'nord','LT':'nord','RO':'se'}
ZCOL={'atl':'#6BAED6','cont':'#74C476','med':'#FD8D3C','nord':'#9E9AC8','se':'#E6550D'}
ZORDER=['atl','cont','med','nord','se']
CTRY=sorted(ZONE, key=lambda c:(ZORDER.index(ZONE[c]),c))  # group by climate zone

# ---------- Fig: ID vs OOD grouped bars ----------
fig,ax=plt.subplots(figsize=(3.4,2.6))
idv=[e1[f]['balanced_accuracy']['mean']*100 for f in FEAT]
ide=[e1[f]['balanced_accuracy']['std']*100 for f in FEAT]
oov=[e2[f]['mean_ba']*100 for f in FEAT]
ooe=[e2[f]['std_ba']*100 for f in FEAT]
x=np.arange(2); w=0.25
for i,f in enumerate(FEAT):
    ax.bar(x+(i-1)*w,[idv[i],oov[i]],w,yerr=[ide[i],ooe[i]],color=COL[f],
           label=FNAME[f],error_kw=dict(lw=0.7,capsize=2))
    for xi,v in zip(x+(i-1)*w,[idv[i],oov[i]]):
        ax.text(xi,v+1.2,f'{v:.1f}',ha='center',fontsize=6.3)
ax.set_xticks(x); ax.set_xticklabels(['In-distribution\n(pooled CV)','Cross-border\n(leave-one-country-out)'])
ax.set_ylabel('Balanced accuracy (%)'); ax.set_ylim(0,80)
ax.legend(frameon=False,fontsize=7,loc='upper right')
stamp_multi(fig,['pooled_cv','loco'],x=0.0,y=-0.02,fontsize=6.4,panel_letters=False,
    extra='A matched pair from the same training-breadth row of the protocol 2×2: both train on 10–11 countries, so the contrast isolates distribution shift.')
for d in ['SLIDES_FIGURES','Manuscript/Paper_Alpha_Earth/0001_draft_v1/figures']:
    fig.savefig(f'{d}/S03b_ID_vs_OOD_gap.pdf' if d=='SLIDES_FIGURES' else f'{d}/fig_id_vs_ood_gap.pdf')
fig.savefig('/tmp/regen_id_vs_ood_gap.png'); plt.close(fig)

# ---------- Fig: transfer matrix (fusion) ----------
def matrix(f):
    tm=e2b[f]['transfer_matrix']; M=np.full((11,11),np.nan)
    for i,s in enumerate(CTRY):
        for j,t in enumerate(CTRY):
            if t in tm.get(s,{}): M[i,j]=tm[s][t]['balanced_accuracy']*100
    return M
for f in FEAT:
    M=matrix(f)
    fig,ax=plt.subplots(figsize=(4.2,3.8))
    im=ax.imshow(M,cmap='YlGnBu',vmin=0,vmax=60)
    for i in range(11):
        for j in range(11):
            if not np.isnan(M[i,j]):
                ax.text(j,i,f'{M[i,j]:.0f}',ha='center',va='center',fontsize=6,
                        color='white' if M[i,j]>36 else '#222')
            else: ax.text(j,i,'–',ha='center',va='center',color='#bbb',fontsize=7)
    ax.set_xticks(range(11)); ax.set_yticks(range(11))
    ax.set_xticklabels(CTRY,fontsize=7); ax.set_yticklabels(CTRY,fontsize=7)
    ax.set_xlabel('Target country'); ax.set_ylabel('Source country')
    # climate-zone strips
    for j,c in enumerate(CTRY):
        ax.add_patch(plt.Rectangle((j-0.5,-1.4),1,0.7,color=ZCOL[ZONE[c]],clip_on=False))
        ax.add_patch(plt.Rectangle((-1.4,j-0.5),0.7,1,color=ZCOL[ZONE[c]],clip_on=False))
    cb=fig.colorbar(im,ax=ax,fraction=0.046,pad=0.04); cb.set_label('Balanced accuracy (%)',fontsize=7)
    cb.ax.tick_params(labelsize=6)
    stamp(fig,'pairwise',x=0.0,y=-0.02,fontsize=6.4,
          extra='Row = source country, column = target country; the diagonal is not defined.')
    tag={'ae':'S05a_matrix_AE','s2':'S05b_matrix_S2','fusion':'S05c_matrix_Fusion'}[f]
    fig.savefig(f'SLIDES_FIGURES/{tag}.pdf'); fig.savefig(f'Manuscript/Paper_Alpha_Earth/0001_draft_v1/figures/fig_transfer_matrix_{f}.pdf'); plt.close(fig)

# ---------- Fig: LOCO per-country ----------
fig,ax=plt.subplots(figsize=(5.2,2.6))
x=np.arange(11); w=0.26
for i,f in enumerate(FEAT):
    vals=[e2[f]['per_country'][c]['balanced_accuracy']*100 for c in CTRY]
    ax.bar(x+(i-1)*w,vals,w,color=COL[f],label=FNAME[f])
ax.set_xticks(x); ax.set_xticklabels(CTRY,fontsize=7)
ax.set_ylabel('Held-out balanced accuracy (%)'); ax.set_xlabel('Held-out country')
ax.set_ylim(0,70); ax.legend(frameon=False,fontsize=7,ncol=3,loc='upper center')
# climate zone strip under x
for j,c in enumerate(CTRY):
    ax.add_patch(plt.Rectangle((j-0.4,-7),0.8,3,color=ZCOL[ZONE[c]],clip_on=False))
stamp(fig,'loco',x=0.0,y=-0.10,fontsize=6.4)
fig.savefig('SLIDES_FIGURES/S04_LOCO_per_country.pdf'); fig.savefig('Manuscript/Paper_Alpha_Earth/0001_draft_v1/figures/fig_loco.pdf'); plt.close(fig)

# quick numeric sanity print
print('ID  BA:', {f:round(e1[f]['balanced_accuracy']['mean']*100,1) for f in FEAT})
print('OOD BA:', {f:round(e2[f]['mean_ba']*100,1) for f in FEAT})
print('country order (by zone):', CTRY)
print('saved /tmp/regen_{id_vs_ood_gap,transfer_matrix_fusion,loco_per_country}.png')
