"""The one figure that shows all four evaluation protocols together.

Core conclusion:
    The four protocols are not four alternative answers to one question. They
    form a 2x2 over (how many countries the model was trained on) x (whether the
    test country was in that training set), and that 2x2 separates two effects
    which a single "in-distribution vs out-of-distribution" contrast confounds:
    crossing a border costs ~26 pp, while widening training from one country to
    ten buys ~13 pp. The border cost is measured twice, independently, at each
    training scale, and the two estimates agree within 1.5-2.6 pp.

Every other result figure shows exactly ONE cell of this 2x2 and says which.

Panel map
    a (hero)  the 2x2 itself: four cells, three feature sets per cell, with the
              horizontal (border) and vertical (training breadth) contrasts.
    b         the two independent border-cost estimates per feature set, which
              a states as numbers but cannot show as agreement.
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from protocol_labels import PROTOCOL
import json
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import FancyArrowPatch
import matplotlib.patheffects as pe

mpl.rcParams.update({
    'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],
    'svg.fonttype':'none','pdf.fonttype':42,'ps.fonttype':42,'font.size':7,
    'axes.spines.right':False,'axes.spines.top':False,'axes.linewidth':0.7,
    'xtick.major.width':0.7,'ytick.major.width':0.7,'legend.frameon':False,
})

C=['AT','BE','DE','EE','LT','LV','PT','RO','SE','SI','SK']; R='results/experiments'
e1=json.load(open(f'{R}/exp1_id_baseline/tier_b_nocoords_2021.json'))
e0=json.load(open(f'{R}/exp0_per_country_cv/tier_b_nocoords_2021.json'))
e2=json.load(open(f'{R}/exp2_loco/tier_b_nocoords_2021.json'))
e2b=json.load(open(f'{R}/exp2b_country_transfer/tier_b_nocoords_2021.json'))
F=[('ae','AlphaEarth'),('s2','Sentinel-2'),('fusion','Fusion')]
CL={'ae':'#0F4D92','s2':'#E08A3C','fusion':'#4F9D5B'}
INK='#272727'; MID='#767676'; LITE='#CFCECE'

def pairs(f):
    tm=e2b[f]['transfer_matrix']
    return np.array([tm[s][t]['balanced_accuracy']*100 for s in C for t in C
                     if s!=t and t in tm.get(s,{})])
POOL=json.load(open(f'{R}/exp1_id_baseline/percountry_tier_b_nocoords_2021.json'))
V={f:{'pooled':np.array([np.mean([POOL[f'{f}|{r}']['per_country'][c]['balanced_accuracy']
                                  for r in range(5)]) for c in C]),
      'within':np.array([e0[f][c]['balanced_accuracy']['mean'] for c in C])*100,
      'loco'  :np.array([e2[f]['per_country'][c]['balanced_accuracy'] for c in C])*100,
      'pair'  :pairs(f),
      # per-SOURCE-country mean, so the narrow-row border cost pairs 11 with 11
      'pair_by_src':np.array([np.mean([e2b[f]['transfer_matrix'][c][t]['balanced_accuracy']*100
                                       for t in e2b[f]['transfer_matrix'][c]]) for c in C])}
   for f,_ in F}

CELL={('narrow','seen'):('within','within_country_cv','11 countries'),
      ('narrow','unseen'):('pair','pairwise','11 source countries'),
      ('broad','seen'):('pooled','pooled_cv','11 countries'),
      ('broad','unseen'):('loco','loco','11 held-out countries')}

fig=plt.figure(figsize=(7.20,5.75))
outer=GridSpec(1,2,width_ratios=[2.02,1.0],wspace=0.40,figure=fig,
               left=0.115,right=0.975,top=0.832,bottom=0.175)
gl=GridSpecFromSubplotSpec(2,2,subplot_spec=outer[0],hspace=0.62,wspace=0.14)

axes={}
for r,br in enumerate(['narrow','broad']):
    for c_,col in enumerate(['seen','unseen']):
        ax=fig.add_subplot(gl[r,c_]); axes[(br,col)]=ax
        key,pkey,unit=CELL[(br,col)]
        for i,(f,fn) in enumerate(F):
            v=V[f][key]; m=v.mean()
            ax.bar(i,m,0.66,color=CL[f],edgecolor='none',zorder=2)
            if len(v)>1:
                rng=np.random.RandomState(42+i)
                x=i+(rng.rand(len(v))-0.5)*0.34
                big=len(v)>20
                ax.scatter(x,v,s=3.4 if big else 9,facecolor='white',zorder=3,
                           alpha=0.75 if big else 0.95,
                           edgecolor='#3A3A3A',linewidth=0.22 if big else 0.45)
            ax.text(i,m-3.0,f'{m:.1f}',ha='center',va='top',fontsize=6.8,
                    fontweight='bold',color='white',zorder=6,
                    path_effects=[pe.withStroke(linewidth=1.9,foreground=CL[f])])
        ax.set_xticks(range(3)); ax.set_xticklabels([fn for _,fn in F],fontsize=6.3)
        ax.set_ylim(0,84); ax.set_yticks([0,20,40,60,80]); ax.tick_params(labelsize=6.3)
        if c_==1: ax.set_yticklabels([])
        else: ax.set_ylabel('Balanced accuracy (%)',fontsize=6.8)
        ax.set_title(PROTOCOL[pkey]['name'],loc='left',fontsize=7.2,fontweight='bold',pad=8)
        ax.text(0,1.015,f'$n$ = {unit}',transform=ax.transAxes,fontsize=6.0,
                color=MID,style='italic')
        ax.grid(axis='y',color='#EEE',lw=0.5); ax.set_axisbelow(True)

fig.canvas.draw()
def bbox(k): return axes[k].get_position()

# column / row headers
b00,b01=bbox(('narrow','seen')),bbox(('narrow','unseen'))
b10=bbox(('broad','seen'))
fig.text((b00.x0+b00.x1)/2,0.902,'test country WAS in training',ha='center',
         fontsize=7.8,fontweight='bold',color=INK)
fig.text((b01.x0+b01.x1)/2,0.902,'test country was NEVER in training',ha='center',
         fontsize=7.8,fontweight='bold',color=INK)
fig.text(0.028,(b00.y0+b00.y1)/2,'trained on\n1 country',rotation=90,va='center',
         ha='center',fontsize=7.8,fontweight='bold',color=INK,linespacing=1.35)
fig.text(0.028,(b10.y0+b10.y1)/2,'trained on\n10–11 countries',rotation=90,va='center',
         ha='center',fontsize=7.8,fontweight='bold',color=INK,linespacing=1.35)

# contrast arrows
def arrow(x0,y0,x1,y1,col):
    fig.add_artist(FancyArrowPatch((x0,y0),(x1,y1),transform=fig.transFigure,
        arrowstyle='-|>',color=col,lw=1.0,mutation_scale=8,zorder=5))
RED='#B64342'; BLU='#0F4D92'
for br in ['narrow','broad']:
    a,b=bbox((br,'seen')),bbox((br,'unseen'))
    y=a.y0+0.55*(a.y1-a.y0)
    arrow(a.x1+0.003,y,b.x0-0.003,y,RED)          # seen -> unseen, points right
    d=np.mean([V[f][CELL[(br,'seen')][0]].mean()-V[f][CELL[(br,'unseen')][0]].mean() for f,_ in F])
    fig.text((a.x1+b.x0)/2,y+0.011,f'−{d:.1f} pp',ha='center',fontsize=6.6,
             color=RED,fontweight='bold')
for col in ['seen','unseen']:
    a,b=bbox(('narrow',col)),bbox(('broad',col))   # a = top row (1 country)
    gap=a.y0-b.y1
    x=a.x0+0.86*(a.x1-a.x0)
    arrow(x,a.y0-0.50*gap,x,b.y1+0.06*gap,BLU)     # 1 country -> 10-11, points DOWN
    d=np.mean([V[f][CELL[('broad',col)][0]].mean()-V[f][CELL[('narrow',col)][0]].mean() for f,_ in F])
    fig.text(x+0.007,b.y1+0.26*gap,f'{d:+.1f} pp',ha='left',va='center',fontsize=6.6,
             color=BLU,fontweight='bold')
fig.text(b00.x0,0.944,'red — cost of crossing a border, at fixed training breadth          '
         'blue — gain from widening training, at fixed distribution shift',
         fontsize=6.4,color=MID,style='italic')

# ---------------------------------------------------------------- b
axb=fig.add_subplot(outer[1])
y=np.arange(len(F))[::-1]
from scipy.stats import wilcoxon
for i,(f,fn) in enumerate(F):
    broad =V[f]['pooled']-V[f]['loco']
    narrow=V[f]['within']-V[f]['pair_by_src']
    axb.plot([broad.mean(),narrow.mean()],[y[i],y[i]],color=LITE,lw=2.6,zorder=1,
             solid_capstyle='round')
    axb.scatter([broad.mean()],[y[i]],s=48,color=CL[f],zorder=3,edgecolor='white',lw=0.8)
    axb.scatter([narrow.mean()],[y[i]],s=48,color=CL[f],zorder=3,edgecolor='white',lw=0.8,marker='D')
    p=wilcoxon(narrow,broad).pvalue
    axb.text(broad.mean(),y[i]+0.17,f'{broad.mean():.1f}',ha='center',fontsize=6.2,color=MID)
    axb.text(narrow.mean(),y[i]+0.17,f'{narrow.mean():.1f}',ha='center',fontsize=6.2,color=MID)
    axb.text((broad.mean()+narrow.mean())/2,y[i]-0.34,
             f'+{(narrow-broad).mean():.1f} pp, $p$ = {p:.3f}',ha='center',fontsize=6.0,color=INK)
axb.set_yticks(y); axb.set_yticklabels([fn for _,fn in F],fontsize=6.8)
axb.set_xlim(6,34); axb.set_xlabel('Cost of crossing a border (pp)',fontsize=6.8)
axb.tick_params(labelsize=6.3); axb.set_ylim(-0.75,2.75)
axb.grid(axis='x',color='#EEE',lw=0.5); axb.set_axisbelow(True)
axb.spines['left'].set_visible(False); axb.tick_params(axis='y',length=0)
axb.set_title('b   A border costs more than twice\n      as much with one training country',
              loc='left',fontsize=7.6,fontweight='bold',pad=58)
from matplotlib.lines import Line2D
axb.legend(handles=[
    Line2D([],[],marker='o',ls='',ms=5.2,color=MID,label='trained on 10–11 countries\n(pooled CV − LOCO)'),
    Line2D([],[],marker='D',ls='',ms=4.8,color=MID,label='trained on 1 country\n(within-country CV − pairwise)')],
    loc='lower left',bbox_to_anchor=(-0.02,1.005),fontsize=6.0,labelspacing=0.6,
    handletextpad=0.5,borderaxespad=0)
axes[('narrow','seen')].set_title('Within-country cross-validation',loc='left',
                                  fontsize=7.2,fontweight='bold',pad=8)
fig.text(0.115,0.968,'a   Four protocols, one 2×2: what changes between them',
         fontsize=8.6,fontweight='bold',color=INK,ha='left')

fig.text(0.115,0.072,
 'Every cell is balanced accuracy computed per country and then averaged over the eleven countries, so all four are on the same footing; white points are the individual\n'
 'countries. Reporting a globally pooled balanced accuracy against a per-country-averaged one inflates the apparent gap by roughly 14 pp, which is why the pooled cell here\n'
 'is not the 68.5% quoted for the pooled model as a whole. Balanced accuracy over the 10 Tier-B classes, LightGBM, no coordinate features. Diagonal comparisons between\n'
 'cells change both factors at once and are not clean estimates of either.',
 fontsize=6.1,color=MID,style='italic',va='top',linespacing=1.5)

for ext in ['pdf','png','svg']:
    fig.savefig(f'Manuscript/Paper_Alpha_Earth/0001_draft_v1/figures/fig_protocol_ladder.{ext}',
                dpi=600,bbox_inches='tight')
fig.savefig('SLIDES_FIGURES/S17_protocol_ladder.pdf',bbox_inches='tight')
fig.savefig('SLIDES_FIGURES/S17_protocol_ladder.png',dpi=300,bbox_inches='tight')
plt.close(fig); print('saved S17_protocol_ladder')
