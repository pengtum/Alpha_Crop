"""独立分析摘要图：延续现有论文样式，区分同OA比较与可估计性。"""
from pathlib import Path
import sys
import numpy as np,pandas as pd
PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT/'label_expansion/storyline/rse_figs'))
import nstyle as ns
import matplotlib.pyplot as plt
OUT=PROJECT/'results/single_source_20261009/analysis_repaired_20261010'
SOURCE=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else OUT
ns.setup();s=pd.read_csv(SOURCE/'representation_summary.csv');s=s[s.target_set=='all11']
keys=[('lgb','Z0'),('rf','Z0'),('lgb','RT200'),('lgb','B_SRS100')]
labels=['No target labels · LGB','No target labels · RF','200 random labels · LGB','100 existing + 100 random · LGB']
fig,axes=plt.subplots(1,2,figsize=(ns.W2,78*ns.MM),gridspec_kw={'width_ratios':[1.1,1]})
ax=axes[0]
for i,(l,a) in enumerate(keys):
    z=s[(s.learner==l)&(s.arm==a)].iloc[0];v=z.mean_ae_minus_s2;lo=z.reweighting_lo;hi=z.reweighting_hi
    ax.errorbar(v,i,xerr=[[v-lo],[hi-v]],fmt='o',color=ns.C['AE'],capsize=2,lw=1)
ax.set(yticks=range(4),yticklabels=labels,xlabel='AE − S2 at matched OA (BA, pp)',title='Representation advantage',xlim=(-2,8),ylim=(3.5,-.6))
ns.zero_line(ax,axis='x');ns.panel(ax,'a',x=-.05)
e=pd.read_csv(SOURCE/'label_equivalents.csv');e=e[e.source!='MULTI5K'];es=e.groupby(['k_ae','status']).size().unstack(fill_value=0)
ax=axes[1];left=np.zeros(len(es));statuses=['already_met_at_zero','interpolated','not_reached_no_extrapolation','insufficient_common_oa']
colors=[ns.C['S2'],ns.C['AE'],ns.C['CONC'],ns.C['LIGHT']]
legend=['S2 already meets at 0','Interpolated within 0–400','S2 does not reach by 400','Insufficient common OA']
for status,color,label in zip(statuses,colors,legend):
    widths=100*es[status].to_numpy()/es.sum(axis=1).to_numpy();ax.barh(range(len(es)),widths,left=left,color=color,height=.58,label=label);left+=widths
ax.set(yticks=range(4),yticklabels=[str(k) for k in es.index],ylabel='AE target labels',xlabel='Share of 660 unit × draw comparisons (%)',title='Can label equivalence be estimated?',xlim=(0,100),ylim=(3.5,-.6))
ax.legend(loc='upper center',bbox_to_anchor=(.48,-.26),ncol=2,fontsize=5.5,columnspacing=.8,handlelength=1.2);ns.panel(ax,'b',x=-.16)
fig.subplots_adjust(left=.26,right=.99,top=.86,bottom=.29,wspace=.55)
ns.save(fig,'single_source_evidence_summary',str(OUT/'figures'))
(OUT/'FIGURE_CAPTIONS.txt').write_text('single_source_evidence_summary: Panel a shows mean AE minus S2 balanced accuracy at matched overall accuracy. Countries are weighted equally, with source countries averaged within target countries and repetitions within spatial units. Bars are 95% country-reweighting sensitivity ranges, not confidence intervals from 110 independent transfers. Available pairs: LGB Z0 98/110; RF Z0 104/110; LGB RT200 and B_SRS100 95/110. Panel b records the feasibility of frozen label-equivalence interpolation across 220 single-source spatial units and three draws (660 dependent comparisons per budget). Curves are not forced monotone and no extrapolation is performed; 311/411 feasible S2 curves are nonmonotone.\n\nsingle_source_zero_label_matrix: Default-decision LGB balanced accuracy for each directed country pair, averaged over the two spatial units. The difference panel compares default BA, not matched-OA BA. Diagonal cells are not part of the transfer experiment.\n\nsingle_source_label_curves: Default-decision LGB BA versus target-label budget. Thin lines represent source countries; thick lines average the ten sources for each target, with spatial units and repeated draws averaged. Curves are descriptive and are not constrained to increase.\n')
