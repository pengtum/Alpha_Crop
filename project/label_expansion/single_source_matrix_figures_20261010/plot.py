"""完整单源国家配对矩阵；科研绘图，无抽样、插补或显著性标记。"""
from pathlib import Path
import sys,json
import numpy as np,pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap,Normalize
PROJECT=Path(__file__).resolve().parents[2]
STYLE=PROJECT/'label_expansion/storyline/rse_figs'
# 只继承项目的配色和排版，统计映射针对本实验重新实现。
sys.path.insert(0,str(STYLE));import nstyle as ns
from audit_panel_alignment import require_matplotlib_panel_alignment
OUT=PROJECT/'results/single_source_20261009/matrix_figures_20261010'
DATA=PROJECT/'results/single_source_20261009/analysis_repaired_20261010'
COUNTRIES=['AT','BE','DE','EE','LT','LV','PT','RO','SE','SI','SK']
ns.setup();mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],'font.size':6.5,'svg.fonttype':'none','pdf.fonttype':42})
SEQ=LinearSegmentedColormap.from_list('Accuracy',['#f5f8fa','#a7cad7','#397b99','#154b65'])
DIV=LinearSegmentedColormap.from_list('Paired difference',['#a35441','#dfb7ab','#fafafa','#a5d0d3','#236b76'])
raw=pd.read_csv(DATA/'default_metrics.csv');raw=raw[(raw.source!='MULTI5K')&(raw.learner=='lgb')&(raw.arm=='Z0')]
assert len(raw)==440 and set(raw.source)==set(COUNTRIES)
absolute=raw.groupby(['source','target','feat'],as_index=False).agg(OA=('OA','mean'),BA=('BA','mean'),n_units=('unit','nunique'))
assert (absolute.n_units==2).all();absolute.to_csv(OUT/'absolute_source_data.csv',index=False)
u=pd.read_csv(DATA/'unit_rows.csv');u=u[(u.level=='D')&(u.source!='MULTI5K')].copy();u['arm']=u.name.str.removeprefix('D_')
keys=['learner','arm','source','target']
units=u.groupby(keys+['unit'],as_index=False).agg(value=('value','mean'),n_valid=('value','count'))
paired=units.groupby(keys,as_index=False).agg(value=('value','mean'),n_units=('value','count'),n_valid=('n_valid','sum'))
paired['n_expected']=np.where(paired.arm=='Z0',2,6)
paired.to_csv(OUT/'matched_oa_source_data.csv',index=False)
ref=pd.read_csv(DATA/'pair_effects.csv');ref=ref[ref.source!='MULTI5K'].sort_values(keys)
assert np.allclose(paired.sort_values(keys).value,ref.value,equal_nan=True)

def matrix(ax,series,cmap,norm,title,letter,support=None):
    mat=series.unstack('target').reindex(index=COUNTRIES,columns=COUNTRIES).to_numpy()
    ax.set_facecolor('#eeeeee');im=ax.imshow(np.ma.masked_invalid(mat),cmap=cmap,norm=norm,interpolation='none')
    ax.set(xticks=range(11),yticks=range(11),xticklabels=COUNTRIES,yticklabels=COUNTRIES,xlabel='Target country',ylabel='Source country')
    ax.tick_params(length=0,pad=3,labelsize=6)
    for spine in ax.spines.values():spine.set_visible(False)
    ax.set_title(title,loc='left',pad=10,fontsize=7)
    ax.annotate(letter,xy=(0,1),xycoords='axes fraction',xytext=(-20,9),textcoords='offset points',fontsize=8,fontweight='bold',ha='left',va='bottom')
    for i,src in enumerate(COUNTRIES):
        for j,tgt in enumerate(COUNTRIES):
            v=mat[i,j]
            if src==tgt:txt='—';color='#8b8b8b'
            elif not np.isfinite(v):txt='×';color='#6b6b6b'
            else:
                txt=f'{v:.0f}' if support is None else ('0.0' if abs(v)<.05 else f'{v:+.1f}')
                rgb=cmap(norm(v))[:3];lum=sum(a*b for a,b in zip(rgb,[.2126,.7152,.0722]));color='white' if lum<.52 else '#252525'
            ax.text(j,i,txt,ha='center',va='center',fontsize=5.5,color=color)
            if support is not None and src!=tgt and np.isfinite(v):
                s=support.loc[(src,tgt)]
                if s.n_valid<s.n_expected:ax.plot(j+.36,i-.34,marker='o',markersize=1.3,markeredgewidth=0,color='#252525')
    return im

def save(fig,axes,caxes,name):
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,axes=list(axes),panel_ids=list('abcd'),exclude_axes=caxes,json_out=OUT/f'{name}.alignment.json',overlay_svg=OUT/f'{name}.alignment.svg',require_panel_labels=True,strict=True)
    fig.savefig(OUT/f'{name}.pdf')
    fig.savefig(OUT/f'{name}.svg')
    fig.savefig(OUT/f'{name}.png',dpi=300)
    fig.savefig(OUT/f'{name}.tiff',dpi=600,pil_kwargs={'compression':'tiff_lzw'})
    plt.close(fig)

def canvas():
    fig,axs=plt.subplots(2,2,figsize=(7.20472440945,7.48031496063))
    fig.subplots_adjust(left=.095,right=.98,top=.935,bottom=.14,wspace=.28,hspace=.36)
    return fig,axs.ravel()
fig,axs=canvas();plans=[('ae','OA'),('s2','OA'),('ae','BA'),('s2','BA')]
for ax,(feat,metric),letter in zip(axs,plans,'abcd'):
    series=absolute[absolute.feat==feat].set_index(['source','target'])[metric]
    im=matrix(ax,series,SEQ,Normalize(0,100),f'{feat.upper()} · {metric}',letter)
cax=fig.add_axes([.28,.065,.5,.013]);cb=fig.colorbar(im,cax=cax,orientation='horizontal',ticks=[0,25,50,75,100]);cb.set_label('Default-decision accuracy (%)',labelpad=3);cb.outline.set_visible(False)
fig.text(.095,.017,'LightGBM · 5,000 source labels · no target labels · mean of two spatial splits; — self-transfer not run',fontsize=5.5)
save(fig,axs,[cax],'01_zero_label_transfer_accuracy')
fig,axs=canvas();plans=[('lgb','Z0','No target labels · LightGBM'),('rf','Z0','No target labels · Random forest'),('lgb','RT200','200 random target labels · LightGBM'),('lgb','B_SRS100','100 existing + 100 random · LightGBM')];summary=[]
for ax,(learner,arm,title),letter in zip(axs,plans,'abcd'):
    g=paired[(paired.learner==learner)&(paired.arm==arm)].set_index(['source','target'])
    im=matrix(ax,g.value,DIV,Normalize(-30,30),title,letter,support=g)
    summary.append({'panel':letter,'learner':learner,'arm':arm,'valid_pairs':int(g.value.notna().sum()),'positive_pairs':int((g.value>0).sum()),'missing_pairs':int(g.value.isna().sum()),'partial_pairs':int(((g.n_valid<g.n_expected)&g.value.notna()).sum())})
cax=fig.add_axes([.28,.065,.5,.013]);cb=fig.colorbar(im,cax=cax,orientation='horizontal',ticks=[-30,-15,0,15,30]);cb.set_label('S2 higher ← BA difference at matched OA (pp) → AE higher',labelpad=3);cb.outline.set_visible(False)
fig.text(.095,.017,'— self-transfer not run; × insufficient common OA; small dot: incomplete split/draw support. No significance symbols.',fontsize=5.5)
save(fig,axs,[cax],'02_paired_representation_advantage')
(OUT/'panel_counts.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
