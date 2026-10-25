"""只调整已有图形对象的字号与布局，保留数据、区间和图元。"""
from pathlib import Path
import sys
import matplotlib.pyplot as plt
from matplotlib.text import Text
from matplotlib.lines import Line2D
MM = 1/25.4
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT.parent/'nature-skills-main/skills/nature-figure/scripts'))
from audit_panel_alignment import require_matplotlib_panel_alignment

def finalize(fig,name,base):
    for t in fig.findobj(Text):
        if t.get_visible() and t.get_text().strip():t.set_fontsize(max(7.5,t.get_fontsize()))
    aa=fig.axes
    ids=[]; rows=[]; cols=[]; exemptions=[]
    def size(w,h):fig.set_size_inches(w*MM,h*MM,forward=True)
    def pos(ax,x,y,w,h):
        W,H=fig.get_size_inches()/MM
        ax.set_position([x/W,y/H,w/W,h/H])
    def label(ax,l):
        for t in list(ax.texts):
            if t.get_text() in list('abcdef') and t.get_fontweight()=='bold':t.remove()
        ax.annotate(l,(0,1),xycoords='axes fraction',xytext=(0,8),textcoords='offset points',fontsize=10,fontweight='bold',va='bottom')
    if name=='fig2_evaluation':
        size(183,126)
        a,b,c=aa
        pos(a,14,70,75,46.5);pos(b,104,70,75,46.5);pos(c,61,12,67,40)
        for t in c.texts:
            tx=t.get_text()
            x,y=t.get_position()
            if tx in ['a','b','c']:continue
            if 'countries' in tx:t.set_position((-10.5,y));t.set_fontsize(8)
            elif tx.startswith('Δ'):t.set_position((-1.8,y))
            elif '[' in tx or tx=='mean [95% CI]':t.set_position((10.1,y))
            elif '/' in tx or tx=='> 0':t.set_position((16.4,y))
        for ax,l in zip(aa,'abc'):label(ax,l)
        ids=list('abc');rows=[['a','b']]
    elif name=='fig6_representation':
        size(183,192)
        a,b,c,d,e,f=aa
        coords=[(37,140,46,40),(113,140,61,40),(37,77,46,37),(113,77,61,37),(37,14,46,43),(114,14,34,43)]
        for ax,co in zip(aa,coords):pos(ax,*co)
        for t in list(fig.texts):t.remove()
        # 六面板按原字母顺序排成两列三行。
        for ax,l in zip(aa,'abcdef'):label(ax,l)
        for ax in (b,c):ax.set_xlabel('ΔBA (%)')
        e.set_xlabel('ΔBA interaction (%)');f.set_xlabel('ΔBA (%)')
        # 两行汇总统计兼作图例，保留原图的均值、置信区间和国家计数。
        for t in b.texts:
            if t.get_text().startswith('LightGBM '):t.set_position((.06,-.31))
            elif t.get_text().startswith('RF '):t.set_position((.06,-.42))
        for i,ln in enumerate([ln for ln in b.lines if ln.get_transform()==b.transAxes]):
            ln.set_ydata([-.31 if i==0 else -.42])
        for t in c.texts:
            if t.get_text()=='11 countries':t.set_position((.99,1.03))
        for ax in (a,c,e):
            for t in ax.texts:
                if hasattr(t,'xy') and t.get_text() not in list('abcdef') and t.xy[0] is not None:
                    x,y=t.xy
                    if isinstance(x,(int,float)) and abs(x)<1.0:
                        t.set_position((8 if x>=0 else -8,2.4));t.set_ha('left' if x>=0 else 'right')
        f.text(.22,1.14,'11 countries; 90% CI',transform=f.transAxes,fontsize=7.5,color='#767676')
        ids=list('abcdef');rows=[['a','b'],['c','d'],['e','f']]
    elif name=='fig4_existing_survey':
        size(183,132)
        a,b,c=aa
        pos(a,15,76,68,44);pos(b,125,76,48,44);pos(c,70,13,103,43)
        for t in c.texts:
            if t.get_text().startswith('vs '):t.set_position((-.56,t.get_position()[1]))
        for ax,l in zip(aa,'abc'):label(ax,l)
        ids=list('abc');rows=[['a','b']]
    elif name=='fig5_exchange_cost':
        size(183,142)
        a,b,c,d,e=aa
        pos(a,15,84,64,41);pos(b,109,84,65,41)
        for ax,x in zip([c,d,e],[15,72,129]):pos(ax,x,18,45,43)
        for leg in list(fig.legends):leg.remove()
        from nstyle import C
        hs=[Line2D([],[],color=C[k],marker='o',ms=3, label=s) for k,s in [('LABEL','Random labels'),('CONC','Survey / local extension'),('SPREAD','Survey + national spread'),('BETTI','Survey + regional allocation')]]
        hs.append(Line2D([],[],color='black',ls='',marker='*',label='Best at each budget'))
        fig.legend(handles=hs,loc='upper center',bbox_to_anchor=(.5,1.01),ncol=3,fontsize=7.5,columnspacing=1.2,handlelength=1.4)
        for t in list(fig.texts):t.remove()
        fig.text(.52,71/142,'c = cost per spread label / cost per concentrated label',ha='center',fontsize=7.5,color='#767676')
        for ax,l in zip([a,b,c],'abc'):label(ax,l)
        d.set_xlabel('Budget (cost units of one concentrated label)')
        ids=['a','b','c1','c2','c3'];rows=[['a','b'],['c1','c2','c3']]
    elif name=='fig3_labels_zero':
        # 第一、三幅图放大；中间条形图用更短标签释放绘图区。
        size(183,111)
        a,b,c=aa
        pos(a,30,61,56,40);pos(b,30,12,56,34);pos(c,107.5,33,75,68)
        c.set_ylim(-3,7.4);c.set_aspect('equal')
        c.set_xlabel('LightGBM ΔBA (%)');c.set_ylabel('Random forest ΔBA (%)')
        a.set_xticks([0,100,200,400])
        a.get_legend().set_bbox_to_anchor((0,-.01))
        a.get_legend()._loc=3
        for t in a.texts:
            if t.get_text().startswith('11 countries:'):t.set_position((320,-.4));t.set_va('top')
        # 图例放在两条附加国家曲线之间的空白区域。
        a.set_ylim(-8.5,7.2);a.get_legend().set_bbox_to_anchor((.02,.01))
        for t in c.texts:
            if t.get_text()=='y = x':t.set_position((-2.5,-1.5))
            if t.get_text()=='SK':t.set_position((4.17,3.35));t.set_ha('center');t.set_va('bottom')
        for t in list(b.texts):
            if t.get_text().startswith('share of'):t.set_text('Gap closed');t.set_position((6.2,4.05))
            if t.get_text().startswith('+16.'):
                t.set_position((.4,5.8));t.set_va('bottom');t.set_fontsize(7.5)
        b.set_yticklabels(['Target-only\n(~30k labels)']+[f'+{k} labels' for k in [50,100,200,400]])
        for t in c.texts:
            tx=t.get_text()
            if tx.startswith('+200 random labels;'):t.set_text('+200 labels; 9-country means')
        for ax,l in zip(aa,'abc'):label(ax,l)
        ids=list('abc');cols=[['a','b']]
    elif name=='fig_zero_label_accuracy':
        size(183,84)
        for ax,x in zip(aa,[12,73,134]):pos(ax,x,16,45,57)
        for ax,l in zip(aa,'abc'):label(ax,l)
        for t in list(fig.texts):t.remove()
        aa[0].set_xlabel('Overall accuracy (%)');aa[1].set_xlabel('Balanced accuracy (%)');aa[2].set_xlabel('ΔBA (%)')
        ids=list('abc');rows=[ids]
    elif name=='fig_zero_label_confusion':
        size(183,113)
        pos(aa[0],29,39,61,61);pos(aa[1],119,39,61,61);pos(aa[2],48,7,88,1.8)
        for t in list(fig.texts):t.remove()
        for ax,l in zip(aa[:2],'ab'):
            label(ax,l)
            for t in ax.get_xticklabels():t.set_rotation(90);t.set_ha('center');t.set_va('top');t.set_rotation_mode('default')
        ids=['a','b'];rows=[ids];aa=aa[:2]
    elif name=='figS4_strategies':
        size(150,78)
        pos(aa[0],33,12,103,60)
        for t in aa[0].texts:
            if t.get_text().startswith('countries'):t.set_text('> 0')
        ids=['a'];rows=[]
    elif name=='fig1_design':
        # 补图的四幅目标区地图保留原有范围，字体增大；总图紧裁边。
        ids=['map','protocol']+[f'local{i+1}' for i in range(4)]
        aa=[fig.axes[i] for i in [0,2,3,4,5,6]]
        rows=[['local1','local2','local3','local4']]
    if not ids:ids=[str(i) for i in range(len(aa))]
    # 实际裁切轴外折线，避免 PDF 的隐藏路径干扰碰撞检查。
    from shapely.geometry import LineString, box
    import numpy as np
    for ax in fig.axes:
        for ln in ax.lines:
            if ln.get_transform()!=ax.transData or not ln.get_clip_on():continue
            xy=ln.get_xydata()
            if len(xy)<2 or not np.isfinite(xy).all():continue
            lim=box(*[min(ax.get_xlim()),min(ax.get_ylim()),max(ax.get_xlim()),max(ax.get_ylim())])
            if lim.covers(LineString(xy)):continue
            cut=LineString(xy).intersection(lim)
            parts=[cut] if cut.geom_type=='LineString' else list(cut.geoms) if hasattr(cut,'geoms') else []
            coords=[]
            for part in parts:
                if part.geom_type=='LineString':coords.extend(list(part.coords)+[(float('nan'),float('nan'))])
            if coords:ln.set_data(np.array(coords).T)
    for t in fig.findobj(Text):
        if '\n' in t.get_text():t.set_linespacing(1.3)
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,axes=aa,panel_ids=ids,row_groups=rows,column_groups=cols,
        strict=True,exemptions=exemptions,json_out=base+'.alignment.json',overlay_svg=base+'.alignment.svg')
