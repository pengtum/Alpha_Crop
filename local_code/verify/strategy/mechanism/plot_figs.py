import sys; sys.path.insert(0,'.')
from load import ci
from wls import slope
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':['Hiragino Sans GB','Arial Unicode MS'],'axes.unicode_minus':False,'font.size':9,
                     'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#8a8984','axes.labelcolor':'#0b0b0b',
                     'xtick.color':'#52514e','ytick.color':'#52514e','axes.grid':True,'grid.color':'#e6e5e0','grid.linewidth':0.6,
                     'figure.facecolor':'#fcfcfb','axes.facecolor':'#fcfcfb','savefig.facecolor':'#fcfcfb'})
C1,C2,C3,GR='#2a78d6','#eb6834','#1baf7a','#8a8984'
NM={'predstrat_srs':'predstrat','margin':'margin','gist_margin':'gist','badge_c':'badge','kcenter':'kcenter','typiclust':'typiclust','kherding':'kherding','lpm2_nogeo':'lpm2'}
A=pd.read_pickle('A.pkl'); A=A[A.cond.isin(['AE','S2'])].reset_index(drop=True); A['cell']=np.arange(len(A))
L=pd.read_pickle('L.pkl')
# ---------------- 图 1
fig,ax=plt.subplots(1,3,figsize=(13.5,4.3),gridspec_kw={'width_ratios':[1,1.15,1.15]})
# A 剂量–反应（格内去均值后分箱）
b=slope(L)
q=pd.qcut(L.dlog_w,20,duplicates='drop')
g=L.groupby(q,observed=True).apply(lambda d: pd.Series({'x':np.average(d.dlog_w,weights=d.w),'y':np.average(d.drec_w,weights=d.w),'n':len(d)}))
a=ax[0]; a.axhline(0,color=GR,lw=0.8); a.axvline(0,color=GR,lw=0.8)
xs=np.linspace(g.x.min(),g.x.max(),10); a.plot(xs,b*xs,color=C2,lw=2,label=f'合并斜率 b = {b:+.1f} pp')
a.scatter(g.x,g.y,s=28,color=C1,edgecolor='#fcfcfb',linewidth=1.5,zorder=3,label='分箱均值（20 分位）')
bs=L.groupby('target').apply(lambda d: slope(d)); r=ci(bs)
a.set_xlabel('该类标签数的变化 Δlog(1+n)（相对 SRS，格内去均值）'); a.set_ylabel('该类召回的变化 Δrecall（pp，格内去均值）')
a.set_title(f'A  逐类剂量–反应\n逐国斜率 {r["mean"]:+.1f} [{r["lo"]:+.1f}, {r["hi"]:+.1f}]，{r["npos"]}/{r["n"]} 国为正',loc='left',fontsize=9.5)
a.legend(frameon=False,fontsize=8,loc='upper left')
# B 构成预测 vs 观测（留一方法估计斜率，9 国斜率）
L9=L[~L.target.isin(['RO','PT'])]
pts=[]
for m in NM:
    bb=slope(L9[L9.method!=m]); pc=L[L.method==m].groupby('cell').dlog.mean()
    for cond in ('AE','S2'):
        x=A[(A.method==m)&(A.cond==cond)].copy(); x['pred']=bb*pc.reindex(x.cell).to_numpy()
        cv=lambda d,col: d.groupby(['clf','k','target'])[col].mean().groupby('target').mean()
        for reg,mask in (('9 国',~x.target.isin(['RO','PT'])),('RO、PT',x.target.isin(['RO','PT']))):
            o=cv(x[mask],'d_ba_ge30'); p=cv(x[mask],'pred')
            c=ci(o) if len(o)>2 else dict(mean=o.mean(),lo=np.nan,hi=np.nan)
            pts.append(dict(m=m,cond=cond,reg=reg,pred=p.mean(),obs=c['mean'],lo=c['lo'],hi=c['hi']))
P=pd.DataFrame(pts); P.to_csv('fig1_panelB_points.csv',index=False)
a=ax[1]; lim=(-4.0,2.5)
a.plot([-1,2],[-1,2],color=GR,lw=1,ls='--'); a.text(1.22,-1.35,'虚线：观测 = 构成预测',color='#52514e',fontsize=7.5,ha='right')
a.axhline(0,color=GR,lw=0.8)
for reg,col,mk in (('9 国',C1,'o'),('RO、PT',C2,'s')):
    d=P[P.reg==reg]
    if reg=='9 国': a.errorbar(d.pred,d.obs,yerr=[d.obs-d.lo,d.hi-d.obs],fmt='none',ecolor=col,alpha=0.35,lw=1)
    for cond,fill in (('AE',True),('S2',False)):
        dd=d[d.cond==cond]
        a.scatter(dd.pred,dd.obs,s=38,marker=mk,facecolor=col if fill else '#fcfcfb',edgecolor=col,linewidth=1.5,zorder=3,
                  label=f'{reg}，{cond} 指导')
for _,rr in P[(P.cond=='AE')].iterrows():
    if (rr.reg=='9 国' and rr.m in ('predstrat_srs','margin','gist_margin')) or (rr.reg=='RO、PT' and rr.m in ('predstrat_srs','margin','kcenter')):
        a.annotate(NM[rr.m],(rr.pred,rr.obs),xytext=(-6,-11) if rr.m=='gist_margin' else (5,-3),textcoords='offset points',fontsize=7.5,color='#52514e',ha='right' if rr.m=='gist_margin' else 'left')
a.set_xlim(-0.35,1.25); a.set_ylim(*lim)
a.set_xlabel('构成通道预测的 ΔBA（pp；剂量斜率 × 平均 Δlog 剂量）'); a.set_ylabel('观测 ΔBA = 方法 − SRS（ba_ge30，pp）')
a.set_title('B  方法效应 ≈ 构成通道 + 地块通道\n9 国点靠近对角线；RO、PT 远在其下',loc='left',fontsize=9.5)
a.legend(frameon=False,fontsize=7.5,loc='lower right')
# C OA：9 国 vs RO vs PT
a=ax[2]
order=['predstrat_srs','margin','gist_margin','badge_c','kcenter','typiclust','kherding','lpm2_nogeo']
y=np.arange(len(order))[::-1]
for j,(reg,col,mask) in enumerate((('9 国',C1,~A.target.isin(['RO','PT'])),('PT',C3,A.target=='PT'),('RO',C2,A.target=='RO'))):
    v=[A[mask&(A.method==m)].groupby(['cond','clf','k','target']).d_oa.mean().groupby('target').mean().mean() for m in order]
    a.scatter(v,y+0.18*(1-j),s=34,color=col,edgecolor='#fcfcfb',linewidth=1.5,zorder=3,label=reg)
a.axvline(0,color=GR,lw=0.8); a.set_yticks(y); a.set_yticklabels([NM[m] for m in order])
a.set_xlabel('ΔOA = 方法 − SRS（pp；AE、S2 指导与两分类器平均）')
a.set_title('C  OA：偏离代表性的代价集中在 RO、PT',loc='left',fontsize=9.5); a.legend(frameon=False,fontsize=8,loc='lower left')
fig.tight_layout(); fig.savefig('fig_mech1_composition_regime.png',dpi=200); plt.close(fig)

# ---------------- 图 2：策略 × 标签用法
d=pd.read_csv('/Users/perng_mini/AlphaCrops_local/verify/pivot_eval/stat_review/ctrl_combo_rows.csv').rename(columns={'t':'target'})
key=['target','s','r','k']
base=d[(d['mode']=='bal')&(d.gamma==0.0)].drop_duplicates(key+['lab'])
U={'AE 重训':base.set_index(key+['lab'])[['ba_ae','oa_ae']].set_axis(['ba','oa'],axis=1),
   'R2V 重训集成':base.set_index(key+['lab'])[['ba_ens','oa_ens']].set_axis(['ba','oa'],axis=1)}
for gm in (0.25,0.4,1.0): U[f'+均衡头\nγ={gm}']=d[(d['mode']=='bal')&(d.gamma==gm)].set_index(key+['lab'])[['ba','oa']]
fig,ax=plt.subplots(1,3,figsize=(13.5,4.2),gridspec_kw={'width_ratios':[1,1,1.25]})
for pi_,met in enumerate(('ba','oa')):
    a=ax[pi_]; a.axhline(0,color=GR,lw=0.8)
    for j,(k,col) in enumerate(((50,C1),(200,C2))):
        ms,lo,hi=[],[],[]
        for nm,T in U.items():
            p=T.reset_index().pivot_table(index=key,columns='lab',values=met).xs(k,level='k')
            v=(p['AE_predstrat_srs']-p['none_srs']).groupby('target').mean(); c=ci(v); ms.append(c['mean']); lo.append(c['lo']); hi.append(c['hi'])
        x=np.arange(len(U))+(j-0.5)*0.18
        a.errorbar(x,ms,yerr=[np.array(ms)-lo,np.array(hi)-ms],fmt='o',color=col,ms=6,mec='#fcfcfb',mew=1.5,lw=1.2,capsize=0,label=f'k={k}')
    a.set_xticks(np.arange(len(U))); a.set_xticklabels(list(U),fontsize=7.8)
    a.set_ylabel(('ΔBA（ba_ge30）' if met=='ba' else 'ΔOA')+'：predstrat − SRS（pp）')
    a.set_title(('A  构成策略的 BA 收益随用法的类别均衡程度而缩小' if met=='ba' else 'B  同一对比的 OA：均衡头下收益转到 OA'),loc='left',fontsize=9.5)
    a.legend(frameon=False,fontsize=8)
# C diag3：同一选样在不同学习方式下的 方法 − SRS（BA）
x3=pd.read_csv('/Users/perng_mini/AlphaCrops_local/verify/pivot_eval/codesign/diag3_sel_rows.csv').rename(columns={'tgt':'target'})
k3=['unit','feat','k','learner']; s3=x3[x3.sel=='none_srs'].set_index(k3).ba
x3=x3[x3.sel!='none_srs'].copy(); x3['d']=x3.ba.to_numpy()-s3.reindex(pd.MultiIndex.from_frame(x3[k3])).to_numpy()
a=ax[2]; a.axhline(0,color=GR,lw=0.8)
meths=['predstrat_srs','margin','gist_margin','badge_c','kcenter','typiclust','kherding','lpm2_nogeo']
for j,(Lr,col,lab) in enumerate((('lgb',C1,'LightGBM 重训（r=50）'),('bias',C3,'均衡偏置头（只学类偏置）'),('feat03',C2,'均衡特征头（用 16 维几何）'))):
    ms,lo,hi=[],[],[]
    for m in meths:
        q=x3[(x3.learner==Lr)&x3.sel.isin([f'AE_{m}',f'S2_{m}'])]
        v=q.groupby(['sel','feat','k','target']).d.mean().groupby('target').mean(); c=ci(v); ms.append(c['mean']); lo.append(c['lo']); hi.append(c['hi'])
    xx=np.arange(len(meths))+(j-1)*0.22
    a.errorbar(xx,ms,yerr=[np.array(ms)-lo,np.array(hi)-ms],fmt='o',color=col,ms=5.5,mec='#fcfcfb',mew=1.2,lw=1,capsize=0,label=lab)
a.set_xticks(np.arange(len(meths))); a.set_xticklabels([NM[m] for m in meths],fontsize=8,rotation=20)
a.set_ylabel('ΔBA = 方法 − SRS（pp；AE、S2 指导与 k 平均）')
a.set_title('C  同一批选样，换标签用法后方法排序改变',loc='left',fontsize=9.5); a.legend(frameon=False,fontsize=7.5,loc='lower left')
fig.tight_layout(); fig.savefig('fig_mech2_strategy_x_usage.png',dpi=200); plt.close(fig)
print('ok')
