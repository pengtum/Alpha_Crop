import os; os.environ['OMP_NUM_THREADS']='2'
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import pandas as pd
for f in ['Hiragino Sans GB','Heiti TC','STHeiti']:
    try:
        fm.findfont(f, fallback_to_default=False); plt.rcParams['font.family']=f; break
    except Exception: pass
plt.rcParams['axes.unicode_minus']=False
BLUE,ORANGE,AQUA='#2a78d6','#eb6834','#1baf7a'; INK='#0b0b0b'; INK2='#52514e'; GRID='#e4e3df'
# (group, label, dOA, dBA, source)
P=[
 ('ach','predstrat_srs AE指导 k200',0.11,1.67,'G 11国'),
 ('ach','predstrat_srs S2指导 k200',0.13,1.23,'G 11国'),
 ('ach','几何4法均值 k200',-0.36,-0.07,'G 11国'),
 ('ach','margin AE指导 k200',-0.69,0.42,'G 11国'),
 ('ach','bait PT k1000',-0.46,1.49,'阶段A PT'),
 ('ach','gist_maxmin PT k1000',-1.43,1.31,'阶段A PT'),
 ('orc','神谕几何 typiclust k200',-0.06,0.85,'G ORACLE2'),
 ('orc','oracle_sqrt k200',-1.33,1.62,'试点4国 r=10'),
 ('orc','oracle_equal k200',-4.34,3.84,'试点4国 r=10'),
 ('orc','oracle_sqrt PT k1000',-1.44,7.65,'阶段A PT'),
 ('orc','oracle_equal PT k1000',-14.15,16.81,'阶段A PT'),
 ('orc','BIAS_ORACLE 大量标签校准决策层',12.40,6.42,'检验3, 相对k=0'),
 ('trn','r 1→50 (k1000)',2.99,0.78,'阶段A 3国'),
 ('trn','200 vs 300 棵树 (S2,k200)',-0.51,0.96,'G 11国'),
 ('trn','真实先验修正 P2 (S2,k200)',1.18,-7.27,'G 11国'),
 ('trn','SRS k 0→200 (S2)',5.26,2.41,'G 11国'),
 ('trn','SRS k 0→200 去RO/PT',1.30,0.66,'G 9国'),
 ('trn','LR 替代 LGB (AT 单单元)',-9.75,8.62,'model_diag 初步'),
]
df=pd.DataFrame(P,columns=['g','label','doa','dba','src']); df.to_csv('headroom_map_points.csv',index=False)
sty={'ach':(BLUE,'o','可实现的选样方法（相对 SRS）'),'orc':(ORANGE,'D','神谕 / 上限（作弊）'),'trn':(AQUA,'s','非选样杠杆：训练/用法/预算')}
fig,axs=plt.subplots(1,2,figsize=(15,7.2),gridspec_kw=dict(width_ratios=[1.15,1]))
for ax,(xl,yl,zoom) in zip(axs,[((-16,14),(-9,18.5),False),((-2.2,3.4),(-0.9,2.3),True)]):
    ax.set_facecolor('#fcfcfb')
    ax.axhline(0,color=INK2,lw=0.8); ax.axvline(0,color=INK2,lw=0.8)
    ax.fill_between([-0.3,xl[1]],1.0,yl[1],color=AQUA,alpha=0.08,lw=0)
    ax.plot([-0.3,-0.3],[1.0,yl[1]],color=INK2,lw=1,ls='--'); ax.plot([-0.3,xl[1]],[1.0,1.0],color=INK2,lw=1,ls='--')
    for g,(c,m,name) in sty.items():
        d=df[df.g==g]; d=d[(d.doa.between(*xl))&(d.dba.between(*yl))]
        ax.scatter(d.doa,d.dba,s=70,c=c,marker=m,edgecolors='#fcfcfb',linewidths=2,label=name,zorder=3)
    ax.set_xlim(*xl); ax.set_ylim(*yl); ax.grid(color=GRID,lw=0.6); ax.set_axisbelow(True)
    for s in ['top','right']: ax.spines[s].set_visible(False)
    for s in ['left','bottom']: ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2)
    ax.set_xlabel('ΔOA（pp）',color=INK); ax.set_ylabel('Δ ba_ge30（pp）',color=INK)
axs[0].set_title('A  全景：选样、神谕与非选样杠杆的 BA–OA 位置',loc='left',color=INK,fontsize=12)
axs[1].set_title('B  放大原点附近：门槛区 = ΔBA ≥ +1 且 ΔOA ≥ -0.3（阴影）',loc='left',color=INK,fontsize=12)
off0={'oracle_equal PT k1000':(8,-4),'oracle_sqrt PT k1000':(8,-4),'LR 替代 LGB (AT 单单元)':(8,2),'BIAS_ORACLE 大量标签校准决策层':(-10,8),
      '真实先验修正 P2 (S2,k200)':(8,-4),'SRS k 0→200 (S2)':(8,2),'oracle_equal k200':(-8,6)}
for _,r in df.iterrows():
    if r.label in off0:
        dx,dy=off0[r.label]; ha='right' if dx<0 else 'left'
        axs[0].annotate(f'{r.label}\n[{r.src}]',(r.doa,r.dba),xytext=(dx,dy),textcoords='offset points',fontsize=8,color=INK2,ha=ha)
off1={'predstrat_srs AE指导 k200':(8,4),'predstrat_srs S2指导 k200':(8,-10),'几何4法均值 k200':(8,-10),'margin AE指导 k200':(8,-12),
      'bait PT k1000':(8,2),'gist_maxmin PT k1000':(8,-14),'神谕几何 typiclust k200':(8,-12),'oracle_sqrt k200':(-8,6),
      'r 1→50 (k1000)':(-8,8),'200 vs 300 棵树 (S2,k200)':(-8,-14),'SRS k 0→200 去RO/PT':(8,-12)}
for _,r in df.iterrows():
    if r.label in off1 and -2.2<=r.doa<=3.4 and -0.9<=r.dba<=2.3:
        dx,dy=off1[r.label]; ha='right' if dx<0 else 'left'
        txt=r.label if r.label in ('gist_maxmin PT k1000','200 vs 300 棵树 (S2,k200)') else f'{r.label} [{r.src}]'
        axs[1].annotate(txt,(r.doa,r.dba),xytext=(dx,dy),textcoords='offset points',fontsize=8,color=INK2,ha=ha)
axs[0].legend(loc='lower left',frameon=False,fontsize=9)
fig.text(0.01,0.005,'注：各点基线不同——选样方法与神谕相对同单元 SRS；非选样杠杆相对各自对照（见标签）。BIAS_ORACLE 相对 k=0 外国模型；LR 为 1 国 1 单元初步结果。'
         '\n数据：guidance_G rows、stageA_merged_interim_PT、headroom_summary、check3 报告、model_diag/altmodels（进行中）。',fontsize=8,color=INK2)
plt.tight_layout(rect=(0,0.05,1,1))
plt.savefig('fig_headroom_map.png',dpi=150,facecolor='#fcfcfb')
print('ok')
