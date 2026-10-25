"""质疑者视角 s1：对策略基准/机制/定位中关键说法的复核（只读 G 行与已有派生表；不训练、不选样）。"""
import os, sys, itertools
os.environ['OMP_NUM_THREADS']='4'
sys.path.insert(0,'/Users/perng_mini/AlphaCrops_local/verify/strategy/benchmark')
from common_bm import *
pd.set_option('display.width',250)
R=load()
OUT='/Users/perng_mini/AlphaCrops_local/verify/strategy/skeptic'
rng=np.random.default_rng(0)

def D(metric, rep0=True):
    d=all_vs_srs(R, metric)
    if rep0: d=d[d.rep==0]
    return d

def country_method(d):
    # 国家 × 方法：先 (方法, 指导, 分类器, k) 内单元均值，再等权
    return d.groupby(['target','method','cond','clf','k']).d.mean().groupby(['target','method']).mean().unstack('method')

print('='*20,'C1：BA 与 OA 的方法排名——跨国均值 vs 国内','='*20)
for rep0 in (False, True):
    CB=country_method(D('ba_ge30',rep0)); CO=country_method(D('oa',rep0))
    rho_mean=stats.spearmanr(CB.mean(),CO.mean())[0]
    within=pd.Series({c: stats.spearmanr(CB.loc[c],CO.loc[c])[0] for c in CB.index})
    print(f'rep0_only={rep0}: 跨国均值 ρ={rho_mean:+.2f}；国内 ρ：中位数 {within.median():+.2f}，>0 的国家 {int((within>0).sum())}/11，≥0.5 的 {int((within>=0.5).sum())}/11')
    print('  ', within.round(2).to_dict())
    rkO=CO.rank(axis=1,ascending=False).mean().sort_values(); rkB=CB.rank(axis=1,ascending=False).mean().sort_values()
    print('  OA 平均秩：', rkO.round(2).to_dict())
    print('  OA 国家中位数：', CO.median().sort_values(ascending=False).round(2).to_dict())
    print('  OA 均值 去 RO、PT：', CO.drop(['RO','PT']).mean().sort_values(ascending=False).round(2).to_dict())
    print('  BA 均值 去 RO、PT：', CB.drop(['RO','PT']).mean().sort_values(ascending=False).round(2).to_dict())
    print('  去 RO、PT 后 BA 与 OA 方法均值 ρ =', round(stats.spearmanr(CB.drop(['RO','PT']).mean(),CO.drop(['RO','PT']).mean())[0],2))
    CB.to_csv(f'{OUT}/c1_country_method_ba_rep0{int(rep0)}.csv'); CO.to_csv(f'{OUT}/c1_country_method_oa_rep0{int(rep0)}.csv')

print('\n'+'='*20,'C2：方差份额的设计依赖（每自由度比较）','='*20)
for metric in ('ba_ge30','oa'):
    d=D(metric)
    ssM=ssG=ssMG=0.0
    for key,g in d.groupby(['target','split','clf','k']):
        Y=g.pivot_table(index='method',columns='cond',values='d')
        Y=Y-Y.values.mean()
        mM=Y.mean(1); mG=Y.mean(0)
        ssM+=2*(mM**2).sum(); ssG+=8*(mG**2).sum(); ssMG+=((Y.values-mM.values[:,None]-mG.values[None,:])**2).sum()
    tot=ssM+ssG+ssMG
    print(f'[{metric}] 份额 方法 {ssM/tot:.1%} 指导 {ssG/tot:.1%} 交互 {ssMG/tot:.1%}；纯噪声下期望 46.7% / 6.7% / 46.7%')
    print(f'   每自由度均方：方法 {ssM/7/88:.3f}，指导 {ssG/1/88:.3f}，交互 {ssMG/7/88:.3f}；方法/交互 = {ssM/ssMG:.2f}，指导/交互 = {ssG*7/ssMG:.2f}')

print('\n'+'='*20,'C3：方法主效应（国家随机）对方法清单的敏感性','='*20)
def method_F(CM):
    # CM: 国家 × 方法（已在 指导/分类器/k 上平均）；两因素无重复 ANOVA，误差=方法×国家
    Y=CM.values; n,m=Y.shape
    gm=Y.mean(); rm=Y.mean(1,keepdims=True); cmn=Y.mean(0,keepdims=True)
    ssm=n*((cmn-gm)**2).sum(); sse=((Y-rm-cmn+gm)**2).sum()
    F=(ssm/(m-1))/(sse/((m-1)*(n-1))); p=stats.f.sf(F,m-1,(m-1)*(n-1))
    return F,p,m-1,(m-1)*(n-1)
for metric in ('ba_ge30','oa'):
    CM=country_method(D(metric))
    for name,drop in [('全部 8',[]),('去 kcenter',['kcenter']),('去 predstrat_srs',['predstrat_srs']),('去 kcenter+predstrat',['kcenter','predstrat_srs']),('只 E1',PRED+HYB),('只预测/混合',E1)]:
        for exRO in (False,True):
            X=CM.drop(columns=drop)
            if exRO: X=X.drop('RO')
            F,p,a,b=method_F(X)
            print(f'[{metric}] {name:18s} {"去RO" if exRO else "11国"}: F({a},{b})={F:.2f}, p={p:.3g}')

print('\n'+'='*20,'C4：predstrat_srs 的稳健性（重复 0，AE+S2 合并）','='*20)
for metric in ('ba_ge30','oa'):
    d=D(metric); ps=d[d.method=='predstrat_srs']
    v=cv(ps)
    for name,excl in [('11 国',[]),('去 SE',['SE']),('去 RO',['RO']),('去 SE、RO',['SE','RO']),('去开发国 PT/BE/SI',['PT','BE','SI']),('去检验2四国',['PT','SE','SK','LV'])]:
        s=ts(v.drop(excl)); print(f'[{metric}] {name:14s} {s["mean"]:+.2f} [{s["lo95"]:+.2f}, {s["hi95"]:+.2f}] {s["n_pos"]}/{s["n"]}  p={s["p"]:.3f}')
    print(f'  中位数 {v.median():+.2f}；各国：', v.round(2).to_dict())
    per=ps.groupby(['cond','clf','k','target']).d.mean().groupby(['cond','clf','k']).agg(['mean',lambda x:(x>0).sum()])
    print(per.round(2).to_string())

print('\n'+'='*20,'C5：族错误率——8 方法相对 SRS 的 maxT（国家符号翻转，保留方法间相关）','='*20)
for metric in ('ba_ge30','oa'):
    for cond in ('AE+S2','AE','S2'):
        d=D(metric, rep0=False)
        if cond!='AE+S2': d=d[d.cond==cond]
        CM=country_method(d)   # 11×8
        V=CM.values
        def tstat(A):
            return A.mean(0)/(A.std(0,ddof=1)/np.sqrt(A.shape[0]))
        t0=tstat(V)
        signs=np.array(list(itertools.product([-1,1],repeat=V.shape[0])))
        mx=np.array([np.abs(tstat(V*s[:,None])).max() for s in signs])
        padj=pd.Series([(mx>=abs(t)-1e-12).mean() for t in t0],index=CM.columns)
        praw=pd.Series([2*stats.t.sf(abs(t),V.shape[0]-1) for t in t0],index=CM.columns)
        out=pd.DataFrame(dict(mean=CM.mean(),t=t0,p_raw=praw,p_maxT=padj,holm8=holm(praw)))
        print(f'[{metric}|{cond}]'); print(out.round(3).to_string())

print('\n'+'='*20,'C6：OA 分散度随标签价值放大——是否只是噪声随学习曲线斜率放大？','='*20)
# 每国：以 (方法) × (划分, 分类器, k, 指导) 的两因素表，F_c = MS_方法 / MS_方法×格（格内部分只含选样/训练噪声 + 设置交互）
srs=R[(R.method.isin(['srs','foreign_only']))&R.clf.isin(['s2','ae'])]
lv={}
for metric in ('ba_ge30','oa'):
    g=srs[srs.k.isin([0,200])].groupby(['target','k'])[metric].mean().unstack('k')
    lv[metric]=g[200]-g[0]
rows=[]
for metric in ('ba_ge30','oa'):
    d=D(metric)
    for c,g in d.groupby('target'):
        Y=g.pivot_table(index='method',columns=['split','clf','k','cond'],values='d').values
        n_m,n_s=Y.shape
        gm=Y.mean(); rm=Y.mean(1,keepdims=True); cm=Y.mean(0,keepdims=True)
        msm=n_s*((rm-gm)**2).sum()/(n_m-1); mse=((Y-rm-cm+gm)**2).sum()/((n_m-1)*(n_s-1))
        # 只用"划分"为重复：先在 clf/k/cond 上平均，误差 = 方法×划分
        Z=g.groupby(['method','split']).d.mean().unstack('split').values
        rm2=Z.mean(1,keepdims=True); cm2=Z.mean(0,keepdims=True); gm2=Z.mean()
        msm2=2*((rm2-gm2)**2).sum()/7; mse2=((Z-rm2-cm2+gm2)**2).sum()/7
        rows.append(dict(metric=metric,target=c,sd_methods=np.sqrt(((rm-gm)**2).sum()/(n_m-1)),F_cells=msm/mse,F_split=msm2/mse2,
                         p_split=stats.f.sf(msm2/mse2,7,7),label_value=lv[metric][c]))
T=pd.DataFrame(rows); T.to_csv(f'{OUT}/c6_signal_to_noise_by_country.csv',index=False)
for metric in ('ba_ge30','oa'):
    X=T[T.metric==metric].set_index('target')
    print(f'[{metric}]'); print(X.round(3).to_string())
    for col in ('sd_methods','F_cells','F_split'):
        r=stats.spearmanr(X.label_value,X[col]); r2=stats.spearmanr(X.drop(['RO','PT']).label_value,X.drop(['RO','PT'])[col])
        print(f'   ρ(标签价值, {col}) = {r[0]:+.2f} (p={r[1]:.3f})；去 RO、PT {r2[0]:+.2f} (p={r2[1]:.3f})')

print('\n'+'='*20,'C7：RO/PT 的大损失是否来自构成（先验）？Δ 与 P2 修正后 Δ','='*20)
for m in ['kcenter','margin','gist_margin','badge_c','predstrat_srs','typiclust']:
    for c in ['RO','PT']:
        vals={}
        for metric in ('oa','oa_P2','ba_ge30','ba_P2','comp_L1','n_classes_sel'):
            dd=pd.concat([vs_srs(R,cond,[m],metric) for cond in ('AE','S2')])
            dd=dd[(dd.target==c)&(dd.rep==0)]
            vals[metric]=dd.d.mean()
        print(f'{m:14s} {c}: ΔOA {vals["oa"]:+.2f}  ΔOA_P2 {vals["oa_P2"]:+.2f} | ΔBA {vals["ba_ge30"]:+.2f}  ΔBA_P2 {vals["ba_P2"]:+.2f} | Δcomp_L1 {vals["comp_L1"]:+.3f}  Δn_cls {vals["n_classes_sel"]:+.2f}')

print('\n'+'='*20,'C8：F5（k200 − k50）按方法族拆分；只用重复 0','='*20)
for metric in ('ba_ge30','oa'):
    for rep0 in (False,True):
        d=D(metric,rep0)
        for fam,ms in [('8 方法',ALL8),('E1 几何',E1),('预测/混合',PRED+HYB),('去 predstrat',[m for m in ALL8 if m!='predstrat_srs'])]:
            x=d[d.method.isin(ms)].groupby(['target','method','cond','clf','k']).d.mean().unstack('k')
            v=(x[200]-x[50]).groupby('target').mean()
            s=ts(v); print(f'[{metric}] rep0_only={rep0} {fam:10s}: {s["mean"]:+.3f} [{s["lo95"]:+.3f}, {s["hi95"]:+.3f}] {s["n_pos"]}/11 p={s["p"]:.4f}')

print('\n'+'='*20,'C9：8 个"设置"并非独立——两种分类器共用同一选样','='*20)
for metric in ('ba_ge30','oa'):
    d=D(metric)
    S=d.groupby(['cond','clf','k','target','method']).d.mean().groupby(['cond','clf','k','method']).mean().unstack('method')
    S4=d.groupby(['cond','k','target','method']).d.mean().groupby(['cond','k','method']).mean().unstack('method')
    def W(M):
        Rk=M.rank(axis=1).values; m_,n_=Rk.shape[0],Rk.shape[1]
        Rs=Rk.sum(0); S_=((Rs-Rs.mean())**2).sum(); return 12*S_/(m_**2*(n_**3-n_))
    print(f'[{metric}] W(8 设置)={W(S):.3f}；W(4 个独立选样配置：指导×k，分类器平均)={W(S4):.3f}')
    print('   4 配置最佳：', S4.idxmax(axis=1).to_dict(), ' 最差：', S4.idxmin(axis=1).to_dict())
    # 留一国：predstrat 是否仍在 4 配置中都最佳
    cnt=0; worst=[]
    for c in COUNTRIES:
        dd=d[d.target!=c]
        S4c=dd.groupby(['cond','k','target','method']).d.mean().groupby(['cond','k','method']).mean().unstack('method')
        b=S4c.idxmax(axis=1); cnt+=int((b==S4c.idxmax(axis=1).mode()[0]).all()); worst.append((c,b.value_counts().to_dict()))
    print('   留一国后各配置最佳：', worst)
    # 分类器之间（同一选样）方法排序一致性
    for cond in ('AE','S2'):
        for k in (50,200):
            a=S.loc[(cond,'ae',k)]; b=S.loc[(cond,'s2',k)]
            print(f'   同一选样、换分类器 {cond} k={k}: Spearman {stats.spearmanr(a,b)[0]:+.2f}')

print('\n'+'='*20,'C10：标签用法改变方法排序？（codesign diag3；C 级）','='*20)
x=pd.read_csv('/Users/perng_mini/AlphaCrops_local/verify/pivot_eval/codesign/diag3_sel_rows.csv').rename(columns={'tgt':'target'})
key=['unit','feat','k','learner']
s=x[x.sel=='none_srs'].set_index(key)[['ba','oa']]
y=x[x.sel!='none_srs'].copy(); kk=pd.MultiIndex.from_frame(y[key])
for m in ('ba','oa'): y['d_'+m]=y[m].to_numpy()-s[m].reindex(kk).to_numpy()
y=y.dropna(subset=['d_ba']); y=y[y.sel.str.match(r'^(AE|S2)_')]
for m in ('d_ba','d_oa'):
    P=y.groupby(['sel','k','learner','target'])[m].mean().groupby(['sel','k','learner']).mean().unstack('learner')
    print(f'[{m}] 方法×指导×k（32 点）在不同用法之间的 Pearson / Spearman：')
    print(P.corr().round(2).to_string()); print(P.corr(method='spearman').round(2).to_string())
    Pm=y.groupby(['sel','learner','target'])[m].mean().groupby(['sel','learner']).mean().unstack('learner')
    Pm.index=[i.split('_',1)[1] for i in Pm.index]
    Pm=Pm.groupby(level=0).mean()
    print('  8 方法（指导与 k 平均）各用法下的值：'); print(Pm.round(2).to_string())
    print('  各用法最佳：', Pm.idxmax().to_dict(), ' 最差：', Pm.idxmin().to_dict())
