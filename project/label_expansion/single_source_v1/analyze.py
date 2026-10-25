"""独立分析：复用冻结OA–BA算法，另报告类别覆盖及单源/等样本量多源对照。"""
from pathlib import Path
import sys,os
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import run as R
import analyze_confirm as A
import numpy as np,pandas as pd,json
from sklearn.metrics import confusion_matrix
from protect_baseline import check
OUT=R.ROOT/'analysis'

def summarize_effects(rows):
    """国家共同权重重抽样的敏感性区间，不把110对当作独立样本。"""
    d=rows[rows.level=='D'].copy()
    d['arm']=d['name'].str.removeprefix('D_')
    keys=['learner','arm','source','target']
    d=d.groupby(keys+['unit'],as_index=False).value.mean().groupby(keys,as_index=False).value.mean()
    single=d[d.source!='MULTI5K']
    rng=np.random.default_rng(9827);weights=rng.exponential(size=(2000,len(R.COUNTRIES)))
    ci={c:i for i,c in enumerate(R.COUNTRIES)};summary=[]
    for (learner,arm),g in single.groupby(['learner','arm']):
        for name,targets in [('all11',R.COUNTRIES),('sensitivity9',[c for c in R.COUNTRIES if c not in ('RO','PT')])]:
            v=g[g.target.isin(targets)].dropna(subset=['value']);bytarget=v.groupby('target').value.mean()
            vals=[]
            if len(bytarget)>=2:
                for w in weights:
                    means=[];tw=[]
                    for target,tg in v.groupby('target'):
                        sw=np.array([w[ci[c]] for c in tg.source]);means.append(np.average(tg.value,weights=sw));tw.append(w[ci[target]])
                    vals.append(np.average(means,weights=tw))
            lo,hi=np.quantile(vals,[.025,.975]) if vals else (np.nan,np.nan)
            summary.append(dict(learner=learner,arm=arm,target_set=name,mean_ae_minus_s2=bytarget.mean(),
                                n_targets=len(bytarget),n_pairs=len(v),reweighting_lo=lo,reweighting_hi=hi))
    pd.DataFrame(summary).to_csv(OUT/'representation_summary.csv',index=False)
    d.to_csv(OUT/'pair_effects.csv',index=False)

def label_equivalents(config,cv):
    rows=[];ks=[0,50,100,200,400]
    zero=[cv[('lgb',f,'Z0',0)][0][A.I_T0] for f in ('ae','s2')];grid=min(zero)+A.OFFS
    for r in range(3):
        env={}
        for f in ('ae','s2'):
            for k in ks:
                arm='Z0' if k==0 else f'RT{k}';key=('lgb',f,arm,0 if k==0 else r)
                env[(f,k)]=A.B.envelope([cv[key]],grid)[0]
        ok=np.logical_and.reduce([np.isfinite(v) for v in env.values()]);coverage=float(ok.mean())
        for k in ks[:-1]:
            row=dict(id=config['id'],source=config['source'],target=config['target'],unit=config['unit'],r=r,k_ae=k,coverage=coverage)
            if coverage<A.MIN_COV:
                rows.append(dict(row,status='insufficient_common_oa',n_s2=np.nan,label_difference=np.nan));continue
            target=float(env[('ae',k)][ok].mean());ys=[float(env[('s2',j)][ok].mean()) for j in ks]
            needed,status=invert_curve(ks,ys,target)
            rows.append(dict(row,status=status,n_s2=needed,label_difference=needed-k if np.isfinite(needed) else np.nan,
                             target_ba=target,s2_curve_nonmonotone=bool(np.any(np.diff(ys)<0))))
    return rows

def invert_curve(xs,ys,target):
    if ys[0]>=target:return float(xs[0]),'already_met_at_zero'
    for i in range(1,len(xs)):
        if ys[i]>=target and ys[i]>ys[i-1]:
            return float(xs[i-1]+(xs[i]-xs[i-1])*(target-ys[i-1])/(ys[i]-ys[i-1])),'interpolated'
    return np.nan,'not_reached_no_extrapolation'

def multi_contrasts(frozen,cache):
    rows=[]
    for config in frozen['configs']:
        if config['source']=='MULTI5K':continue
        cv=cache[config['id']];mv=cache[f'MULTI5K__{config["unit"]}']
        for learner in ('lgb','rf'):
            anchor=min(cur[(learner,f,'Z0',0)][0][A.I_T0] for cur in (cv,mv) for f in ('ae','s2'))
            grid=anchor+A.OFFS
            plan=R.CR.fit_plan(learner)
            for arm,r in sorted({(a,r) for a,f,r in plan if a!='TGT'}):
                es={}
                for label,cur in [('single',cv),('multi',mv)]:
                    for f in ('ae','s2'):es[(label,f)]=A.B.envelope([cur[(learner,f,arm,r)]],grid)[0]
                ok=np.logical_and.reduce([np.isfinite(x) for x in es.values()]);cov=float(ok.mean())
                means={k:float(v[ok].mean()) if cov>=A.MIN_COV else np.nan for k,v in es.items()}
                rows.append(dict(source=config['source'],target=config['target'],unit=config['unit'],learner=learner,arm=arm,r=r,coverage=cov,
                                 multi_minus_single_ae=means[('multi','ae')]-means[('single','ae')],
                                 multi_minus_single_s2=means[('multi','s2')]-means[('single','s2')],
                                 representation_interaction=(means[('multi','ae')]-means[('multi','s2')])-(means[('single','ae')]-means[('single','s2')])) )
    pd.DataFrame(rows).to_csv(OUT/'equal_size_multi_contrasts.csv',index=False)

def plots(raw,effects):
    sys.path.insert(0,str(HERE.parent/'storyline/rse_figs'));import nstyle as ns
    import matplotlib.pyplot as plt
    ns.setup();figdir=OUT/'figures';figdir.mkdir(exist_ok=True)
    z=raw[(raw.learner=='lgb')&(raw.arm=='Z0')&(raw.source!='MULTI5K')]
    z=z.groupby(['source','target','feat']).BA.mean().unstack('feat')
    fig,axes=plt.subplots(1,3,figsize=(ns.W2,62*ns.MM))
    for ax,f,title in zip(axes,['ae','s2','delta'],['AE','S2','AE − S2']):
        vals=z.ae-z.s2 if f=='delta' else z[f];mat=vals.unstack('target').reindex(index=R.COUNTRIES,columns=R.COUNTRIES)
        im=ax.imshow(mat,cmap='RdBu_r' if f=='delta' else 'Blues',vmin=-25 if f=='delta' else 0,vmax=25 if f=='delta' else 100)
        ax.set(xticks=range(11),yticks=range(11),xticklabels=R.COUNTRIES,yticklabels=R.COUNTRIES,title=title,xlabel='Target',ylabel='Source')
        ax.tick_params(labelsize=5.5);fig.colorbar(im,ax=ax,shrink=.7,label='BA difference (pp)' if f=='delta' else 'BA (%)')
    fig.tight_layout();ns.save(fig,'single_source_zero_label_matrix',str(figdir))
    fig,axes=plt.subplots(3,4,figsize=(ns.W2,150*ns.MM));axes=axes.ravel()
    a=raw[(raw.learner=='lgb')&(raw.source!='MULTI5K')&(raw.arm.isin(['Z0','RT50','RT100','RT200','RT400']))]
    for ax,target in zip(axes,R.COUNTRIES):
        for f in ('s2','ae'):
            v=a[(a.target==target)&(a.feat==f)].groupby(['source','k']).BA.mean().unstack('k')
            for _,row in v.iterrows():ax.plot(v.columns,row,color=ns.C[f.upper()],alpha=.18,lw=.5)
            ax.plot(v.columns,v.mean(),color=ns.C[f.upper()],label=f.upper(),lw=1.1)
        ax.set(title=target,xlabel='Local labels',ylabel='Default BA (%)',ylim=(0,100));ax.legend(fontsize=5.5)
    axes[-1].axis('off');fig.tight_layout();ns.save(fig,'single_source_label_curves',str(figdir))

def main():
    check();frozen=R.prepare();OUT.mkdir(exist_ok=True);D=R.CR.G.GData()
    allrows=[];raw=[];cms=[];equiv=[];curves={};coverage=[];numerics=[]
    expected={(l,a,f,r) for l in ('lgb','rf') for a,f,r in R.CR.fit_plan(l) if a!='TGT'}
    for config in frozen['configs']:
        cid=config['id'];sp,sel=R.CR.load_unit(R.OLD,config['unit']);fr=np.load(R.ROOT/'configs'/f'{cid}.npz')['foreign']
        runs=pd.read_csv(R.ROOT/'runs'/f'{cid}.csv',keep_default_na=False)
        assert len(runs)==76 and not runs.duplicated(['learner','arm','feat','r']).any()
        assert set(runs[['learner','arm','feat','r']].itertuples(index=False,name=None))==expected
        assert (runs.status=='ok').all() and (runs.code_hash==frozen['code_hash']).all()
        numerics.extend(runs[['id','unit','source','target','learner','arm','feat','r','logit_abs_max','unseen_raw_argmax_fraction','saturated_fraction']].to_dict('records'))
        P={};test_y=D.y[sp['test']];big=sp['big'];seen=np.flatnonzero(np.bincount(D.y[fr],minlength=10)>0)
        covered=np.intersect1d(big,seen)
        coverage.append(dict(source=config['source'],target=config['target'],unit=config['unit'],eligible_classes=len(big),covered_classes=len(covered),
                             missing_target_fraction=float((~np.isin(test_y,seen)).mean())))
        for row in runs.itertuples():
            ti=R.target_indices(row.arm,row.r,sp,sel);train=np.r_[fr,ti]
            assert row.sel_hash==R.CR.G.sel_hash(ti) and row.train_sha==R.CR.G.sel_hash(train)
            path=R.ROOT/'probs'/cid/f'{row.learner}_{row.arm}_{row.feat}_r{row.r}.npz';assert R.sha(path)==row.prob_sha
            pt=np.load(path)['P_test'];assert pt.dtype==np.float32 and pt.shape==(len(test_y),10)
            assert np.isfinite(pt).all() and np.min(pt)>=0 and np.max(abs(pt.sum(axis=1)-1))<1e-5
            assert (pt[:,~np.isin(np.arange(10),D.y[train])]==0).all()
            P[(row.learner,row.feat,row.arm,row.r)]=pt
            pred=pt.argmax(1);cm=confusion_matrix(test_y,pred,labels=np.arange(10));support=cm.sum(axis=1)
            recall=np.divide(np.diag(cm),support,out=np.full(10,np.nan),where=support>0)
            raw.append(dict(id=cid,source=config['source'],target=config['target'],unit=config['unit'],learner=row.learner,feat=row.feat,arm=row.arm,r=row.r,k=row.k,
                            OA=100*float((pred==test_y).mean()),BA=100*float(recall[big].mean()),
                            BA_source_covered=100*float(recall[covered].mean()) if len(covered) else np.nan,
                            newly_covered_classes=len(np.setdiff1d(np.unique(D.y[train]),seen))))
            if row.learner=='lgb' and row.arm in ('Z0','RT400','B_S0','B_SRS300'):
                for i in range(10):
                    for j in range(10):cms.append(dict(source=config['source'],target=config['target'],unit=config['unit'],feat=row.feat,arm=row.arm,r=row.r,true=i,predicted=j,count=int(cm[i,j]),support=int(support[i])))
        for f in ('ae','s2'):
            P[('lgb',f,'TGT',0)]=np.load(R.OLD/'probs'/config['unit']/f'lgb_TGT_{f}_r0.npz')['P_test']
        cv=A.unit_curves(test_y,big,P);curves[cid]=cv
        rows=A.unit_rows(config['unit'],cv,'formal',3)
        for z in rows:z.update(source=config['source'],target=config['target'],id=cid)
        allrows.extend(rows);equiv.extend(label_equivalents(config,cv))
        print('Analyzed',cid,flush=True)
    pd.DataFrame(numerics).to_csv(OUT/'numerical_diagnostics.csv',index=False)
    d=pd.DataFrame(allrows);d.to_csv(OUT/'unit_rows.csv',index=False)
    raw=pd.DataFrame(raw);raw.to_csv(OUT/'default_metrics.csv',index=False)
    pd.DataFrame(cms).to_csv(OUT/'confusion_counts.csv',index=False)
    pd.DataFrame(coverage).to_csv(OUT/'class_coverage.csv',index=False)
    pd.DataFrame(equiv).to_csv(OUT/'label_equivalents.csv',index=False)
    summarize_effects(d);multi_contrasts(frozen,curves);plots(raw,d)
    summary=pd.read_csv(OUT/'representation_summary.csv')
    text='独立探索性单源迁移实验；未修改RSE稿件。\n训练全部完成：18,392次新拟合，另复用44个目标单独模型。\n\n'
    nd=pd.DataFrame(numerics);text+=f'数值诊断：{int((nd.logit_abs_max>1e4).sum())} 次拟合出现绝对logit>10,000；详见numerical_diagnostics.csv，不能将这类模型问题直接解释为表征本身的限制。\n\n'
    text+=summary[(summary.target_set=='all11')&(summary.learner=='lgb')].to_string(index=False)
    text+='\n\n区间为国家共享权重重抽样敏感性范围，不是110个独立重复的置信区间。共同OA不可达时保持缺失；标签等价不外推。是否入稿等待用户讨论。\n'
    (OUT/'REPORT.txt').write_text(text);check()
if __name__=='__main__':main()
