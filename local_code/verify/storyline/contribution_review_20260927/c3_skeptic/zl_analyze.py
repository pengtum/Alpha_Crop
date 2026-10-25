import numpy as np, pandas as pd
from scipy import stats
D1='/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops/label_expansion/storyline/decision_curve'
Z=pd.read_csv('/private/tmp/c3_skeptic/zerolabel_rows.csv')
U=pd.read_csv(D1+'/decision_curve_units.csv')
key=['target','split','rep','clf']
base=Z[(Z.design=='tau')&(Z.tau==0)].set_index(key)[['ba_ge30','oa']]
bPT=U[U.k==50].pivot_table(index=key,columns='metric',values='base_PT')
print('base check max|d|', (base-bPT.loc[base.index]).abs().max().to_dict())
H=U.pivot_table(index=key+['k'],columns='metric',values='head').reset_index()
def ci(v):
    v=np.asarray(v,float); m=v.mean(); se=v.std(ddof=1)/np.sqrt(len(v)); t=stats.t.ppf(.975,len(v)-1)
    return f'{m:+.2f} [{m-t*se:+.2f},{m+t*se:+.2f}] {int((v>0).sum())}/{len(v)}'
def country(df,cols):
    # average over units, then over features ("both") -> per country
    return df.groupby(['target','clf'])[cols].mean().groupby('target').mean()
def rep(dfc,label):
    for sub,nm in ((None,'11'),(['RO','PT'],'9')):
        x=dfc if sub is None else dfc.drop([c for c in sub if c in dfc.index])
        print(f'  {label} [{nm}]', ' | '.join(f'{c} {ci(x[c])}' for c in x.columns))
T=Z[Z.design=='tau'].copy()
T=T.merge(base.rename(columns={'ba_ge30':'ba0','oa':'oa0'}).reset_index(),on=key)
T['dBA']=T.ba_ge30-T.ba0; T['dOA']=T.oa-T.oa0
print('== 零标签 τ 调整相对 k=0（两特征平均，国家层）==')
for tau in [0.25,0.5,0.75,1.0,1.5]:
    rep(country(T[T.tau==tau],['dBA','dOA']),f'tau={tau}')
E=Z[Z.design=='eq_iter'].merge(base.rename(columns={'ba_ge30':'ba0','oa':'oa0'}).reset_index(),on=key)
E['dBA']=E.ba_ge30-E.ba0; E['dOA']=E.oa-E.oa0
rep(country(E,['dBA','dOA']),'eq_iter')
print('== D1 head(k) 相对 k=0，复核 ==')
HH=H.merge(base.rename(columns={'ba_ge30':'ba0','oa':'oa0'}).reset_index(),on=key)
HH['dBA']=HH.ba_ge30-HH.ba0; HH['dOA']=HH.oa-HH.oa0
for k in [50,200,400,1000]:
    rep(country(HH[HH.k==k],['dBA','dOA']),f'head k={k}')
print('== head(k) − 零标签 τ（固定 τ）==')
for tau in [0.5,1.0]:
    t=T[T.tau==tau].set_index(key)[['ba_ge30','oa']]
    for k in [50,200,400]:
        h=HH[HH.k==k].set_index(key)[['ba_ge30','oa']]
        d=(h-t.loc[h.index]).reset_index().rename(columns={'ba_ge30':'dBA','oa':'dOA'})
        rep(country(d,['dBA','dOA']),f'head{k}-tau{tau}')
print('== OA 匹配前沿：零标签 τ 曲线上 OA ≥ 头的 OA 时能达到的最大 BA（τ 用测试集选，偏向零标签）==')
for k in [50,200,400,1000]:
    out=[]
    for (u),h in HH[HH.k==k].groupby(key):
        tt=T.set_index(key).loc[u]
        ok=tt[tt.oa>=h.oa.values[0]-1e-9]
        fb=ok.ba_ge30.max() if len(ok) else tt[tt.tau==0].ba_ge30.values[0]
        # BA-matched: max OA with BA >= head BA
        ok2=tt[tt.ba_ge30>=h.ba_ge30.values[0]-1e-9]
        fo=ok2.oa.max() if len(ok2) else np.nan
        dom = len(ok)==0
        out.append(dict(zip(key,u),gapBA=h.ba_ge30.values[0]-fb,gapOA=h.oa.values[0]-fo,
                        nan_oa=np.isnan(fo),dom=dom))
    o=pd.DataFrame(out)
    print(f'  k={k} 头的OA高于全部τ点的单元数: {int(o.dom.sum())} 国家 {sorted(o[o.dom].target.unique())}；BA 高于全部τ点的单元数: {int(o.nan_oa.sum())}')
    oc=o.dropna(subset=['gapBA'])
    rep(country(oc,['gapBA']),f'head{k}-frontier@OA')
    oc2=o.dropna(subset=['gapOA'])
    rep(country(oc2,['gapOA']),f'head{k}-frontier@BA(OA差)')
# 每国 τ=0.5 与 head50 对照
t=T[T.tau==0.5]; c1=country(t,['dBA','dOA']); c2=country(HH[HH.k==50],['dBA','dOA']); c3=country(HH[HH.k==400],['dBA','dOA'])
print(pd.concat([c1.add_prefix('tau05_'),c2.add_prefix('h50_'),c3.add_prefix('h400_')],axis=1).round(2))
t=T[T.tau==1.0]; print(country(t,['dBA','dOA']).round(2).T)
