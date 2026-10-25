import pandas as pd, numpy as np, glob
from scipy import stats, optimize
R='/Users/perng_mini/AlphaCrops_local/results/guidance_G/'
df=pd.concat([pd.read_csv(f) for f in sorted(glob.glob(R+'rows_t*.csv'))],ignore_index=True)
df['unit']=df.split.astype(str)+'_'+df.rep.astype(str)
C=sorted(df.target.unique())
E1=['lpm2_nogeo','kcenter','typiclust','kherding']
PRED=['margin','predstrat_srs']; HYB=['gist_margin','badge_c']
K=[50,200]

def ttest(v, label=''):
    v=np.asarray(v,float); n=len(v); m=v.mean(); se=v.std(ddof=1)/np.sqrt(n)
    t95=stats.t.ppf(.975,n-1); t90=stats.t.ppf(.95,n-1)
    p=2*stats.t.sf(abs(m/se),n-1)
    s=f"{label}: {m:+.3f} 95%[{m-t95*se:+.3f},{m+t95*se:+.3f}] 90%[{m-t90*se:+.3f},{m+t90*se:+.3f}] p={p:.3f} >0 {int((v>0).sum())}/{n}"
    print(s); return dict(m=m,se=se,l95=m-t95*se,u95=m+t95*se,l90=m-t90*se,u90=m+t90*se,p=p)

def delta(condA,condB,methods,metric='ba_ge30',ks=K,reps=None,d=df):
    a=d[(d.cond==condA)&d.method.isin(methods)&d.k.isin(ks)]
    b=d[(d.cond==condB)&d.method.isin(methods)&d.k.isin(ks)]
    key=['target','method','clf','k','split','rep','unit']
    if reps is not None: a=a[a.rep.isin(reps)]; b=b[b.rep.isin(reps)]
    m=a[key+[metric]].merge(b[key+[metric]],on=key,suffixes=('_a','_b'))
    assert len(m)==len(a)==len(b), (len(m),len(a),len(b))
    m['d']=m[metric+'_a']-m[metric+'_b']
    return m

def country_vals(m):
    cell=m.groupby(['target','method','clf','k']).d.mean()
    return cell.groupby('target').mean()

print('=== H1')
m=delta('AE','S2',E1)
print('n units per cell', m.groupby(['target','method','clf','k']).size().unique())
cv=country_vals(m)
print(cv.round(3).to_dict())
r=ttest(cv.values,'theta_E1')
tost = (r['l90']>-0.3) and (r['u90']<0.3) and (r['l95']<=0<=r['u95'])
print('TOST equiv:',tost)

# R0
unitv = m.groupby(['target','unit']).d.mean()   # per-unit country value (equal weight cells)
v_i = unitv.groupby('target').var(ddof=1)/unitv.groupby('target').size()
print('v_i', v_i.round(4).to_dict())
y=cv.values; vpool=v_i.mean()
def reml(y,v):
    def nll(t2):
        w=1/(v+t2); mu=(w*y).sum()/w.sum()
        return 0.5*(np.log(v+t2).sum()+np.log(w.sum())+(w*(y-mu)**2).sum())
    res=optimize.minimize_scalar(nll,bounds=(0,10),method='bounded',options={'xatol':1e-12})
    t2=res.x
    if nll(0)<=res.fun: t2=0.0
    w=1/(v+t2); mu=(w*y).sum()/w.sum()
    kk=len(y); q=(w*(y-mu)**2).sum()/(kk-1)
    se=np.sqrt(q/w.sum())
    return mu,t2,se
def reml_fp(y,v,it=1000):
    # Fisher scoring REML iteration
    t2=max(0,y.var(ddof=1)-v.mean())
    for _ in range(it):
        w=1/(v+t2); mu=(w*y).sum()/w.sum()
        P_trace=(w).sum()-(w**2).sum()/w.sum()
        new=t2+ ( (w**2*(y-mu)**2).sum() - P_trace ) / (w**2).sum()
        new=max(0,new)
        if abs(new-t2)<1e-12: break
        t2=new
    return t2
vv=np.full(len(y),vpool)
mu,t2,se=reml(y,vv)
print('pooled v',vpool, 'REML fp t2', reml_fp(y,vv))
tq=stats.t.ppf(.90,len(y)-2)
print(f'R0 mu={mu:+.3f} tau={np.sqrt(t2):.3f} SE_HKSJ={se:.3f} PI80=[{mu-tq*np.sqrt(t2+se**2):+.3f},{mu+tq*np.sqrt(t2+se**2):+.3f}]')
mu2,t22,se2=reml(y,v_i.values)
print(f'REML own v: mu={mu2:+.3f} SE_HKSJ={se2:.3f} tau={np.sqrt(t22):.3f}', 'fp tau', np.sqrt(reml_fp(y,v_i.values)))

print('=== H2 OA'); ttest(country_vals(delta('AE','S2',E1,'oa')).values,'D_OA')
print('=== H3 E2')
g=country_vals(delta('AE','S2',E1)); p=country_vals(delta('AE','S2',PRED)); h=country_vals(delta('AE','S2',HYB))
ttest(((g+p+h)/3).values,'theta_E2')
print('pred units',delta('AE','S2',PRED).unit.unique())
print('=== pathways')
ttest(g.values,'geom'); ttest(p.values,'pred'); ttest(h.values,'hyb')
print('=== gamma, per clf, per k')
cc=m.groupby(['target','method','clf','k']).d.mean().groupby(['target','clf']).mean().unstack()
ttest(((cc['ae']-cc['s2'])/2).values,'gamma')
ttest(cc['s2'].values,'clf s2'); ttest(cc['ae'].values,'clf ae')
ck=m.groupby(['target','method','clf','k']).d.mean().groupby(['target','k']).mean().unstack()
ttest(ck[50].values,'k50'); ttest(ck[200].values,'k200')
print('=== per method raw')
allm=delta('AE','S2',E1+PRED+HYB)
pm=allm.groupby(['target','method','clf','k']).d.mean().groupby(['target','method']).mean().unstack()
raw=pm.mean(); se_m=pm.std(ddof=1)/np.sqrt(len(pm))
print(pd.DataFrame({'raw':raw,'se':se_m}).round(3))
# shrinkage: REML across methods
mu_m,t2m,_=reml(raw.values,(se_m**2).values)
print('tau_methods REML',np.sqrt(t2m),'mu',mu_m)
B=se_m**2/(se_m**2+t2m); shr=B*mu_m+(1-B)*raw
print(pd.DataFrame({'shrunk':shr,'sesh':np.sqrt((1-B)*se_m**2)}).round(3))
print('=== P1 P2 Efar')
m2=delta('AE','S2',E1,'ba_P2'); ttest(country_vals(m2).values,'P2')
ttest(country_vals(delta('AE','S2',E1,'ba_P1')).values,'P1')
ttest(country_vals(delta('AE','S2',E1,'ba_far')).values,'E_far')
ttest((cv-country_vals(m2)).values,'P0-P2')
print('=== XY/S2XY')
XM=['lpm2_nogeo','kcenter']
ttest(country_vals(delta('AE','S2',XM,reps=[0])).values,'AE-S2')
ttest(country_vals(delta('AE','S2XY',XM,reps=[0])).values,'AE-S2XY')
ttest(country_vals(delta('AE','XY',XM,reps=[0])).values,'AE-XY')
ttest(country_vals(delta('S2XY','S2',XM,reps=[0])).values,'S2XY-S2')
print('=== ORACLE')
OM=['lpm2_nogeo','typiclust','margin']
ttest(country_vals(delta('ORACLE','S2',OM,reps=[0])).values,'H')
ttest(country_vals(delta('AE','S2',OM,reps=[0])).values,'AE-S2 same')
print('=== sensitivity')
drop=lambda s: cv.drop(s).values
ttest(drop(['PT','SE','SK','LV']),'drop test2')
ttest(drop(['RO']),'drop RO')
ttest(drop(['PT','BE','SI']),'drop dev')
ttest(country_vals(delta('AE','S2',E1,'ba_i200')).values,'i200')
ttest(country_vals(delta('AE','S2',E1,'ba_i250')).values,'i250')
chk=(df.ba_i300-df.ba_ge30).abs().max(); print('max|ba_i300-ba_ge30|',chk)
m.to_pickle('m_E1.pkl')
