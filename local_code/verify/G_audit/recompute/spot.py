import pandas as pd, numpy as np, glob, hashlib, os
R='/Users/perng_mini/AlphaCrops_local/results/guidance_G/'
DC='/Users/perng_mini/AlphaCrops_local/data/cache_guidance_G/'
y=np.load(DC+'y.npy'); ctry=np.load(DC+'ctry.npy')
df=pd.concat([pd.read_csv(f) for f in sorted(glob.glob(R+'rows_t*.csv'))],ignore_index=True)
sub=df[df.method!='foreign_only']
rng=np.random.default_rng(7)
samp=sub.iloc[rng.choice(len(sub),240,replace=False)]
# include foreign_only rows too
samp=pd.concat([samp,df[df.method=='foreign_only'].sample(10,random_state=1)])
S_cache={}
def scores(yT,ph,big,mask=None):
    if mask is not None: yT,ph=yT[mask],ph[mask]
    rec=np.array([(ph[yT==c]==c).mean() if (yT==c).any() else np.nan for c in range(10)])
    return 100*(ph==yT).mean(), 100*np.nanmean(rec[big]), rec
def cw_from(yr):
    n=np.bincount(yr,minlength=10).astype(float)
    return np.where(n>0,len(yr)/(np.count_nonzero(n)*np.maximum(n,1)),1.0)
maxd={}
def upd(k,v):
    maxd[k]=max(maxd.get(k,0),abs(v))
bad=[]
for _,r in samp.iterrows():
    key=(r.target,r.split)
    if key not in S_cache: S_cache[key]=dict(np.load(R+f'stage0/{r.target}_s{r.split}.npz'))
    S=S_cache[key]
    test=S['test']; yT=y[test]; efar=S['efar'].astype(bool)
    big=np.flatnonzero(np.bincount(yT,minlength=10)>=30)
    assert (ctry[test]==r.target).all()
    if r.method=='foreign_only':
        P={it:S[f'P0test_{r.clf}_i{it}'] for it in (200,250,300)}
    else:
        sp=R+f'sel/{r.target}_s{r.split}_r{r.rep}_{r.cond}_{r.method}_k{r.k}.npz'
        z=np.load(sp); sel=z['sel']
        h=hashlib.md5(np.sort(sel.astype(np.int64)).tobytes()).hexdigest()[:16]
        if h!=r.sel_hash: bad.append(('hash',sp))
        if len(sel)!=r.k or len(np.unique(sel))!=r.k: bad.append(('k',sp))
        n_cand=len(S['cand']); g=np.random.default_rng(10000*r.split+100*r.rep+7)
        pp=np.sort(g.choice(n_cand,min(30000,n_cand),replace=False)); pool=S['cand'][pp]
        if not np.isin(sel,pool).all(): bad.append(('notinpool',sp))
        if np.isin(sel,test).any(): bad.append(('intest',sp))
        pz=np.load(R+f'probs/{r.target}_s{r.split}_r{r.rep}_{r.clf}_k{r.k}_{r.sel_hash}.npz')
        P={int(k_[1:]):pz[k_] for k_ in pz.files}
    P3=P[300].astype(np.float64); P3/=P3.sum(1,keepdims=True)
    oa,ba,rec=scores(yT,P3.argmax(1),big)
    upd('oa',oa-r.oa); upd('ba_ge30',ba-r.ba_ge30)
    _,bf,_=scores(yT,P3.argmax(1),big,efar); upd('ba_far',bf-r.ba_far)
    for it in (200,250):
        _,b,_=scores(yT,P[it].argmax(1),big); upd(f'ba_i{it}',b-r[f'ba_i{it}'])
    # P2
    fr=S['foreign']; cw=cw_from(y[fr])
    if r.method=='foreign_only':
        pit=np.bincount(y[fr],weights=cw[y[fr]],minlength=10); pit/=pit.sum()
        n_cand=len(S['cand']); g=np.random.default_rng(10000*r.split+100*r.rep+7)
        pp=np.sort(g.choice(n_cand,min(30000,n_cand),replace=False)); pool=S['cand'][pp]
    else:
        ys=y[sel]; w=np.r_[cw[y[fr]],50*cw[ys]]; pit=np.bincount(np.r_[y[fr],ys],weights=w,minlength=10)/w.sum()
    pip=np.bincount(y[pool],minlength=10)/len(pool)
    wq=np.divide(pip,pit,out=np.zeros_like(pip),where=pit>0); q=P3*wq; s=q.sum(1,keepdims=True)
    Pc=np.where(s>0,q/np.where(s>0,s,1),P3)
    _,b2,_=scores(yT,Pc.argmax(1),big); upd('ba_P2',b2-r.ba_P2)
print('rows checked',len(samp)); print({k:float(v) for k,v in maxd.items()}); print('bad',bad[:10],len(bad))
