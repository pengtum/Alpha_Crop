import pandas as pd, numpy as np
from scipy import stats
exec(open('rel.py').read().split("# R2 candidates")[0].replace("print(","(lambda *a,**k:None)("))
print('--- R2 (analyze def): mean over cells of log(SD_AE/SD_S2) of (method-SRS) across units')
sd={c:out[c].groupby(['target','method','clf','k']).dm.std(ddof=1) for c in out}
lr=np.log(sd['AE']/sd['S2'])
print('n cells',len(lr),'zero SD',(sd['AE']==0).sum(),(sd['S2']==0).sum())
ttest(lr.groupby('target').mean().values,'R2 analyze-def')
sdb={c:out[c].groupby(['target','method','clf','k']).ba_ge30.std(ddof=1) for c in out}
ttest(np.log(sdb['AE']/sdb['S2']).groupby('target').mean().values,'R2 raw-ba (no SRS subtraction)')
# SD over cells pooled within country of dm
ttest(np.log(out['AE'].groupby('target').dm.std()/out['S2'].groupby('target').dm.std()).values,'R2 SD over all cells')

print('--- WCB Webb (own impl, enumerate-free MC 200k)')
cv=pd.Series(out['AE'].groupby(['target','method','clf','k']).ba_ge30.mean().values-out['S2'].groupby(['target','method','clf','k']).ba_ge30.mean().values,
             index=out['AE'].groupby(['target','method','clf','k']).ba_ge30.mean().index).groupby('target').mean()
v=cv.values; n=len(v); t0=v.mean()/(v.std(ddof=1)/np.sqrt(n))
rng=np.random.default_rng(12345)
w6=np.array([-np.sqrt(1.5),-1,-np.sqrt(.5),np.sqrt(.5),1,np.sqrt(1.5)])
W=rng.choice(w6,size=(200000,n)); ys=W*v
ts=ys.mean(1)/(ys.std(1,ddof=1)/np.sqrt(n))
print('restricted WCB p',(np.abs(ts)>=abs(t0)).mean())
# unrestricted variant (bootstrap-t with centered residuals)
e=v-v.mean(); ys2=v.mean()+W*e; ts2=(ys2.mean(1)-v.mean())/(ys2.std(1,ddof=1)/np.sqrt(n))
print('unrestricted WCB p',(np.abs(ts2)>=abs(t0)).mean())

print('--- label equivalents (own PAV + linear interp)')
def pav(y):
    y=list(map(float,y)); w=[1.0]*len(y); blocks=[[yy,1.0,1] for yy in y]
    out_=[]
    for b in blocks:
        out_.append(b)
        while len(out_)>1 and out_[-2][0]>out_[-1][0]:
            a=out_.pop(); c=out_.pop()
            out_.append([(a[0]*a[1]+c[0]*c[1])/(a[1]+c[1]),a[1]+c[1],a[2]+c[2]])
    res=[]
    for b in out_: res+= [b[0]]*b[2]
    return np.array(res)
def inv(ks,f,k,delta):
    fk=np.interp(k,ks,f); tg=fk+delta
    if tg>f.max()+1e-12: return np.inf
    if tg<f.min()-1e-12: return -np.inf
    # smallest x with f(x)>=tg on piecewise linear
    for j in range(len(ks)):
        if f[j]>=tg:
            if j==0: return ks[0]-k
            x0,x1,y0,y1=ks[j-1],ks[j],f[j-1],f[j]
            return (x1 if y1==y0 else x0+(tg-y0)*(x1-x0)/(y1-y0))-k
th=cv.mean(); se=cv.std(ddof=1)/np.sqrt(n); q=stats.t.ppf(.975,n-1)
D={'+0.3':0.3,'L95':th-q*se,'U95':th+q*se,'θ':th,'−0.3':-0.3}
rows=[]; curves={}
for t in C:
    for clf in ['s2','ae']:
        g=df[(df.target==t)&(df.clf==clf)&df.method.isin(['srs','foreign_only'])]
        c=g.groupby('k').ba_ge30.mean().sort_index()
        ks=c.index.values.astype(float); f=pav(c.values); curves[(t,clf)]=(c.round(2).tolist(),f.round(2).tolist())
        for k in [50,200]:
            for nm,dv in D.items(): rows.append(dict(t=t,clf=clf,k=k,what=nm,lab=inv(ks,f,k,dv)))
L=pd.DataFrame(rows)
print(L.groupby(['what','k']).lab.median().unstack())
print(L.pivot_table(index=['t','clf'],columns=['what','k'],values='lab').round(0).to_string())
for kk,vv in curves.items(): print(kk,vv)
