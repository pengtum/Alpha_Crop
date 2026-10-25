import glob, numpy as np, pandas as pd
from scipy import stats
OUT='/Users/perng_mini/AlphaCrops_local/results/guidance_G'
R=pd.concat([pd.read_csv(f) for f in glob.glob(OUT+'/rows_t*_*.csv')],ignore_index=True)
g=R[R.method.isin(['srs','foreign_only'])]
C=g.groupby(['target','clf','k']).ba_ge30.mean().unstack('k')
ks=np.array([50,100,200,400.]); x=np.log2(ks)
sl=C[[50,100,200,400]].apply(lambda r: np.polyfit(x,r.to_numpy(),1)[0],axis=1)
print('per-curve slopes pp/doubling'); print(sl.round(3).unstack())
print('median',sl.median().round(3),'mean',sl.mean().round(3),'n<=0',(sl<=0).sum())
slx=sl.drop(index=['PT','RO'],level=0)
print('excl PT RO median',slx.median().round(3),'mean',slx.mean().round(3))
# common slope with curve fixed effects = mean of per-curve slopes when x same (balanced) 
xc=x-x.mean()
def common(Cs):
    Y=Cs[[50,100,200,400]].to_numpy(); Y=Y-Y.mean(1,keepdims=True)
    return (Y*xc).sum()/(len(Y)*(xc**2).sum())
print('common all',round(common(C),3),'excl PT RO',round(common(C.drop(index=['PT','RO'],level=0)),3))
# country-level slope (avg clf) and t CI over 11 countries
cs=sl.groupby('target').mean()
m,se=cs.mean(),cs.std(ddof=1)/np.sqrt(len(cs)); q=stats.t.ppf(.975,len(cs)-1)
print('country-mean slope', round(m,3), 'tCI',round(m-q*se,3),round(m+q*se,3))
# bootstrap countries
rng=np.random.default_rng(0); B=[cs.sample(len(cs),replace=True,random_state=int(rng.integers(1e9))).mean() for _ in range(20000)]
print('boot pct CI', np.percentile(B,[2.5,97.5]).round(3))
# doublings for 0.3 pp
for name,s in [('median per-curve',sl.median()),('common exPTRO',common(C.drop(index=['PT','RO'],level=0))),('common all',common(C))]:
    print(name, 'doublings for 0.3pp =', round(0.3/s,2), ' labels at k=50:',round(50*(2**(0.3/s)-1),1),' at k=200:',round(200*(2**(0.3/s)-1),1))
# per-curve label equivalents with log-linear per curve
def le(s,k,d):
    if s<=0: return np.inf if d>0 else -np.inf
    return k*(2**(d/s)-1)
for k in (50,200):
    for nm,d in [('+0.3',.3),('-0.3',-.3),('L95',-0.244),('U95',0.136),('L90',-0.208),('U90',0.101),('theta',-0.054)]:
        v=np.array([le(s,k,d) for s in sl])
        print(k,nm,'median per-curve loglin labels',round(float(np.median(v)),1))
