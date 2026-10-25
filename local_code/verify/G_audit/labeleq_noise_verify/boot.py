import sys, numpy as np, pandas as pd
sys.path.insert(0, "/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops/label_expansion")
from analyze_G import label_equiv
from sklearn.isotonic import IsotonicRegression
W = pd.read_csv('srs_units_mine.csv', index_col=[0,1,2,3]); W.columns = W.columns.astype(int)
KS = np.array([0,50,100,200,400], float)
DEL = {'+0.3':0.3,'θ':-0.054,'L95':-0.244,'U95':0.136,'-0.3':-0.3}
def curve(M):  # M: units x 5
    return IsotonicRegression().fit_transform(KS, M.mean(0))
# 1) reproduce report medians
rows=[]
for (t,c),g in W.groupby(level=[0,1]):
    f=curve(g.values)
    for k in (50,200):
        for n,d in DEL.items(): rows.append(dict(t=t,c=c,k=k,what=n,L=label_equiv(KS,f,k,d)))
L=pd.DataFrame(rows); print(L.groupby(['what','k']).L.median().unstack())
# 2) unit bootstrap per curve
rng=np.random.default_rng(0); B=2000
res=[]; medB={(n,k):[] for n in DEL for k in (50,200)}
per={}
groups={key:g.values for key,g in W.groupby(level=[0,1])}
for b in range(B):
    vals={(n,k):[] for n in DEL for k in (50,200)}
    for key,M in groups.items():
        idx=rng.integers(0,len(M),len(M))
        f=curve(M[idx])
        for k in (50,200):
            for n,d in DEL.items():
                v=label_equiv(KS,f,k,d); vals[(n,k)].append(v)
                per.setdefault((key,n,k),[]).append(v)
    for kk,v in vals.items(): medB[kk].append(np.median(v))
print('\nper-curve P(+0.3 -> inf) at k=200:')
p200=pd.Series({key:np.mean(np.isinf(per[(key,'+0.3',200)])) for key in groups})
print(p200.round(2).to_string()); print('curves with 0.2<=P<=0.8:', ((p200>=0.2)&(p200<=0.8)).sum(), '/22; point value inf in', (L[(L.what=='+0.3')&(L.k==200)].L==np.inf).sum(),'curves')
p50=pd.Series({key:np.mean(np.isinf(per[(key,'+0.3',50)])) for key in groups})
print('k=50 +0.3: curves with 0.2<=P(inf)<=0.8:', ((p50>=0.2)&(p50<=0.8)).sum())
# per-curve bootstrap spread of finite +0.3@50
print('\nper-curve +0.3@k50: point, bootstrap 10/50/90 pct (inf allowed)')
for key in groups:
    v=np.array(per[(key,'+0.3',50)]); pt=L[(L.t==key[0])&(L.c==key[1])&(L.k==50)&(L.what=='+0.3')].L.iloc[0]
    q=np.percentile(np.where(np.isinf(v),1e9*np.sign(v),v),[10,50,90])
    print(key, round(pt,1), np.round(q,1))
print('\nbootstrap distribution of the 22-curve MEDIAN (unit resampling within curves):')
for kk,v in medB.items():
    v=np.array(v); vv=np.where(np.isinf(v),np.sign(v)*1e9,v)
    print(kk, 'pct 5/25/50/75/95:', np.round(np.percentile(vv,[5,25,50,75,95]),1), ' P(inf)=',np.mean(np.isposinf(v)).round(3),' P(-inf)=',np.mean(np.isneginf(v)).round(3))
