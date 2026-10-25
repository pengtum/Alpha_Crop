import sys, numpy as np, pandas as pd
sys.argv=['x']
import analyze_G as A
R=A.load_rows()
rows=[]
for t in sorted(R.target.unique()):
    for clf in ('s2','ae'):
        ks,f=A.srs_curve(R,t,clf)
        for k in (50,200):
            for nm,d in [('+0.3',.3),('-0.3',-.3),('L95',-0.244),('U95',0.136),('U90',0.101),('L90',-0.208)]:
                rows.append(dict(t=t,clf=clf,k=k,nm=nm,lab=A.label_equiv(ks,f,k,d)))
L=pd.DataFrame(rows)
P=L.pivot_table(index=['t','clf'],columns=['k','nm'],values='lab')
pd.set_option('display.width',250)
print(P.round(0))
print(L.groupby(['nm','k']).lab.median().unstack())
for k in (50,200):
    s=L[(L.k==k)&(L.nm=='+0.3')].lab
    print(k,'+0.3: n inf',np.isinf(s).sum(),'n >k',(s>k).sum(),'of',len(s))
    s=L[(L.k==k)&(L.nm=='-0.3')].lab
    print(k,'-0.3: n -inf',np.isinf(s).sum(),'n < -k/2',(s<-k/2).sum())
# isotonic flat curves: how many curves are flat (plateau) over 50..400 after PAV
for t in sorted(R.target.unique()):
    for clf in ('s2','ae'):
        ks,f=A.srs_curve(R,t,clf)
        print(t,clf,np.round(f,2), 'range50-400',round(f[-1]-f[1],2))
