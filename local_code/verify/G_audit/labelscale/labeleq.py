import sys; sys.argv=['x']
import numpy as np, pandas as pd
import analyze_G as A
R=A.load_rows()
vals={'θ':-0.054,'L95':-0.244,'U95':0.136,'L90':-0.208,'U90':0.101,'+0.3':0.3,'-0.3':-0.3}
rows=[]
for t in sorted(R.target.unique()):
    for clf in ('s2','ae'):
        ks,f=A.srs_curve(R,t,clf)
        for k in (50,200):
            d=dict(target=t,clf=clf,k=k,f0=f[0],fk=np.interp(k,ks,f),f400=f[-1])
            for n,v in vals.items(): d[n]=A.label_equiv(ks,f,k,v)
            rows.append(d)
L=pd.DataFrame(rows)
pd.set_option('display.width',250)
print(L.round(1).to_string())
print(L.groupby('k')[list(vals)].median().T.round(1))
for k in (50,200):
    s=L[L.k==k]
    print(k,'share +0.3 = +inf:',(s['+0.3']==np.inf).mean().round(3),' share -0.3 = -inf:',(s['-0.3']==-np.inf).mean().round(3),
          ' +0.3 <= k labels:',(s['+0.3']<=k).mean().round(3))
# isotonic curve gain 0->50, 200->400 per cell
