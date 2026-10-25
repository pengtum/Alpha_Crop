import sys, numpy as np, pandas as pd
sys.path.insert(0, "/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops/label_expansion")
from analyze_G import label_equiv
from sklearn.isotonic import IsotonicRegression
W = pd.read_csv('srs_units_mine.csv', index_col=[0,1,2,3]); W.columns = W.columns.astype(int)
C = W.groupby(level=[0,1]).mean()
KS=np.array([0,50,100,200,400],float); lk=np.log([50,100,200,400])
# PAV plateau at 50
pav={key:IsotonicRegression().fit_transform(KS,row.values) for key,row in C.iterrows()}
plat50=[key for key,f in pav.items() if np.isclose(f[1],f[2])]
print('curves where PAV pools k=50 with k=100 (plateau):',len(plat50), plat50)
print('label eq for tiny Δ=+0.01 at k=50 on these:',[round(label_equiv(KS,pav[k],50,0.01),1) for k in plat50])
# log-linear per curve
out=[]
for key,row in C.iterrows():
    y=row[[50,100,200,400]].values; b=np.polyfit(lk,y,1)[0]
    for k in (50,200):
        for n,d in {'+0.3':0.3,'θ':-0.054,'-0.3':-0.3}.items():
            if b<=0: v=np.inf if d>0 else -np.inf
            else: v=k*(np.exp(d/b)-1)
            out.append(dict(t=key[0],c=key[1],b=b,k=k,what=n,L=v))
O=pd.DataFrame(out)
print('\nper-curve log-linear slope b (pp per ln k):'); print(O.drop_duplicates(['t','c']).set_index(['t','c']).b.round(3).to_string())
print(O.groupby(['what','k']).L.median().unstack())
# pooled across countries: mean curve per clf, and all 22
for lab,sub in [('s2',C.xs('s2',level=1)),('ae',C.xs('ae',level=1)),('all',C)]:
    m=sub.mean(); f=IsotonicRegression().fit_transform(KS,m.values)
    b=np.polyfit(lk,m[[50,100,200,400]].values,1)[0]
    print(f'pooled mean curve {lab}:', np.round(m.values,2),' PAV-LE +0.3@50',round(label_equiv(KS,f,50,0.3),1),'@200',round(label_equiv(KS,f,200,0.3),1),
          '| θ@50',round(label_equiv(KS,f,50,-0.054),1),'θ@200',round(label_equiv(KS,f,200,-0.054),1),
          '| loglin b',round(b,3),' +0.3@50',round(50*(np.exp(0.3/b)-1),1),'@200',round(200*(np.exp(0.3/b)-1),1))
# median curve without RO/PT (steep ones)
sub=C.drop(['PT','RO'],level=0); m=sub.median(); f=IsotonicRegression().fit_transform(KS,m.values)
b=np.polyfit(lk,m[[50,100,200,400]].values,1)[0]
print('median curve excl PT,RO:',np.round(m.values,2),' PAV +0.3@50',round(label_equiv(KS,f,50,0.3),1),'@200',round(label_equiv(KS,f,200,0.3),1),' loglin b',round(b,3),'+0.3@200',round(200*(np.exp(0.3/b)-1),1))
# common slope with country bootstrap
rng=np.random.default_rng(1); cs=C.index.get_level_values(0).unique()
bs=[]
for _ in range(4000):
    pick=rng.choice(cs,len(cs)); sub=pd.concat([C.xs(c,level=0,drop_level=False) for c in pick])
    bs.append(np.polyfit(lk,sub[[50,100,200,400]].mean().values,1)[0])
bs=np.array(bs); b0=np.polyfit(lk,C[[50,100,200,400]].mean().values,1)[0]
print('common slope b=%.3f, country-bootstrap 95%% [%.3f, %.3f]'%(b0,*np.percentile(bs,[2.5,97.5])))
for k in (50,200):
    q=k*(np.exp(0.3/np.percentile(bs,[97.5,50,2.5]))-1)
    print(f' +0.3 @k={k}: {k*(np.exp(0.3/b0)-1):.1f} labels; CI from slope [{q[0]:.1f}, {q[2]:.1f}]')
