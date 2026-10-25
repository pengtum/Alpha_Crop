import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
D = '/Users/perng_mini/AlphaCrops_local/results/check3'; C=10
df2 = pd.read_pickle('b4_rows.pkl'); R = pd.read_pickle('b1_rows.pkl'); A = R[R.kind=='arms'].set_index(['feat','target'])
def load(t,f,m,s):
    p = np.load(f'{D}/{t}_{f}_{m}_{s}.npy').astype(np.float64); return p/p.sum(1,keepdims=True)
def laplace(yl, a=1.0):
    n = np.bincount(yl, minlength=C).astype(float); return (n+a)/(n.sum()+a*C)
out=[]
for f in ('ae','s2'):
    for t in ['PT','SE','SK','LV']:
        sp = np.load(f'{D}/{t}_split.npz'); yp, yt = sp['y_pool'], sp['y_test']
        Lt = load(t,f,'loco','test'); piT = np.bincount(yp,minlength=C)/len(yp)
        L0 = 100*(Lt.argmax(1)==yt).mean(); PO = 100*((Lt*piT).argmax(1)==yt).mean()
        r = dict(feat=f, target=t, c3_L0=L0, c3_PO_gain=PO-L0)
        for k in (50,200):
            # 与检验 2 random 策略完全相同的 k 个 pool 标签
            g = [100*((Lt*laplace(yp[np.random.default_rng(1000*s+k).choice(len(yp),k,replace=False)])).argmax(1)==yt).mean() for s in range(3)]
            r[f'c3_PK{k}_gain'] = np.mean(g) - L0
            d = df2[(df2.feat==f)&(df2.target==t)]
            fo = d[d.strat=='foreign_only'].oa.mean()
            r['c2_FO'] = fo
            r[f'c2_rand{k}_gain'] = d[(d.strat=='random')&(d.k==k)].oa.mean() - fo
        d = df2[(df2.feat==f)&(df2.target==t)]
        r['c2_rand1000_gain'] = d[(d.strat=='random')&(d.k==1000)].oa.mean() - r['c2_FO']
        r['c2_full_gain'] = d[d.strat=='ceiling_w25'].oa.mean() - r['c2_FO']
        out.append(r)
O = pd.DataFrame(out).set_index(['feat','target'])
pd.set_option('display.width',250)
print(O.round(2).to_string())
for f in ('ae','s2'):
    o = O.loc[f]
    print(f'{f.upper()} 4 国均值: c3 PK50 {o.c3_PK50_gain.mean():+.2f}  c3 PK200 {o.c3_PK200_gain.mean():+.2f}  |  c2 rand50 {o.c2_rand50_gain.mean():+.2f}  c2 rand200 {o.c2_rand200_gain.mean():+.2f}  c2 rand1000 {o.c2_rand1000_gain.mean():+.2f}  | c3 L0 {o.c3_L0.mean():.1f} vs c2 FO {o.c2_FO.mean():.1f}')
    print(f'    去掉 PT: c3 PK50 {o.drop("PT").c3_PK50_gain.mean():+.2f}  c2 rand50 {o.drop("PT").c2_rand50_gain.mean():+.2f}  c3 PK200 {o.drop("PT").c3_PK200_gain.mean():+.2f}  c2 rand200 {o.drop("PT").c2_rand200_gain.mean():+.2f}')
O.to_pickle('b4b.pkl')
