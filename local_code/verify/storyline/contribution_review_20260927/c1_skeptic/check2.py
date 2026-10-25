import pandas as pd, numpy as np
R='/Users/perng_mini/AlphaCrops_local/results/'
a=pd.read_csv(R+'check3_arms.csv'); a=a[a.split=='random']
c=pd.read_csv(R+'check3_capacity_arms.csv')
rng=np.random.default_rng(1)
def boot(num,den,B=5000):
    idx=np.arange(len(num)); out=[]
    for _ in range(B):
        s=rng.choice(idx,len(idx)); d=den[s].sum()
        if d!=0: out.append(num[s].sum()/d)
    return np.percentile(out,[2.5,97.5])*100
for cap in ['es','c300']:
  for f in ['ae','s2']:
    d=c[(c.cap==cap)&(c.feat==f)]
    o=d.pivot_table(index='target',columns='arm',values='oa')
    print(f'--- {cap} {f}')
    print(o[['L0','PM','PO','CMN','W0','Wnat']].round(1).to_string())
    tot=(o.Wnat-o.L0).values; comp=(o.PM-o.L0).values; within=(o.Wnat-o.W0).values
    shap=0.5*(comp+within); inter=comp-within
    for lab,m in [('11',np.ones(11,bool)),('exROPT',~o.index.isin(['RO','PT']))]:
        print(f' [{lab}] chain comp share {comp[m].sum()/tot[m].sum()*100:.1f}%  CI {boot(comp[m],tot[m]).round(1)} | Shapley comp share {shap[m].sum()/tot[m].sum()*100:.1f}% CI {boot(shap[m],tot[m]).round(1)} | domain-first comp share {within[m].sum()/tot[m].sum()*100:.1f}% | interaction mean {inter[m].mean():+.2f}pp')
        cs=comp/tot*100
        print(f'    per-country chain share median {np.median(cs[m]):.1f}%, n share<30%: {(cs[m]<30).sum()}/{m.sum()}, n share<=0: {(cs[m]<=0).sum()}')
    print(' per-country comp pp', dict(zip(o.index,np.round(comp,2))))
    print(' per-country total pp', dict(zip(o.index,np.round(tot,2))))
# EM variants early stop
for f in ['ae','s2']:
    o=a[a.feat==f].pivot_table(index='target',columns='arm',values='oa')
    pg=o.PO-o.L0
    for lab,m in [('11',o.index),('exROPT',[t for t in o.index if t not in ['RO','PT']]),('exRO',[t for t in o.index if t!='RO'])]:
        s=' '.join(f'{arm}={((o[arm]-o.L0)[m].sum()/pg[m].sum()*100):.1f}%' for arm in ['PEM','PEM_LS','PEM_bcts','BBSE_soft','PK50_s0','PK_LS50_s0'])
        print(f'{f} [{lab}] {s}')
    print(f, 'per-country PEM rec', ((o.PEM-o.L0)/pg*100).round(0).to_dict())
