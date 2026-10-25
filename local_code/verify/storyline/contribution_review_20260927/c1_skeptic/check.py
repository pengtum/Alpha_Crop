import pandas as pd, numpy as np
from scipy import stats
R='/Users/perng_mini/AlphaCrops_local/results/'
a=pd.read_csv(R+'check3_arms.csv'); c=pd.read_csv(R+'check3_capacity_arms.csv')
a=a[a.split=='random']
def piv(df,feat,metric):
    return df[df.feat==feat].pivot_table(index='target',columns='arm',values=metric)
def tci(x):
    x=np.asarray(x); n=len(x); m=x.mean(); se=x.std(ddof=1)/np.sqrt(n); t=stats.t.ppf(.975,n-1)
    return f"{m:+.2f} [{m-t*se:+.2f},{m+t*se:+.2f}] n>0={int((x>0).sum())}/{n}"
EX=['RO','PT']; REUSE=['PT','SE','SK','LV']
for cap,df in [('es',c[c.cap=='es']),('c300',c[c.cap=='c300'])]:
  for f in ['ae','s2']:
    o=piv(df,f,'oa'); b=piv(df,f,'ba_ge30')
    pk=[x for x in o.columns if x.startswith('PK50_')]
    pk3=[x for x in pk if x in ('PK50_s0','PK50_s1','PK50_s2')]
    o['PK50']=o[pk].mean(1); o['PK50_3']=o[pk3].mean(1); b['PK50']=b[pk].mean(1)
    tot=o.Wnat-o.L0; comp=o.PM-o.L0; pg=o.PO-o.L0
    print(f'==== {cap} {f}')
    for lab,idx in [('11',o.index),('exROPT',[t for t in o.index if t not in EX]),('exReuse4',[t for t in o.index if t not in REUSE]),('exRO',[t for t in o.index if t!='RO'])]:
        print(f' [{lab}] OAcomp share={comp[idx].sum()/tot[idx].sum()*100:.1f}%  total OA {tci(tot[idx])}  PO-L0 {tci(pg[idx])}')
        print(f'    PK50(20) rec={ (o.PK50-o.L0)[idx].sum()/pg[idx].sum()*100:.1f}%  PK50(3)rec={(o.PK50_3-o.L0)[idx].sum()/pg[idx].sum()*100:.1f}%  PK50-PO {tci((o.PK50-o.PO)[idx])}')
        bt=b.W0-b.L0; bs=b.LS-b.L0
        print(f'    BAset share={bs[idx].sum()/bt[idx].sum()*100:.1f}%  BA PO-LS {tci((b.PO-b.LS)[idx])}  BA PK50-LS {tci((b.PK50-b.LS)[idx])} BA PK50-L0 {tci((b.PK50-b.L0)[idx])}')
        print(f'    Wnat-W0 OA (home natural vs balanced) {tci((o.Wnat-o.W0)[idx])}  share of total OA loss={(o.Wnat-o.W0)[idx].sum()/tot[idx].sum()*100:.1f}%')
        print(f'    W0-L0 OA (balanced vs balanced) {tci((o.W0-o.L0)[idx])};  Wnat-PM OA (natural vs natural) {tci((o.Wnat-o.PM)[idx])}')
        print(f'    PM-PO OA {tci((o.PM-o.PO)[idx])}  Wnat-W0 BA {tci((b.Wnat-b.W0)[idx])}')
    print(' RO share of PO gain', round(pg['RO']/pg.sum()*100,1), ' PT share', round(pg['PT']/pg.sum()*100,1))
    print(' per-country PO-L0 OA:', pg.round(2).to_dict())
    print(' per-country comp share:', (comp/tot*100).round(0).to_dict())
    print(' per-country Wnat-W0 OA:', (o.Wnat-o.W0).round(2).to_dict())
