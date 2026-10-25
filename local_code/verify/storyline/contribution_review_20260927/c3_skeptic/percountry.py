import numpy as np, pandas as pd
exec(open('frontier_cmp.py').read().split("R=pd.read_csv")[0])
R=pd.read_csv('/private/tmp/c3_skeptic/retrain_tau_rows.csv'); RI=R.set_index(key)
DR=pd.read_csv(D1+'/decision_curve_rows.csv'); Hg=DR[DR.design=='srs_G_check'][key+['k','oa','ba_ge30']]
Zp=pd.read_csv('/private/tmp/c3_skeptic/zerolabel_rows.csv'); Zp=Zp[Zp.design=='tau']; ZpI=Zp.set_index(key)
U=pd.read_csv(D1+'/decision_curve_units.csv'); Hd=U.pivot_table(index=key+['k'],columns='metric',values='head').reset_index()
out=[]
for nm,H,cur in [('A',Hd,lambda u,k: ZpI.loc[u]),('B',Hg,lambda u,k:(lambda d:d[(d.src=='retrain')&(d.k==k)])(RI.loc[u]))]:
  for _,h in H[H.k.isin([50,200,400])].iterrows():
    u=tuple(h[c] for c in key); c=cur(u,int(h.k)); fb=fr_ba(c,h.oa)
    if np.isnan(fb): fb=c[c.tau==0].ba_ge30.values[0]
    out.append(dict(cmp=nm,target=u[0],clf=u[3],k=int(h.k),gap=h.ba_ge30-fb))
o=pd.DataFrame(out)
t=o.groupby(['cmp','k','clf','target']).gap.mean().unstack('target').round(2)
print(t)
f=o.groupby(['cmp','k','clf','target']).gap.mean().reset_index()
for (c,k,cl),g in f.groupby(['cmp','k','clf']):
    g=g.set_index('target').gap
    print(c,k,cl,'11:',ci(g),' 9:',ci(g.drop(['RO','PT'])))
