import numpy as np, pandas as pd
from scipy import stats
D1='/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops/label_expansion/storyline/decision_curve'
RR=pd.read_csv(D1+'/decision_curve_repr_residual_s0r0.csv')
DR=pd.read_csv(D1+'/decision_curve_rows.csv')
LS=pd.read_csv('/Users/perng_mini/AlphaCrops_local/verify/storyline/d1_review/lam_sens_rows.csv')
print(LS.k.unique()[:10], LS.lam.unique(), LS.columns.tolist())
def ci(v):
    v=np.asarray(v,float); m=v.mean(); se=v.std(ddof=1)/np.sqrt(len(v)); t=stats.t.ppf(.975,len(v)-1)
    return f'{m:+.2f} [{m-t*se:+.2f},{m+t*se:+.2f}] {int((v>0).sum())}/{len(v)}'
s0=DR[(DR.split==0)&(DR.rep==0)]
f4=s0[s0.design=='fullpool_lam1e-4'].set_index(['target','clf'])
RR=RR.set_index(['target','clf'])
RR['full4_ba']=f4.ba_ge30
RR['res4_ba']=RR.tgt300_ba-RR.full4_ba
lf=LS[(LS.k.astype(str).isin(['full','-1']))|(LS.k.astype(str).str.contains('full'))]
print(len(lf))
for nm,col in [('λ=0.1',None),('λ=1e-4','res4_ba')]:
    c = RR.res_ba if col is None else RR[col]
    cc=c.groupby('target').mean()
    print(nm,'11',ci(cc),'9',ci(cc.drop(['RO','PT'])))
# also: tgt300 - zero-label τ frontier? just print BA residual using best-BA of zero-label τ on s0r0
Z=pd.read_csv('/private/tmp/c3_skeptic/zerolabel_rows.csv'); Z=Z[(Z.design=='tau')&(Z.split==0)&(Z.rep==0)]
zb=Z.groupby(['target','clf']).ba_ge30.max()
RR['res_zl']=RR.tgt300_ba-zb
cc=RR.res_zl.groupby('target').mean(); print('tgt300 − 零标签τ最佳BA(测试集选τ)','11',ci(cc),'9',ci(cc.drop(['RO','PT'])))
cc=(RR.tgt300_ba-RR.base_ba).groupby('target').mean(); print('tgt300−k0',ci(cc))
