# 评审对照 R3：同一混合配置下，"2v重训+XV头" − "AE重训+AE头"，即两视图在标签用法之上的净增量（含简单集成部分）；
# 以及扣除简单集成后的部分：[(2v+XV) − 2v] − [(AE+AEhead) − AE]。只读 r2_rows.csv。
import pandas as pd, numpy as np
df=pd.read_csv('r2_rows.csv'); U=['target','split','rep']
def fmt(v):
    v=np.asarray(v); m=v.mean(); se=v.std(ddof=1)/np.sqrt(len(v)); return f'{m:+.2f} [{m-2.228*se:+.2f},{m+2.228*se:+.2f}] {int((v>0).sum())}/{len(v)}'
for meth in ('srs','predstrat_srs'):
  for k in (50,200):
    d=df[(df.meth==meth)&(df.k==k)]
    for a in (0.5,1.0):
      for beta in (0.25,0.4):
        A=d[(d.arm=='ret_2v+head_both')&(d.a==a)&(d.beta==beta)].set_index(U)
        B=d[(d.arm=='ret_ae+head_ae')&(d.a==a)&(d.beta==beta)].set_index(U)
        j=A[['ba','oa']].join(B[['ba','oa']],rsuffix='_b',how='inner')
        dB=(j.ba-j.ba_b).groupby(level=0).mean(); dO=(j.oa-j.oa_b).groupby(level=0).mean()
        print(f'{meth:14s} k={k} a={a} β={beta}: (2v重训+XV头) − (AE重训+AE头)  dBA {fmt(dB)}  dOA {fmt(dO)}')
