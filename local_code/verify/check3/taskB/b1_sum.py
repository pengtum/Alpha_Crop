import numpy as np, pandas as pd
R = pd.read_pickle('b1_rows.pkl')
A = R[R.kind=='arms'].set_index(['feat','target'])
arms = pd.read_csv('/Users/perng_mini/AlphaCrops_local/results/check3_arms.csv')
arms = arms[arms.split=='random'].pivot_table(index=['feat','target'], columns='arm', values=['oa','ba_ge30'])
pd.set_option('display.width', 250)
print('trained_same all:', A.trained_same.all(), ' n_trained:', A.n_trained.unique(), ' PO fallback rows:', A.PO_fallback.sum())
# 与 arms.csv 对比
for mine, theirs in (('L0_oa',('oa','L0')),('LS_oa',('oa','LS')),('PO_oa',('oa','PO')),('EM_oa',('oa','PEM')),
                     ('L0_ba',('ba_ge30','L0')),('LS_ba',('ba_ge30','LS')),('PO_ba',('ba_ge30','PO'))):
    d = (A[mine]-arms[theirs]).abs().max()
    print(f'{mine:8s} vs arms {theirs}: max|diff|={d:.2e}')
pk = R[R.kind!='arms']
for s in range(3):
    for k in (50,200):
        a = pk[(pk.kind=='orig')&(pk.seed==1000+s)&(pk.k==k)].set_index(['feat','target'])
        print(f'PK{k}_s{s}: max|diff| oa = {(a.oa - arms[("oa",f"PK{k}_s{s}")]).abs().max():.2e}, ba = {(a.ba - arms[("ba_ge30",f"PK{k}_s{s}")]).abs().max():.2e}')
