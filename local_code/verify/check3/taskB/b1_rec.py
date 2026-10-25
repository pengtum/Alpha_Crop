import numpy as np, pandas as pd
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
R = pd.read_pickle('b1_rows.pkl')
A = R[R.kind=='arms'].set_index(['feat','target'])
pk = R[R.kind!='arms']
for f in ['ae','s2']:
    a = A.loc[f]
    base = a.PO_oa - a.L0_oa
    print(f'\n===== {f.upper()} =====  PO−L0 均值 {base.mean():+.2f} pp/国；POtest−PO 最大 |差| {(a.POtest_oa-a.PO_oa).abs().max():.2f}')
    tab = pd.DataFrame({'L0':a.L0_oa, 'PO−L0':base})
    tab['EM%'] = 100*(a.EM_oa-a.L0_oa)/base
    tab['EM5k%'] = 100*(a.EM5k_oa-a.L0_oa)/base
    tab['EM_it'] = a.EM_it; tab['EM_conv']=a.EM_conv; tab['EM5k_it']=a.EM5k_it
    tab['EMtest%'] = 100*(a.EMtest_oa-a.L0_oa)/base
    for k in (50,200):
        o = pk[(pk.feat==f)&(pk.kind=='orig')&(pk.k==k)].groupby('target').oa.mean()
        tab[f'PK{k}o%'] = 100*(o-a.L0_oa)/base
    for k in (10,20,50,100,200):
        g = pk[(pk.feat==f)&(pk.kind=='new')&(pk.k==k)]
        o = g.groupby('target').oa.mean()
        tab[f'PK{k}n%'] = 100*(o-a.L0_oa)/base
    print(tab.round(1).to_string())
    S = base.sum()
    print(f'合并恢复率：EM(200) {100*(a.EM_oa-a.L0_oa).sum()/S:.1f}%  EM(5000) {100*(a.EM5k_oa-a.L0_oa).sum()/S:.1f}%  EM(test,200) {100*(a.EMtest_oa-a.L0_oa).sum()/S:.1f}%')
    print(f'EM 收敛(200 次内) {a.EM_conv.sum()}/11；5000 次内 {a.EM5k_conv.sum()}/11；L1(EM200) 均值 {a.EM_l1.mean():.3f}，L1(EM5k) {a.EM5k_l1.mean():.3f}')
    for kind, ks in (('orig',(50,200)),('new',(10,20,50,100,200))):
        for k in ks:
            g = pk[(pk.feat==f)&(pk.kind==kind)&(pk.k==k)]
            o = g.groupby('target').oa.mean()
            pooled = 100*(o-a.L0_oa).sum()/S
            # 按种子的合并恢复率分布（每个种子一个值）
            per_seed = g.pivot(index='seed', columns='target', values='oa')
            rs = 100*(per_seed.sub(a.L0_oa, axis=1)).sum(1)/S
            # 单国单种子最差
            per_ct = 100*per_seed.sub(a.L0_oa,axis=1).div(base,axis=1)
            print(f'  PK{k:<3d} [{kind}] 合并(种子平均) {pooled:5.1f}% | 按种子: 中位 {rs.median():.1f}% 范围 [{rs.min():.1f}, {rs.max():.1f}] | '
                  f'单国×种子恢复率 5% 分位 {np.percentile(per_ct.values,5):.0f}% 最小 {per_ct.values.min():.0f}% ({per_ct.min().idxmin()}) | L1 均值 {g.l1.mean():.3f}')
    # ba_ge30
    d_ls = a.PO_ba - a.LS_ba; d_l0 = a.PO_ba - a.L0_ba
    print(f'ba_ge30: PO−LS 均值 {d_ls.mean():+.2f}（负 {(d_ls<0).sum()}/11，范围 [{d_ls.min():+.1f},{d_ls.max():+.1f}]）；PO−L0 均值 {d_l0.mean():+.2f}（负 {(d_l0<0).sum()}/11）；'
          f'EM−L0 {(a.EM_ba-a.L0_ba).mean():+.2f}（负 {((a.EM_ba-a.L0_ba)<0).sum()}/11）')
    for k in (50,200):
        g = pk[(pk.feat==f)&(pk.kind=='new')&(pk.k==k)].groupby('target').ba.mean()
        dd = g - a.L0_ba
        print(f'   PK{k}(20 种子均值) ba_ge30 − L0 均值 {dd.mean():+.2f}（负 {(dd<0).sum()}/11）; − LS 均值 {(g-a.LS_ba).mean():+.2f}')
