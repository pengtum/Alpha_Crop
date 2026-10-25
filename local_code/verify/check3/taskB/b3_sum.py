import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
df = pd.read_pickle('b3_rows.pkl'); R = pd.read_pickle('b1_rows.pkl')
pk = R[(R.kind=='new')&(R.k==50)]
for f in ('ae','s2'):
    print(f'===== {f.upper()} =====')
    base = df.groupby('target')[f'{f}_PO'].first() - df.groupby('target')[f'{f}_L0'].first(); S = base.sum()
    L0 = df.groupby('target')[f'{f}_L0'].first()
    # 随机 50 的参照
    g = pk[pk.feat==f].pivot(index='seed', columns='target', values='oa')
    rr = 100*g.sub(L0,axis=1).sum(1)/S
    print(f'  随机 50（20 种子）：合并恢复率 均值 {100*(g.mean()-L0).sum()/S:.1f}%  按种子 [{rr.min():.0f}, {rr.max():.0f}]  L1 {pk[pk.feat==f].l1.mean():.3f}')
    for ncl in (20,50):
        for m in (1,2,3):
            d = df[(df.ncl==ncl)&(df.m==m)]
            gm = d.groupby('target')[[f'{f}_PK', f'{f}_Pinf']].mean()
            rec = 100*(gm[f'{f}_PK']-L0).sum()/S
            rec_inf = 100*(gm[f'{f}_Pinf']-L0).sum()/S
            per = d.pivot(index='rep', columns='target', values=f'{f}_PK')
            rrep = 100*per.sub(L0,axis=1).sum(1)/S
            # 单国层面：PK < L0 的比例
            dd = d.merge(L0.rename('L0').reset_index(), on='target')
            worse = 100*(dd[f'{f}_PK'] < dd['L0']).mean()
            gap = (gm[f'{f}_PK'] - (L0+base))
            print(f'  {ncl} 簇取 {m} 簇：恢复率 {rec:5.1f}%（按重复 5–95% [{np.percentile(rrep,5):.0f}, {np.percentile(rrep,95):.0f}]），'
                  f'无限标签极限 {rec_inf:5.1f}%；L1 {d.l1_pk.mean():.3f}（极限 {d.l1_inf.mean():.3f}）；'
                  f'单国×重复 OA<L0 {worse:.1f}%；PK−PO 均值 {gap.mean():+.2f} pp，最差国 {gap.idxmin()} {gap.min():+.2f}；'
                  f'见到的类 {d.n_cls_seen.mean():.1f}/{d.n_cls_pool.mean():.1f}；范围 {d.extent_km.median():.0f} km')
    # 20 簇 2 簇 各国恢复率
    d = df[(df.ncl==20)&(df.m==2)]; gm = d.groupby('target')[f'{f}_PK'].mean()
    q = d.groupby('target')[f'{f}_PK'].quantile(0.1)
    print('  20 簇取 2 簇，各国：恢复率(均值) / 10% 分位 / PK−PO(pp)')
    print(pd.DataFrame({'PO−L0':base, 'rec%':100*(gm-L0)/base, 'rec10%':100*(q-L0)/base, 'PK−PO':gm-(L0+base)}).round(1).T.to_string())
