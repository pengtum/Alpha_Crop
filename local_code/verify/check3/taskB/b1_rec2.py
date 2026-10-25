import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
R = pd.read_pickle('b1_rows.pkl')
A = R[R.kind=='arms'].set_index(['feat','target'])
pk = R[R.kind!='arms']
for f in ['ae','s2']:
    a = A.loc[f]; base = a.PO_oa - a.L0_oa
    print(f'--- {f.upper()} --- EM(200) 收敛 {int(a.EM_conv.astype(bool).sum())}/11, EM(5000) 收敛 {int(a.EM5k_conv.astype(bool).sum())}/11')
    print(f'  RO 占 Σ(PO−L0) 的比例 {100*base["RO"]/base.sum():.0f}%；RO+SI+SK 占 {100*base[["RO","SI","SK"]].sum()/base.sum():.0f}%')
    for drop in ([], ['RO'], ['RO','SI']):
        k = [t for t in a.index if t not in drop]; S = base[k].sum()
        out = [f'EM {100*(a.EM_oa-a.L0_oa)[k].sum()/S:.0f}%']
        for kk in (10,20,50,100,200):
            o = pk[(pk.feat==f)&(pk.kind=='new')&(pk.k==kk)].groupby('target').oa.mean()
            out.append(f'PK{kk} {100*(o-a.L0_oa)[k].sum()/S:.0f}%')
        print(f'  去掉 {drop or "无"}：' + '  '.join(out))
    # 以 pp 表示的缺口（PK − PO）与 PK/EM − L0
    rows = {}
    for kk in (10,20,50,100,200):
        o = pk[(pk.feat==f)&(pk.kind=='new')&(pk.k==kk)].groupby('target').oa.mean()
        rows[f'PK{kk}−PO'] = o - a.PO_oa
    rows['EM−PO'] = a.EM_oa - a.PO_oa
    rows['EM−L0'] = a.EM_oa - a.L0_oa
    t = pd.DataFrame(rows)
    print('  pp 缺口（各国；均值行）：'); print(pd.concat([t, t.mean().to_frame('mean').T]).round(2).to_string())
    # 单种子 PK50 − L0 < 0 的比例（修正后反而比 L0 差）
    for kk in (10,20,50):
        g = pk[(pk.feat==f)&(pk.kind=='new')&(pk.k==kk)].merge(a.L0_oa.rename('L0').reset_index(), on='target')
        print(f'  PK{kk}：单国×单种子 OA < L0 的比例 {100*(g.oa<g.L0).mean():.1f}%，最大下降 {(g.oa-g.L0).min():+.2f} pp ({g.loc[(g.oa-g.L0).idxmin(),"target"]})')
