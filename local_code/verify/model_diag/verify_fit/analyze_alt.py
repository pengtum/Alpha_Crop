import glob, json
import numpy as np, pandas as pd
OUT = '/Users/perng_mini/AlphaCrops_local/verify/model_diag/verify_fit'
L = []
for p in glob.glob(f'{OUT}/jobs/alt_*.json'):
    r = json.load(open(p)); j = r['job']
    for ck, m in r['curve'].items():
        L.append(dict(model=j['model'], tset=j['tset'], tgt=j['tgt'], feat=j['feat'], ck=str(ck), ba=m['test']['ba'], oa=m['test']['oa'],
                      tr_oa=m['train']['oa'], tr_ba=m['train']['ba'], loss=m.get('loss', np.nan), n_iter=m.get('n_iter', np.nan)))
a = pd.DataFrame(L)
a.to_csv(f'{OUT}/alt_long.csv', index=False)
pd.set_option('display.width', 250)
print('作业数', a.groupby(['model', 'tset']).tgt.nunique().to_dict())
for mo in ('mlp', 'lr'):
    x = a[a.model == mo]
    if len(x):
        print(f'\n== {mo} ==')
        print(x.groupby(['tset', 'feat', 'ck'])[['ba', 'oa', 'tr_oa', 'tr_ba', 'loss', 'n_iter']].mean().round(2).to_string())
# LightGBM 对照（同 5 国划分 1）
v = pd.read_csv(f'{OUT}/vf_long.csv')
v = v[(v.ev == 'test') & v.kind.isin(['k0', 'tgt']) & v.tgt.isin(a.tgt.unique())]
lg = v.groupby(['kind', 'feat', 'it'])[['ba', 'oa']].mean().round(2)
print('\nLightGBM 同国对照（第 50/100/300 轮）：\n', lg.loc[(slice(None), slice(None), [50, 100, 300]), :].to_string())
# 最终点与神谕点的差距 tgt − k0
def summ(x, final):
    f = x[x.ck == final].groupby(['tset', 'feat']).ba.mean().unstack(0)
    b = x.groupby(['tset', 'tgt', 'feat']).ba.max().groupby(['tset', 'feat']).mean().unstack(0)
    return f, b
for mo, final in (('mlp', '40'), ('lr', 'C1.0')):
    x = a[a.model == mo]
    if len(x) == 0: continue
    f, b = summ(x, final)
    print(f'\n{mo} 最终点 ba：\n', f.round(2).to_string(), '\n 差距 tgt−k0:', (f['tgt'] - f['k0']).round(2).to_dict())
    print(f'{mo} 神谕最优 ba：\n', b.round(2).to_string(), '\n 差距 tgt−k0:', (b['tgt'] - b['k0']).round(2).to_dict())
lk = v.groupby(['kind', 'tgt', 'feat']).ba.max().groupby(['kind', 'feat']).mean().unstack(0)
print('\nLightGBM 神谕最优 ba：\n', lk.round(2).to_string(), '\n 差距 tgt−k0:', (lk['tgt'] - lk['k0']).round(2).to_dict())
