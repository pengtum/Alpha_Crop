"""第 2 步：逐单元算候选指标 → 与 prep_unified 相同的聚合（抽样平均 → 单元平均 → 域；Δ_L 取两特征平均）→ 任务层。"""
import os
import pickle
import numpy as np
import pandas as pd
from metrics import all_metrics, xfit_bal, xfit_w

HERE = os.path.dirname(os.path.abspath(__file__))
R = 3
KS = (50, 100, 200, 400)
pd.set_option('display.width', 250)
pd.set_option('display.max_columns', 60)
pd.set_option('display.max_rows', 400)

allres = pickle.load(open(os.path.join(HERE, 'curves.pkl'), 'rb'))
rows, xrows = [], []
for task, res in allres.items():
    for unit, U in res.items():
        cv = U['cv']
        base = dict(task=task, unit=unit, domain=U['domain'], primary=U['primary'])
        g = lambda k: cv[k]['all']  # noqa: E731
        # Δ_R：A = S2 Z0、B = AE Z0；锚点模型 = OA★ 较小者
        ref = 'A' if g(('s2', 'Z0', 0))['oa'][40] <= g(('ae', 'Z0', 0))['oa'][40] else 'B'
        m = all_metrics(g(('s2', 'Z0', 0)), g(('ae', 'Z0', 0)), ref)
        m.update(xfit_w({h: cv[('s2', 'Z0', 0)][h] for h in ('h1', 'h2')}, {h: cv[('ae', 'Z0', 0)][h] for h in ('h1', 'h2')}))
        rows += [dict(base, comp='R', feat='ae-s2', r=0, metric=k, value=v) for k, v in m.items()]
        x = xfit_bal({h: cv[('s2', 'Z0', 0)][h] for h in ('h1', 'h2')}, {h: cv[('ae', 'Z0', 0)][h] for h in ('h1', 'h2')})
        xrows.append(dict(base, comp='R', feat='ae-s2', r=0, xfit=x[0], insample_half=x[1], optimism_A=x[2]))
        for f in ('ae', 's2'):
            for arm, rs in [(f'RT{k}', range(R)) for k in KS] + [('REF', [0])]:
                for r in rs:
                    m = all_metrics(g((f, 'Z0', 0)), g((f, arm, r)), 'A')
                    m.update(xfit_w({h: cv[(f, 'Z0', 0)][h] for h in ('h1', 'h2')}, {h: cv[(f, arm, r)][h] for h in ('h1', 'h2')}))
                    rows += [dict(base, comp=arm, feat=f, r=r, metric=k, value=v) for k, v in m.items()]
                    x = xfit_bal({h: cv[(f, 'Z0', 0)][h] for h in ('h1', 'h2')}, {h: cv[(f, arm, r)][h] for h in ('h1', 'h2')})
                    xrows.append(dict(base, comp=arm, feat=f, r=r, xfit=x[0], insample_half=x[1], optimism_A=x[2]))
U = pd.DataFrame(rows)
X = pd.DataFrame(xrows)
U.to_parquet(os.path.join(HERE, 'unit_metrics.parquet'))
X.to_parquet(os.path.join(HERE, 'xfit.parquet'))


def dom_table(U):
    x = U.groupby(['task', 'domain', 'primary', 'unit', 'comp', 'feat', 'metric']).value.mean().reset_index()
    d = x.groupby(['task', 'domain', 'primary', 'comp', 'feat', 'metric']).value.mean().reset_index()
    fe = d.set_index(['task', 'domain', 'primary', 'comp', 'metric', 'feat']).value.unstack('feat')
    both = fe[['ae', 's2']].mean(axis=1, skipna=False)
    out = pd.concat([fe['ae-s2'].dropna().rename('value').reset_index().assign(feat='R'),
                     both.dropna().rename('value').reset_index().assign(feat='both'),
                     fe['ae'].dropna().rename('value').reset_index().assign(feat='ae'),
                     fe['s2'].dropna().rename('value').reset_index().assign(feat='s2'),
                     (fe['ae'] - fe['s2']).dropna().rename('value').reset_index().assign(feat='inter')])
    return out


D = dom_table(U)
D.to_parquet(os.path.join(HERE, 'domain_metrics.parquet'))
print('done', len(U), len(D))
