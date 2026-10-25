"""
检验 3 事后补充分析：早停 vs 固定 300 棵树的链式分解对比（ANALYSIS_ADDENDUM_check3_capacity.md，非预注册）。
输出：RESULTS_DIR/check3_capacity_report.txt、check3_capacity_arms.csv
"""
import os
import sys

import numpy as np
import pandas as pd

from analyze_check3 import Tee, metrics, pooled, C
from check3_priors import as64, trained_mask, uniform_prior, reweight, label_set, laplace_prior
from common import COUNTRIES, RESULTS_DIR, RANDOM_STATE

D = os.path.join(RESULTS_DIR, 'check3')
MODELS = {'L0': 'loco', 'PM': 'loco_pm', 'CMN': 'cmn', 'Wnat': 'wnat', 'CMB': 'cmb', 'W0': 'within'}


def load(tgt, feat, model, cap):
    suffix = '_c300' if cap == 'c300' else ''
    return as64(np.load(os.path.join(D, f'{tgt}_{feat}_{model}{suffix}_test.npy')))


def main():
    sys.stdout = Tee(os.path.join(RESULTS_DIR, 'check3_capacity_report.txt'))
    print('事后稳健性检验（非预注册）：主链模型的训练容量，早停（es）对比固定 300 棵树（c300）')
    rng = np.random.default_rng(RANDOM_STATE)
    rows = []
    targets = [t for t in COUNTRIES
               if all(os.path.exists(os.path.join(D, f'{t}_{f}_within_c300_test.npy')) for f in ['ae', 's2'])]
    print(f'可用目标国（{len(targets)}）：{targets}')
    for tgt in targets:
        sp = np.load(os.path.join(D, f'{tgt}_split.npz'))
        y_pool, y_test = sp['y_pool'], sp['y_test']
        pi_T = np.bincount(y_pool, minlength=C) / len(y_pool)
        present = pi_T > 0
        for feat in ['ae', 's2']:
            for cap in ('es', 'c300'):
                pr = {arm: load(tgt, feat, m, cap) for arm, m in MODELS.items()}
                pi_tr = uniform_prior(trained_mask(pr['L0']))
                pr['LS'] = label_set(pr['L0'], present, pi_tr)[0]
                pr['PO'] = reweight(pr['L0'], pi_T, pi_tr)[0]
                for s in range(20):
                    lab = np.random.default_rng(1000 + s).choice(len(y_pool), 50, replace=False)
                    pr[f'PK50_s{s}'] = reweight(pr['L0'], laplace_prior(y_pool[lab], C), pi_tr)[0]
                for arm, p in pr.items():
                    rows.append(dict(target=tgt, feat=feat, cap=cap, arm=arm, **metrics(y_test, p.argmax(1))))
    A = pd.DataFrame(rows)
    A.to_csv(os.path.join(RESULTS_DIR, 'check3_capacity_arms.csv'), index=False)
    T = A.pivot_table(index=['feat', 'cap', 'target'], columns='arm', values=['oa', 'ba_ge30'])
    pk = [f'PK50_s{s}' for s in range(20)]
    for metric, chain, names in [('oa', ['L0', 'PM', 'CMN', 'Wnat'], ['类别构成', '样本量', '域残差']),
                                 ('ba_ge30', ['L0', 'LS', 'CMB', 'W0'], ['类别集合', '样本量', '域残差'])]:
        print(f'\n### {metric} 链：' + ' → '.join(chain))
        for feat in ['ae', 's2']:
            for cap in ('es', 'c300'):
                r = T[metric].loc[(feat, cap)]
                dfp = pd.DataFrame({'total': r[chain[-1]] - r[chain[0]]})
                for i, nm in enumerate(names):
                    dfp[nm] = r[chain[i + 1]] - r[chain[i]]
                parts = []
                for nm in names:
                    v, lo, hi, _, _ = pooled(dfp, nm, 'total', rng)
                    parts.append(f'{nm} {v:.0%} [{lo:.0%},{hi:.0%}]')
                print(f'  [{feat.upper()} {cap:4s}] L0 均值 {r[chain[0]].mean():.1f}，{chain[-1]} 均值 {r[chain[-1]].mean():.1f}，'
                      f'总损失 {dfp.total.mean():+.2f} pp；' + '；'.join(parts))
            if metric == 'oa':
                es = T['oa'].loc[(feat, 'es')]
                c3 = T['oa'].loc[(feat, 'c300')]
                print(f'  [{feat.upper()}] c300 − es：L0 {(c3.L0 - es.L0).mean():+.2f}，PM {(c3.PM - es.PM).mean():+.2f}，'
                      f'CMN {(c3.CMN - es.CMN).mean():+.2f}，Wnat {(c3.Wnat - es.Wnat).mean():+.2f} pp')
                for cap in ('es', 'c300'):
                    r = T['oa'].loc[(feat, cap)]
                    rec = (r[pk].mean(1) - r.L0).sum() / (r.PO - r.L0).sum()
                    print(f'  [{feat.upper()} {cap}] PO−L0 均值 {(r.PO - r.L0).mean():+.2f} pp；PK50（20 种子）恢复率 {rec:.0%}；'
                          f'PK50−PO 均值 {(r[pk].mean(1) - r.PO).mean():+.2f} pp')
                v = {cap: (T['oa'].loc[(feat, cap)].PM - T['oa'].loc[(feat, cap)].L0).sum() /
                     (T['oa'].loc[(feat, cap)].Wnat - T['oa'].loc[(feat, cap)].L0).sum() for cap in ('es', 'c300')}
                verdict = '稳健（两种设置均 ≥30%）' if min(v.values()) >= 0.30 else '依赖训练设置（固定容量下 <30%）'
                print(f'  → [{feat.upper()}] OA 类别构成合并比例：早停 {v["es"]:.0%}，固定 300 棵 {v["c300"]:.0%}；{verdict}')


if __name__ == '__main__':
    main()
