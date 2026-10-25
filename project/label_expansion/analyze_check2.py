"""
检验 2 分析（按 ANALYSIS_PLAN_check2.md v1，sha256[:16]=13a40f0d2a265943）。

主指标 ba_ge30（测试集地块数 ≥30 的类别上的平衡精度），次要指标 ba、macro_f1。
单元 = (目标国, 预算, 分类特征)，共 4 × 3 × 2 = 24 个；每个单元先对 3 个种子取平均。
判定"有用"：平均差 ≥ 1 pp，24 个单元上 Wilcoxon p < 0.05，且至少 3/4 个目标国方向一致。
补回比例 = (BA_k − BA_foreign_only) / (BA_ceiling_w25 − BA_foreign_only)。

用法：python analyze_check2.py [--noise]（--noise 把精度换成随机数，用于在不看结果的情况下测试脚本）
输出：RESULTS_DIR/check2_report.txt
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from common import RESULTS_DIR, RANDOM_STATE

METRICS = ['ba_ge30', 'ba', 'macro_f1']
REFS = ['foreign_only', 'target_full', 'ceiling_w25', 'foreign+target_nat']
MAIN = ['random', 'geo_strat', 's2_rep', 'ae_rep', 's2_gap', 'ae_gap', 'uncertainty']
BETTI = ['admin_rep', 'img_rep_ae', 'img_rep_s2']


class Tee:
    def __init__(self, path):
        self.f = open(path, 'w')

    def write(self, s):
        sys.__stdout__.write(s)
        self.f.write(s)

    def flush(self):
        self.f.flush()


def load(noise=False):
    a = pd.read_csv(os.path.join(RESULTS_DIR, 'pilot_selection.csv'))
    b = pd.read_csv(os.path.join(RESULTS_DIR, 'pilot_betti_parcel.csv'))
    b['strategy'] = b['strategy'].where(~b['strategy'].isin(BETTI), 'betti_' + b['strategy'])
    d = pd.concat([a, b[a.columns.intersection(b.columns)]], ignore_index=True)
    if noise:
        rng = np.random.default_rng(0)
        for m in METRICS:
            d[m] = rng.uniform(0.2, 0.6, len(d))
    for m in METRICS:
        d[m] = d[m] * 100
    return d


def compare(cell, a, b, metric, rng, label):
    """a − b 在 24 个单元上的比较，按预注册标准判定。"""
    x = cell[metric].unstack('strategy')
    diff = (x[a] - x[b]).dropna()
    by_t = diff.groupby(level='target').mean()
    agree = int((np.sign(by_t) == np.sign(diff.mean())).sum())  # 与 Δ 同号的目标国数
    p = wilcoxon(diff).pvalue if (diff != 0).any() else np.nan
    targets = by_t.index.to_numpy()
    boots = []
    for _ in range(5000):
        s = rng.choice(targets, len(targets))
        boots.append(np.mean([diff.xs(t, level='target').mean() for t in s]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    useful = diff.mean() >= 1 and p < 0.05 and (by_t > 0).sum() >= 3
    print(f'  {label:28s} Δ={diff.mean():+5.2f} pp  CI[{lo:+5.2f},{hi:+5.2f}]  '
          f'正向单元 {int((diff > 0).sum())}/{len(diff)}  Wilcoxon p={p:.3f}  '
          f'与Δ同向的目标国 {agree}/{len(by_t)}（{", ".join(f"{t}{v:+.1f}" for t, v in by_t.items())}）'
          f'  → {"有用" if useful else "不满足"}')
    return diff


def main():
    noise = '--noise' in sys.argv
    sys.stdout = Tee(os.path.join(RESULTS_DIR, 'check2_report_NOISE.txt' if noise else 'check2_report.txt'))
    if noise:
        print('!!! 噪声测试模式：精度已替换为随机数，结果无意义 !!!\n')
    d = load(noise)
    rng = np.random.default_rng(RANDOM_STATE)
    strat_rows = d[~d['strategy'].isin(REFS)]
    n_seed = strat_rows.groupby(['target', 'strategy', 'budget', 'clf_feat']).size()
    print(f'策略单元的种子数：{sorted(n_seed.unique())}；目标国 {sorted(d.target.unique())}')
    print(f'eff_share 范围（策略行）：{strat_rows.eff_share.min():.4f}–{strat_rows.eff_share.max():.4f}')

    ref = d[d['strategy'].isin(REFS)].groupby(['target', 'clf_feat', 'strategy'])[METRICS].mean()
    print('\n=== 参照线（3 个种子平均，ba_ge30 %）===')
    print(ref['ba_ge30'].unstack('strategy')[REFS].round(1).to_string())

    cell = strat_rows.groupby(['target', 'budget', 'clf_feat', 'strategy'])[METRICS].mean()

    for metric in METRICS:
        tag = '主指标' if metric == 'ba_ge30' else '次要指标'
        print(f'\n{"=" * 20} {tag} {metric} {"=" * 20}')
        tab = cell[metric].unstack('strategy')
        print('各策略均值（24 个单元平均）：' + '  '.join(
            f'{s}={tab[s].mean():.2f}' for s in MAIN + ['betti_' + s for s in BETTI] if s in tab))

        print('\n问题 1：AE 作为选样依据是否优于 S2？')
        d1 = compare(cell, 'ae_rep', 's2_rep', metric, rng, 'ae_rep − s2_rep')
        d2 = compare(cell, 'ae_gap', 's2_gap', metric, rng, 'ae_gap − s2_gap')
        comb = ((d1 + d2) / 2).rename('comb')
        tmp = pd.concat({'a': comb, 'b': comb * 0}, axis=1).stack().rename(metric)
        tmp.index = tmp.index.set_names('strategy', level=-1)
        compare(tmp.to_frame(), 'a', 'b', metric, rng, '合并（两组平均）')
        print('\n问题 1′（Betti 框架）：Image-Rep 用 AE 分组 vs 用 S2 分组')
        compare(cell, 'betti_img_rep_ae', 'betti_img_rep_s2', metric, rng, 'img_rep_ae − img_rep_s2')

        print('\n问题 2：各策略相对随机')
        for s in MAIN[1:]:
            compare(cell, s, 'random', metric, rng, f'{s} − random')
        print('问题 2′（Betti 框架）：Rep 类策略相对随机')
        for s in BETTI:
            compare(cell, 'betti_' + s, 'random', metric, rng, f'{s} − random')

        print('\n问题 3：random 策略的补回比例 (BA_k − BA_foreign_only)/(BA_ceiling − BA_foreign_only)')
        r = cell[metric].unstack('strategy')['random'].unstack('budget')
        fo = ref[metric].unstack('strategy')['foreign_only']
        ce = ref[metric].unstack('strategy')['ceiling_w25']
        closed = r.sub(fo, axis=0).div(ce - fo, axis=0)
        print(closed.map(lambda v: f'{v:.0%}').to_string())
        print('  平均：' + '  '.join(f'k={k}: {closed[k].mean():.0%}' for k in closed.columns))
        if metric != 'ba_ge30':
            continue
        print('\n  参考：外国池 5 万个标签 vs 目标国全部候选池（只用目标国）的差（pp）：')
        print((ref[metric].unstack('strategy')['target_full'] - fo).round(1).to_string())


if __name__ == '__main__':
    main()
