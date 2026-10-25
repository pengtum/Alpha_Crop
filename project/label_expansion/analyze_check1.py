"""
检验 1 分析（按 ANALYSIS_PLAN_check1.md 预注册方案）。

输入：pairwise_v2.csv、domain_distance.csv
输出：check1_report.txt、check1_per_target.csv
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

from common import COUNTRIES, RESULTS_DIR, RANDOM_STATE

DESCRIPTORS = ['mmd_ae', 'mmd_s2', 'c2st_ae', 'c2st_s2', 'geo_km', 'climate_diff', 'class_js']
LABEL_FREE = DESCRIPTORS[:-1]


class Tee:
    def __init__(self, path):
        self.f = open(path, 'w')

    def write(self, s):
        sys.__stdout__.write(s)
        self.f.write(s)

    def flush(self):
        self.f.flush()


def main():
    sys.stdout = Tee(os.path.join(RESULTS_DIR, 'check1_report.txt'))
    pw = pd.read_csv(os.path.join(RESULTS_DIR, 'pairwise_v2.csv'))
    dd = pd.read_csv(os.path.join(RESULTS_DIR, 'domain_distance.csv'))
    sym = pd.concat([dd, dd.rename(columns={'c1': 'c2', 'c2': 'c1'})], ignore_index=True)
    cross = pw[pw['source'] != pw['target']].merge(
        sym, left_on=['source', 'target'], right_on=['c1', 'c2'])
    assert len(cross) == 110 * 3, len(cross)

    within = pw[pw['source'] == pw['target']].pivot(index='source', columns='feature', values='ba')
    mean_cross = cross.pivot_table(index='feature', values='ba')
    print('=== 迁移精度概览（v2 数据，平衡精度 %）===')
    print('国内测试（源国 20% 留出）均值：', (within.mean() * 100).round(2).to_dict())
    print('跨国 110 对均值：', (mean_cross['ba'] * 100).round(2).to_dict())

    # 每个目标国：10 个候选源国上 Spearman(−距离, BA)
    per = []
    for feat in ['ae', 's2', 'fusion']:
        for tgt in COUNTRIES:
            g = cross[(cross['feature'] == feat) & (cross['target'] == tgt)]
            best_ba = g['ba'].max()
            for d in DESCRIPTORS:
                # 距离在 10 个源国上全部相同时（如 BE、RO 的气候带）ρ 无定义，记为 NaN 并从均值中剔除
                rho = spearmanr(-g[d], g['ba']).correlation
                # 距离并列时，top-1 与 regret 取并列源国上的期望（等价于随机打破并列）
                tied = g[np.isclose(g[d], g[d].min())]
                per.append(dict(feature=feat, target=tgt, descriptor=d, rho=rho,
                                top1=float((tied['ba'] == best_ba).mean()),
                                regret=100 * float((best_ba - tied['ba']).mean())))
    per = pd.DataFrame(per)
    per.to_csv(os.path.join(RESULTS_DIR, 'check1_per_target.csv'), index=False)

    rng = np.random.default_rng(RANDOM_STATE)

    def boot_ci(v, B=5000):
        v = np.asarray(v)
        v = v[~np.isnan(v)]
        m = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(B)]
        return np.percentile(m, [2.5, 97.5])

    print('\n=== 主指标：每个目标国的 Spearman ρ（−距离 vs 迁移 BA），目标国平均（ρ 无定义的目标国剔除）===')
    tab = per.pivot_table(index='descriptor', columns='feature', values='rho').reindex(DESCRIPTORS)
    tab['avg'] = tab.mean(1)
    print(tab.round(3).to_string())

    print('\n三种分类特征平均后，每个目标国的 ρ 的均值与 95% bootstrap CI（按目标国重抽）：')
    avg_t = per.groupby(['descriptor', 'target'])['rho'].mean().unstack()
    for d in DESCRIPTORS:
        lo, hi = boot_ci(avg_t.loc[d].to_numpy())
        print(f'  {d:13s} {avg_t.loc[d].mean():+.3f}  [{lo:+.3f}, {hi:+.3f}]')

    print('\n=== 主比较：AE-MMD vs S2-MMD（11 个目标国配对）===')
    for feat in ['ae', 's2', 'fusion', 'avg']:
        if feat == 'avg':
            a, b = avg_t.loc['mmd_ae'], avg_t.loc['mmd_s2']
        else:
            p = per[per['feature'] == feat].pivot(index='target', columns='descriptor', values='rho')
            a, b = p['mmd_ae'], p['mmd_s2']
        diff = (a - b).dropna().to_numpy()
        stat = wilcoxon(diff) if np.any(diff != 0) else None
        lo, hi = boot_ci(diff)
        print(f'  分类特征={feat:6s} Δρ(AE−S2)={diff.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]  '
              f'AE 更好的目标国 {int((diff > 0).sum())}/11  Wilcoxon p={stat.pvalue if stat else float("nan"):.3f}')

    print('\n=== 同样比较：AE-MMD vs 地理距离、气候带；C2ST 版本 ===')
    for x, y in [('mmd_ae', 'geo_km'), ('mmd_ae', 'climate_diff'), ('mmd_s2', 'geo_km'),
                 ('c2st_ae', 'c2st_s2'), ('mmd_ae', 'class_js')]:
        diff = (avg_t.loc[x] - avg_t.loc[y]).dropna().to_numpy()
        stat = wilcoxon(diff) if np.any(diff != 0) else None
        print(f'  {x} − {y}: Δρ={diff.mean():+.3f}  {int((diff > 0).sum())}/{len(diff)}  '
              f'p={stat.pvalue if stat else float("nan"):.3f}')

    print('\n=== 敏感性：去掉 RO（标签体系不同，只有 3 类）后的主比较 ===')
    sub = avg_t.drop(columns='RO')
    for x, y in [('mmd_ae', 'mmd_s2'), ('mmd_ae', 'geo_km')]:
        diff = (sub.loc[x] - sub.loc[y]).dropna().to_numpy()
        print(f'  {x} − {y}: Δρ={diff.mean():+.3f}  {int((diff > 0).sum())}/{len(diff)}  p={wilcoxon(diff).pvalue:.3f}'
              f'   （{x} 平均 ρ={sub.loc[x].mean():.3f}，{y}={sub.loc[y].mean():.3f}）')

    print('\n=== 判定（按预注册标准）===')
    d_as = (avg_t.loc['mmd_ae'] - avg_t.loc['mmd_s2']).dropna()
    d_ag = (avg_t.loc['mmd_ae'] - avg_t.loc['geo_km']).dropna()
    ok = d_as.mean() > 0 and wilcoxon(d_as).pvalue < 0.05 and d_ag.mean() >= 0
    print('  "AE 是更好的域地图"：' + ('成立' if ok else '不成立 → 判为"两者相当"'))
    print('  注：C2ST-AE 在 55 对中 38 对 AUC=1.0（饱和），其 ρ 不能解读为 AE 域信息更少。')

    print('\n=== 次要：top-1 命中率与 regret（选距离最近的源国，比最佳源国少多少 BA，pp）===')
    sec = per.groupby(['descriptor', 'feature']).agg(top1=('top1', 'mean'), regret=('regret', 'mean')).unstack()
    print(sec.reindex(DESCRIPTORS).round(3).to_string())
    rnd = []
    for feat in ['ae', 's2', 'fusion']:
        for tgt in COUNTRIES:
            g = cross[(cross['feature'] == feat) & (cross['target'] == tgt)]
            rnd.append(100 * (g['ba'].max() - g['ba'].mean()))
    print(f'  随机挑源国的期望 regret：{np.mean(rnd):.2f} pp')

    print('\n=== 次要：带目标国固定效应的偏 R²（BA ~ 距离 + 目标国 FE）===')
    for feat in ['ae', 's2', 'fusion']:
        g = cross[cross['feature'] == feat].copy()
        g['ba_dm'] = g['ba'] - g.groupby('target')['ba'].transform('mean')
        out = []
        for d in DESCRIPTORS:
            x = g[d] - g.groupby('target')[d].transform('mean')
            r = np.corrcoef(x, g['ba_dm'])[0, 1]
            out.append(f'{d}={r ** 2:.3f}')
        print(f'  {feat:6s} ' + '  '.join(out))

    print('\n=== 描述量之间的相关（55 个国家对，Spearman）===')
    print(dd[DESCRIPTORS].corr(method='spearman').round(2).to_string())


if __name__ == '__main__':
    main()
