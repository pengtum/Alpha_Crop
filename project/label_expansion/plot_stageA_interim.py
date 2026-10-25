"""
阶段 A 期中可视化（只含指定国家、已完成的单元）：各方法相对 SRS 的 Δ（ba_ge30 与 OA，P0），k=50 与 k=1000。
点 = 单元平均，横线 = 分层 bootstrap 90% CI。按家族着色，神谕单列。
用法：python plot_stageA_interim.py PT
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_stageA import boot_ci
from plot_check2 import INK, INK2, MUTED, GRID, BLUE, ORANGE, style
from stageA import OUT

AQUA, VIOLET, GRAYC = '#1baf7a', '#4a3aa7', '#8a8984'
FAMILY = {
    '概率设计': (['srs_nuts3', 'lpm2', 'predstrat', 'predstrat_prop', 'predstrat_equal', 'pmht', 'cube', 'clhs'], BLUE),
    '不确定性与多样化': (['margin', 'entropy', 'margin_prior', 'margin_r2', 'gist_margin', 'gist_maxmin', 'gist_geo',
                  'kcenter', 'kcenter_foreign', 'badge', 'kdpp'], ORANGE),
    '覆盖与代表性': (['clue', 'uherding', 'maxherding', 'typiclust', 'typiclust_strat', 'fass_quota',
                'fass_quota_equal', 'probcover', 'kherding', 'fscs'], AQUA),
    '迁移感知': (['tqs', 'bait', 'hybrid'], VIOLET),
    '神谕（作弊）': (['oracle_prop', 'oracle_sqrt', 'oracle_equal', 'oracle_err'], GRAYC),
}


def main(tag):
    M = pd.read_csv(os.path.join(OUT, f'stageA_merged_interim_{tag}.csv'))
    unit = ['target', 'split', 'rep', 'k']
    base = M[M.method == 'srs'].set_index(unit)
    order, colors = [], []
    for fam, (ms, c) in FAMILY.items():
        for m in ms:
            if m in set(M.method):
                order.append(m)
                colors.append(c)
    fig, axes = plt.subplots(1, 4, figsize=(13, 8.6), sharey=True)
    for j, (k, metric, title) in enumerate([(50, 'ba_ge30', 'k=50 · 平衡精度'), (50, 'oa', 'k=50 · 总体精度'),
                                            (1000, 'ba_ge30', 'k=1000 · 平衡精度'), (1000, 'oa', 'k=1000 · 总体精度')]):
        ax = axes[j]
        style(ax)
        ax.grid(axis='y', visible=False)
        ax.grid(axis='x', color=GRID, lw=0.6)
        ax.axvline(0, color=INK2, lw=1, ls=(0, (4, 3)))
        xs = []
        for i, (m, c) in enumerate(zip(order, colors)):
            g = M[(M.method == m) & (M.k == k)].set_index(unit)
            d = (g[metric] - base.loc[g.index, metric]).to_numpy()
            lo, hi = boot_ci(d, g.index.get_level_values('target'))
            ax.plot([lo, hi], [i, i], color=c, lw=1.6, alpha=0.8)
            ax.scatter(d.mean(), i, s=30, color=c, edgecolor='white', lw=0.8, zorder=3)
            xs += [lo, hi]
        lim = max(abs(np.percentile(xs, 2)), abs(np.percentile(xs, 98)), 1.5)
        ax.set_xlim(-lim * 1.1, lim * 1.1)
        ax.set_title(title, fontsize=10, color=INK, loc='left')
        ax.set_xlabel('相对随机（SRS）的差（pp）')
    axes[0].set_yticks(range(len(order)), order)
    axes[0].invert_yaxis()
    h = [plt.Line2D([], [], marker='o', ls='-', color=c, label=f) for f, (_, c) in FAMILY.items()]
    fig.legend(handles=h, loc='upper center', ncol=5, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 1.0))
    n_units = M.groupby(['target', 'split', 'rep']).ngroups
    fig.suptitle(f'阶段 A 期中（{tag}，{n_units} 个单元）：各采样方法相对随机的差', fontsize=11.5, color=INK,
                 x=0.01, ha='left', y=1.04)
    fig.text(0.01, -0.01, '点 = 单元平均，横线 = 90% CI（单元层面 bootstrap，单元少时偏窄）。S2 分类器，300 棵树，目标样本权重 r=50，'
             '空间分块 + 1 km 缓冲划分，未做先验修正（P0）。期中结果，不作晋级判定。', fontsize=7.5, color=MUTED, va='top')
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures', f'fig_stageA_interim_{tag}.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    print('saved', out)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'PT')
