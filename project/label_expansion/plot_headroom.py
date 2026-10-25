"""
提升空间诊断的可视化。
每个目标国一栏；上行 ba_ge30，下行 OA；每栏按预算（50、200）分组：
  20 次随机选样（点 + 箱线）、两种按真实类别分层的作弊上界（等量、√比例）、不采样基线（虚线）。
栏内文字：诚实的 best-of-20（测试集一半上挑、另一半上报告）相对随机中位数的增益（来自 headroom_summary.csv）。
输出：label_expansion/figures/fig_headroom.png|pdf
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import RESULTS_DIR
from plot_check2 import INK, INK2, MUTED, GRID, BLUE, ORANGE, style

AQUA = '#1baf7a'
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
TNAME = {'PT': '葡萄牙 PT', 'SE': '瑞典 SE', 'SK': '斯洛伐克 SK', 'LV': '拉脱维亚 LV'}


def main():
    H = pd.read_csv(os.path.join(RESULTS_DIR, 'headroom.csv'))
    Sm = pd.read_csv(os.path.join(RESULTS_DIR, 'headroom_summary.csv'))
    H['oa'] *= 100
    H['ba_ge30'] *= 100
    targets = [t for t in ['PT', 'SK', 'SE', 'LV'] if t in set(H.target)]
    fig, axes = plt.subplots(2, len(targets), figsize=(2.9 * len(targets), 6.4))
    rng = np.random.default_rng(0)
    for j, t in enumerate(targets):
        for i, (metric, mname, sname) in enumerate([('ba_ge30', '平衡精度 ba_ge30 (%)', 'ba_ge30'),
                                                    ('oa', '总体精度 OA (%)', 'oa')]):
            ax = axes[i, j]
            style(ax)
            fo = H[(H.target == t) & (H.strategy == 'foreign_only')][metric].iloc[0]
            ax.axhline(fo, color=INK2, lw=1, ls=(0, (4, 3)), zorder=1)
            vals = [fo]
            for g, k in enumerate([50, 200]):
                x0 = g * 1.0
                r = H[(H.target == t) & (H.strategy == 'random') & (H.budget == k)][metric].to_numpy()
                ax.boxplot(r, positions=[x0], widths=0.28, showfliers=False, patch_artist=True,
                           boxprops=dict(facecolor='#e8f0fb', edgecolor=BLUE, lw=1),
                           medianprops=dict(color=BLUE, lw=1.6), whiskerprops=dict(color=BLUE, lw=1),
                           capprops=dict(color=BLUE, lw=1), zorder=2)
                ax.scatter(x0 + rng.uniform(-0.1, 0.1, len(r)), r, s=9, color=BLUE, alpha=0.55, lw=0, zorder=3)
                vals += list(r)
                for dx, strat, c, mk in ((0.3, 'oracle_equal', ORANGE, 'D'), (0.42, 'oracle_sqrt', AQUA, 's')):
                    o = H[(H.target == t) & (H.strategy == strat) & (H.budget == k)][metric]
                    if len(o):
                        ax.errorbar(x0 + dx, o.mean(), yerr=[[o.mean() - o.min()], [o.max() - o.mean()]], fmt=mk,
                                    ms=5.5, color=c, mec='white', mew=0.8, elinewidth=1, capsize=0, zorder=4)
                        vals += list(o)
                s = Sm[(Sm.target == t) & (Sm.k == k) & (Sm.metric == sname)]
                if len(s):
                    ax.text(x0, 0.02, f'挑最好\n{s.U2_gain.iloc[0]:+.1f}', transform=ax.get_xaxis_transform(),
                            ha='center', va='bottom', fontsize=7, color=INK2, linespacing=1.0)
            lo, hi = min(vals), max(vals)
            pad = 0.18 * (hi - lo + 1)
            ax.set_ylim(lo - pad, hi + 0.08 * (hi - lo + 1))
            ax.set_xticks([0.15, 1.15], ['k=50', 'k=200'])
            ax.set_xlim(-0.35, 1.65)
            if i == 0:
                ax.set_title(TNAME[t], fontsize=10, color=INK, loc='left')
            if j == 0:
                ax.set_ylabel(mname)
    h = [plt.Line2D([], [], marker='o', ls='', color=BLUE, ms=5, alpha=0.7, label='随机选样（20 次）'),
         plt.Line2D([], [], marker='D', ls='', color=ORANGE, ms=5.5, label='作弊：按真实类别等量分层（5 次）'),
         plt.Line2D([], [], marker='s', ls='', color=AQUA, ms=5.5, label='作弊：按真实类别 √比例分层（5 次）'),
         plt.Line2D([], [], color=INK2, lw=1, ls=(0, (4, 3)), label='不采样（只用外国标签）')]
    fig.legend(handles=h, loc='upper center', ncol=4, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle('选样的提升空间：随机子集之间差别很小；按真实类别均衡能提高平衡精度，但会拉低总体精度',
                 fontsize=11.5, color=INK, x=0.01, ha='left', y=1.05)
    fig.text(0.01, -0.02, '"挑最好"= 在测试集一半上从 20 个随机子集中挑最好的一个、在另一半上报告，相对随机中位数的增益（pp），'
             '是可实现的子集选择上界。\n误差线 = 5 次的范围。S2 分类，目标样本权重为外国样本的 10 倍；'
             '划分与检验 2 相同（目标国内随机 70/30）。', fontsize=7.5, color=MUTED, va='top')
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs(OUT_DIR, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(OUT_DIR, f'fig_headroom.{ext}'), dpi=200, bbox_inches='tight')
    print('saved')


if __name__ == '__main__':
    main()
