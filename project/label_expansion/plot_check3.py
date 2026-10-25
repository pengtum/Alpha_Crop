"""
检验 3 可视化：跨境损失的链式分解（每国一条横条）。
上行 OA 链（L0 → PM → CMN → W-nat）：类别构成 / 样本量 / 域残差；
下行 ba_ge30 链（L0 → LS → CMB → W0）：类别集合 / 样本量 / 域残差。
正值从 0 向右堆叠，负值向左堆叠；黑色竖线为总损失。
输出：label_expansion/figures/fig_check3_decomposition.png|pdf
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import RESULTS_DIR
from plot_check2 import INK, INK2, MUTED, GRID, BLUE, ORANGE, style

GRAY_SEG = '#b9b8b2'
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
CHAINS = {'oa': (['L0', 'PM', 'CMN', 'Wnat'], ['类别构成（先验 + 类别集合）', '样本量', '域残差（特征分布等）'],
                 '总体精度 OA'),
          'ba_ge30': (['L0', 'LS', 'CMB', 'W0'], ['类别集合', '样本量', '域残差（特征分布等）'],
                      '平衡精度 ba_ge30')}
COLORS = [BLUE, GRAY_SEG, ORANGE]
FNAME = {'ae': 'AE 分类', 's2': 'S2 分类'}


def main():
    A = pd.read_csv(os.path.join(RESULTS_DIR, 'check3_arms.csv'))
    A = A[A.split == 'random']
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2), sharey='row')
    for i, (metric, (chain, names, title)) in enumerate(CHAINS.items()):
        tab = A.pivot_table(index=['feat', 'target'], columns='arm', values=metric)
        order = (tab.loc['s2'][chain[-1]] - tab.loc['s2'][chain[0]]).sort_values().index
        for j, feat in enumerate(['ae', 's2']):
            ax = axes[i, j]
            style(ax)
            ax.grid(axis='y', visible=False)
            ax.grid(axis='x', color=GRID, lw=0.6)
            r = tab.loc[feat].reindex(order)
            parts = np.array([r[chain[k + 1]] - r[chain[k]] for k in range(3)])  # 3 × n
            total = r[chain[-1]] - r[chain[0]]
            y = np.arange(len(order))
            pos_left, neg_left = np.zeros(len(y)), np.zeros(len(y))
            for k in range(3):
                v = parts[k]
                left = np.where(v >= 0, pos_left, neg_left + v)
                ax.barh(y, np.abs(v), left=left, color=COLORS[k], height=0.62,
                        edgecolor='white', linewidth=1.2, label=names[k] if (i == 0 or k != 0) else None)
                pos_left = pos_left + np.where(v >= 0, v, 0)
                neg_left = neg_left + np.where(v < 0, v, 0)
            ax.scatter(total, y, marker='|', s=170, color=INK, lw=1.8, zorder=5)
            ax.axvline(0, color=INK2, lw=0.8)
            ax.set_yticks(y, order)
            share = parts[0].sum() / total.sum()
            ax.set_title(f'{title} · {FNAME[feat]}   {names[0][:4]}占 {share:.0%}',
                         fontsize=10, color=INK, loc='left')
            ax.set_xlabel('相对跨境模型（L0）的精度差（pp）')
            if i == 1:
                ax.set_xlim(-20, 42)
            else:
                ax.set_xlim(-3, 55)
    h = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COLORS] + \
        [plt.Line2D([], [], marker='|', ls='', color=INK, ms=12, mew=1.8)]
    fig.legend(h, ['类别构成 / 类别集合（OA 行含先验；ba 行只有类别集合）', '样本量', '域残差（特征分布、标签语义、类内构成）',
                   '总损失（本国模型 − 跨境模型）'],
               loc='upper center', ncol=4, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle('跨境损失的来源：按总体精度看，约一半是类别比例；按平衡精度看，几乎全是特征分布', fontsize=11.5,
                 color=INK, x=0.01, ha='left', y=1.045)
    fig.text(0.01, -0.02, 'OA 链：跨境模型 → 训练先验改为本国比例 → 按本国各类计数抽取外国样本 → 本国模型；'
             'ba 链：跨境模型 → 只预测本国存在的类 → 按本国计数抽样（类别平衡）→ 本国模型。'
             '国家按 S2 的总损失排序。\n图为预注册设置（早停）。事后稳健性检验（各模型固定 300 棵树）下，'
             'OA 的类别构成占比为 AE 46%、S2 44%；ba 链仍几乎全为域残差。', fontsize=7.5, color=MUTED, va='top')
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs(OUT_DIR, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(OUT_DIR, f'fig_check3_decomposition.{ext}'), dpi=200, bbox_inches='tight')
    print('saved')


if __name__ == '__main__':
    main()
