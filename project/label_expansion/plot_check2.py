"""
检验 2 可视化：不采样（只用外国标签）的精度，以及各采样策略带来的增益。

图 1 fig_check2_curves：每个目标国 × 分类特征一格，横轴为目标国标签数，纵轴为 ba_ge30。
  虚线 = 不采样（外国池 5 万标签），点线 = 天花板（外国池 + 目标国全部候选池），
  随机与不确定性采样突出显示，其余 8 种策略为灰色细线。
图 2 fig_check2_gain：各策略相对"不采样"的增益（pp），按预算分栏；
  空心点为 8 个（目标国 × 分类特征）单元，实心点为平均。
输出：label_expansion/figures/*.png|pdf
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import RESULTS_DIR

plt.rcParams.update({'font.family': ['PingFang SC', 'Arial Unicode MS'], 'axes.unicode_minus': False,
                     'font.size': 9, 'axes.edgecolor': '#8a8984', 'axes.linewidth': 0.6,
                     'xtick.color': '#52514e', 'ytick.color': '#52514e', 'axes.labelcolor': '#0b0b0b'})
INK, INK2, MUTED, GRID = '#0b0b0b', '#52514e', '#8a8984', '#e7e6e2'
BLUE, ORANGE, GRAY = '#2a78d6', '#eb6834', '#b9b8b2'
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
REFS = ['foreign_only', 'target_full', 'ceiling_w25', 'foreign+target_nat']
NAMES = {'random': '随机', 'uncertainty': '不确定性', 'geo_strat': '空间分层', 's2_rep': 'S2 代表性',
         'ae_rep': 'AE 代表性', 's2_gap': 'S2 覆盖缺口', 'ae_gap': 'AE 覆盖缺口',
         'betti_admin_rep': 'Betti·行政区', 'betti_img_rep_ae': 'Betti·AE 分组', 'betti_img_rep_s2': 'Betti·S2 分组'}
TARGETS = ['PT', 'SK', 'SE', 'LV']
TNAME = {'PT': '葡萄牙 PT', 'SK': '斯洛伐克 SK', 'SE': '瑞典 SE', 'LV': '拉脱维亚 LV'}
FNAME = {'ae': 'AE 分类', 's2': 'S2 分类'}
BUDGETS = [50, 200, 1000]
XPOS = {0: 0, 50: 1, 200: 2, 1000: 3}


def load():
    a = pd.read_csv(os.path.join(RESULTS_DIR, 'pilot_selection.csv'))
    b = pd.read_csv(os.path.join(RESULTS_DIR, 'pilot_betti_parcel.csv'))
    b['strategy'] = 'betti_' + b['strategy']
    d = pd.concat([a, b[a.columns.intersection(b.columns)]], ignore_index=True)
    d['m'] = d['ba_ge30'] * 100
    return d


def style(ax):
    ax.grid(axis='y', color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)


def fig_curves(d):
    ref = d[d.strategy.isin(REFS)].groupby(['target', 'clf_feat', 'strategy'])['m'].mean()
    st = d[~d.strategy.isin(REFS)]
    mean = st.groupby(['target', 'clf_feat', 'strategy', 'budget'])['m'].mean()
    lo = st.groupby(['target', 'clf_feat', 'strategy', 'budget'])['m'].min()
    hi = st.groupby(['target', 'clf_feat', 'strategy', 'budget'])['m'].max()
    fig, axes = plt.subplots(2, 4, figsize=(11, 5.6), sharex=True)
    for j, t in enumerate(TARGETS):
        for i, f in enumerate(['ae', 's2']):
            ax = axes[i, j]
            style(ax)
            fo, ce = ref[(t, f, 'foreign_only')], ref[(t, f, 'ceiling_w25')]
            ax.hlines(fo, -0.2, 3.0, color=INK2, lw=1, ls=(0, (4, 3)))
            ax.hlines(ce, -0.2, 3.0, color=INK2, lw=1, ls=(0, (1, 2)))
            xs = [0] + [XPOS[k] for k in BUDGETS]
            for s in NAMES:
                if s in ('random', 'uncertainty'):
                    continue
                ys = [fo] + [mean[(t, f, s, k)] for k in BUDGETS]
                ax.plot(xs, ys, color=GRAY, lw=1, zorder=2)
            for s, c in [('random', BLUE), ('uncertainty', ORANGE)]:
                ys = [fo] + [mean[(t, f, s, k)] for k in BUDGETS]
                ax.fill_between(xs, [fo] + [lo[(t, f, s, k)] for k in BUDGETS],
                                [fo] + [hi[(t, f, s, k)] for k in BUDGETS], color=c, alpha=0.12, lw=0, zorder=3)
                ax.plot(xs, ys, color=c, lw=2, marker='o', ms=4.5, mec='white', mew=1, zorder=4)
            span = ce - fo
            ax.set_ylim(fo - 0.12 * span - 1, ce + 0.12 * span + 1)
            ax.text(3.08, fo, '不采样', va='center', ha='left', color=INK2, fontsize=8)
            ax.text(3.08, ce, '天花板', va='center', ha='left', color=INK2, fontsize=8)
            gain = mean[(t, f, 'random', 1000)] - fo
            ax.text(-0.1, ce - 0.05 * span, f'随机 1000 个：{gain:+.1f} pp\n天花板差距：{span:.1f} pp',
                    va='top', ha='left', fontsize=7.5, color=INK2)
            if i == 0:
                ax.set_title(TNAME[t], fontsize=10, color=INK, loc='left')
            if j == 0:
                ax.set_ylabel(f'{FNAME[f]}\n平衡精度 ba_ge30 (%)')
            ax.set_xticks(range(4), ['0', '50', '200', '1000'])
            ax.set_xlim(-0.2, 3.7)
            if i == 1:
                ax.set_xlabel('目标国标签数')
    h = [plt.Line2D([], [], color=BLUE, lw=2, marker='o', ms=4.5, mec='white', label='随机'),
         plt.Line2D([], [], color=ORANGE, lw=2, marker='o', ms=4.5, mec='white', label='不确定性采样'),
         plt.Line2D([], [], color=GRAY, lw=1, label='其余 8 种策略（含 Betti）'),
         plt.Line2D([], [], color=INK2, lw=1, ls=(0, (4, 3)), label='不采样：只用外国池 5 万标签'),
         plt.Line2D([], [], color=INK2, lw=1, ls=(0, (1, 2)), label='天花板：外国池 + 目标国全部候选池')]
    fig.legend(handles=h, loc='upper center', ncol=5, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle('加入目标国标签后的精度：各策略几乎重合，差距主要取决于目标国', y=1.045, fontsize=11.5,
                 color=INK, x=0.01, ha='left')
    fig.text(0.01, -0.01, '阴影 = 3 个种子的范围。主指标只统计测试集中 ≥30 个地块的类别。'
             '目标国样本权重固定为外国池的 25%。', fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def fig_gain(d):
    ref = d[d.strategy == 'foreign_only'].groupby(['target', 'clf_feat'])['m'].mean()
    st = d[~d.strategy.isin(REFS)]
    cell = st.groupby(['target', 'clf_feat', 'strategy', 'budget'])['m'].mean().reset_index()
    cell['gain'] = cell['m'] - cell.set_index(['target', 'clf_feat']).index.map(ref)
    order = (cell[cell.budget == 1000].groupby('strategy')['gain'].mean().sort_values().index.tolist())
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.4), sharey=True, sharex=True)
    rng = np.random.default_rng(0)
    for ax, k in zip(axes, BUDGETS):
        style(ax)
        ax.grid(axis='x', color=GRID, lw=0.6)
        ax.grid(axis='y', visible=False)
        ax.axvline(0, color=INK2, lw=1, ls=(0, (4, 3)))
        rmean = cell[(cell.budget == k) & (cell.strategy == 'random')]['gain'].mean()
        ax.axvline(rmean, color=BLUE, lw=1, alpha=0.5)
        for y, s in enumerate(order):
            g = cell[(cell.budget == k) & (cell.strategy == s)]
            c = BLUE if s == 'random' else ORANGE if s == 'uncertainty' else INK2
            ax.scatter(g['gain'], y + rng.uniform(-0.18, 0.18, len(g)), s=14, facecolor='none',
                       edgecolor=c, lw=0.8, alpha=0.7, zorder=3)
            ax.scatter(g['gain'].mean(), y, s=46, color=c, edgecolor='white', lw=1.2, zorder=4)
        ax.set_title(f'{k} 个目标国标签', fontsize=10, color=INK, loc='left')
        ax.set_xlabel('相对不采样的增益 (pp)')
    axes[0].set_yticks(range(len(order)), [NAMES[s] for s in order])
    fig.suptitle('各采样策略相对"不采样"的增益：与随机的差别远小于国家之间的差别', fontsize=11.5,
                 color=INK, x=0.01, ha='left')
    fig.text(0.01, -0.02, '实心点 = 4 个目标国 × 2 种分类特征的平均；空心点 = 各单元（3 个种子平均）。'
             '黑色虚线 = 不采样（增益 0）；蓝色竖线 = 随机的平均增益。', fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    d = load()
    for name, fig in [('fig_check2_curves', fig_curves(d)), ('fig_check2_gain', fig_gain(d))]:
        for ext in ('png', 'pdf'):
            fig.savefig(os.path.join(OUT_DIR, f'{name}.{ext}'), dpi=200, bbox_inches='tight')
        print('saved', name)


if __name__ == '__main__':
    main()
