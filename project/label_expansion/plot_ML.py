"""
多学习器稳健性图（rev2）的版式修订版。analyze_ML.py 已登记哈希、不改动；它生成的 fig_multilearner 右栏图例压住了曲线
（LR 与 RF 的 k=0 点）。本脚本只读 summary_ML.csv 与各学习器的行，重画同一张图并覆盖 figures/fig_multilearner.png|pdf。
改动只在版式：图例移到学习曲线面板右侧并拆成"学习器（颜色）"与"特征（线型）"两组；两个 θ_E1 面板用同一横轴范围，等价带宽度可比。
数字与 analyze_ML.py 完全相同（同一 summary 与同一学习曲线聚合式）。重跑 analyze_ML.py 后需再运行本脚本。
用法：python plot_ML.py
"""
import os

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

import analyze_G as A
import analyze_ML as M
from multilearner_G import OUT
from plot_check2 import INK, INK2, MUTED, GRID, BLUE, ORANGE, style

COL = {'lgb': INK, 'lr': BLUE, 'mlp': ORANGE, 'rf': '#1baf7a'}
SHORT = {'lgb': 'LightGBM', 'lr': 'Logistic 回归', 'mlp': 'MLP 集成', 'rf': '随机森林'}


def main():
    S = pd.read_csv(os.path.join(OUT, 'summary_ML.csv'))
    R = M.load_all()
    problems = M.completeness(R)
    assert not problems, problems
    if 'status' in R:
        R = R[R.status.fillna('ok') != 'failed']
    # 与 analyze_ML.main 相同的学习曲线聚合式：单元 → 国家 → 11 国均值
    lc = R[R.method.isin(['srs', 'foreign_only'])].groupby(['learner', 'clf', 'k', 'target'])[['ba_ge30', 'oa']].mean() \
        .groupby(['learner', 'clf', 'k']).mean().unstack('k')

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), gridspec_kw=dict(width_ratios=[1.1, 1.1, 1.3], wspace=0.38))
    th = S[S.what == 'theta_E1']
    lim = max(0.65, th[['lo95', 'hi95']].abs().max().max() + 0.05)
    for ax, met, title in ((axes[0], 'ba_ge30', 'θ_E1 · 平衡精度（ba_ge30）'), (axes[1], 'oa', 'θ_E1 · OA')):
        style(ax)
        ax.grid(axis='y', visible=False)
        ax.grid(axis='x', color=GRID, lw=0.6)
        ax.axvspan(-A.SESOI, A.SESOI, color='#eceae4', zorder=0)
        ax.axvline(0, color=INK2, lw=0.9, ls=(0, (4, 3)))
        sub = th[th.metric == met].set_index('learner')
        Ls = [L for L in M.ORDER if L in sub.index]
        for i, L in enumerate(Ls):
            s = sub.loc[L]
            ax.plot([s.lo95, s.hi95], [i, i], color=COL[L], lw=1.2)
            ax.plot([s.lo90, s.hi90], [i, i], color=COL[L], lw=3.2)
            ax.scatter(s['mean'], i, s=40, color=COL[L], edgecolor='white', lw=0.8, zorder=3)
            verdict = '等价' if str(s.verdict).startswith('等价') else '不确定'
            ax.text(-lim + 0.02, i - 0.2, f'{s["mean"]:+.2f} · {verdict}', ha='left', va='bottom', fontsize=7.5,
                    color=INK2)
        ax.set_yticks(range(len(Ls)), [SHORT[L] for L in Ls])
        ax.set_ylim(len(Ls) - 0.5, -0.7)
        ax.set_xlim(-lim, lim)
        ax.set_title(title, fontsize=10, color=INK, loc='left')
        ax.set_xlabel('AE 指导 − S2 指导（pp）')

    ax = axes[2]
    style(ax)
    Ls = [L for L in M.ORDER if L in lc.index.get_level_values(0)]
    for L in Ls:
        for clf, ls in (('s2', '-'), ('ae', '--')):
            if (L, clf) in lc.index:
                y = lc.loc[(L, clf), 'ba_ge30']
                ax.plot(y.index, y.values, ls=ls, color=COL[L], lw=1.5, marker='o', ms=3.5)
    ax.set_xlabel('随机目标标签数 k')
    ax.set_ylabel('平衡精度 ba_ge30（%）')
    ax.set_xticks([0, 50, 100, 200, 400])
    ax.set_title('随机选样学习曲线（11 国均值）', fontsize=10, color=INK, loc='left')
    h_l = [Line2D([], [], color=COL[L], lw=2) for L in Ls]
    h_f = [Line2D([], [], color=INK2, lw=1.5, ls='-'), Line2D([], [], color=INK2, lw=1.5, ls='--')]
    leg1 = ax.legend(h_l, [SHORT[L] for L in Ls], fontsize=7.5, frameon=False, loc='upper left',
                     bbox_to_anchor=(1.02, 1.0), handlelength=1.6, title='学习器', title_fontsize=7.5, alignment='left')
    ax.add_artist(leg1)
    ax.legend(h_f, ['S2 特征', 'AE 特征'], fontsize=7.5, frameon=False, loc='upper left',
              bbox_to_anchor=(1.02, 0.62), handlelength=2.2, title='分类器特征', title_fontsize=7.5, alignment='left')

    fig.suptitle('多学习器稳健性分析（rev2，预先指定；G s0/s1 测试集已复用，非确认性）', fontsize=11.5, color=INK,
                 x=0.01, ha='left', y=1.03)
    fig.text(0.01, -0.04, '灰带 = 等价界 ±0.3 pp；粗线 = 90% CI（TOST 判定用），细线 = 95% CI（11 国 t 检验，df = 10）。'
             '每行左上为均值与该学习器的判定。学习曲线：实线 = S2 特征、虚线 = AE 特征；k=0 为只用外国数据。\n'
             '训练集、权重、选样与测试集与 G 相同（外国 5 万 + k 个目标样本，目标权重 ×50，类别权重按外国频率）；LR、MLP、RF 在 G 已保存的选样上重训。',
             fontsize=7.5, color=MUTED, va='top', linespacing=1.5)
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(d, f'fig_multilearner.{ext}'), dpi=200, bbox_inches='tight', pad_inches=0.35)
    print('写入', os.path.join(d, 'fig_multilearner.png|pdf'))


if __name__ == '__main__':
    main()
