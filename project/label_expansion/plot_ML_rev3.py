"""
rev3 补充估计量的图（论文计划图 7 与图 3(c)）：各学习器下
  (a) 特征效应 Δ_feat(L,k) = ba_ge30(AE 特征) − ba_ge30(S2 特征)（B1）
  (b) 同上，OA（B1）
  (c) 随机标签价值 G(L,k) = ba_ge30(SRS k) − ba_ge30(k=0)，两种特征平均（B2）
上行 11 国，下行去掉 RO、PT（9 国）。只读 analyze_ML_rev3.py 的正式输出 summary_ML_rev3.csv，不重算。
用法：python plot_ML_rev3.py   → figures/fig_ML_rev3.png|pdf
"""
import os

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from multilearner_G import OUT
from plot_check2 import INK, INK2, MUTED, GRID, BLUE, ORANGE, style

COL = {'lgb': INK, 'lr': BLUE, 'mlp': ORANGE, 'rf': '#1baf7a'}
SHORT = {'lgb': 'LightGBM', 'lr': 'Logistic 回归', 'mlp': 'MLP 集成', 'rf': '随机森林'}
ORDER = ['lgb', 'lr', 'mlp', 'rf']
DODGE = {'lgb': -9, 'lr': -3, 'mlp': 3, 'rf': 9}  # 横向错开（k 单位），避免误差线重叠
SUBSETS = [('11 国', '11 国'), ('去 RO、PT', '去 RO、PT')]


def panel(ax, T, ylab, title, ref=None):
    style(ax)
    ax.axhline(0, color=INK2, lw=0.9, ls=(0, (4, 3)))
    if ref is not None:
        ax.axhline(ref, color=MUTED, lw=0.8, ls=':')
    for L in ORDER:
        d = T[T.learner == L].sort_values('k')
        if d.empty:
            continue
        x = d.k + DODGE[L]
        ax.vlines(x, d.lo95, d.hi95, color=COL[L], lw=1.2, alpha=0.85)
        ax.plot(x, d['mean'], color=COL[L], lw=1.3, marker='o', ms=4, mec='white', mew=0.6)
    ax.set_xticks(sorted(T.k.unique()))
    ax.set_xlabel('随机目标标签数 k')
    ax.set_ylabel(ylab)
    ax.set_title(title, fontsize=9.5, color=INK, loc='left')


def criteria(S):
    """按 analyze_ML_rev3.py 的同一判据重算满足的学习器数（ba_ge30）。"""
    b = S[(S.metric == 'ba_ge30')]
    n1 = 0
    for L in ORDER:
        x = b[(b.block == 'B1') & (b.subset == '11 国') & (b.learner == L)].set_index('k')['mean']
        ch = b[(b.block == 'B1_change') & (b.subset == '11 国') & (b.learner == L)].iloc[0]
        n1 += bool((x[0] > 0) == (x[400] > 0) and ch.lo95 > -1 and ch.hi95 < 1)
    g = b[(b.block == 'B2') & (b.subset == '去 RO、PT') & (b.clf == 'avg') & (b.k == 200)]
    n2 = int((g.hi95 < 2).sum())
    return n1, n2


def main():
    S = pd.read_csv(os.path.join(OUT, 'summary_ML_rev3.csv'))
    assert 'NOISE' not in OUT
    n1, n2 = criteria(S)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2), gridspec_kw=dict(wspace=0.3, hspace=0.45))
    for r, (sub, lab) in enumerate(SUBSETS):
        b1 = S[(S.block == 'B1') & (S.subset == sub)]
        la, lb, lc = 'abc' if r == 0 else 'def'
        panel(axes[r, 0], b1[b1.metric == 'ba_ge30'], 'AE 特征 − S2 特征（pp）', f'({la}) 特征效应 · 平衡精度 · {lab}')
        panel(axes[r, 1], b1[b1.metric == 'oa'], 'AE 特征 − S2 特征（pp）', f'({lb}) 特征效应 · OA · {lab}')
        b2 = S[(S.block == 'B2') & (S.subset == sub) & (S.clf == 'avg')]
        panel(axes[r, 2], b2, 'SRS(k) − 只用外国（pp）', f'({lc}) 随机标签价值 · 平衡精度 · {lab}',
              ref=2 if sub == '去 RO、PT' else None)
        axes[r, 2].set_xlim(0, 450)
    # 同一指标的上下两行用同一纵轴，便于比较
    for c in range(3):
        lo = min(a.get_ylim()[0] for a in axes[:, c])
        hi = max(a.get_ylim()[1] for a in axes[:, c])
        for a in axes[:, c]:
            a.set_ylim(lo, hi)
    axes[0, 0].legend([Line2D([], [], color=COL[L], lw=1.6, marker='o', ms=4) for L in ORDER],
                      [SHORT[L] for L in ORDER], fontsize=7.5, frameon=False, ncol=4, loc='lower left',
                      bbox_to_anchor=(0, 1.1), handlelength=1.6, columnspacing=1.2)
    fig.suptitle('rev3 补充估计量：AE 作为特征的优势与随机标签价值（描述性；G s0/s1 测试集已复用；LR 的 SRS/k=0 行非盲）',
                 fontsize=11, color=INK, x=0.01, ha='left', y=1.0)
    fig.text(0.01, 0.0, '上行 11 国（df = 10），下行去掉 RO、PT（9 国，df = 8）。点 = 国家层面均值，竖线 = 95% CI（以国家为单位的 t 检验）；'
             '同一 k 的四个学习器横向错开显示。\n'
             '特征效应 Δ_feat(L,k) = 同一批 SRS 标签下 AE 特征 − S2 特征（k=0 为只用外国数据）；'
             '随机标签价值 G(L,k) = SRS(k) − 只用外国，两种特征平均。\n'
             f'rev3 判据：B1"整体平移"要求 ≥3/4 个学习器 k=0 与 k=400 同号、且 0→400 变化的 95% CI 在 ±1 pp 内（{n1}/4 满足）；'
             f'B2"边际价值小"要求每个学习器去掉 RO、PT 后 G(L,200) 的 95% 上界 < 2 pp（点线；{n2}/4 满足）。',
             fontsize=7.5, color=MUTED, va='top', linespacing=1.5)
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(d, f'fig_ML_rev3.{ext}'), dpi=200, bbox_inches='tight', pad_inches=0.3)
    print('写入', os.path.join(d, 'fig_ML_rev3.png|pdf'))


if __name__ == '__main__':
    main()
