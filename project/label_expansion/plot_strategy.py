"""
策略基准图（G 实验的描述性结果，非预注册主假设；方法 − SRS 为冻结方案中的 R3 描述量）。
第一行：各策略相对 SRS 的差（国家层面均值与未校正 95% CI），AE 指导 vs S2 指导；平衡精度与 OA × k=50 与 200。
第二行左：逐国热图（平衡精度，方法 − SRS，对指导、分类器、k 取平均）；右：风险（方法比 SRS 低 0.5 pp 以上的比例）。
输出：label_expansion/figures/fig_strategy_benchmark.png|pdf
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap

import analyze_G as A
from guidance_G import E1, PRED, HYB
from plot_check2 import INK, INK2, MUTED, GRID, BLUE, ORANGE, style
from plot_G import wrap_mixed

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
METHODS = E1 + PRED + HYB
LABEL = {'lpm2_nogeo': 'LPM2（空间平衡）', 'kcenter': 'k-center（覆盖）', 'typiclust': 'TypiClust（典型性）',
         'kherding': 'Kernel herding（分布匹配）', 'margin': 'Margin（不确定性）', 'predstrat_srs': '按预测类分层',
         'gist_margin': 'GIST（不确定性 + 多样性）', 'badge_c': 'BADGE'}
FAMILY = {**{m: '只用几何' for m in E1}, **{m: '只用预测' for m in PRED}, **{m: '几何 + 预测' for m in HYB}}
GUIDE = {'AE': BLUE, 'S2': ORANGE}


def summarize(R):
    rows = []
    for m in METHODS:
        for cond in ('AE', 'S2'):
            for metric in ('ba_ge30', 'oa'):
                d = A.vs_srs(R, cond, [m], metric)
                for k in (50, 200):
                    s = A.t_summary(A.country_values(d[d.k == k]))
                    rows.append(dict(method=m, cond=cond, metric=metric, k=k, **s))
    return pd.DataFrame(rows)


def main():
    R = A.load_rows()
    S = summarize(R)
    fig = plt.figure(figsize=(13.5, 10.2))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.95], hspace=0.30)
    gs = outer[0].subgridspec(1, 4, wspace=0.12)
    gb = outer[1].subgridspec(1, 3, width_ratios=[3.0, 0.06, 1.0], wspace=0.14)
    ys = np.arange(len(METHODS))
    off = {'AE': -0.17, 'S2': 0.17}
    panels = [('ba_ge30', 50, '平衡精度 · k=50'), ('ba_ge30', 200, '平衡精度 · k=200'),
              ('oa', 50, 'OA · k=50'), ('oa', 200, 'OA · k=200')]
    lim = {'ba_ge30': max(2.2, S[S.metric == 'ba_ge30'][['lo95', 'hi95']].abs().max().max() * 1.05),
           'oa': max(2.2, S[S.metric == 'oa'][['lo95', 'hi95']].abs().max().max() * 1.05)}
    for j, (metric, k, title) in enumerate(panels):
        ax = fig.add_subplot(gs[0, j])
        style(ax)
        ax.grid(axis='y', visible=False)
        ax.grid(axis='x', color=GRID, lw=0.6)
        ax.axvline(0, color=INK2, lw=1, ls=(0, (4, 3)), zorder=1)
        for cond in ('AE', 'S2'):
            sub = S[(S.metric == metric) & (S.k == k) & (S.cond == cond)].set_index('method').loc[METHODS]
            y = ys + off[cond]
            ax.hlines(y, sub.lo95, sub.hi95, color=GUIDE[cond], lw=1.6, zorder=2)
            ax.scatter(sub['mean'], y, s=30, color=GUIDE[cond], edgecolor='white', lw=0.8, zorder=3,
                       label=f'{cond} 指导')
        for yy in (3.5, 5.5):
            ax.axhline(yy, color=GRID, lw=0.8)
        ax.set_ylim(len(METHODS) - 0.5, -0.5)
        ax.set_xlim(-lim[metric], lim[metric])
        ax.set_title(title, fontsize=10, color=INK, loc='left')
        ax.set_xlabel('方法 − 随机选样（pp）')
        if j == 0:
            ax.set_yticks(ys, [LABEL[m] for m in METHODS])
        else:
            ax.set_yticks(ys, [''] * len(METHODS))
            ax.tick_params(axis='y', length=0)
        if j == 3:
            ax.legend(loc='lower right', fontsize=8, frameon=False, handletextpad=0.3)

    # 逐国热图：平衡精度，方法 − SRS，对指导、分类器、k 平均
    ax = fig.add_subplot(gb[0, 0])
    frames = []
    for m in METHODS:
        for cond in ('AE', 'S2'):
            d = A.vs_srs(R, cond, [m])
            frames.append(d.assign(cond=cond))
    Dd = pd.concat(frames)
    H = Dd.groupby(['method', 'target']).d.mean().unstack('target').loc[METHODS]
    gain = (R[(R.method == 'srs') & (R.k == 200)].groupby('target').ba_ge30.mean()
            - R[R.method == 'foreign_only'].groupby('target').ba_ge30.mean())
    order = gain.sort_values(ascending=False).index
    H = H[order]
    v = float(np.nanmax(np.abs(H.to_numpy())))
    cmap = LinearSegmentedColormap.from_list('div', ['#b4441c', '#ea9a73', '#f1efe9', '#7fa9dc', '#1f5fae'])
    im = ax.imshow(H.to_numpy(), cmap=cmap, norm=TwoSlopeNorm(0, -v, v), aspect='auto')
    for i in range(H.shape[0]):
        for jj in range(H.shape[1]):
            val = H.iat[i, jj]
            ax.text(jj, i, f'{val:+.1f}', ha='center', va='center', fontsize=7.3,
                    color='white' if abs(val) > 0.6 * v else INK)
    ax.set_yticks(range(len(METHODS)), [LABEL[m] for m in METHODS])
    ax.set_xticks(range(len(order)), [f'{c}\n{gain[c]:+.1f}' for c in order])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title('逐国：平衡精度，方法 − 随机选样（pp）', fontsize=10, color=INK, loc='left')
    cb = fig.colorbar(im, cax=fig.add_subplot(gb[0, 1]))
    cb.set_label('pp', fontsize=8)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7.5)

    # 风险：方法比 SRS 低 0.5 pp 以上的比例（按国家平均）
    ax = fig.add_subplot(gb[0, 2])
    style(ax)
    ax.grid(axis='y', visible=False)
    ax.grid(axis='x', color=GRID, lw=0.6)
    for cond in ('AE', 'S2'):
        vals = [A.risk_by_country(R[R.rep == 0] if m not in E1 else R, cond, [m])[0].mean() for m in METHODS]
        ax.scatter(vals, ys + off[cond], s=30, color=GUIDE[cond], edgecolor='white', lw=0.8, zorder=3)
        ax.hlines(ys + off[cond], 0, vals, color=GUIDE[cond], lw=1.2, alpha=0.5)
    ax.set_ylim(len(METHODS) - 0.5, -0.5)
    ax.set_yticks(ys, [''] * len(METHODS))
    ax.tick_params(axis='y', length=0)
    ax.set_xlim(0, 0.75)
    ax.set_xlabel('比随机低 0.5 pp 以上的比例')
    ax.set_title('风险（行与热图对应）', fontsize=10, color=INK, loc='left')
    for yy in (3.5, 5.5):
        ax.axhline(yy, color=GRID, lw=0.8)

    fig.suptitle('样本扩充策略基准（G 实验，11 国）：各策略相对随机选样的收益与风险，AE 指导 vs S2 指导',
                 fontsize=12, color=INK, x=0.01, ha='left', y=0.955)
    fig.text(0.01, 0.045, wrap_mixed(
        '方法从上到下：只用几何（LPM2、k-center、TypiClust、Kernel herding）｜只用预测（Margin、按预测类分层）｜几何 + 预测（GIST、BADGE），灰线分隔。'
        '热图：对 AE/S2 指导、两种分类器、k=50/200 平均；国家按"200 个随机标签相对只用外国数据的增益"（横轴第二行，pp）从大到小排列。'
        '风险：每个国家内"比随机低 0.5 pp 以上"的选样运行比例，再对 11 国平均；纯噪声下（方法与随机无差别）的期望约为 0.26–0.37，'
        '所以除"按预测类分层"外，各方法的风险与噪声分不开（见 STRATEGY_FINDINGS_2026-09-26.md）。RO 的候选池只剩 3 类（标签映射所致），'
        'ba_ge30 只平均 3 个召回，RO 列应谨慎解读。\n'
        '描述性结果，不是预注册的主假设（冻结方案中"方法 − SRS"属 R3 描述量）；区间为国家层面 t 检验（df=10）的未校正 95% CI，'
        '多组对比未做校正。同一单元、同一分类器、同一 k 下与随机选样配对。几何方法用 44 个单元（11 国 × 2 划分 × 2 重复），'
        '其余方法只用重复 0 的 22 个单元。每个点对两种分类器（AE、S2 特征）取平均。学习器为 LightGBM（300 轮，目标样本权重 r=50，'
        '外国 5 万）。"AE/S2 指导"= 选样看到的几何空间与外国模型预测的来源。', 150), fontsize=7.5, color=MUTED, va='top')
    os.makedirs(FIG_DIR, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(FIG_DIR, f'fig_strategy_benchmark.{ext}'), dpi=200, bbox_inches='tight')
    S.to_csv(os.path.join(A.OUT, 'strategy_benchmark_summary.csv'), index=False)
    print('saved')


if __name__ == '__main__':
    main()
