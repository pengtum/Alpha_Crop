"""
G 实验结果图（ANALYSIS_PLAN_guidance.md）。
A：各国 θ_E1（AE 指导 − S2 指导，只用几何的通路，ba_ge30）与合并估计（95%/90% CI、80% 预测区间），灰带 = ±0.3 pp 等价界。
B：次要对比（通路、分类器、k、XY/S2XY、ORACLE 提升空间），点 = 11 国均值，线 = 95% CI。
C：几何的类别信息量（10 近邻平衡精度，AE − S2）与选样价值（θ_E1）的关系，每点一个国家。
输出：label_expansion/figures/fig_G_results.png|pdf
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import analyze_G as A
from guidance_G import E1, FAMILIES, OUT
from plot_check2 import INK, INK2, MUTED, GRID, BLUE, ORANGE, style

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')


def wrap_mixed(text, width=128):
    """按显示宽度换行：中日韩字符记 1，其他字符记 0.55（图注是中英混排）。保留原有换行。"""
    out = []
    for para in text.split('\n'):
        line, w = '', 0.0
        for ch in para:
            cw = 1.0 if ord(ch) > 0x2E80 else 0.55
            if w + cw > width:
                out.append(line)
                line, w = '', 0.0
            line += ch
            w += cw
        out.append(line)
    return '\n'.join(out)
BAND = '#eceae4'


def ci_row(ax, y, s, color, lw=1.6, ms=5.5, marker='o', post=False):
    """post=True：事后追加的行，空心点 + 虚线。"""
    ax.plot([s['lo95'], s['hi95']], [y, y], color=color, lw=lw, solid_capstyle='round', zorder=3,
            ls=(0, (3, 2)) if post else '-')
    ax.scatter(s['mean'], y, s=ms ** 2, marker=marker, zorder=4, lw=1.2 if post else 0.8,
               facecolor='white' if post else color, edgecolor=color if post else 'white')


def main():
    R = A.load_rows()
    d = A.deltas(R, 'AE', 'S2', E1)
    v = A.country_values(d).sort_values()
    s = A.t_summary(v)
    pi = A.prediction_interval(v.to_numpy(), np.full(len(v), A.within_var(d)))
    u = A.unit_level(d).groupby('target')

    fig = plt.figure(figsize=(12.5, 6.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.0, 0.85], wspace=0.72)
    # ---------------- A
    ax = fig.add_subplot(gs[0])
    style(ax)
    ax.grid(axis='y', visible=False)
    ax.grid(axis='x', color=GRID, lw=0.6)
    ax.axvspan(-A.SESOI, A.SESOI, color=BAND, zorder=0)
    ax.axvline(0, color=INK2, lw=0.9, ls=(0, (4, 3)))
    for i, c in enumerate(v.index):
        uv = u.get_group(c).to_numpy()
        ax.scatter(uv, np.full(len(uv), i), s=10, color=MUTED, alpha=0.6, lw=0, zorder=2)
        ax.scatter(v[c], i, s=34, color=BLUE, edgecolor='white', lw=0.8, zorder=4)
    y0 = len(v) + 0.9
    ax.plot([pi['pi_lo'], pi['pi_hi']], [y0, y0], color=BLUE, lw=5, alpha=0.18, solid_capstyle='butt', zorder=2)
    ax.plot([s['lo95'], s['hi95']], [y0, y0], color=INK, lw=1.2, zorder=3)
    ax.plot([s['lo90'], s['hi90']], [y0, y0], color=INK, lw=3, zorder=3)
    ax.scatter(s['mean'], y0, s=46, color=INK, marker='D', edgecolor='white', lw=0.8, zorder=4)
    ax.set_yticks(list(range(len(v))) + [y0], list(v.index) + ['合并（11 国）'])
    ax.set_xlabel('θ_E1：AE 指导 − S2 指导（pp，平衡精度）')
    ax.set_title('A  只用几何的通路：各国与合并估计', fontsize=10, color=INK, loc='left')
    lim = max(0.9, np.abs(np.r_[d.groupby(A.UNIT).d.mean().to_numpy(), pi['pi_lo'], pi['pi_hi']]).max() * 1.08)
    ax.set_xlim(-lim, lim)

    # ---------------- B
    ax = fig.add_subplot(gs[1])
    style(ax)
    ax.grid(axis='y', visible=False)
    ax.grid(axis='x', color=GRID, lw=0.6)
    ax.axvspan(-A.SESOI, A.SESOI, color=BAND, zorder=0)
    ax.axvline(0, color=INK2, lw=0.9, ls=(0, (4, 3)))
    items = []
    names = {'geom': '只用几何（主，E1）', 'pred': '只用预测', 'hyb': '几何 + 预测'}
    for p, ms in FAMILIES.items():
        dd = A.deltas(R, 'AE', 'S2', ms, reps=None if p == 'geom' else [0])
        items.append((names[p], A.t_summary(A.country_values(dd)), BLUE if p == 'geom' else ORANGE))
    for clf in ('s2', 'ae'):
        items.append((f'E1 | {clf.upper()} 分类器', A.t_summary(A.country_values(d[d.clf == clf])), MUTED))
    for k in (50, 200):
        items.append((f'E1 | k={k}', A.t_summary(A.country_values(d[d.k == k])), MUTED))
    two = ['lpm2_nogeo', 'kcenter']
    for a, b, lab in (('AE', 'S2XY', 'AE − (S2 + 坐标)'), ('AE', 'XY', 'AE − 纯坐标'), ('S2XY', 'S2', '(S2 + 坐标) − S2')):
        items.append((lab, A.t_summary(A.country_values(A.deltas(R, a, b, two, reps=[0]))), INK2))
    three = ['lpm2_nogeo', 'typiclust', 'margin']
    items.append(('神谕 H（预注册）',
                  A.t_summary(A.country_values(A.deltas(R, 'ORACLE', 'S2', three, reps=[0]))), INK))
    if 'ORACLE2' in set(R.cond):
        dH = pd.concat([A.deltas(R, 'ORACLE2', 'S2', ['lpm2_nogeo', 'typiclust'], reps=[0]),
                        A.deltas(R, 'ORACLE', 'S2', ['margin'], reps=[0])])
        items.append(('神谕 H（几何修正*）', A.t_summary(A.country_values(dH)), INK))
    for ca, cb, ms, lab in (('RAWZ_AE', 'RAWZ_S2', E1, 'RAWZ：AE − S2*'),
                            ('NATIVE_AE', 'NATIVE_S2', ['lpm2_nogeo', 'typiclust'], 'NATIVE：AE − S2*')):
        if ca in set(R.cond):
            items.append((lab, A.t_summary(A.country_values(A.deltas(R, ca, cb, ms, reps=[0], ks=(50,)))), BLUE))
    for i, (lab, st, c) in enumerate(items):
        ci_row(ax, i, st, c, post=lab.endswith('*'))
    ax.set_yticks(range(len(items)), [it[0] for it in items])
    ax.invert_yaxis()
    for yy in (2.5, 6.5, 9.5, 11.5):
        if yy < len(items) - 0.5:
            ax.axhline(yy, color=GRID, lw=0.8)
    ax.set_xlabel('差（pp，平衡精度；11 国均值与未校正的 95% CI）')
    ax.set_title('B  次要对比', fontsize=10, color=INK, loc='left')

    # ---------------- C
    ax = fig.add_subplot(gs[2])
    style(ax)
    Q = pd.read_csv(os.path.join(OUT, 'guidance_quality_G.csv'))
    q = Q.pivot_table(index='target', columns='feat', values='knn10_ba')
    gap = (q['ae'] - q['s2']).reindex(v.index)
    ax.axhspan(-A.SESOI, A.SESOI, color=BAND, zorder=0)
    ax.axhline(0, color=INK2, lw=0.9, ls=(0, (4, 3)))
    ax.scatter(gap, v, s=34, color=BLUE, edgecolor='white', lw=0.8, zorder=3)
    for c in v.index:
        off = {'AT': (-12, 3), 'RO': (4, -8)}.get(c, (4, 2))
        ax.annotate(c, (gap[c], v[c]), xytext=off, textcoords='offset points', fontsize=7, color=INK2)
    from scipy.stats import spearmanr
    r = np.corrcoef(gap, v)[0, 1]
    rho = spearmanr(gap, v).correlation
    z, se = np.arctanh(r), 1 / np.sqrt(len(v) - 3)
    loo_r = {c: np.corrcoef(gap.drop(c), v.drop(c))[0, 1] for c in v.index}
    ax.set_xlabel('几何的类别信息量：10 近邻平衡精度 AE − S2（pp）')
    ax.set_ylabel('θ_E1（pp）')
    ax.set_title('C  AE 类别信息量优势与 θ_E1（事后、探索性、未登记）', fontsize=10, color=INK, loc='left')
    ax.text(0.98, 0.22, f'Pearson r = {r:+.2f}\n（95% CI {np.tanh(z - 1.96 * se):+.2f} 至 {np.tanh(z + 1.96 * se):+.2f}）\n'
            f'Spearman ρ = {rho:+.2f}\n留一国 r：{min(loo_r.values()):+.2f}（去 {min(loo_r, key=loo_r.get)}）'
            f'至 {max(loo_r.values()):+.2f}',
            transform=ax.transAxes, fontsize=7.3, color=INK2, va='bottom', ha='right')

    fig.suptitle('G 实验：只用几何的通路上，以 AlphaEarth 嵌入代替 Sentinel-2 特征作为选样几何，\n'
                 '11 国等权平均的平衡精度差判为等价（±0.3 pp；OA 与 k=200 单独看为不确定）',
                 fontsize=11.5, color=INK, x=0.01, ha='left', y=1.06)
    fig.text(0.01, -0.05, wrap_mixed(f'A：合并 {s["mean"]:+.2f} pp，90% CI [{s["lo90"]:+.2f}, {s["hi90"]:+.2f}]（粗线），'
             f'95% CI [{s["lo95"]:+.2f}, {s["hi95"]:+.2f}]（细线），80% 预测区间 [{pi["pi_lo"]:+.2f}, {pi["pi_hi"]:+.2f}]'
             '（浅蓝带；国家内方差取合并值，各国自取时为 [−0.18, +0.12]）。'
             '同一单元、同一方法、同一种子，只替换指导数据（几何方法只用其特征空间，预测类方法另用外国模型的预测）。\n'
             '主估计量 θ_E1 = 4 个只用几何的方法 × 2 种分类器 × k∈{50,200} × 4 个单元（2 个空间划分 × 2 次候选池抽样）的等权平均，11 国 t 检验（df=10）。'
             '\n灰带 = 预注册的等价界 ±0.3 pp。A 中灰点 = 单元值。B：只用预测 = margin、predstrat_srs；几何 + 预测 = gist_margin、badge_c；'
             'XY/S2XY 对比 = lpm2_nogeo、kcenter；神谕 H = lpm2_nogeo、typiclust、margin；这些行只用重复 0 的 22 个单元，E1 各行用 44 个单元。'
             '\nB 的区间未做多重比较校正；14 项预注册次要对比经 Holm 校正后均不显著（只用预测：原始 p=0.031，Holm 0.44，maxT 0.14）；等价判定用 90% CI，见报告。'
             '神谕指导 = 用目标国候选池标签交叉拟合的模型与 LDA 空间。LightGBM 300 棵树，目标样本权重 r=50。'
             '\nC：x = 重复 0 候选池上 10 近邻平衡精度 AE − S2（两划分平均），y = 各国 θ_E1，n = 11；未登记于冻结方案与 rev1，p 未校正。'
             '\n* 空心点、虚线 = 看过结果后追加（ANALYSIS_PLAN_guidance_rev1.md B1、B2）。几何修正 = lpm2_nogeo、typiclust 改用修正后的神谕几何'
             '（只保留池中 ≥30 个地块的类的 LDA 列，逐列 z 截断到 ±5），margin 沿用原神谕，k∈{50,200}；'
             'RAWZ = 阶段 A 的原始 z + PCA（4 个几何方法）；NATIVE = AE 均值组 L2 单位化、S2 均值组稳健 z（lpm2_nogeo、typiclust）；RAWZ 与 NATIVE 只用 k=50；均为重复 0。'),
             fontsize=7.3, color=MUTED, va='top')
    os.makedirs(FIG_DIR, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(FIG_DIR, f'fig_G_results.{ext}'), dpi=200, bbox_inches='tight')
    print('saved')


if __name__ == '__main__':
    main()
