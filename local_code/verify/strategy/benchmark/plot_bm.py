"""两张图：(1) 各方法相对 SRS 的国家层面效应（BA、OA；AE/S2 指导；国家点，RO 标出）；
(2) 一致性与机制：设置 × 方法热图、标签价值 vs 方法分散度（OA）、BA vs OA 方法效应。"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from common_bm import *

plt.rcParams.update({'font.family': ['Arial Unicode MS'], 'axes.unicode_minus': False, 'font.size': 9,
                     'axes.edgecolor': '#8a8986', 'axes.linewidth': 0.6, 'xtick.color': '#52514e', 'ytick.color': '#52514e',
                     'axes.labelcolor': '#0b0b0b', 'axes.titlesize': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb', 'savefig.facecolor': '#fcfcfb'})
AE_C, S2_C, INK2, GRID = '#2a78d6', '#eb6834', '#52514e', '#e4e3df'
R = load()
T = pd.read_csv(f'{OUTD}/a_vs_srs_cells.csv')
order = ['predstrat_srs', 'margin', 'gist_margin', 'badge_c', 'kherding', 'typiclust', 'lpm2_nogeo', 'kcenter']
fam = {m: '几何' for m in E1} | {m: '预测' for m in PRED} | {m: '混合' for m in HYB}

# ---------------- 图 1
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), sharey=True)
for ax, metric, lab in ((axes[0], 'ba_ge30', 'ba_ge30：方法 − SRS（pp）'), (axes[1], 'oa', 'OA：方法 − SRS（pp）')):
    D = all_vs_srs(R, metric)
    cm = D.groupby(['target', 'cond', 'clf', 'k', 'method']).d.mean().groupby(['target', 'method']).mean().unstack('method')
    for i, m in enumerate(order):
        y = len(order) - 1 - i
        v = cm[m]
        ax.scatter(v.drop('RO'), np.full(10, y) + np.linspace(-0.08, 0.08, 10), s=9, color='#b5b4ae', zorder=1, linewidths=0)
        ax.scatter([v['RO']], [y], s=22, marker='x', color='#0b0b0b', zorder=3, linewidths=0.9)
        for off, cond, col in ((0.22, 'AE', AE_C), (-0.22, 'S2', S2_C)):
            r = T[(T.metric == metric) & (T.method == m) & (T.cond == cond) & (T.clf == 'pool')].iloc[0]
            ax.plot([r.lo95, r.hi95], [y + off, y + off], color=col, lw=2, solid_capstyle='round', zorder=2)
            ax.scatter([r['mean']], [y + off], s=30, color=col, edgecolor='#fcfcfb', linewidth=1.2, zorder=4)
        m10 = v.drop('RO').mean()
        ax.scatter([m10], [y], s=34, marker='D', facecolor='none', edgecolor='#0b0b0b', linewidth=0.9, zorder=5)
    ax.axvline(0, color=INK2, lw=0.8, zorder=0)
    ax.grid(axis='x', color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel(lab)
    if metric == 'ba_ge30':
        ax.set_xlim(-7, 5)
    else:
        ax.set_xlim(-13.5, 2)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f'{m}（{fam[m]}）' for m in order[::-1]])
    ax.tick_params(axis='y', length=0)
# 图例
from matplotlib.lines import Line2D
h = [Line2D([], [], color=AE_C, lw=2, marker='o', ms=5, label='AE 指导：11 国均值 ± 95% CI'),
     Line2D([], [], color=S2_C, lw=2, marker='o', ms=5, label='S2 指导：11 国均值 ± 95% CI'),
     Line2D([], [], color='#b5b4ae', lw=0, marker='o', ms=4, label='各国（AE+S2 合并）'),
     Line2D([], [], color='#0b0b0b', lw=0, marker='x', ms=5, label='RO'),
     Line2D([], [], color='#0b0b0b', lw=0, marker='D', ms=5, markerfacecolor='none', label='去 RO 的 10 国均值')]
fig.legend(handles=h, loc='lower center', ncol=5, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.01))
fig.suptitle('G 实验：各选样方法相对 SRS（2 分类器 × k∈{50,200} 等权；几何法 44 单元，预测/混合法 22 单元）', fontsize=10, x=0.5, y=0.99)
fig.tight_layout(rect=(0, 0.06, 1, 0.97))
fig.savefig(f'{OUTD}/fig1_methods_vs_srs.png', dpi=200)
plt.close(fig)

# ---------------- 图 2
fig = plt.figure(figsize=(12, 4.4))
gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1, 1], wspace=0.42)
# (a) 设置 × 方法热图（BA）
ax = fig.add_subplot(gs[0])
M = pd.read_csv(f'{OUTD}/c_setting_by_method_ba_ge30.csv').set_index(['cond', 'clf', 'k'])[order]
M = M.reindex([(g, c, k) for g in ('AE', 'S2') for c in ('s2', 'ae') for k in (50, 200)])
cmap = LinearSegmentedColormap.from_list('div', ['#e34948', '#f0efec', '#2a78d6'])
im = ax.imshow(M.T.to_numpy(), cmap=cmap, norm=TwoSlopeNorm(0, -1.5, 1.5), aspect='auto')
for i in range(M.shape[1]):
    for j in range(M.shape[0]):
        val = M.iloc[j, i]
        ax.text(j, i, f'{val:+.2f}', ha='center', va='center', fontsize=6.5, color='#0b0b0b')
ax.set_xticks(range(M.shape[0]))
ax.set_xticklabels([f'{g}\n{c}\nk{k}' for g, c, k in M.index], fontsize=7)
ax.set_yticks(range(len(order)))
ax.set_yticklabels(order, fontsize=8)
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
ax.set_title('(a) ba_ge30 方法 − SRS，11 国均值\n列：指导 / 分类器 / k；Kendall W = 0.57（去 RO 0.81）', fontsize=9)
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
cb.ax.tick_params(labelsize=7)
cb.outline.set_visible(False)
# (b) 标签价值 vs 方法分散度（OA）
ax = fig.add_subplot(gs[1])
L = pd.read_csv(f'{OUTD}/d_label_value_oa.csv', index_col=0)
x, y = L['0->200'], L['sd_methods']
ax.scatter(x, y, s=28, color=AE_C, edgecolor='#fcfcfb', linewidth=1, zorder=3)
for c in L.index:
    ax.annotate(c, (x[c], y[c]), xytext=(4, 2), textcoords='offset points', fontsize=7.5, color=INK2)
ax.set_xscale('log')
ax.set_yscale('log')
from matplotlib.ticker import FuncFormatter
fmt_ = FuncFormatter(lambda v, _: f'{v:g}')
for a_ in (ax.xaxis, ax.yaxis):
    a_.set_major_formatter(fmt_)
    a_.set_minor_formatter(FuncFormatter(lambda v, _: ''))
ax.set_xticks([0.2, 0.5, 1, 2, 5, 10, 20])
ax.set_yticks([0.05, 0.1, 0.2, 0.5, 1, 2, 5])
ax.set_xlabel('SRS 标签价值：OA(k=200) − OA(k=0)，pp')
ax.set_ylabel('8 个方法相对 SRS 的 OA 效应 SD（pp）')
ax.grid(color=GRID, lw=0.6)
ax.set_axisbelow(True)
ax.set_title('(b) 标签越值钱，选样方法之间差得越多（OA）\nSpearman ρ = 0.96（去 RO 0.95）', fontsize=9)
# (c) BA vs OA 方法效应
ax = fig.add_subplot(gs[2])
ba = pd.read_csv(f'{OUTD}/c_country_by_method_ba_ge30.csv', index_col=0)
oa = pd.read_csv(f'{OUTD}/c_country_by_method_oa.csv', index_col=0)
for sel, mk, lab2 in ((ba.index, 'o', '11 国'), (ba.index.drop('RO'), 'D', '去 RO')):
    bx, oy = ba.loc[sel].mean(), oa.loc[sel].mean()
    ax.scatter(bx, oy, s=26, marker=mk, color=AE_C if mk == 'o' else '#fcfcfb', edgecolor=AE_C, linewidth=1.1, zorder=3, label=lab2)
    if mk == 'o':
        for m in ALL8:
            off = {'typiclust': (-48, 4), 'kherding': (6, -1), 'lpm2_nogeo': (-62, -10)}.get(m, (5, -3))
            ax.annotate(m, (bx[m], oy[m]), xytext=off, textcoords='offset points', fontsize=7, color=INK2)
    else:
        for m in ALL8:
            ax.plot([ba[m].mean(), bx[m]], [oa[m].mean(), oy[m]], color='#b5b4ae', lw=0.6, zorder=1)
ax.axhline(0, color=INK2, lw=0.8)
ax.axvline(0, color=INK2, lw=0.8)
ax.grid(color=GRID, lw=0.6)
ax.set_axisbelow(True)
ax.set_xlabel('ba_ge30：方法 − SRS（pp）')
ax.set_ylabel('OA：方法 − SRS（pp）')
ax.legend(frameon=False, fontsize=7.5, loc='lower right')
r10 = stats.spearmanr(ba.drop('RO').mean(), oa.drop('RO').mean())[0]
print('ex-RO BA-OA spearman', r10)
ax.set_title(f'(c) BA 与 OA 上的方法排名互不相关\nSpearman ρ = −0.05（11 国均值）；去 RO {r10:+.2f}', fontsize=9)
fig.savefig(f'{OUTD}/fig2_consistency_mechanism.png', dpi=200, bbox_inches='tight')
print('ok')
