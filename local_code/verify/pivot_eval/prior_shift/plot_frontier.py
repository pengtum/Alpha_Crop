"""草图：同一批 SRS 标签下，不同标签用法相对"G 重训 SRS"的 (ΔOA, ΔBA)。11 国等权均值。"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'Heiti SC']
plt.rcParams['axes.unicode_minus'] = False

OUT = os.path.dirname(os.path.abspath(__file__))
f2 = pd.read_csv(os.path.join(OUT, 'frontier2_rows.csv'))
f1 = pd.read_csv(os.path.join(OUT, 'frontier_rows.csv'))
f3 = pd.read_csv(os.path.join(OUT, 'frontier3_rows.csv'))
key = ['unit', 'target', 'clf', 'k']
INK, INK2, SURF = '#0b0b0b', '#52514e', '#fcfcfb'
S1, S2, S3 = '#2a78d6', '#eb6834', '#1baf7a'


def cm(d):
    return d.groupby('target')[['dba', 'doa']].mean().mean()




fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), facecolor=SURF)
for ax, k in zip(axes, (50, 200)):
    ax.set_facecolor(SURF)
    ref = f2[(f2.design == 'srs_G') & (f2.fam == 'retrain') & (f2.k == k)].set_index(key)[['ba', 'oa']]
    for fam, col, lab in (('F3b', S1, 'F3b：重训 ⊕ 冻结模型 bal 头（用标签）'),
                          ('F3u', S2, 'F3u：重训 ⊕ 冻结模型无标签均衡化'),):
        pts = [(0.0, 0.0)]
        for a in (0.3, 0.4, 0.5, 0.6):
            x = f2[(f2.design == 'srs_G') & (f2.fam == fam) & (f2.alpha == a) & (f2.k == k)].set_index(key)[['ba', 'oa']]
            j = (x - ref).dropna().rename(columns={'ba': 'dba', 'oa': 'doa'}).reset_index()
            m = cm(j)
            pts.append((m.doa, m.dba))
        pts = np.array(pts)
        ax.plot(pts[:, 0], pts[:, 1], '-o', color=col, lw=2, ms=6, label=lab, zorder=3)
        ax.annotate(f'α=0.5', pts[3], textcoords='offset points', xytext=(6, -12), color=INK2, fontsize=8)
    d = f1[(f1.design == 'srs_G') & (f1.fam == 'F2_retrain_biasdir') & (f1.k == k)].copy()
    d['dba'], d['doa'] = d.ba - d.ba_ref, d.oa - d.oa_ref
    pts = d.groupby(['alpha', 'target'])[['doa', 'dba']].mean().groupby('alpha').mean().loc[[0, .2, .4, .6, 1.0]].to_numpy()
    ax.plot(pts[:, 0], pts[:, 1], '-o', color=S3, lw=2, ms=6, label='F2：重训 + 偏置方向（启发式）', zorder=3)
    # 锚点
    r3 = f3[(f3.design == 'srs_G') & (f3['var'] == 'retrain') & (f3.it == 'i300') & (f3.k == k)].set_index(key)[['ba', 'oa']]
    r2 = f3[(f3.design == 'srs_G') & (f3['var'] == 'retrain') & (f3.it == 'i200') & (f3.k == k)].set_index(key)[['ba', 'oa']]
    j = (r2 - r3).dropna().rename(columns={'ba': 'dba', 'oa': 'doa'}).reset_index()
    m = cm(j)
    ax.plot(m.doa, m.dba, 's', color=INK2, ms=7, zorder=4)
    ax.annotate('重训 200 轮', (m.doa, m.dba), textcoords='offset points', xytext=(-70, 8), color=INK2, fontsize=8)
    r0 = r3[r3.index.get_level_values('unit').str.endswith('r0')]
    for des, lab, off in (('predstrat_AE', 'predstrat_srs 重训', (6, -4)),):
        x = f3[(f3.design == des) & (f3['var'] == 'retrain') & (f3.it == 'i300') & (f3.k == k)].set_index(key)[['ba', 'oa']]
        m = cm((x - r0).dropna().rename(columns={'ba': 'dba', 'oa': 'doa'}).reset_index())
        ax.plot(m.doa, m.dba, 'D', color=INK, ms=7, zorder=4)
        ax.annotate(lab, (m.doa, m.dba), textcoords='offset points', xytext=(off if k == 200 else (8, -16)), color=INK, fontsize=8)
        x = f3[(f3.design == des) & (f3['var'] == 'F3b_l0.1_dw1') & (f3.it == 'i300') & (f3.alpha == 0.4) & (f3.k == k)].set_index(key)[['ba', 'oa']]
        m2 = cm((x - r0).dropna().rename(columns={'ba': 'dba', 'oa': 'doa'}).reset_index())
        ax.plot(m2.doa, m2.dba, 'D', color=S1, ms=8, zorder=5, markeredgecolor=INK)
        ax.annotate('predstrat_srs + F3b(α=0.4)', (m2.doa, m2.dba), textcoords='offset points', xytext=(6, 4),
                    color=INK, fontsize=8)
    ax.axhspan(1.0, 3.2, xmin=0, xmax=1, color='#eeeeea', zorder=0)
    ax.axvline(-0.3, color=INK2, lw=0.8, ls='--', zorder=1)
    ax.axhline(0, color=INK2, lw=0.6, zorder=1)
    ax.axvline(0, color=INK2, lw=0.6, zorder=1)
    ax.set_xlim(-3.3, 0.8)
    ax.set_ylim(-0.4, 3.2)
    ax.set_title(f'k = {k}：相对 G 重训 SRS（同一批或同单元标签）', color=INK, fontsize=10)
    ax.set_xlabel('ΔOA（pp）', color=INK2)
    ax.set_ylabel('Δba_ge30（pp）', color=INK2)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK2)
    ax.grid(color='#e6e6e2', lw=0.5)
axes[0].legend(frameon=False, fontsize=8, loc='upper left')
fig.text(0.01, 0.01, '灰带：ΔBA ≥ +1；虚线：ΔOA = −0.3。11 国等权均值；F3b/F3u/F2 为同一批 SRS 标签（44 单元），predstrat 点为重复 0 的 22 单元。'
         'α、λ 事后查看过，属探索性。', fontsize=7.5, color=INK2)
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'Heiti SC']
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(os.path.join(OUT, 'fig_prior_shift_frontier.png'), dpi=160, facecolor=SURF)
print('ok')
