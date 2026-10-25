"""Label efficiency: AE vs S2 iso-accuracy budget ratio.

问题：达到相同 BA τ，用 AE 需要多少样本 vs 用 S2？
方法：
  1. 对每 (source, strategy)，用 5 个 budget 点构建 BA-vs-log(budget) 曲线
  2. 在每个目标 τ 上，用单调样条/线性插值求达到 τ 所需 budget（log 空间）
  3. 报告比例 AE_budget / S2_budget — < 1 说明 AE 更 sample-efficient
"""
import os, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'font.family': 'Arial',
    'font.size': 10, 'figure.dpi': 150, 'savefig.dpi': 150,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.1,
})

OUT_DIR = 'results/experiments/exp6_scalable_mapping'
FIG_DIR = 'Manuscript/Paper_Alpha_Earth/0001_draft_v1/figures'
SOURCES = ['AT','BE','DE','EE','LT','LV','PT','SE','SI','SK']
FEATURES = ['ae','s2','fusion']
FT_LABELS = {'ae':'AlphaEarth','s2':'Sentinel-2','fusion':'Fusion'}
FT_COLORS = {'ae':'#2196F3','s2':'#FF9800','fusion':'#4CAF50'}
BUDGETS = np.array([1000, 5000, 10000, 20000, 50000])
KEY_STRATS = ['balanced','cost_only','random','cost_aware_weak']  # 不同策略下都看


def load():
    dfs = []
    for s in SOURCES:
        for ft in FEATURES:
            f = os.path.join(OUT_DIR, f'scalable_{ft}_{s}_tier_b.csv')
            if not os.path.exists(f): continue
            df = pd.read_csv(f)
            df['source'], df['feature'] = s, ft
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def get_curve(df, source, feature, strategy):
    """返回 (budgets, ba_means, ba_stds)。"""
    sub = df[(df.source==source)&(df.feature==feature)&
             (df.strategy==strategy)&(df.budget>0)]
    grp = sub.groupby('budget')['mean_ood_ba_equal'].agg(['mean','std','count']).reset_index()
    return grp['budget'].values, grp['mean'].values, grp['std'].values


def budget_to_reach(budgets, bas, target):
    """log-budget 空间内线性插值，找最小 budget 使 BA >= target。
    若曲线最高点未达到 target，返回 NaN（外推不可信）。
    若 budget=最小点已达到，返回最小 budget。
    """
    if np.max(bas) < target:
        return np.nan
    if np.min(bas) >= target:
        return budgets[0]  # 已经够了
    # 找首个跨过 target 的区间
    log_b = np.log(budgets)
    for i in range(len(bas)-1):
        if bas[i] < target <= bas[i+1]:
            # 线性插值（log budget 空间）
            t = (target - bas[i]) / (bas[i+1] - bas[i])
            return float(np.exp(log_b[i] + t*(log_b[i+1]-log_b[i])))
    return budgets[-1]


def table_iso_accuracy(df, strategy='balanced'):
    """对每个 source 计算 iso-accuracy budget ratio AE/S2。"""
    print(f'\n{"="*92}')
    print(f' Iso-Accuracy Budget Ratio — Strategy: {strategy}')
    print(f' Lower ratio = AE more sample-efficient')
    print('='*92)
    targets = [0.34, 0.36, 0.38, 0.40, 0.42, 0.44]

    print(f'{"Source":<8}', end='')
    for t in targets:
        print(f'{"τ="+str(t):>14}', end='')
    print()
    print('-'*92)

    all_ratios = {t: [] for t in targets}
    for src in SOURCES:
        b_ae, ba_ae, _ = get_curve(df, src, 'ae', strategy)
        b_s2, ba_s2, _ = get_curve(df, src, 's2', strategy)
        if len(ba_ae)==0 or len(ba_s2)==0:
            continue
        print(f'{src:<8}', end='')
        for t in targets:
            ae_b = budget_to_reach(b_ae, ba_ae, t)
            s2_b = budget_to_reach(b_s2, ba_s2, t)
            if np.isnan(ae_b) and np.isnan(s2_b):
                cell = '— —'
            elif np.isnan(ae_b):
                cell = 'AE未达'
            elif np.isnan(s2_b):
                cell = 'S2未达'
            else:
                r = ae_b / s2_b
                all_ratios[t].append(r)
                cell = f'{r:.2f}x'
            print(f'{cell:>14}', end='')
        print()

    print('-'*92)
    print(f'{"median":<8}', end='')
    for t in targets:
        if all_ratios[t]:
            m = np.median(all_ratios[t])
            print(f'{m:>13.2f}x', end='')
        else:
            print(f'{"N/A":>14}', end='')
    print()
    print(f'{"mean":<8}', end='')
    for t in targets:
        if all_ratios[t]:
            print(f'{np.mean(all_ratios[t]):>13.2f}x', end='')
        else:
            print(f'{"N/A":>14}', end='')
    print()
    print(f'{"n_used":<8}', end='')
    for t in targets:
        print(f'{len(all_ratios[t]):>14d}', end='')
    print()
    return all_ratios


def table_iso_accuracy_aggregate(df):
    """对全局聚合曲线（10 source 平均）做 iso-accuracy 分析。"""
    print(f'\n{"="*78}')
    print(f' Iso-Accuracy Budget Ratio — AGGREGATE (mean over 10 sources × 3 seeds)')
    print(f' Strategy: balanced; budget needed to reach BA τ')
    print('='*78)
    print(f'{"τ (BA target)":<15}{"AE budget":>14}{"S2 budget":>14}{"Fusion budget":>16}{"AE/S2":>10}{"Fus/S2":>10}')
    print('-'*78)
    rows = []
    targets = [0.34, 0.36, 0.38, 0.40, 0.42, 0.44, 0.46]
    for t in targets:
        budgets_per_ft = {}
        for ft in FEATURES:
            sub = df[(df.feature==ft)&(df.strategy=='balanced')&(df.budget>0)]
            grp = sub.groupby('budget')['mean_ood_ba_equal'].mean().reset_index()
            grp = grp.sort_values('budget')
            b = budget_to_reach(grp['budget'].values, grp['mean_ood_ba_equal'].values, t)
            budgets_per_ft[ft] = b

        ae_b, s2_b, fu_b = budgets_per_ft['ae'], budgets_per_ft['s2'], budgets_per_ft['fusion']
        ae_s = f'{ae_b:.0f}' if not np.isnan(ae_b) else 'unreached'
        s2_s = f'{s2_b:.0f}' if not np.isnan(s2_b) else 'unreached'
        fu_s = f'{fu_b:.0f}' if not np.isnan(fu_b) else 'unreached'
        r1 = ae_b/s2_b if not (np.isnan(ae_b) or np.isnan(s2_b)) else np.nan
        r2 = fu_b/s2_b if not (np.isnan(fu_b) or np.isnan(s2_b)) else np.nan
        r1s = f'{r1:.2f}x' if not np.isnan(r1) else 'N/A'
        r2s = f'{r2:.2f}x' if not np.isnan(r2) else 'N/A'
        print(f'τ={t:<14}{ae_s:>14}{s2_s:>14}{fu_s:>16}{r1s:>10}{r2s:>10}')
        rows.append({'target':t,'ae':ae_b,'s2':s2_b,'fusion':fu_b,
                     'ae_over_s2':r1,'fusion_over_s2':r2})
    return pd.DataFrame(rows)


def plot_label_efficiency(df, strategy='balanced'):
    """3 子图：(A) 三 feature 平均曲线；(B) iso-accuracy 散点；(C) per-source AE/S2 ratio"""
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    # ============= 子图 A: BA vs budget 曲线（10 source 聚合）=============
    ax = axes[0]
    for ft in FEATURES:
        sub = df[(df.feature==ft)&(df.strategy==strategy)&(df.budget>0)]
        grp = sub.groupby('budget')['mean_ood_ba_equal'].agg(['mean','std','count']).reset_index()
        grp['se'] = grp['std']/np.sqrt(grp['count'])
        ax.errorbar(grp['budget'], grp['mean'], yerr=grp['se'],
                    marker='o', markersize=8, color=FT_COLORS[ft],
                    linewidth=2, capsize=3, label=FT_LABELS[ft])
    # 画水平参考线
    for tau in [0.40, 0.42, 0.44]:
        ax.axhline(tau, color='gray', linestyle=':', linewidth=0.7, alpha=0.6)
        ax.text(60000, tau+0.001, f'τ={tau}', fontsize=8, color='gray', va='bottom')
    ax.set_xscale('log')
    ax.set_xlabel('Budget (samples, log)', fontsize=11)
    ax.set_ylabel('Mean OOD Balanced Accuracy', fontsize=11)
    ax.set_title(f'(A) Aggregate BA vs Budget\nstrategy={strategy}', fontsize=11)
    ax.grid(alpha=0.3, linestyle='--')
    ax.legend(loc='lower right', fontsize=9)

    # ============= 子图 B: Iso-accuracy budget per feature =============
    ax = axes[1]
    targets = np.array([0.36, 0.38, 0.40, 0.42, 0.44])
    for ft in FEATURES:
        sub = df[(df.feature==ft)&(df.strategy==strategy)&(df.budget>0)]
        grp = sub.groupby('budget')['mean_ood_ba_equal'].mean().reset_index().sort_values('budget')
        budgets_needed = [budget_to_reach(grp['budget'].values, grp['mean_ood_ba_equal'].values, t)
                          for t in targets]
        ax.plot(targets, budgets_needed, marker='o', markersize=8, color=FT_COLORS[ft],
                linewidth=2, label=FT_LABELS[ft])
    ax.set_yscale('log')
    ax.set_xlabel('Target BA (τ)', fontsize=11)
    ax.set_ylabel('Budget needed to reach τ (log)', fontsize=11)
    ax.set_title('(B) Iso-Accuracy Budget per Feature', fontsize=11)
    ax.grid(alpha=0.3, linestyle='--')
    ax.legend(loc='upper left', fontsize=9)

    # ============= 子图 C: AE/S2 ratio per source =============
    ax = axes[2]
    targets = [0.36, 0.38, 0.40, 0.42]
    ratio_data = {t: [] for t in targets}
    src_labels = {t: [] for t in targets}
    for src in SOURCES:
        b_ae, ba_ae, _ = get_curve(df, src, 'ae', strategy)
        b_s2, ba_s2, _ = get_curve(df, src, 's2', strategy)
        for t in targets:
            ae_b = budget_to_reach(b_ae, ba_ae, t)
            s2_b = budget_to_reach(b_s2, ba_s2, t)
            if not (np.isnan(ae_b) or np.isnan(s2_b)):
                ratio_data[t].append(ae_b/s2_b)
                src_labels[t].append(src)

    cmap = plt.cm.plasma
    x = np.arange(len(targets))
    w = 0.7
    parts = ax.boxplot([ratio_data[t] for t in targets], positions=x, widths=w,
                        patch_artist=True, showfliers=False)
    for patch, t in zip(parts['boxes'], targets):
        patch.set_facecolor('#90CAF9'); patch.set_alpha(0.6)
    # 散点叠加，标注 source
    for i, t in enumerate(targets):
        ys = ratio_data[t]
        xs = np.random.normal(i, 0.06, size=len(ys))
        ax.scatter(xs, ys, s=30, c=range(len(ys)), cmap='tab10', edgecolor='black',
                   linewidth=0.5, zorder=3)
        for xj, yj, lbl in zip(xs, ys, src_labels[t]):
            ax.annotate(lbl, (xj, yj), xytext=(2, 2), textcoords='offset points',
                        fontsize=6.5, color='#333')

    ax.axhline(1.0, color='red', linestyle='--', linewidth=1.2, alpha=0.7,
               label='AE = S2 (parity)')
    ax.set_xticks(x); ax.set_xticklabels([f'τ={t}' for t in targets])
    ax.set_yscale('log')
    ax.set_ylabel('Budget ratio AE / S2 (log)\n<1 = AE more efficient', fontsize=10)
    ax.set_title('(C) Per-source AE/S2 budget ratio', fontsize=11)
    ax.grid(alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', fontsize=9)

    fig.suptitle(
        'Label efficiency: AE vs S2 — how many samples to reach the same BA?',
        fontsize=12.5, y=1.02, fontweight='bold')
    fig.tight_layout()
    for ext in ['png','pdf']:
        fig.savefig(os.path.join(FIG_DIR, f'exp6_v2_label_efficiency_ae_vs_s2.{ext}'))
    plt.close(fig)
    print('[OK] exp6_v2_label_efficiency_ae_vs_s2')


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    df = load()
    print(f'Loaded {len(df)} rows')

    # 聚合 iso-accuracy
    agg = table_iso_accuracy_aggregate(df)
    agg.to_csv(os.path.join(OUT_DIR, 'iso_accuracy_aggregate.csv'), index=False)

    # Per-source 详表（balanced）
    table_iso_accuracy(df, 'balanced')
    print()
    # 也看其他策略下是否一致
    table_iso_accuracy(df, 'cost_only')

    plot_label_efficiency(df, 'balanced')
    print('\nDone.')
