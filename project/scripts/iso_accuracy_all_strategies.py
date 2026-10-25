"""跨全部策略的 iso-accuracy AE/S2 比例 — 鲁棒性检验。

对每 (strategy, target_BA τ)：
  1. 聚合 10 sources × 3 seeds 的 BA-vs-budget 曲线
  2. 线性插值（log-budget 空间）求达到 τ 所需 budget
  3. 计算 AE/S2 与 Fusion/S2 比例
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

OUT_DIR='results/experiments/exp6_scalable_mapping'
FIG_DIR='Manuscript/Paper_Alpha_Earth/0001_draft_v1/figures'
SOURCES=['AT','BE','DE','EE','LT','LV','PT','SE','SI','SK']
FEATURES=['ae','s2','fusion']
# 排除 source_only（无 budget 概念）和 weighted（数据不全 4/30）
STRATS=['random','balanced','climate_similar','climate_diverse',
        'representativeness','transfer_strong','transfer_weak',
        'cost_only','cost_aware_strong','cost_aware_weak']
STRAT_LABELS={
    'random':'Random','balanced':'Balanced',
    'climate_similar':'Climate-similar','climate_diverse':'Climate-diverse',
    'representativeness':'Representativeness',
    'transfer_strong':'Transfer-strong','transfer_weak':'Transfer-weak',
    'cost_only':'Cost-only','cost_aware_strong':'Cost-aware-strong',
    'cost_aware_weak':'Cost-aware-weak',
}
TARGETS=[0.34,0.36,0.38,0.40,0.42,0.44]


def load():
    dfs=[]
    for s in SOURCES:
        for ft in FEATURES:
            f=os.path.join(OUT_DIR,f'scalable_{ft}_{s}_tier_b.csv')
            if not os.path.exists(f): continue
            d=pd.read_csv(f); d['source']=s; d['feature']=ft; dfs.append(d)
    return pd.concat(dfs,ignore_index=True)


def b2r(budgets, bas, target):
    """log-budget 线性插值。返回达到 target 的 budget；外推时返回 NaN。"""
    if np.max(bas) < target: return np.nan
    if np.min(bas) >= target: return float(budgets[0])
    lb = np.log(budgets)
    for i in range(len(bas)-1):
        if bas[i] < target <= bas[i+1]:
            t = (target - bas[i]) / (bas[i+1] - bas[i])
            return float(np.exp(lb[i] + t*(lb[i+1]-lb[i])))
    return float(budgets[-1])


def compute_iso(df, strategy, feature):
    sub = df[(df.strategy==strategy)&(df.feature==feature)&(df.budget>0)]
    if len(sub)==0: return None
    g = sub.groupby('budget')['mean_ood_ba_equal'].mean().reset_index().sort_values('budget')
    return g['budget'].values, g['mean_ood_ba_equal'].values


def main():
    df = load()
    print(f'Loaded {len(df)} rows\n')

    # ============ Table 1: Budget needed (per strategy × feature × τ) ============
    print('='*108)
    print(' TABLE 1: Budget needed to reach target BA (averaged over 10 sources × 3 seeds)')
    print('='*108)
    print(f'{"Strategy":<22}{"Feature":<10}', end='')
    for t in TARGETS:
        print(f'{"τ="+str(t):>13}', end='')
    print()
    print('-'*108)
    rows = []
    for strat in STRATS:
        for ft in FEATURES:
            r = compute_iso(df, strat, ft)
            if r is None: continue
            b, ba = r
            line = f'{STRAT_LABELS[strat]:<22}{ft.upper():<10}'
            row = {'strategy':strat,'feature':ft}
            for t in TARGETS:
                bb = b2r(b, ba, t)
                row[f'budget_τ{t}'] = bb
                cell = f'{bb:.0f}' if not np.isnan(bb) else 'X'
                line += f'{cell:>13}'
            print(line)
            rows.append(row)
        print('-'*108)

    bf = pd.DataFrame(rows)

    # ============ Table 2: AE / S2 ratio (核心: 鲁棒性检验) ============
    print('\n'+'='*100)
    print(' TABLE 2: AE/S2 budget ratio — robustness across strategies')
    print(' (<1 = AE more sample-efficient; X = either AE or S2 cannot reach τ within 50k budget)')
    print('='*100)
    print(f'{"Strategy":<22}', end='')
    for t in TARGETS: print(f'{"τ="+str(t):>13}', end='')
    print(f'{"median":>10}')
    print('-'*100)
    ratio_rows = []
    for strat in STRATS:
        line = f'{STRAT_LABELS[strat]:<22}'
        ratios = []
        ratio_row = {'strategy':strat}
        for t in TARGETS:
            ae = bf[(bf.strategy==strat)&(bf.feature=='ae')][f'budget_τ{t}'].values
            s2 = bf[(bf.strategy==strat)&(bf.feature=='s2')][f'budget_τ{t}'].values
            if len(ae)==0 or len(s2)==0 or np.isnan(ae[0]) or np.isnan(s2[0]):
                cell, r = 'X', np.nan
            else:
                r = ae[0]/s2[0]
                cell = f'{r:.2f}x'
                ratios.append(r)
            ratio_row[f'τ{t}'] = r
            line += f'{cell:>13}'
        med = np.median(ratios) if ratios else np.nan
        ratio_row['median'] = med
        line += f'{("%.2fx"%med) if not np.isnan(med) else "—":>10}'
        print(line)
        ratio_rows.append(ratio_row)

    rdf = pd.DataFrame(ratio_rows)

    # ============ Table 3: Fusion / S2 ratio ============
    print('\n'+'='*100)
    print(' TABLE 3: Fusion/S2 budget ratio — Fusion sample efficiency advantage')
    print('='*100)
    print(f'{"Strategy":<22}', end='')
    for t in TARGETS: print(f'{"τ="+str(t):>13}', end='')
    print(f'{"median":>10}')
    print('-'*100)
    for strat in STRATS:
        line = f'{STRAT_LABELS[strat]:<22}'
        ratios = []
        for t in TARGETS:
            fu = bf[(bf.strategy==strat)&(bf.feature=='fusion')][f'budget_τ{t}'].values
            s2 = bf[(bf.strategy==strat)&(bf.feature=='s2')][f'budget_τ{t}'].values
            if len(fu)==0 or len(s2)==0 or np.isnan(fu[0]) or np.isnan(s2[0]):
                cell = 'X'
            else:
                r = fu[0]/s2[0]
                cell = f'{r:.2f}x'
                ratios.append(r)
            line += f'{cell:>13}'
        med = np.median(ratios) if ratios else np.nan
        line += f'{("%.2fx"%med) if not np.isnan(med) else "—":>10}'
        print(line)

    # ============ 汇总判定 ============
    print('\n'+'='*70)
    print(' ROBUSTNESS VERDICT: AE/S2 sample-efficiency ratio')
    print('='*70)
    nstrat_ae_wins = 0
    valid = 0
    for _, row in rdf.iterrows():
        med = row['median']
        if not np.isnan(med):
            valid += 1
            if med < 1.0: nstrat_ae_wins += 1
    print(f'  Strategies where AE is more sample-efficient (median ratio < 1): '
          f'{nstrat_ae_wins}/{valid}')
    print(f'  Mean across strategies of median AE/S2 ratio: '
          f'{rdf["median"].mean():.3f}')
    print(f'  Range of median AE/S2: [{rdf["median"].min():.2f}x, {rdf["median"].max():.2f}x]')

    # 保存 CSV
    bf.to_csv(os.path.join(OUT_DIR, 'iso_accuracy_per_strategy_budget.csv'), index=False)
    rdf.to_csv(os.path.join(OUT_DIR, 'iso_accuracy_per_strategy_ae_over_s2.csv'), index=False)

    # ============ 图：策略×τ 的 AE/S2 ratio 热图 ============
    fig, ax = plt.subplots(figsize=(10, 6))
    mat = []
    for strat in STRATS:
        row = []
        for t in TARGETS:
            ae = bf[(bf.strategy==strat)&(bf.feature=='ae')][f'budget_τ{t}'].values
            s2 = bf[(bf.strategy==strat)&(bf.feature=='s2')][f'budget_τ{t}'].values
            if len(ae)==0 or len(s2)==0 or np.isnan(ae[0]) or np.isnan(s2[0]):
                row.append(np.nan)
            else:
                row.append(ae[0]/s2[0])
        mat.append(row)
    mat = np.array(mat)

    from matplotlib.colors import TwoSlopeNorm
    norm = TwoSlopeNorm(vmin=0.5, vcenter=1.0, vmax=2.0)
    im = ax.imshow(mat, cmap='RdBu_r', norm=norm, aspect='auto')
    ax.set_xticks(range(len(TARGETS)))
    ax.set_xticklabels([f'τ={t}' for t in TARGETS])
    ax.set_yticks(range(len(STRATS)))
    ax.set_yticklabels([STRAT_LABELS[s] for s in STRATS])
    ax.set_xlabel('Target Balanced Accuracy', fontsize=11)
    ax.set_title('AE/S2 budget ratio across strategies\n(<1 blue = AE more efficient; >1 red = S2 more efficient)',
                 fontsize=11, fontweight='bold')
    for i in range(len(STRATS)):
        for j in range(len(TARGETS)):
            v = mat[i,j]
            if np.isnan(v):
                ax.text(j, i, 'X', ha='center', va='center', fontsize=10, color='gray')
            else:
                color = 'white' if (v < 0.7 or v > 1.4) else 'black'
                ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                        fontsize=9, color=color)
    plt.colorbar(im, ax=ax, label='AE/S2 budget ratio')
    fig.tight_layout()
    for ext in ['png','pdf']:
        fig.savefig(os.path.join(FIG_DIR, f'exp6_v2_iso_accuracy_robustness.{ext}'))
    plt.close(fig)
    print('\n[OK] exp6_v2_iso_accuracy_robustness saved')


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()
