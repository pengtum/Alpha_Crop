"""cost_only vs cost_aware_strong/weak 详细对比 - 多维度展开。"""
import os, re
import numpy as np
import pandas as pd

OUT_DIR = 'results/experiments/exp6_scalable_mapping'
COST = {'AT':1.05,'BE':1.00,'DE':1.00,'EE':0.36,'LT':0.40,'LV':0.31,
        'PT':0.48,'SE':1.36,'SI':0.63,'SK':0.38}
SOURCES = ['AT','BE','DE','EE','LT','LV','PT','SE','SI','SK']
FEATURES = ['ae','s2','fusion']
FT_LABELS = {'ae':'AlphaEarth','s2':'Sentinel-2','fusion':'Fusion'}
STRATS = ['balanced','cost_only','cost_aware_strong','cost_aware_weak']


def parse_info(s):
    if pd.isna(s) or s == '{}': return {}
    out = {}
    for c, n in re.findall(r"['\"]([A-Z]{2})['\"]?\s*:\s*(?:np\.int64\()?(\d+)\)?", str(s)):
        out[c] = int(n)
    return out


def compute_cost(s):
    return sum(n * COST.get(c, 1.0) for c, n in parse_info(s).items())


def load():
    dfs = []
    for s in SOURCES:
        for ft in FEATURES:
            f = os.path.join(OUT_DIR, f'scalable_{ft}_{s}_tier_b.csv')
            if not os.path.exists(f): continue
            df = pd.read_csv(f)
            df['source'], df['feature'] = s, ft
            df['aug_cost'] = df['sampling_info'].apply(compute_cost)
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def table1_overall(df):
    """Table 1: 跨 10 source × 5 budget 总览。"""
    print('\n' + '='*95)
    print(' TABLE 1: Overall performance (averaged over 10 sources × 3 seeds, all budgets)')
    print('='*95)
    hdr = f'{"Feature":<12}{"Strategy":<22}{"BA_mean":>9}{"BA_std":>8}{"Cost_mean":>11}{"Cost_std":>10}{"ΔBA_vs_Bal":>12}{"ΔCost%":>9}'
    print(hdr); print('-'*95)
    for ft in FEATURES:
        bal = df[(df.feature==ft)&(df.strategy=='balanced')&(df.budget>0)]
        bal_ba, bal_cost = bal.mean_ood_ba_equal.mean(), bal.aug_cost.mean()
        for strat in STRATS:
            sub = df[(df.feature==ft)&(df.strategy==strat)&(df.budget>0)]
            ba_m, ba_s = sub.mean_ood_ba_equal.mean(), sub.mean_ood_ba_equal.std()
            c_m, c_s = sub.aug_cost.mean(), sub.aug_cost.std()
            dba = (ba_m - bal_ba)*100  # pp
            dc = (c_m - bal_cost)/bal_cost*100
            print(f'{FT_LABELS[ft]:<12}{strat:<22}{ba_m:>9.4f}{ba_s:>8.4f}{c_m:>11.0f}{c_s:>10.0f}{dba:>+11.2f}pp{dc:>+8.1f}%')
        print('-'*95)


def table2_by_budget(df, ft='fusion'):
    """Table 2: 分 budget 细看（某一 feature）。"""
    print(f'\n{"="*95}')
    print(f' TABLE 2: {FT_LABELS[ft]} — by budget (mean over 10 sources × 3 seeds)')
    print('='*95)
    print(f'{"Budget":<10}{"Strategy":<22}{"BA":>9}{"Cost":>10}{"ΔBA_vs_cost_only":>20}{"ΔCost_vs_cost_only":>22}')
    print('-'*95)
    for budget in [1000, 5000, 10000, 20000, 50000]:
        co = df[(df.feature==ft)&(df.strategy=='cost_only')&(df.budget==budget)]
        co_ba, co_cost = co.mean_ood_ba_equal.mean(), co.aug_cost.mean()
        for strat in STRATS:
            sub = df[(df.feature==ft)&(df.strategy==strat)&(df.budget==budget)]
            ba, c = sub.mean_ood_ba_equal.mean(), sub.aug_cost.mean()
            dba = (ba - co_ba)*100
            dc = (c - co_cost)/co_cost*100 if co_cost > 0 else 0
            if strat == 'cost_only':
                print(f'{budget:<10}{strat:<22}{ba:>9.4f}{c:>10.0f}{"(reference)":>20}{"(reference)":>22}')
            else:
                print(f'{budget:<10}{strat:<22}{ba:>9.4f}{c:>10.0f}{dba:>+18.2f}pp{dc:>+20.1f}%')
        print('-'*95)


def table3_pareto(df, budget=10000):
    """Table 3: 每 source @ budget=10k 的 Pareto 判定。"""
    print(f'\n{"="*110}')
    print(f' TABLE 3: Per-source Pareto dominance vs cost_only (budget={budget}, Fusion)')
    print('='*110)
    print(f'{"Source":<8}{"cost_only_BA":>13}{"cost_only_$":>12}  {"cost_aware_strong":<30}  {"cost_aware_weak":<30}')
    print(f'{"":<8}{"":>13}{"":>12}  {"ΔBA":>8}{"Δ$%":>9}{"Pareto":>10}  {"ΔBA":>8}{"Δ$%":>9}{"Pareto":>10}')
    print('-'*110)
    for src in SOURCES:
        co = df[(df.source==src)&(df.feature=='fusion')&(df.strategy=='cost_only')&(df.budget==budget)]
        if len(co)==0: continue
        co_ba, co_c = co.mean_ood_ba_equal.mean(), co.aug_cost.mean()
        line = f'{src:<8}{co_ba:>13.4f}{co_c:>12.0f}'
        for strat in ['cost_aware_strong','cost_aware_weak']:
            sub = df[(df.source==src)&(df.feature=='fusion')&(df.strategy==strat)&(df.budget==budget)]
            if len(sub)==0:
                line += f'  {"N/A":<30}'; continue
            ba, c = sub.mean_ood_ba_equal.mean(), sub.aug_cost.mean()
            dba = (ba - co_ba)*100
            dc = (c - co_c)/co_c*100
            # Pareto: BA 不降 (tol 0.2pp) AND cost 不升
            pareto = 'YES' if (dba >= -0.2 and dc <= 0) else 'no'
            line += f'    {dba:>+6.2f}pp{dc:>+7.1f}%{pareto:>9}'
        print(line)
    print('-'*110)


def table4_allocation_example(df, src='DE', budget=10000):
    """Table 4: 一个 source 下 cost_only vs cost_aware 的实际采样分配。"""
    print(f'\n{"="*80}')
    print(f' TABLE 4: Sample allocation example ({src} source, Fusion, budget={budget}, seed=42)')
    print('='*80)
    print(f'{"Target":<8}{"Cost":>7}  {"cost_only":>12}  {"cost_aware_strong":>20}  {"cost_aware_weak":>18}')
    print('-'*80)
    strat_list = ['cost_only','cost_aware_strong','cost_aware_weak']
    example = {s:{} for s in strat_list}
    for strat in strat_list:
        sub = df[(df.source==src)&(df.feature=='fusion')&(df.strategy==strat)&
                 (df.budget==budget)]
        if len(sub) > 0:
            example[strat] = parse_info(sub.iloc[0]['sampling_info'])
    all_tgts = sorted(set(t for d in example.values() for t in d))
    totals = {s:0 for s in strat_list}
    for tgt in all_tgts:
        row = f'{tgt:<8}{COST.get(tgt,1.0):>7.2f}'
        for strat in ['cost_only','cost_aware_strong','cost_aware_weak']:
            n = example.get(strat,{}).get(tgt,0)
            totals[strat] += n
            pct = n/budget*100
            row += f'  {n:>6} ({pct:>4.1f}%)'
        print(row)
    print('-'*80)
    row = f'{"Total":<8}{"":>7}'
    for strat in ['cost_only','cost_aware_strong','cost_aware_weak']:
        row += f'  {totals[strat]:>6}         '
    print(row)
    # 总成本
    print()
    for strat in ['cost_only','cost_aware_strong','cost_aware_weak']:
        c = sum(n*COST.get(t,1.0) for t,n in example.get(strat,{}).items())
        print(f'  {strat:<22} aug_cost = {c:.0f}')


def table5_ranking(df, budget=10000):
    """Table 5: 三策略谁最优 per source × feature (rank)."""
    print(f'\n{"="*75}')
    print(f' TABLE 5: Best strategy among {{cost_only, cost_aware_strong, cost_aware_weak}}')
    print(f'         by BA (budget={budget}), per (source, feature)')
    print('='*75)
    print(f'{"Source":<8}{"AE":>22}{"S2":>22}{"Fusion":>22}')
    print('-'*75)
    win_count = {'cost_only':0, 'cost_aware_strong':0, 'cost_aware_weak':0}
    for src in SOURCES:
        line = f'{src:<8}'
        for ft in FEATURES:
            best, best_ba = None, -1
            for strat in ['cost_only','cost_aware_strong','cost_aware_weak']:
                sub = df[(df.source==src)&(df.feature==ft)&(df.strategy==strat)&(df.budget==budget)]
                if len(sub) == 0: continue
                ba = sub.mean_ood_ba_equal.mean()
                if ba > best_ba: best_ba, best = ba, strat
            if best:
                win_count[best] += 1
                label = {'cost_only':'cost_only','cost_aware_strong':'cost_aware_str',
                         'cost_aware_weak':'cost_aware_wk'}[best]
                line += f'  {label:>12} ({best_ba:.3f})'
            else:
                line += f'  {"N/A":>22}'
        print(line)
    print('-'*75)
    print(f'\n  Win count:  cost_only={win_count["cost_only"]}/30  '
          f'cost_aware_strong={win_count["cost_aware_strong"]}/30  '
          f'cost_aware_weak={win_count["cost_aware_weak"]}/30')


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    df = load()
    print(f'Loaded {len(df)} rows')
    table1_overall(df)
    table2_by_budget(df, 'fusion')
    table2_by_budget(df, 'ae')
    table3_pareto(df, 10000)
    table4_allocation_example(df, 'DE', 10000)
    table4_allocation_example(df, 'AT', 10000)
    table5_ranking(df, 10000)
