"""主分析（{root}/analysis，nc_main_analyze.py）与独立重写（{root}/indep，nc_main_indep.py）逐值交叉核对。

比对：逐域 d、g200、M、L_saved（删失标记）；e(k)、g(k) 全部曲线值；H1、H2、Holm 判定；M₁ 及其区间、无界、外推；
      附录 RF、LR 的 d > 0 比例；作物锚点逐国 d、g200、M、L_saved。数值容差 1e-9，判定须完全一致。
用法：python nc_main_crosscheck.py --root DIR
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

TOL = 1e-9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    a = ap.parse_args()
    A, I = os.path.join(a.root, 'analysis'), os.path.join(a.root, 'indep')
    errs, rep = [], {}

    def close(name, x, y):
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        bad = ~((np.isnan(x) & np.isnan(y)) | (np.abs(x - y) <= TOL))
        rep[name] = float(np.nanmax(np.abs(x - y))) if np.isfinite(x - y).any() else 0.0
        if bad.any():
            errs.append(f'{name}：{int(bad.sum())} 处不符（最大差 {rep[name]:.3g}）')

    def same(name, x, y):
        if x != y:
            errs.append(f'{name}：{x} ≠ {y}')

    # 逐域值
    da = pd.read_csv(os.path.join(A, 'domain_values.csv'))
    di = pd.read_csv(os.path.join(I, 'domain_values.csv'))
    J = da.merge(di, on=['task', 'domain', 'learner'], how='outer', suffixes=('_a', '_i'), indicator=True)
    same('逐域行集合', int((J._merge != 'both').sum()), 0)
    for c in ('d', 'g200', 'M'):
        close(f'逐域 {c}', J[f'{c}_a'], J[f'{c}_i'])
    lg = J[J.learner == 'lgb']
    same('L_saved 删失标记', lg.censored_a.astype(bool).tolist(), lg.censored_i.astype(bool).tolist())
    nc = ~lg.censored_a.astype(bool)
    close('L_saved（未删失）', lg.L_saved_a[nc], lg.L_saved_i[nc])
    close('k_max', lg.k_max_a, lg.k_max_i)
    # 曲线
    ca = pd.read_csv(os.path.join(A, 'curves_e_g.csv'))
    ci = pd.read_csv(os.path.join(I, 'curves_e_g.csv'))
    # 主实现另含 RF/LR 的 e(0)、e(200)、g(200)（已在逐域值中比对），独立实现另含作物曲线（下面单独比对）
    sel = lambda c: c[(c.learner == 'lgb') & c.task.isin(['lcz', 'lucas'])]  # noqa: E731
    J = sel(ca).merge(sel(ci), on=['task', 'domain', 'learner', 'quantity', 'k'], how='outer', suffixes=('_a', '_i'),
                      indicator=True)
    same('LightGBM 曲线行集合', int((J._merge != 'both').sum()), 0)
    close('e(k)、g(k)', J.value_a, J.value_i)
    # 确认性检验
    Ca = json.load(open(os.path.join(A, 'confirm.json')))
    Ci = json.load(open(os.path.join(I, 'confirm.json')))
    same('n', Ca['n'], Ci['n'])
    same('H1 n_pos', Ca['H1']['n_pos'], Ci['H1']['n_pos'])
    close('H1 p', Ca['H1']['p'], Ci['H1']['p'])
    same('H1 判定', Ca['H1']['rejected'], Ci['H1']['rejected'])
    for k in ('a', 'gamma', 'beta', 't', 'p'):
        close(f'H2 {k}', Ca['H2'][k], Ci['H2'][k])
    close('H2 se_hc3(β)', Ca['H2']['se_hc3'][2], Ci['H2']['se_hc3'])
    same('H2 df', Ca['H2']['df'], Ci['H2']['df'])
    same('H2 判定', Ca['H2']['rejected'], Ci['H2']['rejected'])
    if Ca['H2']['beta'] < 0:
        close('M₁ q', Ca['M1_q_beta_nonneg'], Ci['M1_q_beta_nonneg'])
        for fam in ('lcz', 'lucas'):
            ma, mi = Ca['M1'][fam], Ci['M1'][fam]
            close(f'M₁ {fam}', ma['value'], mi['value'])
            same(f'M₁ {fam} 无界', ma['unbounded'], mi['unbounded'])
            same(f'M₁ {fam} 外推', ma['extrapolated'], mi['extrapolated'])
            if ma['ci95'] is not None or mi['ci95'] is not None:
                close(f'M₁ {fam} 区间', ma['ci95'], mi['ci95'])
    # 附录
    Ea = json.load(open(os.path.join(A, 'estimates.json')))['appendix']
    Ei = json.load(open(os.path.join(I, 'appendix.json')))
    for lr_ in ('rf', 'lr'):
        close(f'附录 {lr_} d>0 比例', Ea[lr_]['frac_d_pos'], Ei[lr_]['d']['pooled']['frac_pos'])
    # 作物
    pa, pi = os.path.join(A, 'crop_anchor.csv'), os.path.join(I, 'crop_anchor.csv')
    if os.path.exists(pa) and os.path.exists(pi):
        xa, xi = pd.read_csv(pa), pd.read_csv(pi)
        J = xa.merge(xi, on='domain', how='outer', suffixes=('_a', '_i'), indicator=True)
        same('作物国家集合', int((J._merge != 'both').sum()), 0)
        for c in ('d', 'g200', 'M'):
            close(f'作物 {c}', J[f'{c}_a'], J[f'{c}_i'])
        same('作物删失', J.censored_a.astype(bool).tolist(), J.censored_i.astype(bool).tolist())
        nc = ~J.censored_a.astype(bool)
        close('作物 L_saved', J.L_saved_a[nc], J.L_saved_i[nc])
        cc = ci[~ci.task.isin(['lcz', 'lucas'])]                 # 独立实现的作物曲线
        if len(cc):
            xe = xa.rename(columns={'d': 'e200'})                  # 主实现把 e(200) 记为 d
            long = xe.melt(id_vars='domain', value_vars=[c for c in xe if c[0] in 'eg' and c[1:].isdigit()])
            long['quantity'], long['k'] = long.variable.str[0], long.variable.str[1:].astype(int)
            J = long.merge(cc, on=['domain', 'quantity', 'k'], how='outer', suffixes=('_a', '_i'), indicator=True)
            same('作物曲线行集合', int((J._merge != 'both').sum()), 0)
            close('作物 e(k)、g(k)', J.value_a, J.value_i)
    print(json.dumps(rep, indent=1, ensure_ascii=False))
    print('PASS' if not errs else 'FAIL\n' + '\n'.join(errs))
    json.dump(dict(max_abs_diff=rep, errors=errs, passed=not errs),
              open(os.path.join(a.root, 'crosscheck.json'), 'w'), indent=1, ensure_ascii=False)


if __name__ == '__main__':
    main()
