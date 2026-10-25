"""
crosscheck_confirm.py —— 方案 CONFIRM_PLAN_v5.md §7 第 6 步：indep_confirm.py 必须与 analyze_confirm.py 一致（误差 ≤ 1e-9）。
逐项核对两者的输出：逐单元量（臂、两臂之差、D）、国家值、汇总（n、均值、SD、CI、t、单侧 p、可判定）、B1 读法类别、
Holm（正式模式）。只打印键数、未配对数、NaN 不一致数与最大 |Δ|，不打印任何指标值。

用法（/Users/perng_mini/AlphaCrops_local/.venv/bin/python -B）：
  crosscheck_confirm.py --analysis <root>/analysis --indep <root>/indep
  crosscheck_confirm.py --retro --analysis <回溯根目录>/analysis --indep <indep 回溯输出目录>
    --retro：indep 的单元名 {国家}_s{划分} 对应 analyze 的 {国家}_F{划分+1}；回溯不做检验，不比 Holm。
退出码：全部一致 0，否则 1。
"""
import os
import re
import sys
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_v] = '2'   # 每进程 ≤ 2 线程（在导入 numpy 之前）
sys.dont_write_bytecode = True
import argparse  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

TOL = 1e-9
LEVEL = {'arm': 'arm', 'q': 'diff', 'D': 'D'}
UKEY = ['level', 'learner', 'unit', 'feat', 'r', 'name']


def name_of(n):
    """indep 量名 → analyze 量名。"""
    n = re.sub(r'^rf_', '', n)
    return re.sub(r'^(B1_(?:SRS-EXT|BA-SRS))_(\d+)$', r'\1\2', n)


def read(p):
    return pd.read_csv(p, float_precision='round_trip')


def compare(label, ind, ana, keys, cols, exact=(), finite_only=()):
    """ind 的每个键都须在 ana 中找到；cols 逐列比较（NaN 位置须相同，最大 |Δ| ≤ TOL）；
    finite_only 只在两边都有限处比较；exact 须完全相等。返回是否通过。"""
    assert not ana.duplicated(keys).any(), f'{label}：analyze 的键不唯一'
    assert not ind.duplicated(keys).any(), f'{label}：indep 的键不唯一'
    M = ind.merge(ana, on=keys, how='left', suffixes=('_i', '_a'), indicator=True)
    miss = int((M._merge != 'both').sum())
    nan_bad, worst = 0, 0.0
    for c in cols + list(finite_only):
        a, b = M[f'{c}_i'].to_numpy(float), M[f'{c}_a'].to_numpy(float)
        fin = np.isfinite(a) & np.isfinite(b)
        if c in cols:
            nan_bad += int((np.isnan(a) != np.isnan(b)).sum())
        if fin.any():
            worst = max(worst, float(np.abs(a[fin] - b[fin]).max()))
    ex_bad = sum(int((M[f'{c}_i'].astype(str) != M[f'{c}_a'].astype(str)).sum()) for c in exact)
    ok = miss == 0 and nan_bad == 0 and ex_bad == 0 and worst <= TOL
    print(f'  {label}：indep {len(ind)} 键，analyze 未找到 {miss}，NaN 不一致 {nan_bad}'
          + (f'，精确列不一致 {ex_bad}（{"/".join(exact)}）' if exact else '')
          + f'，最大 |Δ| {worst:.2e}（{"/".join(cols + list(finite_only))}）→ {"一致" if ok else "不一致"}', flush=True)
    return ok


def reading_cat(s):
    """B1 读法的类别（两份脚本措辞不同，只比类别）。"""
    s = str(s)
    if s.startswith('补采位置'):
        return 'lt1'
    if s.startswith('未能区分'):
        return 'undist'
    if s.startswith('不可'):
        return 'na'
    return 'pos' if '正' in s else 'neg' if '负' in s else '?'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--analysis', required=True, help='analyze_confirm.py 的输出目录（{root}/analysis）')
    ap.add_argument('--indep', required=True, help='indep_confirm.py 的输出目录')
    ap.add_argument('--retro', action='store_true')
    a = ap.parse_args()
    ok = True
    print(f'analyze：{os.path.abspath(a.analysis)}\nindep：{os.path.abspath(a.indep)}\n容差 {TOL:g}；只列差异统计，不列指标值')

    # ---- 逐单元量
    IU, AU = read(os.path.join(a.indep, 'indep_unit.csv')), read(os.path.join(a.analysis, 'unit_rows.csv'))
    IU = IU.assign(level=IU.level.map(LEVEL), name=IU.name.map(name_of))
    if a.retro:
        IU['unit'] = [re.sub(r'_s(\d)$', lambda m: f'_F{int(m.group(1)) + 1}', u) for u in IU.unit]
    same_units = set(IU.unit) == set(AU.unit)
    ok &= same_units
    print(f'纳入单元：indep {IU.unit.nunique()}，analyze {AU.unit.nunique()}，集合相同 {same_units}')
    print('逐单元量：')
    for lev, cols, fo in (('arm', ['OA_star', 'value', 'oa_t0', 'ba_t0'], ['cov']),
                          ('diff', ['OA_star', 'value'], ['cov']),
                          ('D', ['OA_star', 'value'], ['cov'])):
        ok &= compare(lev, IU[IU.level == lev][UKEY + cols + fo], AU[AU.level == lev][UKEY + cols + fo], UKEY, cols,
                      finite_only=fo)

    # ---- 国家值
    IC = read(os.path.join(a.indep, 'indep_country.csv'))
    IC = IC.melt(id_vars=['name', 'learner', 'feat'], var_name='country', value_name='value')
    IC['name'] = IC.name.map(name_of)
    AC = read(os.path.join(a.analysis, 'country_values.csv'))
    print('国家值：')
    ok &= compare('国家值', IC, AC[['learner', 'name', 'feat', 'country', 'value']], ['learner', 'name', 'feat', 'country'],
                  ['value'])

    # ---- 汇总（analyze 同一量可出现在多个编号下，先核对重复行相同再去重）
    IE = read(os.path.join(a.indep, 'indep_estimates.csv')).assign(quantity=lambda d: d.name.map(name_of))
    AE = read(os.path.join(a.analysis, 'estimates.csv'))
    k = ['learner', 'quantity', 'feat', 'set']
    num = ['mean', 'sd', 'lo95', 'hi95', 'lo975', 'hi975', 't', 'p_one']
    dup = AE.groupby(k)[num + ['n']].nunique(dropna=False).max().max()
    ok &= bool(dup == 1)
    print(f'汇总：analyze 同一量的重复行一致 {dup == 1}')
    AE1 = AE.drop_duplicates(k)
    ok &= compare('汇总', IE[k + num + ['n', 'n_pos', 'decidable']], AE1[k + num + ['n', 'n_pos', 'decidable']], k, num,
                  exact=('n', 'n_pos', 'decidable'))
    ib = IE[IE.quantity.str.startswith('B1_') & (IE.feat == 'both')]
    if len(ib):
        ab = AE[(AE.family == 'B1') & (AE.feat == 'both')]
        ok &= compare('B1 读法类别', ib[k].assign(cat=ib.reading.map(reading_cat)),
                      ab[k].assign(cat=ab.reading.map(reading_cat)), k, [], exact=('cat',))

    # ---- Holm（正式模式）
    if not a.retro:
        IT, AT = read(os.path.join(a.indep, 'indep_tests.csv')), read(os.path.join(a.analysis, 'tests.csv'))
        ok &= compare('Holm', IT[['hyp', 'p_one', 'alpha_step', 'reject']], AT[['hyp', 'p_one', 'alpha_step', 'reject']],
                      ['hyp'], ['p_one', 'alpha_step'], exact=('reject',))
    print(f'核对结论：{"全部一致" if ok else "有不一致项"}')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
