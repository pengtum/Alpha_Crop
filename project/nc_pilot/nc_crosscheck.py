"""
交叉核对：nc_analyze（import RSE 冻结函数）与 nc_indep（从头独立实现）的逐域值与判定。

用法：python nc_crosscheck.py --a <nc_analyze 输出目录> --b <nc_indep 输出目录>
通过条件：逐域 dR、dL_k、I_k 的 NaN 位置相同且最大绝对差 < 1e-9；判定用的 m_R、m_L(200)、成立与否、L_AE 相同。
"""
import argparse
import sys

import numpy as np
import pandas as pd

KS = ['50', '100', '200', '400', 'all']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', required=True)
    ap.add_argument('--b', required=True)
    a = ap.parse_args()
    A = pd.read_csv(f'{a.a}/domain_values.csv').set_index(['task', 'dom'])
    B = pd.read_csv(f'{a.b}/indep_domain.csv').set_index(['task', 'dom']).loc[A.index]
    cols = ['dR'] + [f'dL_{k}' for k in KS] + [f'I_{k}' for k in KS]
    ok = True
    worst = 0.0
    for c in cols:
        x, y = A[c].values.astype(float), B[c].values.astype(float)
        if not np.array_equal(np.isnan(x), np.isnan(y)):
            print('NaN 位置不同：', c); ok = False
        m = np.isfinite(x) & np.isfinite(y)
        if m.any():
            worst = max(worst, float(np.abs(x[m] - y[m]).max()))
    TA = pd.read_csv(f'{a.a}/task_summary.csv').set_index('task')
    TB = pd.read_csv(f'{a.b}/indep_task.csv').set_index('task')
    for t in TA.index:
        pairs = [('m_R', 'm_R_dec'), ('m_L_200', 'm_L_200_dec')]
        for ca, cb in pairs:
            x, y = float(TA.loc[t, ca]), float(TB.loc[t, cb])
            if not ((np.isnan(x) and np.isnan(y)) or abs(x - y) < 1e-9):
                print('判定量不同：', t, ca, x, y); ok = False
        if bool(TA.loc[t, 'holds']) != bool(TB.loc[t, 'holds']) or str(TA.loc[t, 'L_AE']) != str(TB.loc[t, 'L_AE']):
            print('成立或 L_AE 不同：', t); ok = False
    print(f'逐域最大绝对差 {worst:.2e}；', '交叉核对通过' if ok and worst < 1e-9 else '交叉核对未通过')
    sys.exit(0 if ok and worst < 1e-9 else 1)


if __name__ == '__main__':
    main()
