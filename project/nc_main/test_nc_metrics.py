"""nc_metrics.delta_v 的三项单元测试（方案 v6 第 3 节，冻结代码前必须通过）。

1. 在 RSE 作物与试点（LCZ、LUCAS）的已存测试概率上，逐值与参考实现 design_calc/metrics.py 的 V_min 相差 ≤ 1e-9；
   另按 design_calc/step2_compare.py 的聚合口径汇到域级，与已存 domain_metrics.parquet 的 V_min 逐值相差 ≤ 1e-9。
2. 反对称：|Δ(A → B) + Δ(B → A)| ≤ 1e-12。
3. 前沿上移 (0, +δ)：|Δ_V − δ| ≤ 1e-9（真实曲线与合成随机路径）。
只读已有文件；结果写入同目录 test_nc_metrics_result.json。
"""
import importlib.util
import json
import os
import sys

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_v] = '2'
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'unified_figs'))
import nc_metrics as NM  # noqa: E402
import prep_unified as PU  # noqa: E402  只 import，不运行 main

_spec = importlib.util.spec_from_file_location('ref_metrics', os.path.join(HERE, 'design_calc', 'metrics.py'))
REF = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(REF)
AC = PU.AC

KS = (50, 100, 200, 400)
R = 3
TOL1, TOL2, TOL3 = 1e-9, 1e-12, 1e-9


def ref_vmin(cA, cB):
    """参考实现的 V_min（与 design_calc/metrics.all_metrics 中同一行）。"""
    gm = min(cA[0][REF.I0], cB[0][REF.I0]) + REF.OFFS
    return REF.paired(REF.env_fd(cB[0], cB[1], gm), REF.env_fd(cA[0], cA[1], gm))


def pairs_of():
    """(comp, feat, r, A 的键, B 的键)：Δ_R、Δ_L（含 g(k)）与 e(k)。"""
    out = [('R', 'ae-s2', 0, ('s2', 'Z0', 0), ('ae', 'Z0', 0))]
    for f in ('ae', 's2'):
        out += [(f'RT{k}', f, r, (f, 'Z0', 0), (f, f'RT{k}', r)) for k in KS for r in range(R)]
        out.append(('REF', f, 0, (f, 'Z0', 0), (f, 'REF', 0)))
    out += [(f'E{k}', 'e', r, ('s2', f'RT{k}', r), ('ae', 'Z0', 0)) for k in KS for r in range(R)]
    return out


def dom_table(U):
    """与 design_calc/step2_compare.dom_table 相同：抽样平均 → 单元平均 → 域；both = 两特征平均。"""
    x = U.groupby(['task', 'domain', 'unit', 'comp', 'feat']).value.mean().reset_index()
    d = x.groupby(['task', 'domain', 'comp', 'feat']).value.mean().reset_index()
    fe = d.set_index(['task', 'domain', 'comp', 'feat']).value.unstack('feat')
    both = fe[['ae', 's2']].mean(axis=1, skipna=False)
    return pd.concat([fe['ae-s2'].dropna().rename('value').reset_index().assign(feat='R'),
                      both.dropna().rename('value').reset_index().assign(feat='both'),
                      fe['ae'].dropna().rename('value').reset_index().assign(feat='ae'),
                      fe['s2'].dropna().rename('value').reset_index().assign(feat='s2')])


def main():
    assert np.array_equal(NM.OFFS, REF.OFFS) and NM.I0 == REF.I0 and len(AC.TAUS) == NM.N_TAU and AC.TAUS[NM.I0] == 0
    rows, t2, t3, curves = [], [], [], []
    for task, units in (('crop', PU.crop_units()), ('lcz', PU.pilot_units('lcz')), ('lucas', PU.pilot_units('lucas'))):
        for U in units:
            CV = AC.unit_curves(U['yT'], U['big'], U['P'])
            curves += list(CV.values())
            for comp, feat, r, ka, kb in pairs_of():
                a, b = CV[ka], CV[kb]
                v = NM.delta_v(a, b)
                rows.append(dict(task=task, domain=U['domain'], unit=U['unit'], comp=comp, feat=feat, r=r,
                                 value=v, ref=ref_vmin(a, b)))
                t2.append(abs(v + NM.delta_v(b, a)))
            print(task, U['unit'], flush=True)
    U = pd.DataFrame(rows)

    # 测试 1a：逐值对参考实现
    err1a = float(np.max(np.abs(U.value - U.ref)))
    # 测试 1b：域级对已存 domain_metrics.parquet（只含 Δ_R 与 Δ_L；e(k) 为新量，只做 1a）
    D = dom_table(U[~U.comp.str.startswith('E')])
    S = pd.read_parquet(os.path.join(HERE, 'design_calc', 'domain_metrics.parquet'))
    S = S[(S.metric == 'V_min') & S.feat.isin(['R', 'both', 'ae', 's2'])]
    J = D.merge(S[['task', 'domain', 'comp', 'feat', 'value']], on=['task', 'domain', 'comp', 'feat'],
                how='outer', suffixes=('', '_stored'), indicator=True)
    assert (J._merge == 'both').all(), J[J._merge != 'both']
    err1b = float(np.max(np.abs(J.value - J.value_stored)))

    # 测试 3：真实曲线与合成随机路径上移 δ
    rng = np.random.default_rng(42)
    synth = []
    for _ in range(500):
        oa = np.clip(70 + np.cumsum(rng.normal(0, 1.5, NM.N_TAU)), 1, 99)
        ba = np.clip(55 + np.cumsum(rng.normal(0, 2.0, NM.N_TAU)), 1, 99)
        synth.append((oa, ba))
    for c in curves + synth:
        for dl in (-5.0, 0.37, 3.0):
            t3.append(abs(NM.delta_v(c, (c[0], c[1] + dl)) - dl))

    res = {
        'test1a_max_abs_err_vs_ref_V_min': err1a, 'test1a_n': int(len(U)),
        'test1a_n_by_task': U.groupby('task').size().to_dict(),
        'test1b_max_abs_err_vs_domain_metrics': err1b, 'test1b_n': int(len(J)),
        'test2_max_abs_antisym': float(max(t2)), 'test2_n': len(t2),
        'test3_max_abs_err_shift': float(max(t3)), 'test3_n': len(t3),
    }
    res['pass'] = bool(err1a <= TOL1 and err1b <= TOL1 and res['test2_max_abs_antisym'] <= TOL2
                       and res['test3_max_abs_err_shift'] <= TOL3)
    json.dump(res, open(os.path.join(HERE, 'test_nc_metrics_result.json'), 'w'), indent=1, ensure_ascii=False)
    print(json.dumps(res, indent=1, ensure_ascii=False))
    assert res['pass']


if __name__ == '__main__':
    main()
