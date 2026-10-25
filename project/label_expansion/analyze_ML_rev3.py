"""
rev3 补充估计量（ANALYSIS_PLAN_guidance_rev3.md 第 B 节）：各学习器下 AE 特征相对 S2 特征的差随 k 的变化（B1）、
随机标签价值（B2）、RF 在 k=0 时的 AE−S2（B3）。
用法：
  python analyze_ML_rev3.py                  # 正式（四个学习器都完成后）
  python analyze_ML_rev3.py --noise          # 指标换成随机数，只测流程
  python analyze_ML_rev3.py --allow-partial  # 允许不全（只测流程）
输出：ML_OUT/report_ML_rev3[_NOISE].txt、summary_ML_rev3[_NOISE].csv
"""
import os
import sys

import numpy as np
import pandas as pd

import analyze_G as A
import analyze_ML as M  # 导入时读取 --noise / --allow-partial
from multilearner_G import OUT

KS = (0, 50, 100, 200, 400)
DROP = ['RO', 'PT']


def unit_feat_delta(R, metric):
    """同一单元、同一 k 下 AE 特征 − S2 特征（SRS 与 k=0 的行；k=0 两个重复相同）。"""
    S = R[R.method.isin(['srs', 'foreign_only']) & R.k.isin(KS)]
    P = S.pivot_table(index=['learner'] + A.UNIT + ['k'], columns='clf', values=metric)
    return (P['ae'] - P['s2']).rename('d').reset_index().dropna()


def unit_gain(R, metric):
    """同一单元、同一特征下 SRS(k) − k=0。"""
    S = R[R.method.isin(['srs', 'foreign_only']) & R.k.isin(KS)]
    P = S.pivot_table(index=['learner'] + A.UNIT + ['clf'], columns='k', values=metric)
    out = []
    for k in (50, 200, 400):
        out.append((P[k] - P[0]).rename('d').reset_index().assign(k=k))
    return pd.concat(out).dropna()


def country(d):
    return d.groupby('target').d.mean()


def summ(v, drop=None):
    if drop:
        v = v.drop([c for c in drop if c in v.index])
    return A.t_summary(v)


def main():
    tag = '_NOISE' if M.NOISE else ''
    sys.stdout = A.Tee(os.path.join(OUT, f'report_ML_rev3{tag}.txt'))
    if M.NOISE:
        print('!!! 噪声模式：指标为随机数，结果无意义，只用于测试流程 !!!')
    print('方案：ANALYSIS_PLAN_guidance_rev3.md（补充估计量；描述性；LR 的 SRS/k=0 行已非盲，见 rev3 §A）')
    R = M.load_all()
    problems = M.completeness(R)
    if problems and not M.PARTIAL:
        raise SystemExit('完整性检查未通过：' + '；'.join(problems))
    if problems:
        print('!!! 完整性检查未通过，只用于流程测试：' + '；'.join(problems) + ' !!!')
    if 'status' in R:
        R = R[R.status.fillna('ok') != 'failed']
    rows = []
    for metric in ('ba_ge30', 'oa'):
        print('\n' + '=' * 16 + f' B1 特征效应 Δ_feat(L, k) = {metric}(AE 特征) − {metric}(S2 特征)，同一批 SRS 标签 ' + '=' * 16)
        D = unit_feat_delta(R, metric)
        for L in M.ORDER:
            DL = D[D.learner == L]
            if DL.empty:
                continue
            for k in KS:
                for lab, drop in (('11 国', None), ('去 RO、PT', DROP)):
                    s = summ(country(DL[DL.k == k]), drop)
                    rows.append(dict(block='B1', metric=metric, learner=L, k=k, subset=lab, **s))
                    print(f'  {M.NAME[L]:<14} k={k:<4}{lab:<8}{A.fmt(s)}')
            ch = DL.pivot_table(index=A.UNIT, columns='k', values='d')
            if {0, 400} <= set(ch.columns):
                c = (ch[400] - ch[0]).rename('d').reset_index()
                for lab, drop in (('11 国', None), ('去 RO、PT', DROP)):
                    s = summ(country(c), drop)
                    rows.append(dict(block='B1_change', metric=metric, learner=L, k=400, subset=lab, **s))
                    print(f'  {M.NAME[L]:<14} 变化 0→400 {lab:<8}{A.fmt(s)}')

    # "整体平移"判据（ba_ge30，11 国）
    T = pd.DataFrame(rows)
    b = T[(T.metric == 'ba_ge30') & (T.subset == '11 国')]
    ok = []
    for L in M.ORDER:
        x0 = b[(b.block == 'B1') & (b.learner == L) & (b.k == 0)]
        x4 = b[(b.block == 'B1') & (b.learner == L) & (b.k == 400)]
        ch = b[(b.block == 'B1_change') & (b.learner == L)]
        if len(x0) and len(x4) and len(ch):
            same = np.sign(x0['mean'].iloc[0]) == np.sign(x4['mean'].iloc[0])
            narrow = ch['lo95'].iloc[0] > -1 and ch['hi95'].iloc[0] < 1
            ok.append((L, bool(same and narrow)))
    n_ok = sum(v for _, v in ok)
    print(f'\n"整体平移"判据（≥3/4 个学习器：k=0 与 k=400 同号，且 0→400 变化的 95% CI 在 ±1 pp 内）：'
          f'{", ".join(f"{M.NAME[L]} {"满足" if v else "不满足"}" for L, v in ok)} → '
          f'{"可写为 B 级：AE 作为特征的优势近似为整体平移（" + str(n_ok) + "/4）" if n_ok >= 3 else "依学习器而定"}')

    print('\n' + '=' * 16 + ' B2 随机标签价值 G(L, k) = ba_ge30(SRS k) − ba_ge30(k=0) ' + '=' * 16)
    Gd = unit_gain(R, 'ba_ge30')
    small = []
    for L in M.ORDER:
        GL = Gd[Gd.learner == L]
        if GL.empty:
            continue
        for k in (50, 200, 400):
            for clf in ('ae', 's2', 'avg'):
                g = GL[GL.k == k] if clf == 'avg' else GL[(GL.k == k) & (GL.clf == clf)]
                if clf == 'avg':
                    g = g.groupby(A.UNIT).d.mean().reset_index()
                for lab, drop in (('11 国', None), ('去 RO、PT', DROP)):
                    s = summ(country(g), drop)
                    rows.append(dict(block='B2', metric='ba_ge30', learner=L, k=k, clf=clf, subset=lab, **s))
                    print(f'  {M.NAME[L]:<14} k={k:<4}{clf:<4}{lab:<8}{A.fmt(s)}')
                    if k == 200 and clf == 'avg' and lab == '去 RO、PT':
                        small.append((L, s['hi95'] < 2))
    print(f'\n"随机标签边际价值小"判据（每个学习器去掉 RO、PT 后 G(L,200) 的 95% 上界 < 2 pp）：'
          f'{", ".join(f"{M.NAME[L]} {"满足" if v else "不满足"}" for L, v in small)} → '
          f'{"可写为 B 级" if small and all(v for _, v in small) and len(small) == 4 else "依学习器而定"}')

    print('\n' + '=' * 16 + ' B3 RF 在 k=0 时的 AE − S2（与 Cheng 主分类器的对照口径，描述） ' + '=' * 16)
    for metric in ('oa', 'ba_ge30'):
        D = unit_feat_delta(R, metric)
        d = D[(D.learner == 'rf') & (D.k == 0)]
        if len(d):
            for lab, drop in (('11 国', None), ('去 RO、PT', DROP)):
                print(f'  RF k=0 {metric:<8}{lab:<8}', A.fmt(summ(country(d), drop)))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, f'summary_ML_rev3{tag}.csv'), index=False)


if __name__ == '__main__':
    main()
