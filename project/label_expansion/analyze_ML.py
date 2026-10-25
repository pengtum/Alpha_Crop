"""
多学习器稳健性分析（ANALYSIS_PLAN_guidance_rev2.md 第 4 节）。
用法：
  python analyze_ML.py                 # 正式分析（三个学习器都完成后）
  python analyze_ML.py --noise         # 指标换成随机数，只用于在不看结果的情况下测试流程
  python analyze_ML.py --allow-partial # 允许单元不全（只用于流程测试）
输出：ML_OUT/report_ML[_NOISE].txt、summary_ML[_NOISE].csv、figures/fig_multilearner[_NOISE].png
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

import analyze_G as A
from guidance_G import E1
from multilearner_G import OUT, LEARNERS

NOISE = '--noise' in sys.argv
PARTIAL = '--allow-partial' in sys.argv
ORDER = ('lgb',) + LEARNERS
NAME = {'lgb': 'LightGBM（G 原结果）', 'lr': 'Logistic 回归', 'mlp': 'MLP 集成', 'rf': '随机森林'}
METRICS = ['ba_ge30', 'oa', 'f1_ge30', 'ba_P1', 'oa_P1', 'ba_P2', 'oa_P2', 'ba_far', 'oa_far']


def load_all():
    G = A.load_rows()
    G = G[G.method.isin(E1 + ['srs', 'foreign_only']) & G.cond.isin(['AE', 'S2', 'none'])
          & G.clf.isin(['s2', 'ae']) & G.k.isin([0, 50, 100, 200, 400])].assign(learner='lgb')
    fs = glob.glob(os.path.join(OUT, 'rows_*_*.csv'))
    M = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()
    R = pd.concat([G, M], ignore_index=True)
    if NOISE:
        rng = np.random.default_rng(0)
        for m in METRICS:
            R[m] = rng.normal(60, 5, len(R))
    return R


def completeness(R):
    """正式分析的完整性条件（rev2）：每个学习器 1,848 行、键无重复、AE/S2 成对、选样哈希与 G 一致。"""
    out = []
    key = ['learner'] + A.UNIT + ['cond', 'method', 'clf', 'k']
    for L in ORDER:
        n = int((R.learner == L).sum())
        if n != 1848:
            out.append(f'{L} 有 {n} 行（应为 1848）')
    if R.duplicated(key).any():
        out.append(f'重复键 {int(R.duplicated(key).sum())} 个')
    e1 = R[R.method.isin(E1)]
    pair = e1.groupby(['learner'] + A.UNIT + ['method', 'clf', 'k']).cond.nunique()
    if (pair != 2).any():
        out.append(f'AE/S2 不成对的格子 {int((pair != 2).sum())} 个')
    g = R[R.learner == 'lgb'].set_index(A.UNIT + ['cond', 'method', 'clf', 'k']).sel_hash
    for L in LEARNERS:
        m = R[(R.learner == L) & (R.k > 0)].set_index(A.UNIT + ['cond', 'method', 'clf', 'k']).sel_hash
        common = m.index.intersection(g.index)
        if len(common) and (m.loc[common] != g.loc[common]).any():
            out.append(f'{L} 的选样哈希与 G 不一致')
    return out


def tost_p(s, sesoi=A.SESOI):
    return max(stats.t.sf((s['mean'] + sesoi) / s['se'], s['n'] - 1), stats.t.sf((sesoi - s['mean']) / s['se'], s['n'] - 1))


def main():
    tag = '_NOISE' if NOISE else ''
    sys.stdout = A.Tee(os.path.join(OUT, f'report_ML{tag}.txt'))
    if NOISE:
        print('!!! 噪声模式：指标为随机数，结果无意义，只用于测试流程 !!!')
    print('方案：ANALYSIS_PLAN_guidance_rev2.md（预先指定的学习器稳健性分析；G s0/s1 测试集已被复用，非确认性）')
    R = load_all()
    problems = completeness(R)
    if problems and not PARTIAL:
        raise SystemExit('完整性检查未通过：' + '；'.join(problems) + '（流程测试请加 --allow-partial）')
    if problems:
        print('!!! 完整性检查未通过，只用于流程测试：' + '；'.join(problems) + ' !!!')
    else:
        print('完整性检查通过：每个学习器 1,848 行（44 单元 × 42），键无重复，AE/S2 成对，选样哈希与 G 一致')
    if 'status' in R:
        f = R[R.status == 'failed']
        print(f'技术失败（剔除）：{len(f)} 行' + ('' if f.empty else '：' + f.groupby('learner').size().to_dict().__repr__()))
    lr = R[(R.learner == 'lr') & (R.cache_hit == False)]  # noqa: E712  只数实际拟合
    if 'info_converged' in lr:
        print(f'LR 未收敛的拟合：{int((lr.info_converged == False).sum())}/{len(lr)}（记录，不剔除）')  # noqa: E712
    cs = [json.load(open(p_)) for p_ in glob.glob(os.path.join(OUT, 'lr_C', '*.json'))]
    if cs:
        print(f'LR 的 C*（{len(cs)} 个 国家×划分×特征）：{pd.Series([c["C"] for c in cs]).value_counts().sort_index().to_dict()}；'
              f'落在网格边界 {sum(c.get("boundary", False) for c in cs)} 个（照用，只作披露）')

    summ, vals = [], {}
    print('\n' + '=' * 20 + ' 各学习器的 θ_E1（AE 指导 − S2 指导；4 几何方法 × 2 特征 × k∈{50,200} × 4 单元；11 国 t 检验）' + '=' * 20)
    for L in ORDER:
        RL = R[R.learner == L]
        if RL.empty:
            continue
        for met in ('ba_ge30', 'oa'):
            v = A.country_values(A.deltas(RL, 'AE', 'S2', E1, met))
            s = A.t_summary(v)
            vals[(L, met)] = v
            summ.append(dict(learner=L, metric=met, what='theta_E1', tost_p=tost_p(s), verdict=A.verdict(s), **s))
            print(f'{NAME[L]:<14} {met:<8}', A.fmt(s), f'；TOST p={tost_p(s):.4f} → {A.verdict(s)}')

    print('\n' + '=' * 20 + ' 学习器稳健性判定（rev2 第 4 节，预先写死；ba_ge30）' + '=' * 20)
    S = pd.DataFrame(summ)
    sb = S[S.metric == 'ba_ge30'].set_index('learner')
    have = [L for L in ORDER if L in sb.index]
    is_eq = {L: sb.loc[L, 'verdict'].startswith('等价') for L in have}
    directional = [L for L in have if sb.loc[L, 'lo95'] > 0 or sb.loc[L, 'hi95'] < 0]
    print('各学习器的判定（冻结方案第 8 节）：' + '；'.join(f'{NAME[L]} {sb.loc[L, "verdict"]}' for L in have))
    if len(have) == 4 and all(is_eq.values()):
        print('总判定：等价不依赖学习器（四个学习器都判为"等价"：90% CI 在 ±0.3 内且 95% CI 含 0）')
    elif directional:
        print('总判定：至少一个学习器下有方向性差异——' + '；'.join(
            f'{NAME[L]} {sb.loc[L, "mean"]:+.3f}，95% CI [{sb.loc[L, "lo95"]:+.3f}, {sb.loc[L, "hi95"]:+.3f}]'
            f'（{sb.loc[L, "verdict"]}）' for L in directional))
    else:
        print(f'总判定：部分学习器下等价（{", ".join(NAME[L] for L in have if is_eq[L]) or "无"}），其余不确定（见上）')

    print('\n' + '=' * 20 + ' 次要（描述性，不校正）' + '=' * 20)
    for L in LEARNERS:
        for met in ('ba_ge30', 'oa'):
            if (L, met) in vals and ('lgb', met) in vals:
                d = (vals[(L, met)] - vals[('lgb', met)]).dropna()
                s = A.t_summary(d)
                summ.append(dict(learner=L, metric=met, what='theta_minus_lgb', **s))
                print(f'θ({NAME[L]}) − θ(LightGBM) {met:<8}', A.fmt(s))
    print('\n先验修正指标上的 θ_E1（只对 LR、MLP 与 LightGBM 解读；RF 概率未校准）：')
    for L in ('lgb', 'lr', 'mlp'):
        RL = R[R.learner == L]
        if RL.empty:
            continue
        for met in ('ba_P1', 'ba_P2', 'oa_P2'):
            s_ = A.t_summary(A.country_values(A.deltas(RL, 'AE', 'S2', E1, met)))
            print(f'  {NAME[L]:<14} {met:<6}', A.fmt(s_))
    print('\n敏感性子集（ba_ge30）：')
    for L in ORDER:
        if (L, 'ba_ge30') not in vals:
            continue
        v = vals[(L, 'ba_ge30')]
        parts = []
        for lab, drop in (('去检验2四国', A.CHECK2), ('去RO', ['RO']), ('去开发国', A.DEV)):
            s = A.t_summary(v.drop([c for c in drop if c in v.index]))
            parts.append(f'{lab} {s["mean"]:+.3f} [{s["lo90"]:+.3f}, {s["hi90"]:+.3f}]')
        print(f'  {NAME[L]:<14}' + '；'.join(parts) + '（90% CI）')

    print('\nSRS 学习曲线（11 国均值；k=0 为只用外国数据）：')
    lc = R[R.method.isin(['srs', 'foreign_only'])].groupby(['learner', 'clf', 'k', 'target'])[['ba_ge30', 'oa']].mean() \
        .groupby(['learner', 'clf', 'k']).mean().unstack('k')
    print(lc.round(2).to_string())

    print('\n各学习器下，几何方法相对 SRS（ba_ge30 与 OA；国家层面均值 [95% CI]，>0 国家数；R3 描述量）：')
    for L in ORDER:
        RL = R[R.learner == L]
        if RL.empty:
            continue
        parts = []
        for m in E1:
            for c in ('AE', 'S2'):
                p = []
                for met in ('ba_ge30', 'oa'):
                    d = A.vs_srs(RL, c, [m], met)
                    s = A.t_summary(A.country_values(d))
                    p.append(f'{met} {s["mean"]:+.2f} [{s["lo95"]:+.2f}, {s["hi95"]:+.2f}] {s["n_pos"]}/11，'
                             f'P(<SRS)={(d.d < 0).mean():.2f}')
                parts.append(f'    {m:<11}{c} 指导：' + '；'.join(p))
        print(f'  {NAME[L]}：\n' + '\n'.join(parts))

    S = pd.DataFrame(summ)
    S.to_csv(os.path.join(OUT, f'summary_ML{tag}.csv'), index=False)
    plot(S, lc, tag)


def plot(S, lc, tag):
    import matplotlib.pyplot as plt
    from plot_check2 import INK, INK2, MUTED, GRID, BLUE, ORANGE, style
    col = {'lgb': INK, 'lr': BLUE, 'mlp': ORANGE, 'rf': '#1baf7a'}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), gridspec_kw=dict(width_ratios=[1.1, 1.1, 1.3], wspace=0.35))
    for ax, met, title in ((axes[0], 'ba_ge30', 'θ_E1 · 平衡精度'), (axes[1], 'oa', 'θ_E1 · OA')):
        style(ax)
        ax.grid(axis='y', visible=False)
        ax.grid(axis='x', color=GRID, lw=0.6)
        ax.axvspan(-A.SESOI, A.SESOI, color='#eceae4', zorder=0)
        ax.axvline(0, color=INK2, lw=0.9, ls=(0, (4, 3)))
        sub = S[(S.what == 'theta_E1') & (S.metric == met)].set_index('learner')
        for i, L in enumerate([L for L in ORDER if L in sub.index]):
            s = sub.loc[L]
            ax.plot([s.lo95, s.hi95], [i, i], color=col[L], lw=1.2)
            ax.plot([s.lo90, s.hi90], [i, i], color=col[L], lw=3.2)
            ax.scatter(s['mean'], i, s=40, color=col[L], edgecolor='white', lw=0.8, zorder=3)
        ax.set_yticks(range(len(sub)), [NAME[L] for L in ORDER if L in sub.index])
        ax.invert_yaxis()
        ax.set_title(title, fontsize=10, color=INK, loc='left')
        ax.set_xlabel('AE 指导 − S2 指导（pp）')
    ax = axes[2]
    style(ax)
    for L in [L for L in ORDER if L in lc.index.get_level_values(0)]:
        for clf, ls in (('s2', '-'), ('ae', '--')):
            if (L, clf) in lc.index:
                y = lc.loc[(L, clf), 'ba_ge30']
                ax.plot(y.index, y.values, ls=ls, color=col[L], lw=1.5, marker='o', ms=3.5,
                        label=f'{NAME[L]}（{clf.upper()} 特征）')
    ax.set_xlabel('随机目标标签数 k')
    ax.set_ylabel('平衡精度 ba_ge30（%）')
    ax.set_title('随机选样学习曲线（11 国均值）', fontsize=10, color=INK, loc='left')
    ax.legend(fontsize=7, frameon=False, ncol=2, loc='lower right')
    fig.suptitle('多学习器稳健性分析（rev2，预先指定；G s0/s1 测试集已复用，非确认性）', fontsize=11.5, color=INK,
                 x=0.01, ha='left', y=1.04)
    fig.text(0.01, -0.06, '灰带 = 等价界 ±0.3 pp；粗线 = 90% CI，细线 = 95% CI（11 国 t 检验）。学习曲线实线 = S2 特征、虚线 = AE 特征。'
             '训练集、权重、测试集与 G 相同（外国 5 万 + k 个目标样本，目标权重 ×50，类别权重按外国频率）。',
             fontsize=7.5, color=MUTED, va='top')
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(d, f'fig_multilearner{tag}.{ext}'), dpi=200, bbox_inches='tight')


if __name__ == '__main__':
    main()
