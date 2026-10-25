"""确认运行方案草稿 v1 的功效粗算（只读已有输出；不训练；C 级、复用测试集，只作功效参考）。
输入：
  - verify/preconf/label/label_rows.csv（W4，新网格；标签线各臂的单元 FA）
  - verify/preconf/label/label_ae7_power.csv（AE7：H_R1 各臂×学习器的 rep0 国家层 SD）
国家值：rep0 单元（划分 0、1）上的单元差先按特征平均到国家，再取两特征平均（both）；9 国 = 去 RO、PT。
mean4 = 用 4 个单元（含 rep1）的回溯均值与 95% t 区间，与 PRECONF/B2 报告的数字对账。
MDE80：单样本 t，非中心 t 精确解（单侧 α；two-sided 列为双侧 0.05）。功效：给定真值时的拒绝概率。"""
import numpy as np
import pandas as pd
from scipy import optimize, stats

LAB = '/Users/perng_mini/AlphaCrops_local/verify/preconf/label/'
d = pd.read_csv(LAB + 'label_rows.csv')
d = d[d.width == 4]
W = d.pivot_table(index=['learner', 'target', 'split', 'rep', 'clf'], columns='arm', values='FA')


def cvals(learner, a, b, reps):
    X = W.loc[learner]
    x = (X[a] - X[b]).reset_index()
    x = x[x.rep.isin(reps)]
    return x.groupby(['target', 'clf'])[0].mean().unstack('clf').mean(axis=1)


def mde(sd, n, alpha, sided=1, power=0.8):
    df = n - 1
    tc = stats.t.ppf(1 - alpha / sided, df)
    return optimize.brentq(lambda z: stats.nct.sf(tc, df, z / (sd / np.sqrt(n))) - power, 1e-6, 50 * sd)


def pw(delta, sd, n, alpha):
    df = n - 1
    tc = stats.t.ppf(1 - alpha, df)
    return stats.nct.sf(tc, df, delta / (sd / np.sqrt(n)))


def contrast(L, a, b, reps, L2=None):
    v = cvals(L, a, b, reps)
    if L2:
        v = v - cvals(L2, a, b, reps)
    return v


Q = [('lgb', 'RT200', 'Z0', None, 'ΔFA(RT200) lgb'), ('lgb', 'RT50', 'Z0', None, 'ΔFA(RT50) lgb'),
     ('lgb', 'RT400', 'Z0', None, 'ΔFA(RT400) lgb'), ('lgb', 'PK200', 'Z0', None, 'A_200 lgb'),
     ('lgb', 'RT200', 'PK200', None, 'R_200 lgb'), ('lgb', 'RTPK200', 'PK200', None, 'R*_200 lgb'),
     ('lgb', 'HD200', 'Z0', None, 'ΔFA(HD200) lgb'), ('lgb', 'RT200', 'HD200', None, 'RT200−HD200 lgb'),
     ('rf', 'RT200', 'Z0', None, 'ΔFA(RT200) rf'), ('mlp', 'RT200', 'Z0', None, 'ΔFA(RT200) mlp'),
     ('lr', 'RT200', 'Z0', None, 'ΔFA(RT200) lr'),
     ('rf', 'RTPK200', 'PK200', None, 'R*_200 rf'), ('mlp', 'RTPK200', 'PK200', None, 'R*_200 mlp'),
     ('lr', 'RTPK200', 'PK200', None, 'R*_200 lr'),
     ('rf', 'RTPK200', 'PK200', 'lgb', 'R*rf−R*lgb'), ('mlp', 'RTPK200', 'PK200', 'lgb', 'R*mlp−R*lgb')]
rows = []
for L, a, b, L2, name in Q:
    for cs, drop in (('9', {'RO', 'PT'}), ('11', set())):
        v4 = contrast(L, a, b, (0, 1), L2)
        v4 = v4[~v4.index.isin(drop)]
        v0 = contrast(L, a, b, (0,), L2)
        v0 = v0[~v0.index.isin(drop)]
        n, sd, m4 = len(v0), v0.std(ddof=1), v4.mean()
        h = stats.t.ppf(0.975, len(v4) - 1) * v4.std(ddof=1) / np.sqrt(len(v4))
        rows.append(dict(q=name, set=cs, n=n, mean4=round(m4, 3), ci95_4=f'[{m4 - h:+.2f}, {m4 + h:+.2f}]',
                         npos4=f'{int((v4 > 0).sum())}/{len(v4)}', sd_r0=round(sd, 3),
                         mde80_a025=round(mde(sd, n, 0.025), 3), mde80_a0125=round(mde(sd, n, 0.0125), 3),
                         mde80_two_sided05=round(mde(sd, n, 0.05, sided=2), 3),
                         pw_a025_at_mean=round(pw(m4, sd, n, 0.025), 3), pw_a025_at_half=round(pw(m4 / 2, sd, n, 0.025), 3),
                         pw_a0125_at_mean=round(pw(m4, sd, n, 0.0125), 3), pw_a0125_at_half=round(pw(m4 / 2, sd, n, 0.0125), 3),
                         pw_a025_half_sdx125=round(pw(m4 / 2, 1.25 * sd, n, 0.025), 3)))
R = pd.DataFrame(rows)
pd.set_option('display.width', 300)
print('== 标签线（label_rows.csv，W4，both）==')
print(R.to_string(index=False))

print('\n== H_R1 / S-1（label_ae7_power.csv 的 sd_r0 与 mean4）==')
P = pd.read_csv(LAB + 'label_ae7_power.csv')
P = P[P.hyp == 'H_R1']
out = []
for _, r in P.iterrows():
    sd, n, m = r.sd_r0, int(r.n), r.mean4
    out.append(dict(q=r.quantity, learner=r.learner, set=r.set, sd_r0=round(sd, 2), mean4=round(m, 2),
                    mde80_a025=round(mde(sd, n, .025), 2), mde80_a0125=round(mde(sd, n, .0125), 2),
                    mde80_a005=round(mde(sd, n, .005), 2),
                    pw_a025_mean=round(pw(m, sd, n, .025), 3), pw_a0125_mean=round(pw(m, sd, n, .0125), 3),
                    pw_a005_mean=round(pw(m, sd, n, .005), 3), pw_a025_half=round(pw(m / 2, sd, n, .025), 3),
                    pw_a0125_half=round(pw(m / 2, sd, n, .0125), 3), pw_a0125_0p6=round(pw(0.6 * m, sd, n, .0125), 3),
                    pw_a0125_half_sdx125=round(pw(m / 2, 1.25 * sd, n, .0125), 3)))
print(pd.DataFrame(out).to_string(index=False))

print('\n== H_R2 估计型：交互量 90% CI 预期半宽（label_ae7_power.csv 的 H_R2 行 sd_r0）==')
H = pd.read_csv(LAB + 'label_ae7_power.csv')
H = H[H.hyp == 'H_R2']
for _, r in H.iterrows():
    n = int(r.n)
    print(f'{r.quantity:14s} {r.set:>3} 国 sd_r0 {r.sd_r0:.2f} 90% CI 半宽 {stats.t.ppf(.95, n - 1) * r.sd_r0 / np.sqrt(n):.2f}'
          f'  TOST±1 功效(真值0) {r.tost_power_true0:.3f}  80% 功效所需界值 {r.margin80_true0:.2f}')
