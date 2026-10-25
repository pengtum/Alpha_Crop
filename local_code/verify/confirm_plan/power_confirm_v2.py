"""确认运行方案草稿 v2 的功效粗算（只读已有输出；不训练；不读任何新格子；C 级、复用测试集，只作功效参考）。

输入（全部为 preconf 已有输出，与 v1 的 power_confirm.py 相同来源）：
  - verify/preconf/label/label_rows.csv：标签线各臂单元级 FA、τ=0 / τ=0.5 的 OA 与 BA（LightGBM：Z0/PK/HD 在 P_T 上，RT/RTPK 在 i300 上）
  - verify/preconf/label/label_ae7_power.csv：H_R1 各臂×学习器的 rep0 国家层 SD 与回溯均值
新增内容（相对 v1）：
  1. 主假设：SD 取 80% 置信上限的列、真值为回溯 60% / 50% 的列、两种 α；97.5% CI 半宽；三种结局概率；两条主假设同时通过的概率（近似独立）。
  2. S-0（标签分解）：A_200 > 0、R* < δ0（δ0 = 1.0 / 1.5）的非优效上界功效。SD 用 P_T 非对称口径（代理；raw 口径无回溯值）。
  3. S-1 改为 3 项、S-2 加 LR 为 3 项后的 Holm 首步 / 末步功效。
  4. S-4（读数纠偏）与 S-5（τ=0.5 可部署点 AE−S2）的回溯 SD 与功效（P_T 口径代理）。
  5. 合并 Holm 摘要闸门（m = 16 项）的最坏步 α 下的功效（下界）。
国家值：rep0 单元的单元差先按特征平均到国家，再取两特征平均（both）；9 国 = 去 RO、PT。mean4 = 4 单元回溯均值。
"""
import numpy as np
import pandas as pd
from scipy import optimize, stats

LAB = '/Users/perng_mini/AlphaCrops_local/verify/preconf/label/'
pd.set_option('display.width', 320)
pd.set_option('display.max_columns', 40)


def mde(sd, n, alpha, power=0.8):
    df = n - 1
    tc = stats.t.ppf(1 - alpha, df)
    return optimize.brentq(lambda z: stats.nct.sf(tc, df, z / (sd / np.sqrt(n))) - power, 1e-6, 50 * sd)


def pw(delta, sd, n, alpha):
    """单侧 H1: θ > 0 的功效。"""
    df = n - 1
    return stats.nct.sf(stats.t.ppf(1 - alpha, df), df, delta / (sd / np.sqrt(n)))


def p_lower_ge(c, delta, sd, n, conf):
    """双侧 conf 置信区间下界 ≥ c 的概率。"""
    df = n - 1
    tc = stats.t.ppf(1 - (1 - conf) / 2, df)
    return stats.nct.sf(tc, df, (delta - c) / (sd / np.sqrt(n)))


def p_upper_lt(c, delta, sd, n, conf):
    df = n - 1
    tc = stats.t.ppf(1 - (1 - conf) / 2, df)
    return stats.nct.sf(tc, df, (c - delta) / (sd / np.sqrt(n)))


def ucl80(n):
    """SD 的 80% 单侧置信上限相对样本 SD 的倍数：sqrt(df / chi2_{0.2, df})。"""
    df = n - 1
    return np.sqrt(df / stats.chi2.ppf(0.2, df))


def halfw(sd, n, conf):
    return stats.t.ppf(1 - (1 - conf) / 2, n - 1) * sd / np.sqrt(n)


# ---------------------------------------------------------------- 1. 主假设
print('== 1. 主假设（单侧；Holm 首步 α 0.0125，末步 0.025）==')
print(f'SD 80% 置信上限倍数：df=8 {ucl80(9):.3f}，df=10 {ucl80(11):.3f}')
PRIM = [('H_L ΔFA(RT200) 9 国', 0.942, 9, 2.102), ('H_R1 D(Z0) 11 国', 3.729, 11, 5.250),
        ('H_R1 D(Z0) 9 国', 3.124, 9, 6.012)]
rows = []
for name, sd, n, m in PRIM:
    for frac in (1.0, 0.6, 0.5):
        for sdlab, s in (('SD', sd), ('SD×1.25', 1.25 * sd), ('SD_UCL80', ucl80(n) * sd)):
            rows.append(dict(q=name, true=f'{frac:.1f}×均值={frac * m:.2f}', sd=sdlab, sd_val=round(s, 3),
                             pw_a025=round(pw(frac * m, s, n, 0.025), 3), pw_a0125=round(pw(frac * m, s, n, 0.0125), 3)))
print(pd.DataFrame(rows).to_string(index=False))

print('\n-- 97.5% / 95% CI 半宽与"下界 ≥ 1"的概率（SD 取回溯值）--')
rows = []
for name, sd, n, m in PRIM:
    for frac in (1.0, 0.5):
        d = frac * m
        rows.append(dict(q=name, true=round(d, 2), hw975=round(halfw(sd, n, 0.975), 2), hw95=round(halfw(sd, n, 0.95), 2),
                         P_L975_ge1=round(p_lower_ge(1, d, sd, n, 0.975), 3), P_L95_ge1=round(p_lower_ge(1, d, sd, n, 0.95), 3)))
print(pd.DataFrame(rows).to_string(index=False))

print('\n-- 三种结局的预期概率（按 97.5% CI；"通过"近似为 97.5% 下界 > 0，即 Holm 首步）--')
rows = []
for name, sd, n, m in PRIM:
    for frac in (1.0, 0.5):
        d = frac * m
        p_ge1 = p_lower_ge(1, d, sd, n, 0.975)
        p_pos = p_lower_ge(0, d, sd, n, 0.975)
        p_ult1 = p_upper_lt(1, d, sd, n, 0.975)
        rows.append(dict(q=name, true=round(d, 2), P_pass_ge1=round(p_ge1, 3), P_pass_lt1=round(p_pos - p_ge1, 3),
                         P_notpass_upper_lt1=round(p_ult1, 3), P_other=round(1 - p_pos - p_ult1, 3)))
print(pd.DataFrame(rows).to_string(index=False))

print('\n-- 两条主假设同时通过（Holm；近似独立；H_L 9 国，H_R1 11 国）--')
for frac in (1.0, 0.6, 0.5):
    L = PRIM[0]; R = PRIM[1]
    a1L, a2L = pw(frac * L[3], L[1], L[2], 0.0125), pw(frac * L[3], L[1], L[2], 0.025)
    a1R, a2R = pw(frac * R[3], R[1], R[2], 0.0125), pw(frac * R[3], R[1], R[2], 0.025)
    both = a1L * a2R + a1R * a2L - a1L * a1R
    print(f'真值 = {frac:.1f}×回溯均值：P(两者都通过) ≈ {both:.3f}；P(H_L 通过) ≈ {a1L + (a2L - a1L) * a1R:.3f}；'
          f'P(H_R1 通过) ≈ {a1R + (a2R - a1R) * a1L:.3f}')

# ---------------------------------------------------------------- 2. S-0
print('\n== 2. S-0 标签分解（LightGBM，9 国；两槽位 Holm：α 0.0125 / 0.025）==')
print('S-0a A_200 > 0（P_T 口径代理，SD 1.373，均值 2.087）：'
      f'MDE80 {mde(1.373, 9, 0.0125):.2f} / {mde(1.373, 9, 0.025):.2f}；功效（回溯均值）{pw(2.087, 1.373, 9, 0.0125):.3f} / '
      f'{pw(2.087, 1.373, 9, 0.025):.3f}；真值一半 {pw(1.0435, 1.373, 9, 0.0125):.3f} / {pw(1.0435, 1.373, 9, 0.025):.3f}')
rows = []
for d0 in (1.0, 1.5):
    for true in (0.5, 0.0, -0.469):
        for sdlab, s in (('SD 1.304', 1.304), ('×1.25', 1.25 * 1.304), ('UCL80', ucl80(9) * 1.304)):
            rows.append(dict(bound=d0, true=true, sd=sdlab, pw_a0125=round(pw(d0 - true, s, 9, 0.0125), 3),
                             pw_a025=round(pw(d0 - true, s, 9, 0.025), 3)))
print('S-0b H1: R* < δ0（非优效上界）的功效（SD 为 P_T 非对称口径 R*_200 的 rep0 SD；raw 口径无回溯值）')
print(pd.DataFrame(rows).to_string(index=False))
# 闭检验（Bonferroni 局部检验；H0b 与 H0c 互斥）：H0b 在 p_a ≤ α/2 时用 α，否则用 α/2（近似独立）
pa1 = pw(2.087, 1.373, 9, 0.0125)
for d0 in (1.0, 1.5):
    for true in (0.0, -0.469):
        pb_a, pb_h = pw(d0 - true, 1.304, 9, 0.025), pw(d0 - true, 1.304, 9, 0.0125)
        print(f'S-0b 闭检验功效（δ0={d0}，真值 {true}，S-0a 真值 = 回溯均值）：{pa1 * pb_a + (1 - pa1) * pb_h:.3f}')
print(f'S-0c H1: R* > 0：真值 +0.5 时 {pw(0.5, 1.304, 9, 0.0125):.3f} / {pw(0.5, 1.304, 9, 0.025):.3f}；'
      f'真值 +1.0 时 {pw(1.0, 1.304, 9, 0.0125):.3f} / {pw(1.0, 1.304, 9, 0.025):.3f}')

# ---------------------------------------------------------------- 3. S-1 / S-2 / S-3
print('\n== 3. S-1（3 项，11 国）、S-2（3 项，9 国）、S-3（2 项，9 国）：Holm 首步 / 末步 ==')
FAM = [('S-1', 'D(RT200) LGB', 4.177, 11, 4.686, 3), ('S-1', 'D(K0) MLP', 3.892, 11, 4.282, 3),
       ('S-1', 'D(K0) RF', 3.882, 11, 7.352, 3),
       ('S-2', 'ΔFA(RT200) RF', 1.571, 9, 2.308, 3), ('S-2', 'ΔFA(RT200) MLP', 1.414, 9, 2.564, 3),
       ('S-2', 'ΔFA(RT200) LR', 2.670, 9, 4.126, 3),
       ('S-3', 'R*_200 RF', 0.527, 9, 0.971, 2), ('S-3', 'R*_200 MLP', 1.600, 9, 1.275, 2)]
rows = []
for fam, q, sd, n, m, mm in FAM:
    a1 = 0.025 / mm
    rows.append(dict(fam=fam, q=q, n=n, sd=sd, mean=m, a_first=round(a1, 5), mde_first=round(mde(sd, n, a1), 2),
                     mde_last=round(mde(sd, n, 0.025), 2), pw_first_mean=round(pw(m, sd, n, a1), 3),
                     pw_last_mean=round(pw(m, sd, n, 0.025), 3), pw_first_half=round(pw(m / 2, sd, n, a1), 3),
                     pw_gate_m16=round(pw(m, sd, n, 0.025 / 16), 3)))
print(pd.DataFrame(rows).to_string(index=False))

# ---------------------------------------------------------------- 4. S-4 / S-5（从 label_rows 重算，P_T 口径代理）
d = pd.read_csv(LAB + 'label_rows.csv')
d = d[(d.width == 4) & (d.learner == 'lgb')]


def country(v_unit, reps, drop):
    v = v_unit.rename('v').reset_index()
    v = v[v.rep.isin(reps)]
    c = v.groupby(['target', 'clf'])['v'].mean().unstack('clf').mean(axis=1)
    return c[~c.index.isin(drop)]


def unit(col, arm):
    return d[d.arm == arm].set_index(['target', 'split', 'rep', 'clf'])[col]


s4a = (unit('FA', 'RT200') - unit('FA', 'Z0')) - (unit('ba_t0', 'RT200') - unit('ba_t0', 'Z0'))
s4b = (unit('ba_t0', 'HD200') - unit('ba_t0', 'RT200')) - (unit('FA', 'HD200') - unit('FA', 'RT200'))
tau0_rt = unit('oa_t0', 'RT200') - unit('oa_t0', 'Z0')
tau0_hd = unit('oa_t0', 'HD200') - unit('oa_t0', 'RT200')
print('\n== 4. S-4 读数纠偏（LightGBM，both；P_T 口径代理：Z0/HD 在 P_T，RT 在 i300）==')
rows = []
for name, v in (('S-4a ΔFA(RT200) − ΔBA_τ0(RT200)', s4a), ('S-4b [BA_τ0(HD−RT)] − [FA(HD−RT)]', s4b),
                ('（同报）ΔOA_τ0(RT200 − Z0)', tau0_rt), ('（同报）ΔOA_τ0(HD200 − RT200)', tau0_hd)):
    for cs, drop in (('9', {'RO', 'PT'}), ('11', set())):
        v4, v0 = country(v, (0, 1), drop), country(v, (0,), drop)
        n, sd, m = len(v0), v0.std(ddof=1), v4.mean()
        h = stats.t.ppf(0.975, n - 1) * v4.std(ddof=1) / np.sqrt(n)
        rows.append(dict(q=name, set=cs, n=n, mean4=round(m, 2), ci95=f'[{m - h:+.2f}, {m + h:+.2f}]',
                         npos=f'{int((v4 > 0).sum())}/{n}', sd_r0=round(sd, 3), mde_a0125=round(mde(sd, n, 0.0125), 2),
                         pw_a0125_mean=round(pw(m, sd, n, 0.0125), 3), pw_a0125_half=round(pw(m / 2, sd, n, 0.0125), 3),
                         pw_gate_m16=round(pw(m, sd, n, 0.025 / 16), 3)))
print(pd.DataFrame(rows).to_string(index=False))

print('\n== 5. S-5 可部署点 τ = 0.5：Z0 的 AE − S2（LightGBM；P_T 口径代理）==')
z = d[d.arm == 'Z0'].set_index(['target', 'split', 'rep', 'clf'])
rows = []
for met in ('ba_t05', 'oa_t05'):
    u = z[met].unstack('clf')
    diff = (u['ae'] - u['s2']).rename(0).reset_index()
    for cs, drop in (('11', set()), ('9', {'RO', 'PT'})):
        g4 = diff.groupby('target')[0].mean()
        g0 = diff[diff.rep == 0].groupby('target')[0].mean()
        g4, g0 = g4[~g4.index.isin(drop)], g0[~g0.index.isin(drop)]
        n, sd, m = len(g0), g0.std(ddof=1), g4.mean()
        h = stats.t.ppf(0.975, n - 1) * g4.std(ddof=1) / np.sqrt(n)
        rows.append(dict(met=met, set=cs, n=n, mean4=round(m, 2), ci95=f'[{m - h:+.2f}, {m + h:+.2f}]',
                         npos=f'{int((g4 > 0).sum())}/{n}', sd_r0=round(sd, 3), pw_a025_mean=round(pw(m, sd, n, 0.025), 3),
                         pw_a025_half=round(pw(m / 2, sd, n, 0.025), 3), pw_gate_m16=round(pw(m, sd, n, 0.025 / 16), 3)))
S5 = pd.DataFrame(rows)
print(S5.to_string(index=False))
r11 = S5[S5.set == '11']
pb, po = r11.pw_a025_mean.values
print(f'IUT（BA 与 OA 都拒绝，各 α 0.025，11 国）功效：上界 min = {min(pb, po):.3f}，独立时 = {pb * po:.3f}（正相关时介于两者之间）')
pbh, poh = r11.pw_a025_half.values
print(f'真值一半：上界 {min(pbh, poh):.3f}，独立 {pbh * poh:.3f}')

print('\n== 6. 合并 Holm 摘要闸门：主假设在 m = 16 项最坏步 α = 0.025/16 下的功效（下界）==')
for name, sd, n, m in PRIM[:2]:
    print(f'{name}: 回溯均值 {pw(m, sd, n, 0.025 / 16):.3f}，真值一半 {pw(m / 2, sd, n, 0.025 / 16):.3f}')
