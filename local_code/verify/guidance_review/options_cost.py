import numpy as np, pandas as pd
from power import V_country, power_t, mde, tost_power
SEC, PAR = 57, 4
def hours(n): return n * SEC / PAR / 3600
S0 = 11 * 2 * 2 * 6  # 阶段 0：11 国 × 2 划分 × 2 特征 × (全量 1 + 分组留出 5)
opts = {
 'D0 草稿':      dict(M=12, J=4, C=2, main=12*2*2*2*44, srs_or=3*2*2*44, xy=6*2*2*44, cross=4*2*2*2*44, s0=S0),
 'A 推荐':        dict(M=6,  J=4, C=2, main=6*2*2*2*44,  srs_or=2*2*44 + 2*44, xy=2*2*2*22, cross=2*2*2*1*22, s0=S0),
 'B 全方法少单元': dict(M=12, J=2, C=2, main=12*2*2*2*22, srs_or=2*2*22 + 2*22, xy=3*2*2*11, cross=2*2*2*2*11, s0=S0),
 'C 仅 S2 分类器': dict(M=12, J=4, C=1, main=12*2*1*2*44, srs_or=2*44 + 2*44,  xy=3*1*2*44, cross=2*2*1*2*22, s0=S0),
}
rows = []
for k, o in opts.items():
    tot = o['main'] + o['srs_or'] + o['xy'] + o['cross'] + o['s0']
    r = dict(方案=k, 主实验=o['main'], SRS与神谕=o['srs_or'], XY=o['xy'], 通路交叉=o['cross'], 阶段0=o['s0'],
             合计=tot, 小时_4进程=round(hours(tot), 1))
    for tau in [0.15, 0.3, 0.5]:
        V = V_country(tau, 1.1, o['M'], o['J'], 2, o['C'], 0.1)
        r[f'MDE80_tau{tau}'] = round(mde(V, 11), 3)
        r[f'pow03_tau{tau}'] = round(power_t(0.3, V, 11), 2)
        r[f'TOST0_tau{tau}'] = round(tost_power(0.0, V, 11, nsim=100_000), 2)
    rows.append(r)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
print(pd.DataFrame(rows).to_string(index=False))
# 单方法估计精度（逐方法 95% CI 半宽，n=11，分类器平均，预算合并；tau_m = 0.3）
print('\n逐方法 95% CI 半宽（tau_m=0.3, sigma_e=1.1）:')
from scipy import stats
for k, o in opts.items():
    Vm = 0.3**2 + 1.1**2 * ((1 + 0.3) / 2) * ((1 + (o['C'] - 1) * 0.1) / o['C']) / o['J']
    hw = stats.t.ppf(0.975, 10) * np.sqrt(Vm / 11)
    p_single = power_t(0.3, Vm, 11); p_holm = power_t(0.3, Vm, 11, alpha=0.05 / o['M'])
    print(f'  {k}: 半宽 {hw:.2f} pp；单方法在 0.3 pp 的功效 {p_single:.2f}，按 Holm 最严阈值 {p_holm:.2f}')
