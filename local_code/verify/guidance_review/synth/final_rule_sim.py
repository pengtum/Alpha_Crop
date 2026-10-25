"""汇总方案的判定规则运行特征（蒙特卡洛）。
国家 c 的主估计量 E1 = 3 个几何方法 × 2 分类器 × 2 档 k × J 单元的平均。
D_cmlbj = mu + s_l*gamma + u_c + w_cl + v_cm + e ;  u~N(0,.7tau^2), w~N(0,.3tau^2), v~N(0,tau_mc^2)
e: sigma=1.1，方法间相关 rho_m，分类器间 0.1，k 间 0.3（单元间独立）。
"""
import numpy as np, pandas as pd
from scipy import stats
N, NSIM = 11, 20000
SIG, RHO_M, RHO_L, RHO_K, TAU_MC = 1.1, 0.1, 0.1, 0.3, 0.3

def country_means(mu, gamma, tau, M, J, rng, n=N):
    L, K = 2, 2
    # 单元内误差协方差：维度 (m,l,b)
    idx = [(m, l, b) for m in range(M) for l in range(L) for b in range(K)]
    S = np.array([[(1 if a[0]==c[0] else RHO_M)*(1 if a[1]==c[1] else RHO_L)*(1 if a[2]==c[2] else RHO_K)
                   for c in idx] for a in idx]) * SIG**2
    e = rng.multivariate_normal(np.zeros(len(idx)), S, size=(NSIM, n, J)).mean(2).reshape(NSIM, n, M, L, K)
    u = rng.normal(0, np.sqrt(.7)*tau, (NSIM, n, 1, 1, 1))
    w = rng.normal(0, np.sqrt(.3)*tau, (NSIM, n, 1, L, 1))
    v = rng.normal(0, TAU_MC, (NSIM, n, M, 1, 1))
    s = np.array([1, -1]).reshape(1, 1, 1, L, 1)
    D = mu + s*gamma + u + w + v + e
    # 单元层面的国家内方差（用于 PI 的 tau 估计）：已知噪声部分
    within = (S.sum()/len(idx)**2)/J
    return D.mean((2, 3, 4)), within

def rules(x, within, n=N):
    m = x.mean(1); sd = x.std(1, ddof=1); se = sd/np.sqrt(n)
    t95, t90 = stats.t.ppf(.975, n-1), stats.t.ppf(.95, n-1)
    lo95, hi95, lo90, hi90 = m-t95*se, m+t95*se, m-t90*se, m+t90*se
    tau2 = np.maximum(0, sd**2 - within)
    tpi = stats.t.ppf(.90, n-2)
    pi_lo = m - tpi*np.sqrt(tau2 + se**2)
    better = lo95 > 0; worse = hi95 < 0
    eq3 = (lo90 > -.3) & (hi90 < .3); eq5 = (lo90 > -.5) & (hi90 < .5)
    return dict(P_better=better.mean(), P_better_and_PIok=(better & (pi_lo > -.5)).mean(),
                P_better_but_eq3=(better & eq3).mean(), P_eq3=eq3.mean(), P_eq5=eq5.mean(),
                P_worse=worse.mean(), P_inconcl_eq3=(~better & ~worse & ~eq3).mean(),
                P_inconcl_eq5=(~better & ~worse & ~eq5).mean(),
                sign9=((x > 0).sum(1) >= 9).mean(), mean_halfwidth95=(t95*se).mean())

if __name__ == "__main__":
    rows = []
    rng = np.random.default_rng(7)
    for tau in [0.15, 0.3, 0.5]:
        for mu, g in [(0, 0), (0, .3), (.1, 0), (.24, .29), (.3, 0), (.3, .3), (.5, 0), (-.3, 0)]:
            x, within = country_means(mu, g, tau, 3, 4, rng)
            rows.append(dict(tau=tau, mu=mu, gamma=g, **rules(x, within)))
    res = pd.DataFrame(rows)
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
    print(res.round(3).to_string(index=False))
    res.to_csv('/Users/perng_mini/AlphaCrops_local/verify/guidance_review/synth/final_rule_sim.csv', index=False)
    # 去掉 RO（n=10）与只用 7 国的精度
    for n in [10, 7]:
        x, within = country_means(.3, 0, .3, 3, 4, rng, n=n)
        r = rules(x, within, n=n); print('n=', n, 'mu=.3 tau=.3', {k: round(v, 3) for k, v in r.items()})
