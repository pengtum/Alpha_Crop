"""G 实验功效与规模计算（国家随机效应模型）。

每国家族平均 Δ 的方差：
  V = tau^2 + sigma_e^2 * gM * gK * gC / J
  gM = rho_m + (1 - rho_m) / M        （M 个方法，方法间噪声相关 rho_m）
  gK = (1 + (K - 1) rho_k) / K        （K 档预算合并，预算间噪声相关 rho_k）
  gC = (1 + (C - 1) rho_l) / C        （C 种分类器合并，分类器间噪声相关 rho_l）
检验：n 个国家均值的单样本 t 检验，df = n - 1，双侧 α = 0.05。
"""
import itertools

import numpy as np
import pandas as pd
from scipy import stats

RHO_K, RHO_L = 0.3, 0.1


def V_country(tau, sig, M, J, K, C, rho_m):
    gM = rho_m + (1 - rho_m) / M
    gK = (1 + (K - 1) * RHO_K) / K
    gC = (1 + (C - 1) * RHO_L) / C
    return tau ** 2 + sig ** 2 * gM * gK * gC / J


def power_t(mu, V, n, alpha=0.05):
    df = n - 1
    tc = stats.t.ppf(1 - alpha / 2, df)
    nc = mu / np.sqrt(V / n)
    return 1 - stats.nct.cdf(tc, df, nc) + stats.nct.cdf(-tc, df, nc)


def n_needed(mu, V, target=0.8, nmax=200):
    for n in range(3, nmax + 1):
        if power_t(mu, V, n) >= target:
            return n
    return np.inf


def tost_power(mu, V, n, delta=0.3, nsim=200_000, seed=0):
    rng = np.random.default_rng(seed)
    df = n - 1
    xb = rng.normal(mu, np.sqrt(V / n), nsim)
    s = np.sqrt(V * rng.chisquare(df, nsim) / df)
    h = stats.t.ppf(0.95, df) * s / np.sqrt(n)
    return float(np.mean((xb - h > -delta) & (xb + h < delta)))


def mde(V, n, target=0.8):
    lo, hi = 0.0, 5.0
    for _ in range(60):
        m = (lo + hi) / 2
        lo, hi = (m, hi) if power_t(m, V, n) < target else (lo, m)
    return hi


if __name__ == '__main__':
    pd.set_option('display.width', 220)
    SIG = [0.9, 1.1, 1.35]
    TAU = [0.15, 0.3, 0.5]
    RHO = [0.0, 0.1, 0.25]
    # ---------------- 1. 草稿设计（M=12, J=4, K=2），分类器单独 (C=1) 与平均 (C=2)
    print('=== 1. 草稿设计 n=11, M=12, J=4, K=2：检测功效 ===')
    rows = []
    for sig, tau, rho, C in itertools.product(SIG, TAU, RHO, [1, 2]):
        V = V_country(tau, sig, 12, 4, 2, C, rho)
        rows.append(dict(sigma_e=sig, tau=tau, rho_m=rho, C=C, sd_country=np.sqrt(V),
                         se_mean=np.sqrt(V / 11), pow03=power_t(0.3, V, 11), pow05=power_t(0.5, V, 11),
                         mde80=mde(V, 11), n03=n_needed(0.3, V), n05=n_needed(0.5, V),
                         tost_mu0=tost_power(0.0, V, 11), tost_mu01=tost_power(0.1, V, 11),
                         tost_mu02=tost_power(0.2, V, 11)))
    t1 = pd.DataFrame(rows)
    print(t1[(t1.sigma_e == 1.1)].round(3).to_string(index=False))
    print('\n(sigma_e 敏感性，rho_m=0.1)')
    print(t1[(t1.rho_m == 0.1)].round(3).to_string(index=False))
    t1.to_csv('power_draft.csv', index=False)

    # ---------------- 2. n=11 时每国所需单元数 J（功效 0.8）
    print('\n=== 2. n=11 时达到功效 0.8 所需的每国单元数 J（M=12, K=2, sigma_e=1.1）===')
    out = []
    for tau, rho, C, mu in itertools.product(TAU, RHO, [1, 2], [0.3, 0.5]):
        J_need = next((J for J in range(1, 101) if power_t(mu, V_country(tau, 1.1, 12, J, 2, C, rho), 11) >= 0.8),
                      np.inf)
        lim = power_t(mu, tau ** 2, 11)  # J → ∞ 的极限功效
        out.append(dict(tau=tau, rho_m=rho, C=C, mu=mu, J_needed=J_need, power_limit_Jinf=round(lim, 3)))
    print(pd.DataFrame(out).pivot_table(index=['tau', 'rho_m'], columns=['mu', 'C'], values='J_needed'))
    print(pd.DataFrame(out).drop_duplicates(['tau', 'mu']).pivot_table(index='tau', columns='mu',
                                                                      values='power_limit_Jinf'))
    # 所需国家数（J=4, M=12, K=2）
    print('\n=== 2b. 所需国家数（J=4, M=12, K=2, sigma_e=1.1）===')
    out = []
    for tau, rho, C in itertools.product(TAU, RHO, [1, 2]):
        V = V_country(tau, 1.1, 12, 4, 2, C, rho)
        out.append(dict(tau=tau, rho_m=rho, C=C, n_mu03=n_needed(0.3, V), n_mu05=n_needed(0.5, V)))
    print(pd.DataFrame(out).pivot_table(index=['tau', 'rho_m'], columns='C', values=['n_mu03', 'n_mu05']))
    # TOST 所需国家数（真值 0 与 0.1）
    print('\n=== 2c. TOST(±0.3, 90% CI) 功效 0.8 所需国家数（J=4, M=12, K=2, sigma_e=1.1）===')
    out = []
    for tau, rho, C in itertools.product(TAU, [0.1], [1, 2]):
        V = V_country(tau, 1.1, 12, 4, 2, C, rho)
        res = {}
        for mu in [0.0, 0.1, 0.2]:
            res[f'n_mu{mu}'] = next((n for n in range(4, 61) if tost_power(mu, V, n, nsim=40_000) >= 0.8), '>60')
        out.append(dict(tau=tau, C=C, **res))
    print(pd.DataFrame(out).to_string(index=False))
