"""判定规则的运行特征 + 削减方案的算力与功效（蒙特卡洛）。

真值模型（国家 c、分类器 l∈{ae,s2}、预算 b∈{50,200}）：
  D_clb = mu + s_l*gamma + u_c + w_cl + e_clb
  s_ae=+1, s_s2=-1（gamma = "指导–分类器匹配"半差；检验 2 估计 ≈ +0.29）
  u_c ~ N(0, 0.7 tau^2)，w_cl ~ N(0, 0.3 tau^2)
  e_clb ~ N(0, sigma^2 * gM / J)，分类器间相关 rho_l，预算间相关 rho_k
"""
import numpy as np
import pandas as pd
from scipy import stats

RHO_K, RHO_L = 0.3, 0.1
N_C = 11
NSIM = 20_000


def simulate(mu, gamma, tau, sig, M, J, rho_m, clfs=('ae', 's2'), n=N_C, nsim=NSIM, seed=1):
    rng = np.random.default_rng(seed)
    gM = rho_m + (1 - rho_m) / M
    L = len(clfs)
    # 误差协方差（l, b）
    idx = [(l, b) for l in range(L) for b in range(2)]
    S = np.zeros((len(idx), len(idx)))
    for i, (l1, b1) in enumerate(idx):
        for j, (l2, b2) in enumerate(idx):
            S[i, j] = (1 if l1 == l2 else RHO_L) * (1 if b1 == b2 else RHO_K)
    S *= sig ** 2 * gM / J
    e = rng.multivariate_normal(np.zeros(len(idx)), S, size=(nsim, n)).reshape(nsim, n, L, 2)
    u = rng.normal(0, np.sqrt(0.7) * tau, (nsim, n, 1, 1))
    w = rng.normal(0, np.sqrt(0.3) * tau, (nsim, n, L, 1))
    s = np.array([1 if c == 'ae' else -1 for c in clfs]).reshape(1, 1, L, 1)
    return mu + s * gamma + u + w + e  # (nsim, n, L, 2)


def tstats(x):
    """x: (nsim, n) 国家均值 → 均值、SE。"""
    n = x.shape[1]
    return x.mean(1), x.std(1, ddof=1) / np.sqrt(n)


def rules(D):
    nsim, n, L, _ = D.shape
    t95, t90 = stats.t.ppf(0.975, n - 1), stats.t.ppf(0.95, n - 1)
    Dk = D.mean(3)  # 合并预算 (nsim, n, L)
    out = {}
    # 草稿规则（按预算合并解读）：每种分类器 CI 下界>0 且点估计≥0.3；平均后 ≥8/11 国 >0；等价：每种分类器 90%CI ⊂ ±0.3
    sup, eq = np.ones(nsim, bool), np.ones(nsim, bool)
    for l in range(L):
        m, se = tstats(Dk[:, :, l])
        sup &= (m - t95 * se > 0) & (m >= 0.3)
        eq &= (m - t90 * se > -0.3) & (m + t90 * se < 0.3)
    sign = (Dk.mean(2) > 0).sum(1) >= 8
    out['draft_better'] = np.mean(sup & sign)
    out['draft_equiv'] = np.mean(eq)
    # 草稿规则的"4 个族"解读：每种分类器 × 每档预算都要满足
    sup4, eq4 = np.ones(nsim, bool), np.ones(nsim, bool)
    for l in range(L):
        for b in range(2):
            m, se = tstats(D[:, :, l, b])
            sup4 &= (m - t95 * se > 0) & (m >= 0.3)
            eq4 &= (m - t90 * se > -0.3) & (m + t90 * se < 0.3)
    out['draft4_better'] = np.mean(sup4 & sign)
    out['draft4_equiv'] = np.mean(eq4)
    # 建议规则：主估计量 = 分类器平均、预算合并；优：95%CI 下界>0；等价：90%CI ⊂ ±0.3；
    # 一致性（描述）：两种分类器点估计都 >0
    m, se = tstats(Dk.mean(2))
    cons = np.all(Dk.mean(1) > 0, axis=1)
    out['prop_better'] = np.mean(m - t95 * se > 0)
    out['prop_better_cons'] = np.mean((m - t95 * se > 0) & cons)
    out['prop_equiv'] = np.mean((m - t90 * se > -0.3) & (m + t90 * se < 0.3))
    if L == 2:  # 匹配交互检验（次要）
        mi, sei = tstats((Dk[:, :, 0] - Dk[:, :, 1]) / 2)
        out['match_detect'] = np.mean(np.abs(mi / sei) > t95)
    return out


OPTIONS = {
    # 名称: (M, J, clfs, 说明)
    'D0 草稿': (12, 4, ('ae', 's2')),
    'A 6 方法×2 分类器×J=4': (6, 4, ('ae', 's2')),
    'B 12 方法×2 分类器×J=2': (12, 2, ('ae', 's2')),
    'C 12 方法×仅 S2 分类器×J=4': (12, 4, ('s2',)),
}
SCEN = [  # (mu, gamma, 说明)
    (0.0, 0.0, '真值为零'),
    (0.0, 0.3, '主效应零 + 匹配交互'),
    (0.24, 0.29, '检验 2 点估计外推'),
    (0.3, 0.0, 'mu=0.3 无交互'),
    (0.3, 0.3, 'mu=0.3 + 匹配交互'),
    (0.5, 0.0, 'mu=0.5 无交互'),
    (0.5, 0.3, 'mu=0.5 + 匹配交互'),
]

if __name__ == '__main__':
    pd.set_option('display.width', 250)
    pd.set_option('display.max_columns', 30)
    rows = []
    for opt, (M, J, clfs) in OPTIONS.items():
        for tau in [0.15, 0.3, 0.5]:
            for mu, g, lab in SCEN:
                D = simulate(mu, g, tau, 1.1, M, J, 0.1, clfs)
                rows.append(dict(option=opt, tau=tau, mu=mu, gamma=g, **rules(D)))
    res = pd.DataFrame(rows)
    res.to_csv('power_options.csv', index=False)
    for tau in [0.15, 0.3, 0.5]:
        print(f'\n===== tau = {tau}（sigma_e=1.1, rho_m=0.1）=====')
        print(res[res.tau == tau].drop(columns='tau').round(3).to_string(index=False))
    # rho_m 敏感性：方案 A 与 B 在 tau=0.3、mu=0.3、gamma=0 下的建议规则功效
    print('\n===== rho_m 敏感性（tau=0.3, mu=0.3, gamma=0, 建议规则 prop_better）=====')
    for rho in [0.0, 0.1, 0.25, 0.4]:
        r_ = {opt: rules(simulate(0.3, 0.0, 0.3, 1.1, M, J, rho, clfs))['prop_better']
              for opt, (M, J, clfs) in OPTIONS.items()}
        print(rho, {k: round(v, 3) for k, v in r_.items()})
