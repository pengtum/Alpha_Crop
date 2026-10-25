import numpy as np, pandas as pd
from scipy import stats
from power_options import simulate
rows = []
for opt, (M, J) in {'D0': (12, 4), 'A': (6, 4), 'B': (12, 2)}.items():
    for tau in [0.15, 0.3, 0.5]:
        for mu in [0.0, 0.1, 0.24, 0.3, 0.5]:
            D = simulate(mu, 0.29, tau, 1.1, M, J, 0.1, nsim=20000)
            x = D.mean(3).mean(2)  # 分类器平均、预算合并的国家均值
            n = x.shape[1]; m = x.mean(1); se = x.std(1, ddof=1) / np.sqrt(n)
            t95, t90 = stats.t.ppf(0.975, n - 1), stats.t.ppf(0.95, n - 1)
            better = m - t95 * se > 0
            eq3 = (m - t90 * se > -0.3) & (m + t90 * se < 0.3)
            eq5 = (m - t90 * se > -0.5) & (m + t90 * se < 0.5)
            rows.append(dict(option=opt, tau=tau, mu=mu, P_better=better.mean(), P_equiv03=eq3.mean(),
                             P_equiv05=eq5.mean(), P_uncertain03=1 - better.mean() - (eq3 & ~better).mean()))
pd.set_option('display.width', 200)
print(pd.DataFrame(rows).round(2).to_string(index=False))
