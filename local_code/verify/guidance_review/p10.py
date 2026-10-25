import numpy as np, pandas as pd
w = pd.read_csv('check2_cell_diffs.csv'); s = w[w.budget.isin([50, 200])]
print('检验 2：方法−随机 的逐单元 SD（ae_rep/s2_rep）：', round(s.d_rr.std(), 2), round(s.d_geo.std(), 2),
      ' 二者相关', round(s.d_rr.corr(s.d_geo), 2))
rng = np.random.default_rng(0)
def sim(n=11, J=4, K=2, M=6, sd=1.2, tau_c=0.4, sd_m=0.3, rho=0.3, nsim=4000):
    out = []
    for _ in range(nsim):
        a = rng.normal(0, tau_c, (n, 1, 1, 1)); b = rng.normal(0, sd_m, (1, 1, 1, M))
        srs = rng.normal(0, sd * np.sqrt(rho), (n, J, K, 1))      # 共用 SRS 带来的相关
        xa = a + b + srs + rng.normal(0, sd * np.sqrt(1 - rho), (n, J, K, M))
        xs = a + b + srs + rng.normal(0, sd * np.sqrt(1 - rho), (n, J, K, M))
        out.append(np.percentile(xa, 10) - np.percentile(xs, 10))
    out = np.array(out); return out.std(), (out >= -0.2).mean()
for nm, (J, M) in {'D0 (J=4,M=12)': (4, 12), 'A (J=4,M=6)': (4, 6), 'B (J=2,M=12)': (2, 12)}.items():
    sd_, p = sim(J=J, M=M)
    print(f'{nm}: 每种分类器 P10 差的 SD {sd_:.3f}；真值相等时满足"差 ≥ −0.2"的概率 {p:.2f}')
