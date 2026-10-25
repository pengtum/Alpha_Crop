"""G 实验判定规则的功效与假阳性模拟。参数范围取自检验 2（配对 Δ_G 单元 SD ≈ 1.0–1.2 pp、国家间 SD 0–0.5）
与阶段 A r 网格（划分内 SRS SD 0.4–0.86 pp）。"""
import numpy as np, itertools
from scipy import stats
rng = np.random.default_rng(1)
G, CELLS, M = 11, 4, 12
def sim(mu, tau_c, sig=1.0, tau_m=0.3, tau_mc=0.3, rho_clf=0.5, srs_sd=0.9, meth_sd_vs_srs=0.9, nsim=2000, boot=False):
    out = dict(ae_better=0, tost=0, ci_lo_pos=0, pt_ge03=0, sign8=0, p10ok=0, p10ok_permethod=0, boot_cover=0, t_cover=0)
    tcrit95, tcrit90 = stats.t.ppf(0.975, G - 1), stats.t.ppf(0.95, G - 1)
    for s in range(nsim):
        ok_all = True; tost_all = True
        # 国家效应在两种分类器间相关
        zc = rng.multivariate_normal([0, 0], [[1, rho_clf], [rho_clf, 1]], size=G) * tau_c
        um = rng.normal(0, tau_m, M)
        for clf in range(2):
            umc = rng.normal(0, tau_mc, (G, M))
            # 单元 × 方法：AE 与 S2 指导下"方法 − SRS"（共用 SRS）
            srs = rng.normal(0, srs_sd, (G, CELLS))
            base = rng.normal(0, 0.5, (G, 1, M))  # 方法相对 SRS 的国家×方法基线（两指导共用）
            e_s2 = rng.normal(0, meth_sd_vs_srs, (G, CELLS, M))
            e_ae = rng.normal(0, meth_sd_vs_srs, (G, CELLS, M))
            # 令 Δ_G = mu + zc + um + umc + (e_ae - e_s2)，其噪声 SD = sqrt(2)*meth_sd ≈ sig
            d_s2 = base + e_s2 - srs[:, :, None]
            d_ae = base + e_ae - srs[:, :, None] + (mu + zc[:, clf][:, None, None] + um[None, None, :] + umc[:, None, :])
            dG = d_ae - d_s2
            fam_cell = dG.mean(2)            # (G, CELLS)
            cm = fam_cell.mean(1)            # 国家均值
            est = cm.mean(); se = cm.std(ddof=1) / np.sqrt(G)
            lo95 = est - tcrit95 * se
            lo90, hi90 = est - tcrit90 * se, est + tcrit90 * se
            out['t_cover'] += (lo95 <= mu <= est + tcrit95 * se) / 2
            if boot:
                bs = np.array([cm[rng.integers(0, G, G)].mean() for _ in range(400)])
                b_lo, b_hi = np.percentile(bs, [2.5, 97.5])
                out['boot_cover'] += (b_lo <= mu <= b_hi) / 2
            sign = (cm > 0).sum() >= 8
            # 下行风险：合并所有方法的单元值的第 10 百分位
            p10_ae = np.percentile(d_ae, 10); p10_s2 = np.percentile(d_s2, 10)
            p10ok = (p10_ae - p10_s2) >= -0.2
            # 逐方法：取中位数方法的判定
            pm = np.array([np.percentile(d_ae[:, :, m], 10) - np.percentile(d_s2[:, :, m], 10) for m in range(M)])
            out['p10ok_permethod'] += (pm >= -0.2).mean() / 2
            out['ci_lo_pos'] += (lo95 > 0) / 2
            out['pt_ge03'] += (est >= 0.3) / 2
            out['sign8'] += sign / 2
            out['p10ok'] += p10ok / 2
            ok_all &= (lo95 > 0) and (est >= 0.3) and sign and p10ok
            tost_all &= (lo90 > -0.3) and (hi90 < 0.3)
        out['ae_better'] += ok_all
        out['tost'] += tost_all
    return {k: round(v / nsim, 3) for k, v in out.items()}
print('mu tau_c | P(判AE更可靠) P(TOST等价) | 单分类器: P(CI下界>0) P(点≥0.3) P(≥8/11) P(P10合并不劣) P(P10逐方法不劣) | t-CI覆盖 boot-CI覆盖')
for tau_c in (0.2, 0.4, 0.6):
    for mu in (0.0, 0.2, 0.3, 0.5, 0.8):
        r = sim(mu, tau_c, nsim=600, boot=(mu == 0.3))
        print(f'{mu:.1f} {tau_c:.1f} | {r["ae_better"]:.3f} {r["tost"]:.3f} | {r["ci_lo_pos"]:.3f} {r["pt_ge03"]:.3f} {r["sign8"]:.3f} {r["p10ok"]:.3f} {r["p10ok_permethod"]:.3f} | {r["t_cover"]:.3f} {r["boot_cover"] if mu==0.3 else "-"}', flush=True)
# 等价区间半宽（单分类器，μ=0）
for tau_c in (0.0, 0.2, 0.4, 0.6):
    se = np.sqrt(tau_c**2/11 + (0.3**2)/(11*12) + (0.3**2)/(11*12) + (1.27**2)/(11*4*12))
    print(f'tau_c={tau_c}: 近似 SE={se:.3f}, 90%CI 半宽={stats.t.ppf(0.95,10)*se:.3f}, 95%半宽={stats.t.ppf(0.975,10)*se:.3f}; 80%功效 MDE(双侧0.05)≈{(stats.t.ppf(0.975,10)+stats.t.ppf(0.8,10))*se:.3f}')
