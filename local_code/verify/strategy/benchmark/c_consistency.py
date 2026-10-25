"""(c) 方法排名的一致性：AE/S2（指导、分类器）、k、国家；方法 × 国家异质性；方法间配对比较。"""
import itertools

from common_bm import *

R = load()
rng = np.random.default_rng(0)


def kendall_w(M, B=20000):
    """M：raters × items。每行内排秩；W 与置换 p（各行独立打乱）。"""
    X = np.asarray(M, float)
    m, n = X.shape
    Rk = np.apply_along_axis(stats.rankdata, 1, X)

    def W_of(Rk):
        S = ((Rk.sum(0) - m * (n + 1) / 2) ** 2).sum()
        return 12 * S / (m ** 2 * (n ** 3 - n))
    w0 = W_of(Rk)
    cnt = 0
    for _ in range(B):
        P = np.array([rng.permutation(r) for r in Rk])
        cnt += W_of(P) >= w0 - 1e-12
    chi2 = m * (n - 1) * w0
    return w0, (cnt + 1) / (B + 1), stats.chi2.sf(chi2, n - 1)


out = {}
lines = []
for metric in ('ba_ge30', 'oa'):
    D = all_vs_srs(R, metric)  # E1 44 单元，其余重复 0
    D0 = all_vs_srs(R, metric, rep0_only=True)
    # ---- 1. 设置 × 方法矩阵（11 国均值）
    set_mat = D.groupby(['cond', 'clf', 'k', 'method', 'target']).d.mean().groupby(['cond', 'clf', 'k', 'method']).mean().unstack('method')[ALL8]
    set_mat.to_csv(f'{OUTD}/c_setting_by_method_{metric}.csv')
    w, pperm, pchi = kendall_w(set_mat.to_numpy())
    lines.append(f'[{metric}] 8 个设置（指导×分类器×k）对 8 方法排名的 Kendall W = {w:.3f}，置换 p = {pperm:.4f}，χ² p = {pchi:.4f}')
    set_mat_noRO = D[D.target != 'RO'].groupby(['cond', 'clf', 'k', 'method', 'target']).d.mean().groupby(['cond', 'clf', 'k', 'method']).mean().unstack('method')[ALL8]
    w2, pp2, _ = kendall_w(set_mat_noRO.to_numpy())
    lines.append(f'[{metric}]   去掉 RO：W = {w2:.3f}，置换 p = {pp2:.4f}')
    # 成对 Spearman
    def rho(a, b):
        return stats.spearmanr(set_mat.loc[a], set_mat.loc[b])[0]
    pairs = {'指导 AE vs S2': [(('AE', c, k), ('S2', c, k)) for c in ('s2', 'ae') for k in (50, 200)],
             '分类器 ae vs s2': [((g, 'ae', k), (g, 's2', k)) for g in ('AE', 'S2') for k in (50, 200)],
             'k 50 vs 200': [((g, c, 50), (g, c, 200)) for g in ('AE', 'S2') for c in ('s2', 'ae')]}
    for lab, pl in pairs.items():
        rs = [rho(a, b) for a, b in pl]
        lines.append(f'[{metric}]   Spearman（{lab}，4 对）：' + '，'.join(f'{r:+.2f}' for r in rs) + f'；均值 {np.mean(rs):+.2f}')
    best = set_mat.idxmax(1)
    worst = set_mat.idxmin(1)
    lines.append(f'[{metric}]   各设置最佳方法：' + '；'.join(f'{i[0]}/{i[1]}/k{i[2]}={b}' for i, b in best.items()))
    lines.append(f'[{metric}]   各设置最差方法：' + '；'.join(f'{i[0]}/{i[1]}/k{i[2]}={b}' for i, b in worst.items()))
    # ---- 2. 国家 × 方法矩阵（合并所有设置）
    cm = D.groupby(['target', 'cond', 'clf', 'k', 'method']).d.mean().groupby(['target', 'method']).mean().unstack('method')[ALL8]
    cm.to_csv(f'{OUTD}/c_country_by_method_{metric}.csv')
    w3, pp3, pc3 = kendall_w(cm.to_numpy())
    lines.append(f'[{metric}] 11 国对 8 方法排名的 Kendall W = {w3:.3f}，置换 p = {pp3:.4f}，χ² p = {pc3:.4f}')
    w4, pp4, _ = kendall_w(cm.drop('RO').to_numpy())
    lines.append(f'[{metric}]   去掉 RO：W = {w4:.3f}，置换 p = {pp4:.4f}')
    lines.append(f'[{metric}]   各国最佳方法：' + str(cm.idxmax(1).value_counts().to_dict()) + '；最差：' + str(cm.idxmin(1).value_counts().to_dict()))
    lines.append(f'[{metric}]   平均秩（1=最好）：' + str((-cm).rank(axis=1).mean().sort_values().round(2).to_dict()))
    # 每个指导下各自的国家 W
    for cond in ('AE', 'S2'):
        cmc = D[D.cond == cond].groupby(['target', 'clf', 'k', 'method']).d.mean().groupby(['target', 'method']).mean().unstack('method')[ALL8]
        wc, pc, _ = kendall_w(cmc.to_numpy(), B=5000)
        lines.append(f'[{metric}]   只看 {cond} 指导：国家间 W = {wc:.3f}，p = {pc:.4f}')
    # Friedman：方法主效应（国家 × 方法）
    fr = stats.friedmanchisquare(*[cm[m] for m in ALL8])
    lines.append(f'[{metric}] Friedman（方法主效应，11 国 × 8 方法，合并）：χ²={fr.statistic:.2f}，p = {fr.pvalue:.4g}')
    # ---- 3. 方法 × 指导交互：每国每方法的 AE−S2
    g = D.groupby(['target', 'cond', 'method', 'clf', 'k']).d.mean().unstack('cond')
    gi = (g['AE'] - g['S2']).groupby(['target', 'method']).mean().unstack('method')[ALL8]
    fr2 = stats.friedmanchisquare(*[gi[m] for m in ALL8])
    lines.append(f'[{metric}] 方法 × 指导交互（各方法 AE−S2 是否不同）Friedman：χ²={fr2.statistic:.2f}，p = {fr2.pvalue:.4f}；各方法 AE−S2 均值 ' + str(gi.mean().round(3).to_dict()))
    # 方法 × 分类器交互
    h = D.groupby(['target', 'cond', 'method', 'clf', 'k']).d.mean().unstack('clf')
    hi = (h['ae'] - h['s2']).groupby(['target', 'method']).mean().unstack('method')[ALL8]
    fr3 = stats.friedmanchisquare(*[hi[m] for m in ALL8])
    lines.append(f'[{metric}] 方法 × 分类器交互 Friedman：χ²={fr3.statistic:.2f}，p = {fr3.pvalue:.4f}；各方法 (clf ae − s2) 均值 ' + str(hi.mean().round(3).to_dict()))
    # 方法 × k 交互
    kk = D.groupby(['target', 'cond', 'method', 'clf', 'k']).d.mean().unstack('k')
    ki = (kk[200] - kk[50]).groupby(['target', 'method']).mean().unstack('method')[ALL8]
    fr4 = stats.friedmanchisquare(*[ki[m] for m in ALL8])
    lines.append(f'[{metric}] 方法 × k 交互 Friedman：χ²={fr4.statistic:.2f}，p = {fr4.pvalue:.4f}；各方法 (k200 − k50) 均值 ' + str(ki.mean().round(3).to_dict()))
    # ---- 4. 方法 × 国家（嵌套 ANOVA；重复 0，所有 8 方法；单元 = 国家内 2 个划分）
    Y = D0.groupby(['target', 'split', 'method']).d.mean().unstack('method')[ALL8]
    arr = np.stack([Y.xs(s, level='split').loc[COUNTRIES].to_numpy() for s in (0, 1)], axis=1)  # T × S × M
    T_, S_, M_ = arr.shape
    gm = arr.mean()
    mt = arr.mean(1)  # T×M
    t_ = arr.mean((1, 2))
    m_ = arr.mean((0, 1))
    ts_ = arr.mean(2)  # T×S
    e_mt = mt - t_[:, None] - m_[None, :] + gm
    ss_mt = S_ * (e_mt ** 2).sum()
    resid = arr - ts_[:, :, None] - mt[:, None, :] + t_[:, None, None]  # M×S(T)
    ss_ms = (resid ** 2).sum()
    df_mt, df_ms = (T_ - 1) * (M_ - 1), T_ * (S_ - 1) * (M_ - 1)
    F = (ss_mt / df_mt) / (ss_ms / df_ms)
    lines.append(f'[{metric}] 方法 × 国家（嵌套：对 方法 × 划分(国家)）：F({df_mt},{df_ms}) = {F:.2f}，p = {stats.f.sf(F, df_mt, df_ms):.4g}')
    arr2 = np.delete(arr, COUNTRIES.index('RO'), axis=0)
    T2 = arr2.shape[0]
    mt2 = arr2.mean(1); t2 = arr2.mean((1, 2)); m2 = arr2.mean((0, 1)); ts2 = arr2.mean(2); gm2 = arr2.mean()
    ss_mt2 = S_ * ((mt2 - t2[:, None] - m2[None, :] + gm2) ** 2).sum()
    ss_ms2 = ((arr2 - ts2[:, :, None] - mt2[:, None, :] + t2[:, None, None]) ** 2).sum()
    F2 = (ss_mt2 / ((T2 - 1) * (M_ - 1))) / (ss_ms2 / (T2 * (S_ - 1) * (M_ - 1)))
    lines.append(f'[{metric}]   去掉 RO：F({(T2 - 1) * (M_ - 1)},{T2 * (M_ - 1)}) = {F2:.2f}，p = {stats.f.sf(F2, (T2 - 1) * (M_ - 1), T2 * (M_ - 1)):.4g}')
    # 每个方法的国家异质性（单元值单因素 F）
    het = []
    for m in ALL8:
        dm = D[D.method == m]
        hh = A.heterogeneity(dm[['target', 'split', 'rep', 'method', 'clf', 'k', 'd']].groupby(UNIT + ['method', 'clf', 'k']).d.mean().reset_index())
        het.append(dict(method=m, F=hh['F'], df1=hh['df1'], df2=hh['df2'], pF=hh['pF']))
    H = pd.DataFrame(het)
    H['pF_holm'] = holm(H.pF).to_numpy()
    lines.append(f'[{metric}] 各方法国家异质性（单元值单因素 F）：\n' + H.round(4).to_string(index=False))
    # ---- 5. 方法间配对比较（重复 0 同单元；国家层面 t；28 对；Holm、BH、maxT 符号翻转）
    P = D0.groupby(['target', 'cond', 'clf', 'k', 'method']).d.mean().unstack('method')[ALL8]
    V = {}
    for a, b in itertools.combinations(ALL8, 2):
        V[f'{a} − {b}'] = (P[a] - P[b]).groupby('target').mean()
    V = pd.DataFrame(V)
    pr = []
    for c in V.columns:
        s = ts(V[c])
        pr.append(dict(contrast=c, mean=s['mean'], lo95=s['lo95'], hi95=s['hi95'], n_pos=s['n_pos'], p=s['p'],
                       mean_noRO=V[c].drop('RO').mean(), n_pos_noRO=int((V[c].drop('RO') > 0).sum())))
    PR = pd.DataFrame(pr)
    PR['p_holm'] = holm(PR.p).to_numpy()
    PR['q_bh'] = bh(PR.p).to_numpy()
    PR['p_maxT'] = A.maxT_signflip(V).to_numpy()
    PR.to_csv(f'{OUTD}/c_pairwise_{metric}.csv', index=False)
    lines.append(f'[{metric}] 方法间配对（重复 0，AE+S2 合并；28 对）：BH q<0.05 的：\n' +
                 PR[PR.q_bh < 0.05].round(4).to_string(index=False))
    lines.append(f'[{metric}]   Holm p<0.05 的：{PR[PR.p_holm < 0.05].contrast.tolist()}；maxT p<0.05 的：{PR[PR.p_maxT < 0.05].contrast.tolist()}')
    out[metric] = cm

# BA 与 OA 的方法排名是否一致（合并国家均值）
ba, oa = out['ba_ge30'].mean(), out['oa'].mean()
r = stats.spearmanr(ba, oa)
lines.append(f'BA 与 OA 的方法效应（11 国均值）Spearman ρ = {r[0]:+.2f}（p={r[1]:.3f}）；BA：{ba.round(2).to_dict()}；OA：{oa.round(2).to_dict()}')
# 国家内 BA 与 OA 效应的相关（每国 8 方法）
rr = [stats.spearmanr(out['ba_ge30'].loc[c], out['oa'].loc[c])[0] for c in COUNTRIES]
lines.append('各国内 BA 与 OA 方法效应的 Spearman：' + str(dict(zip(COUNTRIES, np.round(rr, 2)))))
open(f'{OUTD}/c_consistency.log', 'w').write('\n'.join(lines))
print('\n'.join(lines))
