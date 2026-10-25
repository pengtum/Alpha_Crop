"""(d) 目标标签价值（SRS k0→k200 增益）与策略效应的关系；附：类别构成偏移与 BA/OA 效应。"""
from common_bm import *

R = load()
S = R[R.method.isin(['srs', 'foreign_only'])].groupby(['target', 'split', 'rep', 'clf', 'k'])[['ba_ge30', 'oa']].mean()
lines = []
res = []
for metric in ('ba_ge30', 'oa'):
    X = S[metric].unstack('k')
    gain = {}
    for a, b in ((0, 200), (0, 50), (50, 200)):
        gain[f'{a}->{b}'] = (X[b] - X[a]).groupby(['target', 'clf']).mean().groupby('target').mean()
    G = pd.DataFrame(gain)
    D = all_vs_srs(R, metric)
    cm = D.groupby(['target', 'cond', 'clf', 'k', 'method']).d.mean().groupby(['target', 'method']).mean().unstack('method')[ALL8]
    summ = pd.DataFrame({'mean8': cm.mean(1), 'E1': cm[E1].mean(1), 'pred_hyb': cm[PRED + HYB].mean(1),
                         'sd_methods': cm.std(1, ddof=1), 'max_minus_min': cm.max(1) - cm.min(1), 'best': cm.max(1)})
    tab = pd.concat([G, summ, cm], axis=1)
    tab.to_csv(f'{OUTD}/d_label_value_{metric}.csv')
    lines.append(f'\n=== {metric}：国家层面 SRS 增益与策略效应 ===\n' + tab.round(2).to_string())
    for gcol in ('0->200', '0->50', '50->200'):
        for col in list(summ.columns) + ALL8:
            x, y = G[gcol], tab[col]
            rs, ps = stats.spearmanr(x, y)
            rp, pp = stats.pearsonr(x, y)
            xn, yn = x.drop('RO'), y.drop('RO')
            rs2, ps2 = stats.spearmanr(xn, yn)
            res.append(dict(metric=metric, gain=gcol, effect=col, spearman=rs, p_s=ps, pearson=rp, p_p=pp,
                            spearman_noRO=rs2, p_s_noRO=ps2))
Rz = pd.DataFrame(res)
# 多重比较：每个 (metric, gain) 下 8 个方法 + 6 个汇总 = 14 项
for (m, g), idx in Rz.groupby(['metric', 'gain']).groups.items():
    Rz.loc[idx, 'q_bh_s'] = bh(Rz.loc[idx, 'p_s']).to_numpy()
    Rz.loc[idx, 'q_bh_s_noRO'] = bh(Rz.loc[idx, 'p_s_noRO']).to_numpy()
Rz.to_csv(f'{OUTD}/d_corr.csv', index=False)
pd.set_option('display.width', 250)
lines.append('\n=== 相关（n=11；去 RO n=10）===\n' + Rz[Rz.gain == '0->200'].round(3).to_string(index=False))
lines.append('\n=== 0->50 / 50->200 的汇总列 ===\n' + Rz[(Rz.gain != '0->200') & Rz.effect.isin(['mean8', 'E1', 'pred_hyb', 'sd_methods', 'max_minus_min', 'best'])].round(3).to_string(index=False))

# ---- 附：类别构成偏移（comp_L1、n_classes_sel 相对 SRS）与 BA/OA 效应（方法 × 国家，AE+S2 合并）
K = ['target', 'split', 'rep', 'clf', 'k']
srs = R[R.method == 'srs'].set_index(K)[['comp_L1', 'n_classes_sel', 'ba_ge30', 'oa']]
rows = []
for cond in ('AE', 'S2'):
    g = R[(R.cond == cond) & R.method.isin(ALL8) & R.k.isin([50, 200]) & R.clf.isin(['s2', 'ae'])].set_index(K)
    for c in ('comp_L1', 'n_classes_sel', 'ba_ge30', 'oa'):
        g[f'd_{c}'] = g[c] - srs[c].reindex(g.index).to_numpy()
    rows.append(g.reset_index()[K + ['method'] + [f'd_{c}' for c in ('comp_L1', 'n_classes_sel', 'ba_ge30', 'oa')]].assign(cond=cond))
C = pd.concat(rows)
mc = C.groupby(['method', 'target'])[['d_comp_L1', 'd_n_classes_sel', 'd_ba_ge30', 'd_oa']].mean()
lines.append('\n=== 方法均值：相对 SRS 的构成偏移与精度差（11 国均值）===\n' + mc.groupby('method').mean().round(3).to_string())
for eff in ('d_ba_ge30', 'd_oa'):
    for med in ('d_comp_L1', 'd_n_classes_sel'):
        r_all = stats.spearmanr(mc[med], mc[eff])
        # 国家内（去掉国家均值后）的相关
        w = mc.groupby('target').transform(lambda x: x - x.mean())
        r_w = stats.spearmanr(w[med], w[eff])
        lines.append(f'{eff} ~ {med}：88 个 方法×国家 点 Spearman {r_all[0]:+.2f}（p={r_all[1]:.3g}）；国家内中心化后 {r_w[0]:+.2f}（p={r_w[1]:.3g}）')
open(f'{OUTD}/d_label_value.log', 'w').write('\n'.join(lines))
print('\n'.join(lines))
