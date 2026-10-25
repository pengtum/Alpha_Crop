"""补充：去 RO 的方法 − SRS；k 依赖（k200 − k50）；各方法风险分级所需的汇总。"""
from common_bm import *

R = load()
lines = []
for metric in ('ba_ge30', 'oa'):
    D = all_vs_srs(R, metric)
    cm = D.groupby(['target', 'cond', 'clf', 'k', 'method']).d.mean().groupby(['target', 'method']).mean().unstack('method')[ALL8]
    rows = []
    for m in ALL8:
        s = ts(cm[m]); s2 = ts(cm[m].drop('RO'))
        rows.append(dict(method=m, mean11=s['mean'], lo11=s['lo95'], hi11=s['hi95'], pos11=s['n_pos'], p11=s['p'],
                         mean10=s2['mean'], lo10=s2['lo95'], hi10=s2['hi95'], pos10=s2['n_pos'], p10=s2['p'],
                         median11=cm[m].median()))
    T = pd.DataFrame(rows)
    T['holm11'] = holm(T.p11).to_numpy(); T['holm10'] = holm(T.p10).to_numpy()
    T['bh10'] = bh(T.p10).to_numpy()
    T.to_csv(f'{OUTD}/g_noRO_{metric}.csv', index=False)
    lines.append(f'\n=== {metric}：方法 − SRS（AE+S2 合并）11 国与去 RO 10 国；Holm/BH 族 = 8 方法 ===\n' + T.round(3).to_string(index=False))
    # k 依赖
    kk = D.groupby(['target', 'cond', 'method', 'clf', 'k']).d.mean().unstack('k')
    ki = (kk[200] - kk[50]).groupby(['target', 'method']).mean().unstack('method')[ALL8]
    avg = ki.mean(axis=1)
    s = ts(avg); s10 = ts(avg.drop('RO'))
    lines.append(f'{metric}：8 方法平均的 (k200 − k50) 差：{s["mean"]:+.3f} [{s["lo95"]:+.3f}, {s["hi95"]:+.3f}]，{s["n_pos"]}/11 国为正，p={s["p"]:.3f}；'
                 f'去 RO {s10["mean"]:+.3f} [{s10["lo95"]:+.3f}, {s10["hi95"]:+.3f}]，{s10["n_pos"]}/10，p={s10["p"]:.3f}；符号检验 p={stats.binomtest(s["n_pos"], 11).pvalue:.3f}')
    lines.append(f'  各方法 (k200−k50) 为正的国家数：' + str((ki > 0).sum().to_dict()))
    lines.append(f'  各方法 (k200−k50) 均值：' + str(ki.mean().round(3).to_dict()))
    # k=50 与 k=200 各自：方法 − SRS 为正的方法数（11 国均值）
    for k in (50, 200):
        v = D[D.k == k].groupby(['target', 'cond', 'clf', 'method']).d.mean().groupby(['target', 'method']).mean().unstack('method')[ALL8]
        lines.append(f'  k={k}：8 方法 11 国均值 ' + str(v.mean().round(2).to_dict()) + f'；>0 的方法 {(v.mean() > 0).sum()}/8；'
                     f'8 方法平均 {v.mean(axis=1).mean():+.3f}，>0 国家 {(v.mean(axis=1) > 0).sum()}/11')
open(f'{OUTD}/g_extra.log', 'w').write('\n'.join(lines))
print('\n'.join(lines))
