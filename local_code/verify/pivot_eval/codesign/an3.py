import numpy as np, pandas as pd
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 400)
R = pd.read_csv('diag3_sel_rows.csv')
print(R.groupby(['learner', 'k']).size())
base = R[R.sel == 'none_srs'][['unit', 'feat', 'k', 'learner', 'ba', 'oa']].rename(columns={'ba': 'ba_s', 'oa': 'oa_s'})
D = R.merge(base, on=['unit', 'feat', 'k', 'learner'])
D['dba'] = D.ba - D.ba_s; D['doa'] = D.oa - D.oa_s
# 两种分类器特征平均 → 单元 → 国家 → 11 国
c = D.groupby(['learner', 'k', 'sel', 'tgt'])[['dba', 'doa', 'comp_L1', 'n_cls']].mean().reset_index()
a = c.groupby(['learner', 'k', 'sel']).agg(dba=('dba', 'mean'), doa=('doa', 'mean'), npos=('dba', lambda x: int((x > 0).sum())),
                                          sd_c=('dba', 'std'), L1=('comp_L1', 'mean'), ncls=('n_cls', 'mean')).reset_index()
a['ci'] = 1.96 * a.sd_c / np.sqrt(11)
keep = ['none_srs', 'AE_predstrat_srs', 'S2_predstrat_srs', 'AE_margin', 'S2_margin', 'AE_badge_c', 'AE_gist_margin',
        'AE_typiclust', 'AE_kcenter', 'AE_kherding', 'AE_lpm2_nogeo', 'ORACLE_margin', 'ORACLE_typiclust', 'ORACLE2_typiclust',
        'QORACLE_equal', 'QORACLE_sqrt', 'QORACLE_prop']
for k in (50, 200):
    t = a[(a.k == k) & a.sel.isin(keep)].pivot(index='sel', columns='learner', values=['dba', 'doa', 'npos'])
    t = t.reindex(keep)
    print(f'===== k={k}  (Δ 相对同一学习器的 SRS；两种特征平均；11 国均值)')
    print(t.round(2).to_string())
    # 方法间离散度：G 的非神谕方法
    nm = a[(a.k == k) & ~a.sel.str.contains('ORACLE') & (a.sel != 'none_srs')]
    print('非神谕方法间 ΔBA：', nm.groupby('learner').dba.agg(['mean', 'std', 'min', 'max', 'count']).round(2).to_string())
# 绝对值
S = R[R.sel == 'none_srs'].groupby(['learner', 'k'])[['ba', 'oa']].mean()
print(S.round(2))
