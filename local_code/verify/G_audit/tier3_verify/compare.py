"""比较：主分析 AE−S2 与层级 3 RAWZ_AE−RAWZ_S2、NATIVE_AE−NATIVE_S2（同单元、同方法、同分类器、k=50、重复 0）。
估计量构造与 analyze_G.country_values / t_summary 相同。只读正式 rows_t1 与本目录的 rows_t3。"""
import glob, os, sys
import numpy as np, pandas as pd
LE = '/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops/label_expansion'
sys.path.insert(0, LE); os.chdir(LE)
from analyze_G import deltas, country_values, t_summary, fmt, verdict
E1 = ['lpm2_nogeo', 'kcenter', 'typiclust', 'kherding']
NAT = ['lpm2_nogeo', 'typiclust']
R1 = pd.concat([pd.read_csv(f) for f in glob.glob('/Users/perng_mini/AlphaCrops_local/results/guidance_G/rows_t1_*.csv')])
R3f = glob.glob('/Users/perng_mini/AlphaCrops_local/verify/G_audit/tier3_verify/out/rows_t3_*.csv')
R3 = pd.concat([pd.read_csv(f) for f in R3f]) if R3f else pd.DataFrame(columns=R1.columns)
R = pd.concat([R1, R3], ignore_index=True)
R = R[R.k.isin([50, 200])]
metric = sys.argv[1] if len(sys.argv) > 1 else 'ba_ge30'

def show(lab, d):
    if len(d) == 0:
        print(lab, '无数据'); return None
    v = country_values(d)
    n_units = d.groupby(['target', 'split', 'rep']).ngroups
    s = t_summary(v)
    print(f'{lab}（{n_units} 单元，{len(v)} 国，{len(d)} 配对）：', fmt(s), '|', verdict(s) if len(v) == 11 else '')
    print('   各国', v.round(3).to_dict())
    return d

for name, ca, cb, ms in (('RAWZ', 'RAWZ_AE', 'RAWZ_S2', E1), ('NATIVE', 'NATIVE_AE', 'NATIVE_S2', NAT)):
    print(f'==== {name}（{metric}）')
    dt = deltas(R, ca, cb, ms, metric)
    dt = dt[dt.k == 50]
    units = dt[['target', 'split', 'rep']].drop_duplicates()
    # 只保留该单元内所有方法 × 分类器都齐的单元
    full = dt.groupby(['target', 'split', 'rep']).size()
    keep = full[full == 2 * len(ms)].index
    dt = dt.set_index(['target', 'split', 'rep']).loc[keep].reset_index() if len(keep) else dt.iloc[:0]
    dm = deltas(R, 'AE', 'S2', ms, metric)
    dm = dm[(dm.k == 50) & (dm.rep == 0)]
    dm_same = dm.set_index(['target', 'split', 'rep']).loc[keep].reset_index() if len(keep) else dm.iloc[:0]
    show(f'  层级3 {ca}−{cb}', dt)
    show('  主分析 AE−S2（同单元）', dm_same)
    show('  主分析 AE−S2（k=50，重复 0 全部 22 单元）', dm)
    # 同分类器拆开
    for clf in ('s2', 'ae'):
        a = dt[dt.clf == clf]
        if len(a):
            print(f'   clf={clf}: 层级3 {country_values(a).mean():+.3f}；主分析同单元 {country_values(dm_same[dm_same.clf == clf]).mean():+.3f}')
    for m in ms:
        a = dt[dt.method == m]
        if len(a):
            print(f'   {m}: 层级3 {country_values(a).mean():+.3f}；主分析同单元 {country_values(dm_same[dm_same.method == m]).mean():+.3f}')
    # 预处理本身对各指导的影响：RAWZ_S2 − S2，RAWZ_AE − AE
    for g, gm in ((ca, 'AE'), (cb, 'S2')):
        dd = deltas(R, g, gm, ms, metric)
        dd = dd[dd.k == 50]
        if len(keep):
            dd = dd.set_index(['target', 'split', 'rep']).loc[keep].reset_index()
            print(f'   {g} − {gm}（预处理变化本身）：{country_values(dd).mean():+.3f}，t: {fmt(t_summary(country_values(dd)))}' if dd.target.nunique() > 2 else '')
