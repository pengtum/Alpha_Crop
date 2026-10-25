"""全 11 国 V2 的 H：LV/PT/RO 用原审查者的 V2 复训行（其选样已与本目录独立实现逐位一致，PT 一行已独立复训一致），
其余 8 国用本目录 b5_V2A/V2B/V2L 的复训行。与"只替换 3 国"的混合版本并列。只读。"""
import glob, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.getcwd())
import analyze_G as A
OUT = '/Users/perng_mini/AlphaCrops_local/results/guidance_G'
W = '/Users/perng_mini/AlphaCrops_local/verify/G_audit/oracle_fold_artifact_xv'
R = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(f'{OUT}/rows_t*_*.csv'))], ignore_index=True)
Vo = pd.concat([pd.read_csv(p) for p in glob.glob('/Users/perng_mini/AlphaCrops_local/verify/G_audit/oracle/a6_retrain_V2_*.csv')])
Vm = pd.concat([pd.read_csv(p) for p in glob.glob(f'{W}/b5_V2*.csv')])
Vm = Vm[Vm['var'] == 'V2']
key = ['target', 'split', 'method', 'k', 'clf']


def replace(R, V):
    V = V.rename(columns={'tgt': 'target'}).drop_duplicates(key).set_index(key).ba_ge30
    R2 = R.copy()
    m = R2.cond.eq('ORACLE') & R2.rep.eq(0)
    idx = R2[m].set_index(key).index
    hit = idx.isin(V.index)
    vals = R2.loc[m, 'ba_ge30'].values.copy()
    vals[hit] = V.reindex(idx[hit]).values
    R2.loc[m, 'ba_ge30'] = vals
    return R2, int(hit.sum())


def show(lab, RR):
    for ms in (['lpm2_nogeo', 'typiclust', 'margin'], ['typiclust'], ['lpm2_nogeo'], ['lpm2_nogeo', 'typiclust']):
        s = A.t_summary(A.country_values(A.deltas(RR, 'ORACLE', 'S2', ms, reps=[0])))
        print(f"{lab:14s} {'+'.join(ms):32s} {s['mean']:+.3f} [{s['lo95']:+.3f}, {s['hi95']:+.3f}] p={s['p']:.3f} {s['n_pos']}/11")


R3, n3 = replace(R, Vo)
Rall, nall = replace(R3, Vm)
print('替换行数：3 国', n3, '；全 V2', nall)
show('原始', R)
show('V2 仅 LV/PT/RO', R3)
show('V2 全 11 国', Rall)
for ms in (['typiclust'], ['lpm2_nogeo']):
    a = A.country_values(A.deltas(R, 'ORACLE', 'S2', ms, reps=[0]))
    b = A.country_values(A.deltas(Rall, 'ORACLE', 'S2', ms, reps=[0]))
    print(ms, '各国 原始 / 全V2 / 差：')
    print(pd.DataFrame({'orig': a, 'V2': b, 'diff': b - a}).round(3).to_string())
