import glob, os, numpy as np, pandas as pd
R_G='/Users/perng_mini/AlphaCrops_local/results/guidance_G'
D1='/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops/label_expansion/storyline/decision_curve/decision_curve_rows.csv'
CV=pd.read_csv('/Users/perng_mini/AlphaCrops_local/verify/step0/partB/curves_B.csv')
c0=CV[CV.tau==0.0].set_index(['unit','clf','arm'])
GR=pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(R_G,'rows_t1_*.csv'))])
g=GR[(GR.cond=='none')&(GR.method=='srs')&GR.k.isin([50,200,400])].copy()
g['unit']=g.target+'_s'+g.split.astype(str)+'_r'+g.rep.astype(str); g['arm']='RT'+g.k.astype(str)
j=g.set_index(['unit','clf','arm']).join(c0,rsuffix='_c',how='inner')
print('RT τ=0 vs G rows:',len(j),'max|ΔOA|',(j.oa-j.oa_c).abs().max(),'max|ΔBA|',(j.ba_ge30-j.ba_ge30_c).abs().max(),'oa==oa_i300',(j.oa-j.oa_i300).abs().max())
d=pd.read_csv(D1)
b=d[d.design=='base_PT'].assign(arm='Z0').set_index(['unit','clf','arm'])
j=b.join(c0,rsuffix='_c',how='inner'); print('Z0 τ=0 vs D1 base_PT:',len(j),(j.oa-j.oa_c).abs().max(),(j.ba_ge30-j.ba_ge30_c).abs().max())
h=d[d.design=='srs_G_check'].copy(); h['arm']='HD'+h.k.astype(int).astype(str); h=h[h.arm.isin(['HD50','HD200','HD400'])].set_index(['unit','clf','arm'])
j=h.join(c0,rsuffix='_c',how='inner'); print('HD τ=0 vs D1 srs_G_check:',len(j),(j.oa-j.oa_c).abs().max(),(j.ba_ge30-j.ba_ge30_c).abs().max())
# HD s 与 D1
FA=pd.read_csv('/Users/perng_mini/AlphaCrops_local/verify/step0/partB/fa_rows.csv'); FA=FA[FA.anchor=='Z0']
hs=FA[FA.arm.str.startswith('HD')].set_index(['unit','clf','arm'])['s']
j=h.join(hs.rename('s_m'),how='inner'); print('HD 温度 s vs D1:',len(j),(j.s-j.s_m).abs().max())
# EM 点 vs G foreign_only oa_P1? (底座不同: G 用 P0test_i300；主实现用 P_T) -> 只报 Z0EM τ=0 与 G oa_P1 的差分布
fo=GR[(GR.cond=='none')&(GR.method=='foreign_only')].copy(); fo['unit']=fo.target+'_s'+fo.split.astype(str)+'_r'+fo.rep.astype(str)
em=c0.xs('Z0EM',level='arm')
j=fo.set_index(['unit','clf']).join(em,rsuffix='_c',how='inner')
print('Z0EM τ=0 (P_T 底座) − G oa_P1 (P0test 底座): 中位',(j.oa_c-j.oa_P1).median(),'范围',(j.oa_c-j.oa_P1).min(),(j.oa_c-j.oa_P1).max())
# OA* per country
z=FA[FA.arm=='Z0'].groupby('target').OA_star.mean(); print('OA* 国家均值 RO/PT:',z['RO'].round(2),z['PT'].round(2))
# RT200 在 RO/PT 进入窗口所需 τ
E=CV[(CV.arm=='RT200')&CV.unit.str[:2].isin(['RO','PT'])].merge(FA[FA.arm=='Z0'][['unit','clf','OA_star']],on=['unit','clf'])
w=E[(E.oa<=E.OA_star+1e-9)&(E.oa>=E.OA_star-4-1e-9)]
print('RO/PT RT200 窗口内 τ 格点范围:',w.groupby(E.unit.str[:2]).tau.agg(['min','max']).to_dict())
