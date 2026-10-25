# 只读统计：用于"目标国初始样本"模拟的分组粒度与类别构成偏差（不训练、不看任何精度）
import os
for v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[v]='2'
import numpy as np, pandas as pd, json
D='/Users/perng_mini/AlphaCrops_local/data'
C=f'{D}/cache_guidance_G'
df=pd.read_parquet(f'{D}/dataset_v2.parquet',columns=['parcel_id','country','has_ae','has_s2','area_ha','EC_hcat_n'])
df=df[df.has_ae & df.has_s2].reset_index(drop=True)
y=np.load(f'{C}/y.npy'); ctry=np.load(f'{C}/ctry.npy'); lon=np.load(f'{C}/lon.npy'); lat=np.load(f'{C}/lat.npy')
assert len(df)==len(y) and (df.country.to_numpy()==ctry).all()
nuts=pd.read_parquet(f'{D}/parcel_nuts.parquet',columns=['parcel_id','nuts3','nuts2'])
df=df.merge(nuts,on='parcel_id',how='left',validate='one_to_one')
print('rows',len(df),'nuts3 missing',df.nuts3.isna().sum(), 'hcat n unique', df.EC_hcat_n.nunique())
rows=[]
for t in sorted(set(ctry)):
    m=ctry==t
    yt=y[m]; pi=np.bincount(yt,minlength=10)/m.sum()
    xy=np.c_[lon[m]*111.32*np.cos(np.radians(lat[m].mean())), lat[m]*110.57]
    cell=pd.factorize(pd.Series(np.floor(xy[:,0]/10).astype(int)).astype(str)+'_'+pd.Series(np.floor(xy[:,1]/10).astype(int)).astype(str))[0]
    n3=df.nuts3.to_numpy()[m]
    def grp_stats(g):
        codes,cnt=np.unique(g,return_counts=True)
        L1=[]; w=[]
        for c,n in zip(codes,cnt):
            p=np.bincount(yt[g==c],minlength=10)/n; L1.append(np.abs(p-pi).sum()); w.append(n)
        L1=np.array(L1); w=np.array(w)
        return len(codes), int(np.median(cnt)), int(cnt.min()), float(np.average(L1,weights=w)), float(np.median(L1))
    a=grp_stats(n3); b=grp_stats(cell)
    rows.append(dict(ctry=t,n=int(m.sum()),nuts3=a[0],nuts3_med_n=a[1],nuts3_min_n=a[2],nuts3_L1_wmean=round(a[3],3),
                     cell10=b[0],cell_med_n=b[1],cell_L1_wmean=round(b[3],3),cell_L1_med=round(b[4],3),
                     lat_p10=round(np.percentile(lat[m],10),2),lat_p90=round(np.percentile(lat[m],90),2),
                     ext_km_x=int(np.ptp(xy[:,0])),ext_km_y=int(np.ptp(xy[:,1])),
                     n_class=int((np.bincount(yt,minlength=10)>0).sum())))
out=pd.DataFrame(rows); print(out.to_string(index=False))
out.to_csv('init_sample_stats.csv',index=False)
