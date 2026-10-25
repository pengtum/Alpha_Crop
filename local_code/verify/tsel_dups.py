import numpy as np, pandas as pd, pyarrow.parquet as pq
from sklearn.model_selection import train_test_split
P='/Users/perng_mini/AlphaCrops_local/data/dataset_v2.parquet'
names=pq.ParquetFile(P).schema_arrow.names
meta=pq.read_table(P,columns=['parcel_id','country','longitude','latitude','label','area_ha','has_ae','has_s2']).to_pandas()
keep=np.flatnonzero(meta.has_ae.to_numpy()&meta.has_s2.to_numpy()); df=meta.iloc[keep].reset_index(drop=True)
ctry=df.country.to_numpy()
aecols=[c for c in names if c.startswith('ae_')]
for tgt in ['SE','LV','PT','SK']:
    t=np.flatnonzero(ctry==tgt); pool,test=train_test_split(t,test_size=0.3,random_state=42)
    sub=df.loc[t].copy(); sub['split']=np.where(np.isin(t,pool),'pool','test')
    sub['key']=sub.longitude.round(7).astype(str)+'_'+sub.latitude.round(7).astype(str)
    d=sub[sub.key.duplicated(keep=False)].sort_values('key')
    # 完全相同的 AE 特征向量（pool 与 test 之间）
    tab=pq.read_table(P,columns=aecols).take(keep[t]); A=np.column_stack([tab.column(c).to_numpy() for c in aecols])
    h=pd.util.hash_pandas_object(pd.DataFrame(A.round(6)),index=False).to_numpy()
    hs=pd.Series(h); sp=sub.split.to_numpy()
    dupmask=hs.duplicated(keep=False).to_numpy()
    grp=pd.DataFrame({'h':h[dupmask],'split':sp[dupmask]}).groupby('h').split.nunique()
    print(f'{tgt}: same-centroid rows {len(d)}; identical-AE rows {dupmask.sum()}, groups spanning pool&test {(grp>1).sum()}')
    if len(d): print(d[['parcel_id','label','area_ha','split','key']].head(12).to_string())
csv=pd.read_csv('/Users/perng_mini/AlphaCrops_local/results/pilot_selection.csv')
k=['target','strategy','budget','seed','clf_feat']
print('\nCSV rows',len(csv),'dup keys',csv.duplicated(k).sum(),'dup full rows',csv.duplicated().sum())
print('NaN ba by strategy:',csv[csv.ba.isna()].strategy.value_counts().to_dict())
print('NaN n_classes_sel by strategy:',csv[csv.n_classes_sel.isna()].strategy.value_counts().to_dict())
s=csv[~csv.strategy.isin(['ref','foreign_only','target_full','foreign+target_full'])]
print('cells per (target,strategy):',s.groupby(['target','strategy']).size().to_dict())
print('ba range',s.ba.min(),s.ba.max(),' any ba==0.1:',np.isclose(s.ba,0.1).any())
print(s.pivot_table(index=['budget','strategy'],columns='seed',values='n_classes_sel',aggfunc=lambda x:'/'.join(str(int(v)) for v in x)).to_string())
print(s.groupby(['budget','strategy']).n_classes_sel.mean().unstack(0).round(2).to_string())
