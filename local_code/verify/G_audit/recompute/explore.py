import pandas as pd, glob
R='/Users/perng_mini/AlphaCrops_local/results/guidance_G/'
fs=sorted(glob.glob(R+'rows_t*.csv'))
df=pd.concat([pd.read_csv(f).assign(_file=f.split('/')[-1]) for f in fs],ignore_index=True)
print(len(df))
print(df.groupby(['cond','method','k']).size().to_string())
print(df.groupby(['cond','method']).apply(lambda g: g[['target','split','rep']].drop_duplicates().shape[0]).to_string())
print(df.clf.value_counts())
print(df.prior_src.value_counts(dropna=False))
print(df.isna().sum()[df.isna().sum()>0])
# duplicates
key=['cond','method','k','clf','target','split','rep']
print('dups', df.duplicated(key).sum())
