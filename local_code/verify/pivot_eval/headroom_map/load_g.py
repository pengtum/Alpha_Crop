import glob, pandas as pd, os
os.environ['OMP_NUM_THREADS']='2'
fs = sorted(glob.glob('/Users/perng_mini/AlphaCrops_local/results/guidance_G/rows_t*.csv'))
D = pd.concat([pd.read_csv(f).assign(tier=os.path.basename(f).split('_')[1]) for f in fs], ignore_index=True)
D.to_pickle('G_all.pkl')
print(D.shape)
print(D.groupby(['tier','cond','method','k']).size().to_string())
