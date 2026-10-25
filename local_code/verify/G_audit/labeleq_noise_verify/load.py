import glob, os, numpy as np, pandas as pd
OUT='/Users/perng_mini/AlphaCrops_local/results/guidance_G'
def load():
    fs=glob.glob(os.path.join(OUT,'rows_t*_*.csv'))
    return pd.concat([pd.read_csv(f) for f in fs],ignore_index=True)
