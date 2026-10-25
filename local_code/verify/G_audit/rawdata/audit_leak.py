"""泄漏/重复检查：测试集与外国池、候选池之间的精确重复（坐标+面积；AE 特征行）。"""
import os, hashlib
import numpy as np, pandas as pd
RES='/Users/perng_mini/AlphaCrops_local/results/guidance_G'; CACHE='/Users/perng_mini/AlphaCrops_local/data/cache_guidance_G'
lon=np.load(f'{CACHE}/lon.npy'); lat=np.load(f'{CACHE}/lat.npy'); area=np.load(f'{CACHE}/area.npy')
Xae=np.load(f'{CACHE}/X_ae.npy',mmap_mode='r')
def keys(idx): return set(zip(lon[idx].round(7), lat[idx].round(7), area[idx].round(6)))
def fh(idx): return set(hashlib.md5(np.ascontiguousarray(Xae[i]).tobytes()).hexdigest() for i in idx)
out=[]
for f in sorted(os.listdir(f'{RES}/stage0')):
    S=np.load(f'{RES}/stage0/{f}'); fr,cand,test=S['foreign'],S['cand'],S['test']
    kt=keys(test); ht=fh(test); hf=fh(fr); hc=fh(cand)
    out.append(dict(f=f, test_coord_in_foreign=len(kt&keys(fr)), test_coord_in_cand=len(kt&keys(cand)),
                    test_ae_in_foreign=len(ht&hf), test_ae_in_cand=len(ht&hc), n_test_ae_unique=len(ht), n_test=len(test)))
    print(out[-1], flush=True)
pd.DataFrame(out).to_csv('leak_checks.csv',index=False)
