"""每个选样文件的类别构成、同类空间覆盖与离群比例（只读 G 的 stage0/sel 与数据缓存；不训练）。"""
import os, glob
os.environ['OMP_NUM_THREADS']='4'
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
G='/Users/perng_mini/AlphaCrops_local/results/guidance_G'
Cd='/Users/perng_mini/AlphaCrops_local/data/cache_guidance_G'
H='/Users/perng_mini/AlphaCrops_local/verify/pivot_eval/geo_cost'
y=np.load(f'{Cd}/y.npy'); lon=np.load(f'{Cd}/lon.npy'); lat=np.load(f'{Cd}/lat.npy')
def km_xy(lo,la,lat0): return np.c_[lo*111.32*np.cos(np.radians(lat0)), la*110.57]
def part(A,d=32):
    A=np.asarray(A,dtype=np.float64)[:,:d]; return A/np.sqrt(A.var(0).sum()/d)
def pool_positions(S,split,rep):
    n=len(S['cand']); rng=np.random.default_rng(10_000*split+100*rep+7)
    return np.sort(rng.choice(n,min(30000,n),replace=False))
beta=pd.read_csv(f'{H}/d1b_distance_bins.csv')
bcols=['b_0_5','b_5_10','b_10_20','b_20_50','b_50_100']
beta=beta.groupby(['target','clf'])[bcols].mean()
EDGES=np.array([0,5,10,20,50,100])
CONDS=('AE','S2','none','ORACLE','ORACLE2')
rows=[]; unitrows=[]
for f0 in sorted(glob.glob(f'{G}/stage0/*.npz')):
    tgt,sp=os.path.basename(f0)[:-4].split('_s'); sp=int(sp)
    S=dict(np.load(f0))
    test=S['test']; lat0=float(S['lat0']); xyT=km_xy(lon[test],lat[test],lat0); yT=y[test]
    nT=np.bincount(yT,minlength=10); big=np.flatnonzero(nT>=30)
    pass
    for rep in (0,1):
        pp=pool_positions(S,sp,rep); idx=S['allT'][pp]; yP=y[idx]; xyP=km_xy(lon[idx],lat[idx],lat0)
        piP=np.bincount(yP,minlength=10)/len(yP)
        sp_ = {'ae':part(S['G_ae'][pp]),'s2':part(S['G_s2'][pp])}; sp_['common']=np.hstack([sp_['ae'],sp_['s2']])
        far={}
        for nm,X in sp_.items():
            r=np.linalg.norm(X-X.mean(0),axis=1); far[nm]=r>=np.quantile(r,0.99)
        unitrows.append(dict(target=tgt,split=sp,rep=rep,**{f'pi{c}':piP[c] for c in range(10)},**{f'nT{c}':nT[c] for c in range(10)},nbig=len(big)))
        for p in glob.glob(f'{G}/sel/{tgt}_s{sp}_r{rep}_*.npz'):
            b=os.path.basename(p)[:-4]; rest=b[len(f'{tgt}_s{sp}_r{rep}_'):]
            cond=[c for c in CONDS if rest.startswith(c+'_')]
            if not cond: continue
            cond=cond[0]; rest=rest[len(cond)+1:]; method,k=rest.rsplit('_k',1); k=int(k)
            if cond in('ORACLE','ORACLE2') and False: continue
            s=np.load(p)['sel_pos']; ys=yP[s]; n=np.bincount(ys,minlength=10)
            r=dict(target=tgt,split=sp,rep=rep,cond=cond,method=method,k=k,**{f'n{c}':n[c] for c in range(10)})
            # 同类空间覆盖：每个 big 类的测试地块到最近同类标签的距离
            sc5=[]; pred={'ae':[], 's2':[]}
            for c in big:
                m=yT==c
                if (ys==c).any():
                    d,_=cKDTree(xyP[s][ys==c]).query(xyT[m])
                else:
                    d=np.full(m.sum(),np.inf)
                sc5.append((d<=5).mean())
                bin_=np.searchsorted(EDGES,d,side='right')-1  # 0..4 为各箱，5 及以上为参照
                for clf in ('ae','s2'):
                    bv=np.r_[beta.loc[(tgt,clf)].to_numpy(),0.0]
                    pred[clf].append(bv[np.minimum(bin_,5)].mean())
            r['sc5']=float(np.mean(sc5)); r['spat_ae']=float(np.mean(pred['ae'])); r['spat_s2']=float(np.mean(pred['s2']))
            d5,_=cKDTree(xyP[s]).query(xyT); r['any5']=float((d5<=5).mean())
            for nm in ('ae','s2','common'): r[f'far_{nm}']=float(far[nm][s].mean())
            rows.append(r)
    print(tgt,sp,flush=True)
pd.DataFrame(rows).to_csv('sel_table.csv',index=False)
pd.DataFrame(unitrows).to_csv('unit_table.csv',index=False)
