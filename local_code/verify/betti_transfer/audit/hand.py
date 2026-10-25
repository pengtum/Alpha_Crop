# 手工重算 2 个单元（AT_s1_r0、LV_s1_r0）主设定 K=8/q95/w=1/50/k=200 的分配与 TV_a
# 数据纪律：只读 cand、allT 的 npy 头、G_{f} 前 n_cand 行（流式）、GF_{f}；不读任何标签/概率键
import os, sys
for v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[v]='2'
sys.dont_write_bytecode=True
import zipfile, math
import numpy as np, pandas as pd
from numpy.lib import format as npf
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_info
LE='/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops/label_expansion'
sys.path.insert(0, LE)
from guidance_G import pool_positions
print('线程:', [(i['internal_api'], i['num_threads']) for i in threadpool_info()])
ST='/Users/perng_mini/AlphaCrops_local/results/guidance_G/stage0'
D='/Users/perng_mini/AlphaCrops_local/verify/betti_transfer'
RA=pd.read_csv(f'{D}/diag_rows.csv'); AA=pd.read_csv(f'{D}/diag_alloc.csv')
RI=pd.read_csv(f'{D}/indep/indep_rows.csv'); AI=pd.read_csv(f'{D}/indep/indep_alloc.csv')

def hdr(f):
    v=npf.read_magic(f)
    return npf.read_array_header_1_0(f) if v==(1,0) else npf.read_array_header_2_0(f)
def read_rows(z,key,n=None):
    with z.open(key+'.npy') as f:
        shape,fo,dt=hdr(f); assert not fo
        n=shape[0] if n is None else n
        cols=int(np.prod(shape[1:])) if len(shape)>1 else 1
        need=n*cols*dt.itemsize; b=f.read(need); assert len(b)==need
        a=np.frombuffer(b,dtype=dt).copy()
        return (a.reshape(n,cols) if len(shape)>1 else a), shape

def my_pool(n_cand, split, rep):   # 独立重写 pool_positions
    rng=np.random.default_rng(10000*split+100*rep+7)
    return np.sort(rng.choice(n_cand, min(30000,n_cand), replace=False))

def my_greedy(gamma, cap, n0, k):  # 纯 Python 逐步贪心，平局取编号小的组
    K=len(gamma); n=[0]*K
    for _ in range(min(k,int(sum(cap)))):
        best=None; bg=-math.inf
        for g in range(K):
            if n[g]>=cap[g]: continue
            a=n0[g]+n[g]
            gain=gamma[g]*((a+1)**-0.5-(a+2)**-0.5)
            if gain>bg: bg=gain; best=g      # 严格大于 → 平局保留编号小的
        n[best]+=1
    return np.array(n)

for ctry,split,rep in [('AT',1,0),('LV',1,0)]:
    unit=f'{ctry}_s{split}_r{rep}'
    with zipfile.ZipFile(f'{ST}/{ctry}_s{split}.npz') as z:
        print(unit, 'npz 成员:', sorted(n for n in z.namelist()))  # 只看成员名
        cand,_=read_rows(z,'cand'); n_cand=len(cand)
        with z.open('allT.npy') as f: shT,_,_=hdr(f)
        G={}; GF={}
        for fsp in ['ae','s2']:
            G[fsp],shp=read_rows(z,f'G_{fsp}',n_cand); assert shp[0]==shT[0]
            GF[fsp],_=read_rows(z,f'GF_{fsp}')
    pos=pool_positions({'cand':cand},split,rep); pos2=my_pool(n_cand,split,rep)
    print(f'{unit}: n_cand {n_cand} n_allT {shT[0]} 池 {len(pos)}  pool_positions 与重写一致 {np.array_equal(pos,pos2)}  池内最大位置 {pos.max()} < n_cand')
    for fsp in ['ae','s2']:
        Xp32=G[fsp][pos]; XF=GF[fsp].astype(np.float64)
        ref=RA[(RA.unit==unit)&(RA.space==fsp)&(RA.K==8)&(RA.thr=='q95')&(RA.w=='1/50')&(RA.k==200)].iloc[0]
        refa=AA[(AA.unit==unit)&(AA.space==fsp)&(AA.K==8)&(AA.thr=='q95')&(AA.w=='1/50')&(AA.k==200)].sort_values('group')
        refi=RI[(RI.target==ctry)&(RI.split==split)&(RI.rep==rep)&(RI.space==fsp)&(RI.thr=='q95')&(RI.k==200)].iloc[0]
        for prec in ['f64','f32']:
            X=Xp32.astype(np.float64) if prec=='f64' else Xp32
            km=KMeans(n_clusters=8,n_init=10,random_state=0).fit(X)
            lab=km.labels_; C=km.cluster_centers_.astype(np.float64); X64=Xp32.astype(np.float64)
            dp=np.sqrt(((X64[:,None,:]-C[None])**2).sum(-1))
            cap=np.bincount(lab,minlength=8); gamma=cap/cap.sum()
            r=np.array([np.quantile(dp[lab==g][np.arange((lab==g).sum()), g] if False else dp[lab==g, g],0.95) for g in range(8)])
            dF=np.sqrt(((XF[:,None,:]-C[None])**2).sum(-1)); gs=dF.argmin(1); ds=dF[np.arange(len(XF)),gs]
            m=np.bincount(gs[ds<=r[gs]],minlength=8).astype(float)
            n0=m/50.0
            nt=my_greedy(gamma.tolist(),cap.tolist(),n0.tolist(),200)
            nn=my_greedy(gamma.tolist(),cap.tolist(),[0.0]*8,200)
            TVa=0.5*np.abs(nt-nn).sum()/200; TVb=0.5*np.abs(nt-gamma*200).sum()/200; fr=m.sum()/50000
            # 分组编号可能不同：按 gamma 多重集比较
            same_groups=np.array_equal(np.sort(cap),np.sort(refa.capacity.values))
            exact=(np.array_equal(cap,refa.capacity.values) and np.array_equal(m.astype(int),refa.m_g.values)
                   and np.array_equal(nt,refa.n_transfer.values) and np.array_equal(nn,refa.n_none.values))
            print(f'  {fsp} {prec}: TV_a {TVa:.4f} TV_b {TVb:.6f} 计入率 {fr:.5f} | 主实现 TV_a {ref.TV_a:.4f} TV_b {ref.TV_b:.6f} 计入率 {ref.foreign_counted_rate:.5f} | 独立 tv_a {refi.tv_a:.4f} 计入率 {refi.foreign_rate:.5f}')
            print(f'     与主实现 diag_alloc 逐组完全一致(容量/m_g/n迁移/n无初始): {exact}; 容量多重集一致 {same_groups}')
            print(f'     cap {cap.tolist()}  m {m.astype(int).tolist()}  n迁移 {nt.tolist()}  n无初始 {nn.tolist()}  Σn0 {n0.sum():.2f}')
            if prec=='f64' and not exact:
                print('     主 diag_alloc: cap', refa.capacity.tolist(), 'm', refa.m_g.tolist(), 'n迁移', refa.n_transfer.tolist(), 'n无初始', refa.n_none.tolist())
