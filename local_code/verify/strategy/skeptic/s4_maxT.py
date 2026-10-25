"""质疑者视角 s4：联合族（8 方法 × 指导 × 指标）相对 SRS 的 maxT 符号翻转；与 t 检验 Holm 对照。"""
import os, sys, itertools
os.environ['OMP_NUM_THREADS']='4'
sys.path.insert(0,'/Users/perng_mini/AlphaCrops_local/verify/strategy/benchmark')
from common_bm import *
pd.set_option('display.width',250)
R=load()
cols={}
for metric in ('ba_ge30','oa'):
    for cond in ('AE','S2'):
        d=vs_srs(R,cond,ALL8,metric)
        cm=d.groupby(['target','method','clf','k']).d.mean().groupby(['target','method']).mean().unstack('method')
        for m in ALL8: cols[(metric,cond,m)]=cm[m]
V=pd.DataFrame(cols)
def tstat(A): return A.mean(0)/(A.std(0,ddof=1)/np.sqrt(A.shape[0]))
signs=np.array(list(itertools.product([-1,1],repeat=11)))
for fam,sub in [('BA 16 项',V['ba_ge30']),('BA+OA 32 项',V)]:
    A=sub.values; t0=tstat(A)
    mx=np.array([np.abs(tstat(A*s[:,None])).max() for s in signs])
    pm=[(mx>=abs(t)-1e-12).mean() for t in t0]
    # 单项符号翻转 p
    ps=[(np.abs(np.array([tstat(A[:,[j]]*s[:,None])[0] for s in signs]))>=abs(t0[j])-1e-12).mean() for j in range(A.shape[1])]
    praw=[2*stats.t.sf(abs(t),10) for t in t0]
    out=pd.DataFrame(dict(mean=sub.mean().values,t=t0,p_t=praw,p_signflip=ps,p_maxT=pm),index=sub.columns)
    out['holm_t']=holm(pd.Series(praw,index=out.index)).values
    print(f'== {fam} =='); print(out[(out.p_maxT<0.2)|(out.p_t<0.1)].round(4).to_string())
