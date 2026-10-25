import os, sys
os.environ['OMP_NUM_THREADS']='2'; os.environ['OPENBLAS_NUM_THREADS']='2'; os.environ['MKL_NUM_THREADS']='2'; os.environ['VECLIB_MAXIMUM_THREADS']='2'
sys.dont_write_bytecode=True
import numpy as np, pandas as pd
from scipy import stats
M = pd.read_csv('/Users/perng_mini/AlphaCrops_local/verify/step0/partB2/fa_rows_B2.csv', float_precision='round_trip')
W = M[M.grid=='new'].pivot_table(index=['target','split','rep','unit','clf'], columns='arm', values='FA', aggfunc='first', dropna=False).reset_index()
DROP=['RO','PT']
def cv(Wx, a, b):
    T = Wx.assign(d=Wx[a]-Wx[b])
    ae = T[T.clf=='ae'].groupby('target').d.mean(); s2 = T[T.clf=='s2'].groupby('target').d.mean()
    return pd.concat([ae,s2],axis=1).mean(axis=1, skipna=False)
def ts(v):
    v=v.dropna(); n=len(v); m=v.mean(); sd=v.std(ddof=1); q=stats.t.ppf(.975,n-1); return m, m-q*sd/np.sqrt(n), m+q*sd/np.sqrt(n)
def reading(Wx, drop_extra=()):
    r = cv(Wx,'RT200','PK200').drop(DROP+list(drop_extra)); a = cv(Wx,'PK200','Z0').drop(DROP+list(drop_extra)); d = cv(Wx,'RT200','Z0').drop(DROP+list(drop_extra))
    m, lo, hi = ts(r); sh = a.mean()/d.mean()
    rd = 1 if lo>0 else (2 if lo<=0<=hi and sh>=0.7 else 3)
    return f'R200 {m:+.2f} [{lo:+.2f},{hi:+.2f}] share {sh:.1%} -> {rd}'
print('全部:', reading(W))
for c in sorted(set(W.target)-set(DROP)):
    print('去掉', c, reading(W, [c]))
for rep in (0,1):
    print('只 rep', rep, reading(W[W.rep==rep]))
for sp in (0,1):
    print('只 split', sp, reading(W[W.split==sp]))
# 单元层分布
T = W[~W.target.isin(DROP)]
for a,b,n in (('PK200','Z0','A200'),('RT200','PK200','R200')):
    d = T[a]-T[b]; print(n, '单元层 9 国 n', d.notna().sum(), '均值 %.2f 中位 %.2f P10 %.2f P90 %.2f min %.2f max %.2f >0 %d' % (d.mean(), d.median(), d.quantile(.1), d.quantile(.9), d.min(), d.max(), (d>0).sum()))
# rep0 vs rep1 的 Z0 是否相同（Z0 与 rep 无关？）
z = W.pivot_table(index=['target','split','clf'], columns='rep', values='Z0')
print('Z0 rep0==rep1:', np.allclose(z[0], z[1]))
pk = W.pivot_table(index=['target','split','clf'], columns='rep', values='PKo')
print('PKo rep0==rep1:', np.allclose(pk[0].fillna(-1), pk[1].fillna(-1)))
