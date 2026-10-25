import glob, numpy as np, pandas as pd, sys
sys.path.insert(0,'.')
import guidance_G as G
R=pd.concat([pd.read_csv(f) for f in glob.glob(G.OUT+'/rows_t*_*.csv')],ignore_index=True)
UNIT=['target','split','rep']
Y=R[R.cond.isin(['AE','S2'])&R.method.isin(G.E1)].copy()
C=UNIT+['clf','k']
Y['r']=Y.ba_ge30-Y.groupby(C).ba_ge30.transform('mean')
m=Y.groupby(C+['method']).r.transform('mean')
g=Y.groupby(C+['cond']).r.transform('mean')
res=Y.r-m-g
SS=lambda x:(x**2).sum()
tot=SS(Y.r)
print('格内平方和分解（每格 4 方法×2 指导）：总 %.1f；方法 %.1f (%.1f%%)；指导 %.1f (%.1f%%)；方法×指导 %.1f (%.1f%%)'%(tot,SS(m),100*SS(m)/tot,SS(g),100*SS(g)/tot,SS(res),100*SS(res)/tot))
print('方差（除以 n）：总 %.3f 方法 %.3f 指导 %.3f 交互 %.3f'%(Y.r.var(ddof=0),m.var(ddof=0),g.var(ddof=0),res.var(ddof=0)))
# 纯噪声下的期望：若方法与指导都无系统效应，指导部分的自由度份额 = 1/7
print('纯噪声期望份额：方法 3/7=%.1f%%，指导 1/7=%.1f%%，交互 3/7=%.1f%%'%(300/7,100/7,300/7))
# SRS 学习曲线 50->400 增益
S=R[R.method=='srs'].groupby(['target','clf','k']).ba_ge30.mean().unstack('k')
print(((S[400]-S[50]).groupby('target').mean()).round(2).to_dict())
# pred 通路 AE-S2（复现）
from scipy import stats
def cv(d): return d.groupby(['target','method','clf','k']).d.mean().groupby('target').mean()
KEY=UNIT+['method','clf','k']
def pair(a,b,ms,reps=None):
    A=R[(R.cond==a)&R.method.isin(ms)].set_index(KEY).ba_ge30;B=R[(R.cond==b)&R.method.isin(ms)].set_index(KEY).ba_ge30
    d=(A-B).dropna().rename('d').reset_index()
    return d if reps is None else d[d.rep.isin(reps)]
for ms in (['margin','predstrat_srs'],['margin'],['predstrat_srs']):
    v=cv(pair('AE','S2',ms,[0])); se=v.std(ddof=1)/np.sqrt(11)
    print(ms,'AE-S2 %.3f [%.3f, %.3f] p=%.3f'%(v.mean(),v.mean()-2.228*se,v.mean()+2.228*se,2*stats.t.sf(abs(v.mean()/se),10)))
# E1 仅 rep0（22 单元）AE-S2 的 CI 宽度，用于和 ORACLE（22 单元）对比
v=cv(pair('AE','S2',G.E1,[0])); se=v.std(ddof=1)/np.sqrt(11); print('E1 rep0 AE-S2 %.3f ±%.3f'%(v.mean(),2.228*se))
