"""predstrat 实际达到的构成偏移 vs 真实类 sqrt / equal 配额（确定性计算，不抽样、不训练）；用剂量斜率换算为 ΔBA 的粗略外推。"""
import sys; sys.path.insert(0,'.')
from load import ci, fmt
from wls import slope
import numpy as np, pandas as pd
pd.set_option('display.width',250)
X=pd.read_pickle('X.pkl'); L=pd.read_pickle('L.pkl'); A=pd.read_pickle('A.pkl')
U=pd.read_csv('unit_table.csv')
def quota(N,k,rule):
    w={'sqrt':np.sqrt(N),'equal':(N>0).astype(float)}[rule]
    n=np.floor(k*w/w.sum()).astype(int); n=np.minimum(n,N)
    # 余数按权重补足
    while n.sum()<k:
        j=np.argmax(np.where(n<N,w-n*w.sum()/k,-np.inf)); n[j]+=1
    return n
rows=[]
for _,u in U.iterrows():
    N=np.array([u[f'pi{c}'] for c in range(10)])*30000; N=np.round(N).astype(int)
    big=np.array([u[f'nT{c}']>=30 for c in range(10)])
    for k in (50,200):
        s=X[(X.method=='srs')&(X.target==u.target)&(X.split==u.split)&(X.rep==u.rep)&(X.k==k)].iloc[0]
        ns=np.array([s[f'n{c}'] for c in range(10)])
        r=dict(target=u.target,split=u.split,rep=u.rep,k=k)
        for rule in ('sqrt','equal'):
            nq=quota(N,k,rule); r['dLD_'+rule]=float(np.mean(np.log1p(nq[big])-np.log1p(ns[big])))
        for cond in ('AE','S2'):
            p=X[(X.method=='predstrat_srs')&(X.cond==cond)&(X.target==u.target)&(X.split==u.split)&(X.rep==u.rep)&(X.k==k)]
            if len(p):
                npz=np.array([p.iloc[0][f'n{c}'] for c in range(10)])
                r['dLD_pred_'+cond]=float(np.mean(np.log1p(npz[big])-np.log1p(ns[big])))
        rows.append(r)
D=pd.DataFrame(rows).dropna()
C=D.groupby('target')[['dLD_sqrt','dLD_equal','dLD_pred_AE','dLD_pred_S2']].mean()
C['frac_AE']=C.dLD_pred_AE/C.dLD_sqrt; C['frac_S2']=C.dLD_pred_S2/C.dLD_sqrt
print(C.round(3).to_string())
L9=L[~L.target.isin(['RO','PT'])]; b=slope(L); b9=slope(L9)
print('\n11 国均值：',C.mean().round(3).to_dict())
print('9 国（去 RO、PT）均值：',C.drop(['RO','PT']).mean().round(3).to_dict())
print(f'剂量斜率 b（全部）={b:.2f}，b（9 国）={b9:.2f}')
for g,cc,bb in (('11国',C,b),('9国',C.drop(['RO','PT']),b9)):
    print(g,'按剂量外推的 ΔBA：predstrat AE',round(bb*cc.dLD_pred_AE.mean(),2),'S2',round(bb*cc.dLD_pred_S2.mean(),2),
          '；真实类 sqrt',round(bb*cc.dLD_sqrt.mean(),2),'；equal',round(bb*cc.dLD_equal.mean(),2))
D.to_csv('a7_access.csv',index=False)
