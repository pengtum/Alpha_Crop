"""同一个检验 2 模型上：标签用于先验（PK） vs 用于训练（random-k） vs 两者叠加；以及 PO、EM。"""
import numpy as np, pandas as pd, os, warnings; warnings.filterwarnings('ignore')
C=10; D='/Users/perng_mini/AlphaCrops_local/results/check3'
def laplace(yl, a=1.0):
    n = np.bincount(yl, minlength=C).astype(float); return (n+a)/(n.sum()+a*C)
def em(p, tr, it=5000, tol=1e-6):
    k=np.flatnonzero(tr); P=p[:,k]; pt=np.full(len(k),1/len(k)); pi=pt.copy()
    for _ in range(it):
        a=P*(pi/pt); a/=a.sum(1,keepdims=True); new=a.mean(0)
        if np.abs(new-pi).max()<tol: pi=new; break
        pi=new
    o=np.zeros(C); o[k]=pi; return o
def ld(t,strat,k,s,f):
    d=np.load(f'b5_probs/{t}_{strat}_{k}_{s}_{f}.npz'); pt=d['test'].astype(float); pp=d['pool'].astype(float)
    return pt/pt.sum(1,keepdims=True), pp/pp.sum(1,keepdims=True), d['lab'], float(d['agree'])
rows=[]
import sys
TT = sys.argv[1].split(',') if len(sys.argv)>1 else ['PT','SE','SK','LV']
for t in TT:
    sp=np.load(f'{D}/{t}_split.npz'); yp,yt=sp['y_pool'],sp['y_test']; piT=np.bincount(yp,minlength=C)/len(yp)
    for f in ('ae','s2'):
        for s in range(3):
            p,pp,_,ag=ld(t,'foreign_only',0,s,f); tr=p.max(0)>0
            oa=lambda q: 100*((q).argmax(1)==yt).mean()
            r=dict(target=t,feat=f,seed=s,agree_fo=ag, FO=oa(p), FO_PO=oa(p*piT*tr), FO_EM=oa(p*em(pp,tr)))
            for k in (50,200):
                pr,_,lab,ag2=ld(t,'random',k,s,f)
                lab2=np.random.default_rng(1000*s+k).choice(len(yp),k,replace=False); assert (lab==lab2).all()
                pk=laplace(yp[lab])
                r[f'FO_PK{k}']=oa(p*pk*tr); r[f'R{k}']=oa(pr); r[f'R{k}_PK']=oa(pr*pk*(pr.max(0)>0)); r[f'agree_r{k}']=ag2
                r[f'R{k}_PO']=oa(pr*piT)
            rows.append(r)
X=pd.DataFrame(rows); X.to_pickle('b6.pkl')
pd.set_option('display.width',250)
print('复训与已存预测一致率最小值：', X[['agree_fo','agree_r50','agree_r200']].min().to_dict())
G=X.groupby(['feat','target']).mean(numeric_only=True).drop(columns=['seed','agree_fo','agree_r50','agree_r200'])
T=pd.DataFrame({'FO':G.FO,'PO':G.FO_PO-G.FO,'EM':G.FO_EM-G.FO,'PK50(先验)':G.FO_PK50-G.FO,'R50(训练)':G.R50-G.FO,'R50+PK50':G.R50_PK-G.FO,
                'PK200(先验)':G.FO_PK200-G.FO,'R200(训练)':G.R200-G.FO,'R200+PK200':G.R200_PK-G.FO,'R200+PO':G.R200_PO-G.FO})
print(T.round(2).to_string())
for f in ('ae','s2'):
    print(f.upper(),'4 国均值：', T.loc[f].mean().round(2).to_dict())
    print(f.upper(),'去 PT 均值：', T.loc[f].drop('PT').mean().round(2).to_dict())
