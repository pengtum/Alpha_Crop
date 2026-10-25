import sys, os, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0,'/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops/label_expansion')
os.chdir('/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops/label_expansion')
import analyze_G as A
from sklearn.isotonic import IsotonicRegression
pd.set_option('display.width',250)
R=A.load_rows()
sE1=A.t_summary(A.country_values(A.deltas(R,'AE','S2',A.E1)))
D={'+0.3':0.3,'U95':sE1['hi95'],'θ_E1':sE1['mean'],'L95':sE1['lo95'],'−0.3':-0.3}
print({k:round(float(v),3) for k,v in D.items()})
KS=np.array([0,50,100,200,400.])
C=pd.read_csv('/Users/perng_mini/AlphaCrops_local/verify/G_audit/labeleq/srs_curve_raw.csv',index_col=[0,1]); C.columns=C.columns.astype(int)
def q_inf(v,p):
    v=np.sort(np.asarray(v,float)); pos=p*(len(v)-1); lo=int(np.floor(pos)); hi=int(np.ceil(pos)); w=pos-lo
    a,b=v[lo],v[hi]
    if lo==hi or w==0: return a
    if np.isinf(a) or np.isinf(b):
        if np.isinf(a) and np.isinf(b) and a!=b: return np.nan
        return a if np.isinf(a) else b
    return a+(b-a)*w
def summ(v):
    v=np.asarray(v,float)
    return f"中位 {q_inf(v,.5):7.1f}  Q1 {q_inf(v,.25):7.1f}  Q3 {q_inf(v,.75):7.1f}  +inf {int(np.isposinf(v).sum()):2d}  −inf {int(np.isneginf(v).sum()):2d}  超范围 {100*np.mean(~np.isfinite(v)):4.1f}%  有限值中位 {np.median(v[np.isfinite(v)]):7.1f}"
def pav(f): return IsotonicRegression().fit_transform(KS,f)
def finv(ks,f,y):
    if y>f.max(): return np.inf
    if y<f.min(): return -np.inf
    j=np.searchsorted(f,y,side='left')
    if j==0: return ks[0]
    x0,x1,y0,y1=ks[j-1],ks[j],f[j-1],f[j]
    return x0+(y-y0)*(x1-x0)/(y1-y0)
def leq_anch(ks,f,k,d):
    fk=np.interp(k,ks,f); return finv(ks,f,fk+d)-finv(ks,f,fk)
print('\n=== 1) 方案实现（逐国×分类器，22 条曲线），分布 ===')
for k in (50,200):
    for n,d in D.items():
        v=[A.label_equiv(KS,pav(r.to_numpy()),k,d) for _,r in C.iterrows()]
        print(f'k={k:3d} {n:5s} Δ={d:+.3f}: '+summ(v))
print('\n=== 1b) 锚定版 f⁻¹(f(k)+Δ) − f⁻¹(f(k))（Δ=0 → 0） ===')
for k in (50,200):
    for n,d in D.items():
        v=[leq_anch(KS,pav(r.to_numpy()),k,d) for _,r in C.iterrows()]
        print(f'k={k:3d} {n:5s}: '+summ(v))
print('\n=== 1c) 国家层面：每国两分类器取中位（=均值；有 inf 则 inf）后跨 11 国 ===')
for k in (50,200):
    for n,d in D.items():
        v=pd.Series({ (t,c):A.label_equiv(KS,pav(r.to_numpy()),k,d) for (t,c),r in C.iterrows()})
        vc=v.groupby(level=0).apply(lambda x: q_inf(x.values,.5))
        print(f'k={k:3d} {n:5s}: '+summ(vc.values))
print('\n=== 2) 跨国平均曲线（每个分类器：11 国均值，PAV 后换算） ===')
for lab,sub in (('11 国均值',C),('去 PT、RO 的 9 国均值',C.drop(index=['PT','RO'],level=0))):
    for clf in ('s2','ae','both'):
        f=(sub if clf=='both' else sub.xs(clf,level=1)).mean().to_numpy()
        fp=pav(f)
        out=' '.join(f'{n}:k50={A.label_equiv(KS,fp,50,d):7.1f},k200={A.label_equiv(KS,fp,200,d):7.1f}' for n,d in D.items())
        print(f'{lab:>12s} {clf:4s} 曲线 {np.round(f,2)} PAV变动 {int(np.sum(np.abs(fp-f)>1e-9))}\n      {out}')
    f=(sub.groupby(level=1).median()).mean().to_numpy()
print('\n    跨国中位数曲线（22 条曲线逐点中位）:')
f=C.median().to_numpy(); fp=pav(f); print('   ',np.round(f,2),np.round(fp,2))
print('    '+' '.join(f'{n}:k50={A.label_equiv(KS,fp,50,d):7.1f},k200={A.label_equiv(KS,fp,200,d):7.1f}' for n,d in D.items()))
print('\n=== 3) 对数线性斜率（k∈{50,100,200,400}，pp/翻倍），标签当量 = k·(2^(Δ/b) − 1) ===')
L2=np.log2(KS[1:])
b=pd.Series({idx:np.polyfit(L2,r[[50,100,200,400]].to_numpy(),1)[0] for idx,r in C.iterrows()})
print('逐曲线斜率 b（pp/翻倍）：'); print(b.unstack().round(3).T)
print(f'b 中位 {b.median():.3f}，均值 {b.mean():.3f}，b≤0 的曲线 {int((b<=0).sum())}/22；去 PT/RO 中位 {b.drop(index=["PT","RO"],level=0).median():.3f}')
# 共同斜率：国家×分类器固定截距
Y=C[[50,100,200,400]]; Yc=Y.sub(Y.mean(axis=1),axis=0); xc=L2-L2.mean()
bpool=(Yc.to_numpy()@xc).sum()/(len(Y)*(xc@xc)); 
bpool9=((Yc.drop(index=['PT','RO'],level=0)).to_numpy()@xc).sum()/((len(Y)-4)*(xc@xc))
# 国家层 bootstrap
rng=np.random.default_rng(0); cs=C.index.get_level_values(0).unique()
bb=[]
for _ in range(4000):
    s=rng.choice(cs,len(cs)); Ys=pd.concat([Yc.loc[c] for c in s]); bb.append((Ys.to_numpy()@xc).sum()/(len(Ys)*(xc@xc)))
lo,hi=np.quantile(bb,[.025,.975])
print(f'共同斜率 b = {bpool:.3f} pp/翻倍（国家 bootstrap 95% [{lo:.3f}, {hi:.3f}]）；去 PT/RO：{bpool9:.3f}')
for lab,bv in (('共同 b',bpool),('去PT/RO b',bpool9),('b 中位',b.median())):
    out=' '.join(f'{n}:k50={50*(2**(d/bv)-1):6.1f},k200={200*(2**(d/bv)-1):6.1f}' for n,d in D.items())
    print(f'  {lab:9s}={bv:.3f}: {out}')
# 逐曲线对数线性（b>0 才有定义）
print('  逐曲线对数线性（b≤0 记为超范围）分布：')
for k in (50,200):
    for n,d in D.items():
        v=[k*(2**(d/bi)-1) if bi>0 else (np.inf if d>0 else -np.inf) for bi in b.values]
        print(f'   k={k:3d} {n:5s}: '+summ(v))
print('\n=== 4) 局部割线斜率（跨国均值）：k=50 用 (f100−f0)/100，k=200 用 (f400−f100)/300；标签当量=Δ/斜率 ===')
for lab,sub in (('22 曲线',C),('去 PT、RO',C.drop(index=['PT','RO'],level=0))):
    s50=((sub[100]-sub[0])/100); s200=((sub[400]-sub[100])/300)
    print(f'  {lab}: 斜率均值 k50 {s50.mean()*100:.3f} pp/100 标签，k200 {s200.mean()*100:.3f} pp/100 标签；中位 {s50.median()*100:.3f}, {s200.median()*100:.3f}')
    print('   均值斜率: '+' '.join(f'{n}:k50={d/s50.mean():6.1f},k200={d/s200.mean():6.1f}' for n,d in D.items()))
    print('   中位斜率: '+' '.join(f'{n}:k50={d/s50.median():6.1f},k200={d/s200.median():6.1f}' for n,d in D.items()))
print('\n=== 5) 分 k 的 θ（次要结果 k=50 +0.013、k=200 −0.120）换算 ===')
for k in (50,200):
    th=A.t_summary(A.country_values(A.deltas(R[R.k.isin([k])],'AE','S2',A.E1)))
    v=[A.label_equiv(KS,pav(r.to_numpy()),k,th['mean']) for _,r in C.iterrows()]
    print(f'  k={k}: θ_k={th["mean"]:+.3f} [{th["lo95"]:+.3f},{th["hi95"]:+.3f}] → '+summ(v)+f'；共同 b 换算 {k*(2**(th["mean"]/bpool)-1):.1f}')
