"""任务 B-1/2：独立复算 PO、PK50/200、EM，检查 PK 标签来源与 EM 输入。"""
import numpy as np, pandas as pd, os
D = '/Users/perng_mini/AlphaCrops_local/results/check3'
C = 10
M = np.load('meta.npz', allow_pickle=True); y_all = M['y']
T = ['AT','BE','DE','EE','LT','LV','PT','RO','SE','SI','SK']

def load(t,f,m,s):
    p = np.load(f'{D}/{t}_{f}_{m}_{s}.npy').astype(np.float64)
    return p / p.sum(1, keepdims=True)

def oa(y, yh): return 100*(y==yh).mean()
def ba_ge30(y, yh):
    n = np.bincount(y, minlength=C); big = np.flatnonzero(n>=30)
    return 100*np.mean([(yh[y==c]==c).mean() for c in big])

def correct(p, pi_new, trained):
    # 独立写法：π_train 在训练类上均匀 → q ∝ p·π_new（只在训练类上）；全零行回退到 p
    w = np.where(trained, pi_new, 0.0)
    q = p*w
    s = q.sum(1)
    bad = s<=0
    q[bad] = p[bad]
    return q.argmax(1), bad.sum()

def saerens(p, trained, max_iter=200, tol=1e-6):
    k = np.flatnonzero(trained); P = p[:,k]
    pt = np.full(len(k), 1/len(k)); pi = pt.copy()
    for it in range(1, max_iter+1):
        a = P*(pi/pt); a /= a.sum(1, keepdims=True)
        new = a.mean(0)
        d = np.abs(new-pi).max(); pi = new
        if d < tol: break
    out = np.zeros(C); out[k] = pi
    return out, it, d<tol

def laplace(yl, alpha=1.0):
    n = np.bincount(yl, minlength=C).astype(float)
    return (n+alpha)/(n.sum()+alpha*C)

rows=[]; checks=[]
for t in T:
    sp = np.load(f'{D}/{t}_split.npz')
    pool, test, foreign = sp['pool'], sp['test'], sp['foreign']
    yp, yt = sp['y_pool'], sp['y_test']
    # 划分检查
    checks.append(dict(target=t, n_pool=len(pool), n_test=len(test), overlap_pool_test=len(np.intersect1d(pool,test)),
                       overlap_foreign_T=len(np.intersect1d(foreign, np.r_[pool,test])),
                       ypool_ok=bool((y_all[pool]==yp).all()), ytest_ok=bool((y_all[test]==yt).all())))
    piT = np.bincount(yp, minlength=C)/len(yp)
    for f in ['ae','s2']:
        Lp, Lt = load(t,f,'loco','pool'), load(t,f,'loco','test')
        assert Lp.shape[0]==len(pool) and Lt.shape[0]==len(test)
        tr = (Lt.max(0)>0) | (Lp.max(0)>0)
        tr_test_only = Lt.max(0)>0
        r = dict(target=t, feat=f, trained_same=bool((tr==tr_test_only).all()), n_trained=int(tr.sum()))
        yh0 = Lt.argmax(1)
        r['L0_oa'], r['L0_ba'] = oa(yt,yh0), ba_ge30(yt,yh0)
        present = piT>0
        yls,_ = correct(Lt, present.astype(float), tr); r['LS_oa'], r['LS_ba'] = oa(yt,yls), ba_ge30(yt,yls)
        ypo,nb = correct(Lt, piT, tr); r['PO_oa'], r['PO_ba'] = oa(yt,ypo), ba_ge30(yt,ypo); r['PO_fallback']=nb
        # 用 test 真实比例的 PO（检查 pool/test 比例差异是否要紧）
        piTt = np.bincount(yt, minlength=C)/len(yt)
        r['POtest_oa'] = oa(yt, correct(Lt, piTt, tr)[0])
        # EM：只在 pool 上
        for tag, mi in (('EM', 200), ('EM5k', 5000)):
            pe, it, cv = saerens(Lp, tr, mi)
            yh,_ = correct(Lt, pe, tr)
            r[f'{tag}_oa'], r[f'{tag}_ba'], r[f'{tag}_it'], r[f'{tag}_conv'] = oa(yt,yh), ba_ge30(yt,yh), it, cv
            r[f'{tag}_l1'] = np.abs(pe-piT).sum()
        # EM 在 test 上运行（对照：若误用 test 会怎样）
        pe_t,_,_ = saerens(Lt, tr, 200); r['EMtest_oa'] = oa(yt, correct(Lt, pe_t, tr)[0])
        # PK：原种子 1000+s（复现），另 20 个新种子 5000+s
        for k in (10,20,50,100,200):
            seeds = [('orig',1000+s) for s in range(3)] + [('new',5000+s) for s in range(20)]
            for kind, sd in seeds:
                if kind=='orig' and k not in (50,200): continue
                lab = np.random.default_rng(sd).choice(len(yp), k, replace=False)   # 下标 → pool
                pk = laplace(yp[lab])
                yh,_ = correct(Lt, pk, tr)
                rows.append(dict(target=t, feat=f, k=k, kind=kind, seed=sd, oa=oa(yt,yh), ba=ba_ge30(yt,yh),
                                 l1=np.abs(pk-piT).sum()))
        checks[-1][f'{f}_arm'] = r
        rows.append(dict(target=t, feat=f, k=-1, kind='arms', **{kk:vv for kk,vv in r.items() if kk not in ('target','feat')}))
R = pd.DataFrame(rows)
R.to_pickle('b1_rows.pkl')
Ck = pd.DataFrame([{k:v for k,v in c.items() if not k.endswith('_arm')} for c in checks])
print(Ck.to_string())
