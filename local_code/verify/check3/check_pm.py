import numpy as np, os
D='/Users/perng_mini/AlphaCrops_local/results/check3'
TS=['AT','BE','DE','EE','LT','LV','PT','RO','SE','SI','SK']
cl=list(np.load(f'{D}/AT_split.npz')['classes'])
R=np.load('replay_cm_idx.npz')
import sys; sys.path.insert(0,'.')
for T in TS:
    sp=np.load(f'{D}/{T}_split.npz')
    npool=np.bincount(sp['y_pool'],minlength=10); ntest=np.bincount(sp['y_test'],minlength=10)
    pres=npool>0
    out=[]
    for f in ['ae','s2']:
        for m in ['loco_pm','loco','cmn','cmb','wnat','within']:
            nz=set()
            for s in ['pool','test']:
                a=np.load(f'{D}/{T}_{f}_{m}_{s}.npy')
                nz|=set(np.flatnonzero((a>0).any(0)))
                if m=='loco_pm' and s=='test':
                    rowsum=np.abs(a.sum(1)-1).max()
            nzv=np.zeros(10,bool); nzv[list(nz)]=True
            out.append(f"{f}:{m}={'OK' if (nzv==pres).all() else 'DIFF nz='+str([cl[i] for i in np.flatnonzero(nzv)])}")
    absent=[cl[i] for i in np.flatnonzero(~pres)]
    small={cl[i]:int(npool[i]) for i in np.flatnonzero(pres & (npool<20))}
    tabs={cl[i]:int(ntest[i]) for i in np.flatnonzero(ntest>0) if ntest[i]<30}
    print(T,'absent_in_pool=',absent,'pool<20=',small,'test<30:',tabs,'test_classes_not_in_pool=',[cl[i] for i in np.flatnonzero((ntest>0)&~pres)])
    print('   ',' '.join(out))
