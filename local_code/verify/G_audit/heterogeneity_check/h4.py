import sys, numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, '.')
import analyze_G as A
from guidance_G import E1
R = A.load_rows()
d = A.deltas(R, 'AE', 'S2', E1)
cm = d.groupby(['target','method','split','rep']).d.mean().rename('y').reset_index()
cm['unit']=cm.target+'_'+cm.split.astype(str)+cm.rep.astype(str)
def cmF(cm):
    Y = cm.pivot_table(index=['target','unit'], columns='method', values='y')  # units x methods
    # remove unit means (absorbs C and U(C)), then remove method means within... two-way per country
    Yc = Y.sub(Y.mean(1), axis=0)            # unit-centered
    mCM = Yc.groupby(level=0).mean()          # country x method (unit-centered)
    mM = mCM.mean(0)
    CM = mCM - mM
    nU = Y.groupby(level=0).size()
    SS_CM = (CM.pow(2).sum(1)*nU).sum()
    E = Yc - mCM.reindex(Y.index.get_level_values(0)).to_numpy()
    SS_E = (E**2).to_numpy().sum()
    nC, nM = mCM.shape
    dfCM = (nC-1)*(nM-1); dfE = (len(Y)-nC)*(nM-1)
    F = (SS_CM/dfCM)/(SS_E/dfE)
    return F, dfCM, dfE, stats.f.sf(F, dfCM, dfE), SS_CM
print('all', cmF(cm)[:4])
print('drop SE', cmF(cm[cm.target!='SE'])[:4])
print('drop SE,PT', cmF(cm[~cm.target.isin(['SE','PT'])])[:4])
print('drop typiclust', cmF(cm[cm.method!='typiclust'])[:4])
# replace SE typiclust by NaN-> drop SE unit rows only for typiclust: use method-subset tests
# permutation: within each unit shuffle method labels (null: no method effects at all)
rng=np.random.default_rng(1)
Y = cm.pivot_table(index=['target','unit'], columns='method', values='y')
F0 = cmF(cm)[4]
cnt=0; B=5000
arr=Y.to_numpy()
for b in range(B):
    P = np.array([row[rng.permutation(4)] for row in arr])
    Yp = pd.DataFrame(P, index=Y.index, columns=Y.columns).stack().rename('y').reset_index()
    Yp = Yp.rename(columns={'level_2':'method'}) if 'level_2' in Yp else Yp
    if cmF(Yp)[4] >= F0: cnt+=1
print('perm p (SS_CM, within-unit method shuffle):', (cnt+1)/(B+1))
