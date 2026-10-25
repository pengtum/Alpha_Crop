import sys, os
sys.path.insert(0, '/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops/label_expansion')
sys.argv=['x']
import numpy as np, pandas as pd
from scipy import stats
import analyze_G as A
from guidance_G import E1, FAMILIES
R=A.load_rows()
pd.set_option('display.width',200)
def s(v): return A.t_summary(v)
def show(name, st): print(name, {k:(round(v,6) if isinstance(v,float) else v) for k,v in st.items()})
dE1=A.deltas(R,'AE','S2',E1); vE1=A.country_values(dE1); sE1=s(vE1)
show('E1',sE1)
print(vE1.round(5).to_dict())
q=stats.t.ppf(0.95,10)
print('minbound', max(abs(sE1['mean']-q*sE1['se']),abs(sE1['mean']+q*sE1['se'])))
p_tost=max(stats.t.sf((sE1['mean']+0.3)/sE1['se'],10),stats.t.sf((0.3-sE1['mean'])/sE1['se'],10)); print('tost',p_tost)
# alt PI
u=A.unit_level(dE1).groupby('target'); vc=(u.var(ddof=1)/u.size()).reindex(vE1.index)
pm=A.prediction_interval(vE1.to_numpy(), vc.to_numpy()); print('altPI', pm)
pi=A.prediction_interval(vE1.to_numpy(), np.full(len(vE1), A.within_var(dE1))); print('PI',pi)
# XY
for a,b in (('AE','S2'),('AE','S2XY'),('AE','XY'),('S2XY','S2')):
    show(f'{a}-{b}', s(A.country_values(A.deltas(R,a,b,['lpm2_nogeo','kcenter'],reps=[0]))))
# ORACLE2
two=['lpm2_nogeo','typiclust']
dH=pd.concat([A.deltas(R,'ORACLE2','S2',two,reps=[0]),A.deltas(R,'ORACLE','S2',['margin'],reps=[0])])
show('H2', s(A.country_values(dH)))
for m in two: show('O2 '+m, s(A.country_values(A.deltas(R,'ORACLE2','S2',[m],reps=[0]))))
show('H', s(A.country_values(A.deltas(R,'ORACLE','S2',['lpm2_nogeo','typiclust','margin'],reps=[0]))))
for m in ['lpm2_nogeo','typiclust','margin']:
    for met in ['ba_ge30','oa']:
        v=A.country_values(A.deltas(R,'ORACLE','S2',[m],met,reps=[0])); show(f'O {m} {met}', s(v)); print('  n<0',(v<0).sum(), 'n==0',(v==0).sum())
# tier3
for ca,cb,ms in (('RAWZ_AE','RAWZ_S2',E1),('NATIVE_AE','NATIVE_S2',['lpm2_nogeo','typiclust'])):
    show(ca, s(A.country_values(A.deltas(R,ca,cb,ms,reps=[0],ks=(50,)))))
    show(ca+' main', s(A.country_values(A.deltas(R,'AE','S2',ms,reps=[0],ks=(50,)))))
    show(ca+'-AE', s(A.country_values(A.deltas(R,ca,'AE',ms,reps=[0],ks=(50,)))))
    show(cb+'-S2', s(A.country_values(A.deltas(R,cb,'S2',ms,reps=[0],ks=(50,)))))
# pred multi-metric
for met in ('ba_ge30','oa','ba_P1','ba_P2','ba_far'):
    show('pred '+met, s(A.country_values(A.deltas(R,'AE','S2',FAMILIES['pred'],met,reps=[0]))))
vp=A.country_values(A.deltas(R,'AE','S2',FAMILIES['pred'],reps=[0])); print('pred by country', vp.round(3).to_dict())
# other metrics E1
for met in ('oa','f1_ge30','ba_P1','ba_P2','ba_far','oa_P2','oa_P1','oa_far','ba_i200','ba_i250'):
    st=s(A.country_values(A.deltas(R,'AE','S2',E1,met))); print(met, round(st['mean'],4), round(st['lo90'],4), round(st['hi90'],4), A.verdict(st))
# SE typiclust
dd=dE1[dE1.target=='SE']
print('SE by method', dd.groupby(['method','clf','k']).d.mean().groupby('method').mean().round(3).to_dict())
print('SE cells typiclust >0', (dd[dd.method=='typiclust'].d>0).sum(), '/', (dd.method=='typiclust').sum())
# E1 without typiclust
show('E1 no typ', s(A.country_values(dE1[dE1.method!='typiclust'])))
# R3
for cond in ('AE','S2'):
    d=A.vs_srs(R,cond,E1); print(cond, d.d.mean(), (d.d<0).mean(), (d.d<-0.5).mean(), len(d))
    show('R3 '+cond, s(A.country_values(d)))
# heterogeneity
print(A.heterogeneity(dE1))
# per k
for k in (50,200): show(f'k={k}', s(A.country_values(dE1[dE1.k==k])))
show('k200 noPT', s(A.country_values(dE1[(dE1.k==200)&(dE1.target!='PT')])))
