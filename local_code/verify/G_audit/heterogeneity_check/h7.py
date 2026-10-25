import sys, numpy as np, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
import analyze_G as A
from guidance_G import E1
R = A.load_rows(); d = A.deltas(R, 'AE', 'S2', E1)
for lab, dd in (('drop typiclust', d[d.method!='typiclust']), ('drop SE-typiclust', d[~((d.target=='SE')&(d.method=='typiclust'))])):
    print(lab, A.fmt(A.t_summary(A.country_values(dd))))
    vv=A.country_values(dd); print('   R0-style', {k:round(float(x),3) for k,x in A.prediction_interval(vv.to_numpy(), np.full(len(vv), A.within_var(dd))).items()})
