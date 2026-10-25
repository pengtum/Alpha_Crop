import numpy as np, lightgbm as lgb, sys
CODE = "/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops/label_expansion"
sys.path.insert(0, CODE)
from sklearn.datasets import make_classification
X,y=make_classification(20000,128,n_informative=40,n_classes=10,n_clusters_per_class=2,random_state=0)
w=np.r_[np.ones(19000),np.full(1000,12.5)]
P=dict(n_estimators=60, learning_rate=0.05, num_leaves=63, max_depth=-1, min_child_samples=20, class_weight='balanced',
       subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=0, verbose=-1)
pred={}
for nj in [5,5,10]:
    m=lgb.LGBMClassifier(**P,n_jobs=nj).fit(X[:15000],y[:15000],sample_weight=w[:15000])
    pred.setdefault(nj,[]).append(m.predict_proba(X[15000:]))
print('5 vs 5 max|dp|', np.abs(pred[5][0]-pred[5][1]).max())
print('5 vs 10 max|dp|', np.abs(pred[5][0]-pred[10][0]).max(), ' label disagreement', (pred[5][0].argmax(1)!=pred[10][0].argmax(1)).mean())
