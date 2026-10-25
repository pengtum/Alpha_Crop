# 只读：统计各国测试集/候选池/外国池规模，并检查本地环境可用的模型接口
import numpy as np, glob, os, inspect, importlib
rows=[]
for f in sorted(glob.glob('/Users/perng_mini/AlphaCrops_local/results/guidance_G/stage0/*_s0.npz')):
    S=np.load(f, allow_pickle=True)
    rows.append((os.path.basename(f)[:2], len(S['foreign']), len(S['cand']), len(S['test'])))
for r in rows: print('%s foreign=%d cand=%d test=%d'%r)
print('test median', int(np.median([r[3] for r in rows])), 'max', max(r[3] for r in rows), 'sum', sum(r[3] for r in rows))
import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
print('sklearn', sklearn.__version__)
for C in [MLPClassifier, HistGradientBoostingClassifier, LogisticRegression, KNeighborsClassifier]:
    print(C.__name__, 'fit params:', list(inspect.signature(C.fit).parameters))
for m in ['torch','tabpfn','tabicl','xgboost','catboost','pytabkit','tabm','autogluon']:
    print(m, 'available' if importlib.util.find_spec(m) else 'NOT installed')
