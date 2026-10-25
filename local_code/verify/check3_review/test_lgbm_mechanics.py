"""合成数据：验证 classes_、full_proba、eval_sample_weight、best_iteration、未见标签。"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, lightgbm as lgb
from sklearn.utils.class_weight import compute_sample_weight
rng = np.random.default_rng(0)
C = 10
present = np.array([1, 4, 7, 9])                  # 非连续标签，模拟"只见过部分类"
n = 4000
y = rng.choice(present, n, p=[0.6, 0.25, 0.1, 0.05])
X = rng.normal(size=(n, 5)) + y[:, None] * 0.35
tr, va, te = np.arange(0, 2400), np.arange(2400, 3200), np.arange(3200, 4000)
P = dict(n_estimators=300, learning_rate=0.1, num_leaves=15, class_weight=None, subsample=0.8, subsample_freq=1,
         colsample_bytree=0.8, n_jobs=2, random_state=42, verbose=-1)

def full_proba(m, X, n_classes):
    p = np.zeros((len(X), n_classes), np.float32); p[:, m.classes_] = m.predict_proba(X); return p

w_tr = compute_sample_weight('balanced', y[tr]); w_va = compute_sample_weight('balanced', y[va])
m = lgb.LGBMClassifier(**P).fit(X[tr], y[tr], sample_weight=w_tr, eval_set=[(X[va], y[va])],
                                eval_sample_weight=[w_va], callbacks=[lgb.early_stopping(20, verbose=False)])
print('[1] classes_ =', m.classes_, 'dtype', m.classes_.dtype, '（原标签值，而非 0..k-1）')
fp = full_proba(m, X[te], C)
print('    full_proba 非零列 =', np.flatnonzero(fp.sum(0) > 0), ' argmax 与 predict 一致:',
      bool((fp.argmax(1) == m.predict(X[te])).all()), ' 行和 max|1-s| =', float(np.abs(fp.sum(1) - 1).max()))

# [2] eval_sample_weight 是否进入早停指标
def wlogloss(p, yy, w):
    idx = np.searchsorted(m.classes_, yy); return float(np.sum(-w * np.log(np.clip(p[np.arange(len(yy)), idx], 1e-15, 1))) / w.sum())
res = m.evals_result_['valid_0']['multi_logloss']
it = m.best_iteration_
pb = m.predict_proba(X[va], num_iteration=it)
print(f'[2] best_iteration_={it}, n_trees={m.booster_.num_trees() // len(m.classes_)}, len(evals)={len(res)}')
print(f'    evals_result[best-1]={res[it-1]:.6f}  手算加权 logloss={wlogloss(pb, y[va], w_va):.6f}  '
      f'手算不加权={wlogloss(pb, y[va], np.ones(len(va))):.6f}  best_score_={m.best_score_["valid_0"]["multi_logloss"]:.6f}')
# 不加权时的早停轮次对比
m2 = lgb.LGBMClassifier(**P).fit(X[tr], y[tr], sample_weight=w_tr, eval_set=[(X[va], y[va])],
                                 callbacks=[lgb.early_stopping(20, verbose=False)])
print(f'    不传 eval_sample_weight 时 best_iteration_={m2.best_iteration_}（对照）')

# [3] predict_proba 默认是否用 best_iteration
p_def = m.predict_proba(X[te]); p_best = m.predict_proba(X[te], num_iteration=it); p_all = m.predict_proba(X[te], num_iteration=0)
print(f'[3] 默认 vs best: max|Δ|={np.abs(p_def - p_best).max():.2e}; 默认 vs 全部树: max|Δ|={np.abs(p_def - p_all).max():.2e}')

# [4] 早停验证集含训练集未见标签
yb = y.copy(); yb[va[:1]] = 3
try:
    lgb.LGBMClassifier(**P).fit(X[tr], yb[tr], eval_set=[(X[va], yb[va])], callbacks=[lgb.early_stopping(20, verbose=False)])
    print('[4] 未报错')
except Exception as e:
    print('[4] 验证集含未见标签 ->', type(e).__name__, str(e)[:90])

# [5] float32 下溢：极小概率存成 float32
m3 = lgb.LGBMClassifier(**{**P, 'n_estimators': 400, 'learning_rate': 0.3}).fit(X[tr], y[tr])
pp = m3.predict_proba(X[te]); print(f'[5] float64 最小概率 {pp.min():.2e}; 转 float32 后为 0 的比例 {(pp.astype(np.float32) == 0).mean():.4f}')
