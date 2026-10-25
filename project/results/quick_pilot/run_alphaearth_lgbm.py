"""
AlphaEarth Boosting实验 - 与medium设置一致，但使用Boosting模型
"""
import json, warnings, sys
import numpy as np
import pandas as pd
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from collections import Counter

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── 1. 读取数据 ──
emb = pd.read_parquet("data/processed/parcels_filtered_embeddings_interactive.parquet")
pf = pd.read_parquet("data/processed/parcels_filtered.parquet")

# 合并标签和纬度
pf_sub = pf.iloc[emb["parcel_idx"].values][["label_tier_a", "latitude"]].reset_index(drop=True)
df = pd.concat([emb.reset_index(drop=True), pf_sub], axis=1)

# ── 2. 特征 ──
feat_cols = [c for c in df.columns if c.endswith("_mean") or c.endswith("_stdDev")]
feat_cols.sort()

# ── 3. 标签 ──
label_col = "label_tier_a"
df = df.dropna(subset=[label_col]).reset_index(drop=True)

# ── 4. 采样：每类最多100，总上限10000 ──
MAX_PER_CLASS = 100
MAX_TOTAL = 10000
sampled = []
for cls, grp in df.groupby(label_col):
    n = min(len(grp), MAX_PER_CLASS)
    sampled.append(grp.sample(n=n, random_state=42))
df_s = pd.concat(sampled).reset_index(drop=True)
if len(df_s) > MAX_TOTAL:
    df_s = df_s.sample(n=MAX_TOTAL, random_state=42).reset_index(drop=True)

# 过滤样本数<2的类
cls_counts = df_s[label_col].value_counts()
valid_cls = cls_counts[cls_counts >= 2].index
df_s = df_s[df_s[label_col].isin(valid_cls)].reset_index(drop=True)

# 编码标签
le = LabelEncoder()
y = le.fit_transform(df_s[label_col])
X = df_s[feat_cols].values
lat = df_s["latitude"].values

n_samples = len(y)
n_classes = len(le.classes_)
n_features = X.shape[1]
print(f"Samples={n_samples}, Classes={n_classes}, Features={n_features}")

# ── 5. 模型 ──
model = HistGradientBoostingClassifier(
    max_iter=200, max_depth=6, learning_rate=0.1, random_state=42
)
model_name = "HistGradientBoosting(max_iter=200)"

# ── 6a. ID评估 (80/20 random) ──
from sklearn.model_selection import train_test_split
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model.fit(X_tr, y_tr)
y_pred_id = model.predict(X_te)
id_f1 = f1_score(y_te, y_pred_id, average="macro")

# ── 6b. OOD评估 (纬度中位数：北训南测) ──
lat_median = np.median(lat)
north = lat >= lat_median
south = ~north
model_ood = HistGradientBoostingClassifier(
    max_iter=200, max_depth=6, learning_rate=0.1, random_state=42
)
model_ood.fit(X[north], y[north])
y_pred_ood = model_ood.predict(X[south])
ood_f1 = f1_score(y[south], y_pred_ood, average="macro")

drop = (id_f1 - ood_f1) / id_f1 * 100 if id_f1 > 0 else 0.0

print(f"ID  F1={id_f1:.4f}")
print(f"OOD F1={ood_f1:.4f}")
print(f"DROP={drop:.1f}%")

# ── 7. JSON输出 ──
summary = {
    "experiment": "alphaearth_lgbm",
    "n_samples": n_samples,
    "n_classes": n_classes,
    "n_features": n_features,
    "max_per_class": MAX_PER_CLASS,
    "id_macro_f1": round(id_f1, 4),
    "ood_macro_f1": round(ood_f1, 4),
    "ood_drop_pct": round(drop, 1),
    "model": model_name,
    "label_col": label_col,
    "ood_method": "latitude_median_north_train_south_test",
}
with open("results/quick_pilot/summary_alphaearth_lgbm.json", "w") as f:
    json.dump(summary, f, indent=2)

# ── 8. 文本报告 ──
report = f"""AlphaEarth Boosting Pilot Report
================================
Samples: {n_samples} (max {MAX_PER_CLASS}/class)
Classes: {n_classes} ({label_col})
Features: {n_features} (A00-A63 mean+stdDev)
Model: {model_name}

ID  macro-F1: {id_f1:.4f} (80/20 random split)
OOD macro-F1: {ood_f1:.4f} (north-train / south-test)
DROP: {drop:.1f}%
"""
with open("results/quick_pilot/report_alphaearth_lgbm.txt", "w") as f:
    f.write(report)

# ── 9. 特征重要性 (permutation importance as fallback) ──
# HistGradientBoosting does not expose feature_importances_ in sklearn 0.24
try:
    importances = model.feature_importances_
    fi_source = "built-in"
except AttributeError:
    from sklearn.inspection import permutation_importance
    perm = permutation_importance(model, X_te, y_te, n_repeats=5, random_state=42, n_jobs=-1)
    importances = perm.importances_mean
    fi_source = "permutation"

fi_df = pd.DataFrame({"feature": feat_cols, "importance": importances})
fi_df = fi_df.sort_values("importance", ascending=False).head(20)
fi_df.to_csv("results/quick_pilot/featimp_alphaearth_lgbm.csv", index=False)
print(f"Feature importance ({fi_source}) saved.")

# ── 10. 最终输出 ──
print(f"\nPATH_JSON=results/quick_pilot/summary_alphaearth_lgbm.json")
print(f"ID_F1={id_f1:.4f}")
print(f"OOD_F1={ood_f1:.4f}")
print(f"DROP={drop:.1f}%")
print(f"MODEL={model_name}")
