"""
RF重跑实验 - 沿用alphaearth_large配置
配置：RF(n_estimators=300), label_tier_a, MAX_PER_CLASS=300, MIN_PER_CLASS=5
OOD: 纬度中位数 北训南测
"""
import json, warnings, os, sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
np.random.seed(42)

# 切换到项目根目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
os.chdir(PROJECT_ROOT)

OUT_DIR = SCRIPT_DIR
os.makedirs(OUT_DIR, exist_ok=True)

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

# ── 4. 采样：每类最多300，min 5 ──
MAX_PER_CLASS = 300
MIN_PER_CLASS = 5
MAX_TOTAL = 10000
sampled = []
for cls, grp in df.groupby(label_col):
    n = min(len(grp), MAX_PER_CLASS)
    sampled.append(grp.sample(n=n, random_state=42))
df_s = pd.concat(sampled).reset_index(drop=True)
if len(df_s) > MAX_TOTAL:
    df_s = df_s.sample(n=MAX_TOTAL, random_state=42).reset_index(drop=True)

# 过滤样本数<MIN_PER_CLASS的类
cls_counts = df_s[label_col].value_counts()
valid_cls = cls_counts[cls_counts >= MIN_PER_CLASS].index
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
RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": None,
    "min_samples_leaf": 5,
    "class_weight": "balanced",
    "n_jobs": -1,
    "random_state": 42,
}
model_name = f"RandomForest(n_estimators={RF_PARAMS['n_estimators']})"

# ── 6a. ID评估 (80/20 random) ──
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
clf_id = RandomForestClassifier(**RF_PARAMS)
clf_id.fit(X_tr, y_tr)
y_pred_id = clf_id.predict(X_te)
id_f1 = f1_score(y_te, y_pred_id, average="macro")

# ── 6b. OOD评估 (纬度中位数：北训南测) ──
lat_median = np.median(lat)
north = lat >= lat_median
south = ~north
clf_ood = RandomForestClassifier(**RF_PARAMS)
clf_ood.fit(X[north], y[north])
y_pred_ood = clf_ood.predict(X[south])
ood_f1 = f1_score(y[south], y_pred_ood, average="macro")

drop = (id_f1 - ood_f1) / id_f1 * 100 if id_f1 > 0 else 0.0

print(f"ID  F1={id_f1:.4f}")
print(f"OOD F1={ood_f1:.4f}")
print(f"DROP={drop:.1f}%")

# ── 7. JSON输出 ──
summary = {
    "experiment": "rf_rerun_20260305_1025",
    "n_samples": n_samples,
    "n_classes": n_classes,
    "n_features": n_features,
    "max_per_class": MAX_PER_CLASS,
    "min_per_class": MIN_PER_CLASS,
    "id_macro_f1": round(id_f1, 4),
    "ood_macro_f1": round(ood_f1, 4),
    "ood_drop_pct": round(drop, 1),
    "model": model_name,
    "label_col": label_col,
    "ood_method": "latitude_median_north_train_south_test",
    "rf_params": RF_PARAMS,
}
json_path = os.path.join(OUT_DIR, "summary_rf_rerun.json")
with open(json_path, "w") as f:
    json.dump(summary, f, indent=2)

# ── 8. 文本报告 ──
report = f"""============================================================
RF Rerun Experiment Report (20260305_1025)
============================================================

Samples:    {n_samples}
Classes:    {n_classes} ({label_col})
Max/class:  {MAX_PER_CLASS}
Min/class:  {MIN_PER_CLASS}
Features:   {n_features}
Model:      {model_name}

ID  macro-F1:  {id_f1:.4f} (80/20 random split)
OOD macro-F1:  {ood_f1:.4f} (north-train / south-test)
DROP:          {drop:.1f}%

--- Comparison with alphaearth_large baseline ---
  Baseline ID:  0.5009
  Baseline OOD: 0.2854
  Baseline DROP: 43.0%
  Delta ID:  {id_f1 - 0.5009:+.4f}
  Delta OOD: {ood_f1 - 0.2854:+.4f}
  Delta DROP: {drop - 43.0:+.1f}pp
============================================================
"""
report_path = os.path.join(OUT_DIR, "report_rf_rerun.txt")
with open(report_path, "w") as f:
    f.write(report)
print(report)

# ── 9. 特征重要性 ──
importances = clf_ood.feature_importances_
fi_df = pd.DataFrame({"feature": feat_cols, "importance": importances})
fi_df = fi_df.sort_values("importance", ascending=False)
fi_df.to_csv(os.path.join(OUT_DIR, "featimp_rf_rerun.csv"), index=False)
print(f"Top 10 features:")
print(fi_df.head(10).to_string(index=False))

# ── 10. 最终输出 ──
print(f"\nPATH_JSON={json_path}")
print(f"ID_F1={id_f1:.4f}")
print(f"OOD_F1={ood_f1:.4f}")
print(f"DROP={drop:.1f}%")
print(f"MODEL={model_name}")
