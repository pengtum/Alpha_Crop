"""扩大样本 + OOD 实验（AlphaEarth medium pilot）"""
import json, os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

SEED = 42
MAX_PER_CLASS = 100
MAX_TOTAL = 10000

# --- 读取数据 ---
base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
emb = pd.read_parquet(os.path.join(base, "data/processed/parcels_filtered_embeddings_interactive.parquet"))
par = pd.read_parquet(os.path.join(base, "data/processed/parcels_filtered.parquet"))

# 合并标签和坐标
par_sub = par.loc[emb["parcel_idx"], ["label_tier_a", "latitude"]].reset_index(drop=True)
df = pd.concat([emb.reset_index(drop=True), par_sub], axis=1)

# --- 特征列 ---
feat_cols = [c for c in df.columns if c.startswith("A") and ("mean" in c or "stdDev" in c)]
label_col = "label_tier_a"
print(f"Features: {len(feat_cols)}, Label: {label_col}, Raw N: {len(df)}")

# 去除缺失
df = df.dropna(subset=feat_cols + [label_col, "latitude"])

# --- 采样：每类最多 MAX_PER_CLASS，总量不超 MAX_TOTAL ---
sampled = df.groupby(label_col, group_keys=False).apply(
    lambda g: g.sample(n=min(len(g), MAX_PER_CLASS), random_state=SEED)
)
if len(sampled) > MAX_TOTAL:
    sampled = sampled.sample(n=MAX_TOTAL, random_state=SEED)

# 过滤掉样本数<2的类（stratify需要）
cls_counts = sampled[label_col].value_counts()
valid_cls = cls_counts[cls_counts >= 2].index
sampled = sampled[sampled[label_col].isin(valid_cls)]
print(f"Sampled N: {len(sampled)}, Classes: {sampled[label_col].nunique()}")

X = sampled[feat_cols].values
y = sampled[label_col].values
lat = sampled["latitude"].values

# --- ID split (80/20 random) ---
X_tr, X_te, y_tr, y_te, lat_tr, lat_te = train_test_split(
    X, y, lat, test_size=0.2, random_state=SEED, stratify=y
)

rf = RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)
rf.fit(X_tr, y_tr)
id_f1 = f1_score(y_te, rf.predict(X_te), average="macro")
print(f"ID macro-F1: {id_f1:.4f}")

# --- OOD split（纬度中位数：北训练 南测试）---
lat_median = np.median(lat)
north = lat >= lat_median
south = ~north

X_north, y_north = X[north], y[north]
X_south, y_south = X[south], y[south]

rf_ood = RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)
rf_ood.fit(X_north, y_north)
ood_f1 = f1_score(y_south, rf_ood.predict(X_south), average="macro")
drop = (id_f1 - ood_f1) / id_f1 * 100
print(f"OOD macro-F1: {ood_f1:.4f}, DROP: {drop:.1f}%")

# --- 保存结果 ---
out_dir = os.path.join(base, "results/quick_pilot")
os.makedirs(out_dir, exist_ok=True)

summary = {
    "experiment": "alphaearth_medium",
    "n_samples": int(len(sampled)),
    "n_classes": int(sampled[label_col].nunique()),
    "n_features": len(feat_cols),
    "max_per_class": MAX_PER_CLASS,
    "id_macro_f1": round(id_f1, 4),
    "ood_macro_f1": round(ood_f1, 4),
    "ood_drop_pct": round(drop, 1),
    "model": "RandomForest(n_estimators=200)",
    "label_col": label_col,
    "ood_method": "latitude_median_north_train_south_test",
}

json_path = os.path.join(out_dir, "summary_alphaearth_medium.json")
with open(json_path, "w") as f:
    json.dump(summary, f, indent=2)

report = f"""AlphaEarth Medium Pilot Report
==============================
Samples: {len(sampled)} (max {MAX_PER_CLASS}/class)
Classes: {sampled[label_col].nunique()} ({label_col})
Features: {len(feat_cols)} (A00-A63 mean+stdDev)
Model: RandomForest(200 trees)

ID  macro-F1: {id_f1:.4f} (80/20 random split)
OOD macro-F1: {ood_f1:.4f} (north-train / south-test)
DROP: {drop:.1f}%
"""
txt_path = os.path.join(out_dir, "report_alphaearth_medium.txt")
with open(txt_path, "w") as f:
    f.write(report)

print(f"\nPATH_JSON=results/quick_pilot/summary_alphaearth_medium.json")
print(f"ID_F1={id_f1:.4f}")
print(f"OOD_F1={ood_f1:.4f}")
print(f"DROP={drop:.1f}%")
print(f"N={len(sampled)}")
