#!/usr/bin/env python3
"""快速基线: 最近中心分类器, 仅标准库."""
import csv, json, math, os, random
from collections import Counter, defaultdict

random.seed(42)

# ── 读取CSV ──────────────────────────────────────────────────
def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

FILES = {
    "DE": "14094196/de_bb_2023.csv",
    "BE": "14094196/be_vlg_2021.csv",
    "ES": "14094196/es_2023.csv",
}

data = {k: read_csv(v) for k, v in FILES.items()}

# ── 自动识别标签列 (优先 HCAT2 > HCAT3 > class > crop > label) ──
LABEL_PRIORITY = ["HCAT2_name", "HCAT3_name", "class", "crop", "label"]

def find_label_col(rows):
    cols = set(rows[0].keys())
    for cand in LABEL_PRIORITY:
        for c in cols:
            if c.lower() == cand.lower():
                return c
    raise ValueError(f"无法识别标签列, 列: {cols}")

# 找所有文件共有的标签列 (取最高优先级且所有文件都有的)
all_cols = [set(r.lower() for r in data[k][0].keys()) for k in data]
label_col = None
for cand in LABEL_PRIORITY:
    if all(cand.lower() in cs for cs in all_cols):
        # 取原始大小写
        label_col = cand
        break
if label_col is None:
    # fallback: 每文件各自选
    label_col = find_label_col(data["DE"])
    print(f"警告: 无公共标签列, 回退至 {label_col}")

# 用实际列名匹配 (大小写)
def get_label_col_name(rows, target):
    for c in rows[0].keys():
        if c.lower() == target.lower():
            return c
    return target

label_cols = {k: get_label_col_name(v, label_col) for k, v in data.items()}
print(f"标签列: {label_col}")

# ── 特征: original_code (数值化) ──────────────────────────────
def to_float(s):
    try:
        return float(s)
    except (ValueError, TypeError):
        return None

def make_samples(rows, lbl_col):
    samples = []
    for row in rows:
        val = to_float(row.get("original_code"))
        if val is None:
            continue
        label = row[lbl_col]
        if not label:
            continue
        samples.append((val, label))
    return samples

samples = {k: make_samples(v, label_cols[k]) for k, v in data.items()}

# ── 每国每类最多80 ──────────────────────────────────────────
def subsample(samp, max_per_class=80):
    by_cls = defaultdict(list)
    for feat, lbl in samp:
        by_cls[lbl].append((feat, lbl))
    out = []
    for lbl, items in by_cls.items():
        random.shuffle(items)
        out.extend(items[:max_per_class])
    return out

for k in samples:
    samples[k] = subsample(samples[k], 80)

print(f"样本数  DE={len(samples['DE'])}  BE={len(samples['BE'])}  ES={len(samples['ES'])}")

# ── 最近中心分类器 ────────────────────────────────────────────
def compute_centroids(samp):
    sums = defaultdict(float)
    counts = defaultdict(int)
    for feat, lbl in samp:
        sums[lbl] += feat
        counts[lbl] += 1
    return {lbl: sums[lbl] / counts[lbl] for lbl in sums}

def predict(centroids, feat):
    best_lbl, best_dist = None, float("inf")
    for lbl, c in centroids.items():
        d = abs(feat - c)
        if d < best_dist:
            best_dist, best_lbl = d, lbl
    return best_lbl

# ── Macro-F1 ──────────────────────────────────────────────────
def macro_f1(y_true, y_pred):
    labels = sorted(set(y_true) | set(y_pred))
    f1s = []
    for lbl in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == lbl and p == lbl)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != lbl and p == lbl)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == lbl and p != lbl)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    return sum(f1s) / len(f1s) if f1s else 0.0

# ── 训练 & 评估 ──────────────────────────────────────────────
train_data = samples["DE"] + samples["BE"]
test_ood = samples["ES"]

centroids = compute_centroids(train_data)

# ID: 训练集自身评估
y_true_id = [lbl for _, lbl in train_data]
y_pred_id = [predict(centroids, f) for f, _ in train_data]
id_f1 = macro_f1(y_true_id, y_pred_id)

# OOD: ES测试
y_true_ood = [lbl for _, lbl in test_ood]
y_pred_ood = [predict(centroids, f) for f, _ in test_ood]
ood_f1 = macro_f1(y_true_ood, y_pred_ood)

print(f"ID  macro-F1 = {id_f1:.4f}")
print(f"OOD macro-F1 = {ood_f1:.4f}")

# ── 写结果 ────────────────────────────────────────────────────
result = {
    "label_column": label_col,
    "n_train": len(train_data),
    "n_test_ood": len(test_ood),
    "n_classes_train": len(centroids),
    "n_classes_test": len(set(y_true_ood)),
    "ID_macro_F1": round(id_f1, 4),
    "OOD_macro_F1": round(ood_f1, 4),
    "method": "nearest_centroid",
    "feature": "original_code",
    "max_per_class": 80,
}

out_path = "results/quick_pilot/summary_from_claude.json"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

print(f"\n结果已写入 {out_path}")
print("OK")
