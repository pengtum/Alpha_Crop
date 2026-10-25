import os, json, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore")

LABEL_PRIORITY = ["HCAT2", "HCAT3", "class", "crop", "label"]
FILES = {
    "DE": "14094196/de_bb_2023.csv",
    "BE": "14094196/be_vlg_2021.csv",
    "ES": "14094196/es_2023.csv",
}
MAX_PER_CLASS = 100
OUT_PATH = "results/quick_pilot/summary_from_claude.json"


def find_label_col(df):
    for c in LABEL_PRIORITY:
        matches = [col for col in df.columns if col.strip().lower() == c.lower()]
        if matches:
            return matches[0]
    for col in df.columns:
        if df[col].dtype == object and df[col].nunique() < 500:
            return col
    return None


def load_and_prepare(path, country):
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as e:
        print(f"[WARN] Cannot read {path}: {e}")
        return None, None, None
    label_col = find_label_col(df)
    if label_col is None:
        print(f"[WARN] No label column found in {path}")
        return None, None, None
    df = df.dropna(subset=[label_col])
    df[label_col] = df[label_col].astype(str)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        print(f"[WARN] No numeric features in {path}")
        return None, None, None
    X = df[num_cols].copy()
    y = df[label_col].copy()
    X = X.fillna(0)
    sampled = (
        pd.DataFrame({"y": y, "idx": np.arange(len(y))})
        .groupby("y")
        .apply(lambda g: g.sample(n=min(len(g), MAX_PER_CLASS), random_state=42), include_groups=False)
        .reset_index(drop=True)
    )
    idx = sampled["idx"].values
    return X.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True), label_col


def align_features(*dfs):
    common = set(dfs[0].columns)
    for d in dfs[1:]:
        common &= set(d.columns)
    common = sorted(common)
    if not common:
        return [pd.DataFrame()] * len(dfs)
    return [d[common] for d in dfs]


def main():
    data = {}
    label_cols = {}
    for country, path in FILES.items():
        X, y, lc = load_and_prepare(path, country)
        if X is not None and len(X) > 0:
            data[country] = (X, y)
            label_cols[country] = lc

    if not data:
        os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
        json.dump({"error": "No valid data loaded"}, open(OUT_PATH, "w"), indent=2)
        return

    results = {}

    # --- ID: mix all available, random split ---
    all_Xs, all_ys = [], []
    for country in data:
        all_Xs.append(data[country][0])
        all_ys.append(data[country][1])
    if all_Xs:
        aligned = align_features(*all_Xs)
        if aligned[0].shape[1] > 0:
            X_all = pd.concat(aligned, ignore_index=True).fillna(0)
            y_all = pd.concat(all_ys, ignore_index=True)
            mask = y_all.isin(y_all.value_counts()[y_all.value_counts() >= 2].index)
            X_all, y_all = X_all[mask].reset_index(drop=True), y_all[mask].reset_index(drop=True)
            if len(X_all) > 10:
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
                )
                clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                clf.fit(X_tr, y_tr)
                id_f1 = f1_score(y_te, clf.predict(X_te), average="macro", zero_division=0)
                results["id_macro_f1"] = round(float(id_f1), 4)

    # --- OOD: train DE+BE, test ES ---
    ood_f1 = None
    train_countries = [c for c in ["DE", "BE"] if c in data]
    if train_countries and "ES" in data:
        train_Xs = [data[c][0] for c in train_countries]
        train_ys = [data[c][1] for c in train_countries]
        test_X, test_y = data["ES"]
        all_parts = train_Xs + [test_X]
        aligned = align_features(*all_parts)
        if aligned[0].shape[1] > 0:
            X_tr = pd.concat(aligned[: len(train_Xs)], ignore_index=True).fillna(0)
            y_tr = pd.concat(train_ys, ignore_index=True)
            X_te = aligned[-1].fillna(0)
            y_te = test_y
            common_labels = set(y_tr.unique()) & set(y_te.unique())
            if common_labels:
                mask_tr = y_tr.isin(common_labels)
                mask_te = y_te.isin(common_labels)
                X_tr, y_tr = X_tr[mask_tr].reset_index(drop=True), y_tr[mask_tr].reset_index(drop=True)
                X_te, y_te = X_te[mask_te].reset_index(drop=True), y_te[mask_te].reset_index(drop=True)
                if len(X_tr) > 0 and len(X_te) > 0:
                    clf2 = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                    clf2.fit(X_tr, y_tr)
                    ood_f1 = f1_score(y_te, clf2.predict(X_te), average="macro", zero_division=0)
                    results["ood_macro_f1"] = round(float(ood_f1), 4)

    if "id_macro_f1" in results and "ood_macro_f1" in results:
        results["drop"] = round(results["id_macro_f1"] - results["ood_macro_f1"], 4)

    results["label_cols"] = label_cols
    results["countries_loaded"] = list(data.keys())
    results["n_samples"] = {c: len(data[c][0]) for c in data}

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
