# 只读计数（不训练）：dataset_v2 各国各类地块数，以及 G 测试集（s0/s1）各类地块数。
# 用途：PAPER_PLAN §4 数据表、D2（Cheng 可比 4 类子集）的样本稀疏标注。
import os
for v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[v] = "1"
import json
import numpy as np
import pandas as pd

CACHE = "/Users/perng_mini/AlphaCrops_local/data/cache_guidance_G"
ST0 = "/Users/perng_mini/AlphaCrops_local/results/guidance_G/stage0"
OUT = os.path.dirname(os.path.abspath(__file__))
y = np.load(f"{CACHE}/y.npy")
c = np.load(f"{CACHE}/ctry.npy")
meta = json.load(open(f"{CACHE}/meta.json"))
CL, CT = meta["classes"], meta["countries"]

tot = pd.DataFrame([np.bincount(y[c == t], minlength=10) for t in CT], index=CT, columns=CL)
tot["n_total"] = tot.sum(1)
tot["n_classes_present"] = (tot[CL] > 0).sum(1)
tot.to_csv(f"{OUT}/dataset_v2_class_counts.csv")

rows = []
for t in CT:
    for s in (0, 1):
        S = np.load(f"{ST0}/{t}_s{s}.npz")
        b = np.bincount(y[S["test"]], minlength=10)
        rows.append(dict(country=t, split=s, n_test=len(S["test"]), n_cand=len(S["cand"]),
                         n_foreign=len(S["foreign"]), n_big_ge30=int((b >= 30).sum()),
                         **{f"test_{k}": int(v) for k, v in zip(CL, b)}))
d = pd.DataFrame(rows)
d.to_csv(f"{OUT}/G_test_class_counts.csv", index=False)
four = ["grassland", "maize", "potatoes", "winter_cereals"]
print("dataset_v2 总地块", int(tot.n_total.sum()))
print(tot.to_string())
print(d[["country", "split", "n_test", "n_cand", "n_big_ge30"] + [f"test_{k}" for k in four]].to_string(index=False))
print("n_test 范围", d.n_test.min(), d.n_test.max(), "；n_cand 范围", d.n_cand.min(), d.n_cand.max())
