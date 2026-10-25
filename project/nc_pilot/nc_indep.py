#!/usr/bin/env python3
"""nc_indep.py：独立实现 PILOT_PLAN_v4 第 6、7 节的分析，用于与 nc_analyze.py 交叉核对。

不 import analyze_confirm / step0B，全部度量从头实现：
  - τ 族：ℓ = log P − τ·log π̂，π̂ 为该臂测试集上 P 的列平均；π̂_c = 0 的类偏移取 0；
    P = 0 的类 log 为 −inf；argmax 并列取第一个；τ = −2..2，步长 0.05，共 81 点。
  - OA = 测试正确率 ×100；BA = BA 类别（splits/ba_classes.json）召回率的平均 ×100。
  - 上包络：相邻 τ 点在 (OA, BA) 平面线性插值；OA 相同的平段取两端 BA 较大者；
    多段经过同一 OA 时取最大。
  - 窗口：OA★ − 4 .. OA★，步长 0.1，共 41 点。可达性用整数精确判断
    （OA = 100·m/n，窗口点 = 100·m★/n − 0.1·j，同乘 10n 后比较 1000·m 与 1000·m★ − j·n）。
  - 两曲线之差：只在两曲线都可达的窗口点上求 BA 差再取平均；共同可达点 < 41/2 时缺失。
  - Δ_R、Δ_L(k)、缺失规则、I(k)、判定、L_AE（等权 PAV + 线性插值）按方案第 6、7 节。

用法：python nc_indep.py --root <含 removed.csv 与 probs/ 的目录> --out <输出目录>
输出：{out}/indep_domain.csv、{out}/indep_task.csv、{out}/indep_arm.csv（逐比较明细）
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")

import argparse
import json
import sys

import numpy as np
import pandas as pd

SPLITS = "/Users/perng_mini/AlphaCrops_local/nc_pilot/splits"
TAUS = np.linspace(-2.0, 2.0, 81)
I_TAU0 = 40
assert TAUS.size == 81 and TAUS[I_TAU0] == 0.0
NWIN = 41  # OA★−4 .. OA★，步长 0.1
KS = [50, 100, 200, 400]
KL = ["50", "100", "200", "400", "all"]
ARM_OF = {"50": "RT50", "100": "RT100", "200": "RT200", "400": "RT400", "all": "ALL"}
FEATS = ["ae", "s2"]
NCLS = {"T1": 17, "T2": 7}
T2_CODE = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "H": 6}
NAN = float("nan")


# ---------------------------------------------------------------- 数据

def load_task_meta(task, removed):
    """返回 {域: (期望测试编号集合, 编号->标签 dict)}，测试集按剔除前确定，再去掉被剔除样本。"""
    df = pd.read_parquet(f"{SPLITS}/{task.lower()}_samples.parquet")
    if task == "T1":
        dcol, scol = "city", "row"
        assert df["y"].between(0, 16).all()
    else:
        dcol, scol = "nuts0", "id"
        lab = df["cprn_lc"].str[0].map(T2_CODE)
        assert lab.notna().all() and (lab.astype(int) == df["y"]).all(), "T2 标签与 cprn_lc 不一致"
    assert df[scol].is_unique
    rem = removed[removed["task"] == task]
    rem_sid = set(rem[scol].dropna().astype(np.int64).tolist())
    return df, dcol, scol, rem_sid


def curve(P, y, ba_cls):
    """81 个 τ 点上的 (正确数 m, BA%)。"""
    P = np.asarray(P, dtype=np.float64)
    with np.errstate(divide="ignore"):
        logP = np.log(P)  # P = 0 -> -inf
    pi = P.mean(axis=0)
    logpi = np.zeros_like(pi)
    pos = pi > 0
    logpi[pos] = np.log(pi[pos])  # π̂_c = 0 的类偏移取 0
    masks = [(y == c) for c in ba_cls]
    for c, mk in zip(ba_cls, masks):
        if not mk.any():
            raise RuntimeError(f"BA 类别 {c} 在剔除后的测试集中没有样本")
    m = np.empty(TAUS.size, dtype=np.int64)
    ba = np.empty(TAUS.size, dtype=np.float64)
    for i, t in enumerate(TAUS):
        pred = np.argmax(logP - t * logpi, axis=1)  # 并列取第一个
        ok = pred == y
        m[i] = int(ok.sum())
        ba[i] = 100.0 * float(np.mean([ok[mk].mean() for mk in masks]))
    return m, ba


def envelope(m, ba, n, mstar):
    """在 41 个窗口点上求曲线可达的最大 BA；不可达为 NaN。整数精确判断可达性。"""
    U = 1000 * m.astype(np.int64)
    G = 1000 * int(mstar) - np.arange(NWIN, dtype=np.int64) * int(n)
    out = np.full(NWIN, np.nan)
    for i in range(U.size - 1):
        a, b = int(U[i]), int(U[i + 1])
        lo, hi = (a, b) if a <= b else (b, a)
        sel = (G >= lo) & (G <= hi)
        if not sel.any():
            continue
        if a == b:
            v = np.full(int(sel.sum()), max(ba[i], ba[i + 1]))
        else:
            t = (G[sel] - a) / float(b - a)
            v = ba[i] + t * (ba[i + 1] - ba[i])
        out[sel] = np.fmax(out[sel], v)
    return out


def curve_diff(envA, envB):
    both = ~np.isnan(envA) & ~np.isnan(envB)
    c = int(both.sum())
    if 2 * c < NWIN:  # 共同可达点不足一半
        return NAN, c
    return float(np.mean(envA[both] - envB[both])), c


def nanmean_or_nan(vals):
    v = [x for x in vals if not (x is None or np.isnan(x))]
    return float(np.mean(v)) if v else NAN


def pav(v):
    """等权保序回归（非降）。"""
    blocks = []
    for x in v:
        blocks.append([float(x), 1])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            s, c = blocks.pop()
            blocks[-1][0] += s
            blocks[-1][1] += c
    out = []
    for s, c in blocks:
        out += [s / c] * c
    return out


def k_eq(target, xs, g):
    if target > g[-1]:
        return "> 400"
    if target < g[0]:
        return "< 0"
    for i in range(len(g)):
        if g[i] >= target:
            if i == 0:
                return float(xs[0])
            return float(xs[i - 1] + (target - g[i - 1]) / (g[i] - g[i - 1]) * (xs[i] - xs[i - 1]))
    raise AssertionError("unreachable")


# ---------------------------------------------------------------- 主流程

def analyze_domain(root, task, dom, exp_sid, lab_of, ba_cls, arm_rows):
    ddir = os.path.join(root, "probs", task, dom)
    curves = {}  # (feat, arm, r) -> (m, ba)
    n_ref, sid_ref = None, None
    for feat in FEATS:
        for arm, rs in [("Z0", [0])] + [(f"RT{k}", [0, 1, 2]) for k in KS] + [("ALL", [0])]:
            for r in rs:
                fp = os.path.join(ddir, f"{feat}_{arm}_r{r}.npz")
                if not os.path.exists(fp):
                    print(f"[缺文件] {fp}", file=sys.stderr)
                    continue
                z = np.load(fp)
                P, sid = z["P_test"], z["sid_test"].astype(np.int64)
                assert P.ndim == 2 and P.shape[1] == NCLS[task] and P.shape[0] == sid.size, fp
                assert np.unique(sid).size == sid.size, f"{fp} 测试编号重复"
                if set(sid.tolist()) != exp_sid:
                    raise RuntimeError(f"{fp} 的测试编号与 splits 测试集减去 removed.csv 不一致"
                                       f"（文件 {sid.size}，期望 {len(exp_sid)}）")
                s_sorted = np.sort(sid)
                if sid_ref is None:
                    sid_ref, n_ref = s_sorted, sid.size
                assert np.array_equal(s_sorted, sid_ref)
                y = np.array([lab_of[s] for s in sid.tolist()], dtype=np.int64)
                curves[(feat, arm, r)] = curve(P, y, ba_cls)
    n = n_ref
    row = {"task": task, "dom": dom, "n_test": n}

    # Δ_R
    za, zs = curves.get(("ae", "Z0", 0)), curves.get(("s2", "Z0", 0))
    if za is not None and zs is not None:
        mstar = min(za[0][I_TAU0], zs[0][I_TAU0])
        dR, c = curve_diff(envelope(*za, n, mstar), envelope(*zs, n, mstar))
    else:
        dR, c = NAN, 0
    row["dR"] = dR
    row["cov_R"] = c / NWIN
    arm_rows.append(dict(task=task, dom=dom, feat="ae-s2", arm="Z0", r=0, n_common=c, diff=dR))

    # Δ_L(k)：每特征先对抽样取平均（有值者），再两特征取平均（任一缺失则缺失）
    dLf = {}
    for feat in FEATS:
        z0 = curves.get((feat, "Z0", 0))
        for kl in KL:
            arm = ARM_OF[kl]
            rs = [0] if kl == "all" else [0, 1, 2]
            vals = []
            for r in rs:
                cur = curves.get((feat, arm, r))
                if z0 is None or cur is None:
                    d, c = NAN, 0
                else:
                    mstar = z0[0][I_TAU0]
                    d, c = curve_diff(envelope(*cur, n, mstar), envelope(*z0, n, mstar))
                vals.append(d)
                arm_rows.append(dict(task=task, dom=dom, feat=feat, arm=arm, r=r, n_common=c, diff=d))
            dLf[(feat, kl)] = nanmean_or_nan(vals)
    for kl in KL:
        a, s = dLf[("ae", kl)], dLf[("s2", kl)]
        both = not (np.isnan(a) or np.isnan(s))
        row[f"dL_{kl}"] = (a + s) / 2.0 if both else NAN
        row[f"I_{kl}"] = a - s if both else NAN
    for feat in FEATS:
        for kl in KL:
            row[f"dL_{feat}_{kl}"] = dLf[(feat, kl)]
    row["all_ge_R"] = (row["dL_all"] >= dR) if not (np.isnan(row["dL_all"]) or np.isnan(dR)) else None
    row["all_lt_400"] = (row["dL_all"] < row["dL_400"]) if not (np.isnan(row["dL_all"]) or np.isnan(row["dL_400"])) else None
    return row


def analyze_task(task, drows):
    out = {"task": task}
    d = pd.DataFrame(drows)
    # 各量在自身有值的域上的平均（描述用）
    out["m_R"] = nanmean_or_nan(d["dR"].tolist())
    for kl in KL:
        out[f"m_L_{kl}"] = nanmean_or_nan(d[f"dL_{kl}"].tolist())
    for kl in KL:
        out[f"I_{kl}"] = nanmean_or_nan(d[f"I_{kl}"].tolist())
    # 判定：只用 Δ_R 与 Δ_L(200) 都有值的域
    dec = d[d["dR"].notna() & d["dL_200"].notna()]
    out["n_dec"] = len(dec)
    if len(dec) >= 2:
        mR, mL = float(dec["dR"].mean()), float(dec["dL_200"].mean())
        out["m_R_dec"], out["m_L_200_dec"] = mR, mL
        out["holds"] = bool(mR > 0 and mR >= mL)
    else:
        out["m_R_dec"], out["m_L_200_dec"] = NAN, NAN
        out["holds"] = False
    # L_AE：只用 Δ_R 与四个 Δ_L(k) 全部有值的域
    eq = d[d["dR"].notna() & d[[f"dL_{k}" for k in KS]].notna().all(axis=1)]
    out["n_eq"] = len(eq)
    if len(eq) >= 2:
        mR = float(eq["dR"].mean())
        gains = [0.0] + [float(eq[f"dL_{k}"].mean()) for k in KS]
        g = pav(gains)
        out["m_R_eq"] = mR
        out["gains_eq"] = ";".join(repr(x) for x in gains)
        out["pav_eq"] = ";".join(repr(x) for x in g)
        out["L_AE"] = k_eq(mR, [0, 50, 100, 200, 400], g)
    else:
        out["m_R_eq"], out["gains_eq"], out["pav_eq"] = NAN, "", ""
        out["L_AE"] = "不可计算"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    targets = json.load(open(f"{SPLITS}/targets.json"))
    ba_classes = json.load(open(f"{SPLITS}/ba_classes.json"))
    removed = pd.read_csv(os.path.join(a.root, "removed.csv"))

    drows_all, trows, arm_rows = [], [], []
    for task in ["T1", "T2"]:
        df, dcol, scol, rem_sid = load_task_meta(task, removed)
        drows = []
        for dom in targets[task]:
            sub = df[df[dcol] == dom]
            test = sub[sub["role"] == "test"]
            exp_sid = set(test[scol].astype(np.int64).tolist()) - rem_sid
            lab_of = dict(zip(sub[scol].astype(np.int64).tolist(), sub["y"].astype(np.int64).tolist()))
            ba_cls = [int(c) for c in ba_classes[dom]]
            drows.append(analyze_domain(a.root, task, dom, exp_sid, lab_of, ba_cls, arm_rows))
        drows_all += drows
        trows.append(analyze_task(task, drows))

    nh = sum(bool(t["holds"]) for t in trows)
    decision = {2: "GO", 1: "部分 GO", 0: "STOP"}[nh]
    for t in trows:
        t["decision"] = decision

    pd.DataFrame(drows_all).to_csv(os.path.join(a.out, "indep_domain.csv"), index=False)
    pd.DataFrame(trows).to_csv(os.path.join(a.out, "indep_task.csv"), index=False)
    pd.DataFrame(arm_rows).to_csv(os.path.join(a.out, "indep_arm.csv"), index=False)
    for t in trows:
        print(f"{t['task']}: n_dec={t['n_dec']} m_R={t['m_R_dec']:.4f} m_L(200)={t['m_L_200_dec']:.4f} "
              f"holds={t['holds']}  L_AE={t['L_AE']}")
    print(f"决策：{decision}")
    print(f"输出：{a.out}")


if __name__ == "__main__":
    main()
