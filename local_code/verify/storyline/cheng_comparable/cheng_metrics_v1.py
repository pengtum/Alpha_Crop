"""
与 Cheng et al. 2026（RSE 347:115694）可比的口径：ANALYSIS_PLAN_guidance_rev4.md 第 D2 节（补充材料，描述）。
只读已有测试集概率，不训练任何模型。

指标（每个 学习器 × 特征 × 单元）：
  (a) ba_all：测试集中出现的所有类（≥1 个地块）的平均召回；另附 oa 与 ba_ge30（≥30 个地块的类）用于与 rows 核对。
  (b) 4 类子任务 {grassland, maize, potatoes, winter_cereals}：只保留真实类属于这 4 类的测试地块；
      概率只取这 4 列并重新归一化后 argmax；报告 oa4 与 ba4（子集中出现的类的平均召回）以及逐类召回。
对象：k=0（只用外国数据）为主；SRS k=200 为补充。AE 特征 − S2 特征，11 国 t 检验（df=10）与去掉 RO、PT 的版本。

数据来源（学习器 → 概率来源的映射，--src NAME=KIND:ROOT，可重复）：
  KIND=G ：guidance_G.py 的布局。k=0 取 stage0/{国家}_s{划分}.npz 的 P0test_{特征}_i{最大轮}（float16，以此为准）；
           SRS k 取 rows_t1_{单元}.csv 中 method=srs 的 sel_hash → probs/{单元}_{特征}_k{k}_{哈希}.npz 的最大轮。
  KIND=ML：multilearner_G.py 的布局。k=0 取 probs/{NAME}/{国家}_s{划分}_{特征}_k0.npz 的 'P'；
           SRS k 取 rows_{NAME}_{单元}.csv（status=ok）的 sel_hash → probs/{NAME}/{单元}_{特征}_k{k}_{哈希}.npz 的 'P'。
  默认只有 lgbm=G:<guidance_G.OUT>。rev4 规定：LR/MLP/RF 必须等 rev2 全部完成、analyze_ML.py 正式运行之后才可计算；
  因此任何 KIND=ML 的来源、以及任何位于 guidance_G_ml/ 下的路径，都必须显式加 --ml-unblinded 才会被读取。

用法（必须能 import label_expansion 下的模块；脚本自己把该目录加入 sys.path，且不写 .pyc）：
  python cheng_metrics.py                                   # 只算 LightGBM（G）
  python cheng_metrics.py --src lgbm=G:/…/guidance_G --src rf=ML:/…/guidance_G_ml --ml-unblinded   # rev2 解盲之后
输出（--out，默认本脚本所在目录）：cheng_metrics_rows.csv、cheng_metrics_counts.csv、cheng_metrics_summary.txt
"""
import os
import sys

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
           'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '2')  # 硬性规则：线程 ≤2
sys.dont_write_bytecode = True  # 不在项目目录写 __pycache__

import argparse  # noqa: E402
import glob  # noqa: E402
import hashlib  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

L_DIR = ('/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops/'
         'label_expansion')
sys.path.insert(0, L_DIR)
import guidance_G as G  # noqa: E402
from analyze_G import t_summary  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PLAN = os.path.join(L_DIR, 'ANALYSIS_PLAN_guidance_rev4.md')
PLAN_SHA16 = '7e90ccd5b1c3828f'  # plan_hashes.txt 2026-09-27 14:48
MIN_N = G.MIN_N  # 30
C4 = (0, 1, 4, 9)  # grassland, maize, potatoes, winter_cereals（对应 Cheng 的 MFP、玉米、马铃薯、冬小麦+冬大麦）
DROP = ('RO', 'PT')
FEATS = ('ae', 's2')
KS = (0, 200)  # 0 = 只用外国数据（主）；200 = SRS（补充）
ML_BLIND_DIR = 'guidance_G_ml'
EXPECTED_CLASSES = ['grassland', 'maize', 'orchards', 'other_crops', 'potatoes', 'root_crops', 'strawberries',
                    'tree_wood_forest', 'vegetables', 'winter_cereals']


# ---------------------------------------------------------------- 盲态保护
class BlindGuard:
    """rev4：rev2 多学习器结果解盲之前，任何 guidance_G_ml/ 下的路径都不得读取。"""
    unblinded = False

    @classmethod
    def check(cls, path):
        parts = os.path.normpath(os.path.abspath(path)).split(os.sep)
        if ML_BLIND_DIR in parts and not cls.unblinded:
            raise SystemExit(f'拒绝读取 {path}：rev2 多学习器结果未解盲（rev4 D2）。解盲后加 --ml-unblinded。')
        return path


def _load_npz(path):
    return np.load(BlindGuard.check(path))


def _read_csv(path):
    return pd.read_csv(BlindGuard.check(path))


# ---------------------------------------------------------------- 概率来源
class GSource:
    """guidance_G.py 的输出布局（LightGBM）。"""
    kind = 'G'

    def __init__(self, name, root):
        self.name, self.root = name, BlindGuard.check(root)
        fs = sorted(glob.glob(os.path.join(root, 'rows_t1_*.csv')))
        assert fs, f'{root} 下没有 rows_t1_*.csv'
        R = pd.concat([_read_csv(f) for f in fs], ignore_index=True)
        self.rows = R[R.method.isin(['foreign_only', 'srs'])].copy()

    def k0(self, tgt, split, feat):
        path = os.path.join(self.root, 'stage0', f'{tgt}_s{split}.npz')
        z = _load_npz(path)
        pre = f'P0test_{feat}_i'
        it = max(int(k[len(pre):]) for k in z.files if k.startswith(pre))
        return z[f'{pre}{it}'], f'stage0/{tgt}_s{split}.npz:{pre}{it}', ''

    def srs(self, tgt, split, rep, feat, k):
        r = self._row('srs', tgt, split, rep, feat, k)
        h = r['sel_hash']
        rel = f'probs/{tgt}_s{split}_r{rep}_{feat}_k{k}_{h}.npz'
        z = _load_npz(os.path.join(self.root, rel))
        it = max(int(key[1:]) for key in z.files)
        return z[f'i{it}'], f'{rel}:i{it}', h

    def _row(self, method, tgt, split, rep, feat, k):
        R = self.rows
        r = R[(R.method == method) & (R.target == tgt) & (R.split == split) & (R.rep == rep) & (R.clf == feat) &
              (R.k == k)]
        assert len(r) == 1, (method, tgt, split, rep, feat, k, len(r))
        return r.iloc[0]

    def reference(self, method, tgt, split, rep, feat, k):
        """rows 中已记录的 oa、ba_ge30，用来核对本脚本的重算（应逐位一致）。"""
        r = self._row(method, tgt, split, rep, feat, k)
        return float(r['oa']), float(r['ba_ge30'])


class MLSource(GSource):
    """multilearner_G.py 的输出布局（LR/MLP/RF）。只在 --ml-unblinded 时允许构造。"""
    kind = 'ML'

    def __init__(self, name, root):
        if not BlindGuard.unblinded:
            raise SystemExit(f'学习器 {name}：rev2 结果未解盲，不得读取（rev4 D2）。')
        self.name, self.root = name, BlindGuard.check(root)
        fs = sorted(glob.glob(os.path.join(root, f'rows_{name}_*.csv')))
        assert fs, f'{root} 下没有 rows_{name}_*.csv'
        R = pd.concat([_read_csv(f) for f in fs], ignore_index=True)
        if 'status' in R:
            R = R[R.status == 'ok']
        self.rows = R[R.method.isin(['foreign_only', 'srs'])].copy()

    def k0(self, tgt, split, feat):
        rel = f'probs/{self.name}/{tgt}_s{split}_{feat}_k0.npz'
        return _load_npz(os.path.join(self.root, rel))['P'], f'{rel}:P', 'k0'

    def srs(self, tgt, split, rep, feat, k):
        h = self._row('srs', tgt, split, rep, feat, k)['sel_hash']
        rel = f'probs/{self.name}/{tgt}_s{split}_r{rep}_{feat}_k{k}_{h}.npz'
        return _load_npz(os.path.join(self.root, rel))['P'], f'{rel}:P', h


KINDS = {'G': GSource, 'ML': MLSource}


def parse_sources(specs):
    """['lgbm=G:/path', 'rf=ML:/path'] → {名字: 来源对象}。"""
    out = {}
    for s in specs:
        name, rest = s.split('=', 1)
        kind, root = rest.split(':', 1)
        assert kind in KINDS, f'未知来源类型 {kind}'
        assert name not in out, f'学习器 {name} 重复'
        out[name] = KINDS[kind](name, root)
    return out


# ---------------------------------------------------------------- 指标
def _recalls(y, ph, classes):
    return np.array([(ph[y == c] == c).mean() if (y == c).any() else np.nan for c in classes])


def cheng_scores(P, y):
    """P：(n, 10) 测试集概率；y：真实类。返回 D2 的全部指标（百分数）与诊断量。"""
    P = np.asarray(P, dtype=np.float64)
    P = P / P.sum(1, keepdims=True)  # 与 guidance_G.evaluate 相同
    ph = P.argmax(1)
    cnt = np.bincount(y, minlength=10)
    rec = _recalls(y, ph, range(10))
    srt = np.sort(P, 1)
    out = dict(n_test=len(y), n_cls_present=int((cnt > 0).sum()), n_cls_ge30=int((cnt >= MIN_N).sum()),
               ties10=int((srt[:, -1] == srt[:, -2]).sum()),
               oa=100 * float((ph == y).mean()),
               ba_all=100 * float(np.nanmean(rec[cnt > 0])),
               ba_ge30=100 * float(np.nanmean(rec[cnt >= MIN_N])))
    # (b) 4 类子任务
    m = np.isin(y, C4)
    y4 = y[m]
    P4 = P[m][:, list(C4)]
    s4 = P4.sum(1, keepdims=True)
    zero = (s4[:, 0] <= 0)
    P4 = np.divide(P4, s4, out=np.zeros_like(P4), where=s4 > 0)
    top = np.sort(P4, 1)
    ties = (top[:, -1] == top[:, -2])  # 含 4 列全为 0 的行（float16 下溢）；argmax 取第一个（grassland）
    ph4 = np.asarray(C4)[P4.argmax(1)]
    cnt4 = np.bincount(y4, minlength=10)[list(C4)]
    rec4 = _recalls(y4, ph4, C4)
    out.update(n_test4=int(m.sum()), n_cls4_present=int((cnt4 > 0).sum()),
               oa4=100 * float((ph4 == y4).mean()),
               ba4=100 * float(np.nanmean(rec4[cnt4 > 0])),
               ties4=int(ties.sum()), zero4=int(zero.sum()),
               **{f'rec4_{EXPECTED_CLASSES[c]}': 100 * r for c, r in zip(C4, rec4)},
               **{f'n4_{EXPECTED_CLASSES[c]}': int(n) for c, n in zip(C4, cnt4)})
    return out


# ---------------------------------------------------------------- 计算
def compute_rows(D, sources, countries):
    rows, counts = [], []
    for tgt in countries:
        for split in G.SPLITS:
            S = _load_npz(os.path.join(G.OUT, 'stage0', f'{tgt}_s{split}.npz'))
            y = np.asarray(D.y[S['test']])
            cnt = np.bincount(y, minlength=10)
            counts.append(dict(target=tgt, split=split, **{EXPECTED_CLASSES[c]: int(cnt[c]) for c in range(10)}))
            for name, src in sources.items():
                for feat in FEATS:
                    jobs = [('foreign_only', 0, -1)] + [('srs', k, rep) for k in KS if k > 0 for rep in G.REPS]
                    for method, k, rep in jobs:
                        if k == 0:
                            P, file, h = src.k0(tgt, split, feat)
                        else:
                            P, file, h = src.srs(tgt, split, rep, feat, k)
                        assert P.shape == (len(y), 10), (name, tgt, split, feat, k, P.shape)
                        m = cheng_scores(P, y)
                        ref_oa, ref_ba = src.reference(method, tgt, split, 0 if rep < 0 else rep, feat, k)
                        ok = abs(ref_oa - m['oa']) < 1e-9 and abs(ref_ba - m['ba_ge30']) < 1e-9
                        rows.append(dict(learner=name, method=method, k=k, feat=feat, target=tgt, split=split, rep=rep,
                                         sel_hash=h, source=file, ref_oa=ref_oa, ref_ba_ge30=ref_ba, ref_match=ok, **m))
    return pd.DataFrame(rows), pd.DataFrame(counts)


def country_diff(R, learner, k, metric, drop=()):
    """AE − S2：先在单元内配对相减，再在国家内对单元（k=0：两个划分；SRS：划分 × 重复）平均。"""
    sub = R[(R.learner == learner) & (R.k == k) & ~R.target.isin(drop)]
    w = sub.pivot_table(index=['target', 'split', 'rep'], columns='feat', values=metric)
    d = (w['ae'] - w['s2']).dropna()
    return d.groupby('target').mean(), w.groupby('target').mean()


# ---------------------------------------------------------------- 报告
def pad(text, width):
    """按显示宽度（全角字符算 2）右补空格。"""
    import unicodedata
    w = sum(2 if unicodedata.east_asian_width(ch) in 'WF' else 1 for ch in text)
    return text + ' ' * max(0, width - w)


def fmt_s(s):
    return (f'{s["mean"]:+6.2f} [{s["lo95"]:+6.2f}, {s["hi95"]:+6.2f}]  p={s["p"]:.3f}  '
            f'>0: {s["n_pos"]}/{s["n"]}')


def counts_table(C):
    """每国每类测试样本数（两个划分）；0 标 '缺'，1–29 标 '*'。"""
    lines = ['国家 划分 | ' + ' | '.join(f'{c[:10]:>10}' for c in EXPECTED_CLASSES)]
    for _, r in C.iterrows():
        cells = []
        for c in EXPECTED_CLASSES:
            n = int(r[c])
            cells.append(f'{"缺":>9}' if n == 0 else f'{n:>9}*' if n < MIN_N else f'{n:>10}')
        lines.append(f'{r.target}   s{r.split} | ' + ' | '.join(cells))
    return lines


def flags4(C):
    """4 类子任务中，某类在任一划分为 0 或 <30 的国家。"""
    out = {}
    for t, g in C.groupby('target'):
        f = []
        for c in C4:
            nm = EXPECTED_CLASSES[c]
            mn = int(g[nm].min())
            if mn == 0:
                f.append(f'{nm}=缺')
            elif mn < MIN_N:
                f.append(f'{nm}<30')
        if f:
            out[t] = f
    return out


def summary(R, C, sources, out_lines):
    P = out_lines.append
    P(f'# 与 Cheng et al. 2026 可比的口径（rev4 D2，补充材料，描述，C 级：G s0/s1 测试集已被复用）')
    P(f'学习器：{", ".join(f"{n}（{s.kind}: {s.root}）" for n, s in sources.items())}')
    P('单位：百分点（pp）；Cheng 的 0.04 OA = 4 pp。国家值 = 单元平均（k=0：s0、s1 两个划分；SRS k=200：2 划分 × 2 重复）；'
      't 检验以国家为单位（11 国 df=10；去掉 RO、PT 后 df=8）。')
    P(f'国家：{",".join(sorted(R.target.unique()))}（{R.target.nunique()} 国）。rows.csv 中 k=0 的 rep=-1（外国模型与重复无关，每个划分只有一行）。')
    P(f'核对：本脚本重算的 oa、ba_ge30 与 rows 中记录值逐位一致的行 {int(R.ref_match.sum())}/{len(R)}。')
    P(f'4 类子任务中并列（含 4 列概率全为 0）的测试地块：合计 {int(R.ties4.sum())}（全为 0 的 {int(R.zero4.sum())}），'
      f'涉及 {int((R.ties4 > 0).sum())}/{len(R)} 行；最大单行 {int(R.ties4.max())} 个，'
      f'占该行子集 {100 * (R.ties4 / R.n_test4).max():.3f}%。10 类 argmax 并列：合计 {int(R.ties10.sum())}。')
    P('')
    metrics = [('ba_all', '(a) 全类平衡精度（出现的所有类）'), ('oa4', '(b) 4 类子任务 OA'),
               ('ba4', '(b) 4 类子任务平衡精度'), ('oa', '参照：10 类 OA（自然先验）'),
               ('ba_ge30', '参照：ba_ge30（主文口径）')]
    P('第一行 = 11 国（df=10）；"去掉 RO、PT" = 9 国（df=8）；"方案外" 行不在 rev4 中预先指定，只作描述。')
    fl = flags4(C)
    no_incomplete = tuple(fl)
    for name in sources:
        for k in KS:
            lab = 'k=0（只用外国数据，主）' if k == 0 else f'SRS k={k}（补充）'
            P(f'## 学习器 {name}，{lab}')
            P(f'{pad("指标", 38)} {"AE 均值":>8} {"S2 均值":>8} | AE−S2：均值 [95% CI]  p  >0 国家数')
            for m, desc in metrics:
                d, lv = country_diff(R, name, k, m)
                s = t_summary(d.to_numpy())
                P(f'{pad(desc, 38)} {lv["ae"].mean():8.2f} {lv["s2"].mean():8.2f} | {fmt_s(s)}')
                d2, lv2 = country_diff(R, name, k, m, DROP)
                s2 = t_summary(d2.to_numpy())
                P(f'{pad("  └ 去掉 RO、PT", 38)} {lv2["ae"].mean():8.2f} {lv2["s2"].mean():8.2f} | {fmt_s(s2)}')
                if m in ('oa4', 'ba4'):
                    d3, lv3 = country_diff(R, name, k, m, no_incomplete)
                    s3 = t_summary(d3.to_numpy())
                    P(f'{pad("  └ 方案外：去掉 4 类不全的国家", 38)} {lv3["ae"].mean():8.2f} {lv3["s2"].mean():8.2f} | '
                      f'{fmt_s(s3)}  （去掉 {",".join(no_incomplete)}）')
            P('')
            # 逐国 AE−S2
            P(f'逐国 AE − S2（{lab}）：')
            cols = ['ba_all', 'oa4', 'ba4', 'oa', 'ba_ge30']
            tab = pd.DataFrame({m: country_diff(R, name, k, m)[0] for m in cols})
            P('国家   ' + ''.join(f'{c:>9}' for c in cols) + '   4 类标记')
            for t, r in tab.iterrows():
                P(f'{t:<6} ' + ''.join(f'{r[c]:+9.2f}' for c in cols) + f'   {"、".join(fl.get(t, []))}')
            P('')
            # 4 类子任务逐类召回（描述）
            P(f'4 类子任务逐类召回 AE − S2（{lab}；只含该类两个划分都 ≥30 的国家；描述，不推断）：')
            for c in C4:
                nm = EXPECTED_CLASSES[c]
                ok = C.groupby('target')[nm].min() >= MIN_N
                keep = ok[ok].index
                sub = R[R.target.isin(keep)]
                d, lv = country_diff(sub, name, k, f'rec4_{nm}')
                P(f'  {nm:<15} n={len(d):>2}  AE {lv["ae"].mean():6.2f}  S2 {lv["s2"].mean():6.2f}  '
                  f'差 均值 {d.mean():+6.2f}  中位数 {d.median():+6.2f}  >0: {int((d > 0).sum())}/{len(d)}')
            P('')
    P('## 各国每类测试样本数（k=0 与 SRS 共用同一测试集；"缺" = 0，"*" = 1–29）')
    out_lines.extend(counts_table(C))
    P('')
    P('4 类子任务中类别不全或不足 30 的国家：' +
      ('；'.join(f'{t}（{"、".join(v)}）' for t, v in fl.items()) if fl else '无'))
    lt30 = [(r.target, r.split, c, int(r[c])) for _, r in C.iterrows() for c in EXPECTED_CLASSES
            if 0 < int(r[c]) < MIN_N]
    P(f'(a) 全类平衡精度中计入的 1–29 个地块的格子：{len(lt30)} 个（国家×划分×类），' +
      '；'.join(f'{t} s{s} {c}={n}' for t, s, c, n in lt30))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', action='append', default=None,
                    help='NAME=KIND:ROOT，KIND ∈ {G, ML}；默认 lgbm=G:<guidance_G.OUT>')
    ap.add_argument('--ml-unblinded', action='store_true',
                    help='rev2 全部完成且 analyze_ML.py 正式运行之后才允许加')
    ap.add_argument('--out', default=HERE)
    ap.add_argument('--countries', default=','.join(G.COUNTRIES_G), help='只用于冒烟测试；正式运行用默认 11 国')
    a = ap.parse_args()
    BlindGuard.unblinded = a.ml_unblinded
    with open(PLAN, 'rb') as f:
        sha = hashlib.sha256(f.read()).hexdigest()[:16]
    assert sha == PLAN_SHA16, f'rev4 方案哈希 {sha} 与登记值 {PLAN_SHA16} 不一致'
    D = G.GData()
    assert list(D.classes) == EXPECTED_CLASSES, D.classes
    sources = parse_sources(a.src or [f'lgbm=G:{G.OUT}'])
    countries = a.countries.split(',')
    R, C = compute_rows(D, sources, countries)
    os.makedirs(a.out, exist_ok=True)
    R.to_csv(os.path.join(a.out, 'cheng_metrics_rows.csv'), index=False)
    C.to_csv(os.path.join(a.out, 'cheng_metrics_counts.csv'), index=False)
    lines = [f'rev4 sha256[:16] = {sha}（与 plan_hashes.txt 一致）；脚本 {os.path.abspath(__file__)}']
    summary(R, C, sources, lines)
    txt = '\n'.join(lines) + '\n'
    with open(os.path.join(a.out, 'cheng_metrics_summary.txt'), 'w') as f:
        f.write(txt)
    print(txt)


if __name__ == '__main__':
    main()
