"""
与 Cheng et al. 2026（RSE 347:115694）可比的口径：ANALYSIS_PLAN_guidance_rev4.md 第 D2 节（补充材料，描述）。
只读已有测试集概率，不训练任何模型。v2（2026-09-27，按 D2 复核意见修订；v1 另存为 cheng_metrics_v1.py）。

rev4 D2 预先指定的内容（本脚本不改变）：
  对象：k=0（只用外国数据）下 AE 特征 − S2 特征，按学习器分别报告；11 国与去掉 RO、PT 两个版本。
  (a) 全类平衡精度 ba_all：测试集中出现的所有类（≥1 个地块）的平均召回。
  (b) 4 类子任务 {grassland, maize, potatoes, winter_cereals}：只保留真实类属于这 4 类的测试地块；
      概率只取这 4 列并重新归一化后 argmax；报告 oa4 与 ba4（子集中出现的类的平均召回）。
  各国每类的测试样本数；某类不足 30 个的国家单独标出。
方案外（rev4 未预先指定，汇总中一律标"方案外，描述"，去留交 D3 统一决定）：
  去掉 4 类不全的国家；4 类逐类召回；SRS k=200 整节；10 类 OA 与 ba_ge30 参照行（含逐国表中这两列）。
  oa 与 ba_ge30 同时用于与 rows 中的记录值逐位核对。

数据来源（学习器 → 概率来源的映射，--src NAME=KIND:ROOT，可重复）：
  KIND=G ：guidance_G.py 的布局。k=0 取 stage0/{国家}_s{划分}.npz 的 P0test_{特征}_i{最大轮}（float16，与主文
           foreign_only 行相同）；SRS k 取 rows_t1_{单元}.csv 中 method=srs 的 sel_hash → probs/{单元}_{特征}_k{k}_{哈希}.npz。
           另用同一 stage0 文件中温度校准后的 P_T_{特征} 测试行（D1 的底座，float32）重算 k=0 指标，只用来标注两者之差。
  KIND=ML：multilearner_G.py 的布局。k=0 取 probs/{NAME}/{国家}_s{划分}_{特征}_k0.npz 的 'P'；
           SRS k 取 rows_{NAME}_{单元}.csv（status=ok）的 sel_hash → probs/{NAME}/{单元}_{特征}_k{k}_{哈希}.npz 的 'P'。
  默认只有 lgbm=G:<guidance_G.OUT>。rev4 规定：LR/MLP/RF 必须等 rev2 全部完成、analyze_ML.py 正式运行之后才可计算；
  因此任何 KIND=ML 的来源、以及任何位于 guidance_G_ml/ 下的路径（按原路径与 realpath 两者判断，不区分大小写），
  都必须显式加 --ml-unblinded 才会被读取。
缺失与技术失败（rev2 analyze_ML 的规则：技术失败剔除，不中止）：
  multilearner_G 不重跑 status=failed 的键，所以真实输出中失败单元只有一条 failed 行、没有概率文件。
  本脚本遇到"没有 status=ok 的行"或"概率文件不存在"时记录并跳过；只有 AE 与 S2 都成功的单元才配对进入 AE − S2；
  剔除明细写入 cheng_metrics_skipped.csv，汇总报告剔除数。同键多条 ok 行、概率形状不符仍视为数据错误并中止。

用法（必须能 import label_expansion 下的模块；脚本自己把该目录加入 sys.path，且不写 .pyc）：
  python cheng_metrics.py                                   # 只算 LightGBM（G）
  python cheng_metrics.py --src lgbm=G:/…/guidance_G --src rf=ML:/…/guidance_G_ml --ml-unblinded   # rev2 解盲之后
输出（--out，默认本脚本所在目录）：cheng_metrics_rows.csv、cheng_metrics_counts.csv、cheng_metrics_skipped.csv、
  cheng_metrics_summary.txt
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
REPO_DOC = os.path.join(L_DIR, 'REPOSITION_AFTER_CHENG2026_2026-09-27.md')
MIN_N = G.MIN_N  # 30
C4 = (0, 1, 4, 9)  # grassland, maize, potatoes, winter_cereals（对应 Cheng 的 MFP、玉米、马铃薯、冬小麦+冬大麦）
DROP = ('RO', 'PT')
FEATS = ('ae', 's2')
KS = (0, 200)  # 0 = 只用外国数据（rev4 D2 指定）；200 = SRS（方案外，描述）
ML_BLIND_DIR = 'guidance_G_ml'
EXPECTED_CLASSES = ['grassland', 'maize', 'orchards', 'other_crops', 'potatoes', 'root_crops', 'strawberries',
                    'tree_wood_forest', 'vegetables', 'winter_cereals']
PAIR_KEY = ['learner', 'method', 'k', 'target', 'split', 'rep']
SKIP_COLS = PAIR_KEY + ['feat', 'stage', 'reason']
OFFPLAN = '方案外，描述'
CHENG_S2ONLY_PP = 4.0  # Cheng 只用 S2 的 SITS 0.67 对 AEF 0.71 → AEF +0.04 OA ≈ +4 pp（近均衡测试集，OA≈BA）

SPARSE_PHRASE = '波罗的海国家和 SE 的玉米只有几百个，PT 的 winter_cereals 只有 156'
# 与 Cheng 对照段引用的数字与设定都读自定位文档或 rev4；运行时逐条核对短语仍在原文中（找不到只警告，不中止）。
DOC_PHRASES = [
    (REPO_DOC, '只用 S2 0.67、AEF 0.71', 'Cheng 传感器消融：只用 S2 的 SITS 0.67，AEF 0.71'),
    (REPO_DOC, '跨区 0.71 对 0.75', 'Cheng 跨区 AEF 0.71 对完整 SITS 0.75'),
    (REPO_DOC, 'S2+S1+Landsat，105 维', 'Cheng 的 SITS = S2+S1+Landsat，105 维'),
    (REPO_DOC, '共 168 列', '我们的 S2 = 手工年度与季度统计，168 列'),
    (REPO_DOC, '按每类最多 5000 点近均衡抽样', 'Cheng 近均衡抽样，OA≈BA'),
    (REPO_DOC, '我们的 AE 取 30 m 邻域的 mean+stdDev（128 维），Cheng 取单像元 64 维', 'AE 提取方式'),
    (REPO_DOC, '只有 AT、DE、PT、SE 四国，而且 Cheng 的 DE 是勃兰登堡，我们是 NRW+LS', '重合国家与 DE 区域'),
    (REPO_DOC, '9 区 × 9 区 × 7 年 = 567 对', 'Cheng 跨区 567 对'),
    (REPO_DOC, '交叉点约 1400–1500 °C·d（约 9–10° 纬度）', 'Cheng 交叉点'),
    (REPO_DOC, '除 PT 外，各目标国到最近源国的 |Δlat| ≤ 1.8°', '我们多源池到最近源国的距离'),
    (REPO_DOC, '随机点划分与 10 km 格网加 1 km 缓冲', '测试集划分'),
    (REPO_DOC, 'RF 为主、6 个近均衡的类', 'Cheng 主分类器 RF、6 类'),
    (PLAN, '对应 Cheng 的 MFP、玉米、马铃薯、冬小麦与冬大麦', 'rev4：冬小麦与冬大麦都对应 winter_cereals'),
    (REPO_DOC, SPARSE_PHRASE, '定位文档的"稀疏"说法'),
]


# ---------------------------------------------------------------- 盲态保护
class BlindGuard:
    """rev4：rev2 多学习器结果解盲之前，任何 guidance_G_ml/ 下的路径都不得读取。
    同时检查原路径（normpath+abspath）与 realpath（解析符号链接），路径分量比较不区分大小写（APFS 默认不区分）。"""
    unblinded = False

    @staticmethod
    def blocked(path):
        cands = {os.path.normpath(os.path.abspath(path)), os.path.realpath(path)}
        return any(ML_BLIND_DIR.lower() in [p.lower() for p in c.split(os.sep)] for c in cands)

    @classmethod
    def check(cls, path):
        if not cls.unblinded and cls.blocked(path):
            raise SystemExit(f'拒绝读取 {path}：rev2 多学习器结果未解盲（rev4 D2）。解盲后加 --ml-unblinded。')
        return path


def _load_npz(path):
    return np.load(BlindGuard.check(path))


def _read_csv(path):
    return pd.read_csv(BlindGuard.check(path))


class MissingUnit(Exception):
    """缺失或技术失败的单元：记录并跳过（rev2：技术失败剔除，不中止）。"""


# ---------------------------------------------------------------- 概率来源
class GSource:
    """guidance_G.py 的输出布局（LightGBM）。"""
    kind = 'G'

    def __init__(self, name, root):
        self.name, self.root = name, BlindGuard.check(root)
        fs = sorted(glob.glob(os.path.join(root, 'rows_t1_*.csv')))
        assert fs, f'{root} 下没有 rows_t1_*.csv'
        R = pd.concat([_read_csv(f) for f in fs], ignore_index=True)
        self.n_failed = 0
        self.rows = R[R.method.isin(['foreign_only', 'srs'])].copy()

    def _stage0(self, tgt, split):
        path = os.path.join(self.root, 'stage0', f'{tgt}_s{split}.npz')
        if not os.path.exists(BlindGuard.check(path)):
            raise MissingUnit(f'概率文件不存在：stage0/{tgt}_s{split}.npz')
        return _load_npz(path)

    def k0(self, tgt, split, feat):
        z = self._stage0(tgt, split)
        pre = f'P0test_{feat}_i'
        its = [int(k[len(pre):]) for k in z.files if k.startswith(pre)]
        if not its:
            raise MissingUnit(f'stage0/{tgt}_s{split}.npz 中没有 {pre}*')
        it = max(its)
        return z[f'{pre}{it}'], f'stage0/{tgt}_s{split}.npz:{pre}{it}', ''

    def k0_calibrated(self, tgt, split, feat):
        """D1 的底座：温度校准后的 P_T 测试行（float32）。只用于标注与 P0test 的差别；没有则返回 None。"""
        z = self._stage0(tgt, split)
        key = f'P_T_{feat}'
        if key not in z.files:
            return None
        nc = len(z['cand'])
        assert np.array_equal(z['allT'][nc:], z['test']), (tgt, split)
        return z[key][nc:]

    def srs(self, tgt, split, rep, feat, k):
        h = self._row('srs', tgt, split, rep, feat, k)['sel_hash']
        rel = f'probs/{tgt}_s{split}_r{rep}_{feat}_k{k}_{h}.npz'
        path = os.path.join(self.root, rel)
        if not os.path.exists(BlindGuard.check(path)):
            raise MissingUnit(f'概率文件不存在：{rel}')
        z = _load_npz(path)
        it = max(int(key[1:]) for key in z.files)
        return z[f'i{it}'], f'{rel}:i{it}', h

    def _row(self, method, tgt, split, rep, feat, k):
        R = self.rows
        r = R[(R.method == method) & (R.target == tgt) & (R.split == split) & (R.rep == rep) & (R.clf == feat) &
              (R.k == k)]
        if len(r) == 0:
            raise MissingUnit(f'rows 中没有 status=ok 的行：{method} k={k} {feat} {tgt} s{split} r{rep}')
        assert len(r) == 1, ('同键多条 ok 行（数据错误）', method, tgt, split, rep, feat, k, len(r))
        return r.iloc[0]

    def reference(self, method, tgt, split, rep, feat, k):
        """rows 中已记录的 oa、ba_ge30，用来核对本脚本的重算（应逐位一致）。
        k=0 的外国模型与重复无关（rep0、rep1 共用同一概率），先取 rep0，没有 ok 行再取 rep1。"""
        reps = [rep] + ([r for r in G.REPS if r != rep] if method == 'foreign_only' else [])
        for i, rr in enumerate(reps):
            try:
                r = self._row(method, tgt, split, rr, feat, k)
            except MissingUnit:
                if i == len(reps) - 1:
                    raise
                continue
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
        R = R[R.method.isin(['foreign_only', 'srs'])]
        if 'status' in R:
            self.n_failed = int((R.status != 'ok').sum())
            R = R[R.status == 'ok']
        else:
            self.n_failed = 0
        self.rows = R.copy()

    def k0(self, tgt, split, feat):
        self.reference('foreign_only', tgt, split, 0, feat, 0)  # 两个重复都没有 ok 行 → MissingUnit
        rel = f'probs/{self.name}/{tgt}_s{split}_{feat}_k0.npz'
        path = os.path.join(self.root, rel)
        if not os.path.exists(BlindGuard.check(path)):
            raise MissingUnit(f'概率文件不存在：{rel}')
        return _load_npz(path)['P'], f'{rel}:P', 'k0'

    def k0_calibrated(self, tgt, split, feat):
        return None

    def srs(self, tgt, split, rep, feat, k):
        h = self._row('srs', tgt, split, rep, feat, k)['sel_hash']
        rel = f'probs/{self.name}/{tgt}_s{split}_r{rep}_{feat}_k{k}_{h}.npz'
        path = os.path.join(self.root, rel)
        if not os.path.exists(BlindGuard.check(path)):
            raise MissingUnit(f'概率文件不存在：{rel}')
        return _load_npz(path)['P'], f'{rel}:P', h


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


def cheng_scores(P, y, preds=False):
    """P：(n, 10) 测试集概率；y：真实类。返回 D2 的全部指标（百分数）与诊断量。preds=True 时另返回 _ph、_ph4。"""
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
    if preds:
        out.update(_ph=ph, _ph4=ph4)
    return out


PT_METRICS = ('ba_all', 'oa4', 'ba4', 'oa', 'ba_ge30')


def pt_diff(P0, PT, y):
    """k=0：D1 底座 P_T（温度校准，float32）减 P0test（float16）的指标差与 argmax 不同的地块数。"""
    a, b = cheng_scores(P0, y, preds=True), cheng_scores(PT, y, preds=True)
    return dict(pt_dtype=str(np.asarray(PT).dtype), pt_dargmax10=int((a['_ph'] != b['_ph']).sum()),
                pt_dargmax4=int((a['_ph4'] != b['_ph4']).sum()),
                **{f'pt_d_{m}': b[m] - a[m] for m in PT_METRICS})


# ---------------------------------------------------------------- 计算
def pair_filter(R, K):
    """只保留 AE 与 S2 都成功的单元；另一种特征缺失或失败的行记为"未配对剔除"。"""
    if R.empty:
        return R, K
    n = R.groupby(PAIR_KEY).feat.transform('nunique')
    bad = R[n < len(FEATS)]
    if len(bad):
        add = bad[PAIR_KEY + ['feat']].assign(stage='未配对剔除', reason='同一单元的另一种特征缺失或失败')
        K = pd.concat([K, add], ignore_index=True)
    return R[n == len(FEATS)].reset_index(drop=True), K


def compute_rows(D, sources, countries):
    rows, counts, skipped = [], [], []
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
                        key = dict(learner=name, method=method, k=k, target=tgt, split=split, rep=rep, feat=feat)
                        try:
                            if k == 0:
                                P, file, h = src.k0(tgt, split, feat)
                            else:
                                P, file, h = src.srs(tgt, split, rep, feat, k)
                            ref_oa, ref_ba = src.reference(method, tgt, split, 0 if rep < 0 else rep, feat, k)
                        except MissingUnit as e:
                            skipped.append(dict(key, stage='缺失或失败', reason=str(e)))
                            continue
                        assert P.shape == (len(y), 10), (name, tgt, split, feat, k, P.shape)
                        m = cheng_scores(P, y)
                        ok = abs(ref_oa - m['oa']) < 1e-9 and abs(ref_ba - m['ba_ge30']) < 1e-9
                        extra = {}
                        if k == 0:
                            PT = src.k0_calibrated(tgt, split, feat)
                            if PT is not None:
                                assert PT.shape == P.shape
                                extra = pt_diff(P, PT, y)
                        rows.append(dict(key, sel_hash=h, source=file, p_dtype=str(P.dtype), ref_oa=ref_oa,
                                         ref_ba_ge30=ref_ba, ref_match=ok, **m, **extra))
    R, K = pair_filter(pd.DataFrame(rows), pd.DataFrame(skipped, columns=SKIP_COLS))
    return R, pd.DataFrame(counts), K[SKIP_COLS]


def country_diff(R, learner, k, metric, drop=()):
    """AE − S2：先在单元内配对相减，再在国家内对单元（k=0：两个划分；SRS：划分 × 重复）平均。"""
    sub = R[(R.learner == learner) & (R.k == k) & ~R.target.isin(drop)]
    w = sub.pivot_table(index=['target', 'split', 'rep'], columns='feat', values=metric)
    d = (w['ae'] - w['s2']).dropna()
    return d.groupby('target').mean(), w.dropna().groupby('target').mean()


def national_counts(D, countries):
    """全国（数据集中全部地块，非测试集）每类计数，用来对照定位文档里的"稀疏"数字。"""
    out = {}
    for t in countries:
        out[t] = np.bincount(np.asarray(D.y[np.asarray(D.ctry) == t]), minlength=10)
    return out


# ---------------------------------------------------------------- 报告
def pad(text, width):
    """按显示宽度（全角字符算 2）右补空格。"""
    import unicodedata
    w = sum(2 if unicodedata.east_asian_width(ch) in 'WF' else 1 for ch in text)
    return text + ' ' * max(0, width - w)


def fmt_s(s):
    if s is None:
        return '国家数不足 2，不计算'
    return (f'{s["mean"]:+6.2f} [{s["lo95"]:+6.2f}, {s["hi95"]:+6.2f}]  p={s["p"]:.3f}  '
            f'>0: {s["n_pos"]}/{s["n"]}')


def ci(s):
    return f'{s["mean"]:+.2f} [{s["lo95"]:+.2f}, {s["hi95"]:+.2f}]（{s["n_pos"]}/{s["n"]}）'


def tstat(d):
    return t_summary(d.to_numpy()) if len(d) >= 2 else None


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


def doc_check():
    """逐条核对对照段引用的短语仍在定位文档 / rev4 中。返回 (找到数, 总数, 找不到的说明)。"""
    texts, miss = {}, []
    for path, phrase, desc in DOC_PHRASES:
        if path not in texts:
            with open(path, encoding='utf-8') as f:
                texts[path] = f.read()
        if phrase not in texts[path]:
            miss.append(f'{desc}（{os.path.basename(path)}："{phrase}"）')
    return len(DOC_PHRASES) - len(miss), len(DOC_PHRASES), miss


def doc_lines(path, phrase):
    """短语在文档中出现的行号（1 起）。"""
    with open(path, encoding='utf-8') as f:
        return [i for i, line in enumerate(f, 1) if phrase in line]


def ties_line(R, name):
    """argmax 并列：按 k 分列，含涉及行数与单行最大占比。"""
    sub = R[R.learner == name]
    parts = []
    for k in KS:
        s = sub[sub.k == k]
        if s.empty:
            continue
        lab = 'k=0' if k == 0 else f'SRS k={k}，{OFFPLAN}'
        dt = '/'.join(sorted(s.p_dtype.unique()))
        parts.append(f'{lab}（概率 {dt}）：4 类子任务 {int(s.ties4.sum())} 个（{int((s.ties4 > 0).sum())}/{len(s)} 行，'
                     f'单行最多占该行 4 类子集 {100 * (s.ties4 / s.n_test4).max():.3f}%）；'
                     f'10 类 {int(s.ties10.sum())} 个（{int((s.ties10 > 0).sum())}/{len(s)} 行，'
                     f'单行最多占该行测试集 {100 * (s.ties10 / s.n_test).max():.3f}%）')
    tot4, tot10 = int(sub.ties4.sum()), int(sub.ties10.sum())
    at0 = sub[sub.k == 0]
    where = ('全部出现在 k=0' if tot4 + tot10 > 0 and int(at0.ties4.sum()) == tot4 and int(at0.ties10.sum()) == tot10
             else '')
    return (f'学习器 {name} 的 argmax 并列（含 4 列全为 0 的 {int(sub.zero4.sum())} 个；并列时取靠前的类）：'
            + '；'.join(parts) + (f'。合计 4 类 {tot4} 个、10 类 {tot10} 个，{where}。' if where else
                                  f'。合计 4 类 {tot4} 个、10 类 {tot10} 个。'))


def pt_lines(R, name):
    s = R[(R.learner == name) & (R.k == 0)]
    if 'pt_d_oa' not in s or s.pt_d_oa.isna().all():
        return [f'学习器 {name}：没有 P_T（温度校准）概率，不做与 D1 底座的比较。']
    s = s.dropna(subset=['pt_d_oa'])
    mx = {m: float(s[f'pt_d_{m}'].abs().max()) for m in PT_METRICS}
    d2m = ('ba_all', 'oa4', 'ba4')  # rev4 D2 指定的三项
    worst = max(d2m, key=mx.get)
    base = sorted(s.source.str.split(':').str[-1].str.replace(r'_(ae|s2)_', '_<特征>_', regex=True).unique())
    # 国家层 AE − S2 的差别（两种特征各自移动，理论上界为单元级最大值的 2 倍；这里直接算实际值）
    cd = {}
    for m in PT_METRICS:
        w = s.pivot_table(index=['target', 'split'], columns='feat', values=f'pt_d_{m}')
        cd[m] = float((w['ae'] - w['s2']).groupby('target').mean().abs().max())
    return [f'学习器 {name} 的 k=0 概率底座：{"、".join(base)}（dtype {"/".join(sorted(s.p_dtype.unique()))}，'
            f'与主文 foreign_only 行相同）；D1 的底座是温度校准后的 P_T（dtype {"/".join(sorted(s.pt_dtype.unique()))}）。'
            '标量温度不改变 argmax，差别只来自 float16 舍入。',
            f'  用 P_T 重算 {len(s)} 行：10 类 argmax 不同的地块合计 {int(s.pt_dargmax10.sum())} 个，4 类 '
            f'{int(s.pt_dargmax4.sum())} 个；单元级 |P_T − P0test| 最大：' +
            '、'.join(f'{m} {mx[m]:.4f}' for m in PT_METRICS) + ' pp。',
            f'  → k=0 的 rev4 D2 三项指标（ba_all、oa4、ba4）与 D1 的 P_T 底座相差不超过 {mx[worst]:.4f} pp'
            f'（≈{mx[worst]:.2f} pp，{worst}）；参照 oa、ba_ge30（{OFFPLAN}）分别不超过 {mx["oa"]:.4f}、{mx["ba_ge30"]:.4f} pp。'
            '单一特征的国家值是单元均值，差别同样不超过这些值；国家层 AE − S2 的实际差别最大：' +
            '、'.join(f'{m} {cd[m]:.4f}' for m in PT_METRICS) + ' pp。D3 数字表引用 k=0 数字时应注明所用底座。']


def summary(R, C, K, sources, D, out_lines):
    P = out_lines.append
    names = list(sources)
    fl = flags4(C)
    no_incomplete = tuple(fl)
    P('# 与 Cheng et al. 2026 可比的口径（rev4 D2，补充材料，描述，C 级：G s0/s1 测试集已被复用）')
    P(f'学习器：{", ".join(f"{n}（{s.kind}: {s.root}）" for n, s in sources.items())}')
    P('范围：rev4 D2 预先指定的只有 k=0 下 AE − S2 的 (a) 全类平衡精度、(b) 4 类子任务 OA 与平衡精度，11 国与去掉 RO、PT '
      '两个版本，以及各国每类测试样本数与不足 30 的标记。')
    P(f'  标"{OFFPLAN}"的内容共四类：①去掉 4 类不全的国家；②4 类逐类召回；③SRS k=200 整节；④10 类 OA 与 ba_ge30 参照行'
      '（含逐国表中这两列）。它们不在 rev4 中，去留由 D3 统一决定。rev4 D2 定位为描述，下文的 CI 与 p 也只作描述。')
    P('单位：百分点（pp）；Cheng 的 0.04 OA = 4 pp。国家值 = 单元平均（k=0：s0、s1 两个划分；SRS k=200：2 划分 × 2 重复）；'
      't 检验以国家为单位（df = 国家数 − 1）。')
    P(f'国家：{",".join(sorted(C.target.unique()))}（{C.target.nunique()} 国）。rows.csv 中 k=0 的 rep=-1'
      '（外国模型与重复无关，每个划分只有一行）。')
    P(f'核对：本脚本重算的 oa、ba_ge30 与 rows 中记录值逐位一致的行 {int(R.ref_match.sum())}/{len(R)}。')
    # 剔除（e）
    n_miss = int((K.stage == '缺失或失败').sum())
    n_unp = int((K.stage == '未配对剔除').sum())
    by = ('、'.join(f'{lr} {st} {n}' for (lr, st), n in K.groupby(['learner', 'stage']).size().items())
          if len(K) else '')
    P(f'剔除（rev2 规则：技术失败剔除，不中止）：缺失或失败 {n_miss} 个（学习器 × 特征 × 单元），因此未配对剔除 {n_unp} 个；'
      f'进入 AE − S2 的只有两种特征都成功的单元。' + (f'按学习器：{by}；明细见 cheng_metrics_skipped.csv。' if by else
                                            '明细文件 cheng_metrics_skipped.csv 为空（只有表头）。'))
    for n in names:
        if sources[n].n_failed:
            P(f'  学习器 {n}：rows 中 status≠ok 的 foreign_only/srs 行 {sources[n].n_failed} 条（已剔除）。')
    # 底座（g）与并列（c）
    for n in names:
        for line in pt_lines(R, n):
            P(line)
    for n in names:
        P(ties_line(R, n))
    P('')

    metrics = [('ba_all', '(a) 全类平衡精度（出现的所有类）', False), ('oa4', '(b) 4 类子任务 OA', False),
               ('ba4', '(b) 4 类子任务平衡精度', False), ('oa', f'参照：10 类 OA（{OFFPLAN}）', True),
               ('ba_ge30', f'参照：ba_ge30（{OFFPLAN}）', True)]
    stats = {}
    offplan_sig = []
    for name in names:
        for k in KS:
            lab = 'k=0（只用外国数据；rev4 D2 指定）' if k == 0 else f'SRS k={k}（{OFFPLAN}）'
            if R[(R.learner == name) & (R.k == k)].empty:
                P(f'## 学习器 {name}，{lab}：没有可配对的行')
                continue
            P(f'## 学习器 {name}，{lab}')
            P(f'{pad("指标", 40)} {"AE 均值":>8} {"S2 均值":>8} | AE−S2：均值 [95% CI]  p  >0 国家数')
            for m, desc, off in metrics:
                off = off or k > 0
                versions = [('11', desc, ()), ('drop', '  └ 去掉 RO、PT', DROP)]
                if m in ('oa4', 'ba4'):
                    versions.append(('inc', f'  └ {OFFPLAN}：去掉 4 类不全的国家', no_incomplete))
                for v, text, drop in versions:
                    d, lv = country_diff(R, name, k, m, drop)
                    s = tstat(d)
                    stats[(name, k, m, v)] = (s, d)
                    tail = f'  （去掉 {",".join(no_incomplete)}）' if v == 'inc' else ''
                    P(f'{pad(text, 40)} {lv["ae"].mean():8.2f} {lv["s2"].mean():8.2f} | {fmt_s(s)}{tail}')
                    if s is not None and (off or v == 'inc') and (s['lo95'] > 0 or s['hi95'] < 0):
                        short = 'k=0' if k == 0 else f'SRS k={k}'
                        offplan_sig.append(f'{name} {short} {desc.split("（")[0].strip()}'
                                           f'{"" if v == "11" else "，" + text.strip(" └")}：{ci(s)}')
            P('')
            # 逐国 AE−S2
            P(f'逐国 AE − S2（{lab}；oa、ba_ge30 两列为{OFFPLAN}）：')
            cols = ['ba_all', 'oa4', 'ba4', 'oa', 'ba_ge30']
            tab = pd.DataFrame({m: country_diff(R, name, k, m)[0] for m in cols})
            P('国家   ' + ''.join(f'{c:>9}' for c in cols) + '   4 类标记')
            for t, r in tab.iterrows():
                P(f'{t:<6} ' + ''.join(f'{r[c]:+9.2f}' for c in cols) + f'   {"、".join(fl.get(t, []))}')
            stats[(name, k, 'tab')] = tab
            P('')
            # 4 类子任务逐类召回（方案外，描述）
            P(f'4 类子任务逐类召回 AE − S2（{lab}；{OFFPLAN}，不推断；只含该类两个划分都 ≥30 的国家）：')
            for c in C4:
                nm = EXPECTED_CLASSES[c]
                ok = C.groupby('target')[nm].min() >= MIN_N
                sub = R[R.target.isin(ok[ok].index)]
                d, lv = country_diff(sub, name, k, f'rec4_{nm}')
                P(f'  {nm:<15} n={len(d):>2}  AE {lv["ae"].mean():6.2f}  S2 {lv["s2"].mean():6.2f}  '
                  f'差 均值 {d.mean():+6.2f}  中位数 {d.median():+6.2f}  >0: {int((d > 0).sum())}/{len(d)}')
            P('')

    # ---------------- 与 Cheng 的对照（a、b）
    found, total, miss = doc_check()
    P('## 与 Cheng et al. 2026 的对照（描述）')
    P(f'引用核对：对照段所用 Cheng 数字与设定逐条读自 REPOSITION_AFTER_CHENG2026_2026-09-27.md 与 rev4，'
      f'{found}/{total} 条原文短语在文中找到' + ('。' if not miss else '；找不到：' + '；'.join(miss) + '（需人工复核）。'))
    P('1. Cheng 的数字：跨区迁移 AEF OA 0.71，与 AEF 输入对齐的完整 SITS（S2+S1+Landsat，105 维）0.75，即 AEF 低 0.04；'
      '只用 S2 的 SITS 为 0.67，即 AEF 高 0.04。Cheng 的测试集每类最多 5000 点、近均衡，OA≈BA，所以 ±0.04 OA ≈ ±4 pp，'
      '最接近的是我们的类均衡指标（全类平衡精度、4 类平衡精度），而不是自然先验下的 OA。')
    verdicts = {}
    P('2. 我们的 S2 是只用 Sentinel-2 的手工物候统计（168 列），可比的参照是 Cheng 只用 S2 的对照（+4 pp），'
      '不是完整 SITS（−4 pp）。−4 pp 与我们的 AE − S2 不是同一个量，所以不用我们的区间去对照 −4。')
    for name in names:
        items = [('全类平衡精度，全部国家', ('ba_all', '11')), ('全类平衡精度，去掉 RO、PT', ('ba_all', 'drop')),
                 ('4 类平衡精度，全部国家', ('ba4', '11')), ('4 类平衡精度，去掉 RO、PT', ('ba4', 'drop')),
                 (f'4 类平衡精度，{OFFPLAN}：去掉 4 类不全的 {"、".join(no_incomplete)}', ('ba4', 'inc'))]
        got = [(lab, stats.get((name, 0, m, v), (None, None))[0]) for lab, (m, v) in items]
        got = [(lab.replace('全部国家', f'{s["n"]} 国'), s) for lab, s in got if s is not None]
        if not got:
            P(f'3. 学习器 {name}：k=0 没有足够的国家，不作对照。')
            verdicts[name] = None
            continue
        P(f'3. 我们（学习器 {name}，k=0）类均衡指标的 AE − S2，均值 [95% CI]（AE 更高的国家数）：')
        for lab, s in got:
            P(f'   - {lab}：{ci(s)}')
        has0 = [lab for lab, s in got if s['lo95'] <= 0 <= s['hi95']]
        has4 = [lab for lab, s in got if s['lo95'] <= CHENG_S2ONLY_PP <= s['hi95']]
        pos = [lab for lab, s in got if s['mean'] > 0]
        all_ci = len(has0) == len(got) and len(has4) == len(got)
        all_pos = len(pos) == len(got)
        verdicts[name] = all_ci and all_pos
        if all_ci:
            P(f'   以上 {len(got)} 个区间都同时含 0 与 +4：数据既不能排除"没有差别"，也不能排除与 Cheng 只用 S2 对照相同的 +4 pp。')
        else:
            P(f'   含 0 的区间 {len(has0)}/{len(got)}（不含 0 的：{"；".join(l for l, _ in got if l not in has0) or "无"}）；'
              f'含 +4 的区间 {len(has4)}/{len(got)}（不含 +4 的：{"；".join(l for l, _ in got if l not in has4) or "无"}）。'
              '措辞须按此逐条写，不作概括。')
        if all_pos:
            P('   点估计全部为正，所以只能说方向与 Cheng 只用 S2 的对照一致，不能说幅度一致，也不能说 AE 显著更好。')
        else:
            P(f'   点估计只有 {len(pos)}/{len(got)} 个为正，方向不一致，不能说与 Cheng 只用 S2 的对照方向一致。')
    P('4. RF：rev2 解盲后补上 RF 的 k=0，只能消除学习器这一项差异（LightGBM 对 Cheng 的主分类器 RF），基线差异仍在。'
      '由于没有与 AEF 输入对齐的 SITS，本数据无论如何都检验不了"AEF 比完整 SITS 低 0.04"。')
    P('5. 设定差异（任何一项都使两边数值不能直接比较）：')
    diffs = [
        '基线：只用 S2 的手工年度与季度统计（168 列）对 Cheng 与 AEF 输入对齐的 SITS（S2+S1+Landsat，105 维）。',
        'AE 提取方式：我们取 30 m 邻域的 mean+stdDev（128 维），Cheng 取单像元 64 维。',
        '类别：10 类模型事后限制到 4 类，对 Cheng 直接训练的 6 类模型；事后限制排除了预测到其余 6 类的错误。',
        'winter_cereals：rev4 把 Cheng 的冬小麦与冬大麦都对应到我们的一类 winter_cereals，'
        '所以我们的 4 类任务没有小麦与大麦互相混淆这一难点。上一项与本项都使我们的 4 类任务比 Cheng 的宽松。',
        '先验与测试集：自然先验、10 km 格网加 1 km 缓冲的空间测试集，对 Cheng 每类最多 5000 点的近均衡、区内随机点划分。',
        '训练设定：多源 10 国外国池，对 Cheng 的单源到单目标。',
        '学习器：LightGBM，对 Cheng 的主分类器 RF（rev2 解盲后可补 RF，见第 4 条）。',
        '空间单元与重合：国家对 NUTS 区；重合的国家只有 AT、DE、PT、SE，而且 Cheng 的 DE 是勃兰登堡，我们的是 NRW+LS。',
        '年份：只有 2021 一年，对 Cheng 的 7 年。',
        'Cheng 的 −0.04 是跨区 567 对的平均，大小随迁移距离变化（AEF 衰减更慢，与 SITS 的交叉点约 1400–1500 °C·d，'
        '约 9–10° 纬度）；我们的多源池中，除 PT 外各目标国到最近源国的 |Δlat| ≤ 1.8°（据定位文档），'
        '所以一个 567 对的平均值不对应我们的迁移距离。',
    ]
    for i, t in enumerate(diffs, 1):
        P(f'   ({i}) {t}')
    for name in names:
        if verdicts.get(name):
            P(f'6. 可写的结论（学习器 {name}）：k=0 下 AE − S2 的类均衡指标点估计为正、区间都含 0 与 +4，'
              '与 Cheng 只用 S2 的对照方向一致、不矛盾；既不能证实也不能反驳"AEF 比完整 SITS 低 0.04"。')
        else:
            P(f'6. 可写的结论（学习器 {name}）：按第 3 条逐条写，不作"方向一致"的概括；'
              '无论结果如何，都不能证实或反驳"AEF 比完整 SITS 低 0.04"。')
    P('')

    # ---------------- 说明与限定（f 及其他）
    P('## 说明与限定')
    Nn = national_counts(D, sorted(C.target.unique()))
    cnt = C.set_index(['target', 'split'])
    notes = []
    # 1 稀疏：全国数 vs 测试集数
    sp = []
    if 'PT' in Nn:
        wc = cnt.loc['PT', 'winter_cereals']
        sp.append(f'PT 的 winter_cereals 全国 {int(Nn["PT"][9])} 个，测试集 s0/s1 只有 {int(wc.loc[0])}/{int(wc.loc[1])} 个')
    mz = [t for t in ('EE', 'LT', 'LV', 'SE') if t in Nn]
    if mz:
        nat = [int(Nn[t][1]) for t in mz]
        tst = [int(cnt.loc[t, 'maize'].loc[s]) for t in mz for s in G.SPLITS]
        sp.append(f'{"、".join(mz)} 的 maize 全国 {min(nat)}–{max(nat)} 个，测试集每个划分只有 {min(tst)}–{max(tst)} 个')
    no_pot = [t for t in Nn if Nn[t][4] == 0]
    if no_pot:
        sp.append(f'{"、".join(no_pot)} 全国没有 potatoes')
    few = {t: [EXPECTED_CLASSES[c] for c in range(10) if Nn[t][c] > 0] for t in Nn if (Nn[t] > 0).sum() <= 3}
    for t, v in few.items():
        sp.append(f'{t} 全国只有 {"、".join(v)} {len(v)} 类')
    ln = doc_lines(REPO_DOC, SPARSE_PHRASE)
    notes.append(f'定位文档第 {"、".join(map(str, ln)) or "?"} 行的"稀疏"数字是全国总数（本脚本按数据集全部地块算得：' + '；'.join(sp) +
                 '）。补充材料应报测试集数（见文末计数表）。')
    # 2 4 类 BA 的归因（f）
    if (names[0], 0, 'tab') in stats:
        tab = stats[(names[0], 0, 'tab')]
        d = tab['ba4'].dropna()
        inc = [t for t in no_incomplete if t in d.index]
        full = d.drop(inc)
        if len(inc) and len(full):
            hi_t, lo_t = full.idxmax(), full.idxmin()
            notes.append(
                f'4 类平衡精度（{names[0]}，k=0）{len(d)} 国均值 {d.mean():+.2f} 几乎全部来自 4 类不全的 {"、".join(inc)}'
                f'（三国差值合计 {d[inc].sum():+.2f}，/{len(d)} ≈ {d[inc].sum() / len(d):+.2f} pp；{len(d)} 国合计 {d.sum():+.2f}）；'
                f'4 类齐全的 {len(full)} 国内，{hi_t} {full[hi_t]:+.2f} 与 {lo_t} {full[lo_t]:+.2f} 相互抵消，'
                f'{len(full)} 国均值 {full.mean():+.2f}（{OFFPLAN}）。')
    # 3 4 类 OA 以 grassland 为主（f）
    c4n = [EXPECTED_CLASSES[c] for c in C4]
    share = 100 * C.grassland / C[c4n].sum(axis=1)
    shr = C.assign(share=share).groupby('target').share.agg(['min', 'max'])
    only_g = [t for t, g in C.groupby('target') if (g[c4n[1:]].to_numpy() == 0).all()]
    notes.append(f'4 类 OA 以 grassland 为主：grassland 占 4 类测试子集的 {share.min():.1f}%–{share.max():.1f}%（按国家和划分；'
                 + '，'.join(f'{t} {r["min"]:.1f}–{r["max"]:.1f}%' for t, r in shr.sort_values('min').iterrows())
                 + '），不能与 Cheng 近均衡测试集上的 OA 直接比；可比的是 4 类平衡精度。'
                 + (f'{"、".join(only_g)} 的 4 类子集只有 grassland，其 4 类 OA 与 4 类平衡精度都等于 grassland 的召回。'
                    if only_g else ''))
    # 4 小格子
    lt30 = [(r.target, r.split, c, int(r[c])) for _, r in C.iterrows() for c in EXPECTED_CLASSES if 0 < int(r[c]) < MIN_N]
    notes.append(f'(a) 全类平衡精度计入了 {len(lt30)} 个只有 1–29 个测试地块的格子（国家 × 划分 × 类，最少 '
                 f'{min(n for *_, n in lt30) if lt30 else "-"} 个），噪声很大；列表见文末。')
    # 5 方案外的显著数字与翻号
    if offplan_sig:
        notes.append(f'标"{OFFPLAN}"的行中 95% CI 不含 0 的有 {len(offplan_sig)} 条：' + '；'.join(offplan_sig) +
                     '。它们不属于 rev4 D2 的指标，是否以及如何引用交 D3 统一处理。')
    flips = []
    for name in names:
        if (name, 0, 'tab') in stats and (name, 200, 'tab') in stats:
            a, b = stats[(name, 0, 'tab')], stats[(name, 200, 'tab')]
            for m in ('oa', 'ba_ge30', 'ba_all', 'oa4', 'ba4'):
                for t in a.index.intersection(b.index):
                    if np.sign(a.loc[t, m]) * np.sign(b.loc[t, m]) < 0 and max(abs(a.loc[t, m]), abs(b.loc[t, m])) > 5:
                        flips.append(f'{name} {t} {m}：k=0 {a.loc[t, m]:+.2f}，SRS k=200 {b.loc[t, m]:+.2f}')
    if flips:
        notes.append(f'逐国差值在 k=0 与 SRS k=200（{OFFPLAN}）之间翻号、且较大一侧的绝对值 >5 pp（本脚本的展示阈值）的：' + '；'.join(flips) + '。')
    notes.append('解盲后的运行：rev2 全部完成、analyze_ML.py 正式运行之后，加 --src lgbm=G:<results/guidance_G> '
                 '--src rf=ML:<results/guidance_G_ml> --ml-unblinded（lr、mlp 同样写法）。缺失或 status=failed 的单元会被记录并剔除，'
                 '不会中止；剔除数写在本汇总开头。')
    for i, t in enumerate(notes, 1):
        P(f'{i}. {t}')
    P('')
    P('## 各国每类测试样本数（k=0 与 SRS 共用同一测试集；"缺" = 0，"*" = 1–29）')
    out_lines.extend(counts_table(C))
    P('')
    P('4 类子任务中类别不全或不足 30 的国家：' +
      ('；'.join(f'{t}（{"、".join(v)}）' for t, v in fl.items()) if fl else '无'))
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
    sources = parse_sources(a.src or [f'lgbm=G:{G.OUT}'])  # 先检查盲态，再加载数据
    D = G.GData()
    assert list(D.classes) == EXPECTED_CLASSES, D.classes
    countries = a.countries.split(',')
    R, C, K = compute_rows(D, sources, countries)
    os.makedirs(a.out, exist_ok=True)
    R.to_csv(os.path.join(a.out, 'cheng_metrics_rows.csv'), index=False)
    C.to_csv(os.path.join(a.out, 'cheng_metrics_counts.csv'), index=False)
    K.to_csv(os.path.join(a.out, 'cheng_metrics_skipped.csv'), index=False)
    lines = [f'rev4 sha256[:16] = {sha}（与 plan_hashes.txt 一致）；脚本 {os.path.abspath(__file__)}（v2；v1 见 *_v1.*）']
    summary(R, C, K, sources, D, lines)
    txt = '\n'.join(lines) + '\n'
    with open(os.path.join(a.out, 'cheng_metrics_summary.txt'), 'w') as f:
        f.write(txt)
    print(txt)


if __name__ == '__main__':
    main()
