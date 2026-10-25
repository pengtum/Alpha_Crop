"""NC 主实验第 3 步：目标域名单、划分、BA 类别与抽样顺序（冻结方案 NC_PLAN_v6 第 1–2 节）。

只用试点已登记样本表中的坐标、编号与类别计数；不读任何特征或概率，不训练，不计算精度。
复用冻结代码 nc_pilot/nc_split.split_domain、n_clusters 与 nc_common（seed_of、rng_of、网格与缓冲常量）。

输出 ~/AlphaCrops_local/nc_main/splits/：
  roles.parquet     task、domain、sid、role（test / cand / buffer）：全部非试点域（含不合格域，供核查）
  eligibility.csv   task、domain、n_test、n_cand、n_ba、eligible（剔除缺失之前）
  targets.json      {task: [合格域，按名称升序]}
  ba_classes.json   {task: {domain: [类别下标]}}（只含合格域）
  orders.npz        键 '{task}/{domain}/r{r}'：候选 sid 按抽样顺序排列（剔除缺失之前；运行时剔除缺失并保持相对次序）
  manifest.json     输入与输出文件的 sha256

源池（运行时使用，不在此生成）：目标域以外同任务的全部样本（LCZ 31 城、LUCAS 27 国），与其在试点中的角色无关。
用法：python nc_main_split.py
"""
import json
import os
import sys

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_v] = '2'
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
NCP = os.path.join(os.path.dirname(HERE), 'nc_pilot')
sys.path.insert(0, NCP)
import nc_common as K  # noqa: E402
import nc_split as S  # noqa: E402

PILOT_SPLITS = '/Users/perng_mini/AlphaCrops_local/nc_pilot/splits'
OUT = '/Users/perng_mini/AlphaCrops_local/nc_main/splits'
PREFIX = 'nc_main_v1'
PILOT_TARGETS = {'lcz': ['amsterdam', 'qingdao', 'melbourne'], 'lucas': ['IT', 'UK', 'SE']}
# 任务 → (样本表, 域列, sid 列, 网格, 原点偏移, 缓冲, 类别数)
TASKS = {
    'lcz': ('t1_samples.parquet', 'city', 'row', K.T1_GRID, K.T1_GRID_OFF, K.T1_BUF, K.T1_C),
    'lucas': ('t2_samples.parquet', 'nuts0', 'id', K.T2_GRID, K.T2_GRID_OFF, K.T2_BUF, K.T2_C),
}
MIN_TEST, MIN_CAND, MIN_BA = 200, 400, 3


def ba_classes(task, sub, te, C):
    """BA 类别：测试集中 ≥ BA_MIN 个样本的类；LCZ 另要求这些样本分属 ≥ BA_MIN_CLUSTERS 个按 T1_CLUSTER 连通的簇。"""
    yt, xt, ymt = sub.y.values[te], sub.x.values[te], sub.y_m.values[te]
    out = []
    for k in range(C):
        m = yt == k
        if m.sum() < K.BA_MIN:
            continue
        if task == 'lcz' and S.n_clusters(xt[m], ymt[m], K.T1_CLUSTER) < K.BA_MIN_CLUSTERS:
            continue
        out.append(k)
    return out


def main():
    man_in = json.load(open(os.path.join(PILOT_SPLITS, 'manifest.json')))
    os.makedirs(OUT, exist_ok=True)
    roles, elig, targets, ba, orders = [], [], {}, {}, {}
    for task, (fn, col, idc, grid, off, buf, C) in TASKS.items():
        path = os.path.join(PILOT_SPLITS, fn)
        assert K.sha256(path) == man_in[fn], f'{fn} 与试点登记的哈希不符'
        T = pd.read_parquet(path)
        assert T[idc].is_unique
        assert sorted(PILOT_TARGETS[task]) == sorted(json.load(open(os.path.join(PILOT_SPLITS, 'targets.json')))[
            {'lcz': 'T1', 'lucas': 'T2'}[task]])
        doms = sorted(set(T[col]) - set(PILOT_TARGETS[task]))
        targets[task], ba[task] = [], {}
        for d in doms:
            sub = T[T[col] == d]
            role = S.split_domain(sub.x.values, sub.y_m.values, grid, off, buf, f'{PREFIX}_{task}_split_{d}')
            te = role == 'test'
            cls = ba_classes(task, sub, te, C)
            n_te, n_ca = int(te.sum()), int((role == 'cand').sum())
            ok = n_te >= MIN_TEST and n_ca >= MIN_CAND and len(cls) >= MIN_BA
            roles.append(pd.DataFrame(dict(task=task, domain=d, sid=sub[idc].values.astype(np.int64), role=role.astype(str))))
            elig.append(dict(task=task, domain=d, n_test=n_te, n_cand=n_ca, n_ba=len(cls), eligible=bool(ok)))
            if not ok:
                continue
            targets[task].append(d)
            ba[task][d] = [int(k) for k in cls]
            ids = np.sort(sub[idc].values[role == 'cand'].astype(np.int64))
            for r in range(K.R):
                orders[f'{task}/{d}/r{r}'] = ids[K.rng_of(f'{PREFIX}_{task}_srs_{d}_r{r}').permutation(len(ids))]

    roles = pd.concat(roles, ignore_index=True)
    elig = pd.DataFrame(elig)
    roles.to_parquet(os.path.join(OUT, 'roles.parquet'), index=False)
    elig.to_csv(os.path.join(OUT, 'eligibility.csv'), index=False)
    json.dump(targets, open(os.path.join(OUT, 'targets.json'), 'w'), indent=1)
    json.dump(ba, open(os.path.join(OUT, 'ba_classes.json'), 'w'), indent=1)
    np.savez(os.path.join(OUT, 'orders.npz'), **orders)

    files = ['roles.parquet', 'eligibility.csv', 'targets.json', 'ba_classes.json', 'orders.npz']
    man = {'inputs': {f: man_in[f] for f in ('t1_samples.parquet', 't2_samples.parquet')},
           'outputs': {f: K.sha256(os.path.join(OUT, f)) for f in files}}
    json.dump(man, open(os.path.join(OUT, 'manifest.json'), 'w'), indent=1)

    pd.set_option('display.width', 200)
    print(elig.to_string(index=False))
    for task in TASKS:
        g = elig[elig.task == task]
        print(f'{task}: 合格 {int(g.eligible.sum())}/{len(g)}；不合格 {", ".join(g.domain[~g.eligible]) or "无"}')
    print('确认性域合计', sum(len(v) for v in targets.values()))
    for f, h in man['outputs'].items():
        print(f'{f} sha256[:16]={h[:16]}')


if __name__ == '__main__':
    main()
