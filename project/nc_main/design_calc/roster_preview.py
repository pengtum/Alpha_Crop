"""冻结前预览确认性目标域名单（方案 v6 第 1–2 节的资格规则）。只用坐标、编号与类别计数，不读任何特征、概率或精度。
正式名单在冻结后由划分代码生成并登记；本脚本只用于核对方案中写的域数。"""
import os
import sys

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_v] = '2'
import numpy as np
import pandas as pd

NCP = '/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops/nc_pilot'
SPL = '/Users/perng_mini/AlphaCrops_local/nc_pilot/splits'
sys.path.insert(0, NCP)
import nc_common as K  # noqa: E402
import nc_split as S  # noqa: E402

PILOT = {'lcz': {'amsterdam', 'qingdao', 'melbourne'}, 'lucas': {'IT', 'UK', 'SE'}}
rows = []
for task, fn, col, grid, off, buf, C in (('lcz', 't1_samples.parquet', 'city', K.T1_GRID, K.T1_GRID_OFF, K.T1_BUF, K.T1_C),
                                         ('lucas', 't2_samples.parquet', 'nuts0', K.T2_GRID, K.T2_GRID_OFF, K.T2_BUF, K.T2_C)):
    T = pd.read_parquet(os.path.join(SPL, fn))
    for d in sorted(set(T[col]) - PILOT[task]):
        sub = T[T[col] == d]
        role = S.split_domain(sub.x.values, sub.y_m.values, grid, off, buf, f'nc_main_v1_{task}_split_{d}')  # {域} 取样本表原值
        te = role == 'test'
        yt, xt, ymt = sub.y.values[te], sub.x.values[te], sub.y_m.values[te]
        ba = [k for k in range(C) if (yt == k).sum() >= K.BA_MIN
              and (task == 'lucas' or S.n_clusters(xt[yt == k], ymt[yt == k], K.T1_CLUSTER) >= K.BA_MIN_CLUSTERS)]
        n_te, n_ca = int(te.sum()), int((role == 'cand').sum())
        rows.append(dict(task=task, domain=d, n_test=n_te, n_cand=n_ca, n_ba=len(ba),
                         eligible=n_te >= 200 and n_ca >= 400 and len(ba) >= 3))
R = pd.DataFrame(rows)
pd.set_option('display.width', 200)
print(R.to_string(index=False))
for task, g in R.groupby('task'):
    print(task, '合格', int(g.eligible.sum()), '/', len(g), '；不合格：', ', '.join(g.domain[~g.eligible]))
print('确认性域合计', int(R.eligible.sum()))
