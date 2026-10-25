"""
测试用：把 G s0/s1 的旧数据改写成确认运行的输出接口（splits/、runs/、probs/），只用于检验 indep_confirm.py 的正式模式读取路径。
单元 {国}_F{划分+1}，r = G 的重复 r0/r1。场景 B 的臂用已知答案的符号链接：
  B_S0 → RT100；B_SRS100、B_EXT100 → RT200；B_BA100 → RT100；B_SRS300、B_BA300 → RT400；B_EXT300 → RT50
变体 --variant fail：AT_F1 的 lgb B_SRS100 ae r1 记 failed 并删除文件；SI_F2 的 test 截成 2,000 个（应被剔除）。
"""
import os
import sys
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_v] = '2'
sys.dont_write_bytecode = True
import glob
import shutil

import numpy as np
import pandas as pd

R_G = '/Users/perng_mini/AlphaCrops_local/results/guidance_G'
R_ML = '/Users/perng_mini/AlphaCrops_local/results/guidance_G_ml'
COUNTRIES = 'AT,BE,DE,EE,LT,LV,PT,RO,SE,SI,SK'.split(',')
KS = (50, 100, 200, 400)
LINKS = {'B_S0': 'RT100', 'B_SRS100': 'RT200', 'B_EXT100': 'RT200', 'B_BA100': 'RT100',
         'B_SRS300': 'RT400', 'B_EXT300': 'RT50', 'B_BA300': 'RT400'}
sys.path.insert(0, '/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops/label_expansion')
from guidance_G import GData  # noqa: E402

root = sys.argv[1]
variant = sys.argv[2] if len(sys.argv) > 2 else ''
shutil.rmtree(root, ignore_errors=True)
for d in ('splits', 'runs', 'probs'):
    os.makedirs(os.path.join(root, d))
y = np.asarray(GData().y)
GR = pd.concat([pd.read_csv(p) for p in glob.glob(os.path.join(R_G, 'rows_t1_*.csv'))])
GR = GR[(GR.cond == 'none') & (GR.method == 'srs')]
MR = pd.concat([pd.read_csv(p) for p in glob.glob(os.path.join(R_ML, 'rows_rf_*.csv'))])
MR = MR[(MR.learner == 'rf') & (MR.cond == 'none') & (MR.method == 'srs') & (MR.k == 200)]
for tgt in COUNTRIES:
    for split in (0, 1):
        unit = f'{tgt}_F{split + 1}'
        S = np.load(os.path.join(R_G, 'stage0', f'{tgt}_s{split}.npz'))
        test = S['test']
        big = np.flatnonzero(np.bincount(y[test], minlength=10) >= 30)
        if variant == 'fail' and unit == 'SI_F2':
            test = test[:2000]
        np.savez(os.path.join(root, 'splits', f'{unit}.npz'), test=test, big=big)
        pd_ = os.path.join(root, 'probs', unit)
        os.makedirs(pd_)
        runs = []

        def put(lrn, arm, f, r, P, h):
            np.savez(os.path.join(pd_, f'{lrn}_{arm}_{f}_r{r}.npz'), P_test=np.asarray(P, dtype=np.float32))
            runs.append(dict(learner=lrn, arm=arm, feat=f, r=r, k=0, sel_hash=h, seed=0, n_train=0, t_fit=0.0,
                             status='ok', code_hash=''))
        for f in ('ae', 's2'):
            put('lgb', 'Z0', f, 0, S[f'P0test_{f}_i300'], '')
            put('rf', 'Z0', f, 0, np.load(os.path.join(R_ML, 'probs', 'rf', f'{tgt}_s{split}_{f}_k0.npz'))['P'], '')
            for r in (0, 1):
                hs = {}
                for k in KS:
                    g = GR[(GR.target == tgt) & (GR.split == split) & (GR.rep == r) & (GR.clf == f) & (GR.k == k)]
                    h = g.sel_hash.iloc[0]
                    hs[f'RT{k}'] = h
                    put('lgb', f'RT{k}', f, r,
                        np.load(os.path.join(R_G, 'probs', f'{tgt}_s{split}_r{r}_{f}_k{k}_{h}.npz'))['i300'], h)
                m = MR[(MR.target == tgt) & (MR.split == split) & (MR.rep == r) & (MR.clf == f)]
                put('rf', 'RT200', f, r, np.load(os.path.join(
                    R_ML, 'probs', 'rf', f'{tgt}_s{split}_r{r}_{f}_k200_{m.sel_hash.iloc[0]}.npz'))['P'],
                    m.sel_hash.iloc[0])
                for arm, srcarm in LINKS.items():
                    os.symlink(f'lgb_{srcarm}_{f}_r{r}.npz', os.path.join(pd_, f'lgb_{arm}_{f}_r{r}.npz'))
                    runs.append(dict(learner='lgb', arm=arm, feat=f, r=r, k=0, sel_hash=hs[srcarm], seed=0,
                                     n_train=0, t_fit=0.0, status='ok', code_hash=''))
        R = pd.DataFrame(runs)
        if variant == 'fail' and unit == 'AT_F1':
            q = (R.learner == 'lgb') & (R.arm == 'B_SRS100') & (R.feat == 'ae') & (R.r == 1)
            R.loc[q, 'status'] = 'failed'
            os.remove(os.path.join(pd_, 'lgb_B_SRS100_ae_r1.npz'))
        R.to_csv(os.path.join(root, 'runs', f'{unit}.csv'), index=False)
        print(unit, len(test), big.tolist(), flush=True)
