"""Accelerate/BLAS 线程是否受 threadpool_limits 控制：模拟 run_task（先 import numpy，再设环境变量），测 MLP/LR 拟合的 CPU 时间/墙钟时间。合成数据。"""
import os, sys, time, resource
mode = sys.argv[2]
if mode == 'env_first':
    for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[v] = '1'
sys.path.insert(0, sys.argv[1]); os.chdir(sys.argv[1])
import numpy as np
import multilearner_G as ML
if mode == 'env_late':  # 与 run_task 相同：numpy 已导入后才设
    for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[v] = '1'
from threadpoolctl import threadpool_info
print(mode, [(d['internal_api'], d['num_threads']) for d in threadpool_info()])
r = np.random.default_rng(0)
X = r.normal(size=(50000, 168)).astype(np.float32); y = r.choice(10, 50000); w = r.uniform(0.5, 3, 50000)
Xt = r.normal(size=(20000, 168)).astype(np.float32)
ML.PARAMS['mlp']['max_iter'] = 5; ML.PARAMS['mlp_seeds'] = 1
for L in ('mlp', 'lr'):
    ru0 = resource.getrusage(resource.RUSAGE_SELF); t = time.time()
    P, _ = ML.fit_predict_learner(L, X, y, w, Xt, 0, 1, 1.0)
    ru1 = resource.getrusage(resource.RUSAGE_SELF); wall = time.time() - t
    cpu = ru1.ru_utime - ru0.ru_utime + ru1.ru_stime - ru0.ru_stime
    np.save(f'blas_{mode}_{L}.npy', P)
    print(f'{mode} {L}: wall {wall:.1f}s cpu {cpu:.1f}s  cpu/wall {cpu / wall:.2f}', flush=True)
