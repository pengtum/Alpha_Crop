"""测 Accelerate（vecLib）线程数：运行时改 VECLIB_MAXIMUM_THREADS 是否生效；threadpoolctl 能否限制。合成数据。
用法：VECLIB_MAXIMUM_THREADS=4 python veclib_threads.py {none|runtime1|tpc1} {gemm|lr|mlp}
"""
import os
import resource
import sys
import time

import numpy as np

mode, task = sys.argv[1], sys.argv[2]
if mode == 'runtime1':
    os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'


def cpu():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


r = np.random.default_rng(0)
if task == 'gemm':
    A = r.normal(size=(3000, 3000))
elif task in ('lr', 'mlp'):
    X = r.normal(size=(50_000, 168))
    y = r.choice(10, 50_000)
    X += np.eye(10)[y] @ r.normal(size=(10, 168)) * 0.3
    w = r.uniform(0.3, 3, 50_000)

from contextlib import nullcontext
from threadpoolctl import threadpool_limits
ctx = threadpool_limits(limits=1) if mode == 'tpc1' else nullcontext()
c0, t0 = cpu(), time.time()
with ctx:
    if task == 'gemm':
        for _ in range(6):
            A @ A
    elif task == 'lr':
        from sklearn.linear_model import LogisticRegression
        LogisticRegression(C=1.0, max_iter=150, tol=1e-12).fit(X, y, sample_weight=w / w.mean())
    else:
        from sklearn.neural_network import MLPClassifier
        import warnings
        warnings.simplefilter('ignore')
        MLPClassifier((256, 128), batch_size=256, max_iter=int(os.environ.get("EP", 2)), random_state=0).fit(X, y, sample_weight=w / w.mean())
c1, t1 = cpu(), time.time()
print(f'{mode:<9}{task:<5} wall {t1 - t0:6.2f}s cpu {c1 - c0:6.2f}s ratio {(c1 - c0) / (t1 - t0):.2f}', flush=True)
