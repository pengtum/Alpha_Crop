"""(4) 单进程内存峰值探测：照 run_task/run_unit_ml 的路径加载 GData 与 stage0，构造真实训练集（外国 + SRS k=400），
测试矩阵用同形状的随机数代替（不在真实测试集上预测，不算任何精度），按正式运行的线程数拟合一次。
报告 macOS phys_footprint 峰值（不含可回收的文件映射页）与 ru_maxrss（含 memmap 页）。
用法：python mem_probe.py <L> <learner> <tgt> <split> <feat> [n_fits]"""
import ctypes
import os
import resource
import sys
import time

L, learner, tgt, split, feat = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
n_fits = int(sys.argv[6]) if len(sys.argv) > 6 else 1
threads = {'lr': 1, 'mlp': 1, 'rf': 4, 'c': 1}[learner]
for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[v] = str(threads)
os.environ['ML_OUT'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scratch_mem')
sys.path.insert(0, L)
os.chdir(L)

_lib = ctypes.CDLL('/usr/lib/libproc.dylib')


def fp():
    buf = (ctypes.c_uint64 * 64)()
    r = _lib.proc_pid_rusage(os.getpid(), 4, ctypes.byref(buf))  # RUSAGE_INFO_V4
    assert r == 0
    # 前 16 字节是 uuid（2 个 uint64），之后 v2 18 个字段、v3 9 个、v4: logical_writes, lifetime_max_phys_footprint
    cur = buf[2 + 7]
    life = buf[2 + 28]
    return cur / 2 ** 20, life / 2 ** 20


def maxrss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20


log = []


def mark(tag):
    c, l_ = fp()
    log.append((tag, round(c), round(l_), round(maxrss())))
    print(f'{tag:<28} footprint {c:8.0f} MB  峰值 {l_:8.0f} MB  maxrss {maxrss():8.0f} MB', flush=True)


mark('start')
import numpy as np  # noqa: E402
import guidance_G as G  # noqa: E402
import multilearner_G as ML  # noqa: E402
from stageA import class_weights_from  # noqa: E402
mark('imports')
D = G.GData()
mark('GData')
S = dict(np.load(os.path.join(G.OUT, 'stage0', f'{tgt}_s{split}.npz')))
mark(f'stage0 dict ({sum(v.nbytes for v in S.values()) / 2**20:.0f} MB)')
if learner == 'c':
    t = time.time()
    C = ML.select_C(D, S, tgt, split, feat, threads, out=os.environ['ML_OUT'] + f'_{tgt}_{split}')
    mark('select_C')
    edge = C in (min(ML.C_GRID), max(ML.C_GRID))
    print(f'select_C {tgt} s{split} {feat}: {time.time() - t:.0f}s；选中值在网格边界：{edge}', flush=True)
    sys.exit(0)
rep = 0
pool_pos = G.pool_positions(S, split, rep)
fr, test = S['foreign'], S['test']
z = np.load(ML.sel_path(tgt, split, rep, 'none', 'srs', 400))
sel = z['sel']
cw = class_weights_from(D.y[fr])
ys = D.y[sel]
w = np.r_[cw[D.y[fr]], 50 * cw[ys]]
X = D.X[feat]
Xtr = np.vstack([X[fr], X[sel]])
Xte = np.random.default_rng(0).normal(size=(len(test), X.shape[1])).astype(np.float32)  # 形状同测试集，内容随机
mark(f'train set {Xtr.shape} test {Xte.shape}')
for i in range(n_fits):
    t = time.time()
    if learner == 'rf':
        # 额外记录森林规模（节点数）；重新拟合一次以取得模型对象
        from sklearn.ensemble import RandomForestClassifier
        from threadpoolctl import threadpool_limits
        with threadpool_limits(threads):
            m = RandomForestClassifier(random_state=i, n_jobs=threads, **ML.PARAMS['rf']).fit(
                np.asarray(Xtr, np.float64), np.r_[D.y[fr], ys], sample_weight=w)
        nodes = sum(e.tree_.node_count for e in m.estimators_)
        mark(f'rf fitted, nodes {nodes:,}')
        print(f'  RF 拟合 {time.time() - t:.1f}s；平均每棵树节点 {nodes / 500:,.0f}；最大深度 '
              f'{max(e.tree_.max_depth for e in m.estimators_)}', flush=True)
        t = time.time()
        P = m.predict_proba(Xte.astype(np.float64))
        mark('rf predicted')
        print(f'  RF 预测 {time.time() - t:.1f}s', flush=True)
        del m, P
    else:
        P, info = ML.fit_predict_learner(learner, Xtr, np.r_[D.y[fr], ys], w, Xte, 1000 * split + rep, threads, 1e-3)
        mark(f'{learner} fit+predict #{i}')
        print(f'  {learner} 拟合+预测 {time.time() - t:.1f}s  info {info}', flush=True)
print('SUMMARY', learner, tgt, split, feat, log[-1], flush=True)
