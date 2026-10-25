"""
核查：G 实验 13 个方法在 samplers/ 中实际读取了 ctx 的哪些字段（含间接调用），以及对各字段扰动的不变性。
只读 CODE；不写 __pycache__（PYTHONDONTWRITEBYTECODE=1）。
"""
import os
import sys
import json
import copy
import numpy as np

CODE = "/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops/label_expansion"
sys.path.insert(0, CODE)
import samplers.base as B  # noqa: E402
import samplers.designs, samplers.uncertainty, samplers.coverage, samplers.transfer  # noqa: E402,F401

G_METHODS = ['srs', 'lpm2', 'fscs', 'kcenter', 'typiclust', 'kherding', 'gist_maxmin', 'margin', 'predstrat_srs',
             'gist_margin', 'predstrat', 'badge', 'fass_quota']
GEO0 = {'lpm2': {'geo_w': 0.0}, 'predstrat': {'geo_w': 0.0}, 'predstrat_srs': {'geo_w': 0.0}}


class LoggedArr(np.ndarray):
    """记录 __getitem__ 的键（用于看取了哪些列）。"""
    def __getitem__(self, key):
        LoggedArr.log.append((self._name, repr(key)[:60]))
        return np.asarray(super().__getitem__(key))


LoggedArr.log = []


def logged(a, name):
    v = np.asarray(a).view(LoggedArr)
    v._name = name
    return v


class TraceCtx(B.Ctx):
    def __getattribute__(self, name):
        if not name.startswith('__'):
            object.__getattribute__(self, '_reads').add(name)
        return object.__getattribute__(self, name)


def make(seed=0, N=3000):
    c = B.make_toy_ctx(N=N, seed=seed)
    return c


def to_trace(c):
    d = {f: getattr(c, f) for f in B.Ctx.__dataclass_fields__}
    d['meta'] = {}
    t = TraceCtx(**d)
    object.__setattr__(t, '_reads', set())
    for f in ('T_ae', 'T_s2', 'T_xy', 'F_ae', 'F_s2'):
        object.__setattr__(t, f, logged(getattr(c, f), f))
    return t


def run(c, name, k, seed, hp=None):
    return B.REGISTRY[name](c, k, seed, **(hp or {}))


def main():
    out = {}
    base = make()
    base.oracle = None
    for name in G_METHODS:
        for tag, hp in (('default', {}), ('G', GEO0.get(name, {}))):
            if tag == 'G' and name not in GEO0:
                continue
            t = to_trace(base)
            t.oracle = None
            LoggedArr.log = []
            run(t, name, 50, 3, hp)
            reads = sorted(object.__getattribute__(t, '_reads') - {'_reads'})
            cols = sorted(set(LoggedArr.log))
            out[f'{name}[{tag}]'] = {'reads': reads, 'array_slices': cols}
    for kk, v in out.items():
        print(kk)
        print('   ctx 字段:', v['reads'])
        print('   数组切片:', v['array_slices'])
    json.dump(out, open(os.path.join(os.path.dirname(__file__), 'trace_fields.json'), 'w'), ensure_ascii=False,
              indent=1)


if __name__ == '__main__':
    main()
