# -*- coding: utf-8 -*-
"""诊断：用主实现的调查区划分（partA/draws/*_frame.npz 的 reg）替换独立实现的 KMeans 区，其余照独立实现原样运行。
只读 partA / partA_indep；输出写到本目录下的 indep_mainreg/。"""
import os
import sys

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_v] = '2'
sys.dont_write_bytecode = True
import numpy as np  # noqa: E402

sys.path.insert(0, '/Users/perng_mini/AlphaCrops_local/verify/step0/partA_indep')
import indep_partA as I  # noqa: E402

A_DIR = '/Users/perng_mini/AlphaCrops_local/verify/step0/partA'
HERE = os.path.dirname(os.path.abspath(__file__))
_orig = I.Unit.__init__


def _init(self, D, tgt, split, centroid='mass'):
    _orig(self, D, tgt, split, centroid)
    F = np.load(os.path.join(A_DIR, 'draws', f'{tgt}_s{split}_frame.npz'))
    assert np.array_equal(F['cand'], self.S['cand'])
    reg = F['reg'].astype(np.int64)
    zc = np.full(self.n_cells, -1, np.int64)
    zc[self.cell] = reg
    assert (zc >= 0).all() and np.array_equal(zc[self.cell], reg)   # 同一格内地块同区
    self.zone_cell, self.zone_par = zc, reg
    self.zone_size = np.bincount(reg, minlength=I.N_ZONE)


I.Unit.__init__ = _init
if __name__ == '__main__':
    I.run('mass', os.path.join(HERE, 'indep_mainreg'))
