"""独立复核：按冻结方案层级 3 的定义补跑 RAWZ（E1 四方法）与 NATIVE（lpm2_nogeo、typiclust），只用 k=50、重复 0。
直接调用 guidance_G.run_unit，out 指向本目录；只读取正式结果目录的 stage0 npz，不写入项目或结果目录。
顺序：NATIVE s0 → RAWZ s0 → NATIVE s1 → RAWZ s1（先出 NATIVE）。"""
import os, sys, time
import multiprocessing as mp
import numpy as np
LE = '/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops/label_expansion'
MY = '/Users/perng_mini/AlphaCrops_local/verify/G_audit/tier3_verify/out'
ST0 = '/Users/perng_mini/AlphaCrops_local/results/guidance_G/stage0'
sys.path.insert(0, LE)

def jobs(phase):
    both = ('s2', 'ae')
    E1 = ['lpm2_nogeo', 'kcenter', 'typiclust', 'kherding']
    if phase == 'NATIVE':
        return [(c, m, 50, both) for c in ('NATIVE_AE', 'NATIVE_S2') for m in ('lpm2_nogeo', 'typiclust')]
    # 按方法成对排列（kcenter、kherding 这两个极值型方法先跑），部分完成时也能配对
    return [(c, m, 50, both) for m in ('kcenter', 'kherding') for c in ('RAWZ_AE', 'RAWZ_S2')]

def init():
    os.chdir(LE)
    import guidance_G as G
    init.G = G
    init.D = G.GData()

def task(a):
    phase, tgt, split = a
    G = init.G
    G.jobs_for = lambda tier, s, r: jobs(phase)
    S = dict(np.load(os.path.join(ST0, f'{tgt}_s{split}.npz')))
    t0 = time.time()
    rows = G.run_unit(init.D, S, tgt, split, 0, 3, 2, out=MY, log=lambda s: print(s, flush=True))
    return f'{phase} {tgt} s{split}: {len(rows)} rows {time.time()-t0:.0f}s'

if __name__ == '__main__':
    C = 'AT,BE,DE,EE,LT,LV,PT,RO,SE,SI,SK'.split(',')
    nj = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    phases = [[('RAWZ', t, 0) for t in C]]
    with mp.Pool(nj, initializer=init) as pool:
        for ph in phases:
            for msg in pool.imap_unordered(task, ph):
                print('task done:', msg, flush=True)
