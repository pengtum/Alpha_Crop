"""检查 main() 的多进程路径（spawn、init_worker、c_task、run_task）与 k=0 缓存在两个重复并发时的行为。
不计算任何精度：G.evaluate 被替换成返回常数的桩函数。输出写到本目录 pool_out/。
在 label_expansion 目录下运行：python pool_path.py
"""
import os
import shutil
import sys

HERE = '/Users/perng_mini/AlphaCrops_local/verify/ml_review/static'
os.environ['ML_OUT'] = os.path.join(HERE, 'pool_out')
sys.path.insert(0, os.getcwd())
import pandas as pd
import guidance_G as G
import multilearner_G as ML

# 模块级补丁：spawn 子进程重新导入本文件（__mp_main__）时同样生效
G.COUNTRIES_G = ['SI']
G.unit_order = lambda: [('SI', 0, 0), ('SI', 0, 1)]
G.evaluate = lambda *a, **k: {'stub': 0.0}
ML.unit_jobs = lambda split, rep, _: [('none', 'foreign_only', 0), ('AE', 'lpm2_nogeo', 50)]
ML.build_manifest = lambda: pd.DataFrame({'x': [1]})
G.SPLITS = [0]


def pid_env_probe():
    return os.getpid(), os.environ.get('OMP_NUM_THREADS'), os.environ.get('VECLIB_MAXIMUM_THREADS')


if __name__ == '__main__':
    os.makedirs(os.path.join(ML.OUT, 'lr_C'), exist_ok=True)
    for f in ('ae', 's2'):  # 用 csel_check.py 已算好的 C（只用外国池），避免重算
        shutil.copy(os.path.join(HERE, 'csel_out', 'lr_C', f'SI_s0_{f}.json'), os.path.join(ML.OUT, 'lr_C'))
    sys.argv = ['multilearner_G.py', '--learner', sys.argv[1] if len(sys.argv) > 1 else 'lr', '--jobs', os.environ.get('JOBS', '2'),
                '--threads', '1']
    ML.main()
    for f in sorted(os.listdir(os.path.join(ML.OUT))):
        print('  out:', f)
    for L in os.listdir(os.path.join(ML.OUT, 'probs')):
        print('  probs/', L, sorted(os.listdir(os.path.join(ML.OUT, 'probs', L))))
    for f in sorted(os.listdir(ML.OUT)):
        if f.startswith('rows_'):
            R = pd.read_csv(os.path.join(ML.OUT, f))
            print(f, R[['cond', 'method', 'k', 'clf', 'sel_hash', 'C', 'cache_hit', 't_fit']].to_string())
