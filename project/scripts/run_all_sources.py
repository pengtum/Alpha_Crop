"""
串行运行 Exp 6 scalable mapping 全部实验：
1. 先完成 DE 的 Fusion 剩余 runs（transfer_strong + transfer_weak）
2. 然后启动 8 个剩余 source 国（AT, BE, EE, LT, LV, PT, SE, SI, SK）
"""
import subprocess
import sys
import os
import time
import shutil
import pandas as pd

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUT_DIR = 'results/experiments/exp6_scalable_mapping'
SOURCES_NEW = ['AT', 'BE', 'EE', 'LT', 'LV', 'PT', 'SE', 'SI', 'SK']

STRATEGIES_ALL = ['source_only', 'random', 'climate_similar', 'climate_diverse',
                  'representativeness', 'balanced',
                  'transfer_strong', 'transfer_weak']
BUDGETS = [1000, 5000, 10000, 20000, 50000]
FEATURES = ['ae', 's2', 'fusion']
N_SEEDS = 3

env = os.environ.copy()
env['PYTHONUNBUFFERED'] = '1'
env['PYTHONIOENCODING'] = 'utf-8'


def run_cmd(cmd, label):
    print(f'\n{"=" * 70}')
    print(f'>>> {label}')
    print(f'{"=" * 70}')
    print(f'Cmd: {" ".join(cmd)}')
    start = time.time()
    result = subprocess.run(cmd, env=env, cwd=os.getcwd())
    elapsed = (time.time() - start) / 60.0
    print(f'\n[{label}] finished in {elapsed:.1f} min (exit {result.returncode})')
    return result.returncode


def merge_de_fusion():
    """合并 DE fusion partial + 新跑的 transfer 数据。"""
    fusion_path = os.path.join(OUT_DIR, 'scalable_fusion_DE_tier_b.csv')
    partial_path = os.path.join(OUT_DIR, 'scalable_fusion_DE_tier_b_partial.csv')
    if not os.path.exists(partial_path):
        return
    if not os.path.exists(fusion_path):
        shutil.copy(partial_path, fusion_path)
        return

    partial = pd.read_csv(partial_path)
    new = pd.read_csv(fusion_path)
    combined = pd.concat([partial, new], ignore_index=True).drop_duplicates(
        subset=['strategy', 'budget', 'seed'], keep='last')
    combined.to_csv(fusion_path, index=False)
    print(f'\nMerged DE fusion: partial {len(partial)} + new {len(new)} = '
          f'total {len(combined)}')


if __name__ == '__main__':
    total_start = time.time()
    results = {}

    # 1. 补跑 DE Fusion 的 transfer_strong/weak (剩余 ~25 runs)
    print('\n\n########## STAGE 1: Resume DE Fusion ##########\n')
    rc = run_cmd([
        sys.executable, '-u', '-W', 'ignore',
        '-m', 'src.experiments.run_scalable_mapping',
        '--source', 'DE',
        '--strategies', 'transfer_strong', 'transfer_weak',
        '--budgets', '5000', '10000', '20000', '50000',
        '--n-seeds', '3',
        '--feature-types', 'fusion',
    ], 'DE fusion resume')
    results['DE_fusion_resume'] = rc

    # 合并
    merge_de_fusion()

    # 2. 8 个新 source
    print(f'\n\n########## STAGE 2: Run 8 new sources ##########\n')
    for i, src in enumerate(SOURCES_NEW):
        label = f'[{i+1}/{len(SOURCES_NEW)}] Source={src}'
        rc = run_cmd([
            sys.executable, '-u', '-W', 'ignore',
            '-m', 'src.experiments.run_scalable_mapping',
            '--source', src,
            '--strategies', *STRATEGIES_ALL,
            '--budgets', *[str(b) for b in BUDGETS],
            '--n-seeds', str(N_SEEDS),
            '--feature-types', *FEATURES,
        ], label)
        results[src] = rc

    total_elapsed = (time.time() - total_start) / 3600.0
    print(f'\n{"=" * 70}')
    print(f'ALL COMPLETE in {total_elapsed:.1f} hours')
    print(f'{"=" * 70}')
    for src, rc in results.items():
        status = 'OK' if rc == 0 else f'FAILED (exit {rc})'
        print(f'  {src}: {status}')
