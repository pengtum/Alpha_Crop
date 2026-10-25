"""
智能 resume 脚本：检查每个 source/feature 的 CSV 状态，只跑未完成的部分。
"""
import subprocess
import sys
import os
import time
import pandas as pd

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUT_DIR = 'results/experiments/exp6_scalable_mapping'
SOURCES = ['DE', 'AT', 'BE', 'EE', 'LT', 'LV', 'PT', 'SE', 'SI', 'SK']
FEATURES = ['ae', 's2', 'fusion']
EXPECTED_ROWS = 108  # 8 strategies * (1 + 5) * 3 seeds = 108

STRATEGIES = ['source_only', 'random', 'climate_similar', 'climate_diverse',
              'representativeness', 'balanced',
              'transfer_strong', 'transfer_weak']
BUDGETS = [1000, 5000, 10000, 20000, 50000]

env = os.environ.copy()
env['PYTHONUNBUFFERED'] = '1'
env['PYTHONIOENCODING'] = 'utf-8'


def check_status():
    """返回需要跑的 (source, feature) 对。"""
    todo = []
    for s in SOURCES:
        for ft in FEATURES:
            f = os.path.join(OUT_DIR, f'scalable_{ft}_{s}_tier_b.csv')
            if not os.path.exists(f):
                todo.append((s, ft, 0))
                continue
            try:
                n = len(pd.read_csv(f))
            except Exception:
                n = 0
            if n < EXPECTED_ROWS:
                todo.append((s, ft, n))
    return todo


def run_source_features(source, features):
    """对指定 source 跑指定 features。"""
    cmd = [
        sys.executable, '-u', '-W', 'ignore',
        '-m', 'src.experiments.run_scalable_mapping',
        '--source', source,
        '--strategies', *STRATEGIES,
        '--budgets', *[str(b) for b in BUDGETS],
        '--n-seeds', '3',
        '--feature-types', *features,
    ]
    print(f'\n{"=" * 70}')
    print(f'>>> Source={source}, Features={features}')
    print(f'{"=" * 70}')
    start = time.time()
    result = subprocess.run(cmd, env=env, cwd=os.getcwd())
    elapsed = (time.time() - start) / 60.0
    print(f'\n[{source} {features}] finished in {elapsed:.1f} min (exit {result.returncode})')
    return result.returncode


if __name__ == '__main__':
    total_start = time.time()

    todo = check_status()
    print(f'\nTotal items to run: {len(todo)}')
    for s, ft, n in todo:
        print(f'  {s} {ft}: {n}/{EXPECTED_ROWS}')

    # 按 source 分组
    by_source = {}
    for s, ft, _ in todo:
        by_source.setdefault(s, []).append(ft)

    for source, features in by_source.items():
        rc = run_source_features(source, features)
        if rc != 0:
            print(f'WARNING: {source} failed with exit {rc}')

    total_elapsed = (time.time() - total_start) / 3600.0
    print(f'\n{"=" * 70}')
    print(f'ALL DONE in {total_elapsed:.1f} hours')
    print(f'{"=" * 70}')

    # 最终状态
    final = check_status()
    if not final:
        print('ALL EXPERIMENTS COMPLETE ✓')
    else:
        print(f'Still missing: {len(final)} items')
        for s, ft, n in final:
            print(f'  {s} {ft}: {n}/{EXPECTED_ROWS}')
