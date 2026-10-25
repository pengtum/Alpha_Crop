"""
一键运行实验管线（分步骤执行）。

Usage:
    python run_pipeline.py --step 1       # 数据准备
    python run_pipeline.py --step 2       # AlphaEarth GEE 提取
    python run_pipeline.py --step 3       # S2 GEE 提取
    python run_pipeline.py --step 4       # 合并嵌入
    python run_pipeline.py --step 5       # 构建特征矩阵
    python run_pipeline.py --step 6       # 运行全部实验
    python run_pipeline.py --step all     # 依次运行 1→5（不含 GEE 步骤）
"""
import argparse
import subprocess
import sys
import os
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

# 临时采样上限（设为 0 则不限制）
MAX_PER_COUNTRY = 50000

# 默认 tier
TIER = 'b'


def run(cmd, desc):
    """运行命令并打印状态。"""
    print(f"\n{'='*60}")
    print(f"▶ {desc}")
    print(f"  命令: {cmd}")
    print(f"{'='*60}\n")
    t0 = time.time()
    result = subprocess.run(cmd, shell=True)
    elapsed = time.time() - t0
    status = "✓ 完成" if result.returncode == 0 else "✗ 失败"
    print(f"\n{status} ({elapsed/60:.1f} min)\n")
    return result.returncode


def step1():
    """数据准备：生成 parcels_eu.parquet"""
    cmd = (f'python -m src.data.prepare_parcels --mode eu '
           f'--max-per-country {MAX_PER_COUNTRY}')
    return run(cmd, f'Step 1: 泛欧数据准备 (max_per_country={MAX_PER_COUNTRY:,})')


def step2():
    """AlphaEarth GEE 提取"""
    cmd = 'python -m src.extraction.gee_centroid_extract --mode eu'
    return run(cmd, 'Step 2: AlphaEarth 嵌入提取 (GEE batch export)')


def step3():
    """S2 GEE 提取"""
    cmd = 'python -m src.extraction.gee_s2_extract --mode eu'
    return run(cmd, 'Step 3: Sentinel-2 光谱特征提取 (GEE batch export)')


def step4():
    """合并嵌入"""
    rc1 = run('python -m src.extraction.merge_embeddings --mode eu --merge-all',
              'Step 4a: 合并 AlphaEarth 嵌入')
    rc2 = run('python -m src.extraction.merge_embeddings --mode eu --source s2 --merge-all',
              'Step 4b: 合并 S2 嵌入')
    return max(rc1, rc2)


def step5():
    """构建特征矩阵"""
    rc = 0
    for ft in ['ae', 's2', 'fusion']:
        r = run(f'python -m src.features.build_features --mode eu --feature-type {ft}',
                f'Step 5: 构建 {ft} 特征矩阵')
        rc = max(rc, r)
    return rc


def step6():
    """运行全部实验"""
    experiments = [
        ('python -m src.experiments.run_id_baseline --tier {tier}',
         'Exp 1: ID Baseline (5-fold CV)'),
        ('python -m src.experiments.run_loco --tier {tier}',
         'Exp 2: Leave-One-Country-Out'),
        ('python -m src.experiments.run_climate_zone --tier {tier}',
         'Exp 3: Climate Zone Transfer'),
        ('python -m src.experiments.run_label_efficiency --tier {tier}',
         'Exp 4: Label Efficiency'),
        ('python -m src.experiments.run_feature_analysis --tier {tier}',
         'Exp 5: Feature Analysis + SHAP'),
        ('python -m src.experiments.run_ablation --tier {tier}',
         'Exp 6: Ablation Study'),
    ]
    rc = 0
    for cmd_template, desc in experiments:
        cmd = cmd_template.format(tier=TIER)
        r = run(cmd, desc)
        rc = max(rc, r)
        if r != 0:
            print(f"[WARN] {desc} 失败，继续下一个实验...")
    return rc


def main():
    parser = argparse.ArgumentParser(description='Alpha_Crops 实验管线')
    parser.add_argument('--step', required=True,
                        choices=['1', '2', '3', '4', '5', '6', 'all'],
                        help='要运行的步骤')
    args = parser.parse_args()

    t0 = time.time()

    if args.step == '1':
        step1()
    elif args.step == '2':
        step2()
    elif args.step == '3':
        step3()
    elif args.step == '4':
        step4()
    elif args.step == '5':
        step5()
    elif args.step == '6':
        step6()
    elif args.step == 'all':
        # 运行本地步骤（不含 GEE 提取）
        for fn in [step1, step4, step5, step6]:
            rc = fn()
            if rc != 0:
                print(f"\n管线在此步骤中断，请检查错误。")
                sys.exit(1)

    total = time.time() - t0
    print(f"\n{'='*60}")
    print(f"管线完成，总耗时: {total/60:.1f} min ({total/3600:.1f} h)")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
