"""
cheng_metrics.py 的冒烟测试（不读取真实的 guidance_G_ml）：
  1. 盲态保护：任何路径含 guidance_G_ml/ 且未加 --ml-unblinded 时必须拒绝；KIND=ML 未加标志时也必须拒绝。
  2. ML 适配器：把 G（LightGBM）的概率按 multilearner_G.py 的布局复制到一个假的 guidance_G_ml/（位于 --tmp 下），
     以 ML 来源读入，指标应与 G 来源逐位相同。
  3. cheng_scores 的手算小例子（4 类重新归一化、缺类、并列）。
用法：python test_cheng_metrics.py --tmp <临时目录>（默认本目录下 _smoke/，用完可删）
"""
import argparse
import os
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cheng_metrics as CM  # noqa: E402

G = CM.G
PY = sys.executable
CTRY = ['LT', 'PT', 'RO', 'SE']  # LT 缺 potatoes；PT winter_cereals 少；RO 4 类中只有 grassland；去掉 RO、PT 后仍剩 2 国


def test_scores():
    y = np.array([0, 0, 1, 4, 9, 3, 3])
    P = np.zeros((7, 10))
    P[0, 0] = 1                      # 对
    P[1, [0, 3]] = [0.2, 0.8]        # 10 类错成 3；4 类内只剩 grassland → 对
    P[2, [1, 4]] = [0.3, 0.3]        # 4 类内并列 → argmax 取 maize（C4 中靠前）→ 对
    P[3, [1, 4]] = [0.6, 0.4]        # 错成 maize
    P[4, [9, 3]] = [0.1, 0.9]        # 10 类错；4 类内对
    P[5, 3] = 1
    P[6, [3, 0]] = [0.4, 0.6]
    m = CM.cheng_scores(P, y)
    assert m['n_test4'] == 5 and m['oa4'] == 80.0, m
    # 4 类召回：grassland 1、maize 1、potatoes 0、winter_cereals 1 → 75
    assert abs(m['ba4'] - 75.0) < 1e-12, m
    assert m['ties4'] == 1, m
    # 10 类：召回 grassland 1/2、maize 1、potatoes 0、other_crops 1/2、winter_cereals 0 → 均值 40
    assert abs(m['ba_all'] - 40.0) < 1e-12 and m['n_cls_present'] == 5, m
    assert abs(m['oa'] - 100 * 3 / 7) < 1e-12, m
    print('cheng_scores 手算例子：通过')


def build_fake_ml(root, name):
    """按 multilearner_G.run_unit_ml 的文件名与列名，把 G 的 k=0 与 SRS k=200 概率复制过去。"""
    os.makedirs(os.path.join(root, 'probs', name), exist_ok=True)
    src = CM.GSource('lgbm', G.OUT)
    R = src.rows
    for tgt in CTRY:
        for split in G.SPLITS:
            for feat in CM.FEATS:
                P, _, _ = src.k0(tgt, split, feat)
                np.savez_compressed(os.path.join(root, 'probs', name, f'{tgt}_s{split}_{feat}_k0.npz'), P=P,
                                    info='{}', params='x')
            for rep in G.REPS:
                unit = f'{tgt}_s{split}_r{rep}'
                sub = R[(R.target == tgt) & (R.split == split) & (R.rep == rep) &
                        (((R.method == 'srs') & (R.k == 200)) | (R.method == 'foreign_only'))].copy()
                sub['sel_hash'] = np.where(sub.method == 'foreign_only', 'k0', sub.sel_hash)
                sub['status'] = 'ok'
                sub['learner'] = name
                bad = sub[sub.method == 'srs'].iloc[:1].copy()
                bad['status'], bad['oa'] = 'failed', np.nan  # 与正常行同键的失败行：不剔除则 _row 的唯一性断言会失败
                pd.concat([sub, bad]).to_csv(os.path.join(root, f'rows_{name}_{unit}.csv'), index=False)
                for feat in CM.FEATS:
                    P, _, h = src.srs(tgt, split, rep, feat, 200)
                    np.savez_compressed(os.path.join(root, 'probs', name, f'{unit}_{feat}_k200_{h}.npz'), P=P,
                                        info='{}', params='x')


def run(args):
    return subprocess.run([PY, os.path.join(HERE, 'cheng_metrics.py')] + args, capture_output=True, text=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tmp', default=os.path.join(HERE, '_smoke'))
    a = ap.parse_args()
    test_scores()
    fake = os.path.join(a.tmp, 'guidance_G_ml')  # 故意用这个目录名，检验保护
    build_fake_ml(fake, 'fake')
    out = os.path.join(a.tmp, 'out')
    cty = ['--countries', ','.join(CTRY), '--out', out]
    r = run(['--src', f'fake=ML:{fake}'] + cty)
    assert r.returncode != 0 and '未解盲' in (r.stderr + r.stdout), r
    r = run(['--src', f'fake=G:{fake}'] + cty)
    assert r.returncode != 0 and '拒绝读取' in (r.stderr + r.stdout), r
    print('盲态保护：通过')
    r = run(['--src', f'lgbm=G:{G.OUT}', '--src', f'fake=ML:{fake}', '--ml-unblinded'] + cty)
    assert r.returncode == 0, r.stderr
    R = pd.read_csv(os.path.join(out, 'cheng_metrics_rows.csv'))
    assert R.ref_match.all()
    cols = ['oa', 'ba_all', 'ba_ge30', 'oa4', 'ba4', 'ties4', 'ties10']
    key = ['method', 'k', 'feat', 'target', 'split', 'rep']
    a_ = R[R.learner == 'lgbm'].set_index(key)[cols].sort_index()
    b_ = R[R.learner == 'fake'].set_index(key)[cols].sort_index()
    assert len(a_) == len(b_) == len(CTRY) * 2 * 2 * 3, (len(a_), len(b_))
    assert (a_.to_numpy() == b_.to_numpy()).all()
    print(f'ML 适配器：{len(b_)} 行与 G 来源逐位相同，通过')


if __name__ == '__main__':
    main()
