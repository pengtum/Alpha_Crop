"""
cheng_metrics.py 的冒烟测试（不读取真实的 guidance_G_ml）。v2（按 D2 复核意见补用例；v1 另存为 test_cheng_metrics_v1.py）。
  1. cheng_scores 的手算小例子（4 类重新归一化、缺类、并列）。
  2. 盲态保护：
     - KIND=ML 未加 --ml-unblinded → 拒绝；
     - 路径含 guidance_G_ml/ → 拒绝；
     - 经符号链接指向 guidance_G_ml/ 的路径（原路径不含该名）→ 拒绝（realpath）；
     - 大小写不同的路径（APFS 默认不区分大小写时）→ 拒绝。
  3. ML 适配器：把 G（LightGBM）的概率按 multilearner_G.py 的布局复制到一个假的 guidance_G_ml/（位于 --tmp 下），
     以 ML 来源读入，指标应与 G 来源逐位相同。其中按真实输出的样子构造两处技术失败
     （multilearner_G 不重跑 failed 键，所以只有一条 failed 行、没有概率文件）：
       a. SE s1 r0 的 SRS k=200、特征 ae：failed 行，无概率文件；
       b. PT s0 的 k=0、特征 s2：两个重复的 foreign_only 行都是 failed，无 k0 概率文件。
     预期：不中止；两处记为"缺失或失败"，其配对的另一特征记为"未配对剔除"；汇总报告剔除数；其余行与 G 逐位相同。
     另保留一处"同键 failed 行 + ok 行"（真实输出中不会出现，只作额外稳健性检查）：应取 ok 行。
用法：python test_cheng_metrics.py --tmp <临时目录>（默认本目录下 _smoke/，用完可删）
"""
import argparse
import os
import shutil
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
FAIL_SRS = ('SE', 1, 0, 'ae', 200)  # (国家, 划分, 重复, 特征, k)
FAIL_K0 = ('PT', 0, 's2')  # (国家, 划分, 特征)


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
    # pt_diff：同一概率 → 差为 0（小例子里没有 ≥30 的类，ba_ge30 为 NaN）
    d = CM.pt_diff(P, P.astype(np.float32), y)
    assert d['pt_dargmax10'] == 0 and d['pt_dargmax4'] == 0, d
    assert all(d[f'pt_d_{k}'] == 0 for k in CM.PT_METRICS if k != 'ba_ge30') and np.isnan(d['pt_d_ba_ge30']), d
    print('cheng_scores / pt_diff 手算例子：通过')


def build_fake_ml(root, name):
    """按 multilearner_G.run_unit_ml 的文件名与列名，把 G 的 k=0 与 SRS k=200 概率复制过去，并构造技术失败。"""
    if os.path.exists(root):
        shutil.rmtree(root)
    os.makedirs(os.path.join(root, 'probs', name), exist_ok=True)
    src = CM.GSource('lgbm', G.OUT)
    R = src.rows
    first = True
    for tgt in CTRY:
        for split in G.SPLITS:
            for feat in CM.FEATS:
                if (tgt, split, feat) == FAIL_K0:
                    continue  # 失败：没有 k0 概率文件
                P, _, _ = src.k0(tgt, split, feat)
                np.savez_compressed(os.path.join(root, 'probs', name, f'{tgt}_s{split}_{feat}_k0.npz'), P=P,
                                    info='{}', params='x')
            for rep in G.REPS:
                unit = f'{tgt}_s{split}_r{rep}'
                sub = R[(R.target == tgt) & (R.split == split) & (R.rep == rep) &
                        (((R.method == 'srs') & (R.k == 200)) | (R.method == 'foreign_only'))].copy()
                sub['sel_hash'] = np.where(sub.method == 'foreign_only', 'k0', sub.sel_hash)
                sub['status'], sub['error'], sub['learner'] = 'ok', '', name
                # 真实形态的失败：只有一条 failed 行（指标为空），没有概率文件
                fk0 = (sub.method == 'foreign_only') & ((tgt, split) == FAIL_K0[:2]) & (sub.clf == FAIL_K0[2])
                fsrs = ((sub.method == 'srs') & ((tgt, split, rep) == FAIL_SRS[:3]) & (sub.clf == FAIL_SRS[3]) &
                        (sub.k == FAIL_SRS[4]))
                for f in (fk0, fsrs):
                    sub.loc[f, 'status'], sub.loc[f, 'error'] = 'failed', "ValueError('fake')"
                    sub.loc[f, ['oa', 'ba_ge30']] = np.nan
                parts = [sub]
                if first:  # 额外稳健性：同键 failed + ok（真实输出中不会出现）
                    bad = sub[(sub.method == 'srs') & (sub.status == 'ok')].iloc[:1].copy()
                    bad['status'], bad['oa'] = 'failed', np.nan
                    parts.append(bad)
                    first = False
                pd.concat(parts).to_csv(os.path.join(root, f'rows_{name}_{unit}.csv'), index=False)
                for feat in CM.FEATS:
                    if (tgt, split, rep, feat, 200) == FAIL_SRS:
                        continue  # 失败：没有概率文件
                    P, _, h = src.srs(tgt, split, rep, feat, 200)
                    np.savez_compressed(os.path.join(root, 'probs', name, f'{unit}_{feat}_k200_{h}.npz'), P=P,
                                        info='{}', params='x')


def run(args):
    return subprocess.run([PY, '-B', os.path.join(HERE, 'cheng_metrics.py')] + args, capture_output=True, text=True)


def test_guard(tmp, fake, cty):
    r = run(['--src', f'fake=ML:{fake}'] + cty)
    assert r.returncode != 0 and '未解盲' in (r.stderr + r.stdout), r
    r = run(['--src', f'fake=G:{fake}'] + cty)
    assert r.returncode != 0 and '拒绝读取' in (r.stderr + r.stdout), r
    # 符号链接：原路径不含 guidance_G_ml
    alias = os.path.join(tmp, 'innocent_alias')
    if os.path.lexists(alias):
        os.remove(alias)
    os.symlink(fake, alias)
    assert 'guidance_g_ml' not in alias.lower()
    assert CM.BlindGuard.blocked(alias) and CM.BlindGuard.blocked(os.path.join(alias, 'probs', 'x.npz'))
    r = run(['--src', f'fake=G:{alias}'] + cty)
    assert r.returncode != 0 and '拒绝读取' in (r.stderr + r.stdout), r
    # 指向单个文件的符号链接
    flink = os.path.join(tmp, 'innocent_file.csv')
    if os.path.lexists(flink):
        os.remove(flink)
    target = sorted(f for f in os.listdir(fake) if f.endswith('.csv'))[0]
    os.symlink(os.path.join(fake, target), flink)
    assert CM.BlindGuard.blocked(flink)
    try:
        CM._read_csv(flink)
        raise AssertionError('经符号链接读取 guidance_G_ml 下的文件应被拒绝')
    except SystemExit as e:
        assert '拒绝读取' in str(e)
    # 大小写变体（APFS 默认不区分大小写时，这个路径指向同一目录）
    upper = os.path.join(tmp, 'GUIDANCE_G_ML')
    assert CM.BlindGuard.blocked(upper)
    if os.path.exists(upper):
        r = run(['--src', f'fake=G:{upper}'] + cty)
        assert r.returncode != 0 and '拒绝读取' in (r.stderr + r.stdout), r
        print('  （文件系统不区分大小写：已用 GUIDANCE_G_ML 路径实跑确认被拒绝）')
    # 不相关的路径不应被误拦
    assert not CM.BlindGuard.blocked(G.OUT)
    assert not CM.BlindGuard.blocked(os.path.join(tmp, 'guidance_G_ml_notes'))
    print('盲态保护（含符号链接、大小写）：通过')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tmp', default=os.path.join(HERE, '_smoke'))
    a = ap.parse_args()
    os.makedirs(a.tmp, exist_ok=True)
    test_scores()
    fake = os.path.join(a.tmp, 'guidance_G_ml')  # 故意用这个目录名，检验保护
    build_fake_ml(fake, 'fake')
    out = os.path.join(a.tmp, 'out')
    cty = ['--countries', ','.join(CTRY), '--out', out]
    test_guard(a.tmp, fake, cty)

    r = run(['--src', f'lgbm=G:{G.OUT}', '--src', f'fake=ML:{fake}', '--ml-unblinded'] + cty)
    assert r.returncode == 0, r.stderr
    R = pd.read_csv(os.path.join(out, 'cheng_metrics_rows.csv'))
    K = pd.read_csv(os.path.join(out, 'cheng_metrics_skipped.csv'))
    txt = open(os.path.join(out, 'cheng_metrics_summary.txt'), encoding='utf-8').read()
    assert R.ref_match.all()
    # 剔除记录：两处"缺失或失败"，两处"未配对剔除"，都属于 fake
    assert (K.learner == 'fake').all() and len(K) == 4, K
    miss = K[K.stage == '缺失或失败']
    unp = K[K.stage == '未配对剔除']
    got_m = set(map(tuple, miss[['method', 'k', 'feat', 'target', 'split', 'rep']].to_numpy().tolist()))
    exp_m = {('srs', 200, 'ae', 'SE', 1, 0), ('foreign_only', 0, 's2', 'PT', 0, -1)}
    assert got_m == exp_m, (got_m, exp_m)
    got_u = set(map(tuple, unp[['method', 'k', 'feat', 'target', 'split', 'rep']].to_numpy().tolist()))
    exp_u = {('srs', 200, 's2', 'SE', 1, 0), ('foreign_only', 0, 'ae', 'PT', 0, -1)}
    assert got_u == exp_u, (got_u, exp_u)
    assert '缺失或失败 2 个' in txt and '未配对剔除 2 个' in txt, '汇总没有报告剔除数'
    assert 'status≠ok 的 foreign_only/srs 行 4 条' in txt, '汇总没有报告 failed 行数'  # 2（k0 两个重复）+1（srs）+1（同键）
    # 行数与逐位一致
    cols = ['oa', 'ba_all', 'ba_ge30', 'oa4', 'ba4', 'ties4', 'ties10']
    key = ['method', 'k', 'feat', 'target', 'split', 'rep']
    a_ = R[R.learner == 'lgbm'].set_index(key)[cols].sort_index()
    b_ = R[R.learner == 'fake'].set_index(key)[cols].sort_index()
    n_all = len(CTRY) * 2 * 2 * 3
    assert len(a_) == n_all and len(b_) == n_all - 4, (len(a_), len(b_))
    assert (a_.loc[b_.index].to_numpy() == b_.to_numpy()).all()
    # 配对：fake 的每个单元都恰有两种特征
    assert (R.groupby(CM.PAIR_KEY).feat.nunique() == 2).all()
    # P_T 比较只对 G 来源做
    assert R[(R.learner == 'lgbm') & (R.k == 0)].pt_d_oa.notna().all()
    assert R[R.learner == 'fake'].pt_d_oa.isna().all()
    print(f'ML 适配器：{len(b_)} 行与 G 来源逐位相同；2 处技术失败被记录并剔除（另 2 行未配对剔除），未中止：通过')


if __name__ == '__main__':
    main()
