"""
确认运行 v1 的训练运行器（CONFIRM_PLAN_v5.md §2、§7；sha256[:16]=5384fb7efa67f495）。只训练并保存概率，不计算任何指标。

读 {root}/splits/{unit}.npz 与 {root}/selection/{unit}.npz（confirm_split.py 生成），按方案 §2 拟合：
  LightGBM（stageA.lgb_params，300 轮，取第 300 轮）
    Z0             外国 5 万；种子 confirm_v1_stage0_{单元}
    RT{k}          外国 + 候选池 π_r 的前 k 个，k ∈ 50/100/200/400，r = 0..2
    TGT            只用候选池，类别权重按池频率；种子 confirm_v1_tgt_{单元}
    B_S0           外国 + S0_r
    B_{SRS,EXT,BA}{100,300}   外国 + S0_r ∪ S1
  RF（multilearner_G.fit_predict_learner，500 棵，sqrt，n_jobs = 线程数）：Z0 与 RT200
  外国样本权重 cw[y]，目标样本权重 50·cw[y]，cw = stageA.class_weights_from(外国池标签)。
  其余拟合的种子 confirm_v1_fit_{单元}_r{r}（同一抽样内各 k、各特征、各臂共用）；种子 = stable_hash(串) % 2**31。
输出（按拟合原子写入，可续跑）：
  probs/{unit}/{learner}_{arm}_{feat}_r{r}.npz   P_test（float32，测试行）、P_pool（float16，候选池行）；Z0、TGT 写 r0
  runs/{unit}.csv   只记元数据：learner、arm、feat、r、k（目标标签数）、sel_hash、train_sha、seed、n_train、trees、
                    threads、t_fit、status、attempts、error、peak_rss_mb、code_hash、splits_hash、selection_hash
失败的拟合用同一种子重试一次，仍失败记 status=failed（不写概率文件）。
启动时断言：方案与冻结模块的 sha256、数据 source stamp、库版本；正式模式另要求本代码与 split_manifest.csv 的
sha256[:16] 已登记在 plan_hashes.txt、每个单元的划分与选样内容哈希与 split_manifest.csv 一致、每进程 2 线程
（§2.3；LightGBM 的 deterministic 只在线程数相同时可复现）。续跑时已有记录的哈希与线程数须与当前相同。
工作进程被系统杀死（OOM、段错误）时主进程退出并提示续跑（已完成的拟合已原子写入）。

用法（/Users/perng_mini/AlphaCrops_local/.venv/bin/python -B）：
  confirm_run.py --root /Users/perng_mini/AlphaCrops_local/results/confirm_v1 --learner lgb --jobs 5
  confirm_run.py --root /Users/perng_mini/AlphaCrops_local/results/confirm_v1 --learner rf --jobs N   # N 按冒烟实测峰值内存
  confirm_run.py --smoke --root /Users/perng_mini/AlphaCrops_local/verify/confirm_code_scratch/smoke --countries AT --jobs 2
      冒烟：旧 G s0/s1 划分（confirm_split.build_smoke_unit），LightGBM 5 轮、RF 10 棵；两个学习器依次运行
"""
import os
import sys
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '2'
sys.dont_write_bytecode = True  # 不在项目目录生成 __pycache__
import argparse  # noqa: E402
import fcntl  # noqa: E402
import hashlib  # noqa: E402
import multiprocessing as mp  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from concurrent.futures.process import BrokenProcessPool  # noqa: E402
import re  # noqa: E402
import resource  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import confirm_split as CS  # noqa: E402  （导入时把 label_expansion 加入 sys.path）
from confirm_split import need  # noqa: E402
G = CS.G
import multilearner_G as ML  # noqa: E402
from stageA import class_weights_from, lgb_params, save_csv  # noqa: E402

FEATS = ('ae', 's2')
R_MULT = 50.0
TREES = {'lgb': 300, 'rf': 500}
SMOKE_TREES = {'lgb': 5, 'rf': 10}
MAX_THREADS = 2
RF_PARAMS = dict(n_estimators=500, max_features='sqrt', min_samples_leaf=1, bootstrap=True, class_weight=None)  # rev2
ARMS_B = ['B_S0'] + [f'B_{s}{k1}' for s in CS.STRATS for k1 in CS.K1]
ARMS = ['Z0'] + [f'RT{k}' for k in CS.K_A] + ['TGT'] + ARMS_B      # 与接口约定的 13 个臂相同
RUN_COLS = ['unit', 'learner', 'arm', 'feat', 'r', 'k', 'sel_hash', 'train_sha', 'seed', 'n_train', 'trees', 'threads',
            't_fit', 'status', 'attempts', 'error', 'peak_rss_mb', 'code_hash', 'splits_hash', 'selection_hash']
STR_COLS = {c: str for c in ('unit', 'learner', 'arm', 'feat', 'sel_hash', 'train_sha', 'status', 'error', 'code_hash',
                               'splits_hash', 'selection_hash')}
need(G.R_MULT == R_MULT, 'guidance_G.R_MULT 与方案的 50 不一致')


def read_runs(path):
    """runs CSV：哈希列按字符串读（避免被解析成数字）。"""
    return pd.read_csv(path, dtype=STR_COLS, keep_default_na=False)


def fit_plan(learner):
    """[(arm, feat, r)]，顺序即运行顺序。LightGBM 每单元 70 次，RF 8 次。"""
    if learner == 'lgb':
        J = [('Z0', f, 0) for f in FEATS]
        J += [(f'RT{k}', f, r) for r in range(CS.R) for k in CS.K_A for f in FEATS]
        J += [('TGT', f, 0) for f in FEATS]
        J += [(a, f, r) for r in range(CS.R) for a in ARMS_B for f in FEATS]
        return J
    need(learner == 'rf', learner)
    return [('Z0', f, 0) for f in FEATS] + [('RT200', f, r) for r in range(CS.R) for f in FEATS]


def probs_path(root, unit, learner, arm, feat, r):
    return os.path.join(root, 'probs', unit, f'{learner}_{arm}_{feat}_r{r}.npz')


def train_rows(arm, r, sp, sel):
    """返回 (外国下标, 目标下标, 种子串)。"""
    fr, pool, unit = sp['foreign'], sp['pool'], str(sp['unit'])
    none = np.array([], np.int64)
    if arm == 'Z0':
        return fr, none, f'confirm_v1_stage0_{unit}'
    if arm == 'TGT':
        return none, pool, f'confirm_v1_tgt_{unit}'
    seed = f'confirm_v1_fit_{unit}_r{r}'
    if arm.startswith('RT'):
        need(arm[2:].isdigit() and int(arm[2:]) in CS.K_A, f'未知的臂 {arm}')
        return fr, pool[sel[f'pi_r{r}'][:int(arm[2:])]], seed
    S0 = sel[f'S0_r{r}']
    if arm == 'B_S0':
        return fr, S0, seed
    m = re.fullmatch(r'B_(SRS|EXT|BA)(\d+)', arm)
    need(m is not None and int(m.group(2)) in CS.K1, f'未知的臂 {arm}')
    return fr, np.r_[S0, sel[f'S1_{m.group(1)}_k{m.group(2)}_r{r}']], seed


def sample_weights(arm, y_f, y_t, cw):
    """外国 cw[y]、目标 50·cw[y]（cw 按外国池频率）；TGT 只有目标样本，权重按池频率。"""
    if arm == 'TGT':
        return class_weights_from(y_t)[y_t]
    return np.r_[cw[y_f], R_MULT * cw[y_t]]


def fit_predict(learner, X_tr, y_tr, w_tr, X_te, seed, threads, trees):
    """训练一次，返回 X_te 的概率 (n, 10) float64；概率不合法时报错（触发重试）。"""
    if learner == 'lgb':
        import lightgbm as lgb
        m = lgb.train(lgb_params(threads, seed), lgb.Dataset(X_tr, y_tr, weight=w_tr, free_raw_data=True),
                      num_boost_round=trees)
        P = m.predict(X_te, num_iteration=trees)
    else:
        need(ML.PARAMS['rf'] == dict(RF_PARAMS, n_estimators=trees), f'RF 参数不符：{ML.PARAMS["rf"]}')
        P, _ = ML.fit_predict_learner('rf', X_tr, y_tr, w_tr, X_te, seed, threads)
    P = np.asarray(P, dtype=np.float64)
    if P.shape != (len(X_te), 10) or not np.isfinite(P).all() or np.abs(P.sum(1) - 1).max() > 1e-4:
        raise FloatingPointError(f'{learner}：概率形状或取值不合法 {P.shape}')
    return P


def peak_rss_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(r / 2 ** 20 if sys.platform == 'darwin' else r / 2 ** 10, 1)


def load_unit(root, unit):
    with np.load(os.path.join(root, 'splits', f'{unit}.npz')) as z:
        sp = dict(z)
    with np.load(os.path.join(root, 'selection', f'{unit}.npz')) as z:
        sel = dict(z)
    return sp, sel


# ---------------------------------------------------------------- 一个 (单元, 学习器)
def run_unit(D, root, unit, learner, threads, trees, code_hash, log=print, fitter=fit_predict):
    """按 fit_plan 依次拟合；已完成（ok 且概率文件存在，或 failed）的跳过。返回本次新拟合的次数。"""
    sp, sel = load_unit(root, unit)
    need(sp['feasible'].dtype == np.bool_, f'{unit}：划分未冻结 big（dry-run 输出），不能用于训练')
    if not bool(sp['feasible']):
        log(f'{unit}：单元不可行（方案 §1），跳过')
        return 0
    h_sp, h_sel = CS.content_hash(sp), CS.content_hash(sel)
    y = np.asarray(D.y)
    test, pool, fr = sp['test'], sp['pool'], sp['foreign']
    cw = class_weights_from(y[fr])
    os.makedirs(os.path.join(root, 'probs', unit), exist_ok=True)
    os.makedirs(os.path.join(root, 'runs'), exist_ok=True)
    os.makedirs(os.path.join(root, 'locks'), exist_ok=True)
    rpath = os.path.join(root, 'runs', f'{unit}.csv')
    with open(os.path.join(root, 'locks', f'{unit}.lock'), 'w') as lk:
        fcntl.flock(lk, fcntl.LOCK_EX | fcntl.LOCK_NB)  # 同一单元同时只允许一个进程写
        rows = read_runs(rpath).to_dict('records') if os.path.exists(rpath) else []
        for r_ in rows:
            need(r_['splits_hash'] == h_sp and r_['selection_hash'] == h_sel and r_['code_hash'] == code_hash,
                 f'{unit}：已有记录的划分、选样或代码哈希与当前不同，不续跑')
            need(int(r_['threads']) == threads, f'{unit}：已有记录的线程数 {r_["threads"]} 与当前 {threads} 不同，不续跑')
        rows = [r_ for r_ in rows if r_['status'] == 'failed' or
                os.path.exists(probs_path(root, unit, r_['learner'], r_['arm'], r_['feat'], r_['r']))]
        done = {(r_['learner'], r_['arm'], r_['feat'], int(r_['r'])) for r_ in rows}
        n_new = 0
        for arm, feat, r in fit_plan(learner):
            if (learner, arm, feat, r) in done:
                continue
            fi, ti, seed_str = train_rows(arm, r, sp, sel)
            seed = CS.seed_of(seed_str)
            need(not np.isin(ti, test).any() and not np.isin(fi, test).any(), f'{unit} {arm}：训练集含测试地块')
            X = D.X[feat]
            X_tr = np.vstack([np.asarray(X[fi]), np.asarray(X[ti])])
            y_tr = np.r_[y[fi], y[ti]]
            w_tr = sample_weights(arm, y[fi], y[ti], cw)
            X_te = np.vstack([np.asarray(X[test]), np.asarray(X[pool])])
            t0 = time.time()
            status, err, att = 'failed', '', 0
            for att in (1, 2):  # 失败用同一种子重试一次
                try:
                    P = fitter(learner, X_tr, y_tr, w_tr, X_te, seed, threads, trees)
                    status, err = 'ok', ''
                    break
                except Exception as e:  # noqa: BLE001
                    err = repr(e)[:300]
            t_fit = time.time() - t0
            if status == 'ok':
                pp = probs_path(root, unit, learner, arm, feat, r)
                tmp = f'{pp}.{os.getpid()}.tmp.npz'
                np.savez_compressed(tmp, P_test=P[:len(test)].astype(np.float32), P_pool=P[len(test):].astype(np.float16))
                os.replace(tmp, pp)
            train_idx = np.r_[fi, ti].astype(np.int64)
            rows.append(dict(unit=unit, learner=learner, arm=arm, feat=feat, r=r, k=len(ti),
                             sel_hash=G.sel_hash(ti) if len(ti) else 'k0',
                             train_sha=hashlib.sha256(train_idx.tobytes()).hexdigest()[:16], seed=seed,
                             n_train=len(train_idx), trees=trees, threads=threads, t_fit=round(t_fit, 1), status=status,
                             attempts=att,
                             error=err, peak_rss_mb=peak_rss_mb(), code_hash=code_hash, splits_hash=h_sp,
                             selection_hash=h_sel))
            save_csv([{c: r_[c] for c in RUN_COLS} for r_ in rows], rpath)
            n_new += 1
            log(f'{unit} {learner} {arm} {feat} r{r} k={len(ti)} {t_fit:.1f}s {status}')
    return n_new


# ---------------------------------------------------------------- 进程与主程序
def init_worker(learner, trees):
    task.D = G.GData()
    if learner == 'rf':
        ML.PARAMS['rf'] = dict(RF_PARAMS, n_estimators=trees)  # 正式 500（与 rev2 相同）；冒烟 10


def task(args):
    root, unit, learner, threads, trees, code_hash = args
    n = run_unit(task.D, root, unit, learner, threads, trees, code_hash, log=lambda s: print(s, flush=True))
    return f'{unit} {learner}：新拟合 {n} 次'


def unit_list(root, countries, smoke):
    if smoke:
        return [f'{c}_s{s}' for c in countries for s in (0, 1)]
    return [f'{c}_F{h}' for c in countries for h in CS.HALVES]


def check_manifest(root, units):
    """正式模式：split_manifest.csv 的 sha256[:16] 已登记（§7 第 4 步），每个单元的划分、选样内容哈希与之一致。"""
    CS.check_registered(os.path.join(root, 'split_manifest.csv'))
    M = pd.read_csv(os.path.join(root, 'split_manifest.csv'), dtype=str).set_index('unit')
    for u in units:
        sp, sel = load_unit(root, u)
        need(CS.content_hash(sp) == M.loc[u, 'splits_hash'] and CS.content_hash(sel) == M.loc[u, 'selection_hash'],
             f'{u}：划分或选样与 split_manifest.csv 不一致')


def build_smoke(root, countries):
    """冒烟：在旧 G s0/s1 上生成划分与选样清单（已存在则核对内容相同）。"""
    D = G.GData()
    rows = []
    for c in countries:
        for s in (0, 1):
            sp, sel = CS.build_smoke_unit(D, c, s)
            CS.write_unit(root, sp, sel)
            rows.append(CS.manifest_row(sp, sel))
    save_csv(rows, os.path.join(root, 'split_manifest.csv'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--learner', choices=['lgb', 'rf'], default='lgb')
    ap.add_argument('--jobs', type=int, default=5)
    ap.add_argument('--threads', type=int, default=MAX_THREADS)
    ap.add_argument('--countries', default=None, help='逗号分隔；默认全部 11 国（冒烟默认 AT）')
    ap.add_argument('--smoke', action='store_true', help='旧 G s0/s1 划分，LightGBM 5 轮、RF 10 棵，依次跑两个学习器')
    a = ap.parse_args()
    need(1 <= a.threads <= MAX_THREADS, f'每进程线程数 ≤ {MAX_THREADS}')
    need(a.smoke or a.threads == MAX_THREADS, f'正式运行每进程线程数须为 {MAX_THREADS}（方案 §2.3）')
    root = os.path.realpath(a.root)
    lines = CS.check_env()
    if a.smoke:
        need(CS.under(root, os.path.join(CS.SCRATCH, 'smoke')), f'--smoke 只能写到 {CS.SCRATCH}/smoke 下')
        countries = (a.countries or 'AT').split(',')
        learners, trees = ['lgb', 'rf'], SMOKE_TREES
    else:
        need(not CS.under(root, CS.SCRATCH), '正式运行不写到 scratch')
        CS.check_frozen_self()
        countries = a.countries.split(',') if a.countries else CS.COUNTRIES
        learners, trees = [a.learner], TREES
    need(set(countries) <= set(CS.COUNTRIES), countries)
    code_hash = CS.code_hash()
    if a.smoke:
        build_smoke(root, countries)
    units = unit_list(root, countries, a.smoke)
    if not a.smoke:
        check_manifest(root, units)
    print('\n'.join(lines), flush=True)
    print(f'confirm_run.py code_hash={code_hash}；{"冒烟" if a.smoke else "正式"}；学习器 {learners}；轮数/棵数 '
          f'{ {k: trees[k] for k in learners} }；{a.jobs} 进程 × {a.threads} 线程；单元 {len(units)}；输出 {root}',
          flush=True)
    t0 = time.time()
    ctx = mp.get_context('spawn')
    for learner in learners:
        ex = ProcessPoolExecutor(a.jobs, mp_context=ctx, initializer=init_worker, initargs=(learner, trees[learner]))
        try:
            futs = [ex.submit(task, (root, u, learner, a.threads, trees[learner], code_hash)) for u in units]
            for f in as_completed(futs):
                print(f'[{time.time() - t0:.0f}s] 完成：{f.result()}', flush=True)
        except BrokenProcessPool:
            ex.shutdown(wait=False, cancel_futures=True)
            raise SystemExit('工作进程异常退出（可能是内存不足或段错误）。已完成的拟合已原子写入；'
                             '调整 --jobs 后用同样的命令重新运行即可续跑。')
        except BaseException:
            ex.shutdown(wait=False, cancel_futures=True)
            raise
        ex.shutdown(wait=True)
    # 完整性（只数行与状态，不读概率）
    for u in units:
        p = os.path.join(root, 'runs', f'{u}.csv')
        if not os.path.exists(p):
            print(f'{u}：无记录（不可行或未运行）', flush=True)
            continue
        R_ = read_runs(p)
        for learner in learners:
            x = R_[R_.learner == learner]
            print(f'{u} {learner}：应 {len(fit_plan(learner))}，ok {int((x.status == "ok").sum())}，'
                  f'failed {int((x.status == "failed").sum())}，峰值 RSS {x.peak_rss_mb.max()} MB', flush=True)
    print(f'总耗时 {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
