"""审计 1：ORACLE 指导在候选池（重复 0）上的袋外质量 vs 外国模型 P_T_ae / P_T_s2；行对齐检查。只读。"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.getcwd())
import guidance_G as G

OUT = '/Users/perng_mini/AlphaCrops_local/results/guidance_G'
D = G.GData()


def metrics(P, y, classes_big):
    ph = P.argmax(1)
    acc = (ph == y).mean()
    rec = np.array([(ph[y == c] == c).mean() if (y == c).any() else np.nan for c in range(10)])
    ba_all = np.nanmean(rec)
    ba_big = np.nanmean(rec[classes_big])
    nll = -np.log(np.clip(P[np.arange(len(y)), y], 1e-12, None)).mean()
    return 100 * acc, 100 * ba_all, 100 * ba_big, nll


rows = []
for tgt in G.COUNTRIES_G:
    for split in (0, 1):
        S = dict(np.load(os.path.join(OUT, 'stage0', f'{tgt}_s{split}.npz')))
        O = dict(np.load(os.path.join(OUT, 'oracle', f'{tgt}_s{split}.npz')))
        pp = G.pool_positions(S, split, 0)
        idx = S['allT'][pp]
        y = D.y[idx]
        assert O['P_T'].shape[0] == len(pp) == O['G'].shape[0], (tgt, split)
        # 候选池内所有位置都 < n_cand（不含测试集）
        assert pp.max() < len(S['cand'])
        assert not np.isin(idx, S['test']).any()
        cnt = np.bincount(y, minlength=10)
        big = np.flatnonzero(cnt >= 30)
        r = dict(tgt=tgt, split=split, N=len(y), temp=float(O['temp']), n_cls_pool=int((cnt > 0).sum()))
        for name, P in (('orc', O['P_T']), ('orc_raw', O['P_oof']), ('ae', S['P_T_ae'][pp]), ('s2', S['P_T_s2'][pp])):
            a, b, bb, nll = metrics(np.asarray(P, float), y, big)
            r.update({f'acc_{name}': a, f'ba_{name}': b, f'babig_{name}': bb, f'nll_{name}': nll})
        # LDA 判别分数 argmax = LDA 预测
        a, b, bb, _ = metrics(O['G'].astype(float) - O['G'].max(1, keepdims=True), y, big)
        r.update(acc_lda=a, ba_lda=b, babig_lda=bb)
        # 对齐反证：把 oracle 行随机打乱后准确率应掉到约先验最大值
        perm = np.random.default_rng(0).permutation(len(y))
        r['acc_orc_shuffled'] = 100 * (O['P_T'][perm].argmax(1) == y).mean()
        r['acc_majority'] = 100 * cnt.max() / len(y)
        # G 各列尺度（检查方案"对 C 列做稳健 z 截断"是否执行）
        Gs = O['G'].std(0)
        r['G_colstd_min'], r['G_colstd_max'] = float(Gs.min()), float(Gs.max())
        r['G_absmax'] = float(np.abs(O['G']).max())
        rows.append(r)
        print(tgt, split, {k: round(v, 3) if isinstance(v, float) else v for k, v in r.items() if k.startswith(('acc', 'ba_', 'temp', 'G_'))}, flush=True)

df = pd.DataFrame(rows)
df.to_csv('/Users/perng_mini/AlphaCrops_local/verify/G_audit/oracle/a1_oracle_quality.csv', index=False)
cols = [c for c in df.columns if c not in ('tgt',)]
print(df.groupby('tgt')[[c for c in cols if c.startswith(('acc', 'ba'))]].mean().round(1).to_string())
print('\n总体均值：')
print(df[[c for c in cols if c.startswith(('acc', 'ba', 'nll', 'temp', 'G_'))]].mean().round(3).to_string())
