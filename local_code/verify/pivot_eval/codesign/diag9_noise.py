"""诊断 9：目视解译噪声敏感性（只扰动目标头的训练标签；LGB 用无噪声标签的缓存结果，所以只反映"头"对噪声的敏感度）。
噪声：以概率 e 把标签均匀改成其他 9 类之一（e = 0, 0.05, 0.10, 0.20），各 3 个噪声种子。"""
import sys, hashlib
import numpy as np, pandas as pd
sys.path.insert(0, '/Users/perng_mini/AlphaCrops_local/verify/pivot_eval/codesign')
from headlib import *  # noqa
y_all = np.load(f'{CACHE}/y.npy')
rows = []
for tgt in COUNTRIES:
    for split in (0, 1):
        S = load_stage0(tgt, split)
        n_cand = len(S['cand']); test = S['test']; yT = y_all[test]
        big = np.flatnonzero(np.bincount(yT, minlength=C) >= MIN_N)
        unit = f'{tgt}_s{split}_r0'
        pp = pool_positions(S, split, 0)
        for f in ('ae', 's2'):
            Zall = logp(S[f'P_T_{f}']); Zp, Zt = Zall[pp], Zall[n_cand:]
            Gf = S[f'G_{f}'].astype(np.float64); Gf = Gf / Gf[:, :1].std()
            Xp, Xt = Gf[pp][:, :16], Gf[n_cand:][:, :16]
            for k in (50, 200):
                sel_pos, sel = load_sel(unit, 'AE', 'predstrat_srs', k)
                ys = y_all[sel]
                h = hashlib.md5(np.sort(np.asarray(sel, dtype=np.int64)).tobytes()).hexdigest()[:16]
                Lg = logp(np.load(f'{GOUT}/probs/{unit}_{f}_k{k}_{h}.npz')['i300']); Lg -= Lg.mean(1, keepdims=True)
                oa_b, ba_b = scores(yT, Lg.argmax(1), big)
                for e in (0.0, 0.05, 0.1, 0.2):
                    for s in range(3 if e > 0 else 1):
                        rng = np.random.default_rng(100 * s + int(e * 100))
                        yn = ys.copy()
                        flip = rng.random(k) < e
                        yn[flip] = (yn[flip] + rng.integers(1, C, flip.sum())) % C
                        pr = fit_head(Zp[sel_pos], yn, 'feat', Xp[sel_pos], 0.03, 1.0)
                        Lh = logp(pr(Zt, Xt)); Lh -= Lh.mean(1, keepdims=True)
                        for nm, L in (('ENS75', 0.75 * Lg + 0.25 * Lh), ('ENS50', 0.5 * Lg + 0.5 * Lh)):
                            oa, ba = scores(yT, L.argmax(1), big)
                            rows.append(dict(tgt=tgt, split=split, feat=f, k=k, e=e, s=s, learner=nm, dba=ba - ba_b, doa=oa - oa_b))
R = pd.DataFrame(rows); R.to_csv(f'{OUT}/diag9_noise_rows.csv', index=False)
c = R.groupby(['learner', 'k', 'e', 'tgt'])[['dba', 'doa']].mean().reset_index()
print(c.groupby(['learner', 'k', 'e']).agg(dba=('dba', 'mean'), npos=('dba', lambda x: int((x > 0).sum())), doa=('doa', 'mean')).round(2).unstack('e').to_string())
