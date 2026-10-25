"""诊断 6：固定配置的配对统计（τ=0，不再调参）。
候选学习器：ENS = 0.5·log p_LGB + 0.5·log p_head(feat, λ=0.03, 类均衡)；ENS75（β=0.75）；LGB+0.5·b（外国模型上学到的偏置）。
(a) 重复 0：SRS 与 AE_predstrat_srs 两种选样（LGB 概率已缓存）；(b) 重复 1：只有 SRS（β、γ 只在重复 0 上看过，作半独立核对）。"""
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
        for rep in (0, 1):
            unit = f'{tgt}_s{split}_r{rep}'
            pp = pool_positions(S, split, rep)
            for f in ('ae', 's2'):
                Zall = logp(S[f'P_T_{f}']); Zp, Zt = Zall[pp], Zall[n_cand:]
                Gf = S[f'G_{f}'].astype(np.float64); Gf = Gf / Gf[:, :1].std()
                Xp, Xt = Gf[pp][:, :16], Gf[n_cand:][:, :16]
                for k in (50, 200):
                    for cm in ('none_srs', 'AE_predstrat_srs', 'S2_predstrat_srs'):
                        ld = load_sel(unit, *cm.split('_', 1), k)
                        if ld is None:
                            continue
                        sel_pos, sel = ld
                        ys = y_all[sel]
                        h = hashlib.md5(np.sort(np.asarray(sel, dtype=np.int64)).tobytes()).hexdigest()[:16]
                        Lg = logp(np.load(f'{GOUT}/probs/{unit}_{f}_k{k}_{h}.npz')['i300']); Lg -= Lg.mean(1, keepdims=True)
                        pr = fit_head(Zp[sel_pos], ys, 'feat', Xp[sel_pos], 0.03, 1.0)
                        Lh = logp(pr(Zt, Xt)); Lh -= Lh.mean(1, keepdims=True)
                        prb = fit_head(Zp[sel_pos], ys, 'bias', None, 0.01, 1.0)
                        b = prb.theta[1:1 + C] * (np.bincount(ys, minlength=C) > 0)
                        for name, L in (('LGB', Lg), ('HEAD', Lh), ('ENS50', 0.5 * Lg + 0.5 * Lh),
                                        ('ENS75', 0.75 * Lg + 0.25 * Lh), ('LGB+b50', Lg + 0.5 * b)):
                            oa, ba = scores(yT, L.argmax(1), big)
                            rows.append(dict(unit=unit, tgt=tgt, split=split, rep=rep, feat=f, k=k, sel=cm, learner=name, ba=ba, oa=oa))
R = pd.DataFrame(rows)
R.to_csv(f'{OUT}/diag6_confirm_rows.csv', index=False)
