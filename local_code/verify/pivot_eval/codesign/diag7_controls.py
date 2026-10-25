"""诊断 7：对照。ENS 的收益来自"目标头学到的东西"，还是只是"和外国模型平均"或"类均衡"？
C0：0.75·log p_LGB + 0.25·log P_T（不用目标标签的外国模型，无头）；C1：头用普通 CE（alpha=0，不做类均衡）；
C2：LGB 换成第 200 轮（更好调的 LGB），ENS 同样用第 200 轮。全部 τ=0，SRS 与 AE_predstrat_srs（重复 0）+ SRS（重复 1）。"""
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
                Z0 = Zt - Zt.mean(1, keepdims=True)
                Gf = S[f'G_{f}'].astype(np.float64); Gf = Gf / Gf[:, :1].std()
                Xp, Xt = Gf[pp][:, :16], Gf[n_cand:][:, :16]
                for k in (50, 200):
                    for cm in ('none_srs', 'AE_predstrat_srs'):
                        ld = load_sel(unit, *cm.split('_', 1), k)
                        if ld is None:
                            continue
                        sel_pos, sel = ld
                        ys = y_all[sel]
                        h = hashlib.md5(np.sort(np.asarray(sel, dtype=np.int64)).tobytes()).hexdigest()[:16]
                        z = np.load(f'{GOUT}/probs/{unit}_{f}_k{k}_{h}.npz')
                        Lg = logp(z['i300']); Lg -= Lg.mean(1, keepdims=True)
                        Lg2 = logp(z['i200']); Lg2 -= Lg2.mean(1, keepdims=True)
                        pr = fit_head(Zp[sel_pos], ys, 'feat', Xp[sel_pos], 0.03, 1.0)
                        Lh = logp(pr(Zt, Xt)); Lh -= Lh.mean(1, keepdims=True)
                        pr0 = fit_head(Zp[sel_pos], ys, 'feat', Xp[sel_pos], 0.03, 0.0)
                        Lh0 = logp(pr0(Zt, Xt)); Lh0 -= Lh0.mean(1, keepdims=True)
                        for name, L in (('LGB', Lg), ('ENS75', 0.75 * Lg + 0.25 * Lh), ('ENS50', 0.5 * Lg + 0.5 * Lh),
                                        ('C0_75', 0.75 * Lg + 0.25 * Z0), ('C0_50', 0.5 * Lg + 0.5 * Z0),
                                        ('C1_75', 0.75 * Lg + 0.25 * Lh0), ('C1_50', 0.5 * Lg + 0.5 * Lh0),
                                        ('LGB200', Lg2), ('ENS75_200', 0.75 * Lg2 + 0.25 * Lh), ('ENS50_200', 0.5 * Lg2 + 0.5 * Lh)):
                            oa, ba = scores(yT, L.argmax(1), big)
                            rows.append(dict(unit=unit, tgt=tgt, rep=rep, feat=f, k=k, sel=cm, learner=name, ba=ba, oa=oa))
R = pd.DataFrame(rows); R.to_csv(f'{OUT}/diag7_controls_rows.csv', index=False)
m = R.merge(R[R.learner == 'LGB'][['unit', 'feat', 'k', 'sel', 'ba', 'oa']], on=['unit', 'feat', 'k', 'sel'], suffixes=('', '_b'))
m['dba'] = m.ba - m.ba_b; m['doa'] = m.oa - m.oa_b
m2 = R.merge(R[R.learner == 'LGB200'][['unit', 'feat', 'k', 'sel', 'ba', 'oa']], on=['unit', 'feat', 'k', 'sel'], suffixes=('', '_b'))
m2['dba'] = m2.ba - m2.ba_b; m2['doa'] = m2.oa - m2.oa_b
pd.set_option('display.width', 250)
for nm, mm in (('相对 LGB(300 轮)', m), ('相对 LGB(200 轮)', m2[m2.learner.str.endswith('200')])):
    c = mm.groupby(['sel', 'learner', 'k', 'tgt'])[['dba', 'doa']].mean().reset_index()   # 两种特征、单元平均到国家
    a = c.groupby(['sel', 'learner', 'k']).agg(dba=('dba', 'mean'), npos=('dba', lambda x: int((x > 0).sum())), doa=('doa', 'mean'))
    print('=====', nm); print(a.round(2).unstack('k').to_string())
