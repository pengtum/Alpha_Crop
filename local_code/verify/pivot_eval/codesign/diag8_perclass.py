"""诊断 8：ENS 的收益落在哪些类上（逐类召回，k=200，重复 0，AE_predstrat_srs 与 SRS）。"""
import sys, hashlib, json
import numpy as np, pandas as pd
sys.path.insert(0, '/Users/perng_mini/AlphaCrops_local/verify/pivot_eval/codesign')
from headlib import *  # noqa
y_all = np.load(f'{CACHE}/y.npy')
CLS = json.load(open(f'{CACHE}/meta.json'))['classes']
rows = []
for tgt in COUNTRIES:
    for split in (0, 1):
        S = load_stage0(tgt, split)
        n_cand = len(S['cand']); test = S['test']; yT = y_all[test]
        nT = np.bincount(yT, minlength=C)
        unit = f'{tgt}_s{split}_r0'
        pp = pool_positions(S, split, 0)
        for f in ('ae', 's2'):
            Zall = logp(S[f'P_T_{f}']); Zp, Zt = Zall[pp], Zall[n_cand:]
            Gf = S[f'G_{f}'].astype(np.float64); Gf = Gf / Gf[:, :1].std()
            Xp, Xt = Gf[pp][:, :16], Gf[n_cand:][:, :16]
            for cm in ('none_srs', 'AE_predstrat_srs'):
                sel_pos, sel = load_sel(unit, *cm.split('_', 1), 200)
                ys = y_all[sel]
                h = hashlib.md5(np.sort(np.asarray(sel, dtype=np.int64)).tobytes()).hexdigest()[:16]
                Lg = logp(np.load(f'{GOUT}/probs/{unit}_{f}_k200_{h}.npz')['i300']); Lg -= Lg.mean(1, keepdims=True)
                pr = fit_head(Zp[sel_pos], ys, 'feat', Xp[sel_pos], 0.03, 1.0)
                Lh = logp(pr(Zt, Xt)); Lh -= Lh.mean(1, keepdims=True)
                nsel = np.bincount(ys, minlength=C)
                for name, L in (('LGB', Lg), ('ENS75', 0.75 * Lg + 0.25 * Lh), ('HEAD', Lh)):
                    ph = L.argmax(1)
                    for c in range(C):
                        if nT[c] >= MIN_N:
                            rows.append(dict(tgt=tgt, split=split, feat=f, sel=cm, learner=name, cls=CLS[c],
                                             rec=100 * (ph[yT == c] == c).mean(), prec_n=int((ph == c).sum()), n=int(nT[c]),
                                             n_sel=int(nsel[c]), share=nT[c] / len(yT)))
R = pd.DataFrame(rows)
p = R.pivot_table(index=['tgt', 'split', 'feat', 'sel', 'cls'], columns='learner', values='rec').reset_index()
p = p.merge(R[R.learner == 'LGB'][['tgt', 'split', 'feat', 'sel', 'cls', 'share', 'n_sel']], on=['tgt', 'split', 'feat', 'sel', 'cls'])
p['d_ens'] = p.ENS75 - p.LGB; p['d_head'] = p.HEAD - p.LGB
pd.set_option('display.width', 250)
g = p.groupby(['sel', 'cls']).agg(n_ctry_units=('d_ens', 'size'), share=('share', 'mean'), n_sel=('n_sel', 'mean'),
                                   rec_lgb=('LGB', 'mean'), d_ens=('d_ens', 'mean'), d_head=('d_head', 'mean'))
print(g.round(2).to_string())
# 收益与类别在测试集中的占比、LGB 基线召回的关系
q = p[p.sel == 'AE_predstrat_srs']
print('corr(d_ens, rec_lgb) =', round(q[['d_ens', 'LGB']].corr().iloc[0, 1], 2), ' corr(d_ens, share) =', round(q[['d_ens', 'share']].corr().iloc[0, 1], 2))
q = q.assign(bin=pd.cut(q.LGB, [-1, 20, 50, 80, 101]))
print(q.groupby('bin', observed=True)[['d_ens', 'd_head']].agg(['mean', 'size']).round(2))
