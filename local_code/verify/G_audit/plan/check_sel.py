"""重现若干选样，核对 ctx 构造与调用参数；并比较 ORACLE 几何按方案做 z 截断后的选样差异。"""
import os, sys, time, numpy as np
sys.path.insert(0, '.')
import guidance_G as G
OUT = G.OUT
D = G.GData()
def load(tgt, s):
    return dict(np.load(f'{OUT}/stage0/{tgt}_s{s}.npz'))
cases = [('AT', 0, 0, 'AE', 'lpm2_nogeo', 50), ('AT', 0, 0, 'S2', 'kcenter', 200), ('AT', 0, 0, 'S2XY', 'kcenter', 50),
         ('AT', 0, 0, 'XY', 'lpm2_nogeo', 200), ('AT', 0, 0, 'AE', 'badge_c', 50), ('AT', 0, 0, 'S2', 'predstrat_srs', 200),
         ('AT', 0, 0, 'AE', 'gist_margin', 50), ('AT', 0, 0, 'S2', 'margin', 50), ('AT', 0, 0, 'ORACLE', 'lpm2_nogeo', 50),
         ('AT', 0, 0, 'ORACLE', 'typiclust', 50), ('SE', 1, 1, 'AE', 'kherding', 50), ('SE', 1, 1, 'S2', 'typiclust', 200)]
cache = {}
for tgt, s, r, cond, m, k in cases:
    if (tgt, s) not in cache:
        cache[(tgt, s)] = load(tgt, s)
    S = cache[(tgt, s)]
    pp = G.pool_positions(S, s, r)
    orc = dict(np.load(f'{OUT}/oracle/{tgt}_s{s}.npz')) if cond == 'ORACLE' else None
    t0 = time.time()
    sel_pos, meta = G.select_one(D, S, pp, cond, m, k, 1000 * s + r, orc)
    saved = np.load(f'{OUT}/sel/{tgt}_s{s}_r{r}_{cond}_{m}_k{k}.npz')['sel_pos']
    extra = ''
    if cond == 'ORACLE' and m in ('lpm2_nogeo', 'typiclust'):
        Gz = orc['G'].astype(np.float64)
        Gz = np.clip((Gz - Gz.mean(0)) / (Gz.std(0) + 1e-9), -5, 5)
        o2 = dict(orc, G=Gz.astype(np.float32))
        sp2, _ = G.select_one(D, S, pp, cond, m, k, 1000 * s + r, o2)
        jac = len(np.intersect1d(sp2, saved)) / len(np.union1d(sp2, saved))
        y = D.y[S['allT'][pp]]
        extra = f' | 按方案 z 截断后 Jaccard={jac:.3f}; 类别数 原 {len(np.unique(y[saved]))} 方案 {len(np.unique(y[sp2]))}'
    print(tgt, s, r, cond, m, k, '一致' if np.array_equal(np.sort(sel_pos), np.sort(saved)) else '不一致',
          f'{time.time()-t0:.0f}s' + extra, flush=True)
# ctx 检查
S = cache[('AT', 0)]; pp = G.pool_positions(S, 0, 0)
for cond in ('AE', 'S2', 'XY', 'S2XY'):
    ctx = G.build_ctx_guided(D, S, pp, cond, d=16)
    print(cond, 'T_ae', None if ctx.T_ae is None else ctx.T_ae.shape, 'P_T', None if ctx.P_T is None else ctx.P_T.shape,
          'others None:', all(getattr(ctx, f) is None for f in ('T_s2', 'T_xy', 'T_nuts3', 'F_ae', 'F_s2', 'F_xy', 'F_country', 'P_T_loco', 'dom', 'P_F_oof', 'oracle')))
    if cond == 'S2XY':
        X = G.X_of(ctx, 'ae', 18)
        v = X.var(0)
        print('  S2XY X_of(d+2) 列数', X.shape[1], '坐标方差份额', round(v[-2:].sum() / v.sum(), 4), '理论 8/(d+8)=', round(8 / 24, 4))
    if cond == 'AE':
        assert np.array_equal(ctx.T_ae, S['G_ae'][pp]); assert np.allclose(ctx.P_T, S['P_T_ae'][pp])
    if cond == 'S2':
        assert np.array_equal(ctx.T_ae, S['G_s2'][pp]); assert np.allclose(ctx.P_T, S['P_T_s2'][pp])
