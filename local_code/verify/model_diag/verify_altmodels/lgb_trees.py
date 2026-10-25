"""复核"LightGBM 不欠拟合、树多了目标精度下降"：在 s1r0（altmodels 只看了 s0r0 的 SE/PT）上训练到 500 棵，
记录 25–500 棵各检查点的原生 ba/OA、τ=0.5 的 ba/OA、τ 网格最大 ba，以及各检查点在"i300 原生 OA"处的 Pareto 包络 ba。
参数与 G 完全相同（guidance_G.lgb_params，种子 1000*split+rep），i300 原生 ba 必须与 G 的 CSV 一致。
用法：python lgb_trees.py SE:1,PT:1,AT:1,RO:1"""
from vlib import *
import lightgbm as lgb
D = G.GData()
CK = (25, 50, 100, 150, 200, 300, 400, 500)
path = os.path.join(OUT, 'lgb_trees_rows.csv')
rows = pd.read_csv(path).to_dict('records') if os.path.exists(path) else []
done = {(r['target'], r['split'], r['feat']) for r in rows}
for item in sys.argv[1].split(','):
    tgt, split = item.split(':'); split = int(split); rep = 0
    S, fr, test, yT, big, cw = unit(D, tgt, split)
    R = pd.read_csv(f'{GOUT}/rows_t1_{tgt}_s{split}_r{rep}.csv')
    sel = sel_of(tgt, split, rep, 'none', 'srs', 200)
    for feat in ('ae', 's2'):
        if (tgt, split, feat) in done: continue
        ref = R[(R.method == 'srs') & (R.k == 200) & (R.clf == feat)].iloc[0]
        Xtr, ytr, wtr, ist = build(D, feat, fr, sel, cw)
        Xte = np.asarray(D.X[feat][test], np.float32)
        t0 = time.time()
        m = lgb.train(G.lgb_params(THREADS, 1000 * split + rep), lgb.Dataset(Xtr, ytr, weight=wtr), num_boost_round=max(CK))
        tfit = time.time() - t0
        Cs = {it: curve(m.predict(Xte, num_iteration=it), yT, big) for it in CK}
        n0 = int(np.flatnonzero(TAUS == 0)[0])
        assert abs(Cs[300][n0, 2] - ref.ba_ge30) < 1e-6, (tgt, feat, Cs[300][n0, 2], ref.ba_ge30)
        oa_ref = Cs[300][n0, 1]
        for it in CK:
            C = Cs[it]; o5, b5 = at_tau(C, 0.5)
            rows.append(dict(target=tgt, split=split, feat=feat, it=it, ba0=C[n0, 2], oa0=C[n0, 1], ba05=b5, oa05=o5,
                             ba_max=C[:, 2].max(), env_at_oa300=env(C, oa_ref), t_fit=tfit))
        pd.DataFrame(rows).to_csv(path + '.tmp', index=False); os.replace(path + '.tmp', path)
        print(tgt, split, feat, f'{tfit:.0f}s', ' '.join(f"{r['it']}:{r['ba0']:.1f}/{r['oa0']:.1f}|{r['ba05']:.1f}|env{r['env_at_oa300']:.1f}"
                                                   for r in rows if r['target'] == tgt and r['split'] == split and r['feat'] == feat), flush=True)
print('TREES DONE', flush=True)
