"""公平性对照：MLP 的 3 种子概率平均 vs LightGBM 的 3 种子概率平均（s1r0，k=200 SRS）。
G 在 s1r0 的种子为 1000（缓存概率）；这里另训 1001、1002（参数与 G 相同）。保存各单种子与 3 种子平均的 τ 曲线。
用法：python lgb_seeds.py AT,BE,...  输出 lgb_seeds_rows_<tag>.csv 与 lgb_seeds_curves/。"""
from vlib import *
import lightgbm as lgb
D = G.GData()
split, rep = 1, 0
countries = sys.argv[1].split(','); tag = sys.argv[2]
path = os.path.join(OUT, f'lgb_seeds_rows_{tag}.csv'); cdir = os.path.join(OUT, 'lgb_seeds_curves'); os.makedirs(cdir, exist_ok=True)
rows = pd.read_csv(path).to_dict('records') if os.path.exists(path) else []
done = {(r['target'], r['feat']) for r in rows}
for tgt in countries:
    S, fr, test, yT, big, cw = unit(D, tgt, split)
    R = pd.read_csv(f'{GOUT}/rows_t1_{tgt}_s{split}_r{rep}.csv')
    sel = sel_of(tgt, split, rep, 'none', 'srs', 200)
    for feat in ('ae', 's2'):
        if (tgt, feat) in done: continue
        t0 = time.time()
        h = R[(R.method == 'srs') & (R.k == 200) & (R.clf == feat)].sel_hash.iloc[0]; assert h == G.sel_hash(sel)
        Ps = {1000: np.load(f'{GOUT}/probs/{tgt}_s{split}_r{rep}_{feat}_k200_{h}.npz')['i300'].astype(np.float64)}
        Xtr, ytr, wtr, ist = build(D, feat, fr, sel, cw)
        Xte = np.asarray(D.X[feat][test], np.float32)
        for sd in (1001, 1002):
            m = lgb.train(G.lgb_params(THREADS, sd), lgb.Dataset(Xtr, ytr, weight=wtr), num_boost_round=300)
            Ps[sd] = m.predict(Xte)
        Ps['ens'] = (Ps[1000] + Ps[1001] + Ps[1002]) / 3
        C = {str(k): curve(P, yT, big) for k, P in Ps.items()}
        np.savez_compressed(os.path.join(cdir, f'{tgt}_{feat}.npz'), **C)
        n0 = int(np.flatnonzero(TAUS == 0)[0])
        for k, Cm in C.items():
            o5, b5 = at_tau(Cm, 0.5)
            rows.append(dict(target=tgt, feat=feat, seed=k, ba0=Cm[n0, 2], oa0=Cm[n0, 1], ba05=b5, oa05=o5, ba_max=Cm[:, 2].max()))
        pd.DataFrame(rows).to_csv(path + '.tmp', index=False); os.replace(path + '.tmp', path)
        print(tgt, feat, f'{time.time()-t0:.0f}s', {r['seed']: (round(r['ba0'], 2), round(r['oa0'], 2)) for r in rows if r['target'] == tgt and r['feat'] == feat}, flush=True)
print('SEEDS DONE', flush=True)
