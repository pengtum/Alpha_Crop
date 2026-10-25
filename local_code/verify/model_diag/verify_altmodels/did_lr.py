"""复核 altmodels 结论 6：E1×k=200 的 AE−S2 在 LR 下是否系统性高于 LGB（差中差）。在 s0r0 以外的单元独立重训 LR。
用法：python did_lr.py 1,0 0,1 1,1   （split,rep）"""
from vlib import *
import warnings; warnings.filterwarnings('ignore')
D = G.GData()
units = [tuple(map(int, u.split(','))) for u in sys.argv[1:]]
path = os.path.join(OUT, 'did_lr_rows.csv')
rows = pd.read_csv(path).to_dict('records') if os.path.exists(path) else []
done = {(r['target'], r['split'], r['rep'], r['cond'], r['method'], r['clf']) for r in rows}
for split, rep in units:
    for tgt in D.countries:
        S, fr, test, yT, big, cw = unit(D, tgt, split)
        R = pd.read_csv(f'{GOUT}/rows_t1_{tgt}_s{split}_r{rep}.csv')
        Xte = {f: np.asarray(D.X[f][test], np.float32) for f in ('ae', 's2')}
        t0 = time.time()
        for method in G.E1:
            for cond in ('AE', 'S2'):
                sel = sel_of(tgt, split, rep, cond, method, 200)
                h = G.sel_hash(sel)
                for feat in ('ae', 's2'):
                    if (tgt, split, rep, cond, method, feat) in done: continue
                    ref = R[(R.cond == cond) & (R.method == method) & (R.k == 200) & (R.clf == feat)]
                    assert len(ref) == 1 and ref.sel_hash.iloc[0] == h, (tgt, split, rep, cond, method, feat)
                    Xtr, ytr, wtr, ist = build(D, feat, fr, sel, cw)
                    A, B = std(Xtr, Xte[feat])
                    oa, ba = score(yT, lr_proba(A, ytr, wtr, B).argmax(1), big)
                    rows.append(dict(target=tgt, split=split, rep=rep, cond=cond, method=method, clf=feat,
                                     ba_lr=ba, oa_lr=oa, ba_lgb=float(ref.ba_ge30.iloc[0]), oa_lgb=float(ref.oa.iloc[0])))
        pd.DataFrame(rows).to_csv(path + '.tmp', index=False); os.replace(path + '.tmp', path)
        print(f's{split}r{rep} {tgt} done {time.time()-t0:.0f}s', flush=True)
print('DID DONE', flush=True)
