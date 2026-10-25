"""第 2 部分：G 核心对比（E1 几何通路 AE−S2）对分类器是否敏感。
11 国 × 划分 0 × 重复 0 × k=200 × E1 4 个方法 × 指导 {AE, S2} × 分类器特征 {ae, s2}；用 G 的 sel 文件重训替代模型。
用法：python run_part2.py lr,mlp2"""
import sys; sys.path.insert(0, '/Users/perng_mini/AlphaCrops_local/verify/model_diag/altmodels')
from altlib import *
from altlib import _std
import pandas as pd
import warnings; warnings.filterwarnings('ignore')

models = sys.argv[1].split(',')
D = G.GData()
path = os.path.join(OUT, 'part2_rows.csv')
rows = pd.read_csv(path).to_dict('records') if os.path.exists(path) else []
done = {(r['target'], r['cond'], r['method'], r['clf'], r['model']) for r in rows}
lgb_e1 = pd.read_csv(os.path.join(OUT, 'lgb_e1_k200_s0r0.csv'))


def fit(model, X_tr, y_tr, w_tr, X_te):
    if model == 'lr':
        return m_lr(X_tr, y_tr, w_tr, X_te)
    if model == 'mlp2':
        return m_mlp(X_tr, y_tr, w_tr, X_te, hidden=(256, 128))[0]
    if model == 'lgb':
        return m_lgb(X_tr, y_tr, w_tr, X_te, seed=0)
    raise ValueError(model)


for model in models:
    for tgt in D.countries:
        S, fr, test, yT, big, cw = load_unit(D, tgt)
        for method in G.E1:
            for cond in ('AE', 'S2'):
                sel = load_sel(tgt, cond, method, 200)
                h = G.sel_hash(sel)
                ref = lgb_e1[(lgb_e1.target == tgt) & (lgb_e1.cond == cond) & (lgb_e1.method == method)]
                assert set(ref.sel_hash) == {h}, (tgt, cond, method)
                for feat in ('ae', 's2'):
                    if (tgt, cond, method, feat, model) in done:
                        continue
                    X_tr, y_tr, w_tr, is_t = train_set(D, feat, fr, sel, cw)
                    X_te = np.asarray(D.X[feat][test], dtype=np.float32)
                    ph, t = timed(fit, model, X_tr, y_tr, w_tr, X_te)
                    oa, ba = scores(yT, ph, big)
                    rows.append(dict(target=tgt, cond=cond, method=method, clf=feat, model=model, ba_ge30=ba, oa=oa,
                                     t_fit=round(t, 2), sel_hash=h))
                    pd.DataFrame(rows).to_csv(path + '.tmp', index=False)
                    os.replace(path + '.tmp', path)
        print(model, tgt, 'done', flush=True)
print('PART2 DONE', flush=True)
