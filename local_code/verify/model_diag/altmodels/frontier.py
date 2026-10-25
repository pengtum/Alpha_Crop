"""工作点诊断：LR/MLP 的 ba 提升是更好的模型，还是只是换了 OA–ba 工作点？
对每个（国家, 特征, k）：LightGBM 测试概率（G 缓存，k=200 SRS 取 probs/，k=0 取阶段 0 P0test_*_i300）、LR/MLP2 测试概率（重训）。
先验调整：logit_c − τ·log π̂_c，π̂ = 该模型在目标测试集上的平均预测概率（无标签），τ ∈ [−3, 3]。
比较 1：把 LightGBM 调到与替代模型原生 OA 相同的工作点，比 ba；比较 2：把替代模型调到 LightGBM 原生 OA，比 ba。
另报各模型在 τ 网格上的最大 ba（有标签选 τ，只作上界参考）。"""
import sys; sys.path.insert(0, '/Users/perng_mini/AlphaCrops_local/verify/model_diag/altmodels')
from altlib import *
from altlib import _std
import pandas as pd
import warnings; warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
D = G.GData()
TAUS = np.round(np.arange(-3, 3.001, 0.05), 2)
path = os.path.join(OUT, 'frontier_rows.csv')
rows = pd.read_csv(path).to_dict('records') if os.path.exists(path) else []
done = {(r['target'], r['feat'], r['k'], r['model']) for r in rows}
curves = {}


def curve(P, yT, big):
    L = np.log(np.clip(P.astype(np.float64), 1e-12, None))
    pi = np.clip(P.mean(0), 1e-6, None)
    out = []
    for t in TAUS:
        oa, ba = scores(yT, (L - t * np.log(pi)).argmax(1), big)
        out.append((t, oa, ba))
    return np.array(out)


def at_oa(C, oa_target):
    j = np.argmin(np.abs(C[:, 1] - oa_target))
    return C[j]


for tgt in D.countries:
    S, fr, test, yT, big, cw = load_unit(D, tgt)
    R = pd.read_csv(f'{GOUT}/rows_t1_{tgt}_s0_r0.csv')
    sel200 = load_sel(tgt, 'none', 'srs', 200)
    for feat in ('ae', 's2'):
        X_te = np.asarray(D.X[feat][test], np.float32)
        for k in (0, 200):
            if k == 0:
                P_L = S[f'P0test_{feat}_i300'].astype(np.float32)
                sel = np.array([], np.int64)
            else:
                h = R[(R.method == 'srs') & (R.k == 200) & (R.clf == feat)].sel_hash.iloc[0]
                P_L = np.load(f'{GOUT}/probs/{tgt}_s0_r0_{feat}_k200_{h}.npz')['i300']
                sel = sel200
            C_L = curve(P_L, yT, big)
            X_tr, y_tr, w_tr, is_t = train_set(D, feat, fr, sel, cw)
            A, B_ = _std(X_tr, X_te)
            for name in ('lr', 'mlp2'):
                if (tgt, feat, k, name) in done:
                    continue
                if name == 'lr':
                    m = LogisticRegression(C=1.0, max_iter=1000).fit(A, y_tr, sample_weight=w_tr / w_tr.mean())
                else:
                    m = MLPClassifier(hidden_layer_sizes=(256, 128), alpha=1e-4, batch_size=256, learning_rate_init=1e-3,
                                      early_stopping=True, validation_fraction=0.1, n_iter_no_change=8, max_iter=60,
                                      random_state=0).fit(A, y_tr, sample_weight=w_tr / w_tr.mean())
                P = np.zeros((len(B_), 10)); P[:, m.classes_] = m.predict_proba(B_)
                C_M = curve(P, yT, big)
                n0 = int(np.flatnonzero(TAUS == 0)[0])
                oaL, baL = C_L[n0, 1:]; oaM, baM = C_M[n0, 1:]
                L_at = at_oa(C_L, oaM); M_at = at_oa(C_M, oaL)
                rows.append(dict(target=tgt, feat=feat, k=k, model=name, oa_lgb=oaL, ba_lgb=baL, oa_m=oaM, ba_m=baM,
                                 lgb_tau_at_mOA=L_at[0], lgb_oa_at_mOA=L_at[1], lgb_ba_at_mOA=L_at[2],
                                 m_tau_at_lgbOA=M_at[0], m_oa_at_lgbOA=M_at[1], m_ba_at_lgbOA=M_at[2],
                                 lgb_ba_max=C_L[:, 2].max(), m_ba_max=C_M[:, 2].max(),
                                 lgb_oa_max=C_L[:, 1].max(), m_oa_max=C_M[:, 1].max()))
                curves[(tgt, feat, k, name)] = C_M
                curves[(tgt, feat, k, 'lgb')] = C_L
                pd.DataFrame(rows).to_csv(path + '.tmp', index=False); os.replace(path + '.tmp', path)
                r = rows[-1]
                print(f"{tgt} {feat} k{k} {name}: native lgb ({oaL:.1f},{baL:.1f}) {name} ({oaM:.1f},{baM:.1f}) | "
                      f"lgb@{name}OA ({r['lgb_oa_at_mOA']:.1f},{r['lgb_ba_at_mOA']:.1f}) τ={r['lgb_tau_at_mOA']} | "
                      f"{name}@lgbOA ({r['m_oa_at_lgbOA']:.1f},{r['m_ba_at_lgbOA']:.1f}) τ={r['m_tau_at_lgbOA']}", flush=True)
np.savez_compressed(os.path.join(OUT, 'frontier_curves.npz'), **{'|'.join(map(str, k)): v for k, v in curves.items()})
print('FRONTIER DONE', flush=True)
