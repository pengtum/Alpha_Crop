"""复核 altmodels 结论 2、3、7：MLP 是否欠训练/欠调参；LR/MLP 相对 LGB 的 ba 提升是否只是工作点。
独立单元 s1r0（altmodels 只用了 s0r0），11 国 × {ae,s2} × k∈{0,200}（k=200 = G 的 SRS 选样）。
模型：lr（C=1）；mlpES（altmodels 配置：early_stopping 随机 10% 验证，种子 0）；
mlpF：不早停、用 partial_fit 逐轮训练（全部目标样本都参与训练），α=1e-4，种子 0/1/2 及 3 种子概率平均（ens），记录第 5/10/20/40 轮；
mlpFa3：α=1e-3，种子 0（只 k=200）。
预先指定的主 MLP：mlpF_ens 第 20 轮。其余轮数与 α 为敏感性（按测试集挑选即为乐观上界）。
LGB：G 缓存概率（k=200：probs/ 的 i300；k=0：阶段 0 P0test_*_i300）。
每个模型存 τ 曲线（logit − τ·log 测试集平均预测概率，τ∈[−3,3] 步长 0.1）。"""
from vlib import *
import warnings; warnings.filterwarnings('ignore')
from sklearn.neural_network import MLPClassifier
D = G.GData()
split, rep = int(sys.argv[1]), int(sys.argv[2])
countries = sys.argv[3].split(',') if len(sys.argv) > 3 else D.countries
TAU = np.round(np.arange(-3, 3.001, 0.1), 2)
def curve2(P, yT, big):
    L = np.log(np.clip(P.astype(np.float64), 1e-12, None)); pi = np.clip(P.mean(0), 1e-6, None)
    return np.array([(t, *score(yT, (L - t * np.log(pi)).argmax(1), big)) for t in TAU])
tag = f's{split}r{rep}'
path = os.path.join(OUT, f'mlp_fair_rows_{tag}.csv'); cpath = os.path.join(OUT, f'mlp_fair_curves_{tag}')
os.makedirs(cpath, exist_ok=True)
rows = pd.read_csv(path).to_dict('records') if os.path.exists(path) else []
done = {(r['target'], r['feat'], r['k']) for r in rows}
CK = (5, 10, 20, 40)

def proba(m, B):
    P = np.zeros((len(B), 10)); P[:, m.classes_] = m.predict_proba(B); return P

def mlp_full(A, ytr, w, B, seed, alpha):
    m = MLPClassifier(hidden_layer_sizes=(256, 128), alpha=alpha, batch_size=256, learning_rate_init=1e-3,
                      random_state=seed, shuffle=True)
    out, tr = {}, {}
    rng = np.random.default_rng(1000 + seed)  # partial_fit 每次调用都用同一整数种子重建 RNG，洗牌顺序会逐轮相同；这里先自行随机置换
    for ep in range(1, max(CK) + 1):
        pm = rng.permutation(len(ytr))
        m.partial_fit(A[pm], ytr[pm], sample_weight=w[pm], classes=np.arange(10))
        if ep in CK:
            out[ep] = proba(m, B)
            tr[ep] = m.predict(A)
    return out, tr

for tgt in countries:
    S, fr, test, yT, big, cw = unit(D, tgt, split)
    R = pd.read_csv(f'{GOUT}/rows_t1_{tgt}_s{split}_r{rep}.csv')
    sel200 = sel_of(tgt, split, rep, 'none', 'srs', 200)
    for feat in ('ae', 's2'):
        Xte = np.asarray(D.X[feat][test], np.float32)
        for k in (200, 0):
            if (tgt, feat, k) in done: continue
            t0 = time.time()
            if k == 0:
                P_L = S[f'P0test_{feat}_i300'].astype(np.float32); sel = np.array([], np.int64)
                refba = R[(R.method == 'foreign_only') & (R.clf == feat)].ba_ge30.iloc[0]
            else:
                rr = R[(R.method == 'srs') & (R.k == 200) & (R.clf == feat)]
                h = rr.sel_hash.iloc[0]; assert h == G.sel_hash(sel200)
                P_L = np.load(f'{GOUT}/probs/{tgt}_s{split}_r{rep}_{feat}_k200_{h}.npz')['i300']; sel = sel200
                refba = rr.ba_ge30.iloc[0]
            Xtr, ytr, wtr, ist = build(D, feat, fr, sel, cw)
            A, B = std(Xtr, Xte)
            w = wtr / wtr.mean()
            Ps, trba = {'lgb': P_L}, {}
            Ps['lr'] = lr_proba(A, ytr, wtr, B)
            m = MLPClassifier(hidden_layer_sizes=(256, 128), alpha=1e-4, batch_size=256, learning_rate_init=1e-3,
                              early_stopping=True, validation_fraction=0.1, n_iter_no_change=8, max_iter=60,
                              random_state=0).fit(A, ytr, sample_weight=w)
            Ps['mlpES'] = proba(m, B); nES = m.n_iter_
            big_tr = np.flatnonzero(np.bincount(ytr[~ist], minlength=10) >= 30)
            trba['mlpES'] = score(ytr[~ist], m.predict(A[~ist]), big_tr)[1]
            seeds = (0, 1, 2) if k == 200 else (0,)
            ens = {ep: 0 for ep in CK}
            for sd in seeds:
                o, tr = mlp_full(A, ytr, w, B, sd, 1e-4)
                for ep in CK:
                    Ps[f'mlpF_s{sd}_e{ep}'] = o[ep]; ens[ep] = ens[ep] + o[ep] / len(seeds)
                    trba[f'mlpF_s{sd}_e{ep}'] = score(ytr[~ist], tr[ep][~ist], big_tr)[1]
                    if k == 200: trba[f'mlpF_s{sd}_e{ep}_tgtacc'] = 100 * float((tr[ep][ist] == ytr[ist]).mean())
            if k == 200:
                for ep in CK: Ps[f'mlpF_ens_e{ep}'] = ens[ep]
                o, tr = mlp_full(A, ytr, w, B, 0, 1e-3)
                for ep in CK:
                    Ps[f'mlpFa3_s0_e{ep}'] = o[ep]; trba[f'mlpFa3_s0_e{ep}'] = score(ytr[~ist], tr[ep][~ist], big_tr)[1]
            C = {name: curve2(P, yT, big) for name, P in Ps.items()}
            np.savez_compressed(os.path.join(cpath, f'{tgt}_{feat}_k{k}.npz'), **C)
            CL = C['lgb']; n0 = int(np.flatnonzero(TAU == 0)[0]); n5 = int(np.flatnonzero(TAU == 0.5)[0])
            assert abs(CL[n0, 2] - refba) < 1e-6, (tgt, feat, k, CL[n0, 2], refba)
            for name, Cm in C.items():
                rows.append(dict(target=tgt, feat=feat, k=k, model=name,
                                 ba0=Cm[n0, 2], oa0=Cm[n0, 1], ba05=Cm[n5, 2], oa05=Cm[n5, 1],
                                 env_at_lgbOA0=env(Cm, CL[n0, 1]), env_at_lgbOA05=env(Cm, CL[n5, 1]),
                                 lgbenv_at_mOA0=env(CL, Cm[n0, 1]), lgbenv_at_mOA05=env(CL, Cm[n5, 1]),
                                 ba_max=Cm[:, 2].max(), train_ba_foreign=trba.get(name, np.nan),
                                 tgt_train_acc=trba.get(name + '_tgtacc', np.nan), n_iter_es=nES if name == 'mlpES' else np.nan))
            pd.DataFrame(rows).to_csv(path + '.tmp', index=False); os.replace(path + '.tmp', path)
            g = {r['model']: r for r in rows if r['target'] == tgt and r['feat'] == feat and r['k'] == k}
            print(f"{tag} {tgt} {feat} k{k}: lgb {g['lgb']['ba0']:.1f}/{g['lgb']['oa0']:.1f} lr {g['lr']['ba0']:.1f}/{g['lr']['oa0']:.1f} "
                  f"mlpES {g['mlpES']['ba0']:.1f}/{g['mlpES']['oa0']:.1f} (it {nES}) mlpF_s0_e20 {g['mlpF_s0_e20']['ba0']:.1f}/{g['mlpF_s0_e20']['oa0']:.1f} "
                  f"{time.time()-t0:.0f}s", flush=True)
print('MLP DONE', flush=True)
