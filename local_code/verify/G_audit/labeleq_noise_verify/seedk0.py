import sys, os, time, numpy as np, pandas as pd
sys.path.insert(0, "/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops/label_expansion")
os.chdir("/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops/label_expansion")
import guidance_G as G
from stageA import class_weights_from
D = G.GData()
tgt, split, clf = sys.argv[1], int(sys.argv[2]), sys.argv[3]
seeds = [int(s) for s in sys.argv[4].split(',')]
threads = int(sys.argv[5]) if len(sys.argv) > 5 else 3
S = dict(np.load(f'{G.OUT}/stage0/{tgt}_s{split}.npz'))
fr, test = S['foreign'], S['test']
yF, yT = D.y[fr], D.y[test]
big = np.flatnonzero(np.bincount(yT, minlength=10) >= G.MIN_N)
cw = class_weights_from(yF)
X = D.X[clf]
stored = S[f'P0test_{clf}_i300']
ba_stored = G._scores(yT, stored.astype(np.float32).argmax(1), big)[1]
out = f'seed_{tgt}_s{split}_{clf}.csv'
rows = pd.read_csv(out).to_dict('records') if os.path.exists(out) else []
for sd in seeds:
    t0 = time.time()
    P = G.train_predict(X[fr], yF, cw[yF], X[test], threads, sd)[300]
    ba32 = G._scores(yT, P.argmax(1), big)[1]
    P16 = P.astype(np.float16).astype(np.float32)
    ba16 = G._scores(yT, P16.argmax(1), big)[1]
    same = float(np.abs(P16 - stored.astype(np.float32)).max()) if sd == 0 else np.nan
    rows.append(dict(tgt=tgt, split=split, clf=clf, seed=sd, ba32=ba32, ba16=ba16, ba_stored=ba_stored,
                     maxabs_vs_stored=same, n_argmax_flip16=int((P16.argmax(1) != P.argmax(1)).sum()), sec=time.time() - t0))
    pd.DataFrame(rows).to_csv(out, index=False)
    print(rows[-1], flush=True)
