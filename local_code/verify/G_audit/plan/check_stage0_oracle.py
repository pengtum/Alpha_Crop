import glob, os, numpy as np
OUT = '/Users/perng_mini/AlphaCrops_local/results/guidance_G'
for p in sorted(glob.glob(f'{OUT}/oracle/*.npz'))[:4]:
    z = np.load(p); G = z['G']
    zs = (G - G.mean(0)) / (G.std(0) + 1e-9)
    print(os.path.basename(p), 'G shape', G.shape, 'col mean range', G.mean(0).min().round(1), G.mean(0).max().round(1),
          'col sd range', G.std(0).min().round(2), G.std(0).max().round(2), '|z|max', np.abs(zs).max().round(1),
          'frac |z|>5', (np.abs(zs) > 5).mean().round(4), 'temp', float(z['temp']))
print()
for p in sorted(glob.glob(f'{OUT}/stage0/*.npz')):
    z = np.load(p)
    g = z['groups']; u, c = np.unique(g, return_counts=True)
    ef = z['efar'].astype(bool)
    print(os.path.basename(p), 'groups', dict(zip(u.tolist(), c.tolist())), 'efar frac', round(ef.mean(), 3), 'n_test', len(z['test']),
          'temp ae/s2', round(float(z['temp_ae']), 3), round(float(z['temp_s2']), 3), 'G_ae', z['G_ae'].shape, 'G_s2', z['G_s2'].shape,
          'keys P0', sorted(k for k in z.files if k.startswith('P0test'))[:6] if p.endswith('AT_s0.npz') else '')
