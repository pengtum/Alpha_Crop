import numpy as np, glob, os
R='/Users/perng_mini/AlphaCrops_local/results/guidance_G/oracle'
rows=[]
for f in sorted(glob.glob(R+'/*.npz')):
    o=np.load(f); G=o['G'].astype(np.float64)
    sd=G.std(0); mu=G.mean(0)
    z=(G-mu)/(sd+1e-9)
    # columns with constant fill?
    rows.append((os.path.basename(f), G.shape, sd.min(), sd.max(), mu.min(), mu.max(), np.abs(z).max(0).min(), np.abs(z).max(), (np.abs(z)>5).mean()*100, sd.max()/sd.min()))
for r in rows:
    print(f"{r[0]:12s} {r[1]} sd[{r[2]:.2f},{r[3]:.2f}] ratio {r[9]:.2f} mean[{r[4]:.1f},{r[5]:.1f}] max|z| {r[7]:.1f} pct|z|>5 {r[8]:.3f}%")
