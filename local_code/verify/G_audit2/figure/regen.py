import plot_G, numpy as np
plot_G.FIG_DIR = '/Users/perng_mini/AlphaCrops_local/verify/G_audit2/figure/regen'
plot_G.main()
from PIL import Image
a = np.asarray(Image.open('figures/fig_G_results.png').convert('RGB')).astype(int)
b = np.asarray(Image.open(plot_G.FIG_DIR + '/fig_G_results.png').convert('RGB')).astype(int)
print(a.shape, b.shape)
if a.shape == b.shape:
    diff = np.abs(a-b).sum(2); print('n diff px', (diff>0).sum(), 'max', diff.max())
