import sys, numpy as np
sys.path.insert(0, "/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops/label_expansion")
from common import load_dataset
df = load_dataset()
classes = sorted(df['label'].unique())
y = df['label'].map({c:i for i,c in enumerate(classes)}).to_numpy()
np.savez('meta.npz', y=y, ctry=df['country'].to_numpy().astype(str), lon=df['longitude'].to_numpy(), lat=df['latitude'].to_numpy(), classes=np.array(classes))
print(len(df), classes)
