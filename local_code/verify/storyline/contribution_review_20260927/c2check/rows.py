import pandas as pd, sys
p="/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops/label_expansion/storyline/numbers_table/NUMBERS.csv"
d=pd.read_csv(p,encoding='utf-8-sig')
ids=sys.argv[1:]
for i in ids:
    sub=d[d.id.str.startswith(i)]
    for _,r in sub.iterrows():
        print("=== ",r['id'])
        for c in d.columns:
            if c=='id': continue
            v=r[c]
            if pd.isna(v): continue
            print(f"  {c}: {v}")
