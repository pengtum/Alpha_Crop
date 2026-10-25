import pandas as pd
p="/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops/label_expansion/storyline/numbers_table/NUMBERS.csv"
d=pd.read_csv(p,encoding='utf-8-sig')
print(d.shape)
print(d.group.value_counts())
pd.set_option('display.width',250); pd.set_option('display.max_colwidth',90); pd.set_option('display.max_rows',400)
print(d[['id','value','ci95_lo','ci95_hi','ci90_lo','ci90_hi','n_pos','p','grade']].to_string())
