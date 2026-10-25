"""把已核对图件与新增描述性数据复制到稿件包，不修改原有补充数据。"""
from pathlib import Path
import shutil
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PAPER = ROOT / 'Manuscript/Paper_Alpha_Earth/0002_after_transfer_rse'

def main():
    for name in ('fig_dataset_imbalance','fig_zero_label_accuracy','fig_zero_label_confusion','fig_pairwise_transfer'):
        shutil.copy2(HERE/'out'/f'{name}.pdf',PAPER/'figures'/f'{name}.pdf')
    mapping = {'S12_class_counts.csv':'class_counts.csv', 'S13_class_shares.csv':'class_shares.csv',
        'S14_zero_label_accuracy.csv':'zero_label_countries.csv',
        'S15_zero_label_confusion_by_country.csv':'confusion_countries.csv',
        'S16_zero_label_confusion_country_equal.csv':'confusion_country_equal.csv',
        'S17_pairwise_transfer.csv':'pairwise_transfer.csv', 'S18_spatial_label_counts.csv':'spatial_density.csv',
        'S19_zero_label_unit_accuracy.csv':'zero_label_units.csv', 'S20_country_label_support.csv':'country_support.csv'}
    for name,src in mapping.items():shutil.copy2(HERE/'source_data'/src,PAPER/'supplementary'/name)
    shutil.copy2(HERE/'out/confusion_by_country.pdf',PAPER/'supplementary/zero_label_confusion_by_country.pdf')
    # 逐国精度表，便于审稿人读取精确数值。
    d=pd.read_csv(HERE/'source_data/zero_label_countries.csv')
    d=d.pivot(index='country',columns='feat',values=['OA','BA'])
    text=[r'\begin{tabular}{lrrrr}',r'\toprule',r'Country & AE OA (\%) & S2 OA (\%) & AE BA (\%) & S2 BA (\%) \\',r'\midrule']
    for cc,r in d.iterrows():
        text.append(cc+' & '+' & '.join(f'{r[k]:.2f}' for k in [('OA','ae'),('OA','s2'),('BA','ae'),('BA','s2')])+r' \\')
    text.extend([r'\bottomrule',r'\end{tabular}'])
    (PAPER/'tables/tab_zero_label_accuracy.tex').write_text('\n'.join(text)+'\n')
    print('Packaged four figures, nine data tables and the eleven-country confusion atlas.')

if __name__=='__main__':main()
