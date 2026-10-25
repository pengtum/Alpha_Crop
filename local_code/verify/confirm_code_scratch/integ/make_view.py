"""把冒烟输出（旧 G s0/s1，单元 {国家}_s{划分}）以符号链接改名成正式接口的单元名 {国家}_F{划分+1}，
供 analyze_confirm.py（正式模式）与 indep_confirm.py --root 走正式读取路径并互相核对。只建链接，不改冒烟文件。"""
import os
import shutil
import sys

src, dst = sys.argv[1], sys.argv[2]
shutil.rmtree(dst, ignore_errors=True)
for sub in ('splits', 'selection', 'runs', 'probs'):
    os.makedirs(os.path.join(dst, sub))
n = 0
for c in ('AT', 'BE', 'DE', 'EE', 'LT', 'LV', 'PT', 'RO', 'SE', 'SI', 'SK'):
    for s in (0, 1):
        a, b = f'{c}_s{s}', f'{c}_F{s + 1}'
        for sub, ext in (('splits', '.npz'), ('selection', '.npz'), ('runs', '.csv'), ('probs', '')):
            os.symlink(os.path.join(src, sub, a + ext), os.path.join(dst, sub, b + ext))
        n += 1
print(f'{n} 个单元：{src} → {dst}')
