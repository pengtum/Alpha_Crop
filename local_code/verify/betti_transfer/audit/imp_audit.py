# 用审计钩子记录 import guidance_G / pilot_betti 时打开的所有文件（检查模块级副作用：是否读数据/标签、是否写文件）
import os, sys
for v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[v]='2'
sys.dont_write_bytecode=True
LE='/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops/label_expansion'
sys.path.insert(0, LE)
opened=[]
def hook(ev, args):
    if ev=='open':
        p=str(args[0]); mode=args[1] if len(args)>1 else None
        if p.endswith('.py') or '/site-packages/' in p or '/lib/python' in p or p.endswith(('.so','.pyc','.dylib')) or '/.venv/' in p: return
        opened.append((p,mode))
    elif ev in ('os.mkdir','os.remove','os.rename','shutil.rmtree'):
        opened.append((ev,str(args[0])))
import numpy
sys.addaudithook(hook)
import guidance_G
print('guidance_G 导入时打开的非代码文件:', opened); opened.clear()
import pilot_betti
print('pilot_betti 导入时打开的非代码文件:', opened)
