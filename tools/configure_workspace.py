"""将代码复制到新工作目录并配置路径，保留归档及原有冻结校验。"""
from pathlib import Path
import argparse
import hashlib
import json
import re
import shutil

ROOT=Path(__file__).resolve().parents[1]
OLD_PROJECT='/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops'
OLD_LOCAL='/Users/perng_mini/AlphaCrops_local'
OLD_NATURE=str(Path(OLD_PROJECT).parent/'nature-skills-main')

def sha(data):return hashlib.sha256(data).hexdigest()

def configure(dest,gee_project):
    dest=dest.expanduser().absolute()
    if dest.exists():raise SystemExit('Destination already exists; select a new directory.')
    if ROOT == dest or ROOT in dest.parents:
        raise SystemExit('Choose a workspace outside this release archive.')
    if not re.fullmatch(r'[a-z][a-z0-9-]{4,61}[a-z0-9]',gee_project):
        raise SystemExit('Supply your Google Cloud project ID (lowercase letters, digits and hyphens).')
    dest.mkdir(parents=True)
    mapping={OLD_PROJECT:str(dest/'Alpha_Crops'),OLD_LOCAL:str(dest/'AlphaCrops_local'),
             OLD_NATURE:str(dest/'nature-skills-main')}
    for src,name in [('project','Alpha_Crops'),('local_code','AlphaCrops_local'),
                     ('third_party/nature-skills-main','nature-skills-main')]:
        shutil.copytree(ROOT/src,dest/name)
    changes,residual=[],[]
    for p in sorted(dest.rglob('*')):
        if not p.is_file() or p.suffix not in {'.py','.sh','.ipynb','.R'}:continue
        old=p.read_bytes();t=old.decode('utf-8')
        for a,b in mapping.items(): t=t.replace(a,b)
        if p==dest/'Alpha_Crops/config/config.py':
            t,n=re.subn(r"^GEE_PROJECT\s*=\s*['\"][^'\"]+['\"]",f'GEE_PROJECT = {gee_project!r}',t,flags=re.M)
            if n!=1:raise RuntimeError('Expected one GEE_PROJECT assignment.')
        new=t.encode()
        if new!=old:
            p.write_bytes(new)
            changes.append({'path':p.relative_to(dest).as_posix(),'before_sha256':sha(old),'after_sha256':sha(new)})
        if re.search(r'/Users/|/home/|[A-Z]:[\\/]',t):residual.append(p.relative_to(dest).as_posix())
    report={'path_mapping':mapping,'gee_project':gee_project,'changed_files':changes,
            'files_with_other_machine_paths':residual,'archive_changed':False,
            'frozen_checks_bypassed':False,'new_run_registration_required':True,
            'experiments_started':False}
    (dest/'configuration_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    (dest/'WORKSPACE_README.txt').write_text(
        'Code working copy. No data were copied and no experiment was started.\n'
        'Set ALPHA_LOCAL to '+str(dest/'AlphaCrops_local')+'\n'
        'Use Alpha_Crops as the working directory for data preparation.\n'
        'Historical registration and hash checks were not rewritten. Path changes invalidate some frozen code hashes; '
        'new data may also have a different source stamp. Review and register a new run before formal training.\n'
        'configuration_report.json lists path rewrites and other machine-specific references that may need review.\n')
    return {'workspace':str(dest),'changed_files':len(changes),'other_path_files':len(residual),
            'new_run_registration_required':True,'experiments_started':False}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dest',required=True,type=Path)
    parser.add_argument('--gee-project',required=True)
    args=parser.parse_args()
    print(json.dumps(configure(args.dest,args.gee_project),ensure_ascii=False,indent=2))
