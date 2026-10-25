"""只读保护当前RSE；快照和校验结果只写到独立备份目录。"""
from pathlib import Path
import hashlib,json,shutil
PROJECT=Path(__file__).resolve().parents[2]
BASE=PROJECT/'Manuscript/Paper_Alpha_Earth/0002_after_transfer_rse'
BACK=PROJECT/'backups/rse_frozen_20261009'
def inventory(root):
    return {str(p.relative_to(root)):hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for p in sorted(root.rglob('*')) if p.is_file()}
def check():
    manifest=json.loads((BACK/'manifest.json').read_text())
    assert inventory(BASE)==manifest, 'RSE原目录与冻结基线不同；停止，不自动恢复或覆盖用户文件'
    assert inventory(BACK/'snapshot')==manifest, '备份校验失败'
    return len(manifest)
def create():
    BACK.mkdir(parents=True,exist_ok=True)
    if (BACK/'manifest.json').exists():return check()
    before=inventory(BASE)
    assert not (BACK/'snapshot').exists(), '已有不完整快照，请人工核查'
    shutil.copytree(BASE,BACK/'snapshot',copy_function=shutil.copy2)
    assert inventory(BASE)==before==inventory(BACK/'snapshot')
    (BACK/'manifest.json').write_text(json.dumps(before,indent=2))
    return check()
if __name__=='__main__':print('Frozen RSE files verified:',create())
