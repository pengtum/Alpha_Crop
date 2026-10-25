"""只读检查代码包；不导入实验代码、不读取数据、不启动训练。"""
from pathlib import Path
import argparse
import hashlib
import json
import re
import sys
import warnings

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = {'.csv', '.tsv', '.parquet', '.npy', '.npz', '.pkl', '.joblib',
             '.shp', '.shx', '.dbf', '.gpkg', '.geojson', '.tif', '.tiff',
             '.png', '.jpg', '.jpeg', '.pdf', '.docx', '.zip', '.pyc'}
CREDENTIALS = [r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
               r'\bAIza[0-9A-Za-z_-]{35}\b', r'\bgh[pousr]_[A-Za-z0-9]{30,}\b',
               r'\bsk-(?:proj-)?[A-Za-z0-9_-]{40,}\b']

def verify():
    errors, checked, syntax = [], 0, 0
    manifest = json.loads((ROOT/'source_manifest.json').read_text())
    for row in manifest:
        path = ROOT/row['path']
        if not path.is_file():
            errors.append(f"Missing source: {row['path']}"); continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != row['sha256']: errors.append(f"Changed source: {row['path']}")
        checked += 1
    for path in sorted(ROOT.rglob('*')):
        if not path.is_file() or '.git' in path.parts: continue
        rel = path.relative_to(ROOT).as_posix()
        if path.suffix.lower() in FORBIDDEN: errors.append(f'Excluded data/binary: {rel}')
        if path.name.startswith('.env') or re.search(r'(credentials|service.account)',path.name,re.I):
            errors.append(f'Credential-like filename: {rel}')
        try: text = path.read_text()
        except UnicodeError:
            errors.append(f'Non-text file: {rel}'); continue
        if any(re.search(p,text) for p in CREDENTIALS): errors.append(f'Potential embedded credential: {rel}')
        if path.suffix == '.py':
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', SyntaxWarning)
                    compile(text,str(path),'exec')
                syntax += 1
            except SyntaxError as e: errors.append(f'Python syntax: {rel}:{e.lineno}: {e.msg}')
        if path.suffix == '.ipynb':
            for i,cell in enumerate(json.loads(text)['cells']):
                if cell.get('outputs') or cell.get('execution_count') is not None or cell.get('attachments'):
                    errors.append(f'Notebook output/attachment: {rel}, cell {i}')
    return {'source_files_checked':checked,'python_files_compiled':syntax,
            'errors':errors,'passed':not errors,'training_executed':False,'data_included':False}

if __name__ == '__main__':
    if sys.version_info < (3,12):
        raise SystemExit('Python 3.12 or newer is required to compile the archived Python 3.12 syntax.')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,help='Optional report path; verification itself is read-only.')
    args=parser.parse_args()
    result=verify()
    output=json.dumps(result,ensure_ascii=False,indent=2)+'\n'
    print(output,end='')
    if args.report: args.report.write_text(output)
    sys.exit(0 if result['passed'] else 1)
