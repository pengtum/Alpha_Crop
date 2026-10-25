"""只读进度与RSE校验；可在实验运行期间调用。"""
from pathlib import Path
import json,sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run as R
from protect_baseline import check
import pandas as pd
if __name__=='__main__':
    files=list((R.ROOT/'runs').glob('*.csv'))
    counts={};failed=0
    if files:
        d=pd.concat([pd.read_csv(p,keep_default_na=False) for p in files]);ok=d[d.status=='ok']
        counts=ok.groupby('learner').size().to_dict();failed=int((d.status=='failed').sum())
    state=json.loads((R.ROOT/'status.json').read_text()) if (R.ROOT/'status.json').exists() else {'state':'not_started'}
    print(json.dumps(dict(status=state,completed=counts,total_completed=sum(counts.values()),total_planned=18392,failed=failed,rse_files_verified=check()),indent=2))
