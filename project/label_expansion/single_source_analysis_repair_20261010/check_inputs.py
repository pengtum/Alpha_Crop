"""独立核验源池、空间隔离和采样元数据，只写修复分析的验证记录。"""
from pathlib import Path
import sys,json
PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT/'label_expansion/single_source_v1'))
import run as R
import numpy as np,pandas as pd
f=R.prepare();data=R.CR.G.GData();checked=0
for config in f['configs']:
    path=R.ROOT/'configs'/f'{config["id"]}.npz'
    assert R.sha(path)==config['config_sha']
    fr=np.load(path)['foreign'];sp,sel=R.CR.load_unit(R.OLD,config['unit'])
    assert len(fr)==len(np.unique(fr))==5000
    assert R.CR.G.sel_hash(fr)==config['foreign_hash']
    assert np.array_equal(np.bincount(data.y[fr],minlength=10),config['class_counts'])
    assert not np.isin(fr,sp['test']).any()
    if config['source']=='MULTI5K':
        cs,counts=np.unique(data.ctry[fr],return_counts=True)
        assert set(cs)==set(R.COUNTRIES)-{config['target']} and (counts==500).all()
    else:assert (data.ctry[fr]==config['source']).all()
    assert R.CR.CS.content_hash(sp)==config['splits_hash']
    assert R.CR.CS.content_hash(sel)==config['selection_hash']
    runs=pd.read_csv(R.ROOT/'runs'/f'{config["id"]}.csv')
    assert (runs.source==config['source']).all() and (runs.target==config['target']).all()
    assert (runs.n_train==5000+runs.k).all() and (runs.threads==2).all()
    for arm,r in set(runs[['arm','r']].itertuples(index=False,name=None)):
        ti=R.target_indices(arm,r,sp,sel)
        assert not np.isin(ti,sp['test']).any()
    checked+=1
out=R.ROOT/'analysis_repaired_20261010/input_validation.json'
out.write_text(json.dumps({'status':'passed','configs':checked,'checks':['frozen input and code hashes','source pool hash and class counts','single source country or ten countries x 500','training and test index separation','split and selection metadata match','RSE manuscript and backup hashes'],'rse_files':R.check()},indent=2))
print(out.read_text())
