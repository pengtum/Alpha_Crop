"""独立单源运行器：只写新结果目录；复用冻结采样，不修改旧确认代码。"""
import os,sys
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='2'
sys.dont_write_bytecode=True
from pathlib import Path
HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[1]
ROOT=PROJECT/'results/single_source_20261009'
OLD=Path('/Users/perng_mini/AlphaCrops_local/results/confirm_v1')
sys.path.insert(0,str(HERE.parent/'confirm'))
import confirm_run as CR
import numpy as np
import pandas as pd
import argparse,hashlib,json,time,shutil,fcntl,traceback,multiprocessing,subprocess
from concurrent.futures import ProcessPoolExecutor,as_completed
from protect_baseline import check
COUNTRIES=CR.CS.COUNTRIES

def sha(path):
    with open(path,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def atomic_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,default=str));os.replace(tmp,path)
def code_hash():
    files=sorted(HERE.glob('*.py'))+[HERE/'PLAN.txt']
    return hashlib.sha256(''.join(str(p.name)+sha(p) for p in files).encode()).hexdigest()
def provenance():
    files=[Path(p) for p in CR.CS.FROZEN]+[Path(CR.__file__),Path(CR.CS.__file__),Path(CR.CS.PLAN)]
    files += list((OLD/'splits').glob('*.npz'))+list((OLD/'selection').glob('*.npz'))
    files += [OLD/'split_manifest.csv']
    files += list((OLD/'probs').glob('*/lgb_TGT_*.npz'))
    cache=Path(CR.G.GData.cache_dir())
    files += [cache/'meta.json']+[cache/f'{k}.npy' for k in CR.G.GData.KEYS]
    return {str(p):sha(p) for p in files}
def prepare():
    check();CR.CS.check_env();ROOT.mkdir(exist_ok=True)
    lock=ROOT/'frozen.json'
    if lock.exists():
        z=json.loads(lock.read_text());assert z['code_hash']==code_hash(),'代码与本扩展冻结版本不同'
        assert z['inputs']==provenance(),'输入文件与扩展基线不同'
        return z
    D=CR.G.GData();configs=[];manifest=[]
    for tgt in COUNTRIES:
        for half in (1,2):
            unit=f'{tgt}_F{half}';sp,sel=CR.load_unit(OLD,unit)
            assert bool(sp['feasible']);assert len(np.unique(sp['foreign']))==50000
            assert not np.isin(sp['foreign'],sp['test']).any()
            assert not np.isin(sp['pool'],sp['test']).any()
            foreign={c:sp['foreign'][D.ctry[sp['foreign']]==c] for c in COUNTRIES if c!=tgt}
            assert all(len(a)==5000 for a in foreign.values())
            rng=np.random.default_rng(CR.CS.seed_of(f'single_source_v1_multi5k_{unit}'))
            mixed=np.concatenate([rng.choice(foreign[c],500,replace=False) for c in sorted(foreign)])
            for src,fr in list(foreign.items())+[('MULTI5K',mixed)]:
                cid=f'{src}__{unit}';out=ROOT/'configs'/f'{cid}.npz';out.parent.mkdir(exist_ok=True)
                np.savez(out,foreign=fr)
                counts=np.bincount(D.y[fr],minlength=10)
                row=dict(id=cid,source=src,target=tgt,unit=unit,foreign_hash=CR.G.sel_hash(fr),
                         splits_hash=CR.CS.content_hash(sp),selection_hash=CR.CS.content_hash(sel),
                         config_sha=sha(out),class_counts=counts.tolist(),n_test=len(sp['test']),
                         n_foreign=len(fr),n_absent_classes=int((counts==0).sum()))
                configs.append(row);manifest.append({k:v for k,v in row.items() if k!='class_counts'})
    frozen=dict(code_hash=code_hash(),inputs=provenance(),configs=configs,created=time.time(),
                expected=dict(core=6292,survey=10164,rf=1936,total=18392))
    pd.DataFrame(manifest).to_csv(ROOT/'config_manifest.csv',index=False)
    atomic_json(lock,frozen);check();return frozen

def phase_plan(phase):
    if phase=='rf':return CR.fit_plan('rf')
    plan=[v for v in CR.fit_plan('lgb') if v[0]!='TGT']
    if phase=='core':return [v for v in plan if v[0]=='Z0' or v[0].startswith('RT')]
    if phase=='survey':return [v for v in plan if v[0].startswith('B_')]
    if phase=='pilot':return [('Z0',f,0) for f in CR.FEATS]+[('RT200',f,0) for f in CR.FEATS]
    raise ValueError(phase)
def target_indices(arm,r,sp,sel):return CR.train_rows(arm,r,sp,sel)[1]
def mask_unseen(P,y_train):
    P=np.asarray(P,dtype=np.float64).copy();seen=np.unique(y_train)
    P[:,~np.isin(np.arange(10),seen)]=0
    P/=P.sum(axis=1,keepdims=True)
    assert np.isfinite(P).all() and np.min(P)>=0
    return P

def stable_seen_softmax(logits,y_train):
    """先在logit空间排除未见类别，再稳定softmax；避免先softmax后条件化的下溢。"""
    logits=np.asarray(logits,dtype=np.float64)
    assert np.isfinite(logits).all(), 'non-finite model logits'
    seen=np.unique(y_train);z=logits[:,seen];z=z-z.max(axis=1,keepdims=True)
    e=np.exp(z);p=np.zeros_like(logits);p[:,seen]=e/e.sum(axis=1,keepdims=True)
    return p

def fit_safely(learner,Xtr,ytr,weights,Xte,seed,trees):
    if learner=='lgb':
        import lightgbm as lgb
        m=lgb.train(CR.lgb_params(2,seed),lgb.Dataset(Xtr,ytr,weight=weights),num_boost_round=trees)
        logits=m.predict(Xte,num_iteration=trees,raw_score=True)
        p=stable_seen_softmax(logits,ytr)
        diagnostics=dict(logit_abs_max=float(np.abs(logits).max()),
                         unseen_raw_argmax_fraction=float((~np.isin(logits.argmax(axis=1),np.unique(ytr))).mean()))
    else:
        p=mask_unseen(CR.fit_predict(learner,Xtr,ytr,weights,Xte,seed,2,trees),ytr)
        diagnostics=dict(logit_abs_max=np.nan,unseen_raw_argmax_fraction=0.)
    assert np.isfinite(p).all() and np.min(p)>=0 and np.max(abs(p.sum(axis=1)-1))<1e-8
    diagnostics['saturated_fraction']=float((p.max(axis=1)>1-1e-7).mean())
    return p,diagnostics

def init_worker():
    global D
    D=CR.G.GData()
def run_config(config,phase,signature,trees=None):
    global D
    cid=config['id'];learner='rf' if phase=='rf' else 'lgb'
    sp,sel=CR.load_unit(OLD,config['unit'])
    assert CR.CS.content_hash(sp)==config['splits_hash']
    assert CR.CS.content_hash(sel)==config['selection_hash']
    cp=ROOT/'configs'/f'{cid}.npz';assert sha(cp)==config['config_sha']
    fr=np.load(cp)['foreign'];assert CR.G.sel_hash(fr)==config['foreign_hash']
    assert len(fr)==5000 and not np.isin(fr,sp['test']).any()
    if config['source']!='MULTI5K':assert np.all(D.ctry[fr]==config['source'])
    else:assert set(D.ctry[fr])==set(COUNTRIES)-{config['target']}
    cw=CR.class_weights_from(D.y[fr]);out=ROOT/'probs'/cid;out.mkdir(parents=True,exist_ok=True)
    records=ROOT/'runs'/f'{cid}.csv';records.parent.mkdir(exist_ok=True)
    locks=ROOT/'locks';locks.mkdir(exist_ok=True)
    nnew=0
    with (locks/f'{cid}.lock').open('w') as lk:
        fcntl.flock(lk,fcntl.LOCK_EX|fcntl.LOCK_NB)
        rows=pd.read_csv(records,keep_default_na=False).to_dict('records') if records.exists() else []
        assert all(z['code_hash']==signature for z in rows),'resume code mismatch'
        done={(z['learner'],z['arm'],z['feat'],int(z['r'])):z for z in rows}
        for arm,feat,r in phase_plan(phase):
            key=(learner,arm,feat,r);dest=out/f'{learner}_{arm}_{feat}_r{r}.npz'
            if key in done and done[key]['status']=='ok' and dest.exists():
                assert sha(dest)==done[key]['prob_sha'],'已保存概率损坏';continue
            assert shutil.disk_usage(ROOT).free>4*1024**3,'可用空间小于4GB，停止以保留进度'
            ti=target_indices(arm,r,sp,sel);assert not np.isin(ti,sp['test']).any()
            if len(ti):assert np.all(D.ctry[ti]==config['target'])
            train=np.r_[fr,ti];seedstr=CR.train_rows(arm,r,sp,sel)[2];seed=CR.CS.seed_of(seedstr)
            t0=time.time();X=D.X[feat];Xtr=np.asarray(X[train]);Xte=np.asarray(X[sp['test']]);ytr=D.y[train]
            weights=CR.sample_weights(arm,D.y[fr],D.y[ti],cw)
            status='failed';error='';attempt=0;diagnostics={}
            for attempt in (1,2):
                try:
                    P,diagnostics=fit_safely(learner,Xtr,ytr,weights,Xte,seed,trees or CR.TREES[learner])
                    assert P.shape==(len(sp['test']),10)
                    tmp=dest.with_suffix('.tmp.npz');np.savez_compressed(tmp,P_test=P.astype(np.float32));os.replace(tmp,dest)
                    status='ok';break
                except Exception:
                    error=traceback.format_exc()
            duration=time.time()-t0
            row=dict(id=cid,unit=config['unit'],source=config['source'],target=config['target'],learner=learner,
                     arm=arm,feat=feat,r=r,k=len(ti),n_train=len(train),sel_hash=CR.G.sel_hash(ti),
                     train_sha=CR.G.sel_hash(train),seed=seed,trees=trees or CR.TREES[learner],threads=2,
                     seconds=round(duration,3),peak_rss_mb=CR.peak_rss_mb(),status=status,attempts=attempt,
                     error=error if status!='ok' else '',code_hash=signature,prob_sha=sha(dest) if status=='ok' else '',
                     source_missing=int((np.bincount(D.y[fr],minlength=10)==0).sum()),
                     train_missing=int((np.bincount(ytr,minlength=10)==0).sum()))
            row.update(diagnostics)
            rows=[z for z in rows if (z['learner'],z['arm'],z['feat'],int(z['r']))!=key]+[row]
            tmp=records.with_suffix('.tmp');pd.DataFrame(rows).to_csv(tmp,index=False);os.replace(tmp,records)
            print(f'{phase} {cid} {learner} {arm} {feat} r{r}: {status} {duration:.1f}s',flush=True)
            if status!='ok':raise RuntimeError(f'{cid} {key} failed: {error}')
            nnew+=1
    return nnew

def run_phase(frozen,phase,jobs=5):
    configs=frozen['configs']
    if phase=='pilot':
        wanted={'AT__BE_F1','RO__BE_F1','SE__PT_F1','BE__RO_F1','MULTI5K__BE_F1'}
        configs=[v for v in configs if v['id'] in wanted]
    t0=time.time();atomic_json(ROOT/'status.json',dict(state='running',phase=phase,started=t0,pid=os.getpid()))
    pool=ProcessPoolExecutor(max_workers=jobs,mp_context=multiprocessing.get_context('spawn'),initializer=init_worker)
    try:
        fs=[pool.submit(run_config,c,phase,frozen['code_hash']) for c in configs]
        for f in as_completed(fs):f.result()
    except BaseException:
        pool.shutdown(wait=True,cancel_futures=True)
        raise
    else:pool.shutdown(wait=True)
    check()
    elapsed=time.time()-t0
    report=dict(phase=phase,seconds=elapsed,configs=len(configs),finished=time.time())
    atomic_json(ROOT/f'{phase}_complete.json',report)
    if phase=='pilot':
        selected=pd.concat([pd.read_csv(ROOT/'runs'/f'{c["id"]}.csv') for c in configs])
        report.update(fits=len(selected),median_seconds=float(selected.seconds.median()),mean_seconds=float(selected.seconds.mean()),
                      projected_fit_hours=float(selected.seconds.mean()*18392/jobs/3600),peak_rss_mb=float(selected.peak_rss_mb.max()))
        atomic_json(ROOT/'pilot_timing.json',report)
    print('PHASE COMPLETE',report,flush=True)
    return report

def pipeline():
    lockpath=ROOT/'pipeline.lock';ROOT.mkdir(exist_ok=True)
    with lockpath.open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            frozen=prepare()
            for phase in ('pilot','core','survey','rf'):
                run_phase(frozen,phase)
            atomic_json(ROOT/'status.json',dict(state='analyzing',pid=os.getpid(),time=time.time()))
            subprocess.run([sys.executable,'-B',str(HERE/'analyze.py')],check=True,cwd=PROJECT)
            check();atomic_json(ROOT/'status.json',dict(state='complete',finished=time.time(),fits=18392,manuscript_unchanged=True))
        except BaseException:
            error=traceback.format_exc();atomic_json(ROOT/'status.json',dict(state='failed',time=time.time(),error=error));raise

def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','pilot','core','survey','rf','pipeline']);a=ap.parse_args()
    if a.action=='pipeline':pipeline()
    else:
        frozen=prepare()
        if a.action!='prepare':run_phase(frozen,a.action)
        else:print('Prepared configurations:',len(frozen['configs']),flush=True)
if __name__=='__main__':main()
