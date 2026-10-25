"""有意义的正确性测试：缺失类别、嵌套采样、配对分析与禁止外推。"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run as R
import analyze as N
import numpy as np,pandas as pd,tempfile
from protect_baseline import check

def main():
    check()
    assert len(R.phase_plan('core'))==26 and len(R.phase_plan('survey'))==42 and len(R.phase_plan('rf'))==8
    assert not (set(R.phase_plan('core'))&set(R.phase_plan('survey')))
    sp,sel=R.CR.load_unit(R.OLD,'BE_F1')
    for r in range(3):
        a=[R.target_indices(f'RT{k}',r,sp,sel) for k in (50,100,200,400)]
        assert all(np.array_equal(a[i],a[i+1][:len(a[i])]) for i in range(3))
        for strategy in ('SRS','EXT','BA'):
            a=R.target_indices(f'B_{strategy}100',r,sp,sel);b=R.target_indices(f'B_{strategy}300',r,sp,sel)
            assert len(a)==200 and len(b)==400 and set(a)<=set(b)
            assert np.array_equal(a[:100],sel[f'S0_r{r}'])
    rng=np.random.default_rng(6);X=rng.normal(size=(240,8)).astype(np.float32);y=np.tile([0,2,7],80)
    p,diag=R.fit_safely('lgb',X,y,np.ones(240),X[:31],42,5)
    assert p.shape==(31,10) and np.all(p[:,[1,3,4,5,6,8,9]]==0) and np.allclose(p.sum(axis=1),1)
    # 极端未见类别logit曾导致softmax后可见类别全部变零；logit条件化必须稳定。
    zz=np.array([[0.,1.e6,2.]+[0.]*7,[0.,-1.e6,1.]+[0.]*7]);q=R.stable_seen_softmax(zz,np.array([0,2]));assert np.isfinite(q).all() and np.allclose(q.sum(axis=1),1) and np.all(q[:,1]==0)
    assert np.allclose(q[0,[0,2]],np.exp([0.,2.])/np.exp([0.,2.]).sum())
    # RF确实把稀疏类编号补到十列，而不是误用列的位置。
    old=R.CR.ML.PARAMS['rf'].copy()
    R.CR.ML.PARAMS['rf']=dict(R.CR.RF_PARAMS,n_estimators=5)
    q=R.mask_unseen(R.CR.fit_predict('rf',X,y,np.ones(240),X[:31],42,2,5),y)
    R.CR.ML.PARAMS['rf']=old
    assert np.all(q[:,[1,3,4,5,6,8,9]]==0) and np.allclose(q.sum(axis=1),1)
    assert N.invert_curve([0,50,100],[20,40,60],50)==(75.,'interpolated')
    assert N.invert_curve([0,50,100],[20,40,60],10)==(0.,'already_met_at_zero')
    assert np.isnan(N.invert_curve([0,50,100],[20,40,60],70)[0])
    assert N.invert_curve([0,50,100],[20,15,30],25)[0]>50
    # 相同概率输入应得到精确零差，且全套臂均能用冻结分析接口读取。
    yy=rng.choice(10,size=1000,p=[.4,.15,.12,.08,.07,.06,.04,.03,.03,.02]);pred=rng.dirichlet([8,2,1,1,1,1,1,1,1,1],size=1000);pred[np.arange(1000),yy]+=.3;pred/=pred.sum(axis=1,keepdims=True)
    probs={(l,f,a,r):pred.astype(np.float32) for l in ('lgb','rf') for a,f,r in R.CR.fit_plan(l)}
    cv=N.A.unit_curves(yy,np.arange(10),probs)
    rows=pd.DataFrame(N.A.unit_rows('BE_F1',cv,'formal',3))
    assert set(rows.level)=={'arm','D','diff','t0'}
    assert np.nanmax(abs(rows.loc[rows.level=='D','value']))<1e-9
    config=dict(id='AT__BE_F1',source='AT',target='BE',unit='BE_F1')
    eq=N.label_equivalents(config,cv);assert all(z['n_s2']==0 for z in eq)
    original=N.OUT
    with tempfile.TemporaryDirectory(prefix='alpha_single_tests_') as temp:
        N.OUT=Path(temp)
        rows['source']='AT';rows['target']='BE';rows['id']=config['id']
        N.summarize_effects(rows)
        assert len(pd.read_csv(N.OUT/'representation_summary.csv'))>0
        N.multi_contrasts({'configs':[config]}, {config['id']:cv,'MULTI5K__BE_F1':cv})
        c=pd.read_csv(N.OUT/'equal_size_multi_contrasts.csv');assert c.representation_interaction.abs().max()<1e-9
    N.OUT=original;check();print('PASS: plan sizes, frozen nested selections, absent classes (LGB/RF), probability alignment, frozen scoring, aggregation, contrasts, no extrapolation, unchanged RSE.')
if __name__=='__main__':main()
