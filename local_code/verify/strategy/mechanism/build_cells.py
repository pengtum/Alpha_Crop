"""合并 rows × 选样表 × 单元表，得到每个 (单元, 条件, 方法, 分类器, k) 相对同单元 SRS 的差（指标 + 中介 + 逐类）。"""
import sys; sys.path.insert(0,'.')
from load import *
R=load()
S=pd.read_csv('sel_table.csv'); U=pd.read_csv('unit_table.csv')
KEYS=['target','split','rep','cond','method','k']
R=R[R.cond.isin(['AE','S2','none','ORACLE','ORACLE2'])&R.k.isin([0,50,100,200,400])&R.clf.isin(['ae','s2'])]
X=R.merge(S,on=KEYS,how='left').merge(U,on=['target','split','rep'],how='left')
C=range(10)
n=X[[f'n{c}' for c in C]].to_numpy(float); n=np.nan_to_num(n)  # foreign_only: 0
X[[f'n{c}' for c in C]]=n
nT=X[[f'nT{c}' for c in C]].to_numpy(); big=nT>=30
X['nbig_sel']=((n>0)&big).sum(1)
ld=np.where(big,np.log1p(n),np.nan); X['LD']=np.nanmean(ld,1)
wT=np.where(big,nT,0); wT=wT/wT.sum(1,keepdims=True)
X['LD_oa']=np.nansum(np.where(big,np.log1p(n),0)*wT,1)
# 多数类（候选池占比最大的类）的份额变化
pi=X[[f'pi{c}' for c in C]].to_numpy(); jm=pi.argmax(1)
X['maj_share']=n[np.arange(len(X)),jm]/np.maximum(X.k.to_numpy(),1)
X['pi_maj']=pi[np.arange(len(X)),jm]
X['nonbig_share']=np.where(~big,n,0).sum(1)/np.maximum(X.k.to_numpy(),1)
X.to_pickle('X.pkl')
# 相对 SRS 的差
ref=X[X.method=='srs'].set_index(['target','split','rep','clf','k'])
A=X[X.cond.isin(['AE','S2','ORACLE','ORACLE2'])&X.k.isin([50,200])].copy()
key=pd.MultiIndex.from_frame(A[['target','split','rep','clf','k']])
cols=['ba_ge30','oa','ba_P2','oa_P2','ba_P1','f1_ge30','ba_far','oa_far','comp_L1','n_classes_sel','nbig_sel','LD','LD_oa','maj_share','nonbig_share','sc5','any5','spat_ae','spat_s2','far_ae','far_s2','far_common']+[f'rec{c}' for c in C]+[f'n{c}' for c in C]
for c in cols:
    A['d_'+c]=A[c].to_numpy()-ref[c].reindex(key).to_numpy()
# 分类器对应的空间预测贡献
A['d_spat']=np.where(A.clf=='ae',A.d_spat_ae,A.d_spat_s2)
A['d_far_own']=np.where(A.cond=='AE',A.d_far_ae,np.where(A.cond=='S2',A.d_far_s2,np.nan))
A['far_own']=np.where(A.cond=='AE',A.far_ae,np.where(A.cond=='S2',A.far_s2,np.nan))
A.to_pickle('A.pkl')
print(A.shape, A.d_ba_ge30.isna().sum())
print(A.groupby(['cond','method']).size())
