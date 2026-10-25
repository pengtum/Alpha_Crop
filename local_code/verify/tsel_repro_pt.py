# 任务B 审查：复现 PT 的选样（不训练模型），核对 n_classes_sel、kmeans_pick 补足次数、有效目标权重
import sys, time, numpy as np, pandas as pd, pyarrow.parquet as pq
from threadpoolctl import threadpool_limits
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.class_weight import compute_sample_weight
P='/Users/perng_mini/AlphaCrops_local/data/dataset_v2.parquet'; R='/Users/perng_mini/AlphaCrops_local/results/'
COUNTRIES = ['AT', 'BE', 'DE', 'EE', 'LT', 'LV', 'PT', 'RO', 'SE', 'SI', 'SK']
FULL1000 = len(sys.argv) > 1
pf=pq.ParquetFile(P); names=pf.schema_arrow.names
meta=pq.read_table(P,columns=['country','longitude','latitude','label','has_ae','has_s2']).to_pandas()
keep=np.flatnonzero(meta.has_ae.to_numpy() & meta.has_s2.to_numpy())
df=meta.iloc[keep].reset_index(drop=True)
classes=sorted(df.label.unique()); y=df.label.map({c:i for i,c in enumerate(classes)}).to_numpy(); ctry=df.country.to_numpy()
tgt='PT'
rng=np.random.default_rng(42)
foreign=np.concatenate([rng.choice(np.flatnonzero(ctry==c),5000,replace=False) for c in COUNTRIES if c!=tgt])
t_idx=np.flatnonzero(ctry==tgt); pool,test=train_test_split(t_idx,test_size=0.3,random_state=42)
need=np.r_[foreign,pool]
X={}
for f in ['ae','s2']:
    cols=[c for c in names if c.startswith(f+'_')]
    tab=pq.read_table(P,columns=cols).take(keep[need])
    X[f]=np.column_stack([tab.column(c).to_numpy().astype(np.float32) for c in cols]); del tab
nF=len(foreign)
# 与原代码一致：float32 -> float64 标准化 + PCA32
desc={}
with threadpool_limits(2):
    for f in ['ae','s2']:
        Z=X[f].astype(np.float64); Z=(Z-Z.mean(0))/(Z.std(0)+1e-9)
        Z=PCA(32,random_state=42).fit_transform(Z); desc[f]=(Z[:nF],Z[nF:])
    gap={f:NearestNeighbors(n_neighbors=1).fit(desc[f][0]).kneighbors(desc[f][1])[0].ravel() for f in ['ae','s2']}
lonlat=df[['longitude','latitude']].to_numpy()
ll=lonlat[pool]; ll=(ll-ll.mean(0))/ll.std(0)
fill_log=[]
def kmeans_pick(Z,k,seed,tag):
    k=min(k,len(Z))
    km=KMeans(n_clusters=k,n_init=1,random_state=seed,max_iter=100).fit(Z)
    nn=NearestNeighbors(n_neighbors=1).fit(Z)
    idx=nn.kneighbors(km.cluster_centers_,return_distance=False).ravel()
    u=np.unique(idx); nfill=k-len(u)
    empty=k-len(np.unique(km.labels_))
    fill_log.append((tag,k,seed,nfill,empty))
    idx=u
    if len(idx)<k:
        rest=np.setdiff1d(np.arange(len(Z)),idx)
        idx=np.r_[idx,np.random.default_rng(seed).choice(rest,k-len(idx),replace=False)]
    return idx
csv=pd.read_csv(R+'pilot_selection.csv')
probs={cf:np.load(R+f'pilot_probs_{tgt}_{cf}.npy') for cf in ['ae','s2']}
print('probs shape',{k:v.shape for k,v in probs.items()},'rowsum',{k:float(np.abs(v.sum(1)-1).max()) for k,v in probs.items()})
out=[]
with threadpool_limits(2):
  for k in [50,200,1000]:
    seeds=[0,1,2] if (k<1000 or FULL1000) else [0]
    for seed in seeds:
      for strat in ['random','geo_strat','s2_rep','ae_rep','s2_gap','ae_gap','uncertainty']:
        t0=time.time()
        r=np.random.default_rng(1000*seed+k)
        for cf in ['ae','s2']:
          if strat=='random': pk=r.choice(len(pool),k,replace=False) if cf=='ae' else pk
          elif strat=='geo_strat': pk=kmeans_pick(ll,k,seed,strat) if cf=='ae' else pk
          elif strat in ('s2_rep','ae_rep'): pk=kmeans_pick(desc[strat[:2]][1],k,seed,strat) if cf=='ae' else pk
          elif strat in ('s2_gap','ae_gap'):
            if cf=='ae':
              f=strat[:2]; cand=np.argsort(-gap[f])[:max(k,int(0.3*len(pool)))]
              pk=cand[kmeans_pick(desc[f][1][cand],k,seed,strat)]
          else:
            p0=probs[cf]; ent=-(p0*np.log(p0+1e-12)).sum(1)
            cand=np.argsort(-ent)[:max(k,int(0.3*len(pool)))]
            pk=cand[kmeans_pick(desc[cf][1][cand],k,seed,strat+'_'+cf)]
          sel=pool[pk]
          tr=np.r_[foreign,sel]; w=np.r_[np.ones(nF),np.full(k,0.25*nF/k)]
          ew=w*compute_sample_weight('balanced',y[tr]); share=ew[nF:].sum()/ew[:nF].sum()
          ncls=len(np.unique(y[sel]))
          c=csv[(csv.target==tgt)&(csv.strategy==strat)&(csv.budget==k)&(csv.seed==seed)&(csv.clf_feat==cf)].n_classes_sel.tolist()
          out.append(dict(strategy=strat,k=k,seed=seed,cf=cf,ncls=ncls,csv_ncls=c[0] if c else None,
                          n_unique=len(np.unique(sel)),eff_share=round(share,3),
                          n_orchards=int((y[sel]==classes.index('orchards')).sum())))
        print(f'{k} {seed} {strat} {time.time()-t0:.1f}s',flush=True)
o=pd.DataFrame(out); pd.set_option('display.width',200); pd.set_option('display.max_rows',500)
o.to_csv('/Users/perng_mini/AlphaCrops_local/verify/tsel_repro_pt.csv',index=False)
print(o.to_string())
print('ncls mismatch rows:',(o.ncls!=o.csv_ncls).sum())
print('fill log (tag,k,seed,n_fill,empty_clusters):'); [print(x) for x in fill_log if x[3]>0 or x[4]>0]
print('total kmeans calls',len(fill_log))
print(o.groupby(['k','strategy']).eff_share.agg(['min','max','mean']).to_string())
