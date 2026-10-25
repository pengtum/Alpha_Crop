import numpy as np, pandas as pd
from scipy import stats
D1='/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops/label_expansion/storyline/decision_curve'
key=['target','split','rep','clf']
def ci(v):
    v=np.asarray(v,float); m=v.mean(); se=v.std(ddof=1)/np.sqrt(len(v)); t=stats.t.ppf(.975,len(v)-1)
    return f'{m:+.2f} [{m-t*se:+.2f},{m+t*se:+.2f}] {int((v>0).sum())}/{len(v)}'
def country(df,cols): return df.groupby(['target','clf'])[cols].mean().groupby('target').mean()
def rep(dfc,label):
    for sub,nm in ((None,'11'),(['RO','PT'],'9')):
        x=dfc if sub is None else dfc.drop([c for c in sub if c in dfc.index])
        print(f'  {label} [{nm}]', ' | '.join(f'{c} {ci(x[c])}' for c in x.columns))
def fr_ba(curve,oa_t):
    c=curve.sort_values('tau'); o=c.oa.values; b=c.ba_ge30.values
    cand=list(b[o>=oa_t])
    for i in range(len(o)-1):
        lo,hi=sorted((o[i],o[i+1]))
        if lo<=oa_t<=hi and o[i]!=o[i+1]:
            cand.append(b[i]+(oa_t-o[i])/(o[i+1]-o[i])*(b[i+1]-b[i]))
    return max(cand) if cand else np.nan
def fr_oa(curve,ba_t):
    c=curve.sort_values('tau'); o=c.oa.values; b=c.ba_ge30.values
    cand=list(o[b>=ba_t])
    for i in range(len(o)-1):
        lo,hi=sorted((b[i],b[i+1]))
        if lo<=ba_t<=hi and b[i]!=b[i+1]:
            cand.append(o[i]+(ba_t-b[i])/(b[i+1]-b[i])*(o[i+1]-o[i]))
    return max(cand) if cand else np.nan
R=pd.read_csv('/private/tmp/c3_skeptic/retrain_tau_rows.csv')
Zp=pd.read_csv('/private/tmp/c3_skeptic/zerolabel_rows.csv'); Zp=Zp[Zp.design=='tau']
DR=pd.read_csv(D1+'/decision_curve_rows.csv')
U=pd.read_csv(D1+'/decision_curve_units.csv')
Hd=U.pivot_table(index=key+['k'],columns='metric',values='head').reset_index()   # D1 head, 5-rep mean
Hg=DR[DR.design=='srs_G_check'][key+['k','oa','ba_ge30']]                    # head on G picks
def matched(heads, curves_by_unit, label, base_curve=None):
    out=[]
    for _,h in heads.iterrows():
        u=tuple(h[c] for c in key)
        cur=curves_by_unit(u,int(h.k))
        fb=fr_ba(cur,h.oa); fo=fr_oa(cur,h.ba_ge30)
        dom=np.isnan(fb)
        if dom:  # head OA above whole curve: compare to τ=0 point
            fb=cur[cur.tau==0].ba_ge30.values[0]
        out.append(dict(zip(key,u),k=int(h.k),gapBA=h.ba_ge30-fb,gapOA=(h.oa-fo) if not np.isnan(fo) else np.nan,dom=dom))
    o=pd.DataFrame(out)
    for k in sorted(o.k.unique()):
        ok=o[o.k==k]
        print(f' {label} k={k}: head OA 高于整条曲线的单元 {int(ok.dom.sum())}/{len(ok)}')
        rep(country(ok,['gapBA']),f'   BA@matchedOA')
        rep(country(ok.dropna(subset=['gapOA']),['gapOA']),f'   OA@matchedBA')
    return o
ZpI=Zp.set_index(key)
print('== A. D1 头（5 重复平均）对 外国底座 + 零标签 τ（π̂=候选池均值），插值前沿 ==')
matched(Hd[Hd.k.isin([50,200,400])], lambda u,k: ZpI.loc[u], 'A')
RI=R.set_index(key)
print('== A2. 同上但 π̂=测试集均值（转导）==')
matched(Hd[Hd.k.isin([50,200,400])], lambda u,k: (lambda d: d[(d.src=='base')])(RI.loc[u]), 'A2')
print('== B. 标签配对头（G 同一批选样）对 同一批标签重训 + 零标签 τ（π̂=测试均值）==')
matched(Hg[Hg.k.isin([50,200,400])], lambda u,k: (lambda d: d[(d.src=='retrain')&(d.k==k)])(RI.loc[u]), 'B')
print('== C. 重训(k)+τ 前沿 对 外国底座+τ 前沿：重训中标签的贡献（控制工作点）: 取重训 τ=0 点对底座前沿 ==')
Rt0=R[(R.src=='retrain')&(R.tau==0)][key+['k','oa','ba_ge30']]
matched(Rt0[Rt0.k.isin([50,200,400])], lambda u,k: (lambda d: d[(d.src=='base')])(RI.loc[u]), 'C')
print('== D. 固定 τ（不看测试标签可预先设定）：重训(k)+τ 与 标签配对头 的差 ==')
for tau in [0.25,0.5]:
    rr=R[(R.src=='retrain')&(np.isclose(R.tau,tau))].set_index(key+['k'])[['ba_ge30','oa']]
    bb=R[(R.src=='base')&(np.isclose(R.tau,tau))].set_index(key)[['ba_ge30','oa']]
    hg=Hg.set_index(key+['k'])[['ba_ge30','oa']]
    for k in [50,200,400]:
        d=(hg.xs(k,level='k')-rr.xs(k,level='k')).reset_index().rename(columns={'ba_ge30':'dBA','oa':'dOA'})
        rep(country(d,['dBA','dOA']),f'head_G{k} − retrain{k}+τ{tau}')
    b0=R[(R.src=='base')&(R.tau==0)].set_index(key)[['ba_ge30','oa']]
    d=(bb-b0).reset_index().rename(columns={'ba_ge30':'dBA','oa':'dOA'})
    rep(country(d,['dBA','dOA']),f'base+τ{tau}(test-mean) − base')
    for k in [200]:
        r0=R[(R.src=='retrain')&(R.tau==0)].set_index(key+['k'])[['ba_ge30','oa']].xs(k,level='k')
        d=(rr.xs(k,level='k')-r0).reset_index().rename(columns={'ba_ge30':'dBA','oa':'dOA'})
        rep(country(d,['dBA','dOA']),f'retrain{k}+τ{tau} − retrain{k}')
