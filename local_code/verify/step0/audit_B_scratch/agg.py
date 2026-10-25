import re, numpy as np, pandas as pd
from scipy import stats
B='/Users/perng_mini/AlphaCrops_local/verify/step0/partB/'
F=pd.read_csv(B+'fa_rows.csv'); FZ=F[F.anchor=='Z0']
W=FZ.pivot_table(index=['target','split','rep','clf'],columns='arm',values='FA',dropna=False)
def tstat(v):
    v=v.dropna(); n=len(v); m=v.mean(); sd=v.std(ddof=1); q=stats.t.ppf(.975,n-1)
    return f'{m:+.2f} [{m-q*sd/np.sqrt(n):+.2f}, {m+q*sd/np.sqrt(n):+.2f}]，>0 {(v>0).sum()}/{n}'
def cvals(d):
    d=d.rename('d').reset_index()
    ae=d[d.clf=='ae'].groupby('target').d.mean(); s2=d[d.clf=='s2'].groupby('target').d.mean()
    return {'ae':ae,'s2':s2,'both':pd.concat([ae,s2],axis=1).mean(axis=1,skipna=False)}
summ=open(B+'summary_B.txt').read()
bad=0; tot=0
pairs={f'ΔFA(RT{k}) = FA(RT{k}) − FA(Z0+)':(f'RT{k}','Z0+') for k in (50,200,400)}
pairs.update({f'ΔFA(HD{k}) = FA(HD{k}) − FA(Z0+)':(f'HD{k}','Z0+') for k in (50,200,400)})
pairs.update({f'FA(RT{k}) − FA(HD{k})':(f'RT{k}',f'HD{k}') for k in (50,200,400)})
for title,(a,b) in pairs.items():
    cv=cvals(W[a]-W[b])
    blk=summ.split(f'-- {title} --')[1].split('--')[0]
    for f in ('ae','s2','both'):
        for drop in (False,True):
            v=cv[f].drop(['RO','PT']) if drop else cv[f]
            s=tstat(v); tag='9 国' if drop else '11 国'
            line=[l for l in blk.splitlines() if l.strip().startswith(f'{f}') and tag in l][0]
            tot+=1
            if s not in line: bad+=1; print('不一致',title,f,tag,s,'|',line)
print(f'B3 FA 块：{tot} 行，不一致 {bad}')
# 可达率表
for arm in ['Z0+','RT50','RT200','RT400','HD50','HD200','HD400','RT200mlp','RT200lr']:
    x=FZ[FZ.arm==arm]; print(arm,'全',int((x.reach>=1).sum()),'部分',int(x.partial.sum()),'缺失',int(x.fa_missing.sum()),'缺失国',sorted(x[x.fa_missing].target.tolist()))
# B4 重算（both 9 国 RT200）
for a,b in (('RT200','Z0+'),('HD200','Z0+'),('RT200','HD200')):
    d=(W[a]-W[b]); cv4=cvals(d)['both'].drop(['RO','PT'])
    dr0=d[d.index.get_level_values('rep')==0]; cvr=cvals(dr0)['both'].drop(['RO','PT'])
    ub=d.unstack('clf').mean(axis=1,skipna=False).unstack('rep'); ub=ub[~ub.index.get_level_values('target').isin(['RO','PT'])].dropna()
    s2rep=((ub[0]-ub[1])**2/2).mean(); sdp=np.sqrt(cv4.std()**2+s2rep/4)
    print(f'B4 {a}-{b} both 9 国：SD4 {cv4.std():.2f} SDr0 {cvr.std():.2f} SDproj {sdp:.2f}')
# B5
S=pd.read_csv(B+'splithalf_rows.csv')
SB=S.groupby(['target','split','rep','unit','clf','arm']).bias.mean()
for arm in ['Z0','Z0+','RT200','HD200','HD50']:
    x=SB.xs(arm,level='arm'); print('B5',arm,'n',x.notna().sum(),'均值',round(x.mean(),3),'max',round(x.max(),2))
hd=S[S.arm=='HD50'].groupby(['unit','clf']).bias.apply(lambda v:v.notna().sum())
print('HD50 单方向可用的单元×特征数',int((hd==1).sum()),'两方向都缺',int((hd==0).sum()))
