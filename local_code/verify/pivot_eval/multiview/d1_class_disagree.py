# D1：逐（国家, 真实类）看两视图的召回、分歧率，以及"隐藏类"（两视图召回都低）是否集中在分歧区。
from common_mv import *
import pandas as pd
rows = []
for t in COUNTRIES:
    for s in (0, 1):
        u = unit(t, s, 0)
        y, ya, ys = u['y'], u['Pa'].argmax(1), u['Ps'].argmax(1)
        dis = ya != ys
        for c in u['big']:
            m = y == c
            if m.sum() == 0: continue
            rows.append(dict(target=t, split=s, cls=CLASSES[c], n_pool=int(m.sum()), share=100*m.mean(),
                             rec_ae=100*(ya[m]==c).mean(), rec_s2=100*(ys[m]==c).mean(),
                             rec_either=100*((ya[m]==c)|(ys[m]==c)).mean(),
                             dis_rate=100*dis[m].mean(), share_in_dis=100*(y[dis]==c).mean(),
                             pred_share_ae=100*(ya==c).mean(), pred_share_s2=100*(ys==c).mean()))
        # 汇总：隐藏类
df = pd.DataFrame(rows)
df.to_csv('d1_class_disagree.csv', index=False)
g = df.groupby(['target', 'cls']).mean(numeric_only=True).drop(columns='split')
g['hidden'] = g[['rec_ae', 'rec_s2']].max(1) < 30
g['enrich'] = g.share_in_dis / g.share
pd.set_option('display.width', 250)
print(g.round(1).to_string())
H = g[g.hidden]; V = g[~g.hidden]
print('\n隐藏类（max 召回<30%）数量', len(H), '/', len(g))
print('隐藏类：平均分歧率 %.1f，富集比中位数 %.2f，rec_either 均值 %.1f' % (H.dis_rate.mean(), H.enrich.median(), H.rec_either.mean()))
print('可见类：平均分歧率 %.1f，富集比中位数 %.2f' % (V.dis_rate.mean(), V.enrich.median()))
# 国家层：分歧集合中隐藏类占比 vs 全池中隐藏类占比
cc = []
for t, gg in g.groupby(level=0):
    cc.append(dict(target=t, hid_share_pool=gg[gg.hidden].share.sum(), hid_share_dis=gg[gg.hidden].share_in_dis.sum(),
                   n_hidden=int(gg.hidden.sum()), n_big=len(gg)))
print(pd.DataFrame(cc).round(1).to_string())
