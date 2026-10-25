"""对修复分析作独立核验和描述性汇总；保留所有不利和缺失结果。"""
from pathlib import Path
import sys,json,hashlib
import numpy as np,pandas as pd
PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT/'label_expansion/single_source_v1'))
from protect_baseline import check
ROOT=PROJECT/'results/single_source_20261009'
OUT=ROOT/'analysis_repaired_20261010'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def equal_target(d,cols):
    keys=['source','target']
    return d.groupby(keys+['unit'])[cols].mean().groupby(keys).mean().groupby('target').mean().mean()
def main():
    assert (OUT/'REPORT.txt').exists()
    assert json.loads((OUT/'input_validation.json').read_text())['status']=='passed'
    names=sorted(p.name for p in (ROOT/'analysis').glob('*.csv'))
    same={name:digest(OUT/name)==digest(ROOT/'analysis'/name) for name in names}
    assert all(same.values()),same
    d=pd.read_csv(OUT/'default_metrics.csv');assert len(d)==18392
    assert not d.duplicated(['id','learner','arm','feat','r']).any()
    n=pd.read_csv(OUT/'numerical_diagnostics.csv');assert len(n)==18392
    lgb=n[n.learner=='lgb'];assert np.isfinite(lgb.logit_abs_max).all()
    assert n[n.learner=='rf'].logit_abs_max.isna().all()
    s=d[d.source!='MULTI5K'];summ=[]
    for (l,a,f),g in s.groupby(['learner','arm','feat']):
        m=equal_target(g,['OA','BA'])
        summ.append(dict(learner=l,arm=a,feat=f,OA=m.OA,BA=m.BA))
    pd.DataFrame(summ).to_csv(OUT/'audit_default_country_means.csv',index=False)
    rows=pd.read_csv(OUT/'unit_rows.csv');delta=rows[(rows.level=='D')&(rows.source!='MULTI5K')].copy()
    diags=n.pivot(index=['id','learner','arm','r'],columns='feat',values='logit_abs_max')
    bad=diags.gt(1e4).any(axis=1).to_dict();delta['extreme_in_pair']=[bad.get((z.id,z.learner,z.name.removeprefix('D_'),z.r),False) for z in delta.itertuples()]
    sensitivity=[]
    for (l,a),g in delta.groupby(['learner','name']):
        for tag,v in [('all',g),('excluding_extreme_diagnostic',g[~g.extreme_in_pair])]:
            pair=v.groupby(['source','target','unit']).value.mean().groupby(['source','target']).mean().dropna();tm=pair.groupby('target').mean()
            sensitivity.append(dict(learner=l,comparison=a,subset=tag,mean=tm.mean(),pairs=len(pair),positive_pairs=int((pair>0).sum()),targets=len(tm)))
    pd.DataFrame(sensitivity).to_csv(OUT/'audit_extreme_sensitivity.csv',index=False)
    multi=pd.read_csv(OUT/'equal_size_multi_contrasts.csv');ms=[]
    for (l,a),g in multi.groupby(['learner','arm']):
        cols=['multi_minus_single_ae','multi_minus_single_s2','representation_interaction'];m=equal_target(g,cols)
        pairs=g.groupby(['source','target'])[cols].mean().dropna()
        ms.append(dict(learner=l,arm=a,valid_rows=int(g[cols[0]].notna().sum()),total_rows=len(g),valid_pairs=len(pairs),**m.to_dict()))
    pd.DataFrame(ms).to_csv(OUT/'audit_multi_country_means.csv',index=False)
    e=pd.read_csv(OUT/'label_equivalents.csv');e=e[e.source!='MULTI5K']
    es=e.groupby(['k_ae','status']).size().unstack(fill_value=0);es.to_csv(OUT/'audit_label_equivalence_status.csv')
    valid=e[e.status!='insufficient_common_oa'].drop_duplicates(['id','r'])
    c=pd.read_csv(OUT/'class_coverage.csv');c=c[c.source!='MULTI5K']
    summary=pd.read_csv(OUT/'representation_summary.csv');primary=summary[summary.target_set=='all11']
    positive_pairs={}
    for (l,a),g in delta.groupby(['learner','name']):
        p=g.groupby(['source','target','unit']).value.mean().groupby(['source','target']).mean().dropna();positive_pairs[f'{l}:{a}']={'positive':int((p>0).sum()),'available':len(p),'planned':110}
    # 额外复核：混淆计数与对应默认OA一致，每个真实类别support一致。
    cm=pd.read_csv(OUT/'confusion_counts.csv');keys=['source','target','unit','feat','arm','r']
    cm['correct']=np.where(cm.true==cm.predicted,cm['count'],0)
    totals=cm.groupby(keys)[['count','correct']].sum();oa=100*totals.correct/totals['count']
    ref=d[(d.learner=='lgb')&d.arm.isin(cm.arm.unique())].set_index(keys).OA
    assert np.allclose(oa.sort_index(),ref.sort_index(),atol=1e-10)
    checked=check()
    validation={'status':'complete','training_models':len(d),'frozen_rse_files_verified':checked,'original_csv_byte_identical':same,'confusion_OA_check':'passed','numeric_conversion':'16456 finite-or-numeric LGB diagnostics; 1936 RF N/A values','extreme_logit_fits':int((lgb.logit_abs_max>1e4).sum()),'extreme_logit_total':len(lgb),'positive_matched_oa_pairs':positive_pairs,'nonmonotone_s2_label_curves':int(valid.s2_curve_nonmonotone.astype(bool).sum()),'valid_s2_label_curves':len(valid),'class_missing_units':int((c.covered_classes<c.eligible_classes).sum()),'class_units':len(c),'max_missing_test_fraction':float(c.missing_target_fraction.max())}
    (OUT/'validation.json').write_text(json.dumps(validation,indent=2))
    report=(OUT/'REPORT.txt').read_text()
    report+='\n\n补充核验与解释（探索性，未改论文）\n'
    report+=f'原分析的 {len(same)} 个CSV与修复重跑逐字节一致；18,392份预测通过原分析的哈希、概率、训练采样检查；混淆计数重新计算OA与默认指标一致。\n'
    report+='\n一、相同OA下的AE−S2差异（BA百分点）\n'+primary.to_string(index=False)+'\n'
    report+='零目标标签下，AE的平均优势在LGB和RF上方向一致。加入200个随机目标标签，或在已有100个样本上补100个随机标签后，LGB差异接近零，国家权重敏感性范围跨零。不能声称AE在所有单源任务及标签预算上稳定占优。\n'
    report+='可用国家对及正向计数：'+json.dumps(positive_pairs,ensure_ascii=False)+'\n'
    report+='\n二、等5,000源标签的多源对照（共同OA窗口内）\n'+pd.DataFrame(ms).to_string(index=False)+'\n'
    report+='多源对照在源样本总量相同条件下比较，平均增益均为正，但增益大小及表征交互依赖场景和学习器；共同OA不可达的比较保留缺失。不同臂的有效对可能不同，不能把其均值差直接当作配对交互检验。\n'
    report+='\n三、标签节省的可识别性\n'+es.to_string()+'\n'
    report+=f'660个单源空间单元×重复中，满足共同OA窗口要求的{len(valid)}条S2标签曲线有{validation["nonmonotone_s2_label_curves"]}条非单调。已有数值依照冻结算法作局部线性内插，无外推；不应把可算子集的平均值表述为普遍节省标签比例。计数中的重复和半区不是独立国家实验。\n'
    report+='\n四、数值问题与类别覆盖\n'
    report+=f'{validation["extreme_logit_fits"]}/{len(lgb)}次LGB拟合出现|logit|>10,000；Z0两种特征均无此标记。极端值是数值诊断信号，不等于这些拟合全部无效；全部原始结果保留。附加剔除标记模型的配对敏感性仅供诊断，不能替代完整样本主结果：\n'+pd.DataFrame(sensitivity).to_string(index=False)+'\n'
    report+=f'{validation["class_missing_units"]}/220个单源×空间单元缺少至少一个目标评估类别；未见源类别涉及的目标测试样本占比最大为{100*c.missing_target_fraction.max():.2f}%。这说明跨国单源任务包含类别覆盖不足，但不能据此将模型差异全归因于表征。\n'
    report+='\n结论：支持AE在零目标标签单源迁移中的优势；对“更丰富表征普遍减少标签需求”的强叙事，证据仍不充分。建议保留为独立探索性结果，讨论后再决定是否纳入论文。RSE正文、图片、备份、冻结实验代码和训练结果均未修改。历史失败状态另存prior_status.json；本目录validation.json记录修复分析完成。\n'
    (OUT/'REPORT.txt').write_text(report)
    print(json.dumps(validation,indent=2))
if __name__=='__main__':main()
