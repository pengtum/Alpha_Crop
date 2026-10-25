"""汇总合成实验：各修正的 OA/BA、PO 与 EM 的恢复率、2×2 路径依赖与 Shapley 分解。"""
import pandas as pd

R = pd.read_csv('/Users/perng_mini/AlphaCrops_local/verify/check3_review/synth_arms.csv')
D = pd.read_csv('/Users/perng_mini/AlphaCrops_local/verify/check3_review/synth_diag.csv')
pd.set_option('display.width', 250)
M = R.pivot_table(index=['T', 'delta', 'seed'], columns='arm', values='OA') * 100
B = R.pivot_table(index=['T', 'delta', 'seed'], columns='arm', values='BA') * 100
arms = ['BAYES_T', 'Wnat', 'WPO', 'W0', 'WPO_n3k', 'BIAS_ORACLE', 'PO', 'PO_implied', 'PO_bcts', 'PEM', 'PEM_bcts',
        'BBSE', 'PK50_lap1', 'PK50_lap.5', 'LS', 'L0']
print('OA（%，2 个种子平均）'); print(M[arms].groupby(['T', 'delta']).mean().round(1).T.to_string())
print('\nBA（%）'); print(B[['W0', 'WPO', 'LS', 'PO', 'L0', 'BIAS_ORACLE']].groupby(['T', 'delta']).mean().round(1).T.to_string())
# 恢复率（相对 PO）
rec = pd.DataFrame({k: (M[k] - M['L0']) / (M['PO'] - M['L0']) for k in ['PEM', 'PEM_bcts', 'BBSE', 'PK50_lap1', 'PK50_lap.5', 'PO_bcts']})
print('\n恢复率 (X−L0)/(PO−L0)'); print(rec.groupby(['T', 'delta']).mean().round(2).to_string())
# 2×2：基于 {LS, PO, W0, WPO}
S = M['LS'] - M['L0']
PA = M['PO'] - M['LS']; CA = M['WPO'] - M['PO']          # 先修先验
CB = M['W0'] - M['LS']; PB = M['WPO'] - M['W0']          # 先修条件
tot = M['WPO'] - M['L0']
X = pd.DataFrame(dict(total=tot, S=S, P_priorfirst=PA, P_condfirst=PB, P_shapley=(PA + PB) / 2,
                      C_priorfirst=CA, C_condfirst=CB, C_shapley=(CA + CB) / 2, interaction=PB - PA,
                      share_prior_planA=(S + PA) / tot, share_prior_condfirst=(S + PB) / tot,
                      share_prior_shapley=(S + (PA + PB) / 2) / tot))
print('\n2×2 分解（OA，pp）'); print(X.groupby(['T', 'delta']).mean().round(2).to_string())
# 样本量混杂：WITHIN 用 3k 时总损失与条件部分
Y = pd.DataFrame(dict(total_full=M['WPO'] - M['L0'], total_n3k=M['WPO_n3k'] - M['L0'],
                      C_full=M['WPO'] - M['PO'], C_n3k=M['WPO_n3k'] - M['PO'],
                      Wnat_minus_WPO=M['Wnat'] - M['WPO'], BO_recover=(M['BIAS_ORACLE'] - M['L0']) / (M['WPO'] - M['L0'])))
print('\n样本量与上界'); print(Y.groupby(['T', 'delta']).mean().round(2).to_string())
print('\n诊断'); print(D.groupby(['T', 'delta']).mean().round(3).drop(columns='seed').to_string())
