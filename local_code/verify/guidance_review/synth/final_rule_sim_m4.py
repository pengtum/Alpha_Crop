import numpy as np, pandas as pd
from final_rule_sim import country_means, rules
rng = np.random.default_rng(21); rows = []
for tau in [0.15, 0.3, 0.5]:
    for mu, g in [(0, 0), (0, .3), (.1, 0), (.24, .29), (.3, 0), (.5, 0), (-.3, 0)]:
        x, w = country_means(mu, g, tau, 4, 4, rng)
        rows.append(dict(tau=tau, mu=mu, gamma=g, **rules(x, w)))
res = pd.DataFrame(rows); res.to_csv('final_rule_sim_m4.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
print(res[['tau','mu','gamma','P_better','P_better_but_eq3','P_eq3','P_eq5','P_worse','P_inconcl_eq3','sign9','mean_halfwidth95']].round(3).to_string(index=False))
