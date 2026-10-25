import numpy as np
from final_rule_sim import country_means, rules
rng = np.random.default_rng(5)
for lab, M, J in [('XY/S2+XY 2法×22单元', 2, 2), ('神谕 2几何法×22单元', 2, 2), ('E2 其余通路 4法×22单元', 4, 2)]:
    for tau in [0.15, 0.3, 0.5]:
        x, w = country_means(0.3, 0, tau, M, J, rng)
        r = rules(x, w)
        print(lab, 'tau', tau, 'power@.3', round(r['P_better'], 2), 'hw95', round(r['mean_halfwidth95'], 2))
