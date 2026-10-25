import numpy as np
from final_rule_sim import country_means, rules
rng = np.random.default_rng(11)
for M in [2, 3, 4, 7]:
    for tau in [0.15, 0.3, 0.5]:
        out = []
        for mu in [0.2, 0.3, 0.4, 0.5]:
            x, w = country_means(mu, 0, tau, M, 4, rng)
            out.append(round(rules(x, w)['P_better'], 2))
        x, w = country_means(0, 0, tau, M, 4, rng)
        r = rules(x, w)
        print(f'M={M} tau={tau} power@mu=.2/.3/.4/.5 = {out}  TOST.3@0={r["P_eq3"]:.2f} TOST.5@0={r["P_eq5"]:.2f} hw95={r["mean_halfwidth95"]:.2f}')
