"""Numbers for the new results, with block-bootstrap 95% intervals."""
from convection.config import CACHE
import sys
import numpy as np
from convection.common.stats import slope, per_frame_slopes, integral_time, block_bootstrap_slope

TAGS = ("Ra1e9", "Ra1e12", "Ra1e15")

def ci_slope(S, lo, hi, nboot=1000):
    tv, _ = integral_time(per_frame_slopes(S, lo, hi))
    L = max(1, int(np.ceil(4 * tv)))
    bs = block_bootstrap_slope(S, lo, hi, L, nboot=nboot)
    return slope(S.mean(0), lo, hi), *np.percentile(bs, [2.5, 97.5])

def crossover(Ee, Ep):
    r = Ee / Ep; k = np.arange(1, len(r) + 1)
    above = np.where(r >= 1)[0]
    return k[above[0]] if len(above) else None

if __name__ == "__main__":
    out = {}
    for t in TAGS:
        d = np.load(f"{CACHE}/pass3_{t}.npz")
        Ee = d["E_ent"].astype(float).mean(1); Ep = d["E_prs"].astype(float).mean(1)   # frames x k
        ve, vp, cv = (d[k].astype(float).mean(0).mean() for k in ("var_ent", "var_prs", "cov_ep"))
        kc = crossover(Ee.mean(0), Ep.mean(0))
        sp = ci_slope(Ep, 10, 30); se = ci_slope(Ee, 10, 30)
        out[t] = dict(share_ent=ve / (ve + vp + 2 * cv), share_prs=vp / (ve + vp + 2 * cv), kc=kc, sp=sp, se=se)
        print("%-7s entropy share %.2f  pressure share %.2f  crossover k_c = %s" % (t, out[t]["share_ent"], out[t]["share_prs"], kc))
        print("        pressure-part slope k10-30 %+.2f [%+.2f, %+.2f]   entropy-part slope %+.2f [%+.2f, %+.2f]" % (*sp, *se))
        for lo, hi in ((5, 15), (10, 50), (20, 60), (30, 100)):
            print("          band k%d-%d: pressure %+.2f  entropy %+.2f" % (lo, hi, slope(Ep.mean(0), lo, hi), slope(Ee.mean(0), lo, hi)))
    np.save(f"{CACHE}/results2.npy", out, allow_pickle=True)
