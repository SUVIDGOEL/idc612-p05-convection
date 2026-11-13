"""Each movie's own power-law range, chosen by a fixed rule (not by eye).

Rule: inside k = 6 .. (noise floor, or where the windowed local slope first drops below -3),
find the widest range [a, b] with b/a >= 2 in which every windowed local slope (fit from k/1.5 to 1.5k)
stays within +-0.15 of the regression slope over [a, b]. Then a block-bootstrap 95% interval for that slope."""
from convection.config import CACHE
import sys
import numpy as np
from convection.common.stats import slope, per_frame_slopes, integral_time, block_bootstrap_slope
FLOOR = {"Ra1e9": 177, "Ra1e12": 873, "Ra1e15": 155}

def window_slope(E, c, w=1.5):
    k = np.arange(1, len(E) + 1.0); m = (k >= c / w) & (k <= c * w)
    return np.polyfit(np.log10(k[m]), np.log10(E[m]), 1)[0]

for t in FLOOR:
    S = np.load(f"{CACHE}/spec_{t}.npz")["clean"].astype(float)
    E = S.mean(0)
    ks = np.arange(6, int(FLOOR[t] / 1.5) + 1)
    loc = np.array([window_slope(E[:FLOOR[t]], c) for c in ks])
    stop = ks[np.argmax(loc < -3)] if np.any(loc < -3) else ks[-1]
    best = None
    for i, a in enumerate(ks):
        for b in ks[i:]:
            if b > stop or b / a < 2: continue
            s = slope(E, a, b)
            m = (ks >= a) & (ks <= b)
            if np.all(np.abs(loc[m] - s) <= 0.15):
                if best is None or np.log(b / a) > np.log(best[1] / best[0]):
                    best = (a, b, s)
    a, b, s = best
    tv, _ = integral_time(per_frame_slopes(S, a, b))
    bs = block_bootstrap_slope(S, a, b, max(1, int(np.ceil(4 * tv))), nboot=1000)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print("%-7s own range k = %3d to %3d (factor %.1f): slope %+.2f [%+.2f, %+.2f]   | common band k=10-30: %+.2f" % (
        t, a, b, b / a, s, lo, hi, slope(E, 10, 30)))
