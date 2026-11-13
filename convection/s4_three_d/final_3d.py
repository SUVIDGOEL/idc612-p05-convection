"""Final numbers for the 3D movie (Ra = 1e8), with 95% moving-block-bootstrap intervals.

Walls: 1D x-spectra (periodic along the wall). Lid and floor: shell-averaged 2D spectrum, and 1D spectra along
x and along y separately (isotropy check). Own straight range with the same rule as own_bands.py.
"""
from convection.config import CACHE
import sys
import numpy as np
from convection.common.stats import slope, per_frame_slopes, integral_time, block_bootstrap_slope
from convection.s3_entropy.entropy import S_of

d = np.load(f"{CACHE}/spec3d.npz")
WALLS = ["wall_xz", "wall_yz", "wall_back1", "wall_back2"]


def ci(S, lo, hi):
    tv, _ = integral_time(per_frame_slopes(S, lo, hi)); L = max(1, int(np.ceil(4 * tv)))
    bs = block_bootstrap_slope(S, lo, hi, L, nboot=1000)
    return (slope(S.mean(0), lo, hi), *np.percentile(bs, [2.5, 97.5]))


def own_range(E, kmax, tol=0.15):
    k = np.arange(1, len(E) + 1.0)
    def ws(c, w=1.5):
        m = (k >= c / w) & (k <= c * w); return np.polyfit(np.log10(k[m]), np.log10(E[m]), 1)[0]
    ks = np.arange(4, int(kmax / 1.5) + 1); loc = np.array([ws(c) for c in ks])
    stop = ks[np.argmax(loc < -3)] if np.any(loc < -3) else ks[-1]
    best = None
    for i, a in enumerate(ks):
        for b in ks[i:]:
            if b > stop or b / a < 2: continue
            s = slope(E, a, b); m = (ks >= a) & (ks <= b)
            if np.all(np.abs(loc[m] - s) <= tol) and (best is None or b / a > best[1] / best[0]):
                best = (a, b, s)
    return best


fmt = lambda t: "%+.2f [%+.2f, %+.2f]" % t
out = {}
W = np.stack([d[w].astype(float) for w in WALLS]).mean(0)          # frames x k
out["walls_k5_15"] = ci(W, 5, 15); out["walls_k10_30"] = ci(W, 10, 30)
for tol in (0.15, 0.20):
    r = own_range(W.mean(0), 120, tol)
    if r:
        a, b, _ = r; out[f"walls_own_{tol}"] = (a, b) + ci(W, a, b)
        print("walls own range (±%.2f): k=%d-%d %s" % (tol, a, b, fmt(out[f"walls_own_{tol}"][2:])))
    else:
        out[f"walls_own_{tol}"] = None; print("walls own range (±%.2f): none of factor 2" % tol)
print("walls   k5-15 %s   k10-30 %s" % (fmt(out["walls_k5_15"]), fmt(out["walls_k10_30"])))
for face in ("lid", "floor"):
    A = d[face].astype(float); n = (A.shape[1] - 2) // 3
    Es, Ex, Ey = A[:, :n], A[:, n + 1:2 * n + 1], A[:, 2 * n + 2:]
    r = own_range(Es.mean(0), 120) or own_range(Es.mean(0), 120, 0.2)
    if r is None:
        print(face, "no straight range"); continue
    a, b, _ = r
    out[face] = {"shell_own": (a, b) + ci(Es, a, b), "x_own": ci(Ex, a, b), "y_own": ci(Ey, a, b),
                 "peak_k": int(np.argmax(Es.mean(0) * np.arange(1, n + 1))) + 1}
    print("%-6s shell own k=%d-%d %s | along x %s | along y %s | k·E peak at k=%d" % (
        face, a, b, fmt(out[face]["shell_own"][2:]), fmt(out[face]["x_own"]), fmt(out[face]["y_own"]), out[face]["peak_k"]))
St = S_of(W[:, :150]); tau, _ = integral_time(St)
out["S_walls"] = (St.mean(), St.std() / np.sqrt(len(St) / (2 * tau)))
print("hydrodynamic entropy, walls, k<=150: %.2f ± %.2f bits" % out["S_walls"])
np.save(f"{CACHE}/final_3d.npy", out, allow_pickle=True)
