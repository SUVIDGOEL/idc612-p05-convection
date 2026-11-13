"""Final numbers for the density panel (2D), each with a 95% moving-block-bootstrap interval.

From read_density_panel (per frame, per height band: E_T, E_rho, co-spectrum) and split_entropy_pressure (entropy and pressure parts):
  - slope of the density spectrum, k = 10-30
  - T-rho coherence at large (k = 1-10), middle (10-30) and small (30-100) sizes, all heights / top / bottom
  - pressure share of the temperature variance and the k where entropy part = pressure part (temperature units)
"""
from convection.config import CACHE
import sys
import numpy as np
from convection.common.fits_and_errors import slope, per_frame_slopes, integral_time

TAGS = ("Ra1e9", "Ra1e12", "Ra1e15")
NBOOT = 1000


def blocks(n, L, nboot, seed=0):
    rng = np.random.default_rng(seed)
    starts = np.arange(n - L + 1); nb = int(np.ceil(n / L))
    for _ in range(nboot):
        yield (rng.choice(starts, nb)[:, None] + np.arange(L)[None, :]).ravel()[:n]


def ci(stat, n, L, point):
    vals = np.array([stat(idx) for idx in blocks(n, L, NBOOT)])
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return point, lo, hi


def fmt(t):
    return "%+.2f [%+.2f, %+.2f]" % t


def main():
    out = {}
    for t in TAGS:
        d2 = np.load(f"{CACHE}/density_panel_{t}.npz"); d3 = np.load(f"{CACHE}/entropy_pressure_split_{t}.npz")
        ET, ER, CO = (d2[k].astype(float) for k in ("ET", "ER", "CO"))          # frames x bands x k
        n, nb, K = ET.shape
        tau, _ = integral_time(per_frame_slopes(ET.mean(1), 10, 30)); L = max(1, int(np.ceil(4 * tau)))
        r = {"L": L}
        # density slope (frames x k, all heights)
        R = ER.mean(1)
        r["rho_slope"] = ci(lambda i: slope(R[i].mean(0), 10, 30), n, L, slope(R.mean(0), 10, 30))
        # coherence: reduce each frame to sums over k-ranges per band first, then resample frames
        rng_ = {"large": (1, 10), "mid": (10, 30), "small": (30, 100)}
        red = {nm: np.stack([X[:, :, lo - 1:hi].sum(2) for X in (CO, ET, ER)], -1) for nm, (lo, hi) in rng_.items()}
        allb, top, bot = list(range(nb)), [0, 1], [nb - 2, nb - 1]
        def coh(arr, i, bands):
            c, a_, b_ = arr[i][:, bands].sum((0, 1)); return c / np.sqrt(a_ * b_)
        for name in rng_:
            for hn, bands in (("all", allb), ("top", top), ("bottom", bot)):
                r[f"coh_{name}_{hn}"] = ci(lambda i: coh(red[name], i, bands), n, L, coh(red[name], np.arange(n), bands))
        # entropy / pressure parts in temperature units: per-frame weighted quantities first
        zc = 1 - (np.arange(nb) + 0.5) / nb
        Tbar = (1 - 0.5 * zc) + np.interp(zc, d3["z"][::-1], d3["prof_theta"][::-1]); w = Tbar ** 2
        Ew = (w[None, :, None] * d3["E_ent"].astype(float)).mean(1)
        Pw = (w[None, :, None] * d3["E_prs"].astype(float)).mean(1)
        vew = (w * d3["var_ent"].astype(float)).sum(1); vpw = (w * d3["var_prs"].astype(float)).sum(1)
        def share(i):
            return vpw[i].sum() / (vew[i].sum() + vpw[i].sum())
        kk = np.arange(1, Ew.shape[1] + 1)
        edges = np.unique(np.round(np.logspace(0, np.log10(len(kk)), 40)).astype(int)) - 1
        def cross(i):
            e = Ew[i].mean(0); p_ = Pw[i].mean(0)
            for a_, b_ in zip(edges[:-1], edges[1:]):
                if b_ > a_ and e[a_:b_].sum() >= p_[a_:b_].sum():
                    return kk[a_:b_].mean()
            return np.nan
        def pslope(i):
            return slope(Pw[i].mean(0), 10, 30)
        allidx = np.arange(n)
        r["pressure_share"] = ci(share, n, L, share(allidx))
        r["crossover_k"] = ci(cross, n, L, cross(allidx))
        r["pressure_slope"] = ci(pslope, n, L, pslope(allidx))
        out[t] = r
        print(f"== {t}  (block length {L} frames)")
        print("   density slope k10-30      ", fmt(r["rho_slope"]))
        for name in ("large", "mid", "small"):
            print("   coherence %-6s all %s  top %s  bottom %s" % (name, fmt(r[f"coh_{name}_all"]), fmt(r[f"coh_{name}_top"]), fmt(r[f"coh_{name}_bottom"])))
        print("   pressure share (T units)  ", "%.2f [%.2f, %.2f]" % r["pressure_share"])
        print("   parts equal at k ≈        ", "%.0f [%.0f, %.0f]" % r["crossover_k"])
        print("   pressure-part slope k10-30", fmt(r["pressure_slope"]))
    np.save(f"{CACHE}/pressure_share.npy", out, allow_pickle=True)


if __name__ == "__main__":
    main()
