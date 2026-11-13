"""Hydrodynamic entropy (Verma & Chatterjee 2022) of the temperature x-spectra.

p_k = E(k) / sum E,   S = -sum p_k log2 p_k,   over k = 1..KMAX (same KMAX for every movie,
below every noise floor, so S is comparable). S is computed for each frame, then averaged.
S_max = log2(KMAX) is white noise (flat spectrum). A power law k^-a gives a known S(a).
"""
from convection.config import CACHE
import sys
import numpy as np
from convection.common.fits_and_errors import integral_time

KMAX = 150
TAGS = ("Ra1e9", "Ra1e12", "Ra1e15")

def S_of(E):
    p = E / E.sum(-1, keepdims=True)
    return -(p * np.log2(p)).sum(-1)

def S_powerlaw(a, kmax=KMAX):
    k = np.arange(1, kmax + 1.0)
    return S_of(k ** (-a))

if __name__ == "__main__":
    print("S_max = log2(%d) = %.3f bits" % (KMAX, np.log2(KMAX)))
    print("power law reference: " + "  ".join("a=%.2f:S=%.2f" % (a, S_powerlaw(a)) for a in (1.4, 5/3, 2, 2.4, 3)))
    res = {}
    for t in TAGS:
        d = np.load(f"{CACHE}/x_spectra_{t}.npz")
        St = S_of(d["clean"][:, :KMAX].astype(np.float64))          # one value per frame
        tau, _ = integral_time(St)
        neff = len(St) / (2 * tau)
        Sb = S_of(d["bands"][:, :, :KMAX].astype(np.float64))      # frames x bands
        res[t] = dict(St=St, tau=tau, Sb=Sb)
        print("%-7s S = %.3f +- %.3f bits (std %.3f, tau %.0f fr, n_eff %.0f)  time-avg-spectrum S = %.3f"
              % (t, St.mean(), St.std() / np.sqrt(neff), St.std(), tau, neff,
                 S_of(d["clean"][:, :KMAX].astype(np.float64).mean(0))))
        print("        by height (top->bottom): " + " ".join("%.2f" % v for v in Sb.mean(0)))
    np.save(f"{CACHE}/spectral_entropy.npy", res, allow_pickle=True)
