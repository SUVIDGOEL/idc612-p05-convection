"""Slopes, correlation time and moving-block bootstrap on cached per-frame spectra."""
import numpy as np

def kgrid(n):
    return np.arange(1, n + 1)

def slope(E, lo, hi):
    k = kgrid(len(E)); m = (k >= lo) & (k <= hi)
    return np.polyfit(np.log10(k[m]), np.log10(E[m]), 1)[0]

def per_frame_slopes(S, lo, hi):
    """Slope of each frame's own spectrum -> a time series."""
    k = kgrid(S.shape[1]); m = (k >= lo) & (k <= hi)
    X = np.log10(k[m]); Y = np.log10(S[:, m])
    Xc = X - X.mean()
    return (Y - Y.mean(1, keepdims=True)) @ Xc / (Xc @ Xc)

def integral_time(x, maxlag=None):
    """Integral autocorrelation time (frames), summed until the ACF first crosses 0."""
    x = x - x.mean(); n = len(x)
    maxlag = maxlag or n // 3
    acf = np.correlate(x, x, "full")[n - 1:n - 1 + maxlag] / (x @ x)
    cut = np.argmax(acf <= 0) if np.any(acf <= 0) else maxlag
    return 0.5 + acf[1:cut].sum(), acf

def block_bootstrap_slope(S, lo, hi, L, nboot=2000, seed=0):
    """Moving-block bootstrap of the slope of the frame-averaged spectrum."""
    rng = np.random.default_rng(seed)
    n = S.shape[0]; nb = int(np.ceil(n / L))
    starts = np.arange(n - L + 1)
    out = np.empty(nboot)
    for b in range(nboot):
        idx = (rng.choice(starts, nb)[:, None] + np.arange(L)[None, :]).ravel()[:n]
        out[b] = slope(S[idx].mean(0), lo, hi)
    return out
