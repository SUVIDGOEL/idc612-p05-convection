"""Clean figures for the 5-slide presentation deck (deck_*.png), drawn from cached results only.

deck_spec1d.png   1D x-FFT spectra with -5/3 reference
deck_spec2d.png   2D FFT shell spectra (x FFT x z sine series)
deck_upper.png    mean-profile (upper-branch) sine modes: measured dots vs jump model, no fitted constant
deck_sound.png    Ra 1e15 frequency-wavenumber map with the ideal-gas sound line (no fit)
Run from P05_code/:  python -m convection.figures.deck_figures
"""
from convection.config import CACHE, FIGS
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator, FuncFormatter
from convection.common.fits_and_errors import slope
from convection.spectrum_2d import sound_waves as SW
EPS = 0.1
BG, INK, INK2, GRID = "#F7F6F2", "#14213D", "#4A5568", "#DEDCD4"
COL = {"Ra1e9": "#2a78d6", "Ra1e12": "#eb6834", "Ra1e15": "#1baf7a"}
LAB = {"Ra1e9": "Ra 10⁹", "Ra1e12": "Ra 10¹²", "Ra1e15": "Ra 10¹⁵"}
TAGS = ("Ra1e9", "Ra1e12", "Ra1e15")
OUT = f"{FIGS}/"
plt.rcParams.update({"font.size": 17, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
                     "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2.6,
                     "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "savefig.bbox": "tight"})
plain = FuncFormatter(lambda v, _: f"{v:g}")


def log_ticks(ax, xs=None, ys=None):
    if xs is not None:
        ax.xaxis.set_major_locator(FixedLocator(xs)); ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(plain)
    if ys is not None:
        ax.yaxis.set_major_locator(FixedLocator(ys)); ax.yaxis.set_minor_locator(NullLocator())


def spec1d():
    """1D x-FFT, mean over rows and frames (same data and fit as FINAL_RESULTS section 1)."""
    fig, a = plt.subplots(figsize=(8, 6.1))
    floor = {"Ra1e9": 177, "Ra1e12": 873, "Ra1e15": 155}
    a.axvspan(10, 30, color=INK, alpha=0.06, lw=0)
    for t in TAGS:
        E = np.load(f"{CACHE}/x_spectra_{t}.npz")["clean"].astype(np.float64).mean(0)
        k = np.arange(1, len(E) + 1); m = k <= min(floor[t], 400)
        En = E / np.exp(np.mean(np.log(E[(k >= 10) & (k <= 30)])))
        a.loglog(k[m], En[m], color=COL[t])
        print(f"1D {t}: slope k=10-30 {slope(E, 10, 30):+.3f}")
    kk = np.array([4, 60.0]); a.loglog(kk, 3 * (kk / 17.3) ** (-5 / 3), color=INK, lw=1.4, ls="--")
    a.text(5, 3 * (5 / 17.3) ** (-5 / 3) * 1.25, "−5/3", color=INK, fontsize=17)
    a.text(10.5, 1e-3 * 0.7, "fit range\nk = 10–30", color=INK2, fontsize=15, va="bottom")
    a.text(45, 7e-4, "10⁹: −1.52", color=COL["Ra1e9"], fontsize=17, ha="left")
    a.text(230, 3e-2, "10¹²: −1.59", color=COL["Ra1e12"], fontsize=17, ha="left")
    a.text(1.3, 150, "10¹⁵: −2.35", color=COL["Ra1e15"], fontsize=17, ha="left", va="bottom")
    a.set_xlim(1, 600); a.set_ylim(5e-4, 300)
    log_ticks(a, xs=[1, 3, 10, 30, 100, 300], ys=[1e-3, 1e-2, 1e-1, 1, 10, 100])
    a.set_xlabel("k  (waves across the box)"); a.set_ylabel("E(k), scaled at k = 10–30")
    fig.savefig(OUT + "deck_spec1d.png", dpi=150); plt.close(fig)


def spec2d():
    """2D FFT shells (x FFT x z sine series), fluctuations only (mean profile removed)."""
    fig, a = plt.subplots(figsize=(8, 6.1))
    a.axvspan(5, 15, color=INK, alpha=0.06, lw=0)
    sl = {"Ra1e9": "−1.27", "Ra1e12": "−1.62", "Ra1e15": "−2.27"}
    vmax = {"Ra1e9": 40, "Ra1e12": 40, "Ra1e15": 100}
    for t in TAGS:
        r = np.load(f"{CACHE}/analysis_2d_{t}.npy", allow_pickle=True).item()
        K, E = r["K"], r["shell"]
        k = (K >= 1) & (K <= vmax[t]) & np.isfinite(E) & (E > 0)
        En = E[k] / np.exp(np.mean(np.log(E[k][(K[k] >= 5) & (K[k] <= 15)])))
        Es = np.exp(np.convolve(np.log(En), [0.5, 0.5], "same")); Es[0] = En[0]   # 2-shell mean, as f31
        a.loglog(K[k], Es, color=COL[t])
        x0, y0 = K[k][-1], Es[-2]
        dy = {"Ra1e9": 0.33, "Ra1e12": 1.6, "Ra1e15": 1.0}[t]
        a.text(x0 * 1.08, y0 * dy, f"{LAB[t][3:]}: {sl[t]}", color=COL[t], fontsize=17, va="center")
    kk = np.array([2.5, 35.0]); a.loglog(kk, 4 * (kk / 8.7) ** (-5 / 3), color=INK, lw=1.4, ls="--")
    a.text(2.6, 4 * (2.6 / 8.7) ** (-5 / 3) * 1.3, "−5/3", color=INK, fontsize=17)
    a.text(5.3, 2e-3, "fit range\nK/π = 5–15", color=INK2, fontsize=15, va="bottom")
    a.set_xlim(1, 250); a.set_ylim(1e-3, 300)
    log_ticks(a, xs=[1, 3, 10, 30, 100], ys=[1e-3, 1e-2, 1e-1, 1, 10, 100])
    a.set_xlabel(r"$K/\pi$,   $K = \sqrt{k_x^2 + k_z^2}$"); a.set_ylabel("E(K), scaled at K/π = 5–15")
    fig.savefig(OUT + "deck_spec2d.png", dpi=150); plt.close(fig)


def upper():
    fig, ax = plt.subplots(figsize=(8, 6.1))
    for t in TAGS:
        r = np.load(f"{CACHE}/analysis_2d_{t}.npy", allow_pickle=True).item()
        Em = r["E_mean"]; n = np.arange(1, len(Em) + 1)
        k = (n <= 40) & (n % 2 == 0)
        ax.loglog(n[k], Em[k] / EPS ** 2, "o", ms=8, color=COL[t], mec=BG, mew=1)
        ax.loglog(n[k], r["E_mean_pred"][k] / EPS ** 2, color=COL[t], lw=1.6, alpha=0.8)
        rat = np.sqrt(Em[1:10:2] / r["E_mean_pred"][1:10:2]).mean()
        print(f"upper {t}: mean amplitude ratio, even n<=10: {rat:.3f}")
    ax.text(41, 1.0e-4 * 1.6, "10¹⁵", color=COL["Ra1e15"], fontsize=17, va="center")
    ax.text(41, 0.75e-4, "10¹²", color=COL["Ra1e12"], fontsize=17, va="center")
    ax.text(43, 8e-7, "10⁹:\nlayers\nresolved", color=COL["Ra1e9"], fontsize=17, va="center")
    ax.text(2.1, 1.2e-6, "dots: measured\nlines: jump model, no fitted constant", color=INK, fontsize=16)
    ax.set_xlim(1.8, 75); ax.set_ylim(2.5e-7, 0.12)
    log_ticks(ax, xs=[2, 4, 10, 20, 40], ys=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
    ax.set_xlabel(r"$n$   (vertical wavenumber $k_z = n\pi$, even $n$)"); ax.set_ylabel("mean-profile mode energy / ε²")
    fig.savefig(OUT + "deck_upper.png", dpi=150); plt.close(fig)


def sound():
    f, kb, S = SW.k_omega_spectrum("Ra1e15", "cT")
    Sn = S / np.maximum(S.sum(1, keepdims=True), 1e-300)
    fig, a = plt.subplots(figsize=(8, 6.1))
    a.pcolormesh(kb, f[1:], np.log10(Sn[:, 1:].T * f[1:, None] + 1e-30), vmin=-5, vmax=-0.5,
                 shading="nearest", cmap="Blues", rasterized=True)
    zg = np.linspace(0, 1, 401); cs = np.sqrt(0.3 * (1 - 0.5 * zg) / (0.5 * 0.1))
    c_eff = 1 / np.mean(1 / cs)
    a.plot(kb, c_eff * kb / 2, color=COL["Ra1e12"], lw=2.4, ls="--")
    a.text(52, 2.5, f"dashed: ω = cK\nc = {c_eff:.2f}, ideal gas,\nno fit\ndark = more power", color=INK, fontsize=17,
           bbox=dict(facecolor=BG, edgecolor="none", alpha=0.92, pad=6))
    a.set_ylim(0, 50); a.set_xlim(1, 90); a.grid(False)
    a.set_xlabel(r"$K/\pi$   (2D shell)"); a.set_ylabel("frequency f  (1 / time unit)")
    fig.savefig(OUT + "deck_sound.png", dpi=150); plt.close(fig)
    print(f"sound: c_eff = {c_eff:.3f}")


if __name__ == "__main__":
    spec1d(); spec2d(); upper(); sound()
