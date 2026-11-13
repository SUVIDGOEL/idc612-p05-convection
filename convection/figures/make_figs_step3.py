"""Teaching figures for step 3: why log-log, the regression, the local slope, the compensated spectrum."""
from convection.config import CACHE, FIGS
import sys
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
INK, INK2, GRID, BG = "#17181c", "#5a5f6b", "#e4e5e8", "#f7f7f5"
C = {"Ra1e9": "#2a78d6", "Ra1e12": "#eb6834", "Ra1e15": "#1baf7a"}
LAB = {"Ra1e9": "Ra = 10⁹", "Ra1e12": "Ra = 10¹²", "Ra1e15": "Ra = 10¹⁵"}
FLOOR = {"Ra1e9": 177, "Ra1e12": 873, "Ra1e15": 155}
BAND = "#8a1c5c"
plt.rcParams.update({"font.size": 17, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.facecolor": BG, "figure.facecolor": BG, "savefig.facecolor": BG,
                     "axes.grid": True, "grid.color": GRID, "axes.spines.top": False, "axes.spines.right": False,
                     "lines.linewidth": 2.4, "legend.frameon": False})
E = {t: np.load(f"{CACHE}/spec_{t}.npz")["clean"].astype(float).mean(0) * 0.01 for t in C}
OUT = f"{FIGS}/"

# ---- A: the same spectrum on linear and on log-log axes
t = "Ra1e12"; e = E[t]; k = np.arange(1, len(e) + 1.0)
fig, ax = plt.subplots(1, 2, figsize=(16, 5.8), dpi=140)
ax[0].plot(k[:200], e[:200], color=C[t]); ax[0].set_xlabel("k"); ax[0].set_ylabel("E(k)")
ax[0].set_title("Linear axes: a curve that hugs the axes", loc="left", fontsize=17, color=INK)
ax[1].loglog(k[:FLOOR[t]], e[:FLOOR[t]], color=C[t]); ax[1].axvspan(10, 30, color=BAND, alpha=0.08)
ax[1].text(10.5, e[0] * 1.2, "k = 10–30", color=BAND, fontsize=15)
ax[1].set_xlabel("k  (log scale)"); ax[1].set_ylabel("E(k)  (log scale)")
ax[1].set_title("Log–log axes: close to a straight line", loc="left", fontsize=17, color=INK)
plt.tight_layout(); plt.savefig(OUT + "f16_loglog.png"); plt.close()

# ---- B: the regression on the 21 points
m = (k >= 10) & (k <= 30); x = np.log10(k[m]); y = np.log10(e[m])
slope, icpt = np.polyfit(x, y, 1)
x2 = np.array([1.0, np.log10(30)]); y2 = np.log10([e[9], e[29]]); s2 = (y2[1] - y2[0]) / (x2[1] - x2[0])
fig, ax = plt.subplots(figsize=(12, 6.4), dpi=140)
ax.plot(x, y, "o", color=C[t], ms=9, label="the 21 points, k = 10 … 30")
xx = np.linspace(0.97, 1.51, 50)
ax.plot(xx, icpt + slope * xx, color=INK, lw=2.6, label=("least-squares line: slope %.2f" % slope).replace("-", "−"))
ax.plot(x2, y2, "--", color=INK2, lw=1.8, label=("line through 2 points only: slope %.2f" % s2).replace("-", "−"))
yb = icpt + slope * x.mean()
ax.plot(xx, yb + (-5 / 3) * (xx - x.mean()) - 0.12, ":", color=BAND, lw=2.2, label="theory −5/3 (shifted down to compare)")
ax.set_xlabel("x = log₁₀ k"); ax.set_ylabel("y = log₁₀ E(k)")
ax.set_title("Ra = 10¹²: the regression in log space", loc="left", fontsize=18, color=INK)
ax.legend(loc="lower left", fontsize=15)
plt.tight_layout(); plt.savefig(OUT + "f17_regression.png"); plt.close()

# ---- C: local slope of the three spectra (straight-line fit in a sliding window k/1.5 .. 1.5k, block-bootstrap band)
from convection.common.stats import per_frame_slopes, integral_time
def window_slopes(E, centres, w=1.5):
    k = np.arange(1, len(E) + 1.0); out = []
    for c in centres:
        m = (k >= c / w) & (k <= c * w)
        out.append(np.polyfit(np.log10(k[m]), np.log10(E[m]), 1)[0] if m.sum() >= 3 else np.nan)
    return np.array(out)
fig, ax = plt.subplots(figsize=(13, 6.6), dpi=140)
for t in C:
    S = np.load(f"{CACHE}/spec_{t}.npz")["clean"].astype(float)[:, :FLOOR[t]]
    n = S.shape[0]
    tau, _ = integral_time(per_frame_slopes(S, 10, 30)); L = max(1, int(np.ceil(4 * tau)))
    centres = np.unique(np.round(np.logspace(np.log10(6), np.log10(min(FLOOR[t] / 1.5, 350)), 28)))
    s0 = window_slopes(S.mean(0), centres)
    rng = np.random.default_rng(0); starts = np.arange(n - L + 1); nb = int(np.ceil(n / L)); Bs = []
    for _ in range(300):
        idx = (rng.choice(starts, nb)[:, None] + np.arange(L)).ravel()[:n]
        Bs.append(window_slopes(S[idx].mean(0), centres))
    lo, hi = np.percentile(np.array(Bs), [2.5, 97.5], axis=0)
    ax.fill_between(centres, lo, hi, color=C[t], alpha=0.18, lw=0)
    ax.semilogx(centres, s0, "o-", color=C[t], ms=5, label=LAB[t])
ax.axvspan(10, 30, color=BAND, alpha=0.08); ax.text(10.5, -0.75, "fit band", color=BAND, fontsize=15)
for v, nm in ((-5 / 3, "−5/3"), (-2, "−2")):
    ax.axhline(v, color=INK2, ls="--", lw=1.3); ax.text(390, v + 0.05, nm, fontsize=15, color=INK2, ha="right")
ax.set_ylim(-4.6, -0.6); ax.set_xlim(5.5, 400)
ax.set_xlabel("wavenumber k"); ax.set_ylabel("local slope near k")
ax.set_title("Local slope (fit from k/1.5 to 1.5k), with 95% bands", loc="left", fontsize=17, color=INK)
ax.legend(loc="lower left", fontsize=15)
plt.tight_layout(); plt.savefig(OUT + "f18_localslope.png"); plt.close()

# ---- D: compensated spectra
fig, ax = plt.subplots(figsize=(13, 6.4), dpi=140)
for t in C:
    e = E[t][:FLOOR[t]]; k = np.arange(1, len(e) + 1.0)
    cmp_ = e * k ** (5 / 3); cmp_ /= np.exp(np.log(cmp_[9:30]).mean())
    ax.loglog(k, cmp_, color=C[t], label=LAB[t])
ax.axhline(1, color=INK2, ls="--", lw=1.3)
ax.axvspan(10, 30, color=BAND, alpha=0.08)
ax.set_xlim(1, 900); ax.set_ylim(0.05, 6)
ax.set_xlabel("wavenumber k"); ax.set_ylabel("k^(5/3) · E(k), scaled to 1 in the band")
ax.set_title("Compensated spectra: flat = exactly −5/3; falling = steeper than −5/3", loc="left", fontsize=17, color=INK)
ax.legend(loc="lower left", fontsize=15)
plt.tight_layout(); plt.savefig(OUT + "f19_compensated.png"); plt.close()
print("slope %.3f intercept %.3f two-point %.3f" % (slope, icpt, s2))
