"""Result 2 figure without the Bolgiano-Obukhov line: slope vs Ra with block-bootstrap 95% intervals."""
from convection.config import CACHE, FIGS
import sys
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from convection.common.stats import slope, per_frame_slopes, integral_time, block_bootstrap_slope
INK, INK2, GRID, BG = "#17181c", "#5a5f6b", "#e4e5e8", "#f7f7f5"
plt.rcParams.update({"font.size": 19, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.facecolor": BG, "figure.facecolor": BG, "savefig.facecolor": BG,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1, "axes.spines.top": False,
                     "axes.spines.right": False, "lines.linewidth": 2.6, "legend.frameon": False})
TAGS = ("Ra1e9", "Ra1e12", "Ra1e15")
D = {t: np.load(f"{CACHE}/spec_{t}.npz") for t in TAGS}
res = {}
for t in TAGS:
    for v in ("clean", "lowz"):
        S = D[t][v].astype(np.float64)
        tv, _ = integral_time(per_frame_slopes(S, 10, 30))
        bs = block_bootstrap_slope(S, 10, 30, max(1, int(np.ceil(4 * tv))), nboot=2000)
        res[(t, v)] = (slope(S.mean(0), 10, 30), *np.percentile(bs, [2.5, 97.5]))
x = np.array([0, 1, 2])
fig, ax = plt.subplots(figsize=(12, 6.6), dpi=150)
for v, col, off, lab in (("clean", INK, -0.08, "all heights"), ("lowz", "#8a1c5c", 0.08, "top layer removed (z < 0.72)")):
    s = np.array([res[(t, v)][0] for t in TAGS]); lo = np.array([res[(t, v)][1] for t in TAGS]); hi = np.array([res[(t, v)][2] for t in TAGS])
    ax.errorbar(x + off, s, yerr=[s - lo, hi - s], fmt="o", color=col, capsize=7, ms=10, lw=2.4, label=lab)
for val, name in ((-5 / 3, "−5/3  Obukhov–Corrsin"), (-2, "−2  sharp fronts")):
    ax.hlines(val, -0.35, 2.25, color=INK2, ls="--", lw=1.3); ax.text(2.32, val, name, va="center", fontsize=16, color=INK2)
ax.set_xticks(x); ax.set_xticklabels(["10⁹", "10¹²", "10¹⁵"]); ax.set_xlim(-0.35, 3.45)
ax.set_xlabel("Rayleigh number"); ax.set_ylabel("slope, k = 10–30"); ax.legend(loc="lower left")
plt.tight_layout(); plt.savefig(f"{FIGS}/f5_exponent_v2.png"); plt.close()
for k_, v in res.items(): print(k_, "%+.2f [%+.2f, %+.2f]" % v)
