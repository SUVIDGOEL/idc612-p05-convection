"""Final summary figure: T-rho coherence and pressure share against Ra, 2D (three movies) and 3D (Ra = 1e8)."""
from convection.config import CACHE, FIGS
import sys
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from convection.common.stats import integral_time
INK, INK2, GRID, BG = "#17181c", "#5a5f6b", "#e4e5e8", "#f7f7f5"
C2, C3 = "#8a1c5c", "#2a78d6"
plt.rcParams.update({"font.size": 17, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.facecolor": BG, "figure.facecolor": BG, "savefig.facecolor": BG, "axes.grid": True, "grid.color": GRID,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})
fd = np.load(f"{CACHE}/final_density.npy", allow_pickle=True).item()
d3 = np.load(f"{CACHE}/rho3d.npz")
# 3D wall coherence with a block bootstrap over the stored per-frame band spectra
walls = ("wall_xz", "wall_yz", "wall_back1", "wall_back2")
Bf = np.stack([d3[w + "_bands"].astype(float) for w in walls]).mean(0)       # frames x 5 x 3 x k
def coh3(idx, lo, hi):
    X = Bf[idx][:, :, :, lo - 1:hi].sum((0, 1, 3)); return X[2] / np.sqrt(X[0] * X[1])
n = Bf.shape[0]; series = np.array([coh3([i], 10, 30) for i in range(n)]); tau, _ = integral_time(series); L = max(1, int(np.ceil(4 * tau)))
rng = np.random.default_rng(0); starts = np.arange(n - L + 1); nb = int(np.ceil(n / L))
c3 = {}
for nm, (lo, hi) in (("large", (1, 10)), ("small", (30, 100))):
    bs = [coh3((rng.choice(starts, nb)[:, None] + np.arange(L)).ravel()[:n], lo, hi) for _ in range(1000)]
    c3[nm] = (coh3(np.arange(n), lo, hi), *np.percentile(bs, [2.5, 97.5]))
x2 = np.array([9, 12, 15]); TAGS = ("Ra1e9", "Ra1e12", "Ra1e15")
fig, ax = plt.subplots(1, 2, figsize=(16, 6.2), dpi=140)
a = ax[0]
for nm, mk, lab in (("large", "o", "large sizes, k = 1–10"), ("small", "s", "small sizes, k = 30–100")):
    v = np.array([fd[t][f"coh_{nm}_all"] for t in TAGS])
    a.errorbar(x2, v[:, 0], yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], fmt=mk + "-", color=C2, ms=10, capsize=6, lw=2,
               mfc=C2 if nm == "large" else BG, label="2D, " + lab)
    v3 = c3[nm]
    a.errorbar([8], [v3[0]], yerr=[[v3[0] - v3[1]], [v3[2] - v3[0]]], fmt=mk, color=C3, ms=10, capsize=6, mfc=C3 if nm == "large" else BG,
               label="3D, " + lab)
a.axhline(0, color=INK2, lw=1)
a.text(8.1, 0.93, "cold = light (vortex cores)", fontsize=14, color=INK2); a.text(8.1, -0.97, "cold = dense (thermal plumes)", fontsize=14, color=INK2)
a.set_xticks([8, 9, 12, 15]); a.set_xticklabels(["10⁸\n(3D)", "10⁹", "10¹²", "10¹⁵"]); a.set_ylim(-1.05, 1.05)
a.set_xlabel("Rayleigh number"); a.set_ylabel("coherence of T and ρ"); a.legend(loc="upper left", bbox_to_anchor=(0.0, 0.60), fontsize=12, frameon=True, facecolor=BG, edgecolor=BG, framealpha=1)
a.set_title("Temperature–density coherence", loc="left", fontsize=18, color=INK)
b = ax[1]
v = np.array([fd[t]["pressure_share"] for t in TAGS]) * 100
b.errorbar(x2, v[:, 0], yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], fmt="o-", color=C2, ms=10, capsize=6, lw=2, label="2D movies")
p3 = np.load(f"{CACHE}/three_d_pressure.npy", allow_pickle=True).item()["share"]; s3 = np.array(p3) * 100
b.errorbar([8], [s3[0]], yerr=[[s3[0] - s3[1]], [s3[2] - s3[0]]], fmt="o", color=C3, ms=10, capsize=6, label="3D movie (walls)")
for xi, vi in zip(list(x2) + [8], list(v[:, 0]) + [s3[0]]):
    b.annotate("%.0f%%" % vi, (xi, vi), textcoords="offset points", xytext=(10, -18 if vi > 90 else 8), fontsize=15, color=INK)
b.set_xticks([8, 9, 12, 15]); b.set_xticklabels(["10⁸\n(3D)", "10⁹", "10¹²", "10¹⁵"]); b.set_ylim(0, 105)
b.set_xlabel("Rayleigh number"); b.set_ylabel("pressure share of T variance (%)"); b.legend(loc="upper left", fontsize=14)
b.set_title("How much of the temperature is vortex pressure", loc="left", fontsize=18, color=INK)
plt.tight_layout(); plt.savefig(f"{FIGS}/f26_final_summary.png")
print("3D coherence large %+.2f [%+.2f, %+.2f], small %+.2f [%+.2f, %+.2f]" % (*c3["large"], *c3["small"]))
np.save(f"{CACHE}/final_3d_coherence.npy", c3, allow_pickle=True)
