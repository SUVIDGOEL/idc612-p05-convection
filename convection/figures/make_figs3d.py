"""Figures for the 3D movie: straightened faces, and spectra compared with the 2D movies."""
from convection.config import CACHE, FIGS
import sys
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
INK, INK2, GRID, BG = "#17181c", "#5a5f6b", "#e4e5e8", "#f7f7f5"
C3, C9 = "#8a1c5c", "#2a78d6"
plt.rcParams.update({"font.size": 17, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.facecolor": BG, "figure.facecolor": BG, "savefig.facecolor": BG,
                     "axes.grid": True, "grid.color": GRID, "axes.spines.top": False, "axes.spines.right": False,
                     "lines.linewidth": 2.4, "legend.frameon": False})
d = np.load(f"{CACHE}/spec3d.npz")
# ---- faces
fig = plt.figure(figsize=(16, 9.2), dpi=140)
gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.36], hspace=0.28, wspace=0.06)
for j, (key, title) in enumerate((("lid_500", "Lid, near the cold top plate (z ≈ 1)"), ("floor_500", "Floor, near the hot bottom plate (z ≈ 0)"))):
    a = fig.add_subplot(gs[0, j]); a.imshow(d[key], cmap="inferno", vmin=-0.1, vmax=0, interpolation="nearest")
    a.set_title(title, loc="left", fontsize=17, color=INK); a.set_xticks([]); a.set_yticks([]); a.grid(False)
for j, (key, title) in enumerate((("wall_xz_500", "Side wall (x–z): plumes fall from the top"), ("wall_yz_500", "Side wall (y–z)"))):
    a = fig.add_subplot(gs[1, j]); a.imshow(d[key], cmap="inferno", vmin=-0.1, vmax=0, interpolation="nearest", aspect="auto")
    a.set_title(title, loc="left", fontsize=17, color=INK); a.set_xticks([]); a.set_yticks([]); a.grid(False)
plt.savefig(f"{FIGS}/f13_3d_faces.png", bbox_inches="tight"); plt.close()

# ---- spectra and local slopes
walls = ["wall_xz", "wall_yz", "wall_back1", "wall_back2"]
Ew = np.stack([d[w].astype(float) for w in walls]).mean((0, 1)); kw = np.arange(1, len(Ew) + 1.0)
A = d["lid"].astype(float).mean(0); n = (len(A) - 2) // 3; El = A[:n]
A = d["floor"].astype(float).mean(0); Ef = A[:n]; ks = np.arange(1, n + 1.0)
E9 = np.load(f"{CACHE}/spec_Ra1e9.npz")["clean"].astype(float).mean(0); k9 = np.arange(1, len(E9) + 1.0)
def logbin(k, E, nb=30):
    e = np.unique(np.round(np.logspace(0, np.log10(len(k)), nb)).astype(int)) - 1
    kc = np.array([k[a:b].mean() for a, b in zip(e[:-1], e[1:]) if b > a]); Ec = np.array([E[a:b].mean() for a, b in zip(e[:-1], e[1:]) if b > a])
    return kc, Ec, np.gradient(np.log(Ec), np.log(kc))
fig, ax = plt.subplots(1, 2, figsize=(16, 6.4), dpi=140)
m = kw <= 120; ax[0].loglog(kw[m], Ew[m] / Ew[4], color=C3, label="3D, Ra = 10⁸, side walls (along x or y)")
m = k9 <= 160; ax[0].loglog(k9[m], E9[m] / E9[4], color=C9, label="2D, Ra = 10⁹, panel (along x)")
m = ks <= 120
ax[0].loglog(ks[m], El[m] / El[4], color=INK, ls="--", lw=1.8, label="3D lid, full 2D FFT")
ax[0].loglog(ks[m], Ef[m] / Ef[4], color=INK2, ls=":", lw=2.2, label="3D floor, full 2D FFT")
ax[0].axvspan(10, 30, color="#8a1c5c", alpha=0.07)
ax[0].set_xlabel("wavenumber k"); ax[0].set_ylabel("E(k), scaled at k = 5"); ax[0].legend(loc="lower left", fontsize=14)
ax[0].set_title("Spectra", loc="left", fontsize=18, color=INK)
for (k, E, col, lab, kmax, kd) in ((kw, Ew, C3, "3D, Ra = 10⁸ (walls)", 70, 27), (k9, E9, C9, "2D, Ra = 10⁹", 160, 71)):
    kc, Ec, g = logbin(k[:kmax], E[:kmax])
    ok = kc >= 2; ax[1].semilogx(kc[ok], g[ok], color=col, label=lab)
    ax[1].annotate("", xy=(kd, -3), xytext=(kd, -4.8), arrowprops=dict(arrowstyle="->", color=col, lw=2))
for v, nm in ((-5 / 3, "−5/3"), (-3, "−3")):
    ax[1].axhline(v, color=INK2, ls="--", lw=1.2); ax[1].text(100, v + 0.1, nm, fontsize=15, color=INK2)
ax[1].set_ylim(-5, 0.5); ax[1].set_xlabel("wavenumber k"); ax[1].set_ylabel("local slope")
ax[1].set_title("Local slope; arrows: the steep fall starts", loc="left", fontsize=16, color=INK)
ax[1].legend(loc="lower left", fontsize=14)
plt.tight_layout(); plt.savefig(f"{FIGS}/f14_3d_spectra.png"); plt.close()
print("ok")
