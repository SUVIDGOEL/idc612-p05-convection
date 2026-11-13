"""Ra = 1e15: the temperature spectrum as the sum of a pressure part and an entropy part, and the break."""
from convection.config import CACHE, FIGS
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
INK, INK2, GRID, BG = "#17181c", "#5a5f6b", "#e4e5e8", "#f7f7f5"
GREEN, PRS, ENT = "#1baf7a", "#17181c", "#c2410c"
plt.rcParams.update({"font.size": 17, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.facecolor": BG, "figure.facecolor": BG, "savefig.facecolor": BG,
                     "axes.grid": True, "grid.color": GRID, "axes.spines.top": False, "axes.spines.right": False,
                     "lines.linewidth": 2.4, "legend.frameon": False})
d = np.load(f"{CACHE}/pass3_Ra1e15.npz")
B = {kk: d[kk].astype(float).mean(0) for kk in ("E_lnT", "E_prs", "E_ent")}          # bands x k
nb = B["E_lnT"].shape[0]; zc = 1 - (np.arange(nb) + .5) / nb
Tbar = (1 - 0.5 * zc) + np.interp(zc, d["z"][::-1], d["prof_theta"][::-1])         # mean temperature of each band
w = Tbar ** 2                                                                         # ln T -> T units: T' = Tbar (ln T)'
Et, Ep, Ee = ((w[:, None] * B[kk]).mean(0) for kk in ("E_lnT", "E_prs", "E_ent"))
k = np.arange(1, len(Et) + 1.0); m = k <= 300
kc = 61
def wslope(E, cs, w=1.5):
    out = []
    for c in cs:
        mm = (k >= c / w) & (k <= c * w); out.append(np.polyfit(np.log10(k[mm]), np.log10(E[mm]), 1)[0])
    return np.array(out)
fig, ax = plt.subplots(1, 2, figsize=(17, 6.4), dpi=140)
a = ax[0]
a.loglog(k[m], Et[m], color=GREEN, lw=3.2, label="temperature, the total")
a.loglog(k[m], Ep[m], color=PRS, lw=1.8, ls="--", label="pressure part (vortex cores)")
a.loglog(k[m], Ee[m], color=ENT, lw=1.8, ls=":", label="entropy part (thermal)")
a.axvline(kc, color=INK2, lw=1.2); a.text(kc * 1.06, Et[1] * 0.6, "parts equal\nk ≈ 60", fontsize=14, color=INK2)
a.axvspan(10, 30, color="#8a1c5c", alpha=0.07)
a.set_xlabel("wavenumber k"); a.set_ylabel("E(k)"); a.set_xlim(1, 300)
a.set_title("Ra = 10¹⁵: two parts of the temperature", loc="left", fontsize=17, color=INK)
a.legend(loc="lower left", fontsize=14)
b = ax[1]
cs = np.unique(np.round(np.logspace(np.log10(6), np.log10(190), 26)))
b.semilogx(cs, wslope(Et, cs), "o-", color=GREEN, lw=3, ms=5, label="total")
b.semilogx(cs, wslope(Ep, cs), "--", color=PRS, lw=1.8, label="pressure part")
b.semilogx(cs, wslope(Ee, cs), ":", color=ENT, lw=2.2, label="entropy part")
b.axvline(kc, color=INK2, lw=1.2); b.axvspan(10, 30, color="#8a1c5c", alpha=0.07)
for v, nm in ((-7 / 3, "−7/3"), (-5 / 3, "−5/3")):
    b.axhline(v, color=INK2, ls="-.", lw=1.0); b.text(185, v + 0.05, nm, fontsize=14, color=INK2, ha="right")
b.set_ylim(-3.2, -0.2); b.set_xlabel("wavenumber k"); b.set_ylabel("local slope")
b.set_title("Local slope: the total follows the larger part", loc="left", fontsize=17, color=INK)
b.legend(loc="lower right", fontsize=14)
plt.tight_layout(); plt.savefig(f"{FIGS}/f20_break15.png")
for c, t, p_, e in zip(cs, wslope(Et, cs), wslope(Ep, cs), wslope(Ee, cs)):
    print("k%3d total %+.2f  pressure %+.2f  entropy %+.2f" % (c, t, p_, e))
