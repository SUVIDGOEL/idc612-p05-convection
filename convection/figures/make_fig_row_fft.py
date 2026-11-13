"""Teaching figure for step 2: one random row of T along x, and its FFT power spectrum."""
from convection.config import CACHE, FIGS
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
INK, INK2, GRID, BG = "#17181c", "#5a5f6b", "#e4e5e8", "#f7f7f5"
ACC = "#eb6834"
plt.rcParams.update({"font.size": 17, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.facecolor": BG, "figure.facecolor": BG, "savefig.facecolor": BG,
                     "axes.grid": True, "grid.color": GRID, "axes.spines.top": False, "axes.spines.right": False,
                     "lines.linewidth": 1.8, "legend.frameon": False})
d = np.load(f"{CACHE}/pass2_Ra1e12.npz")
T = d["T300"].astype(float)                       # frame 300, cleaned field, row 0 = top (z = 1)
nr, nx = T.shape
rng = np.random.default_rng(7)
r = int(rng.integers(nr // 5, 4 * nr // 5))      # a random row away from the plates
z = 1 - (r + 2.5) / (nr + 4)
row = T[r]
x = np.arange(nx) * 4.0 / nx
P = np.abs(np.fft.rfft(row - row.mean())) ** 2 / nx
k = np.arange(len(P))
Eavg = np.load(f"{CACHE}/spec_Ra1e12.npz")["clean"].astype(float).mean(0)   # all rows, all frames
ka = np.arange(1, len(Eavg) + 1)

fig, ax = plt.subplots(2, 1, figsize=(15, 10.5), dpi=140, gridspec_kw={"hspace": 0.62})
a = ax[0]
a.plot(x, row, color=ACC, lw=1.4)
a.axhline(row.mean(), color=INK2, ls="--", lw=1.4)
a.text(4.02, row.mean(), " mean of the row\n (removed before the FFT)", va="center", fontsize=14, color=INK2)
a.set_xlim(0, 4); a.set_xlabel("x  (0 to 4, across the box)"); a.set_ylabel("T − T_A")
a.set_title(f"One row: Ra = 10¹², frame 300, height z = {z:.2f}, {nx} pixels", loc="left", fontsize=18, color=INK)
b = ax[1]
b.loglog(k[1:], P[1:], color=ACC, lw=1.0, alpha=0.9, label="this one row")
b.loglog(ka, Eavg * (P[1:31].sum() / Eavg[:30].sum()), color=INK, lw=2.4, label="average of all 525 525 rows = E(k)")
b.set_xlim(1, 1100); b.set_xlabel("wavenumber k  (waves across the box)"); b.set_ylabel("power")
b.set_title("Its FFT: the power of each wave size", loc="left", fontsize=18, color=INK)
sec = b.secondary_xaxis("top", functions=(lambda kk: 4.0 / np.maximum(kk, 1e-9), lambda w: 4.0 / np.maximum(w, 1e-9)))
sec.set_xlabel("wavelength = 4 / k", fontsize=15, color=INK2); sec.tick_params(colors=INK2)
sec.set_xticks([4, 1, 0.4, 0.1, 0.04, 0.01, 0.004]); sec.set_xticklabels(["4", "1", "0.4", "0.1", "0.04", "0.01", "0.004"])
b.legend(loc="lower left", fontsize=15)
plt.savefig(f"{FIGS}/f15_row_fft.png", bbox_inches="tight")
print("row", r, "z = %.3f" % z, "mean %.4f std %.4f" % (row.mean(), row.std()))
