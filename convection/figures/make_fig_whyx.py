"""Teaching figure: why the FFT is done along x only. A row repeats smoothly; a column jumps."""
from convection.config import CACHE, FIGS
import sys
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
INK, INK2, GRID, BG = "#17181c", "#5a5f6b", "#e4e5e8", "#f7f7f5"
ACC, COLD = "#8a1c5c", "#2350b8"
plt.rcParams.update({"font.size": 17, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.facecolor": BG, "figure.facecolor": BG, "savefig.facecolor": BG,
                     "axes.grid": True, "grid.color": GRID, "axes.spines.top": False, "axes.spines.right": False,
                     "lines.linewidth": 2.2, "legend.frameon": False})
d = np.load(f"{CACHE}/pass2_Ra1e9.npz")
T = d["T300"].astype(float); nr, nx = T.shape
row = T[nr // 2]; col = T[:, nx // 2][::-1]                       # column from z = 0 (bottom) to z = 1 (top)
fig = plt.figure(figsize=(17, 9.6), dpi=140)
gs = fig.add_gridspec(2, 2, width_ratios=[1.5, 1], hspace=0.45, wspace=0.22)
a = fig.add_subplot(gs[0, 0])
x = np.arange(2 * nx) * 4.0 / nx
a.plot(x, np.r_[row, row], color=ACC); a.axvline(4, color=INK2, ls="--", lw=1.4)
a.set_title("One row (along x), drawn twice: the end joins the start smoothly", loc="left", fontsize=17, color=INK)
a.set_xlabel("x  (0 → 4, then the same row again)"); a.set_ylabel("T − T_A")
b = fig.add_subplot(gs[1, 0])
z = np.arange(2 * nr) / nr
b.plot(z, np.r_[col, col], color=COLD); b.axvline(1, color=INK2, ls="--", lw=1.4)
j = col[-1] - col[0]
b.annotate(f"jump {abs(j):.3f}\n(the top plate is colder)", xy=(1, (col[-1] + col[0]) / 2), xytext=(1.42, col.min() + 0.04 * np.ptp(col)),
           fontsize=15, color=INK, arrowprops=dict(arrowstyle="->", color=INK2))
b.set_title("One column (along z), drawn twice: a jump at the join", loc="left", fontsize=17, color=INK)
b.set_xlabel("z  (0 → 1, then the same column again)"); b.set_ylabel("T − T_A")
c = fig.add_subplot(gs[:, 1])
Ers, Ecs = [], []
for f in (5, 300):
    F = d[f"T{f}"].astype(float)
    Ers.append((np.abs(np.fft.rfft(F - F.mean(1, keepdims=True), axis=1)) ** 2).mean(0)[1:] / nx)
    C = F.T
    Ecs.append((np.abs(np.fft.rfft(C - C.mean(1, keepdims=True), axis=1)) ** 2).mean(0)[1:] / nr)
Er, Ec = np.mean(Ers, 0), np.mean(Ecs, 0)
kr = np.arange(1, len(Er) + 1) / 4.0; kc = np.arange(1, len(Ec) + 1) / 1.0   # waves per unit length
c.loglog(kr[:200], Er[:200] / Er[39], color=ACC, label="rows, FFT along x")
s = Ec[9] / Er[39] * 0 + 1
c.loglog(kc[:120], Ec[:120] / np.interp(10, kc, Ec), color=COLD, label="columns, FFT along z")
kk = np.array([8, 80]); c.loglog(kk, 3 * (kk / 10.0) ** -2, color=INK2, ls="--", lw=1.5); c.text(45, 3 * 4.5 ** -2 * 1.4, "−2: the jump", color=INK2, fontsize=15)
c.set_xlabel("waves per unit length"); c.set_ylabel("E, scaled at 10 waves per unit")
c.set_title("Spectra, Ra = 10⁹", loc="left", fontsize=17, color=INK); c.legend(loc="lower left", fontsize=15)
plt.savefig(f"{FIGS}/f12_why_x.png", bbox_inches="tight")
print("row end jump %.4f, column end jump %.4f" % (abs(row[-1] - row[0]), abs(j)))
