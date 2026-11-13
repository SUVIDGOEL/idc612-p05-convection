"""Step 1 figure: the colour bar from the video, and the brightness -> temperature table built from it."""
from convection.config import CACHE, DATA, FIGS
import sys
import av, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from convection.s1_x_spectra.spectra_pass import build_lut
INK, INK2, GRID, BG, ACC = "#17181c", "#5a5f6b", "#e4e5e8", "#f7f7f5", "#8a1c5c"
plt.rcParams.update({"font.size": 18, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.facecolor": BG, "figure.facecolor": BG, "savefig.facecolor": BG,
                     "axes.grid": True, "grid.color": GRID, "axes.spines.top": False, "axes.spines.right": False,
                     "lines.linewidth": 2.6})
st = np.load(f"{CACHE}/tstats_Ra1e12.npz"); r0, r1, c0, c1 = np.load(f"{CACHE}/cbar_Ra1e12.npz")["cbar"]
c = av.open(f"{DATA}/2d_Ra1e12.mp4")
for i, fr in enumerate(c.decode(video=0)):
    if i == 300:
        rgb = fr.to_ndarray(format="rgb24"); break
c.close()
bar = rgb[r0 + 3:r1 - 3, c0:c1]
lut = build_lut(st["mean"], np.array([r0, r1, c0, c1]))         # brightness -> position (1 = top = 0, 0 = bottom = -0.1)
Tval = -0.1 + 0.1 * lut
strip = np.median(st["mean"][r0 + 3:r1 - 3, c0:c1], axis=1)
fig = plt.figure(figsize=(12, 7.4), dpi=150)
gs = fig.add_gridspec(1, 2, width_ratios=[0.22, 1], wspace=0.35)
a = fig.add_subplot(gs[0]); a.imshow(bar, aspect="auto", extent=[0, 1, -0.1, 0]); a.set_xticks([])
a.set_yticks([0, -0.05, -0.1]); a.set_yticklabels(["0", "−0.05", "−0.1"]); a.grid(False)
a.set_title("colour bar\n(from the video)", fontsize=16, color=INK)
a.set_ylabel("T − T_A")
b = fig.add_subplot(gs[1])
lo, hi = int(np.floor(strip.min())), int(np.ceil(strip.max()))
y = np.arange(lo, hi + 1)
b.step(y, Tval[y], where="mid", color=ACC)
b.plot([99, 99], [-0.105, Tval[99]], color=INK2, ls="--", lw=1.4); b.plot([0, 99], [Tval[99], Tval[99]], color=INK2, ls="--", lw=1.4)
b.annotate(("brightness 99  →  T − T_A = %.3f" % Tval[99]).replace("-", "−"), xy=(99, Tval[99]), xytext=(115, -0.075), fontsize=16, color=INK,
           arrowprops=dict(arrowstyle="->", color=INK2))
b.set_xlim(0, 255); b.set_ylim(-0.105, 0.004)
b.set_xlabel("brightness Y of a pixel (0 – 255)"); b.set_ylabel("temperature T − T_A")
b.set_title(f"The table: brightness only increases → one brightness, one temperature\n{hi - lo} levels, steps of ≈ {0.1 / (hi - lo):.4f}",
            loc="left", fontsize=16, color=INK)
plt.savefig(f"{FIGS}/f21_colourbar.png", bbox_inches="tight")
print(lo, hi, Tval[99])
