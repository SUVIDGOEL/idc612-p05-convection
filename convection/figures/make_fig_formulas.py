"""Typeset formula panel for slide 9: the ideal gas, the background, the entropy, and the split of ln T."""
from convection.config import FIGS
import matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"mathtext.fontset": "cm", "font.family": "DejaVu Sans"})
INK, INK2, ACC, BG, CARD = "#17181c", "#3d414b", "#8a1c5c", "#f7f7f5", "#ffffff"
W, H = 16.6, 2.9
fig = plt.figure(figsize=(W, H), dpi=150); fig.patch.set_facecolor(BG)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
from matplotlib.patches import FancyBboxPatch
for x0, w in ((0.05, 6.55), (6.85, 9.7)):
    ax.add_patch(FancyBboxPatch((x0, 0.08), w, H - 0.16, boxstyle="round,pad=0,rounding_size=0.12", fc=CARD, ec="#d9dbe0", lw=1))
# left card: the gas
ax.text(0.3, 2.5, "Ideal gas  (all values ÷ bottom values)", fontsize=15, color=ACC, weight="bold", va="center")
L = [(r"$p=\rho\,T$", r"$\theta,\ \varrho$ = the two painted panels"),
     (r"$T=T_A(z)+\theta,\qquad \rho=\rho_A(z)+\varrho$", ""),
     (r"$T_A=1-Dz,\qquad \rho_A=(1-Dz)^{1/(\gamma-1)}$", ""),
     (r"$s/c_v=\ln T-(\gamma-1)\,\ln\rho$", "entropy")]
for i, (eq, note) in enumerate(L):
    y = 1.95 - 0.52 * i
    ax.text(0.35, y, eq, fontsize=21, color=INK, va="center")
    if note:
        ax.text(6.4, y, note, fontsize=13, color=INK2, va="center", ha="right", style="italic")
# right card: the split
ax.text(7.1, 2.5, "The split of the temperature", fontsize=15, color=ACC, weight="bold", va="center")
r = fig.canvas.get_renderer()
def put(x, y, s, size=30, color=INK):
    t = ax.text(x, y, s, fontsize=size, color=color, va="center")
    bb = t.get_window_extent(renderer=r).transformed(ax.transData.inverted())
    return bb
y0 = 1.62
b1 = put(7.25, y0, r"$\ln T\ =$")
b2 = put(b1.x1 + 0.25, y0, r"$\dfrac{1}{\gamma}\,\dfrac{s}{c_v}$")
b3 = put(b2.x1 + 0.3, y0, r"$+$")
b4 = put(b3.x1 + 0.3, y0, r"$\dfrac{\gamma-1}{\gamma}\,\ln p$")
for b, lab, col in ((b2, "entropy part\n(thermal plumes)", "#c2410c"), (b4, "pressure part\n(vortex cores)", INK)):
    yb = b.y0 - 0.1
    ax.plot([b.x0, b.x0, b.x1, b.x1], [yb + 0.08, yb, yb, yb + 0.08], color=col, lw=1.6)
    ax.text((b.x0 + b.x1) / 2, yb - 0.08, lab, fontsize=14, color=col, ha="center", va="top", weight="bold")
ax.text(b4.x1 + 0.55, 2.0, "plume: cold and dense", fontsize=14, color=INK2, va="center")
ax.text(b4.x1 + 0.55, 1.62, "vortex core: cold and light", fontsize=14, color=INK2, va="center")
ax.text(b4.x1 + 0.55, 1.12, r"no heat exchange:  $\dfrac{T'}{T}=(\gamma-1)\,\dfrac{\rho'}{\rho}$", fontsize=16, color=INK2, va="center")
ax.text(b4.x1 + 0.55, 0.55, r"$\gamma = 1.3,\ \ D = 0.5$", fontsize=16, color=INK2, va="center")
fig.savefig(f"{FIGS}/f22_formulas.png", facecolor=BG)
print("ok", b4.x1)
