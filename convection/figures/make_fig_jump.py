"""Slide 3 inset: why a jump gives power ~ 1/k^2. Sketch of the product (row x wave) plus the formulas."""
from convection.config import FIGS
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"mathtext.fontset": "cm", "font.family": "DejaVu Sans"})
INK, INK2, ACC, BG, CARD = "#17181c", "#3d414b", "#8a1c5c", "#f7f7f5", "#ffffff"
POS, NEG = "#2a78d6", "#eb6834"
fig = plt.figure(figsize=(8.64, 4.5), dpi=250); fig.patch.set_facecolor(BG)
# sketch
a = fig.add_axes([0.035, 0.58, 0.93, 0.34]); a.set_facecolor(CARD)
x = np.linspace(0, 1, 2000); k = 6
f = np.where(x < 0.5, -0.6, 0.6); w = np.sin(2 * np.pi * k * x + 0.6)
prod = f * w
a.fill_between(x, 0, prod, where=prod >= 0, color=POS, alpha=0.35, lw=0)
a.fill_between(x, 0, prod, where=prod < 0, color=NEG, alpha=0.35, lw=0)
a.plot(x, w, color=INK2, lw=0.9, alpha=0.6, label="wave k")
a.plot(x, f, color=INK, lw=2.4, label="row with a jump")
a.axhline(0, color=INK2, lw=0.6)
a.set_xticks([]); a.set_yticks([]); [sp.set_visible(False) for sp in a.spines.values()]
a.text(0.02, 1.08, "row × wave: flat parts cancel (+ and −); at the jump one half-wave is left", fontsize=10.5, color=INK, transform=a.transAxes)
a.legend(loc="lower right", fontsize=9, frameon=False, ncol=2)
a.annotate("two + humps in a row:\none is left over", xy=(0.5, 0.42), xytext=(0.17, 0.78), fontsize=9.5, color=ACC,
           arrowprops=dict(arrowstyle="->", color=ACC, lw=1.2), ha="center", bbox=dict(fc="white", ec="none", pad=1.5))
# math
b = fig.add_axes([0, 0.03, 1, 0.5]); b.axis("off"); b.set_xlim(0, 1); b.set_ylim(0, 1)
b.text(0.035, 0.86, r"amplitude of wave $k$:   $a_k=\dfrac{1}{L}\int_0^L f(x)\,e^{-2\pi i k x/L}\,dx$", fontsize=14, color=INK, va="center")
b.text(0.035, 0.50, r"one half-wave (width $\dfrac{L}{2k}$) left at a jump of height $J$:   $|a_k|\approx\dfrac{J}{2\pi k}$", fontsize=14, color=INK, va="center")
b.text(0.035, 0.15, r"power:   $E(k)=|a_k|^2\approx\dfrac{J^2}{4\pi^2}\,k^{-2}$   →   a straight line with slope $-2$", fontsize=14, color=ACC, va="center")
fig.savefig(f"{FIGS}/f23_jump.png", facecolor=BG)
print("ok")
