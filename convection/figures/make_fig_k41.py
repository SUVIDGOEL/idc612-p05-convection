"""Slide 8 inset: -5/3 from dimensional analysis (Kolmogorov, then Obukhov-Corrsin for temperature)."""
from convection.config import FIGS
import matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams.update({"mathtext.fontset": "cm", "font.family": "DejaVu Sans"})
INK, INK2, ACC, BG, CARD = "#17181c", "#3d414b", "#8a1c5c", "#f7f7f5", "#ffffff"
W, H = 6.08, 3.5
fig = plt.figure(figsize=(W, H), dpi=250); fig.patch.set_facecolor(BG)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
ax.add_patch(FancyBboxPatch((0.02, 0.03), W - 0.04, H - 0.06, boxstyle="round,pad=0,rounding_size=0.08", fc=CARD, ec="#d9dbe0", lw=0.8))
x = 0.16
ax.text(x, 3.18, "Where −5/3 comes from: only the units", fontsize=12.5, color=ACC, weight="bold", va="center")
ax.text(x, 2.80, r"Kolmogorov 1941: in the inertial range $E$ depends only on $\varepsilon$ and $k$", fontsize=10.5, color=INK2, va="center")
ax.text(x, 2.40, r"$E\ [\mathrm{m^3\,s^{-2}}] = C\,\varepsilon^{a}\,k^{b}$,     $\varepsilon\ [\mathrm{m^2\,s^{-3}}]$ energy flux,   $k\ [\mathrm{m^{-1}}]$", fontsize=11.5, color=INK, va="center")
ax.text(x, 1.98, r"seconds:  $-2=-3a\ \Rightarrow\ a=\frac{2}{3}$        metres:  $3=2a-b\ \Rightarrow\ b=-\frac{5}{3}$", fontsize=11.5, color=INK, va="center")
ax.plot([x, W - 0.16], [1.68, 1.68], color="#e4e5e8", lw=1)
ax.text(x, 1.40, r"Obukhov–Corrsin: add the temperature flux $\varepsilon_T\ [\mathrm{K^2\,s^{-1}}]$", fontsize=10.5, color=INK2, va="center")
ax.text(x, 1.00, r"$E_T\ [\mathrm{K^2\,m}] = C\,\varepsilon_T\,\varepsilon^{a}\,k^{b}$:    s: $0=-1-3a\Rightarrow a=-\frac{1}{3}$,   m: $1=2a-b\Rightarrow b=-\frac{5}{3}$", fontsize=11, color=INK, va="center")
ax.text(x, 0.45, r"$E_T(k)=C\,\varepsilon_T\,\varepsilon^{-1/3}\,k^{-5/3}$     →   slope $-5/3=-1.67$", fontsize=14, color=ACC, va="center")
fig.savefig(f"{FIGS}/f24_k41.png", facecolor=BG)
print("ok")
