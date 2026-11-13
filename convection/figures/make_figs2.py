"""Figures for the new results (density panel, entropy/pressure split, zoom check, dissipation range).
Same style as make_slide_figs.py."""
from convection.config import CACHE, FIGS
import sys
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from convection.common.stats import slope
from convection.s2_density.adiabat_fit import fit, beta_model

INK, INK2, GRID, BG = "#17181c", "#5a5f6b", "#e4e5e8", "#f7f7f5"
C = {"Ra1e9": "#2a78d6", "Ra1e12": "#eb6834", "Ra1e15": "#1baf7a"}
LAB = {"Ra1e9": "Ra = 10⁹", "Ra1e12": "Ra = 10¹²", "Ra1e15": "Ra = 10¹⁵"}
plt.rcParams.update({"font.size": 19, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.facecolor": BG, "figure.facecolor": BG, "savefig.facecolor": BG,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1, "axes.spines.top": False,
                     "axes.spines.right": False, "lines.linewidth": 2.6, "legend.frameon": False})
OUT = f"{FIGS}/"
TAGS = ("Ra1e9", "Ra1e12", "Ra1e15")
P2 = {t: np.load(f"{CACHE}/pass2_{t}.npz") for t in TAGS}
P3 = {t: np.load(f"{CACHE}/pass3_{t}.npz") for t in TAGS}


def logbin(k, E, nb=36):
    e = np.unique(np.round(np.logspace(0, np.log10(len(k)), nb)).astype(int)) - 1
    kc = np.array([k[a:b].mean() for a, b in zip(e[:-1], e[1:]) if b > a])
    Ec = np.array([E[..., a:b].mean(-1) for a, b in zip(e[:-1], e[1:]) if b > a])
    return kc, np.moveaxis(Ec, 0, -1)


# ---------- F7: T-rho coherence map, k vs z ----------
fig, ax = plt.subplots(1, 3, figsize=(16, 5.6), dpi=150, sharey=True)
norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
for a, t in zip(ax, TAGS):
    d = P2[t]
    ET, ER, CO = (d[k].astype(float).mean(0) for k in ("ET", "ER", "CO"))
    k = np.arange(1, ET.shape[1] + 1.0)
    kc, Tb = logbin(k, ET); _, Rb = logbin(k, ER); _, Cb = logbin(k, CO)
    coh = Cb / np.sqrt(Tb * Rb)
    m = kc <= 400
    nb = coh.shape[0]; ze = np.linspace(1, 0, nb + 1)
    ke = np.sqrt(kc[m][:-1] * kc[m][1:]); ke = np.r_[kc[m][0] ** 2 / ke[0], ke, kc[m][-1] ** 2 / ke[-1]]
    pc = a.pcolormesh(ke, ze, coh[:, m], cmap="RdBu_r", norm=norm, shading="flat")
    a.set_xscale("log"); a.set_xlim(1, 400); a.set_ylim(0, 1); a.grid(False)
    a.set_title(LAB[t], loc="left", fontsize=20, color=INK); a.set_xlabel("wavenumber k")
ax[0].set_ylabel("height z")
cb = fig.colorbar(pc, ax=ax, fraction=0.025, pad=0.02)
cb.set_label("coherence of T and ρ")
cb.set_ticks([-1, -0.5, 0, 0.5, 1])
fig.text(0.01, 0.005, "red: cold fluid is light (low pressure, vortex cores)     blue: cold fluid is dense (thermal plumes)",
         fontsize=16, color=INK2)
plt.subplots_adjust(left=0.06, right=0.88, bottom=0.2, top=0.9, wspace=0.08)
plt.savefig(OUT + "f7_coherence.png"); plt.close()

# ---------- F8: adiabatic ratio -> gamma and D ----------
fig, ax = plt.subplots(figsize=(12, 6.6), dpi=150)
zz = np.linspace(0, 1, 200)
ax.plot(beta_model(zz, 0.3, 0.5), zz, color=INK2, ls="--", lw=2, label="theory with γ = 1.3, D = 0.5 (printed on the frame)")
fits = {}
for t in TAGS:
    z, b, p, ci = fit(t, nboot=300)
    fits[t] = (p, ci)
    lab = f"{LAB[t]}" + (f"   fit: γ−1 = {p[0]:.2f},  D = {p[1]:.2f}" if t != "Ra1e9" else "   (does not follow: mixed)")
    ax.plot(b, z, "o", color=C[t], ms=11, label=lab)
ax.set_xlabel("T′ / ρ′ at large scales (k = 1–10)"); ax.set_ylabel("height z"); ax.set_ylim(0, 1)
ax.set_xlim(-1.1, 1.4); ax.axvline(0, color=INK2, lw=1)
ax.legend(loc="lower left", fontsize=15)
plt.tight_layout(); plt.savefig(OUT + "f8_adiabat.png"); plt.close()

# ---------- F9: entropy part vs pressure part ----------
fig, ax = plt.subplots(1, 3, figsize=(17, 6.0), dpi=150, sharey=True)
res2 = np.load(f"{CACHE}/results2.npy", allow_pickle=True).item()
for a, t in zip(ax, TAGS):
    d = P3[t]
    Ee = d["E_ent"].astype(float).mean((0, 1)); Ep = d["E_prs"].astype(float).mean((0, 1))
    k = np.arange(1, len(Ee) + 1.0); m = k <= 500
    nrm = Ee[:30].sum() + Ep[:30].sum()
    a.loglog(k[m], Ep[m] / nrm, color=INK, label="pressure part")
    a.loglog(k[m], Ee[m] / nrm, color=C[t], label="entropy part")
    kr = np.array([8, 80]); a.loglog(kr, 0.3 * (Ep[9] / nrm) * (kr / 10.0) ** (-7 / 3), color=INK2, ls="--", lw=1.6)
    a.text(60, 0.3 * (Ep[9] / nrm) * (6.0) ** (-7 / 3), "−7/3", color=INK2, fontsize=16)
    r = res2[t]
    fe = r['share_ent'] / (r['share_ent'] + r['share_prs'])
    a.set_title(f"{LAB[t]}\npressure part {100 * (1 - fe):.0f}%  ·  entropy part {100 * fe:.0f}%",
                loc="left", fontsize=18, color=INK)
    if r["kc"] and r["kc"] > 1:
        a.axvline(r["kc"], color=C[t], ls=":", lw=1.6)
    a.set_xlabel("wavenumber k"); a.set_xlim(1, 500)
ax[0].set_ylabel("E(k) of each part of ln T")
ax[0].legend(loc="lower left", fontsize=16)
plt.tight_layout(); plt.savefig(OUT + "f9_parts.png"); plt.close()

# ---------- F10: zoom panel vs main panel ----------
fig, ax = plt.subplots(1, 2, figsize=(15, 6.0), dpi=150)
for a, t in zip(ax, ("Ra1e12", "Ra1e15")):
    d = P2[t]
    top = d["topE"].astype(float).mean(0); zo = d["zoomE"].astype(float).mean(0)
    km = np.arange(1, len(top) + 1.0); kz = 16.0 * np.arange(1, len(zo) + 1)
    s = top[63] / zo[3]
    a.loglog(km, top / top[63], color=C[t], label="main panel, z > 0.75")
    a.loglog(kz, zo * s / top[63], color=INK, ls="--", label="zoomed panel, 4× finer pixels")
    a.set_xlim(8, 4000); a.set_ylim(1e-4, 60); a.set_xlabel("wavenumber k (main-panel units)")
    a.set_title(LAB[t], loc="left", fontsize=20, color=INK)
ax[0].set_ylabel("E(k), scaled at k = 64"); ax[0].legend(loc="lower left", fontsize=16)
plt.tight_layout(); plt.savefig(OUT + "f10_zoom.png"); plt.close()

# ---------- F11: local slope, dissipation range moves to larger k ----------
fig, ax = plt.subplots(figsize=(12, 6.6), dpi=150)
kd = {"Ra1e9": 58, "Ra1e12": 250, "Ra1e15": 630}
for t in TAGS:
    d = P2[t]
    if t == "Ra1e9":
        E = d["topE"].astype(float).mean(0); k = np.arange(1, len(E) + 1.0); E, k = E[:125], k[:125]
    else:
        E = d["zoomE"].astype(float).mean(0); k = 16.0 * np.arange(1, len(E) + 1); keep = k <= {"Ra1e12": 720, "Ra1e15": 1100}[t]; E, k = E[keep], k[keep]
    kc, Ec = logbin(k, E, nb=30)
    g = np.gradient(np.log(Ec), np.log(kc))
    ok = kc >= 8
    ax.semilogx(kc[ok], g[ok], "-", color=C[t], label=LAB[t] + ("  (main panel)" if t == "Ra1e9" else "  (zoomed panel)"))
    ax.annotate("", xy=(kd[t], -3), xytext=(kd[t], -4.9), arrowprops=dict(arrowstyle="->", color=C[t], lw=2))
for v, nm in ((-5 / 3, "−5/3"), (-3, "−3")):
    ax.axhline(v, color=INK2, ls="--", lw=1.2); ax.text(1250, v + 0.12, nm, va="bottom", fontsize=16, color=INK2)
ax.set_ylim(-5, 0); ax.set_xlim(8, 2000)
ax.set_xlabel("wavenumber k (main-panel units), top layer z > 0.75"); ax.set_ylabel("local slope  d log E / d log k")
ax.legend(loc="lower left", fontsize=16)
plt.tight_layout(); plt.savefig(OUT + "f11_dissipation.png"); plt.close()

np.save(OUT + "results2_figs.npy", {"fits": fits, "kd": kd}, allow_pickle=True)
for t, (p, ci) in fits.items():
    print(t, "A=%.3f [%.3f,%.3f] D=%.3f [%.3f,%.3f]" % (p[0], ci[0, 0], ci[1, 0], p[1], ci[0, 1], ci[1, 1]))
