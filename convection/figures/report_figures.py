"""Figures for the written report (figures/report/*.png), drawn from cached results only.

rep_mirror.png      one real temperature frame (Ra 1e12) and the mirror trick applied to one of its columns
rep_sound_map.png   large sound-ridge map (report section 3.5)
rep_sound_side.png  sound speed against height and sound share
rep_sound.png       sound waves: frequency-wavenumber map with ridge points, sound speed against height, sound share
rep_validation.png  synthetic movies with known slopes: recovered against true slope
deck_*.png          the slide figures again, on a white background
Run from P05_code/:  python -m convection.figures.report_figures
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.fft import dst
from convection.config import CACHE, FIGS, SYNTH
from convection.figures import deck_figures as DF
from convection.spectrum_2d import sound_waves as SW

OUT = f"{FIGS}/report/"
os.makedirs(OUT, exist_ok=True)
EPS, G, D = 0.1, 1.3, 0.5
INK, INK2, GRID, ACC = DF.INK, DF.INK2, DF.GRID, "#d1495b"
COL = DF.COL
plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white"})


def panel_label(ax, s):
    ax.text(-0.02, 1.04, s, transform=ax.transAxes, fontsize=19, fontweight="bold", color=INK, ha="right", va="bottom")


def mirror():
    d = np.load(f"{CACHE}/mirror_modes_2d_Ra1e12.npz")
    T = d["T300"].astype(float)                                # T - T_A, row 0 = bottom plate
    Nz, Nx = T.shape
    z = np.linspace(0, 1, Nz)
    th = T + EPS * z[:, None]                                   # theta' = 0 at both plates
    ix = int(0.62 * Nx)
    fig = plt.figure(figsize=(12.6, 7.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.05, 1], hspace=0.42, wspace=0.42)
    a = fig.add_subplot(gs[0, :])
    im = a.imshow(T, origin="lower", extent=[0, 4, 0, 1], cmap="inferno", vmin=-0.1, vmax=0, aspect="equal")
    a.axvline(4 * ix / Nx, color="#62A2EE", lw=2.5)
    a.text(4 * ix / Nx + 0.04, 0.80, "one column", color="white", fontsize=16, fontweight="bold", bbox=dict(facecolor=INK, edgecolor="none", pad=3))
    a.set_xlabel("x"); a.set_ylabel("z"); a.grid(False); a.set_yticks([0, 1])
    cb = fig.colorbar(im, ax=a, fraction=0.02, pad=0.01); cb.set_label(r"$T - T_A$", fontsize=15)
    a.set_title("Ra 10¹², one frame: temperature read back from the colours", loc="left", fontsize=17, color=INK)
    panel_label(a, "a")
    col = th[:, ix] / EPS
    b = fig.add_subplot(gs[1, 0])
    b.plot(col, z, color=INK, lw=2.4); b.axvline(0, color=INK2, lw=1)
    b.plot([0, 0], [0, 1], "o", color=ACC, ms=7)
    b.set_xlabel("θ′ / ε"); b.set_ylabel("z"); b.set_title("θ′ = 0 at both plates", loc="left", fontsize=16, color=INK)
    panel_label(b, "b")
    c = fig.add_subplot(gs[1, 1])
    zz = np.linspace(0, 2, 2 * Nz - 1)
    mir = np.concatenate([col, -col[::-1][1:]])
    c.plot(mir[:Nz], zz[:Nz], color=INK, lw=2.4); c.plot(mir[Nz - 1:], zz[Nz - 1:], color=ACC, lw=2.4)
    c.axhline(1, color=INK2, lw=1, ls="--"); c.axvline(0, color=INK2, lw=1)
    c.set_xlabel("θ′ / ε"); c.set_ylabel("z"); c.set_yticks([0, 1, 2])
    c.set_title("mirror: θ′(2 − z) = −θ′(z)", loc="left", fontsize=16, color=INK)
    panel_label(c, "c")
    e = fig.add_subplot(gs[1, 2])
    inner = th[1:-1, ix]
    bn = dst(inner, type=1) / (Nz - 1)                          # sine coefficients b_n
    n = np.arange(1, len(bn) + 1)
    e.loglog(n[:200], np.abs(bn[:200]) / EPS, ".", color=INK, ms=4)
    e.loglog([1, 200], [0.3, 0.3 / 200], color=ACC, lw=2, ls="--")
    e.text(12, 0.06, "∝ 1/n", color=ACC, fontsize=16)
    e.set_xlabel(r"sine wave $n$   ($k_z = n\pi$)"); e.set_ylabel("|bₙ| / ε")
    e.set_title("FFT in z = sine series", loc="left", fontsize=16, color=INK)
    panel_label(e, "d")
    fig.savefig(OUT + "rep_mirror.png", dpi=170, bbox_inches="tight"); plt.close(fig)


def sound():
    fig, ax = plt.subplots(1, 3, figsize=(12.6, 4.3), gridspec_kw={"width_ratios": [1.35, 0.8, 1.05]})
    f, kb, S = SW.k_omega_spectrum("Ra1e15", "cT")
    Sn = S / np.maximum(S.sum(1, keepdims=True), 1e-300)
    a = ax[0]
    a.pcolormesh(kb, f[1:], np.log10(Sn[:, 1:].T * f[1:, None] + 1e-30), vmin=-5, vmax=-0.5, shading="nearest", cmap="Blues", rasterized=True)
    fr = SW.ridge(f, kb, S); ok = np.isfinite(fr) & (kb >= 4) & (kb <= 60)
    zg = np.linspace(0, 1, 401); cs = np.sqrt((G - 1) * (1 - D * zg) / (D * EPS)); c_eff = 1 / np.mean(1 / cs)
    a.plot(kb, c_eff * kb / 2, color=ACC, lw=2.2, ls="--", label=f"theory c = {c_eff:.2f}")
    a.plot(kb[ok], fr[ok], "o", ms=4.5, color="#F2A65A", mec=INK, mew=0.5, label="ridge peak per ring")
    a.set_xlim(1, 70); a.set_ylim(0, 40); a.grid(False)
    a.set_xlabel("K/π"); a.set_ylabel("frequency f"); a.legend(loc="upper left", fontsize=13, framealpha=0.9)
    a.set_title("Ra 10¹⁵: sound ridge", loc="left", fontsize=16, color=INK); panel_label(a, "a")
    b = ax[1]
    b.plot(cs, zg, color=INK, lw=2.4)
    b.axvspan(1.93, 2.08, color="#F2A65A", alpha=0.35, lw=0)
    b.axvline(c_eff, color=ACC, lw=2, ls="--")
    b.text(1.35, 0.05, "measured\n1.93–2.08", fontsize=13, color=INK)
    b.text(c_eff + 0.03, 0.88, f"{c_eff:.2f}", color=ACC, fontsize=14)
    b.set_xlim(1.3, 2.6); b.set_xlabel(r"sound speed $c_s(z)$"); b.set_ylabel("z")
    b.set_title("speed vs height", loc="left", fontsize=16, color=INK); panel_label(b, "b")
    c = ax[2]
    sp = np.load(f"{CACHE}/sound_share.npy", allow_pickle=True).item()
    for t in DF.TAGS:
        s = sp[(t, "cT")]; k = (s["kb"] >= 2) & (s["kb"] <= 40)
        y = np.convolve(s["acoustic"], [0.5, 0.5], "same")
        c.semilogy(s["kb"][k], 100 * y[k], color=COL[t], lw=2.4, label=DF.LAB[t])
    c.set_xlabel("K/π"); c.set_ylabel("% of T variance in sound"); c.legend(fontsize=13, frameon=False, loc="lower right")
    c.set_title("sound grows with Ra", loc="left", fontsize=16, color=INK); panel_label(c, "c")
    fig.tight_layout(w_pad=2.2); fig.savefig(OUT + "rep_sound.png", dpi=170); plt.close(fig)
    print(f"sound: theory c = {c_eff:.3f}; c_s from {cs[0]:.2f} (bottom) to {cs[-1]:.2f} (top)")


def sound_big():
    """Large ridge map for the report: measured ridge peaks, fitted speed and the ideal-gas speed (no fit)."""
    f, kb, S = SW.k_omega_spectrum("Ra1e15", "cT")
    Sn = S / np.maximum(S.sum(1, keepdims=True), 1e-300)
    zg = np.linspace(0, 1, 401); cs = np.sqrt((G - 1) * (1 - D * zg) / (D * EPS)); c_th = 1 / np.mean(1 / cs)
    c_fit = np.load(f"{CACHE}/sound_ridge.npy", allow_pickle=True).item()[("Ra1e15", "cT")][0]
    fr = SW.ridge(f, kb, S); ok = np.isfinite(fr) & (kb >= 4) & (kb <= 60)
    fig, a = plt.subplots(figsize=(12.6, 9.6))
    im = a.pcolormesh(kb, f[1:], np.log10(Sn[:, 1:].T * f[1:, None] + 1e-30), vmin=-5, vmax=-0.5,
                      shading="nearest", cmap="Blues", rasterized=True)
    cb = fig.colorbar(im, ax=a, fraction=0.03, pad=0.015); cb.set_label("log₁₀ power (normalised per ring)", fontsize=15)
    a.plot(kb, c_th * kb / 2, color=ACC, lw=3, ls="--", label=f"ideal-gas theory:  c = {c_th:.2f}  (no fit)")
    a.plot(kb, c_fit * kb / 2, color="#F2A65A", lw=2.2, label=f"line through the ridge:  c = {c_fit:.2f}")
    a.plot(kb[ok], fr[ok], "o", ms=6.5, color="#F2A65A", mec=INK, mew=0.8, label="measured ridge peak in each ring")
    a.set_xlim(1, 60); a.set_ylim(0, 34); a.grid(False)
    a.set_xlabel("K/π   (size of the 2D wave)", fontsize=18); a.set_ylabel("frequency f   (1 / time unit)", fontsize=18)
    a.tick_params(labelsize=16)
    a.legend(loc="upper left", fontsize=15, framealpha=0.95)
    a.text(0.985, 0.04, "Ra 10¹⁵, temperature panel\nslow plumes and swirls: bottom\nsound waves: the straight ridge",
           transform=a.transAxes, ha="right", va="bottom", fontsize=14, color=INK,
           bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=6))
    a.set_title(f"Sound waves: measured {c_fit:.2f}–2.08 against ideal-gas {c_th:.2f}", loc="left", fontsize=19, color=INK)
    fig.tight_layout(); fig.savefig(OUT + "rep_sound_map.png", dpi=170); plt.close(fig)


def sound_side():
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.3))
    zg = np.linspace(0, 1, 401); cs = np.sqrt((G - 1) * (1 - D * zg) / (D * EPS)); c_eff = 1 / np.mean(1 / cs)
    b = ax[0]
    b.plot(cs, zg, color=INK, lw=2.6)
    b.axvspan(1.93, 2.08, color="#F2A65A", alpha=0.35, lw=0)
    b.axvline(c_eff, color=ACC, lw=2.2, ls="--")
    b.text(1.36, 0.06, "measured\n1.93–2.08", fontsize=14, color=INK)
    b.text(c_eff + 0.03, 0.9, f"average {c_eff:.2f}", color=ACC, fontsize=14)
    b.set_xlim(1.3, 2.6); b.set_xlabel(r"ideal-gas sound speed $c_s(z)$"); b.set_ylabel("height z")
    b.set_title("faster in the hot bottom", loc="left", fontsize=16, color=INK); panel_label(b, "b")
    c = ax[1]
    sp = np.load(f"{CACHE}/sound_share.npy", allow_pickle=True).item()
    for t in DF.TAGS:
        s_ = sp[(t, "cT")]; k = (s_["kb"] >= 2) & (s_["kb"] <= 40)
        y = np.convolve(s_["acoustic"], [0.5, 0.5], "same")
        c.semilogy(s_["kb"][k], 100 * y[k], color=COL[t], lw=2.6, label=DF.LAB[t])
    c.set_xlabel("K/π"); c.set_ylabel("% of T variance in sound"); c.legend(fontsize=14, frameon=False, loc="lower right")
    c.set_title("sound grows with Ra", loc="left", fontsize=16, color=INK); panel_label(c, "c")
    fig.tight_layout(w_pad=3); fig.savefig(OUT + "rep_sound_side.png", dpi=170); plt.close(fig)


def validation():
    fig, ax = plt.subplots(1, 2, figsize=(11.0, 4.4))
    a = ax[0]
    rows = np.load(f"{SYNTH}/x_spectra_tests.npy", allow_pickle=True)
    mk = {"1063k": ("o", INK), "300k": ("s", ACC)}
    for al, br, tr, me in rows:
        m, cl = mk[br]; a.plot(-tr, -me, m, color=cl, ms=8, mfc="white" if br == "300k" else cl, mew=2)
    a.plot([1.3, 2.6], [1.3, 2.6], color=INK2, lw=1)
    a.plot([], [], "o", color=INK, label="real bitrate (Ra 10⁹ movie)"); a.plot([], [], "s", color=ACC, mfc="white", mew=2, label="3.5× lower stress test")
    a.set_xlabel("true slope (−)"); a.set_ylabel("recovered slope (−)"); a.legend(fontsize=12, frameon=False, loc="upper left")
    a.set_title("1D FFT along x, k = 10–30", loc="left", fontsize=16, color=INK); panel_label(a, "a")
    b = ax[1]
    for s_ in ("set1", "set2"):
        for r in np.load(f"{SYNTH}/spectra_2d_tests_{s_}.npy", allow_pickle=True):
            lo = r["br"] == "300k"
            for key, m in (("shell_5_15", "o"), ("shell_10_40", "^")):
                tr, me = r[key]
                b.plot(-tr, -me, m, color=ACC if lo else INK, ms=8, mfc="white" if lo else (INK if m == "o" else "#62A2EE"), mew=2)
    b.plot([1.4, 2.5], [1.4, 2.5], color=INK2, lw=1)
    b.plot([], [], "o", color=INK, label="K/π 5–15"); b.plot([], [], "^", color=INK, mfc="#62A2EE", mew=2, label="K/π 10–40")
    b.plot([], [], "o", color=ACC, mfc="white", mew=2, label="stress-test bitrate")
    b.set_xlabel("true slope (−)"); b.set_ylabel("recovered slope (−)"); b.legend(fontsize=12, frameon=False, loc="upper left")
    b.set_title("2D FFT with the mirror trick", loc="left", fontsize=16, color=INK); panel_label(b, "b")
    fig.tight_layout(w_pad=2.5); fig.savefig(OUT + "rep_validation.png", dpi=170); plt.close(fig)


def slide_figures_white():
    DF.BG = "white"; DF.OUT = OUT
    DF.spec1d(); DF.spec2d(); DF.upper(); DF.entropy_share()


if __name__ == "__main__":
    mirror(); sound(); sound_big(); sound_side(); validation(); slide_figures_white()
