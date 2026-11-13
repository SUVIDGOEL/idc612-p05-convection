"""Figures for the Thursday progress review. One style, large fonts, 16:9."""
from convection.config import CACHE, DATA, FIGS
import sys
import av, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from convection.common.stats import slope, per_frame_slopes, integral_time, block_bootstrap_slope
from convection.s1_x_spectra.spectra_pass import build_lut, overlay_mask, make_interpolator, row_spectra

INK, INK2, GRID, BG = "#17181c", "#5a5f6b", "#e4e5e8", "#f7f7f5"
C = {"Ra1e9": "#2a78d6", "Ra1e12": "#eb6834", "Ra1e15": "#1baf7a"}
LAB = {"Ra1e9": "Ra = 10⁹", "Ra1e12": "Ra = 10¹²", "Ra1e15": "Ra = 10¹⁵"}
plt.rcParams.update({"font.size": 19, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.facecolor": BG, "figure.facecolor": BG, "savefig.facecolor": BG,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1, "axes.spines.top": False,
                     "axes.spines.right": False, "lines.linewidth": 2.6, "legend.frameon": False})
OUT = f"{FIGS}/"
TAGS = ("Ra1e9", "Ra1e12", "Ra1e15")
D = {t: np.load(f"{CACHE}/spec_{t}.npz") for t in TAGS}

# ---------- F1: 2D FFT vs 1D x-FFT (Ra1e12, 60 frames, cleaned field) ----------
t = "Ra1e12"; st = np.load(f"{CACHE}/tstats_{t}.npz"); cb = np.load(f"{CACHE}/cbar_{t}.npz")["cbar"]
lut = build_lut(st["mean"], cb); r0, r1, c0, c1 = D[t]["crop"]
interp, _ = make_interpolator(D[t]["mask"])
P2 = None; P1 = None; n = 0
c = av.open(f"{DATA}/2d_Ra1e12.mp4")
for i, fr in enumerate(c.decode(video=0)):
    if i < 200: continue
    if i >= 260: break
    Y = fr.to_ndarray(format="yuv420p")[:fr.height, :fr.width]
    F = interp(lut[Y[r0:r1, c0:c1]].astype(np.float64))
    G = F - F.mean()
    A = np.abs(np.fft.fft2(G))**2 / G.size
    P2 = A if P2 is None else P2 + A
    p1 = row_spectra(F).mean(0)
    P1 = p1 if P1 is None else P1 + p1
    n += 1
c.close()
P2 /= n; P1 /= n
ny, nx = P2.shape
kx = np.fft.fftfreq(nx) * nx; ky = np.fft.fftfreq(ny) * ny
K = np.hypot(kx[None, :], ky[:, None]).ravel(); Pf = P2.ravel()
kb = np.arange(1, ny // 2)
idx = np.round(K).astype(int); sel = (idx >= 1) & (idx < ny // 2)
E2 = np.bincount(idx[sel], weights=Pf[sel], minlength=ny // 2)[1:ny // 2]
k1 = np.arange(1, len(P1) + 1)
fig, ax = plt.subplots(figsize=(12, 6.4), dpi=150)
ax.loglog(k1, P1 / P1[9], color=C["Ra1e12"], label="1D along x (periodic)  — our method")
ax.loglog(kb, E2 / E2[9], color=INK2, ls="--", label="2D FFT (z has walls)  — first attempt")
ax.set_xlabel("wavenumber k"); ax.set_ylabel("E(k), scaled to 1 at k = 10")
ax.set_xlim(1, 1100); ax.legend(loc="lower left")
ax.set_title("Ra = 10¹²: the 1D transform along x reaches k ≈ 1000; the 2D transform stops at k ≈ 260", fontsize=18, loc="left", color=INK)
plt.tight_layout(); plt.savefig(OUT + "f1_2d_vs_1d.png"); plt.close()

# ---------- F2: overlay contamination ----------
fig, ax = plt.subplots(1, 2, figsize=(14, 6.0), dpi=150, sharey=True)
for a, t in zip(ax, ("Ra1e12", "Ra1e15")):
    Et, Ec = D[t]["tight"].mean(0), D[t]["clean"].mean(0); k = np.arange(1, len(Et) + 1)
    for h in range(16, 129, 16): a.axvline(h, color=INK2, lw=1, ls=":")
    a.plot(k, Et / Ec, color=C[t]); a.axhline(1, color=INK2, lw=1.2)
    a.set_xlim(1, 130); a.set_ylim(0.9, 3.0); a.set_xlabel("wavenumber k")
    a.set_title(LAB[t], loc="left", fontsize=20, color=INK)
ax[0].set_ylabel("power with box ÷ power without")
plt.tight_layout(); plt.savefig(OUT + "f2_overlay.png"); plt.close()

# ---------- F3: correlated frames (Ra1e12) ----------
S = D["Ra1e12"]["clean"].astype(np.float64)
ts = per_frame_slopes(S, 10, 30); tau, acf = integral_time(ts)
fig, ax = plt.subplots(1, 2, figsize=(14, 5.6), dpi=150, gridspec_kw={"width_ratios": [1.6, 1]})
ax[0].plot(ts, color=C["Ra1e12"], lw=1.6); ax[0].set_xlabel("frame number"); ax[0].set_ylabel("slope of one frame")
ax[0].set_title("Ra = 10¹²: the slope changes slowly from frame to frame", loc="left", fontsize=18, color=INK)
lag = np.arange(len(acf)); ax[1].plot(lag, acf, color=INK); ax[1].axhline(0, color=INK2, lw=1)
ax[1].fill_between(lag[:int(2*tau)], 0, acf[:int(2*tau)], color=C["Ra1e12"], alpha=0.25)
ax[1].set_xlim(0, 330); ax[1].set_xlabel("lag (frames)"); ax[1].set_ylabel("autocorrelation")
ax[1].set_title(f"correlation time τ ≈ {tau:.0f} frames", loc="left", fontsize=18, color=INK)
plt.tight_layout(); plt.savefig(OUT + "f3_correlation.png"); plt.close()

# ---------- F4: spectra of the three Ra ----------
fig, ax = plt.subplots(figsize=(12, 6.6), dpi=150)
floor = {"Ra1e9": 177, "Ra1e12": 873, "Ra1e15": 155}
for t in TAGS:
    E = D[t]["clean"].mean(0); k = np.arange(1, len(E) + 1); m = k <= floor[t]
    En = E / np.exp(np.mean(np.log(E[(k >= 10) & (k <= 30)])))
    ax.loglog(k[m], En[m], color=C[t], label=f"{LAB[t]}   slope {slope(E, 10, 30):+.2f}")
ax.axvspan(10, 30, color="#8a1c5c", alpha=0.08); ax.text(11, 0.0025, "fit band\nk = 10–30", color="#8a1c5c", fontsize=17)
ax.set_xlabel("wavenumber k"); ax.set_ylabel("E(k), scaled at k = 10–30"); ax.legend(loc="lower left")
ax.set_xlim(1, 900)
plt.tight_layout(); plt.savefig(OUT + "f4_spectra.png"); plt.close()

# ---------- F5: exponent vs Ra with CIs ----------
res = {}
for t in TAGS:
    for v in ("clean", "lowz"):
        Sv = D[t][v].astype(np.float64)
        tv, _ = integral_time(per_frame_slopes(Sv, 10, 30))
        bs = block_bootstrap_slope(Sv, 10, 30, max(1, int(np.ceil(4 * tv))), nboot=2000)
        res[(t, v)] = (slope(Sv.mean(0), 10, 30), *np.percentile(bs, [2.5, 97.5]))
x = np.array([0, 1, 2])
fig, ax = plt.subplots(figsize=(12, 6.6), dpi=150)
for v, col, off, lab in (("clean", INK, -0.08, "all heights"), ("lowz", "#8a1c5c", 0.08, "top layer removed (z < 0.72)")):
    s = np.array([res[(t, v)][0] for t in TAGS]); lo = np.array([res[(t, v)][1] for t in TAGS]); hi = np.array([res[(t, v)][2] for t in TAGS])
    ax.errorbar(x + off, s, yerr=[s - lo, hi - s], fmt="o", color=col, capsize=7, ms=10, lw=2.4, label=lab)
for val, name in ((-7 / 5, "−7/5  Bolgiano–Obukhov (T)"), (-5 / 3, "−5/3  Obukhov–Corrsin"), (-2, "−2  sharp fronts")):
    ax.axhline(val, color=INK2, ls="--", lw=1.3); ax.text(2.32, val, name, va="center", fontsize=16, color=INK2)
ax.set_xticks(x); ax.set_xticklabels(["10⁹", "10¹²", "10¹⁵"]); ax.set_xlim(-0.35, 3.45)
ax.set_xlabel("Rayleigh number"); ax.set_ylabel("slope, k = 10–30"); ax.legend(loc="lower left")
plt.tight_layout(); plt.savefig(OUT + "f5_exponent.png"); plt.close()

# ---------- F6: slope vs height ----------
fig, ax = plt.subplots(figsize=(12, 6.6), dpi=150)
hres = {}
for t in TAGS:
    B = D[t]["bands"].astype(np.float64); nb = B.shape[1]
    zc = 1 - (np.arange(nb) + 0.5) / nb            # band 0 is the top row (z = 1)
    sl, lo, hi = [], [], []
    for b in range(nb):
        Sb = B[:, b, :]
        tb, _ = integral_time(per_frame_slopes(Sb, 10, 30))
        bs = block_bootstrap_slope(Sb, 10, 30, max(1, int(np.ceil(4 * tb))), nboot=600, seed=b)
        sl.append(slope(Sb.mean(0), 10, 30)); l, h = np.percentile(bs, [2.5, 97.5]); lo.append(l); hi.append(h)
    sl, lo, hi = map(np.array, (sl, lo, hi))
    ax.fill_betweenx(zc, lo, hi, color=C[t], alpha=0.18); ax.plot(sl, zc, "o-", color=C[t], ms=7, label=LAB[t])
    hres[t] = (zc, sl)
ax.set_xlabel("slope, k = 10–30"); ax.set_ylabel("height z  (0 = bottom plate, 1 = top plate)")
ax.legend(loc="upper left"); ax.set_ylim(0, 1)
plt.tight_layout(); plt.savefig(OUT + "f6_height.png"); plt.close()

np.save(OUT + "results.npy", {"exp": res, "height": hres, "tau_Ra1e12": tau}, allow_pickle=True)
print("tau Ra1e12 =", round(tau, 1))
for k_, v in res.items(): print(k_, "%+.2f [%+.2f, %+.2f]" % v)
for t in TAGS:
    zc, sl = hres[t]; print(t, "slope by height (top->bottom):", " ".join("%+.2f" % s for s in sl))
