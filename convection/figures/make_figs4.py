"""Figures for the vertical (mirror-trick) analysis. Saves figures/f27..f37."""
from convection.config import CACHE, FIGS, SYNTH
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
EPS = 0.1
COL = {"Ra1e9": "#2a78d6", "Ra1e12": "#eb6834", "Ra1e15": "#1baf7a", "3d": "#c98500"}
LAB = {"Ra1e9": "2D, Ra = 10⁹", "Ra1e12": "2D, Ra = 10¹²", "Ra1e15": "2D, Ra = 10¹⁵", "3d": "3D, Ra = 10⁸"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#dddcd8"
plt.rcParams.update({"font.size": 12, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2,
                     "figure.facecolor": "white", "axes.facecolor": "white", "savefig.bbox": "tight"})
OUT = f"{FIGS}/"
R = {t: np.load(f"{CACHE}/analyse4_{t}.npy", allow_pickle=True).item() for t in ("Ra1e9", "Ra1e12", "Ra1e15")}
R3 = np.load(f"{CACHE}/analyse4_3d.npy", allow_pickle=True).item()
R["3d"] = R3["wall_xz"]
TAGS = ("Ra1e9", "Ra1e12", "Ra1e15", "3d")


def label_end(ax, x, y, text, color, dx=1.04, va="center"):
    ax.text(x * dx, y, text, color=INK, fontsize=10, va=va)


# ---------------------------------------------------------------- f27: the mirror trick on a real column
def f27():
    d = np.load(f"{CACHE}/pass4_Ra1e12.npz")
    th = d["T300"].astype(float); Nz = th.shape[0] - 1
    z = np.arange(Nz + 1) / Nz
    col = th[:, 1200] + EPS * z
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.6, 1]})
    a = ax[0]
    zz = np.concatenate([-z[::-1], z, 2 - z[::-1]])
    ff = np.concatenate([-col[::-1], col, -col[::-1]])
    a.axvspan(0, 1, color="#eef3fb", zorder=0)
    a.plot(zz, ff, color=COL["Ra1e9"], lw=1.6)
    a.axhline(0, color=INK2, lw=0.8)
    for zp in (0, 1):
        a.axvline(zp, color=INK2, lw=1, ls="--")
    a.text(0.5, ff.max() * 1.05, "real column (one frame)", ha="center", color=INK)
    a.text(-0.5, ff.max() * 1.05, "odd mirror", ha="center", color=INK2)
    a.text(1.5, ff.max() * 1.05, "odd mirror", ha="center", color=INK2)
    a.set_xlabel("z  (plates at z = 0 and z = 1)"); a.set_ylabel("θ′ = (T − T_A) + εz")
    a.set_title("θ′ is 0 at both plates → mirror it → period 2, no jump", fontsize=12, loc="left")
    a = ax[1]
    from scipy.fft import dst
    b = dst(col[1:-1], type=1) / Nz
    n = np.arange(1, Nz)
    a.loglog(n[1::2], b[1::2] ** 2 / 2, "o", ms=4, color=COL["Ra1e12"], label="even n")
    a.loglog(n[::2], b[::2] ** 2 / 2, "s", ms=3, color=COL["Ra1e9"], label="odd n")
    a.loglog(n, (b[1] ** 2 / 2) * (n / 2.0) ** -2, color=INK2, lw=1, ls="--")
    a.text(30, (b[1] ** 2 / 2) * 15 ** -2 * 1.6, "n⁻²", color=INK2)
    a.set_xlabel("n   (k_z = nπ)"); a.set_ylabel("energy of sin(nπz) mode")
    a.set_title("sine series of that column", fontsize=12, loc="left"); a.legend(frameon=False)
    fig.savefig(OUT + "f27_mirror.png", dpi=150); plt.close(fig)


# ---------------------------------------------------------------- f28: the two branches, every mode
def f28():
    d = np.load(f"{CACHE}/pass4_Ra1e12.npz")
    E = d["acc_T2d"].astype(float)
    Nz, Nx = int(d["Nz"]), int(d["Nx"])
    n = np.arange(1, Nz)[:, None]; m = np.arange(E.shape[1])[None, :]
    K = np.hypot(m / 2.0, n)
    fig, ax = plt.subplots(figsize=(9.5, 6))
    sel = (m >= 1) & (K <= 150) & (E > 0)
    ax.hexbin(K[sel], E[sel], xscale="log", yscale="log", gridsize=70, bins="log", cmap="Greys", mincnt=1)
    ev = (np.arange(1, Nz) % 2 == 0)
    Em = E[:, 0]
    ax.loglog(np.arange(1, Nz)[ev], Em[ev], "o", ms=5, color=COL["Ra1e12"], label="mean-profile modes θ̂(0, n), even n")
    ax.loglog(np.arange(1, Nz)[~ev], Em[~ev], "s", ms=4, color=COL["Ra1e9"], label="mean-profile modes, odd n")
    r = R["Ra1e12"]; nn = np.arange(1, Nz); pr = r["E_mean_pred"]; k = nn <= 80
    ax.loglog(nn[k & ev], pr[k & ev], color=INK, lw=1.2, label="jump model 2/(nπ)² [J₀ − (−1)ⁿJ₁]²  (no fit)")
    ax.loglog(nn[k & ~ev], pr[k & ~ev], color=INK, lw=1.2)
    ax.axvline(Nz / 16, color=INK2, lw=1, ls=":"); ax.text(Nz / 16 * 1.05, 1e-4, "method limit\n(synthetic test)", color=INK2, fontsize=10)
    ax.set_xlim(0.8, 160); ax.set_ylim(1e-14, 2e-3)
    ax.set_xlabel("K / π"); ax.set_ylabel("energy of one mode")
    ax.text(25, 3e-12, "all other modes\n(fluctuations):\nlower branch", color=INK2, fontsize=11)
    ax.text(3, 1.2e-4, "upper branch", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=10, loc="lower left")
    ax.set_title("Ra = 10¹²: every (k_x, k_z) mode of θ′ — two branches", loc="left", fontsize=12)
    fig.savefig(OUT + "f28_dual_branch.png", dpi=150); plt.close(fig)


# ---------------------------------------------------------------- f29: upper branch for all movies
def f29():
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.2))
    for t in TAGS:
        r = R[t]; Em = r["E_mean"]; n = np.arange(1, len(Em) + 1)
        k = (n <= 40) & (n % 2 == 0)
        ax[0].loglog(n[k], Em[k] / EPS ** 2, "o", ms=4, color=COL[t], label=LAB[t])
        ax[0].loglog(n[k], r["E_mean_pred"][k] / EPS ** 2, color=COL[t], lw=1.2)
        ratio = np.sqrt(Em[:12] / r["E_mean_pred"][:12])
        ax[1].plot(np.arange(1, 13), ratio, "o-", ms=5, color=COL[t], lw=1.5, label=LAB[t])
    ax[0].set_xlabel("n   (k_z = nπ)"); ax[0].set_ylabel("mean-profile mode energy / ε²")
    ax[0].set_title("even n. Dots: measured. Lines: jump model, no fitted constant", loc="left", fontsize=12)
    ax[0].legend(frameon=False, fontsize=10)
    ax[1].axhline(1, color=INK2, lw=1)
    ax[1].set_ylim(0, 1.6); ax[1].set_xlabel("n"); ax[1].set_ylabel("measured / predicted amplitude")
    ax[1].set_title("ratio = 1 until n ~ 1/(πδ): resolved layers (10⁹, 3D) roll off", loc="left", fontsize=12)
    fig.savefig(OUT + "f29_upper_branch.png", dpi=150); plt.close(fig)


# ---------------------------------------------------------------- f30: up-down asymmetry of the profile
def f30():
    fig, ax = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for t in TAGS:
        r = R[t]; p = r["prof_T"]; z = np.linspace(0, 1, len(p))
        ax[0].plot(p / EPS, z, color=COL[t], label=LAB[t])
        ax[1].plot(r["rms_profile"] / EPS, z, color=COL[t])
        ax[2].plot(r["skew_z"], z, color=COL[t])
    ax[0].plot([0, -1], [0, 1], color=INK2, lw=1, ls="--"); ax[0].text(-0.62, 0.55, "conduction", color=INK2, rotation=-52)
    ax[0].set_xlabel("(T − T_A) / ε"); ax[0].set_ylabel("z"); ax[0].legend(frameon=False, fontsize=9, loc="lower left")
    ax[0].set_title("mean: big drop at the TOP plate", loc="left", fontsize=12)
    ax[1].set_xlabel("rms / ε"); ax[1].set_title("rms: larger near the top", loc="left", fontsize=12)
    ax[2].axvline(0, color=INK2, lw=1); ax[2].set_xlabel("skewness"); ax[2].set_xlim(-3.2, 1.2)
    ax[2].set_title("skewness: cold events dominate", loc="left", fontsize=12)
    fig.savefig(OUT + "f30_profiles.png", dpi=150); plt.close(fig)


# ---------------------------------------------------------------- f31: lower branch shells, compensated
def f31():
    fig, ax = plt.subplots(figsize=(10, 5.6))
    vmax = {"Ra1e9": 40, "Ra1e12": 40, "Ra1e15": 100, "3d": 30}
    for t in TAGS:
        r = R[t]; K = r["K"]; E = r["shell"]
        k = (K >= 1) & (K <= 150) & np.isfinite(E)
        y = E[k] * K[k] ** (5 / 3)
        y = np.convolve(y, [0.5, 0.5], "same"); y[0] = E[k][0] * K[k][0] ** (5 / 3)
        y = y / np.nanmedian(y[(K[k] >= 5) & (K[k] <= 15)])
        ok = K[k] <= vmax[t]
        ax.loglog(K[k][ok], y[ok], color=COL[t], label=LAB[t])
        ax.loglog(K[k][~ok], y[~ok], color=COL[t], lw=1, ls=":")
    ax.axhline(1, color=INK2, lw=1)
    ax.set_xlabel("K / π   (2D shell, k_x and k_z together)"); ax.set_ylabel("E(K) · K^(5/3)   (scaled to 1 at K/π = 5–15)")
    ax.set_title("fluctuation (lower-branch) spectra. Flat = −5/3. Dotted: beyond the range the synthetic test validates", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=10); ax.set_ylim(0.03, 30)
    fig.savefig(OUT + "f31_shells.png", dpi=150); plt.close(fig)


# ---------------------------------------------------------------- f32: isotropy
def f32():
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.2))
    r = R["Ra1e12"]
    ax[0].loglog(r["kx"], r["Ex"], color=COL["Ra1e9"], label="E(k_x): all heights")
    ax[0].loglog(r["kz"], r["Ez"], color=COL["Ra1e12"], label="E(k_z): all heights")
    if "Ex_bulk" in r:
        ax[0].loglog(r["kx"], r["Ex_bulk"], color=COL["Ra1e9"], lw=1, ls="--", label="E(k_x): bulk, same window")
        ax[0].loglog(r["kz_hann"][1:], r["Ez_hann"][1:] / (r["kz_hann"][1] - r["kz_hann"][0]), color=COL["Ra1e12"], lw=1, ls="--", label="E(k_z): bulk, same window")
    ax[0].set_xlim(0.5, 200); ax[0].set_xlabel("k / π"); ax[0].set_ylabel("1D spectral density")
    ax[0].set_title("Ra = 10¹²: horizontal vs vertical", loc="left", fontsize=12); ax[0].legend(frameon=False, fontsize=9)
    labs, full, bulk, fe, be = [], [], [], [], []
    for t in ("Ra1e9", "Ra1e12", "Ra1e15"):
        labs.append(LAB[t].replace("2D, ", ""))
        v = R[t]["iso_z_over_x_3_10"]; full.append(v[0]); fe.append([[v[0] - v[1]], [v[2] - v[0]]])
        v = R[t]["iso_bulk_3_10"]; bulk.append(v[0]); be.append([[v[0] - v[1]], [v[2] - v[0]]])
    x = np.arange(3)
    for i in range(3):
        ax[1].bar(x[i] - 0.18, full[i], 0.34, color=COL["Ra1e12"], yerr=fe[i], capsize=3, ecolor=INK2)
        ax[1].bar(x[i] + 0.18, bulk[i], 0.34, color=COL["Ra1e9"], yerr=be[i], capsize=3, ecolor=INK2)
    ax[1].axhline(1, color=INK, lw=1); ax[1].set_xticks(x, labs)
    ax[1].set_ylabel("E(k_z) / E(k_x) at k/π = 3–10")
    ax[1].text(-0.4, max(full) * 1.08, "orange: whole box   blue: bulk only (same window)", color=INK2, fontsize=10)
    ax[1].set_title("1 = isotropic. The excess is at the plates; the bulk is isotropic", loc="left", fontsize=12)
    fig.savefig(OUT + "f32_isotropy.png", dpi=150); plt.close(fig)


# ---------------------------------------------------------------- f33: sound waves
def f33():
    from convection.s5_vertical import kw as KW
    f, kb, S = KW.kw("Ra1e15", "cT")
    Sn = S / np.maximum(S.sum(1, keepdims=True), 1e-300)
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.4))
    a = ax[0]
    a.pcolormesh(kb, f[1:], np.log10(Sn[:, 1:].T * f[1:, None] + 1e-30), vmin=-5, vmax=-0.5, shading="nearest", cmap="Greys")
    zg = np.linspace(0, 1, 401); cs = np.sqrt(0.3 * (1 - 0.5 * zg) / (0.5 * 0.1))
    c_eff = 1 / np.mean(1 / cs)
    a.plot(kb, c_eff * kb / 2, color=COL["Ra1e12"], lw=1.5, ls="--")
    a.text(45, 4, f"dashed: sound, ω = c_s K, c_s = {c_eff:.2f}\nfrom γ = 1.3, D = 0.5, ε = 0.1 (no fit)", color=INK, fontsize=10,
           bbox=dict(facecolor="white", edgecolor="none", alpha=0.9))
    a.set_ylim(0, 50); a.set_xlim(1, 90)
    a.set_xlabel("K / π"); a.set_ylabel("frequency f (1 / time unit)")
    a.set_title("Ra = 10¹⁵, temperature: a sound-wave ridge", loc="left", fontsize=12)
    kwr = np.load(f"{CACHE}/kw.npy", allow_pickle=True).item()
    sp = np.load(f"{CACHE}/kw_split.npy", allow_pickle=True).item()
    a = ax[1]
    for t in ("Ra1e9", "Ra1e12", "Ra1e15"):
        s = sp[(t, "cT")]
        k = (s["kb"] >= 2) & (s["kb"] <= 60)
        y = np.convolve(s["acoustic"], [0.5, 0.5], "same")
        a.semilogy(s["kb"][k], y[k] * 100, color=COL[t], label=LAB[t])
    a.set_xlabel("K / π"); a.set_ylabel("% of T variance on the sound ridge")
    a.set_title("sound waves grow with Ra (Mach number)", loc="left", fontsize=12); a.legend(frameon=False, fontsize=10)
    fig.savefig(OUT + "f33_sound.png", dpi=150); plt.close(fig)
    return c_eff, kwr


# ---------------------------------------------------------------- f34: PDFs
def f34():
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    he = np.linspace(-0.1, 0.1, 401); xc = 0.5 * (he[1:] + he[:-1])
    names = ("near bottom (z 0.1–0.2)", "middle (z 0.4–0.5)", "near top (z 0.8–0.9)")
    for j, b in enumerate((1, 4, 8)):
        for t in ("Ra1e9", "Ra1e12", "Ra1e15"):
            Hk = np.load(f"{CACHE}/pass4_{t}.npz")["H_T"][b].astype(float)
            h = Hk / Hk.sum(); mu = (h * xc).sum(); sd = np.sqrt((h * (xc - mu) ** 2).sum())
            y = (xc - mu) / sd; dens = h / (he[1] - he[0]) * sd
            k = dens > 0
            ax[j].semilogy(y[k], dens[k], color=COL[t], label=LAB[t])
        g = np.linspace(-6, 6, 200)
        ax[j].semilogy(g, np.exp(-g ** 2 / 2) / np.sqrt(2 * np.pi), color=INK2, lw=1, ls="--", label="Gaussian")
        ax[j].set_xlim(-10, 6); ax[j].set_ylim(1e-6, 1); ax[j].set_xlabel("(θ − mean) / rms"); ax[j].set_title(names[j], loc="left", fontsize=12)
    ax[0].set_ylabel("PDF"); ax[0].legend(frameon=False, fontsize=9, loc="lower center")
    fig.savefig(OUT + "f34_pdfs.png", dpi=150); plt.close(fig)


# ---------------------------------------------------------------- f35: structure functions / intermittency
def f35():
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    for t in TAGS:
        r = R[t]
        ax[0].loglog(r["rx"], r["flat_r_x_T"], color=COL[t], label=LAB[t])
        ax[1].plot(np.arange(1, 7), r["ess_x_T"], "o-", color=COL[t], ms=5, lw=1.5, label=LAB[t])
    ax[0].axhline(3, color=INK2, lw=1, ls="--"); ax[0].text(0.3, 3.3, "Gaussian = 3", color=INK2)
    ax[0].set_xlabel("separation r (along x)"); ax[0].set_ylabel("flatness ⟨δθ⁴⟩ / ⟨δθ²⟩²")
    ax[0].set_title("small-scale jumps are rare and violent (intermittent)", loc="left", fontsize=12); ax[0].legend(frameon=False, fontsize=9)
    p = np.arange(1, 7)
    ax[1].plot(p, p / 3, color=INK2, lw=1, ls="--"); ax[1].text(4.6, 1.75, "no intermittency: p/3", color=INK2)
    ax[1].set_xlabel("order p"); ax[1].set_ylabel("ζ_p / ζ_3   (extended self-similarity)")
    ax[1].set_title("exponents bend below p/3", loc="left", fontsize=12)
    fig.savefig(OUT + "f35_intermittency.png", dpi=150); plt.close(fig)


# ---------------------------------------------------------------- f36: decorrelation time
def f36():
    fig, ax = plt.subplots(figsize=(9, 5.4))
    for t in TAGS:
        r = R[t]
        noisy = t in ("Ra1e12", "Ra1e15")
        ax.loglog(r["tau_K_env"], r["tau_v_env"], "o-", ms=3 if noisy else 4, color=COL[t], lw=1 if noisy else 1.8,
                  alpha=0.45 if noisy else 1, label=LAB[t] + ("  (sound waves mixed in)" if noisy else ""))
    K = np.array([3, 30.0])
    ax.loglog(K, 1.0 * (K / 3) ** -1, color=INK2, lw=1, ls="--"); ax.text(32, 0.085, "K⁻¹ (sweeping)", color=INK2)
    ax.loglog(K, 0.6 * (K / 3) ** (-2 / 3), color=INK2, lw=1, ls=":"); ax.text(32, 0.6 * 10 ** (-2 / 3) * 1.1, "K^(−2/3)", color=INK2)
    from matplotlib.ticker import ScalarFormatter, NullFormatter
    ax.set_xticks([2, 5, 10, 20, 50]); ax.xaxis.set_major_formatter(ScalarFormatter()); ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("K / π"); ax.set_ylabel("decorrelation time τ (time units)")
    ax.set_title("how long a pattern of size 2/K lives", loc="left", fontsize=12); ax.legend(frameon=False, fontsize=9)
    fig.savefig(OUT + "f36_tau.png", dpi=150); plt.close(fig)


# ---------------------------------------------------------------- f37: validation
def f37():
    rows = list(np.load(f"{SYNTH}/vertical_results_set1.npy", allow_pickle=True)) + list(np.load(f"{SYNTH}/vertical_results_set2.npy", allow_pickle=True))
    fig, ax = plt.subplots(figsize=(11, 5))
    keys = [("shell_5_15", "shell 5–15"), ("shell_10_40", "shell 10–40"), ("z_5_15", "z 5–15"), ("z_10_40", "z 10–40"),
            ("x_10_30", "x old 10–30"), ("mean_2_20", "mean modes 2–20")]
    names = {("1063k", 30): "Ra 10⁹ bitrate", ("300k", 30): "¼ of it", ("2360k", 30): "Ra 10¹² bitrate", ("2758k", 10): "Ra 10¹⁵ bitrate"}
    seen = set(); x = np.arange(len(keys)); w = 0.8 / 5; j = 0
    palette = ["#2a78d6", "#eb6834", "#1baf7a", "#c98500", "#6250d6"]
    for r in rows:
        key = (r["br"], r["fps"])
        if r["alpha"] > 2 or r["db"] < 0.001 and key in seen:
            continue
        lab = names.get(key, key[0]) + (", sharp layer" if r["db"] < 0.001 else "")
        err = [r[k][1] - r[k][0] for k, _ in keys]
        ax.bar(x - 0.4 + w * (j + 0.5), err, w, color=palette[j % 5], label=lab); j += 1; seen.add(key)
        if j >= 5:
            break
    ax.axhline(0, color=INK, lw=1); ax.axhspan(-0.03, 0.03, color="#eef3fb", zorder=0)
    ax.set_xticks(x, [k[1] for k in keys]); ax.set_ylabel("recovered − true slope")
    ax.set_title("synthetic movies with a known answer, true slope −5/3: errors (shaded = ±0.03)", loc="left", fontsize=12)
    ax.legend(frameon=False, fontsize=9, ncol=3)
    fig.savefig(OUT + "f37_validation.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    import sys as _s
    todo = _s.argv[1:] or ['27','28','29','30','31','32','33','34','35','36','37']
    for t_ in todo: globals()['f' + t_]()
