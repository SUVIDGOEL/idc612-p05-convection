"""Wavenumber-frequency spectra of the low 2D modes (complex amplitudes stored by mirror_modes_2d.py).

Looks for a ridge omega = c K (sound waves). For each K-shell the power is normalised by its total, so the ridge
is visible at every K. The ridge is fitted by the frequency of the high-frequency maximum per shell
(above the low-frequency convective part), then omega = 2 pi f against K = pi (K/pi).
Theory check: if velocities are in free-fall units U = sqrt(g d eps) and the gas is ideal with T_A = 1 - D z,
the sound speed is c_s / U = sqrt(gamma R T) / U = sqrt((gamma - 1) T / (D eps)).
"""
from convection.config import CACHE, FIGS
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DT, LX = 0.01, 4.0
G, D, EPS = 1.3, 0.5, 0.1


def k_omega_spectrum(tag, key):
    d = np.load(f"{CACHE}/mirror_modes_2d_{tag}.npz")
    a = d[key].astype(np.complex128)
    if key == "cR":
        a = a[:, 1:]                                 # drop n = 0 of the cosine series
    a = a - a.mean(0)
    nf = a.shape[0]
    W = np.hanning(nf)[:, None, None]
    P = np.abs(np.fft.fft(a * W, axis=0)) ** 2
    f = np.fft.fftfreq(nf, DT)
    Pf = P[: nf // 2 + 1].copy(); Pf[1: nf // 2] += P[::-1][: nf // 2 - 1]       # fold +- f
    f = np.abs(f[: nf // 2 + 1])
    nn = np.arange(1, a.shape[1] + 1); mm = np.arange(a.shape[2])
    K = np.hypot(mm[None, :] * 2 / LX, nn[:, None])
    flu = (mm[None, :] >= 1) & np.ones((len(nn), 1), bool)
    kb = np.arange(1, int(K[flu].max()) + 1)
    S = np.zeros((len(kb), len(f)))
    for i, k in enumerate(kb):
        sel = flu & (np.abs(K - k) < 0.5)
        if sel.any():
            S[i] = Pf[:, sel].sum(1)
    return f, kb, S


def ridge(f, kb, S, fmin_factor=1.0):
    """Per shell: frequency of the maximum of f * P(f) above a floor frequency (skips the convective part)."""
    out = []
    for i, k in enumerate(kb):
        s = S[i] * f
        m = f >= 1.0 + 0.3 * k * fmin_factor
        if m.sum() < 5 or S[i].sum() == 0:
            out.append(np.nan); continue
        j = np.argmax(np.convolve(s[m], np.ones(5) / 5, "same"))
        # require a real bump: peak > 3 x the value at the high-f end of the window
        out.append(f[m][j] if s[m][j] > 3 * np.median(s[m][-10:]) else np.nan)
    return np.array(out)


if __name__ == "__main__":
    fig, ax = plt.subplots(2, 3, figsize=(17, 9))
    res = {}
    for c, tag in enumerate(("Ra1e9", "Ra1e12", "Ra1e15")):
        for r, key in enumerate(("cT", "cR")):
            f, kb, S = k_omega_spectrum(tag, key)
            Sn = S / np.maximum(S.sum(1, keepdims=True), 1e-300)
            a = ax[r, c]
            a.pcolormesh(kb, f[1:], np.log10(Sn[:, 1:].T * f[1:, None] + 1e-30), vmin=-5, vmax=-0.5, shading="nearest", cmap="magma")
            fr = ridge(f, kb, S)
            ok = np.isfinite(fr) & (kb >= 4) & (kb <= 60)
            if ok.sum() >= 5:
                cfit = np.sum(2 * np.pi * fr[ok] * np.pi * kb[ok]) / np.sum((np.pi * kb[ok]) ** 2)
                resid = np.std(2 * np.pi * fr[ok] / (np.pi * kb[ok]))
                a.plot(kb[ok], fr[ok], "c.", ms=4)
                a.plot(kb, cfit * np.pi * kb / (2 * np.pi), "c--", lw=1)
                res[(tag, key)] = (cfit, resid, int(ok.sum()))
                lab = f"c = {cfit:.2f} (scatter {resid:.2f}, {ok.sum()} shells)"
            else:
                lab = "no ridge"
                res[(tag, key)] = (np.nan, np.nan, int(ok.sum()))
            a.set_title(f"{tag} {'T' if key == 'cT' else 'rho'}: {lab}", fontsize=9)
            a.set_xlabel("K / pi"); a.set_ylabel("frequency f (1/time)")
            a.set_ylim(0, 50)
    plt.tight_layout(); plt.savefig(f"{FIGS}/sound_ridge_all.png", dpi=80)
    for k, v in res.items():
        print(k, "c = %.3f, scatter %.3f, shells %d" % v)
    zg = np.linspace(0, 1, 5)
    print("theory, free-fall units: c_s = sqrt((g-1) T / (D eps)) at z = 0, .25, .5, .75, 1:",
          np.round(np.sqrt((G - 1) * (1 - D * zg) / (D * EPS)), 2))
    np.save(f"{CACHE}/sound_ridge.npy", res, allow_pickle=True)


def split(tag, key, c_ac=2.0):
    """Acoustic fraction per K (power within 0.75..1.35 x the ridge frequency) and the rms frequency of the
    slow (convective) band f < 0.5 f_ridge. f_ridge = c K / 2 with K in units of pi."""
    f, kb, S = k_omega_spectrum(tag, key)
    fr = c_ac * kb / 2.0
    ac, wr = [], []
    for i, k in enumerate(kb):
        tot = S[i].sum()
        a = S[i][(f >= 0.75 * fr[i]) & (f <= 1.35 * fr[i])].sum() if 1.35 * fr[i] < f.max() else np.nan
        slow = f < 0.5 * fr[i]
        w = np.sqrt((S[i][slow] * (2 * np.pi * f[slow]) ** 2).sum() / S[i][slow].sum())
        ac.append(a / tot); wr.append(w)
    return kb, np.array(ac), np.array(wr)


if __name__ == "__main__":
    out = {}
    for tag in ("Ra1e9", "Ra1e12", "Ra1e15"):
        for key in ("cT", "cR"):
            kb, ac, wr = split(tag, key)
            ok = (kb >= 3) & (kb <= 30)
            Us = np.polyfit(np.log(np.pi * kb[ok]), np.log(wr[ok]), 1)
            out[(tag, key)] = dict(kb=kb, acoustic=ac, wrms=wr, slope=Us[0], U=np.median(wr[ok] / (np.pi * kb[ok])))
            sel = lambda lo, hi: np.nanmean(ac[(kb >= lo) & (kb <= hi)])
            print(f"{tag} {'T  ' if key == 'cT' else 'rho'}: acoustic fraction K/pi 3-10 {sel(3, 10):.3f}, 10-30 {sel(10, 30):.3f}, 30-60 {sel(30, 60):.3f} | "
                  f"slow band: omega_rms ~ K^{Us[0]:.2f} (K/pi 3-30), omega_rms/K = {out[(tag, key)]['U']:.3f}")
    np.save(f"{CACHE}/sound_share.npy", out, allow_pickle=True)
