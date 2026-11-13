"""Statistics of the pass4 (2D) and pass5 (3D walls) caches. All intervals: 95% moving-block bootstrap over frames.

Prints a report and saves cache/analyse4_<tag>.npy (dict of results) for figures and tables.
usage: python -m convection.s5_vertical.analyse4 Ra1e9|Ra1e12|Ra1e15|3d
"""
from convection.config import CACHE
import sys
import numpy as np
from convection.common.stats import integral_time

EPS = 0.1
DT = 0.01                                           # time units per frame (all movies)
NBOOT = 1000


# ------------------------------------------------------------------ data access
class Src:
    """Uniform access to a 2D movie (pass4) or one 3D wall face (pass5, prefix 'wall__')."""

    def __init__(self, path, prefix=""):
        self.d = np.load(path)
        self.p = prefix

    def __getitem__(self, k):
        return self.d[self.p + k]

    def has(self, k):
        return (self.p + k) in self.d.files


def fit(x, y, lo, hi):
    m = (x >= lo) & (x <= hi) & (y > 0) & np.isfinite(y)
    if m.sum() < 3:
        return np.nan
    return np.polyfit(np.log10(x[m]), np.log10(y[m]), 1)[0]


def boot(n, L, fn, nboot=None, seed=0):
    """Batch means (non-overlapping batches of length 2 tau = L/2, at least 3 batches).
    Returns (value on all frames, lower, upper) with a Student-t 95% interval on the batch spread."""
    from scipy.stats import t as tdist
    B = max(2, L // 2)
    nb = n // B
    if nb < 3:
        nb = 3; B = n // 3
    v = fn(np.arange(n))
    vals = np.array([fn(np.arange(b * B, (b + 1) * B)) for b in range(nb)], dtype=float)
    se = np.nanstd(vals, ddof=1, axis=0) / np.sqrt(nb)
    h = tdist.ppf(0.975, nb - 1) * se
    return v, v - h, v + h


def fmt(t):
    v, lo, hi = t
    return f"{v:+.3f} [{lo:+.3f}, {hi:+.3f}]" if np.ndim(v) == 0 else str(np.round(v, 3))


# ------------------------------------------------------------------ the analyses
def analyse(S, label, ranges_shell, ranges_x, nmax_mean, lx=4.0):
    R = {}
    Nz, Nx = int(S["Nz"]), int(S["Nx"])
    nf = S["T_shell"].shape[0]
    Kb = np.arange(S["T_shell"].shape[1]).astype(float)               # K/pi
    cnt = S["count_sin"].astype(float)
    corr = np.where(cnt > 0, np.pi * Kb / np.maximum(cnt, 1), np.nan)   # mode-count normalisation
    Tsh = S["T_shell"].astype(float) * corr[None, :]
    tau, _ = integral_time(np.array([fit(Kb, s, 5, 15) for s in Tsh]))
    L = max(2, int(np.ceil(4 * tau)))
    R["block"] = L; R["n_batches"] = max(3, nf // max(2, L // 2)); R["nframes"] = nf; R["Nz"] = Nz; R["Nx"] = Nx
    print(f"\n=== {label}: {nf} frames, grid {Nz} x {Nx}, correlation time {tau:.0f} frames -> {R['n_batches']} batches ===")

    # ---- 1. upper branch: mean-profile modes
    n = np.arange(1, Nz).astype(float)
    Em = S["T_mean"].astype(float)
    prof = S["prof_T"].astype(float).mean(0)                     # theta on the grid, j = 0 bottom
    zg = np.arange(Nz + 1) / Nz
    thp = prof + EPS * zg
    lo_ = (zg >= 0.05) & (zg <= 0.25); hi_ = (zg >= 0.75) & (zg <= 0.95)
    J0 = np.polyval(np.polyfit(zg[lo_], thp[lo_], 1), 0.0)
    J1 = np.polyval(np.polyfit(zg[hi_], thp[hi_], 1), 1.0)
    R["J0"], R["J1"] = J0, J1
    pred = 2 / (n ** 2 * np.pi ** 2) * (J0 - (-1) ** n * J1) ** 2
    ev, od = (n % 2 == 0), (n % 2 == 1)
    nm = n <= nmax_mean
    R["mean_slope_even"] = boot(nf, L, lambda i: fit(n[ev], Em[i].mean(0)[ev], 2, nmax_mean))
    R["mean_slope_odd"] = boot(nf, L, lambda i: fit(n[od], Em[i].mean(0)[od], 1, nmax_mean))
    R["mean_amp_ratio_even_n2_10"] = boot(nf, L, lambda i: np.sqrt(Em[i].mean(0)[ev & (n <= 10)] / pred[ev & (n <= 10)]).mean())
    R["mean_amp_ratio_odd_n1_9"] = boot(nf, L, lambda i: np.sqrt(Em[i].mean(0)[od & (n <= 9)] / pred[od & (n <= 9)]).mean())
    R["even_odd_ratio_n_le_20"] = boot(nf, L, lambda i: Em[i].mean(0)[ev & (n <= 20)].sum() / Em[i].mean(0)[od & (n <= 20)].sum())
    R["even_odd_pred"] = pred[ev & (n <= 20)].sum() / pred[od & (n <= 20)].sum()
    R["mode_ratio_n1_12"] = np.sqrt(Em.mean(0)[:12] / pred[:12])
    # BL thickness from the roll-off: fit E_n = 2/(n pi)^2 [J0 g(n pi d0) - (-1)^n J1 g(n pi d1)]^2, g = 1/(1+q^2)
    from scipy.optimize import least_squares
    Emm = Em.mean(0)
    sel = (n <= nmax_mean) & (Emm > 0)

    def model(q, nn):
        d0, d1 = np.abs(q)
        return 2 / (nn ** 2 * np.pi ** 2) * (J0 / (1 + (nn * np.pi * d0) ** 2) - (-1) ** nn * J1 / (1 + (nn * np.pi * d1) ** 2)) ** 2

    best = None
    for g0 in (1e-3, 5e-3, 2e-2):
        r = least_squares(lambda q: np.log(model(q, n[sel])) - np.log(Emm[sel]), [g0, g0])
        if best is None or r.cost < best.cost:
            best = r
    R["delta_spec"] = np.abs(best.x)                                       # bottom, top
    # direct estimates: slope at the plate from the first data point, and the rms peak
    m2 = S["mom_T"][:, 0].astype(float).mean(0)
    rms = np.sqrt(m2)
    jb = int(np.argmax(rms[: Nz // 4])); jt = Nz // 4 * 3 + int(np.argmax(rms[Nz // 4 * 3:]))
    R["delta_rms"] = (zg[jb], 1 - zg[jt])
    R["rms_profile"] = rms; R["prof_T"] = prof
    dz = 1.0 / Nz
    g0 = abs(thp[1] - thp[0]) / dz; g1 = abs(thp[-1] - thp[-2]) / dz
    R["delta_slope_upper_bound"] = (abs(J0) / g0, abs(J1) / g1)
    R["Nu_lower_bound"] = (g0 / EPS, g1 / EPS)
    print(f"[upper branch] jumps J0 (bottom) = {J0:+.4f} = {J0 / EPS:+.2f} eps, J1 (top) = {J1:+.4f} = {J1 / EPS:+.2f} eps "
          f"(Boussinesq: -0.5, +0.5 eps)")
    print(f"   even-n slope n<= {nmax_mean}: {fmt(R['mean_slope_even'])}   odd-n slope: {fmt(R['mean_slope_odd'])}   (theory -2)")
    print(f"   amplitude / jump-model prediction: even {fmt(R['mean_amp_ratio_even_n2_10'])}, odd {fmt(R['mean_amp_ratio_odd_n1_9'])}")
    print("   amplitude / prediction, n = 1..12: " + " ".join(f"{v:.2f}" for v in R["mode_ratio_n1_12"]))
    print(f"   even/odd energy (n<=20): {fmt(R['even_odd_ratio_n_le_20'])}  predicted from jumps {R['even_odd_pred']:.2f}")
    print(f"   BL thickness from roll-off fit: bottom {R['delta_spec'][0]:.4f}, top {R['delta_spec'][1]:.4f} "
          f"(pixel = {dz:.4f}); rms peak at {R['delta_rms'][0]:.4f} / {R['delta_rms'][1]:.4f}; "
          f"slope bound {R['delta_slope_upper_bound'][0]:.4f} / {R['delta_slope_upper_bound'][1]:.4f}")
    print(f"   wall gradient / eps (= Nu if resolved; lower bound otherwise): bottom {R['Nu_lower_bound'][0]:.1f}, top {R['Nu_lower_bound'][1]:.1f}")
    R["E_mean"] = Emm; R["E_mean_pred"] = pred
    # share of theta' variance in the mean profile
    R["var_mean_frac"] = boot(nf, L, lambda i: Em[i].mean(0).sum() / (Em[i].mean(0).sum() + S["T_z"][i].astype(float).mean(0).sum()))
    print(f"   fraction of theta' variance in the mean profile: {fmt(R['var_mean_frac'])}")

    # ---- 2. lower branch: fluctuation shells, 1D x and z, isotropy
    R["shell"] = Tsh.mean(0); R["K"] = Kb
    for lo, hi in ranges_shell:
        R[f"shell_{lo}_{hi}"] = boot(nf, L, lambda i: fit(Kb, Tsh[i].mean(0), lo, hi))
    Ex = S["T_x"].astype(float) / 0.5                                 # density per unit k/pi
    kx = np.arange(1, Ex.shape[1] + 1) * 2 / lx
    Ez = S["T_z"].astype(float) / 1.0
    kz = np.arange(1, Ez.shape[1] + 1).astype(float)
    R["Ex"], R["kx"], R["Ez"], R["kz"] = Ex.mean(0), kx, Ez.mean(0), kz
    for lo, hi in ranges_shell:
        R[f"x_{lo}_{hi}"] = boot(nf, L, lambda i: fit(kx, Ex[i].mean(0), lo, hi))
        R[f"z_{lo}_{hi}"] = boot(nf, L, lambda i: fit(kz, Ez[i].mean(0), lo, hi))
    # old convention: k = whole waves across width 4 (= m); old band k = 10-30
    R["x_old_10_30"] = boot(nf, L, lambda i: fit(np.arange(1, Ex.shape[1] + 1), S["T_x"][i].astype(float).mean(0), 10, 30))
    # Hann-window and DCT variants of the z-spectrum
    jb0, jb1 = S["jb"] if S.has("jb") else (int(.05 * Nz), int(.95 * Nz))
    Lw = (jb1 - jb0) / Nz
    if S.has("T_zhann"):
        Eh = S["T_zhann"].astype(float)
        kh = np.arange(Eh.shape[1]) * 2 / Lw
        R["Ez_hann"], R["kz_hann"] = Eh.mean(0), kh
    Ezc = S["Tc_z"].astype(float)
    for lo, hi in ranges_shell[:2]:
        if S.has("T_zhann"):
            R[f"zhann_{lo}_{hi}"] = boot(nf, L, lambda i: fit(kh, Eh[i].mean(0), lo, hi))
        R[f"zcos_{lo}_{hi}"] = boot(nf, L, lambda i: fit(np.arange(Ezc.shape[1]).astype(float), Ezc[i].mean(0), lo, hi))
    # isotropy: E_z / E_x at equal k (k/pi integer -> m = k lx/2)
    def iso(i, lo, hi):
        ex, ez = Ex[i].mean(0), Ez[i].mean(0)
        ks = np.arange(lo, hi + 1)
        mx = (ks * lx / 2).astype(int) - 1
        ok = (mx < len(ex)) & (ks - 1 < len(ez))
        return np.exp(np.mean(np.log(ez[ks[ok] - 1] / ex[mx[ok]])))
    for lo, hi in ((1, 3), (3, 10), (10, 40)):
        R[f"iso_z_over_x_{lo}_{hi}"] = boot(nf, L, lambda i: iso(i, lo, hi))
    # bulk only: x-spectra of the bands z = 0.1-0.9 against the Hann z-spectrum of z = 0.05-0.95
    if S.has("xband_T") and S.has("T_zhann"):
        xb = S["xband_T"].astype(float)
        nbd = xb.shape[1]
        # weight each height band by the mean squared Hann weight of the z-window over that band
        be = np.linspace(0, 1, nbd + 1); zz = np.linspace(0, 1, 2001)
        ww = np.where((zz >= jb0 / Nz) & (zz <= jb1 / Nz), np.sin(np.pi * (zz - jb0 / Nz) / (Lw)) ** 2, 0) ** 2
        wb = np.array([ww[(zz >= be[b]) & (zz < be[b + 1])].mean() for b in range(nbd)])
        Exb = (xb * wb[None, :, None]).sum(1) / wb.sum() / 0.5
        def iso_b(i, lo, hi):
            ex, eh = Exb[i].mean(0), Eh[i].mean(0) / (2 / Lw)
            ks = np.arange(lo, hi + 1, dtype=float)
            return np.exp(np.mean(np.log(np.interp(ks, kh, eh) / np.interp(ks, kx, ex))))
        for lo, hi in ((3, 10), (10, 40)):
            R[f"iso_bulk_{lo}_{hi}"] = boot(nf, L, lambda i: iso_b(i, lo, hi))
        R["Ex_bulk"] = Exb.mean(0)
    sc = S["sect_count_sin"].astype(float)
    Tsec = S["T_sect"].astype(float)
    def sect(i, lo, hi):
        e = Tsec[i].mean(0)
        ks = (Kb >= lo) & (Kb <= hi)
        return (e[2, ks].sum() / sc[2, ks].sum()) / (e[0, ks].sum() / sc[0, ks].sum())
    for lo, hi in ((2, 5), (5, 15), (15, 40)):
        R[f"sector_vert_over_horiz_{lo}_{hi}"] = boot(nf, L, lambda i: sect(i, lo, hi))
    print("[lower branch = fluctuations]")
    for lo, hi in ranges_shell:
        print(f"   K/pi {lo:>3}-{hi:<3}: shell {fmt(R[f'shell_{lo}_{hi}'])} | 1D x {fmt(R[f'x_{lo}_{hi}'])} | 1D z {fmt(R[f'z_{lo}_{hi}'])}")
    for lo, hi in ranges_shell[:2]:
        extra = f"Hann {fmt(R[f'zhann_{lo}_{hi}'])}" if S.has("T_zhann") else ""
        print(f"   z-slope variants K/pi {lo}-{hi}: DCT {fmt(R[f'zcos_{lo}_{hi}'])} {extra}")
    print(f"   old convention x-slope k=10-30 (all rows, new grid): {fmt(R['x_old_10_30'])}")
    for lo, hi in ((1, 3), (3, 10), (10, 40)):
        print(f"   isotropy E_z/E_x at k/pi {lo}-{hi}: {fmt(R[f'iso_z_over_x_{lo}_{hi}'])}")
    for lo, hi in ((3, 10), (10, 40)):
        if f"iso_bulk_{lo}_{hi}" in R:
            print(f"   isotropy in the bulk only (both weighted by the same Hann window in z, 0.05-0.95) at k/pi {lo}-{hi}: {fmt(R[f'iso_bulk_{lo}_{hi}'])}")
    for lo, hi in ((2, 5), (5, 15), (15, 40)):
        print(f"   per-mode energy, vertical sector / horizontal sector, K/pi {lo}-{hi}: {fmt(R[f'sector_vert_over_horiz_{lo}_{hi}'])}")

    # ---- 3. entropy (Verma & Chatterjee 2022) on 2D shells, K/pi <= 75 (old x k <= 150)
    def S_of(E):
        p = E / E.sum(); p = p[p > 0]
        return -(p * np.log2(p)).sum()
    KM = min(75, len(Kb) - 1)
    raw = S["T_shell"].astype(float)
    def S_all(j):
        e = raw[j, 1:KM + 1].copy(); e[:KM] += Em[j, :KM]
        return S_of(e)
    s1 = np.array([S_of(raw[j, 1:KM + 1]) for j in range(nf)])
    s2 = np.array([S_all(j) for j in range(nf)])
    s3 = np.array([S_of(S["T_z"][j, :KM].astype(float)) for j in range(nf)])
    R["S_shell_fluct"] = boot(nf, L, lambda i: s1[i].mean())
    R["S_shell_all"] = boot(nf, L, lambda i: s2[i].mean())
    R["S_z_fluct"] = boot(nf, L, lambda i: s3[i].mean())
    print(f"[entropy, bits, K/pi 1-{KM}] 2D shells, fluctuations {fmt(R['S_shell_fluct'])}; with mean-profile modes {fmt(R['S_shell_all'])}; "
          f"1D z {fmt(R['S_z_fluct'])}  (flat = {np.log2(KM):.2f})")

    # ---- 4. T-rho: coherence and pressure share by K (cosine basis)
    if S.has("R_shell"):
        cc = S["count_cos"].astype(float) if S.has("count_cos") else None
        Xs, Ts, Rs = S["X_TR_shell"].astype(float), S["Tc_shell"].astype(float), S["R_shell"].astype(float)
        def coh(i, lo, hi):
            ks = slice(lo, hi + 1)
            return Xs[i].mean(0)[ks].sum() / np.sqrt(Ts[i].mean(0)[ks].sum() * Rs[i].mean(0)[ks].sum())
        for lo, hi in ((1, 5), (5, 15), (15, 50)):
            R[f"coh_{lo}_{hi}"] = boot(nf, L, lambda i: coh(i, lo, hi))
        R["coh_K"] = Xs.mean(0) / np.sqrt(Ts.mean(0) * Rs.mean(0))
        Xz, Tz, Rz = S["X_TR_z"].astype(float), S["Tc_z"].astype(float), S["R_z"].astype(float)
        R["coh_kz"] = Xz.mean(0) / np.sqrt(Tz.mean(0) * Rz.mean(0))
        R["Rshell"] = Rs.mean(0) * corr[None, :len(Rs[0])].ravel()[:len(Rs[0])] if False else Rs.mean(0)
        for lo, hi in ranges_shell[:2]:
            R[f"rho_shell_{lo}_{hi}"] = boot(nf, L, lambda i: fit(Kb, Rs[i].mean(0) * corr, lo, hi))
        print("[T-rho]  coherence " + "  ".join(f"K/pi {lo}-{hi}: {fmt(R[f'coh_{lo}_{hi}'])}" for lo, hi in ((1, 5), (5, 15), (15, 50))))
        print("   rho shell slopes " + "  ".join(f"{lo}-{hi}: {fmt(R[f'rho_shell_{lo}_{hi}'])}" for lo, hi in ranges_shell[:2]))
    if S.has("P_shell"):
        Ps, Es_ = S["P_shell"].astype(float), S["E_shell"].astype(float)
        R["share_total"] = boot(nf, L, lambda i: Ps[i].mean(0)[1:].sum() / (Ps[i].mean(0)[1:].sum() + Es_[i].mean(0)[1:].sum()))
        R["share_K"] = Ps.mean(0) / (Ps.mean(0) + Es_.mean(0))
        for lo, hi in ((1, 3), (5, 15), (15, 50)):
            R[f"share_{lo}_{hi}"] = boot(nf, L, lambda i: Ps[i].mean(0)[lo:hi + 1].sum() / (Ps[i].mean(0)[lo:hi + 1].sum() + Es_[i].mean(0)[lo:hi + 1].sum()))
        for lo, hi in ranges_shell[:2]:
            R[f"P_shell_{lo}_{hi}"] = boot(nf, L, lambda i: fit(Kb, Ps[i].mean(0) * corr, lo, hi))
            R[f"E_shell_{lo}_{hi}"] = boot(nf, L, lambda i: fit(Kb, Es_[i].mean(0) * corr, lo, hi))
        Pz, Ez_ = S["P_z"].astype(float), S["E_z"].astype(float)
        R["share_kz"] = Pz.mean(0) / (Pz.mean(0) + Ez_.mean(0))
        cross = np.nonzero(R["share_K"][1:] < 0.5)[0]
        R["K_equal"] = (cross[0] + 1) if len(cross) and R["share_K"][1] > 0.5 else (np.nonzero(R["share_K"][1:] > 0.5)[0][:1] + 1)
        print(f"[pressure part] share of T fluctuation energy: total {fmt(R['share_total'])}; " +
              "  ".join(f"K/pi {lo}-{hi}: {fmt(R[f'share_{lo}_{hi}'])}" for lo, hi in ((1, 3), (5, 15), (15, 50))))
        print("   P-part shell slopes " + "  ".join(f"{lo}-{hi}: {fmt(R[f'P_shell_{lo}_{hi}'])}" for lo, hi in ranges_shell[:2]) +
              " | E-part " + "  ".join(f"{lo}-{hi}: {fmt(R[f'E_shell_{lo}_{hi}'])}" for lo, hi in ranges_shell[:2]))

    # ---- 5. profiles: moments by height (skewness, flatness), up-down asymmetry
    mom = S["mom_T"].astype(float).mean(0)
    sk = mom[1] / np.maximum(mom[0], 1e-30) ** 1.5
    fl = mom[2] / np.maximum(mom[0], 1e-30) ** 2
    R["skew_z"], R["flat_z"] = sk, fl
    def zavg(a, z0, z1):
        m = (zg >= z0) & (zg <= z1)
        return a[m].mean()
    for nm_, a in (("skew", sk), ("flat", fl), ("rms", rms)):
        R[f"{nm_}_bottom"] = zavg(a, 0.03, 0.2); R[f"{nm_}_mid"] = zavg(a, 0.4, 0.6); R[f"{nm_}_top"] = zavg(a, 0.8, 0.97)
    if S.has("mom_R"):
        mr = S["mom_R"].astype(float).mean(0)
        R["skew_R_z"] = mr[1] / np.maximum(mr[0], 1e-30) ** 1.5
        R["flat_R_z"] = mr[2] / np.maximum(mr[0], 1e-30) ** 2
        R["prof_R"] = S["prof_R"].astype(float).mean(0); R["rms_R"] = np.sqrt(mr[0])
    # up-down symmetry: Boussinesq gives rms(z) = rms(1-z), skew(z) = -skew(1-z), theta'(z) = -theta'(1-z)
    R["asym_profile"] = np.sqrt(np.mean((thp + thp[::-1]) ** 2) / np.mean(thp ** 2))
    R["asym_rms"] = np.sqrt(np.mean((rms - rms[::-1]) ** 2) / np.mean(rms ** 2))
    print(f"[profiles] rms theta: bottom {R['rms_bottom']:.4f} mid {R['rms_mid']:.4f} top {R['rms_top']:.4f}; "
          f"skewness: bottom {R['skew_bottom']:+.2f} mid {R['skew_mid']:+.2f} top {R['skew_top']:+.2f}; "
          f"flatness: {R['flat_bottom']:.1f} / {R['flat_mid']:.1f} / {R['flat_top']:.1f}")
    print(f"   up-down asymmetry (0 = Boussinesq symmetric): mean profile {R['asym_profile']:.2f}, rms profile {R['asym_rms']:.2f}")

    # ---- 6. PDFs by height band (theta, rho, P, E)
    if S.has("H_T"):
        he = S["hedges"] if S.has("hedges") else np.linspace(-0.1, 0.1, 401)
        xc = 0.5 * (he[1:] + he[:-1])
        pdfs = {}
        for k in ("T", "R", "P", "E"):
            if not S.has("H_" + k):
                continue
            Hk = S["H_" + k].astype(float)
            res = []
            for b in range(Hk.shape[0]):
                h = Hk[b] / Hk[b].sum()
                mu = (h * xc).sum(); sd = np.sqrt((h * (xc - mu) ** 2).sum())
                s3 = (h * (xc - mu) ** 3).sum() / sd ** 3; s4 = (h * (xc - mu) ** 4).sum() / sd ** 4
                y = (xc - mu) / sd; dens = h / (he[1] - he[0]) * sd
                tails = []
                for sgn in (-1, 1):
                    m = (sgn * y >= 2) & (sgn * y <= 5) & (dens > 0)
                    tails.append(-np.polyfit(sgn * y[m], np.log(dens[m]), 1)[0] if m.sum() > 4 else np.nan)
                res.append((sd, s3, s4, tails[0], tails[1]))
            pdfs[k] = np.array(res)
        R["pdf_stats"] = pdfs
        print("[PDFs by height band, bottom -> top]  T: skew " + " ".join(f"{v:+.2f}" for v in pdfs["T"][:, 1]) +
              " | flat " + " ".join(f"{v:.1f}" for v in pdfs["T"][:, 2]))
        print("   T exponential tail rates (cold side, hot side), mid band: %.2f, %.2f (Gaussian tails: steeper and growing)" %
              tuple(pdfs["T"][len(pdfs['T']) // 2, 3:5]))
        for k in ("R", "P", "E"):
            if k in pdfs:
                print(f"   {k}: skew " + " ".join(f"{v:+.2f}" for v in pdfs[k][:, 1]) + " | flat " + " ".join(f"{v:.1f}" for v in pdfs[k][:, 2]))

    # ---- 7. structure functions (bulk), x and z
    RS = S["RS"] if S.has("RS") else (S.d["RS"] if "RS" in S.d.files else None)
    if S.has("sfx_T"):
        rx = RS * lx / Nx; rz = RS / Nz
        for fld in ("T", "R"):
            if not S.has("sfx_" + fld):
                continue
            sx, sz = S["sfx_" + fld].astype(float), S["sfz_" + fld].astype(float)
            nfs = sx.shape[0]; Ls = max(2, L // 2)
            for nm_, s_, r_ in (("x", sx, rx), ("z", sz, rz)):
                R[f"sf{nm_}_{fld}"] = np.nanmean(s_, 0); R[f"r{nm_}"] = r_
                R[f"zeta2_{nm_}_{fld}"] = boot(nfs, Ls, lambda i: fit(r_, np.nanmean(s_[i], 0)[:, 1], 0.015, 0.1))
                R[f"flat_r_{nm_}_{fld}"] = np.nanmean(s_, 0)[:, 3] / np.nanmean(s_, 0)[:, 1] ** 2
                # ESS: exponents relative to <|d|^3>
                S3 = np.nanmean(s_, 0)[:, 2]
                m = (r_ >= 0.01) & (r_ <= 0.15) & np.isfinite(S3)
                R[f"ess_{nm_}_{fld}"] = [np.polyfit(np.log(S3[m]), np.log(np.nanmean(s_, 0)[m, p]), 1)[0] if m.sum() > 2 else np.nan
                                         for p in range(6)]
            print(f"[structure functions, {fld}] zeta2 (S2 ~ r^zeta2, r 0.015-0.1): x {fmt(R[f'zeta2_x_{fld}'])}, z {fmt(R[f'zeta2_z_{fld}'])} "
                  f"(OC 2/3, BO 2/5) ; flatness at smallest/largest r (x): {R[f'flat_r_x_{fld}'][0]:.1f} / {np.nanmax(R[f'flat_r_x_{fld}'][-4:]):.1f}")
            print("   ESS zeta_p/zeta_3, p=1..6 (x): " + " ".join(f"{v:.2f}" for v in R[f'ess_x_{fld}']) + "  (K41: 0.33 0.67 1 1.33 1.67 2)")

    # ---- 8. time correlations of modes
    if S.has("cT") and nf >= 60:
        cT = S["cT"]
        nn = np.arange(1, cT.shape[1] + 1); mm = np.arange(cT.shape[2])
        Kmode = np.hypot(mm[None, :] * 2 / lx, nn[:, None])
        a = cT - cT.mean(0, keepdims=True)
        maxlag = min(nf // 2, 300)
        var = (np.abs(a) ** 2).mean(0)
        taus = np.full(Kmode.shape, np.nan)
        F = np.fft.fft(a, n=2 * nf, axis=0)
        acfc = np.fft.ifft(np.abs(F) ** 2, axis=0)[:maxlag] / (nf - np.arange(maxlag))[:, None, None]
        acfc = acfc / np.maximum(var[None], 1e-40)
        def tau_of(acf):
            out = np.full(Kmode.shape, np.nan)
            for ii in range(Kmode.shape[0]):
                for jj in range(Kmode.shape[1]):
                    r_ = acf[:, ii, jj]
                    k = np.argmax(r_ < np.exp(-1))
                    if k > 0:
                        out[ii, jj] = (k - 1 + (r_[k - 1] - np.exp(-1)) / (r_[k - 1] - r_[k])) * DT
            return out
        bins = np.unique(np.round(np.logspace(0, np.log10(Kmode.max()), 25)))
        flu = mm[None, :] >= 1
        for nm_, acf in (("real", acfc.real), ("env", np.abs(acfc))):
            taus = tau_of(acf)
            tk, tv = [], []
            for b0, b1 in zip(bins[:-1], bins[1:]):
                sel = flu & (Kmode >= b0) & (Kmode < b1) & np.isfinite(taus)
                if sel.sum() >= 3:
                    tk.append(np.sqrt(b0 * b1)); tv.append(np.median(taus[sel]))
            tk, tv = np.array(tk), np.array(tv)
            R[f"tau_K_{nm_}"], R[f"tau_v_{nm_}"] = tk, tv
            R[f"tau_slope_{nm_}_3_30"] = fit(tk, tv, 3, 30)
            R[f"tau_mean_modes_{nm_}"] = taus[:, 0]
            print(f"[time, {nm_}] decorrelation time (1/e): tau(K/pi=3) ~ {np.interp(3, tk, tv):.3f}, tau(K/pi=30) ~ {np.interp(30, tk, tv):.3f}; "
                  f"tau ~ K^{R[f'tau_slope_{nm_}_3_30']:+.2f} (K/pi 3-30)")
        # mean horizontal drift speed from the phase rotation of low modes: omega = d(phase)/dt, c = omega / k_x
        ph = np.angle(acfc[1:6])                                   # lags 1..5
        om = np.nanmedian(ph / (np.arange(1, 6)[:, None, None] * DT), axis=0)
        kxm = 2 * np.pi * mm[None, :] / lx
        sel = (mm[None, :] >= 1) & (mm[None, :] <= 20) & (nn[:, None] <= 10)
        R["drift_speed"] = float(np.nanmedian((om / kxm)[sel]))
        print(f"   apparent horizontal drift speed from phase rotation (m 1-20, n 1-10): {R['drift_speed']:+.3f} (length/time)")
        taus = tau_of(acfc.real)
        R["tau_mean_n1_n2"] = taus[:2, 0]
        print(f"   mean-profile modes n=1,2: tau = {taus[0, 0]:.3f}, {taus[1, 0]:.3f} (nan = longer than {maxlag * DT:.1f})")
    return R


if __name__ == "__main__":
    tag = sys.argv[1]
    if tag == "3d":
        out = {}
        for face in ("wall_xz", "wall_yz", "wall_back1", "wall_back2"):
            S = Src(f"{CACHE}/pass5_3d.npz", face + "__")
            out[face] = analyse(S, f"3D Ra1e8 {face}", ((3, 8), (5, 15), (10, 30)), None, nmax_mean=20)
        np.save(f"{CACHE}/analyse4_3d.npy", out, allow_pickle=True)
    else:
        S = Src(f"{CACHE}/pass4_{tag}.npz")
        rs = ((3, 8), (5, 15), (10, 40), (40, 100))
        R = analyse(S, tag, rs, None, nmax_mean=20)
        np.save(f"{CACHE}/analyse4_{tag}.npy", R, allow_pickle=True)
