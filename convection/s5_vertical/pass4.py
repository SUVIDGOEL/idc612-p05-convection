r"""Fourth pass over the 2D movies: full (k_x, k_z) analysis with the mirror trick, plus physical-space statistics.

Fields on a grid z_j = j/Nz (j = 0 bottom plate .. Nz top plate), x_i = 4 i/Nx (periodic):
  theta' = (T - T_A) + eps z    zero at both plates -> sine series in z (DST-I)       [main]
  varrho = rho - rho_A          not fixed at plates -> cosine series in z (DCT-I)
  P, E   = pressure and entropy parts of the temperature, in temperature units:
           P = T_A (g-1)/g (ln p)',   E = T_A (s/c_v)'/g,   P + E = T_A (ln T)' ~ theta'      (DCT-I)
Per frame we store shell spectra (fluctuations, m >= 1), the mean-profile modes (m = 0), 1D x and z spectra,
three angular sectors, T-rho and P-E co-spectra, profiles and moments, PDFs by height, structure functions
in x and z (bulk), and complex mode amplitudes (low modes) for time correlations.

usage: python -m convection.s5_vertical.pass4 TAG [zshift_px] [every]
  zshift_px moves both plates by that many pixels (sensitivity test); every = use every n-th frame.
"""
from convection.config import CACHE
import sys, json, time
import av
import numpy as np
from scipy.fft import rfft, dst, dct
from convection.s2_density.pass2 import FILES, setup, invert_rho, yuv, TLIM
from convection.s1_x_spectra.spectra_pass import overlay_mask
from convection.common import vspec as V
class L32(list):
    """List that stores float32 / complex64 copies (keeps memory low on long movies)."""
    def append(self, v):
        a = np.asarray(v)
        super().append(a.astype(np.complex64 if np.iscomplexobj(a) else np.float32))


EPS, G, D = 0.1, 1.3, 0.5
ZOOM_X0 = {"Ra1e12": 3.0, "Ra1e15": 2.0}
RS = np.array([1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256])
NBAND = 10
HB, HR = 400, 0.1                         # histogram bins and half-range for fluctuations
NC_M, NC_N = 129, 64                      # complex amplitudes kept: m = 0..128, n = 1..64 (sine) / 0..63 (cos)


def panel(tag, nm, geo, st_std, zshift):
    g = geo[tag][nm]
    gT = geo[tag]["T"]
    w = 0.5 * (g["widths"][2] + g["widths"][3])
    z1, z0 = g["z1"] + zshift, g["z0"] - zshift            # zshift > 0: plates moved outward
    H = z0 - z1
    r0 = int(np.ceil(g["z1"] + w / 2 + 1)); r1 = int(np.floor(g["z0"] - w / 2 - 1))
    c0 = int(np.ceil(gT["x_left"] + w / 2 + 1)); c1 = int(np.floor(gT["x_right"] - w / 2 - 1))
    rows = np.arange(r0, r1 + 1)
    zr = (z0 - rows) / H
    W, xl = gT["W"], gT["x_left"]
    m = overlay_mask(st_std[r0:r1 + 1, c0:c1 + 1])
    box = None
    if tag in ZOOM_X0:
        x0 = ZOOM_X0[tag]
        box = (g["z1"] - r0, g["z1"] + 0.25 * H - r0, xl + x0 * W / 4 - c0, xl + (x0 + .25) * W / 4 - c0)
    mask = V.clean_overlay_mask(m, box)
    return dict(r0=r0, r1=r1, c0=c0, c1=c1, zr=zr, mask=mask, plan=V.fill_plan(mask),
                cref=int(round(xl)), H=H)


def place(F, P, Nx):
    """Data columns -> periodic grid of Nx columns, gap filled across the wrap."""
    nr = F.shape[0]
    out = np.empty((nr, Nx))
    i0 = P["c0"] - P["cref"]; i1 = P["c1"] - P["cref"]
    out[:, i0:i1 + 1] = F
    gidx, wl, wf = P["gap"]
    out[:, gidx] = F[:, -1:] * wl[None, :] + F[:, :1] * wf[None, :]
    return out


def run(tag, zshift=0.0, every=1):
    geo = json.load(open(f"{CACHE}/geom2d.json"))
    st = np.load(f"{CACHE}/tstats_{tag}.npz")
    S = setup(tag)
    lut_T = S["lut_T"]
    Nz = int(round(geo[tag]["T"]["z0"] - geo[tag]["T"]["z1"]))
    Nx = int(round(geo[tag]["T"]["W"]))
    PT = panel(tag, "T", geo, st["std"], zshift)
    PR = panel(tag, "rho", geo, st["std"], zshift)
    for P in (PT, PR):
        P["gap"] = V.gap_plan(P["c0"] - P["cref"], P["c1"] - P["cref"], Nx)
    MT, _, _ = V.zinterp_matrix(PT["zr"], Nz, walls=(0.0, 0.0))     # theta' = 0 at plates
    MR, _, _ = V.zinterp_matrix(PR["zr"], Nz)                        # held end values
    zg = np.arange(Nz + 1) / Nz
    TA, RA = 1 - D * zg, (1 - D * zg) ** (1 / (G - 1))
    Ssin, Scos = V.Shells(Nz, Nx, "sin"), V.Shells(Nz, Nx, "cos")
    nb = Ssin.nb + 1
    xw = V.xw(Nx)
    jb0, jb1 = int(round(0.05 * Nz)), int(round(0.95 * Nz))           # bulk rows for SF and Hann
    bedges = np.linspace(1, Nz, NBAND + 1).astype(int)                # bands of interior grid rows
    band_of_row = np.clip(np.searchsorted(bedges, np.arange(Nz + 1), side="right") - 1, 0, NBAND - 1)
    hedges = np.linspace(-HR, HR, HB + 1)
    raw_edges = np.linspace(-0.12, 0.02, HB + 1)
    print(f"{tag}: grid Nz={Nz} Nx={Nx}; T rows {PT['r0']}-{PT['r1']} cols {PT['c0']}-{PT['c1']} "
          f"(gap {len(PT['gap'][0])} cols, masked {int(PT['mask'].sum())} px); rho rows {PR['r0']}-{PR['r1']} "
          f"(masked {int(PR['mask'].sum())}); zshift {zshift}", flush=True)

    keys = ("T_mean", "T_shell", "T_sect", "T_x", "T_z", "Tc_shell", "Tc_z", "T_zhann",
            "R_mean", "R_shell", "R_sect", "R_x", "R_z", "X_TR_shell", "X_TR_z", "X_TR_x",
            "P_shell", "E_shell", "X_PE_shell", "P_z", "E_z", "P_x", "E_x", "X_PE_z", "X_PE_x",
            "prof_T", "prof_R", "prof_s", "prof_lnp", "mom_T", "mom_R", "mom_P", "mom_E",
            "sfx_T", "sfz_T", "sfx_R", "sfz_R", "sat", "cT", "cR", "xband_T", "xband_R")
    out = {k: L32() for k in keys}
    acc = {k: 0.0 for k in ("T2d", "R2d", "Tc2d", "XTR2d", "P2d", "E2d", "XPE2d")}
    H = {k: np.zeros((NBAND, HB)) for k in ("T", "R", "P", "E")}
    Hraw = np.zeros((NBAND, HB))
    samples = {}
    nfr = 0
    t0 = time.time()
    c = av.open(FILES[tag])
    for i, fr in enumerate(c.decode(video=0)):
        if i % every:
            continue
        Y, Cb, Cr = yuv(fr)
        # ---- invert the two panels on their data pixels
        yT = Y[PT["r0"]:PT["r1"] + 1, PT["c0"]:PT["c1"] + 1]
        th = TLIM[0] + (TLIM[1] - TLIM[0]) * lut_T[yT]
        th = V.apply_fill(th, PT["plan"])
        sl = (slice(PR["r0"], PR["r1"] + 1), slice(PR["c0"], PR["c1"] + 1))
        vr = V.apply_fill(invert_rho(Y[sl], Cb[sl], Cr[sl], S["lut_pos"], S["lut_neg"]).astype(float), PR["plan"])
        pos = lut_T[yT]
        out["sat"].append([(pos <= 0.005).mean(), (pos >= 0.995).mean(), (np.abs(vr) >= 0.099).mean()])
        # ---- onto the grid
        thp = place(th + EPS * PT["zr"][:, None], PT, Nx)
        GT = MT @ thp                                          # theta', zero at plates
        GR = MR @ place(vr, PR, Nx)
        th_g = GT - EPS * zg[:, None]                          # theta = T - T_A on the grid
        # ---- theta': sine series
        A = rfft(GT[1:-1], axis=1)
        C = dst(A.real, type=1, axis=0) + 1j * dst(A.imag, type=1, axis=0)
        E = np.abs(C) ** 2 * xw[None, :] / (2.0 * Nx ** 2 * Nz ** 2)
        acc["T2d"] = acc["T2d"] + E
        out["T_mean"].append(E[:, 0]); out["T_shell"].append(Ssin.shell(E)); out["T_sect"].append(Ssin.sectors(E))
        out["T_x"].append(E[:, 1:].sum(0)); out["T_z"].append(E[:, 1:].sum(1))
        out["cT"].append(C[:NC_N, :NC_M].astype(np.complex64))
        # x-spectra by height band (rows of the grid), as in the earlier passes
        Pr = np.abs(A[:, 1:]) ** 2 * xw[None, 1:] / Nx ** 2
        out["xband_T"].append(np.stack([Pr[bedges[b] - 1:bedges[b + 1] - 1].mean(0) for b in range(NBAND)]))
        # ---- cosine series for theta, rho and the parts
        Ac = rfft(GT, axis=1)
        Cc = dct(Ac.real, type=1, axis=0) + 1j * dct(Ac.imag, type=1, axis=0)
        AR = rfft(GR, axis=1)
        CR = dct(AR.real, type=1, axis=0) + 1j * dct(AR.imag, type=1, axis=0)
        wn = np.ones(Nz + 1); wn[[0, -1]] = 0.5
        nrm = xw[None, :] * wn[:, None] / (2.0 * Nx ** 2 * Nz ** 2)
        Ec, ER = np.abs(Cc) ** 2 * nrm, np.abs(CR) ** 2 * nrm
        XTR = (Cc * np.conj(CR)).real * nrm
        acc["Tc2d"] = acc["Tc2d"] + Ec; acc["R2d"] = acc["R2d"] + ER; acc["XTR2d"] = acc["XTR2d"] + XTR
        out["Tc_shell"].append(Scos.shell(Ec)); out["Tc_z"].append(Ec[:, 1:].sum(1))
        out["R_mean"].append(ER[:, 0]); out["R_shell"].append(Scos.shell(ER)); out["R_sect"].append(Scos.sectors(ER))
        out["R_x"].append(ER[:, 1:].sum(0)); out["R_z"].append(ER[:, 1:].sum(1))
        out["X_TR_shell"].append(Scos.shell(XTR)); out["X_TR_z"].append(XTR[:, 1:].sum(1)); out["X_TR_x"].append(XTR[:, 1:].sum(0))
        out["cR"].append(CR[:NC_N, :NC_M].astype(np.complex64))
        PrR = np.abs(AR[:, 1:]) ** 2 * xw[None, 1:] / Nx ** 2
        out["xband_R"].append(np.stack([PrR[bedges[b]:bedges[b + 1]].mean(0) for b in range(NBAND)]))
        # pressure and entropy parts (temperature units)
        Tt = TA[:, None] + th_g
        Rt = np.maximum(RA[:, None] + GR, 0.02)
        lnT, lnr = np.log(Tt), np.log(Rt)
        lnp = lnT + lnr
        s = lnT - (G - 1) * lnr
        Pf = TA[:, None] * ((G - 1) / G) * lnp
        Ef = TA[:, None] * (s / G)
        AP, AE = rfft(Pf, axis=1), rfft(Ef, axis=1)
        CP = dct(AP.real, type=1, axis=0) + 1j * dct(AP.imag, type=1, axis=0)
        CE = dct(AE.real, type=1, axis=0) + 1j * dct(AE.imag, type=1, axis=0)
        EP, EE = np.abs(CP) ** 2 * nrm, np.abs(CE) ** 2 * nrm
        XPE = (CP * np.conj(CE)).real * nrm
        acc["P2d"] = acc["P2d"] + EP; acc["E2d"] = acc["E2d"] + EE; acc["XPE2d"] = acc["XPE2d"] + XPE
        out["P_shell"].append(Scos.shell(EP)); out["E_shell"].append(Scos.shell(EE)); out["X_PE_shell"].append(Scos.shell(XPE))
        out["P_z"].append(EP[:, 1:].sum(1)); out["E_z"].append(EE[:, 1:].sum(1)); out["X_PE_z"].append(XPE[:, 1:].sum(1))
        out["P_x"].append(EP[:, 1:].sum(0)); out["E_x"].append(EE[:, 1:].sum(0)); out["X_PE_x"].append(XPE[:, 1:].sum(0))
        # ---- physical space: profiles, moments, PDFs
        fT = th_g - th_g.mean(1, keepdims=True)
        fR = GR - GR.mean(1, keepdims=True)
        fP = Pf - Pf.mean(1, keepdims=True)
        fE = Ef - Ef.mean(1, keepdims=True)
        out["prof_T"].append(th_g.mean(1)); out["prof_R"].append(GR.mean(1))
        out["prof_s"].append(s.mean(1)); out["prof_lnp"].append(lnp.mean(1))
        for nm_, f_ in (("T", fT), ("R", fR), ("P", fP), ("E", fE)):
            f2 = f_ * f_
            out["mom_" + nm_].append(np.stack([f2.mean(1), (f2 * f_).mean(1), (f2 * f2).mean(1)]))
            ib = np.clip(((f_[1:-1] + HR) / (2 * HR) * HB).astype(int), 0, HB - 1)
            H[nm_] += np.bincount((band_of_row[1:-1, None] * HB + ib).ravel(), minlength=NBAND * HB).reshape(NBAND, HB)
        ib = np.clip(((th_g[1:-1] + 0.12) / 0.14 * HB).astype(int), 0, HB - 1)
        Hraw += np.bincount((band_of_row[1:-1, None] * HB + ib).ravel(), minlength=NBAND * HB).reshape(NBAND, HB)
        # ---- structure functions in the bulk (every 2nd stored frame)
        if nfr % 2 == 0:
            out["sfx_T"].append(V.increments(fT, RS, 1, rows=(jb0, jb1)))
            out["sfz_T"].append(V.increments(fT, RS, 0, rows=(jb0, jb1)))
            out["sfx_R"].append(V.increments(fR, RS, 1, rows=(jb0, jb1)))
            out["sfz_R"].append(V.increments(fR, RS, 0, rows=(jb0, jb1)))
        out["T_zhann"].append(V.zspec_hann(fT, jb0, jb1))
        if i in (5, 300, 600, 100):
            samples[f"T{i}"] = th_g.astype(np.float32); samples[f"R{i}"] = GR.astype(np.float32)
        nfr += 1
        if nfr == 3:
            print(f"   {(time.time() - t0) / 3:.2f} s/frame", flush=True)
    c.close()
    arr = {}
    for k, v in out.items():
        if len(v):
            a = np.asarray(v)
            arr[k] = a.astype(np.complex64 if np.iscomplexobj(a) else np.float32)
    for k, v in acc.items():
        arr["acc_" + k] = (v / nfr).astype(np.float32)
    for k, v in H.items():
        arr["H_" + k] = v
    tagz = "" if zshift == 0 else f"_z{zshift:+.1f}"
    np.savez_compressed(f"{CACHE}/pass4_{tag}{tagz}.npz", **arr, **samples, H_raw=Hraw, hedges=hedges,
                        raw_edges=raw_edges, zg=zg, Nz=Nz, Nx=Nx, RS=RS, bedges=bedges, jb=np.array([jb0, jb1]),
                        sect_count_sin=Ssin.sector_counts(), sect_count_cos=Scos.sector_counts(),
                        count_sin=Ssin.count, count_cos=Scos.count, nfr=nfr, every=every,
                        maskT=PT["mask"], maskR=PR["mask"], rowsT=np.array([PT["r0"], PT["r1"]]),
                        rowsR=np.array([PR["r0"], PR["r1"]]))
    print(f"{tag}: {nfr} frames, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    tag = sys.argv[1]
    zs = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    ev = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    run(tag, zs, ev)
