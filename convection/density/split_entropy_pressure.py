r"""Third pass: split the temperature into an ENTROPY part and a PRESSURE part.

Ideal gas, values scaled by the bottom values (checked by fit_gamma_and_D.py: A = gamma - 1, D = 0.48):
    T_A(z) = 1 - D z,   rho_A(z) = (1 - D z)^(1/(gamma - 1))
    T = T_A + theta,    rho = rho_A + varrho          (theta, varrho = the two painted panels)
    ln p   = ln T + ln rho                              (p = rho T)
    s/c_v  = ln T - (gamma - 1) ln rho                  (entropy)
    ln T   = (1/gamma) s/c_v  +  ((gamma - 1)/gamma) ln p
             \_ entropy part _/   \___ pressure part ___/
Cold, dense fluid (a thermal plume) is an entropy fluctuation. Cold, light fluid (a vortex core,
low pressure) is a pressure fluctuation. The x-spectra of the two parts are stored per band.
"""
from convection.config import CACHE
import sys, time
import av
import numpy as np
from convection.density.read_density_panel import FILES, setup, invert_rho, yuv, TLIM, NB
from convection.spectrum_1d.measure_x_spectra import make_interpolator

G = 1.3
D = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5


def spec(F):
    Fh = np.fft.rfft(F - F.mean(1, keepdims=True), axis=1)[:, 1:]
    return Fh


def run(tag):
    S = setup(tag)
    interp, _ = make_interpolator(S["mask"])
    r0, r1, c0, c1 = S["r0"], S["r1"], S["c0"], S["c1"]
    rs, cs = S["rshift"], S["cshift"]
    nr, nx = r1 - r0, c1 - c0
    z = 1 - (np.arange(nr) + 2.5) / (nr + 4)                       # crop is inset 2 px
    TA = (1 - D * z)[:, None]
    RA = ((1 - D * z) ** (1 / (G - 1)))[:, None]
    edges = np.linspace(0, nr, NB + 1).astype(int)
    bands = list(zip(edges[:-1], edges[1:]))

    keys = ("E_lnT", "E_ent", "E_prs", "C_ep", "var_ent", "var_prs", "cov_ep")
    out = {k: [] for k in keys}
    prof = {k: np.zeros(nr) for k in ("theta", "theta2", "rho", "rho2", "s", "lnp")}
    nfr = 0
    t0 = time.time()
    c = av.open(FILES[tag])
    for i, fr in enumerate(c.decode(video=0)):
        Y, Cb, Cr = yuv(fr)
        th = interp(TLIM[0] + (TLIM[1] - TLIM[0]) * S["lut_T"][Y[r0:r1, c0:c1]])
        sl = (slice(r0 + rs, r1 + rs), slice(c0 + cs, c1 + cs))
        vr = interp(invert_rho(Y[sl], Cb[sl], Cr[sl], S["lut_pos"], S["lut_neg"]))
        T = TA + th
        rho = np.maximum(RA + vr, 0.02)
        lnT, lnr = np.log(T), np.log(rho)
        lnp = lnT + lnr
        s = lnT - (G - 1) * lnr
        ent, prs = s / G, (G - 1) / G * lnp

        fT, fe, fp = spec(lnT), spec(ent), spec(prs)
        out["E_lnT"].append(np.stack([(np.abs(fT[a:b]) ** 2).mean(0) for a, b in bands]) / nx)
        out["E_ent"].append(np.stack([(np.abs(fe[a:b]) ** 2).mean(0) for a, b in bands]) / nx)
        out["E_prs"].append(np.stack([(np.abs(fp[a:b]) ** 2).mean(0) for a, b in bands]) / nx)
        out["C_ep"].append(np.stack([(fe[a:b] * np.conj(fp[a:b])).real.mean(0) for a, b in bands]) / nx)
        e0 = ent - ent.mean(1, keepdims=True); p0 = prs - prs.mean(1, keepdims=True)
        out["var_ent"].append([(e0[a:b] ** 2).mean() for a, b in bands])
        out["var_prs"].append([(p0[a:b] ** 2).mean() for a, b in bands])
        out["cov_ep"].append([(e0[a:b] * p0[a:b]).mean() for a, b in bands])

        prof["theta"] += th.mean(1); prof["theta2"] += (th ** 2).mean(1)
        prof["rho"] += vr.mean(1); prof["rho2"] += (vr ** 2).mean(1)
        prof["s"] += s.mean(1); prof["lnp"] += lnp.mean(1)
        nfr += 1
    c.close()
    arrs = {k: np.asarray(v, dtype=np.float32) for k, v in out.items()}
    profs = {"prof_" + k: v / nfr for k, v in prof.items()}
    tagD = "" if D == 0.5 else f"_D{D}"
    np.savez_compressed(f"{CACHE}/entropy_pressure_split_{tag}{tagD}.npz", **arrs, **profs, z=z,
                        band_edges=edges, D=D, gamma=G)
    print(f"{tag} D={D}: {nfr} frames, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    run(sys.argv[1])
