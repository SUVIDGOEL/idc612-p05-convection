"""3D movie (Ra = 1e8): pressure part and entropy part of the temperature, from the stored T and rho spectra.

Same split as pass3.py (2D), linearised (validated in 2D: it reproduces the exact shares 0.29/0.78/0.97):
    ln T' = theta/T_A,   ln rho' = varrho/rho_A,   ln p' = ln T' + ln rho'
    pressure part of ln T = ((g-1)/g) ln p'      entropy part = ln T' - pressure part
In temperature units (x T_A), with c = (g-1)/g and a = T_A/rho_A:
    P = c (theta + a varrho)                     E = theta/g - c a varrho
    |P|^2 = c^2 (E_T + a^2 E_R + 2 a C)          |E|^2 = E_T/g^2 + c^2 a^2 E_R - 2 (c a / g) C
E_T, E_R = x-spectra of theta and varrho, C = Re(co-spectrum). gamma = 1.3, D = 0.5 (printed on the frames).
Walls: 5 height bands, row 0 = top (z = 1). Lid: z = 1. Floor: z = 0.
"""
from convection.config import CACHE
import sys
import numpy as np
from convection.common.stats import slope, integral_time, per_frame_slopes

G, D = 1.3, 0.5
c = (G - 1) / G
TA = lambda z: 1 - D * z
RA = lambda z: (1 - D * z) ** (1 / (G - 1))


def parts(ET, ER, CO, z):
    a = TA(z) / RA(z)
    P = c**2 * (ET + a**2 * ER + 2 * a * CO)
    E = ET / G**2 + c**2 * a**2 * ER - 2 * (c * a / G) * CO
    return P, E


def blocks(n, L, nboot=1000, seed=0):
    rng = np.random.default_rng(seed); starts = np.arange(n - L + 1); nb = int(np.ceil(n / L))
    for _ in range(nboot):
        yield (rng.choice(starts, nb)[:, None] + np.arange(L)[None, :]).ravel()[:n]


d = np.load(f"{CACHE}/rho3d.npz")
WALLS = ("wall_xz", "wall_yz", "wall_back1", "wall_back2")
zc = 1 - (np.arange(5) + 0.5) / 5                      # band centres, top band first
B = np.stack([d[f"{w}_bands"].astype(float) for w in WALLS]).mean(0)   # frames x 5 bands x 3 x k
nf, nb, _, K = B.shape
Pw = np.zeros((nf, nb, K)); Ew = np.zeros((nf, nb, K))
for b in range(nb):
    Pw[:, b], Ew[:, b] = parts(B[:, b, 0], B[:, b, 1], B[:, b, 2], zc[b])

P_f, E_f = Pw.mean(1), Ew.mean(1)                     # frames x k (all heights)
share_f = lambda i: P_f[i].sum() / (P_f[i].sum() + E_f[i].sum())
tau, _ = integral_time(per_frame_slopes(B[:, :, 0].mean(1), 5, 15)); L = max(1, int(np.ceil(4 * tau)))
idx = np.arange(nf)
res = {}
for name, fn in (("share", share_f),
                 ("P_slope_k5_15", lambda i: slope(P_f[i].mean(0), 5, 15)),
                 ("E_slope_k5_15", lambda i: slope(E_f[i].mean(0), 5, 15)),
                 ("T_slope_k5_15", lambda i: slope(B[i][:, :, 0].mean(1).mean(0), 5, 15))):
    vals = np.array([fn(j) for j in blocks(nf, L)])
    res[name] = (fn(idx), *np.nanpercentile(vals, [2.5, 97.5]))
print(f"block length {L} frames")
print("WALLS  pressure share of T variance: %.3f [%.3f, %.3f]" % res["share"])
for nm in ("T", "P", "E"):
    print("WALLS  %s-part slope k=5-15: %.2f [%.2f, %.2f]" % ((nm,) + res[f"{nm}_slope_k5_15"]))
print("WALLS  share by height (top -> bottom):",
      " ".join("z=%.1f:%.3f" % (zc[b], Pw[:, b].sum() / (Pw[:, b].sum() + Ew[:, b].sum())) for b in range(nb)))
for face, z in (("lid", 1.0), ("floor", 0.0)):
    P, E = parts(d[f"{face}_ET"].astype(float), d[f"{face}_ER"].astype(float), d[f"{face}_CO"].astype(float), z)
    res[face] = P.sum() / (P.sum() + E.sum())
    print("%-6s pressure share: %.3f   P-part slope k5-15 %.2f" % (face, res[face], slope(P, 5, 15)))
np.save(f"{CACHE}/three_d_pressure.npy", res, allow_pickle=True)
