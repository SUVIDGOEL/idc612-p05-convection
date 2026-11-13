"""Large-scale T and rho fluctuations are isentropic (pressure) fluctuations:
      T'/T = (gamma - 1) rho'/rho
With an adiabatic background T_A = 1 - D z, rho_A = (1 - D z)^(1/(gamma-1)) (bottom values = 1):
      beta(z) = cov(T', rho') / var(rho') = (gamma - 1) (1 - D z)^(-(2 - gamma)/(gamma - 1))
Fit A = (gamma - 1) and D to the measured beta(z) at large scales (k = 1..10).
"""
from convection.config import CACHE
import sys
import numpy as np
from scipy.optimize import curve_fit

G = 1.3

def beta_model(z, A, D):
    return A * (1 - D * z) ** (-(2 - G) / (G - 1))

def fit(tag, klo=1, khi=10, nboot=500):
    d = np.load(f"{CACHE}/density_panel_{tag}.npz")
    ER, CO = d["ER"].astype(float), d["CO"].astype(float)
    nb = ER.shape[1]; z = 1 - (np.arange(nb) + 0.5) / nb
    def one(idx):
        b = CO[idx][:, :, klo - 1:khi].sum((0, 2)) / ER[idx][:, :, klo - 1:khi].sum((0, 2))
        return b, curve_fit(beta_model, z, b, p0=(0.3, 0.5))[0]
    n = ER.shape[0]
    b, p = one(np.arange(n))
    # block bootstrap over frames (block = 10% of the movie)
    L = max(1, n // 10); rng = np.random.default_rng(1); ps = []
    for _ in range(nboot):
        st = rng.integers(0, n - L + 1, int(np.ceil(n / L)))
        idx = (st[:, None] + np.arange(L)).ravel()[:n]
        ps.append(one(idx)[1])
    ps = np.array(ps)
    return z, b, p, np.percentile(ps, [2.5, 97.5], axis=0)

if __name__ == "__main__":
    for t in (sys.argv[1:] or ("Ra1e9", "Ra1e12", "Ra1e15")):
        z, b, p, ci = fit(t)
        print("%-7s A = %.3f [%.3f, %.3f]   D = %.3f [%.3f, %.3f]   (expected A = gamma-1 = 0.30, D = 0.5)"
              % (t, p[0], ci[0, 0], ci[1, 0], p[1], ci[0, 1], ci[1, 1]))
        print("        beta(z) top->bottom: " + " ".join("%.3f" % v for v in b))
        print("        model:               " + " ".join("%.3f" % v for v in beta_model(z, *p)))
