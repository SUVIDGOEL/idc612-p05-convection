"""Markdown tables of the vertical-analysis results, read from the saved caches (no hand copying)."""
from convection.config import CACHE
import numpy as np
R = {t: np.load(f"{CACHE}/analyse4_{t}.npy", allow_pickle=True).item() for t in ("Ra1e9", "Ra1e12", "Ra1e15")}
R3 = np.load(f"{CACHE}/analyse4_3d.npy", allow_pickle=True).item()
kw = np.load(f"{CACHE}/kw.npy", allow_pickle=True).item()
sp = np.load(f"{CACHE}/kw_split.npy", allow_pickle=True).item()
cols = [("Ra1e9", "10⁹"), ("Ra1e12", "10¹²"), ("Ra1e15", "10¹⁵")]
W3 = ("wall_xz", "wall_yz")


def ci(v, d=2):
    if isinstance(v, tuple):
        return f"{v[0]:+.{d}f} [{v[1]:+.{d}f}, {v[2]:+.{d}f}]"
    return f"{v:+.{d}f}"


def p(v, d=2):
    return f"{v[0]:.{d}f} [{v[1]:.{d}f}, {v[2]:.{d}f}]" if isinstance(v, tuple) else f"{v:.{d}f}"


def w3(k, f=ci, d=2):
    return " / ".join(f(R3[w][k], d) for w in W3)


def row(name, key, f=ci, d=2, three=True):
    cells = [f(R[t][key], d) for t, _ in cols]
    c3 = w3(key, f, d) if three and key in R3["wall_xz"] else "—"
    return f"| {name} | " + " | ".join(cells) + f" | {c3} |"


hdr = "| quantity | 2D 10⁹ | 2D 10¹² | 2D 10¹⁵ | 3D 10⁸ (plane y=0 / plane x=0) |\n|---|---|---|---|---|"
print("### A. Upper branch (mean-profile modes)\n"); print(hdr)
print("| bottom jump J₀/ε | " + " | ".join(f"{R[t]['J0'] / 0.1:+.2f}" for t, _ in cols) + " | " + " / ".join(f"{R3[w]['J0'] / 0.1:+.2f}" for w in W3) + " |")
print("| top jump J₁/ε | " + " | ".join(f"{R[t]['J1'] / 0.1:+.2f}" for t, _ in cols) + " | " + " / ".join(f"{R3[w]['J1'] / 0.1:+.2f}" for w in W3) + " |")
print(row("share of θ′ variance in the mean profile", "var_mean_frac", p))
print(row("measured/predicted amplitude, even n ≤ 10", "mean_amp_ratio_even_n2_10", p))
print(row("measured/predicted amplitude, odd n ≤ 9", "mean_amp_ratio_odd_n1_9", p))
print(row("even-n slope, n ≤ 20", "mean_slope_even"))
print("| BL thickness from roll-off (bottom / top) | " + " | ".join(f"{R[t]['delta_spec'][0]:.4f} / {R[t]['delta_spec'][1]:.4f}" for t, _ in cols) + " | " + " ; ".join(f"{R3[w]['delta_spec'][0]:.3f} / {R3[w]['delta_spec'][1]:.3f}" for w in W3) + " |")
print("| pixel size 1/Nz | " + " | ".join(f"{1 / R[t]['Nz']:.4f}" for t, _ in cols) + " | " + " / ".join(f"{1 / R3[w]['Nz']:.4f}" for w in W3) + " |")
print("| plate gradient/ε (bottom / top) | " + " | ".join(f"{R[t]['Nu_lower_bound'][0]:.0f} / {R[t]['Nu_lower_bound'][1]:.0f}" for t, _ in cols) + " | " + " ; ".join(f"{R3[w]['Nu_lower_bound'][0]:.0f} / {R3[w]['Nu_lower_bound'][1]:.0f}" for w in W3) + " |")
print("\n### B. Lower branch (fluctuations), slopes\n"); print(hdr)
for lo, hi in ((3, 8), (5, 15), (10, 40)):
    for nm, pre in (("2D shell", "shell"), ("1D x", "x"), ("1D z", "z")):
        k = f"{pre}_{lo}_{hi}"
        k3 = k if k in R3["wall_xz"] else f"{pre}_{lo}_{min(hi, 30)}"
        cells = [ci(R[t][k]) for t, _ in cols]
        c3 = " / ".join(ci(R3[w][k3]) for w in W3) if k3 in R3["wall_xz"] else "—"
        print(f"| {nm}, K/π {lo}–{hi} | " + " | ".join(cells) + f" | {c3} |")
print(row("old x-slope, k = 10–30 (check vs step 1–4)", "x_old_10_30"))
print("\n### C. Isotropy\n"); print(hdr)
for k, nm in (("iso_z_over_x_3_10", "E_z/E_x, whole box, k/π 3–10"), ("iso_z_over_x_10_40", "E_z/E_x, whole box, k/π 10–40"),
              ("iso_bulk_3_10", "E_z/E_x, bulk (same window), k/π 3–10"), ("iso_bulk_10_40", "E_z/E_x, bulk (same window), k/π 10–40")):
    print(row(nm, k, p))
print("\n### D. T–ρ and the pressure part (2D shells, cosine basis)\n"); print(hdr)
for k, nm in (("coh_1_5", "coherence K/π 1–5"), ("coh_5_15", "coherence K/π 5–15"), ("coh_15_50", "coherence K/π 15–50"),
              ("share_total", "pressure share, total"), ("share_1_3", "pressure share K/π 1–3"), ("share_5_15", "pressure share K/π 5–15"),
              ("share_15_50", "pressure share K/π 15–50"), ("P_shell_5_15", "P-part shell slope 5–15"), ("E_shell_5_15", "E-part shell slope 5–15")):
    print(row(nm, k, p if "share" in k or "coh" in k else ci))
print("\n### E. Sound waves (k–ω ridge)\n")
print("| | 2D 10⁹ | 2D 10¹² | 2D 10¹⁵ | 3D 10⁸ |\n|---|---|---|---|---|")
print("| ridge speed c from T / from ρ | " + " | ".join(("none" if t == "Ra1e9" else f"{kw[(t, 'cT')][0]:.2f} / {kw[(t, 'cR')][0]:.2f}") for t, _ in cols) + " | none |")
def fr(t, key, lo, hi):
    s = sp[(t, key)]; m = (s["kb"] >= lo) & (s["kb"] <= hi); return np.nanmean(s["acoustic"][m]) * 100
print("| % of T variance on the ridge, K/π 10–30 | " + " | ".join(f"{fr(t, 'cT', 10, 30):.1f}%" for t, _ in cols) + " | < 0.4% |")
print("| % of ρ variance on the ridge, K/π 10–30 | " + " | ".join(f"{fr(t, 'cR', 10, 30):.1f}%" for t, _ in cols) + " | — |")
print("\n### F. Statistics in physical space\n"); print(hdr)
for k, nm in (("skew_bottom", "skewness z 0.03–0.2"), ("skew_mid", "skewness z 0.4–0.6"), ("skew_top", "skewness z 0.8–0.97"),
              ("flat_mid", "flatness z 0.4–0.6")):
    print(row(nm, k, ci))
print(row("ζ₂ along x (S₂ ∝ r^ζ₂, r 0.015–0.1)", "zeta2_x_T"))
print(row("ζ₂ along z", "zeta2_z_T"))
print("| ESS ζ₆/ζ₃ (no intermittency: 2) | " + " | ".join(f"{R[t]['ess_x_T'][5]:.2f}" for t, _ in cols) + " | " + " / ".join(f"{R3[w]['ess_x_T'][5]:.2f}" for w in W3) + " |")
print("| flatness of δθ at the smallest r (x) | " + " | ".join(f"{R[t]['flat_r_x_T'][0]:.0f}" for t, _ in cols) + " | " + " / ".join(f"{R3[w]['flat_r_x_T'][0]:.0f}" for w in W3) + " |")
print("| τ ∝ K^β, β (K/π 3–30, envelope) | " + " | ".join(f"{R[t]['tau_slope_env_3_30']:+.2f}" for t, _ in cols) + " | " + " / ".join(f"{R3[w]['tau_slope_env_3_30']:+.2f}" for w in W3) + " |")
print(row("entropy S, 2D shells, fluctuations (bits)", "S_shell_fluct", p))
print(row("entropy S, 2D shells, with mean-profile modes", "S_shell_all", p))
print("| frames / independent batches | " + " | ".join(f"{R[t]['nframes']} / {R[t]['n_batches']}" for t, _ in cols) + " | " + " / ".join(f"{R3[w]['nframes']} / {R3[w]['n_batches']}" for w in W3) + " |")
