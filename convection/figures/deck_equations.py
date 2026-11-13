"""Typeset the deck's equations with LaTeX (Computer Modern, amsmath) as transparent PNGs: slides_figs/eq_<name>.png.
Rendered at 2x; the slide shows each at half its pixel size. Prints name, width, height (display px).
Run from P05_code/:  python -m convection.figures.deck_equations
"""
from convection.config import FIGS
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK = "#14213D"
plt.rcParams.update({"text.usetex": True, "font.family": "serif",
                     "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}\usepackage{bm}"})
OUT = f"{FIGS}/"

EQ = {
    # slide 1: 1D spectrum
    "lut": r"$\theta(x,z) = T - T_A = -0.1 + 0.1\,P\!\left(Y(x,z)\right)$",
    "fft": r"$\hat{T}(k,z) = \frac{1}{N_x}\sum_{j=0}^{N_x-1} T(x_j,z)\,e^{-2\pi i\,k x_j/L}$",
    "avg": r"$E(k) = \left\langle\, |\hat{T}(k,z,t)|^2 \,\right\rangle_{z,\,t}$",
    "fit": r"$\log E = \log A - \alpha \log k, \qquad 10 \leq k \leq 30$",
    # slide 2: 2D spectrum
    "zero": r"$\begin{aligned}\theta'(x,z) &= T - (1 - Dz) + \varepsilon z\\ \theta'(x,0) &= \theta'(x,1) = 0\end{aligned}$",
    "sine": r"$\begin{aligned}\theta'(x,z) &= \sum_{m,n} \hat{\theta}(m,n)\; e^{\,i k_x x} \sin(k_z z)\\ k_x &= \frac{2\pi m}{L}, \qquad k_z = n\pi\end{aligned}$",
    "shell": r"$\begin{aligned}E(K) &= \sum_{K-\frac{1}{2}\,\le\, |\mathbf{k}|/\pi \,<\, K+\frac{1}{2}} \tfrac{1}{2}|\hat{\theta}(m,n)|^2\\ |\mathbf{k}| &= \sqrt{k_x^2 + k_z^2}\end{aligned}$",
    "fit2": r"$E(K) \propto K^{-\alpha}, \qquad 5 \leq K/\pi \leq 15$",
    # slide 3: entropy and density split
    "prob": r"$p_k = \frac{E(k)}{\sum_{k'} E(k')}$",
    "ent": r"$S = -\sum_{k=1}^{150} p_k \log_2 p_k, \qquad 0 \leq S \leq \log_2 150 = 7.2$",
    "split": r"$\ln T = \underbrace{\frac{1}{\gamma}\,\frac{s}{c_v}}_{\Sigma\ \mathrm{(entropy)}} + \underbrace{\frac{\gamma-1}{\gamma}\,\ln p}_{\Pi\ \mathrm{(pressure)}}, \qquad p = \rho T$",
    "share": r"$\mathrm{share} = \frac{\mathrm{var}\,\Pi}{\mathrm{var}\,\Sigma + \mathrm{var}\,\Pi}$",
    # slide 4: mean profile
    "mean": r"$\Theta(z) = \left\langle\, \theta'(x,z,t) \,\right\rangle_{x,\,t}$",
    "modes": r"$\hat{\Theta}_n = 2\int_0^1 \Theta(z)\,\sin(n\pi z)\,dz, \qquad E_n = \frac{1}{2}\hat{\Theta}_n^{\,2}$",
    "jump": r"$\begin{aligned}\hat{\Theta}_n &= \frac{2}{n\pi}\left[J_0 - (-1)^n J_1\right]\\ E_n &= \frac{2}{(n\pi)^2}\left[J_0 - (-1)^n J_1\right]^2 \;\propto\; n^{-2}\end{aligned}$",
    "ratio": r"$\mathrm{ratio} = \sqrt{E_n^{\,\mathrm{measured}} \,/\, E_n^{\,\mathrm{model}}}$",
    # slide 5: sound
    "series": r"$\hat{\theta}(m,n,t_j), \qquad t_j = j\,\Delta t, \qquad \Delta t = 0.01$",
    "tfft": r"$P(K,f) = \sum_{|\mathbf{k}|\,\in\,K} \left|\, \sum_j w_j\, \hat{\theta}(m,n,t_j)\, e^{-2\pi i f t_j} \right|^2$",
    "ridge": r"$\omega = 2\pi f = c\,|\mathbf{k}|$",
    "cs": r"$\begin{aligned}c_s(z) &= \sqrt{\frac{(\gamma-1)\,T_A(z)}{D\,\varepsilon}}\\ \bar{c} &= \left[\int_0^1 \frac{dz}{c_s(z)}\right]^{-1} = 2.09\end{aligned}$",
}


def render(name, tex, size=27):
    fig = plt.figure(figsize=(0.01, 0.01))
    fig.text(0, 0, tex.replace("$", "$\\displaystyle ", 1), fontsize=size, color=INK)
    fig.savefig(OUT + f"eq_{name}.png", dpi=200, transparent=True, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


if __name__ == "__main__":
    from PIL import Image
    for k, v in EQ.items():
        render(k, v)
        w, h = Image.open(OUT + f"eq_{k}.png").size
        print(f"{k} {w // 2} {h // 2}")
