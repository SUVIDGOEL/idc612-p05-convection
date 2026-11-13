# P05 · Spectra of compressible convection, measured from movie pixels

IDC612 project. The only data are four simulation movies: 2D convection at Ra = 10⁹, 10¹², 10¹⁵ and 3D convection at Ra = 10⁸ (ideal gas, γ = 1.3, D = 0.5, ε = 0.1). Every number below comes from the pixels.

## Main results and the code that found them

| # | Result | Value | Method (what the code does) | Code | Output |
|---|---|---|---|---|---|
| 1 | Temperature spectrum slope, 1D x-FFT, k = 10–30 | −1.52, −1.59, −2.35 (Ra 10⁹, 10¹², 10¹⁵); 10¹² = −5/3 | colour → T with the frame's colour bar; FFT of each row (x is periodic); average over rows and frames; log–log fit; block bootstrap | `s1_x_spectra/spectra_pass.py`, `common/stats.py` | `cache/spec_<Ra>.npz` |
| 2 | Own straight range and small-scale slope | 10¹²: −1.97 (k 40–150); 10¹⁵: −1.92 (k 35–103) ≈ −2 (sharp fronts) | local slope ±0.15 rule over the per-frame spectra | `s1_x_spectra/own_bands.py` | printed |
| 3 | 2D FFT spectrum (shells) | −1.27, −1.62, −2.27 at K/π = 5–15 | mirror trick: θ′ = T − (1 − Dz) + εz is 0 at both plates → odd mirror → sine series in z; × FFT in x; sum in rings of K | `common/vspec.py`, `s5_vertical/pass4.py`, `s5_vertical/analyse4.py` | `cache/pass4_<Ra>.npz`, `cache/analyse4_<Ra>.npy` |
| 4 | Hydrodynamic entropy of the spectrum | 4.44, 3.67, 2.82 bits (1D); 4.36, 3.97, 3.00 (2D shells); 3D walls 4.00 | p_k = E(k)/ΣE, S = −Σ p_k log₂ p_k per frame, then averaged | `s3_entropy/entropy.py`, `s5_vertical/analyse4.py` | `cache/entropy.npy` |
| 5 | Pressure share of the temperature variance | 28%, 77%, 97% (2D); 7% (3D walls) | ρ panel inverted (diverging colour map); ln T split into entropy and pressure parts using p = ρT | `s2_density/pass2.py`, `s2_density/pass3.py`, `s2_density/final_density.py`, `s4_three_d/three_d_pressure.py` | `cache/final_density.npy`, `cache/three_d_pressure.npy` |
| 6 | γ and D from pixels | γ − 1 = 0.29, D = 0.48 (true 0.30, 0.50) | fit the large-scale T/ρ covariance profile to the ideal-gas adiabat | `s2_density/adiabat_fit.py` | printed |
| 7 | Upper branch from the mean profile | E_n ∝ n⁻²; measured ÷ predicted = 0.98, 1.00 (10¹², 10¹⁵), no fitted constant | sine transform of the mean profile; jump model E_n = 2/(nπ)²·[J₀ − (−1)ⁿJ₁]² with jumps measured from the profile | `s5_vertical/analyse4.py` | `cache/analyse4_<Ra>.npy` |
| 8 | Top–bottom asymmetry | jumps 0.25 ε (bottom), 0.75 ε (top); Boussinesq: 0.5 each | bulk profile extrapolated to each plate | `s5_vertical/analyse4.py` | `cache/analyse4_<Ra>.npy` |
| 9 | Bulk isotropy | E_z/E_x = 1.01, 1.06, 1.17 | same Hann window in x and z, bulk rows only | `s5_vertical/analyse4.py` | `cache/analyse4_<Ra>.npy` |
| 10 | Sound waves | ridge speed 1.93–2.08; ideal-gas theory 2.09; 20% of T variance at 10¹⁵ | FFT in time of the stored 2D mode amplitudes → (K, ω) map → ridge ω = cK | `s5_vertical/kw.py` | `cache/kw.npy`, `cache/kw_split.npy` |
| 11 | 3D movie | walls slope −1.53 (k 5–15); T–ρ coherence negative (plumes) | per-face lighting correction and perspective straightening | `s4_three_d/three_d.py`, `three_d_rho.py`, `final_3d.py`, `s5_vertical/pass5_3d.py` | `cache/spec3d.npz`, `cache/rho3d.npz`, `cache/pass5_3d.npz` |
| 12 | Method checks | slopes recovered within 0.03–0.09; plates within 0.03 px | synthetic movies with known spectra, same layout, colours and H.264 compression | `validation/synthetic_test.py`, `validation/synth_vertical.py` | `synth/*.npy` |

Square-bracket intervals in the full results are 95% (moving-block bootstrap, or batch means where a movie holds few correlation times). Full tables: `results/FINAL_RESULTS.md` and `results/report4.md`.

## Folder layout

```
P05_code/
├── README.md
├── run_all.sh                 every stage in order (heavy passes first)
├── requirements.txt
├── convection/                the Python package (run modules from P05_code/ with python -m)
│   ├── config.py              paths (data, cache, figures) and the simulation constants
│   ├── common/                shared tools
│   │   ├── stats.py           slope fits, correlation time, block bootstrap
│   │   ├── vspec.py           mirror trick: sine/cosine transforms, 2D shells, overlay mask, increments
│   │   ├── pixel_stats.py     step 0: temporal mean/std of every pixel, colour-bar boxes
│   │   └── geom2d.py          plate and side-wall positions to ±0.5 px
│   ├── s1_x_spectra/          result 1–2: 1D temperature spectra along x
│   ├── s2_density/            result 5–6: density panel, T–ρ split, γ and D
│   ├── s3_entropy/            result 4: hydrodynamic entropy
│   ├── s4_three_d/            result 11: the 3D movie
│   ├── s5_vertical/           result 3, 7–10: mirror trick, 2D spectra, sound waves
│   ├── validation/            result 12: synthetic movies
│   └── figures/               every figure (f1–f37 and the deck figures)
├── tools/
│   ├── guard.sh               kills a job above a memory limit (the machine has 13 GB)
│   ├── verify_fast.sh         re-runs all fast stages and compares with a reference copy
│   └── compare_outputs.py
├── figures/                   generated figures (png)
├── results/                   text results: FINAL_RESULTS.md, report4.md, analysis logs
├── data/                      the four movies (not in git; links)
├── cache/                     intermediate arrays (not in git; ≈ 2 GB)
└── synth/                     synthetic-movie results
```

## How to run

```bash
conda activate idc612            # numpy, scipy, matplotlib, av (PyAV), opencv, pillow
cd P05_code
bash run_all.sh                  # all stages; the heavy passes take ≈ 1.5 h in total
bash tools/verify_fast.sh REF_CACHE REF_FIGS   # re-run analysis + figures and compare
```

Each stage reads only from `cache/` and writes back to it, so any single step can be run alone, for example
`python -m convection.s5_vertical.analyse4 Ra1e12`. Paths can be moved with `P05_DATA`, `P05_CACHE`, `P05_FIGS`.

Heavy passes read every frame of a movie. Run at most two at once, each under the guard:
`bash tools/guard.sh 4000 logs/pass4.txt python -m convection.s5_vertical.pass4 Ra1e12`.

## Verification of this codebase (2026-11-13)

- Every analysis stage and every figure was re-run from this package and compared with the original outputs: 49 of 53 files identical. The only differences: three deck figures and f28, which were changed on purpose (labels), and `final_3d.npy`, whose old copy came from an earlier version of the script (the original script run today gives the same as this package).
- Heavy pass check: `pass4` on the raw Ra 10⁹ movie (every 50th frame) gives per-frame arrays identical to the original run.

## Known limits

- Ra 10⁹: video compression limits wavenumbers above k ≈ 80.
- Ra 10¹²: the movie holds about 4 independent time samples, so its intervals are wide.
- 3D: the four drawn walls are only 2 independent planes, each drawn twice; the lid and floor sit at z ≈ 0.98 and 0.02.
- The Ra 10¹⁵ slope (−2.35) matches no single law; vortex cores and sound waves are the likely causes, not proved.
