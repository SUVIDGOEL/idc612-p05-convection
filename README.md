# P05 · Spectra of compressible convection, measured from movie pixels

IDC612 project. The only data are four simulation movies: 2D convection at Ra = 10⁹, 10¹², 10¹⁵ and 3D convection at Ra = 10⁸ (ideal gas, γ = 1.3, D = 0.5, ε = 0.1). Every number below comes from the pixels.

## Results and the code that found them

| # | Result | Value | Method | Code (in `convection/`) |
|---|---|---|---|---|
| 1 | Temperature spectrum slope, 1D FFT along x, k = 10–30 | −1.52, −1.59, −2.35 (Ra 10⁹, 10¹², 10¹⁵); Ra 10¹² = −5/3 | colour → temperature with the frame's colour bar; FFT of each row (x is periodic); average over rows and frames; log–log fit; block bootstrap | `spectrum_1d/measure_x_spectra.py`, `common/fits_and_errors.py` |
| 1b | Small-scale slope | −1.97 (Ra 10¹², k 40–150), −1.92 (Ra 10¹⁵, k 35–103) ≈ −2, sharp fronts | longest straight range of the local slope | `spectrum_1d/find_straight_ranges.py` |
| 2 | 2D FFT spectrum | −1.27, −1.62, −2.27 at K/π = 5–15 | mirror trick: θ′ = T − (1 − Dz) + εz is 0 on both plates → odd mirror → sine series in z, FFT in x; energy summed in rings of K | `common/mirror_transform.py`, `spectrum_2d/mirror_modes_2d.py`, `spectrum_2d/analyse_2d_spectra.py` |
| 3a | Hydrodynamic entropy of the spectrum | 4.44, 3.67, 2.82 bits (1D); 4.36, 3.97, 3.00 (2D) | pₖ = E(k)/ΣE, S = −Σ pₖ log₂ pₖ per frame, then averaged | `entropy/spectral_entropy.py`, `spectrum_2d/analyse_2d_spectra.py` |
| 3b | Pressure share of the temperature variance | 28%, 77%, 97% (2D); 7% (3D) | density panel read; ln T split into entropy and pressure parts using p = ρT | `density/read_density_panel.py`, `density/split_entropy_pressure.py`, `density/pressure_share.py`, `movie_3d/pressure_share_3d.py` |
| 3c | γ and D from pixels | γ − 1 = 0.29, D = 0.48 (frames print 0.30, 0.50) | large-scale T/ρ ratio fitted to the ideal-gas adiabat | `density/fit_gamma_and_D.py` |
| 4 | k⁻² branch from the mean profile | measured ÷ model = 0.98, 1.00 (Ra 10¹², 10¹⁵), nothing fitted; jumps 0.25 ε (bottom), 0.75 ε (top) | sine transform of the mean profile; jump model Eₙ = 2/(nπ)² [J₀ − (−1)ⁿ J₁]² | `spectrum_2d/analyse_2d_spectra.py` |
| 4b | Bulk isotropy | vertical ÷ horizontal energy = 1.01, 1.06, 1.17 | same window in x and z, bulk rows only | `spectrum_2d/analyse_2d_spectra.py` |
| 5 | Sound waves | ridge speed 1.93–2.08; ideal-gas theory 2.09; 20% of the T variance at Ra 10¹⁵ | FFT in time of the 2D mode amplitudes → (K, ω) map → ridge ω = cK | `spectrum_2d/sound_waves.py` |
| 6 | Method checks | slopes recovered within 0.03–0.09; plates within 0.03 px | synthetic movies with known spectra, same layout, colours and H.264 compression | `validation/test_x_spectra_on_synthetic_movies.py`, `validation/test_2d_spectra_on_synthetic_movies.py` |

Short report: `P05_report.pdf`. Full tables with 95% intervals: `results/full_results.md` and `results/summary_tables.md`.

## Folder layout

```
P05_code/
├── README.md
├── P05_report.pdf                      the written report (6 pages, figures and tables)
├── report/                             LaTeX source of the report, PowerPoint builder, speaker notes
├── run_all.sh                          every step in order
├── requirements.txt
├── convection/                         the Python package (run from P05_code/ with python -m)
│   ├── config.py                       paths and simulation constants
│   ├── common/
│   │   ├── fits_and_errors.py          power-law fits, correlation time, block bootstrap
│   │   └── mirror_transform.py         sine/cosine transforms, 2D shells, overlay mask
│   ├── geometry/
│   │   ├── frame_statistics.py         per-pixel mean/std over time, colour-bar boxes
│   │   └── plate_positions.py          plates and side walls to ±0.5 px
│   ├── spectrum_1d/
│   │   ├── measure_x_spectra.py        colour → temperature, FFT of every row, every frame
│   │   └── find_straight_ranges.py     longest power-law range of each spectrum
│   ├── spectrum_2d/
│   │   ├── mirror_modes_2d.py          mirror trick: 2D modes of every frame (reads the movie)
│   │   ├── analyse_2d_spectra.py       shell spectra, mean-profile modes, jumps, isotropy, entropy
│   │   ├── sound_waves.py              frequency–wavenumber map and the sound ridge
│   │   └── summary_tables.py           markdown tables of all 2D/3D results
│   ├── density/
│   │   ├── read_density_panel.py       density colours → ρ, T–ρ cross spectra
│   │   ├── split_entropy_pressure.py   ln T = entropy part + pressure part
│   │   ├── fit_gamma_and_D.py          γ and D from the adiabatic T/ρ ratio
│   │   └── pressure_share.py           pressure share with 95% intervals
│   ├── entropy/
│   │   └── spectral_entropy.py         S = −Σ pₖ log₂ pₖ of the spectra
│   ├── movie_3d/
│   │   ├── read_3d_temperature.py      straighten the 3D faces, temperature spectra
│   │   ├── read_3d_density.py          density on the 3D faces
│   │   ├── pressure_share_3d.py        pressure share in 3D
│   │   └── mirror_modes_3d.py          mirror trick on the 3D walls
│   ├── validation/
│   │   ├── test_x_spectra_on_synthetic_movies.py
│   │   └── test_2d_spectra_on_synthetic_movies.py
│   └── figures/
│       ├── deck_figures.py             the four result figures of the presentation
│       └── deck_equations.py           the presentation equations, typeset with LaTeX
├── tools/
│   ├── memory_guard.sh                 stops a job above a memory limit
│   ├── verify_outputs.sh               re-runs the fast steps and compares with a reference
│   └── compare_outputs.py
├── figures/                            presentation figures and equation images
├── results/                            full_results.md, summary_tables.md, analysis logs
├── synth/                              results of the synthetic-movie tests
├── data/                               the four movies (not in git)
└── cache/                              intermediate arrays, ≈ 2 GB (not in git)
```

## How to run

```bash
conda activate idc612            # numpy, scipy, matplotlib, av (PyAV), opencv, pillow; LaTeX for deck_equations
cd P05_code
bash run_all.sh                  # every step; the movie-reading steps take ≈ 1.5 h in total
```

Each step reads from `cache/` and writes back to it, so one step can be run alone, for example
`python -m convection.spectrum_2d.analyse_2d_spectra Ra1e12`. The steps that read the movies should run one at a time under the guard:
`bash tools/memory_guard.sh 4000 guard.txt python -m convection.spectrum_2d.mirror_modes_2d Ra1e12`.

## Verification of this codebase (2026-11-13)

- Every step that reads only `cache/` was re-run from this package: all 33 outputs and the summary tables are identical to the original project outputs.
- `mirror_modes_2d` on the raw Ra 10⁹ movie (every 50th frame) gives per-frame arrays identical to the original run.

## Known limits

- Ra 10⁹: video compression limits wavenumbers above k ≈ 80.
- Ra 10¹²: the movie holds about 4 independent time samples, so its intervals are wide.
- 3D: the four drawn walls are only 2 independent planes, each drawn twice; the lid and floor sit at z ≈ 0.98 and 0.02.
- The Ra 10¹⁵ slope (−2.35) matches no single law; vortex cores and sound waves are the likely causes, not proved.
