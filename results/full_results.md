# P05 · Full results

Working record of every number. Figure labels such as f27 refer to the course-page figures, which are not part of this repository; the deck figures are in figures/.

All intervals are 95% moving-block-bootstrap intervals (blocks of 4 correlation times), unless marked "± SE".
Slopes are least-squares fits of log E against log k. k = whole waves across the box (box width 4).
Data: 2D movies at Ra = 10⁹, 10¹², 10¹⁵ (temperature and density panels, plus zoomed panels); 3D movie at Ra = 10⁸.

## 1. Temperature spectra, 2D (steps 1–4)

| Ra | slope k = 10–30 | own straight range | slope in own range | slope k = 40–150 |
|---|---|---|---|---|
| 10⁹  | −1.52 [−1.65, −1.39] | k = 8–18   | −1.49 [−1.68, −1.33] | −3.67 (dissipation) |
| 10¹² | −1.59 [−1.69, −1.45] | k = 17–71  | −1.75 [−1.79, −1.70] | −1.97 [−1.99, −1.94] |
| 10¹⁵ | −2.35 [−2.42, −2.31] | k = 35–103 | −1.92 [−1.94, −1.89] | −1.88 [−1.92, −1.86] |

- Theory (Obukhov–Corrsin): −5/3 = −1.67. Inside the 10¹² interval; just outside the 10⁹ 95% interval, inside its 99% interval [−1.69, −1.36].
- Local slope: only 10¹² is straight across k = 10–30. 10⁹ steepens across the band (short inertial range). 10¹⁵ is −2.2…−2.8 with bumps, then ≈ −1.9 after k ≈ 35.
- Top layer removed (z < 0.72): −1.74, −1.94, −2.61 — steeper, same order.
- Slope against height (k = 10–30): shallow at the top plate (−1.1…−1.5), steep at the bottom (−2.2…−3.1), all three movies.
- Dissipation range starts (local slope −3, top layer): k ≈ 60, 250, 630. **Caveat (§7.8): at the Ra = 10⁹ bitrate the synthetic test shows compression steepening for k > 80, so the 10⁹ value is near the compression limit.** The zoomed panels (4× finer pixels) agree with the main panels within 0.1–0.2 in slope for k = 16–500.
- Method check: movies with known slopes are recovered within 0.04–0.09 (always slightly steeper).

## 2. Density panel, 2D

| Ra | density slope k = 10–30 | T–ρ coherence, k = 1–10 | T–ρ coherence, k = 30–100 | pressure share of T | parts equal at k |
|---|---|---|---|---|---|
| 10⁹  | −2.70 [−2.83, −2.57] | +0.42 [+0.37, +0.47] | −0.76 [−0.79, −0.72] | 28% [27, 31] | ≈ 2 |
| 10¹² | −2.25 [−2.51, −1.85] | +0.82 [+0.80, +0.83] | −0.04 [−0.05, −0.03] | 77% [77, 78] | ≈ 16 |
| 10¹⁵ | −2.51 [−2.62, −2.46] | +0.90 [+0.90, +0.92] | +0.47 [+0.46, +0.48] | 97% [97, 97] | ≈ 60–70 |

- Positive coherence = cold and light = low-pressure vortex cores. Negative = cold and dense = thermal plumes.
- Near the top plate the coherence is always lower (more thermal) than near the bottom.
- Pressure-part slope, k = 10–30 (temperature units): −3.67, −2.37 [−2.55, −2.12], −2.48 [−2.54, −2.46]. 10¹²: the interval contains −7/3. 10¹⁵: −2.48 in temperature units misses −7/3 by 0.13; with ln T and equal weight per height it is −2.36 [−2.41, −2.32], which contains it. So: close to −7/3, not proven (short range; needs velocity).
- The pixels give back the simulation constants from the adiabatic T/ρ ratio: γ − 1 = 0.291 [0.279, 0.298] and D = 0.475 [0.463, 0.487] (10¹²); 0.294 [0.289, 0.304] and 0.479 [0.458, 0.494] (10¹⁵). Printed on the frames: γ = 1.3, D = 0.5.
- Robust to D = 0.45–0.55: pressure share changes by ≤ 0.01, slopes by ≤ 0.05.

## 3. Hydrodynamic entropy (Verma and Chatterjee 2022), k ≤ 150

| Ra | S (bits) |
|---|---|
| 10⁸ (3D walls) | 4.00 ± 0.02 SE |
| 10⁹  | 4.44 [4.30, 4.58] |
| 10¹² | 3.67 [3.56, 3.78] |
| 10¹⁵ | 2.82 [2.64, 2.90] |

S decreases significantly at each step in 2D: the temperature variance sits in fewer, larger structures at higher Ra.
The 3D value (4.00) is not on the same Ra trend: its k counts waves across a 3D wall face, not across the 2D box of width 4, the movie is at a lower Ra, and it is a side wall of a 3D box. S hardly depends on k_max (3D: 3.83 at k≤27, 4.07 at k≤150; 97% of the variance is at k ≤ 27). Reference: a pure k^(−5/3) spectrum on k = 1…150 gives 3.19 bits; white noise gives log₂150 = 7.23.

## 4. 3D movie, Ra = 10⁸

- Read after a per-face lighting correction (T: inferno × 0.41/0.53/0.88; ρ: fitted colour model, RGB distance 1.3–2.9) and a perspective straightening of each face.
- **Correction (§7.7): the four drawn walls are only two planes (y = 0 and x = 0), each drawn twice** (copies correlate at 0.91–0.98). Averages over "4 walls" are averages over 2 planes. The lid is the plane z ≈ 0.98 and the floor z ≈ 0.02 (matched to the wall temperature profile), i.e. inside the thermal boundary layers.
- Walls: slope k = 5–15: −1.53 [−1.60, −1.40]; k = 10–30: −2.40 [−2.56, −2.25] (already in the dissipation range). No straight range of a factor 2 (rule ±0.15 or ±0.20): too weak a flow for a clear inertial range.
- Lid and floor (repeat in x and y): 1D slopes along x and along y agree (lid k = 10–30: −3.16 / −3.09; floor: −1.51 / −1.57) → the horizontal planes are isotropic. Cell size (peak of k·E): lid k ≈ 9, floor k ≈ 21. Dissipation starts at k ≈ 22 (lid), 47 (floor), 27 (walls).
- **T–ρ coherence is negative on every face and at every size:** walls k = 1–10: −0.70 [−0.73, −0.67]; k = 30–100: −0.54 [−0.56, −0.53]; lid and floor −0.6…−0.9. **Pressure share (convection/movie_3d/pressure_share_3d.py): walls 7.3% [6.8, 7.8], lid 7.7%, floor 1.7%.** Walls by height (top→bottom): 7.3, 8.5, 7.9, 5.3, 4.5%. Pressure-part slope, walls k = 5–15: −1.24 [−1.39, −1.05] (not steep like the 2D vortex pressure; the part is small, so its spectrum is uncertain). Earlier quick values (6/5/2%) came from an unsaved check and are replaced. The 3D cold structures are thermal plumes, not vortex cores.

## 5. The conclusion

1. All three 2D temperature spectra are close to power laws at k = 10–30; Ra = 10¹² follows −5/3 (Obukhov–Corrsin) best.
2. The spectrum becomes steeper with Ra. At Ra = 10¹⁵ the steepening is at large sizes (k ≈ 10–40).
3. The density panel explains it: in 2D, low-pressure vortex cores take over the temperature as Ra grows (pressure share 28% → 77% → 97%), and their pressure spectrum is steep.
4. In 3D (Ra = 10⁸) the cold structures are thermal plumes (pressure share 2–8%; walls 7%). The large vortices are a 2D effect.
5. From the pixels alone we recover γ and D of the simulation, which validates both colour inversions.
6. **Vertical analysis (§7).** With the mirror trick (sine series in z) we get the full 2D spectrum. It has two branches: the mean profile gives a k⁻² upper branch whose amplitude a parameter-free jump model predicts within 0–3% (unresolved layers). The jumps show a strong top–bottom asymmetry (top 0.75 ε, bottom 0.25 ε): a compressible, non-Boussinesq effect.
7. The bulk is nearly isotropic (E_z/E_x = 1.0–1.2); the plates make the whole box anisotropic.
8. **Sound waves** are visible at 10¹² and 10¹⁵. Their speed matches the ideal-gas value with no fit (1.93–2.08 against 2.09). They carry about 20% of the temperature variance at 10¹⁵, inside the pressure part.
9. Patterns live for τ ∝ K^(−0.8) (sweeping); the statistics are strongly intermittent with long cold tails.

## 6. Theory vs results

| Theory | Predicts | Measured | Match? |
|---|---|---|---|
| Obukhov–Corrsin | −5/3 | 10¹² −1.59 [−1.69, −1.45] | yes |
| Obukhov–Corrsin | −5/3 | 10⁹ −1.52 [−1.65, −1.39] | close (inside 99%; short range k 8–18) |
| Obukhov–Corrsin | −5/3 | 10¹⁵ −2.35 [−2.42, −2.31] | no (steeper; pressure part explains it) |
| Sharp fronts | −2 | 10¹² −1.97 (k 40–150), 10¹⁵ −1.92 (k 35–103) | yes |
| Pressure of a cascade | −7/3 | 10¹² −2.37; 10¹⁵ −2.36 (ln T) / −2.48 (T units) | close, not proof |
| Adiabatic ideal gas | γ−1 = 0.3, D = 0.5 | 0.29, 0.48 | yes (3%, 5%) |
| 2D turbulence (Kraichnan 1967) | vortices in 2D only | pressure share 2D 28→97%, 3D 7% | yes (3D at lower Ra) |
| Short inertial range at low Ra | no long power law at 10⁸ | none over a factor 2 | yes |
| Kolmogorov dissipation size | k_d ×9 per ×1000 Ra | ×4, then ×2.5 | no (grid and, at 10⁹, compression limit it) |
| Bolgiano–Obukhov (temperature) | −7/5 | 10⁹ −1.52 [−1.65, −1.39]; 10¹², 10¹⁵ steeper | cannot be told apart from −5/3 at 10⁹; no at higher Ra |
| Upper branch from the boundary layers | k⁻², θ̂(2n) = −1/(2nπ) | n^(−2.09…−2.15); amplitudes = jump model within 0–3% (10¹², 10¹⁵) | yes (generalised to unequal jumps) |
| Boussinesq up–down symmetry | jumps ε/2 and ε/2; odd modes 0 | 0.25 ε and 0.75 ε; odd modes large | no: compressible asymmetry |
| Near-isotropic bulk (Nath et al. 2016) | E_z/E_x ≈ 1 | bulk 1.01–1.17; whole box 1.3–2.4 | yes in the bulk |
| Sound speed of an ideal gas | c_s = 2.09 (depth mean) | 1.93–2.08 | yes (no fit) |
| Random sweeping (Kraichnan; random Taylor hypothesis) | τ ∝ K⁻¹ | K^(−0.84) (10⁹), K^(−0.74…−0.85) (3D) | close |
| Exponential tails (Castaing et al. 1989) | exponential PDF tails | exponential cold tails; hot side cut | yes (cold side) |

Method bias: on synthetic movies (Ra 10⁹ layout and bitrate) slopes come out ≈ 0.05 too steep, so the true 10⁹ slope is probably ≈ −1.47.

## 7. Vertical analysis with the mirror trick (all 4 movies)

### 7.1 Method
- z is not periodic (plates at z = 0 and z = 1). The plates are fixed-temperature walls: T − T_A = 0 at z = 0 and −ε at z = 1. So θ′ = (T − T_A) + εz is exactly 0 at both plates.
- Mirror each column about a plate (odd mirror, z → −z). The column then has period 2 and no jump. Its Fourier series is a sine series: θ′(z) = Σ aₙ sin(nπz), k_z = nπ (DST-I). The spectral code Tarang uses the same basis. Density and the pressure/entropy parts are not 0 at the plates: they use the even mirror (cosine series, DCT-I).
- 2D transform = sine (or cosine) in z × Fourier in x. K = √(k_x² + k_z²); shells of width π. "K/π" below: K/π = 5–15 is the same size range as the old x-band k = 10–30.
- Plates found to ±0.5 px from the white axis lines (centroid from the black side). Data rows start 1–3 px inside; the gap to the plate is filled linearly to θ′ = 0. x gap under the side lines (3–5 columns) filled across the periodic wrap. Overlay mask keeps only the dashed zoom box and ticks (two slow vortices at 10¹⁵ that the old mask removed are now kept).
- Uncertainty: 95% batch-means intervals (batch = 2 correlation times, t-distribution). The movies are short against the correlation time: 10¹² has only 4 independent batches, so its intervals are wide.
- Scripts: geometry/plate_positions.py, common/mirror_transform.py, spectrum_2d/mirror_modes_2d.py (2D), movie_3d/mirror_modes_3d.py (3D), spectrum_2d/analyse_2d_spectra.py, spectrum_2d/sound_waves.py, validation/test_2d_spectra_on_synthetic_movies.py, spectrum_2d/summary_tables.py.

### 7.2 Upper branch: the mean profile (the θ̂(0, n) modes)
96% of the θ′ variance is in the mean profile (m = 0 modes). Its modes follow a parameter-free **jump model** (the −1/(2nπ) result for equal jumps, generalised to unequal plate jumps):
```
E_n = 2/(nπ)² · [J₀ − (−1)ⁿ J₁]²        J₀ = θ′(0⁺), J₁ = θ′(1⁻)  (bulk profile extrapolated to each plate)
Boussinesq: J₀ = −ε/2, J₁ = +ε/2  →  odd modes 0, even modes θ̂(2n) = −ε/(nπ)
```
| quantity | 2D 10⁹ | 2D 10¹² | 2D 10¹⁵ | 3D 10⁸ (plane y=0 / plane x=0) |
|---|---|---|---|---|
| bottom jump J₀/ε | -0.20 | -0.27 | -0.25 | -0.25 / -0.23 |
| top jump J₁/ε | +0.75 | +0.74 | +0.79 | +0.58 / +0.60 |
| share of θ′ variance in the mean profile | 0.96 [0.96, 0.97] | 0.96 [0.96, 0.97] | 0.96 [0.96, 0.97] | 0.94 [0.93, 0.95] / 0.96 [0.96, 0.97] |
| measured/predicted amplitude, even n ≤ 10 | 0.89 [0.88, 0.90] | 0.98 [0.97, 0.99] | 1.00 [0.97, 1.03] | 0.84 [0.81, 0.87] / 0.84 [0.82, 0.86] |
| measured/predicted amplitude, odd n ≤ 9 | 0.89 [0.88, 0.90] | 1.00 [0.98, 1.03] | 1.00 [0.98, 1.02] | 0.81 [0.77, 0.85] / 0.84 [0.81, 0.87] |
| even-n slope, n ≤ 20 | -3.11 [-3.16, -3.05] | -2.15 [-2.17, -2.13] | -2.09 [-2.10, -2.09] | -3.65 [-3.72, -3.58] / -3.80 [-3.91, -3.69] |
| BL thickness from roll-off (bottom / top) | 0.0104 / 0.0255 | 0.0070 / 0.0063 | 0.0060 / 0.0033 | 0.035 / 0.029 ; 0.037 / 0.032 |
| pixel size 1/Nz | 0.0031 | 0.0019 | 0.0037 | 0.0055 / 0.0056 |
| plate gradient/ε (bottom / top) | 11 / 19 | 42 / 80 | 30 / 123 | 13 / 6 ; 12 / 18 |

- Unresolved layers (10¹², 10¹⁵): measured amplitudes = model within 0–3%, mode by mode, odd and even (f28, f29). The k⁻² upper branch is confirmed with no fitted constant.
- Resolved layers (10⁹, 3D): the modes roll off after n ≈ 1/(πδ). Roll-off fit: δ ≈ 0.010–0.016 (bottom) and 0.025–0.028 (top) at 10⁹ (the value depends on the fit range); δ ≈ 0.03 in 3D.
- **Up–down asymmetry (non-Boussinesq):** the top jump is 0.74–0.79 ε and the bottom jump 0.20–0.27 ε in all 2D movies (3D: 0.6 ε and 0.24 ε). Boussinesq gives 0.5 and 0.5. The odd modes, which are 0 for an up–down symmetric profile, are therefore large. The plate gradients are also unequal (10¹⁵: 30 ε at the bottom against 123 ε at the top; lower bounds where the layer is below 1 px). The 3D top gradient differs between the two copies (6 against 18), so it is not reliable.

### 7.3 Lower branch: fluctuations
| quantity | 2D 10⁹ | 2D 10¹² | 2D 10¹⁵ | 3D 10⁸ (plane y=0 / plane x=0) |
|---|---|---|---|---|
| 2D shell, K/π 3–8 | -1.11 [-1.36, -0.86] | -1.68 [-1.92, -1.44] | -2.43 [-2.55, -2.30] | -1.19 [-1.34, -1.04] / -0.74 [-1.09, -0.38] |
| 1D x, K/π 3–8 | -1.42 [-1.59, -1.24] | -1.43 [-1.57, -1.28] | -2.54 [-2.63, -2.46] | -1.53 [-1.79, -1.27] / -1.54 [-1.68, -1.39] |
| 1D z, K/π 3–8 | -1.16 [-1.30, -1.02] | -1.84 [-2.31, -1.38] | -2.56 [-2.69, -2.43] | -1.57 [-1.69, -1.46] / -1.35 [-1.57, -1.13] |
| 2D shell, K/π 5–15 | -1.27 [-1.38, -1.16] | -1.62 [-2.04, -1.20] | -2.27 [-2.34, -2.20] | -2.05 [-2.19, -1.91] / -1.93 [-2.08, -1.79] |
| 1D x, K/π 5–15 | -1.52 [-1.67, -1.38] | -1.56 [-1.82, -1.31] | -2.34 [-2.39, -2.29] | -2.44 [-2.57, -2.30] / -2.19 [-2.41, -1.96] |
| 1D z, K/π 5–15 | -1.53 [-1.60, -1.47] | -1.56 [-1.81, -1.32] | -1.87 [-1.95, -1.80] | -2.42 [-2.53, -2.32] / -2.67 [-2.79, -2.56] |
| 2D shell, K/π 10–40 | -2.40 [-2.48, -2.33] | -1.51 [-1.60, -1.42] | -2.11 [-2.13, -2.09] | -3.49 [-3.55, -3.43] / -2.56 [-2.72, -2.40] |
| 1D x, K/π 10–40 | -2.38 [-2.44, -2.31] | -1.75 [-1.86, -1.65] | -2.05 [-2.10, -2.01] | -3.62 [-3.70, -3.54] / -3.49 [-3.66, -3.32] |
| 1D z, K/π 10–40 | -2.84 [-2.93, -2.75] | -1.49 [-1.56, -1.41] | -2.02 [-2.04, -2.00] | -3.32 [-3.41, -3.23] / -1.87 [-2.06, -1.69] |
| old x-slope, k = 10–30 (check vs step 1–4) | -1.52 [-1.67, -1.38] | -1.56 [-1.82, -1.31] | -2.34 [-2.39, -2.29] | -2.44 [-2.57, -2.30] / -2.19 [-2.41, -1.96] |

- The new x-slopes (all rows, new grid, x-gap filled) reproduce steps 1–4: −1.52, −1.56, −2.34 against −1.52, −1.59, −2.35.
- 2D shell slopes at K/π = 5–15: −1.27, −1.62, −2.27. Only 10¹² is consistent with −5/3 (wide interval; 4 batches). The steepening with Ra seen in x is also seen in z and in the full 2D shells.
- At 10¹⁵ the shell slope at K/π = 10–40 is −2.11 [−2.13, −2.09]: close to the sharp-front value −2, not −5/3 or −7/5 (Bolgiano–Obukhov).

### 7.4 Isotropy
| quantity | 2D 10⁹ | 2D 10¹² | 2D 10¹⁵ | 3D 10⁸ (plane y=0 / plane x=0) |
|---|---|---|---|---|
| E_z/E_x, whole box, k/π 3–10 | 1.33 [1.18, 1.49] | 1.48 [1.35, 1.60] | 1.58 [1.48, 1.69] | 1.16 [1.08, 1.25] / 1.16 [1.01, 1.31] |
| E_z/E_x, whole box, k/π 10–40 | 1.05 [0.98, 1.12] | 1.68 [1.53, 1.82] | 2.36 [2.22, 2.49] | 1.43 [1.34, 1.52] / 2.77 [2.48, 3.06] |
| E_z/E_x, bulk (same window), k/π 3–10 | 1.01 [0.92, 1.10] | 1.06 [0.91, 1.21] | 1.17 [1.11, 1.24] | — |
| E_z/E_x, bulk (same window), k/π 10–40 | 0.62 [0.50, 0.74] | 1.01 [0.95, 1.06] | 1.04 [1.02, 1.05] | — |

- Over the whole box, vertical wavenumbers carry 1.3–2.4 × more energy than horizontal ones. This comes from the plates: thin layers and the strong change of rms with height.
- With the same window in both directions (bulk only), the ratio is 1.01–1.17 at k/π = 3–10. **The bulk is nearly isotropic**, as Nath, Pandey, Kumar & Verma (PRF 1, 064302, 2016) found in 3D. (The 10⁹ value 0.62 at k/π = 10–40 is at the compression limit.) A synthetic isotropic field gives 1.01–1.14, so the test is valid.

### 7.5 T–ρ coherence and the pressure part, by 2D shell
| quantity | 2D 10⁹ | 2D 10¹² | 2D 10¹⁵ | 3D 10⁸ (plane y=0 / plane x=0) |
|---|---|---|---|---|
| coherence K/π 1–5 | 0.47 [0.41, 0.52] | 0.81 [0.79, 0.84] | 0.86 [0.84, 0.89] | -0.72 [-0.75, -0.69] / -0.73 [-0.78, -0.68] |
| coherence K/π 5–15 | 0.03 [-0.04, 0.10] | 0.65 [0.49, 0.80] | 0.91 [0.91, 0.92] | -0.69 [-0.72, -0.67] / -0.74 [-0.79, -0.70] |
| coherence K/π 15–50 | -0.68 [-0.72, -0.64] | 0.04 [-0.01, 0.09] | 0.58 [0.57, 0.59] | -0.42 [-0.44, -0.40] / -0.47 [-0.49, -0.46] |
| pressure share, total | 0.25 [0.23, 0.27] | 0.71 [0.70, 0.72] | 0.95 [0.95, 0.96] | — |
| pressure share K/π 1–3 | 0.45 [0.41, 0.49] | 0.95 [0.93, 0.96] | 0.99 [0.99, 0.99] | — |
| pressure share K/π 5–15 | 0.12 [0.11, 0.14] | 0.60 [0.50, 0.69] | 0.94 [0.94, 0.95] | — |
| pressure share K/π 15–50 | 0.01 [0.01, 0.02] | 0.17 [0.16, 0.19] | 0.66 [0.65, 0.66] | — |
| P-part shell slope 5–15 | -3.65 [-3.86, -3.45] | -2.10 [-2.64, -1.57] | -2.41 [-2.46, -2.37] | — |
| E-part shell slope 5–15 | -1.06 [-1.24, -0.87] | -0.31 [-0.59, -0.03] | -0.54 [-0.69, -0.39] | — |

- These agree with the x-only results of §2 (shares 25/71/95% against 28/77/97%). Vortex cores (positive coherence) take over as Ra grows; 3D stays negative (plumes).

### 7.6 Sound waves (new)
| | 2D 10⁹ | 2D 10¹² | 2D 10¹⁵ | 3D 10⁸ |
|---|---|---|---|---|
| ridge speed c from T / from ρ | none | 1.93 / 2.08 | 1.93 / 2.00 | none |
| % of T variance on the ridge, K/π 10–30 | 0.0% | 4.2% | 20.5% | < 0.4% |
| % of ρ variance on the ridge, K/π 10–30 | 0.1% | 7.2% | 17.2% | — |

- The k–ω spectrum of the stored modes (f33) shows a straight ridge ω = cK at 10¹² and 10¹⁵, in both T and ρ.
- Theory with no fit: c_s = √((γ − 1) T / (D ε)) in free-fall units; 2.45 at the bottom, 1.73 at the top; travel-time mean over the depth 2.09. Measured 1.93–2.08 (scatter 3–5%). **The pixels give the sound speed of the simulation**, a third independent check of γ, D and ε (after the T/ρ ratio and the adiabat).
- Sound grows with Ra: 0% (10⁹, 3D 10⁸), about 4–7% (10¹²), about 17–21% (10¹⁵) of the variance at K/π = 10–30. Sound is adiabatic, so it sits entirely in the pressure part. **At 10¹⁵ part of the "pressure part" is sound, not vortex cores.**
- At 10⁹ (no sound) the time spectrum widens as ω_rms ∝ K^0.85–1.04: the sweeping law (K¹), with a sweep speed of 0.14–0.17.

### 7.7 Statistics in physical space
| quantity | 2D 10⁹ | 2D 10¹² | 2D 10¹⁵ | 3D 10⁸ (plane y=0 / plane x=0) |
|---|---|---|---|---|
| skewness z 0.03–0.2 | -0.93 | -1.49 | -1.40 | -0.13 / -0.08 |
| skewness z 0.4–0.6 | -1.61 | -0.02 | +0.15 | -1.96 / -2.81 |
| skewness z 0.8–0.97 | -1.86 | -0.27 | +0.24 | -1.18 / -1.69 |
| flatness z 0.4–0.6 | +9.69 | +4.20 | +2.59 | +9.67 / +16.82 |
| ζ₂ along x (S₂ ∝ r^ζ₂, r 0.015–0.1) | +1.02 [+1.00, +1.03] | +0.92 [+0.86, +0.98] | +1.22 [+1.21, +1.23] | +1.43 [+1.42, +1.45] / +1.37 [+1.31, +1.42] |
| ζ₂ along z | +1.06 [+1.05, +1.08] | +0.83 [+0.80, +0.87] | +1.06 [+1.05, +1.08] | +1.43 [+1.42, +1.45] / +1.47 [+1.43, +1.51] |
| ESS ζ₆/ζ₃ (no intermittency: 2) | 1.62 | 1.65 | 1.76 | 1.79 / 1.73 |
| flatness of δθ at the smallest r (x) | 37 | 143 | 31 | 21 / 41 |
| τ ∝ K^β, β (K/π 3–30, envelope) | -0.84 | -0.38 | -0.97 | -0.85 / -0.74 |
| entropy S, 2D shells, fluctuations (bits) | 4.36 [4.29, 4.43] | 3.97 [3.86, 4.08] | 3.00 [2.89, 3.11] | 3.83 [3.78, 3.88] / 4.33 [4.19, 4.48] |
| entropy S, 2D shells, with mean-profile modes | 2.12 [2.10, 2.14] | 2.57 [2.54, 2.59] | 2.58 [2.57, 2.60] | 2.09 [2.07, 2.11] / 1.98 [1.95, 2.02] |
| frames / independent batches | 1001 / 12 | 1001 / 4 | 201 / 15 | 1001 / 16 / 1001 / 12 |

- PDFs (f34): long exponential cold tails at all heights, hot side near-Gaussian or cut. Cold events dominate. At 10¹² and 10¹⁵ the skewness is most negative near the bottom (cold vortex cores sink there); at 10⁹ and in 3D it is most negative in the middle and top (cold plumes from the top).
- Entropy part: flatness 15–34 (very spiky fronts). Pressure part: flatness 3–8.
- Structure functions: ζ₂ = 0.8–1.5, consistent with the spectral slopes (slope ≈ −(1 + ζ₂)). ζ_p/ζ₃ bends below p/3 and the flatness grows to 20–140 at the smallest r: strong intermittency (f35), as is usual for scalars.
- Decorrelation time (f36): τ ∝ K^(−0.84) at 10⁹ and K^(−0.74…−0.85) in 3D, close to sweeping (K⁻¹, as in the random-Taylor picture), not local straining (K^(−2/3)). At 10¹² and 10¹⁵ the sound waves mix into τ, so τ there is not a clean turbulence measure.
- 3D: the two copies of each plane agree within the intervals for most quantities; they differ by about 0.2 in the fluctuation slopes and up to 0.6 in the roll-off slope. This is the rendering/extraction error of the 3D movie.
- Entropy with 2D shells (closest to the original definition): 4.36, 3.97, 3.00 bits for 10⁹, 10¹², 10¹⁵ (3D 3.8–4.3). Same falling trend as the 1D values of §3.

### 7.8 Validation and robustness
- Synthetic movies with a known 2D spectrum and known boundary layers, drawn with the same axis lines and colour bar, H.264 at the real bitrates (f37). Plates found within 0.03 px. Slope errors ≤ 0.03 at K/π ≤ 40 (10⁹ and 10¹² bitrates) and ≤ 0.02 up to K/π = 100 (10¹⁵ bitrate). Mean-profile amplitudes within 1%. Exception: 1D z at K/π = 10–40 at the 10⁹ bitrate, −0.09.
- **Compression limit:** at the 10⁹ bitrate, K/π > 40 (k_x > 80) is steepened by 0.6–0.9. At the 10¹² bitrate, by 0.1–0.2 at K/π 40–100.
- Sharp (sub-pixel) layers: the n⁻² slope is recovered for n ≤ 20; beyond n ≈ Nz/16 the method itself steepens it.
- Plates moved by ±0.5 px (10⁹): slopes change by ≤ 0.03, jumps/entropy/share/coherence by ≤ 1%, roll-off slope by 0.13.
- An encoder artefact at a 4-frame period is visible in the 10⁹ time spectra at 10⁻⁷ of the power: negligible.

## Scripts
See README.md: every result is mapped to the script in `convection/` that produces it.
