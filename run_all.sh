#!/usr/bin/env bash
# Every stage of P05 in order. Run from P05_code/ inside the idc612 conda env.
# Heavy passes (read every movie frame) run one at a time under a 4 GB memory guard.
set -euo pipefail
mkdir -p logs
G="bash tools/guard.sh 4000"
py() { python -m "$@" 2>&1 | tee "logs/$(echo "$*" | tr ' /' '__').log"; }

# 0. geometry: pixel statistics, colour-bar boxes, plate positions
py convection.common.pixel_stats
py convection.common.geom2d

# 1. 1D temperature spectra along x (heavy) + straight ranges
$G logs/g_spectra.txt python -m convection.s1_x_spectra.spectra_pass
py convection.s1_x_spectra.own_bands

# 2. density panel: T-rho cross spectra, entropy/pressure split, gamma and D
$G logs/g_pass2.txt python -m convection.s2_density.pass2
for t in Ra1e9 Ra1e12 Ra1e15; do
  for D in 0.5 0.45 0.55; do $G logs/g_pass3.txt python -m convection.s2_density.pass3 $t $D; done
done
py convection.s2_density.adiabat_fit
py convection.s2_density.results2
py convection.s2_density.final_density

# 3. hydrodynamic entropy
py convection.s3_entropy.entropy

# 4. 3D movie
$G logs/g_three_d.txt python -m convection.s4_three_d.three_d
$G logs/g_three_d_rho.txt python -m convection.s4_three_d.three_d_rho
py convection.s4_three_d.final_3d
py convection.s4_three_d.three_d_pressure

# 5. vertical direction: mirror trick, 2D spectra, sound waves
for t in Ra1e9 Ra1e12 Ra1e15; do $G logs/g_pass4.txt python -m convection.s5_vertical.pass4 $t; done
$G logs/g_pass4.txt python -m convection.s5_vertical.pass4 Ra1e9 0.5 5      # plate shift +0.5 px
$G logs/g_pass4.txt python -m convection.s5_vertical.pass4 Ra1e9 -0.5 5     # plate shift -0.5 px
$G logs/g_pass5.txt python -m convection.s5_vertical.pass5_3d
for t in Ra1e9 Ra1e12 Ra1e15 3d; do py convection.s5_vertical.analyse4 $t; done
py convection.s5_vertical.kw
py convection.s5_vertical.report4

# 6. validation on synthetic movies (needs /usr/bin/ffmpeg with libx264)
py convection.validation.synthetic_test
py convection.validation.synth_vertical set1
py convection.validation.synth_vertical set2

# 7. figures
for f in make_slide_figs make_figs_step3 make_f5_v2 make_figs2 make_fig_whyx make_fig_row_fft make_figs3d \
         make_fig_break15 make_fig_colourbar make_fig_formulas make_fig_jump make_fig_k41 make_fig_final \
         make_figs4 make_deck_figs make_deck_math; do py convection.figures.$f; done
