#!/usr/bin/env bash
# Every stage of P05 in order. Run from P05_code/ inside the idc612 conda env.
# Heavy steps read every movie frame; they run one at a time under a 4 GB memory guard.
set -euo pipefail
mkdir -p logs
GUARD="bash tools/memory_guard.sh 4000"
run() { python -m "$@" 2>&1 | tee "logs/$(echo "$*" | tr ' /' '__').log"; }
heavy() { $GUARD "logs/guard_$(echo "$1" | sed 's/.*\.//').txt" python -m "$@"; }

# 0. geometry of the frames: animated panels, colour bars
run convection.geometry.frame_statistics

# 1. 1D temperature spectra along x (result 1)
heavy convection.spectrum_1d.measure_x_spectra
run convection.spectrum_1d.find_straight_ranges

# 2. density panel and the entropy/pressure split (result 3)
heavy convection.density.read_density_panel
run convection.geometry.plate_positions                  # needs the density panel for T/rho alignment
for t in Ra1e9 Ra1e12 Ra1e15; do
  for D in 0.5 0.45 0.55; do heavy convection.density.split_entropy_pressure $t $D; done
done
run convection.density.fit_gamma_and_D
run convection.density.pressure_share

# 3. hydrodynamic entropy of the spectra (result 3)
run convection.entropy.spectral_entropy

# 4. 2D spectra with the mirror trick, mean profile, sound waves (results 2, 4, 5)
for t in Ra1e9 Ra1e12 Ra1e15; do heavy convection.spectrum_2d.mirror_modes_2d $t; done
heavy convection.spectrum_2d.mirror_modes_2d Ra1e9 0.5 5    # robustness: plates shifted +0.5 px
heavy convection.spectrum_2d.mirror_modes_2d Ra1e9 -0.5 5   # robustness: plates shifted -0.5 px

# 5. the 3D movie
heavy convection.movie_3d.read_3d_temperature
heavy convection.movie_3d.read_3d_density
run convection.movie_3d.pressure_share_3d
heavy convection.movie_3d.mirror_modes_3d

for t in Ra1e9 Ra1e12 Ra1e15 3d; do run convection.spectrum_2d.analyse_2d_spectra $t; done
run convection.spectrum_2d.sound_waves
python -m convection.spectrum_2d.summary_tables > cache/summary_tables.md

# 6. method checks on synthetic movies (needs /usr/bin/ffmpeg with libx264)
run convection.validation.test_x_spectra_on_synthetic_movies
run convection.validation.test_2d_spectra_on_synthetic_movies set1
run convection.validation.test_2d_spectra_on_synthetic_movies set2

# 7. presentation figures and equations
run convection.figures.deck_figures
run convection.figures.deck_equations
