#!/usr/bin/env bash
# Re-run every fast step (everything that reads only cache/) and compare the outputs with a reference copy.
# usage: bash tools/verify_outputs.sh REF_CACHE REF_FIGS      (run from P05_code/)
set -u
REF_CACHE=${1:?reference cache folder}; REF_FIGS=${2:?reference figures folder}
mkdir -p logs; T0=$(date +%s)
run() { echo "== $*"; python -m "$@" > "logs/$(echo "$*" | tr ' /' '__').log" 2>&1 || echo "   FAILED (see logs)"; }
run convection.spectrum_1d.find_straight_ranges
run convection.density.fit_gamma_and_D
run convection.density.pressure_share
run convection.entropy.spectral_entropy
run convection.movie_3d.pressure_share_3d
for t in Ra1e9 Ra1e12 Ra1e15 3d; do run convection.spectrum_2d.analyse_2d_spectra $t; done
run convection.spectrum_2d.sound_waves
python -m convection.spectrum_2d.summary_tables > cache/summary_tables.md
run convection.figures.deck_figures
run convection.figures.deck_equations
python tools/compare_outputs.py "$REF_CACHE" "$REF_FIGS" "$T0"
diff -q cache/summary_tables.md "$REF_CACHE/summary_tables.md" && echo "OK   summary_tables.md"
