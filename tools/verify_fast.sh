#!/usr/bin/env bash
# Re-run every fast stage (analysis of cached passes + all figures) and compare with a reference copy.
# usage: bash tools/verify_fast.sh REF_CACHE REF_FIGS      (run from P05_code/)
set -u
REF_CACHE=${1:?reference cache folder}; REF_FIGS=${2:?reference figures folder}
mkdir -p logs; T0=$(date +%s)
run() { echo "== $*"; python -m "$@" > "logs/$(echo "$*" | tr ' /' '__').log" 2>&1 || echo "   FAILED (see logs)"; }
run convection.s1_x_spectra.own_bands
run convection.s2_density.adiabat_fit
run convection.s2_density.results2
run convection.s2_density.final_density
run convection.s3_entropy.entropy
run convection.s4_three_d.final_3d
run convection.s4_three_d.three_d_pressure
for t in Ra1e9 Ra1e12 Ra1e15 3d; do run convection.s5_vertical.analyse4 $t; done
run convection.s5_vertical.kw
run convection.s5_vertical.report4
for f in make_slide_figs make_figs_step3 make_f5_v2 make_figs2 make_fig_whyx make_fig_row_fft make_figs3d \
         make_fig_break15 make_fig_colourbar make_fig_formulas make_fig_jump make_fig_k41 make_fig_final \
         make_figs4 make_deck_figs; do run convection.figures.$f; done
python tools/compare_outputs.py "$REF_CACHE" "$REF_FIGS" "$T0"
