"""All paths and physical constants in one place.

Paths can be moved with environment variables (used for test runs into a scratch cache):
  P05_DATA   folder holding the movies       (default: P05_code/data)
  P05_CACHE  folder for intermediate results (default: P05_code/cache)
  P05_FIGS   folder for figures              (default: P05_code/figures)
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("P05_DATA", ROOT / "data"))
CACHE = Path(os.environ.get("P05_CACHE", ROOT / "cache"))
FIGS = Path(os.environ.get("P05_FIGS", ROOT / "figures"))
SYNTH = Path(os.environ.get("P05_SYNTH", ROOT / "synth"))
FFMPEG = "/usr/bin/ffmpeg"          # needs libx264 (the conda ffmpeg has none)

# simulation constants printed on the frames
GAMMA, D_ADIABAT, EPS = 1.3, 0.5, 0.1
DT_FRAME, LX = 0.01, 4.0            # time between frames; box width (height = 1)

for _d in (CACHE, FIGS, SYNTH):
    _d.mkdir(parents=True, exist_ok=True)
