"""One streaming pass over every frame of a movie.

For each frame, the T panel is inverted to a scalar field and its 1D x-spectrum
is computed for four crop/cleaning variants, so that their effect can be
compared on identical frames:

  legacy  the crop used by p5_v3.py (includes a few static border columns)
  tight   the dynamic region found from temporal std, inset 2 px
  clean   tight + static overlay pixels (zoom box, tick marks) interpolated along x
  lowz    clean, but only rows with z < 0.72 (excludes the zoom-box rows entirely)

Per-frame spectra are cached so every later statistic is post-processing.
"""
from convection.config import CACHE, DATA
import sys, time
import av
import numpy as np
from scipy.ndimage import median_filter, binary_dilation

FILES = {"Ra1e9": f"{DATA}/2d_Ra1e9.mp4", "Ra1e12": f"{DATA}/2d_Ra1e12.mp4",
         "Ra1e15": f"{DATA}/2d_Ra1e15.mp4"}
LEGACY = {"Ra1e9": (118, 439, 145, 1411), "Ra1e12": (355, 889, 139, 2254),
          "Ra1e15": (177, 444, 69, 1127)}          # Ra1e15 = Ra1e12 geometry / 2
DYN = {"Ra1e9": (120, 439, 147, 1410), "Ra1e12": (358, 887, 142, 2252),
       "Ra1e15": (184, 450, 73, 1125)}             # from temporal std
INSET = 2
NBANDS = 10
LOWZ_CUT = 0.28                                    # top 28% of rows = z > 0.72


def build_lut(mean, cbar):
    """Luma (0..255) -> position on the colourbar (1 = vmax at top, 0 = vmin)."""
    r0, r1, c0, c1 = cbar
    r0, r1 = r0 + 3, r1 - 3                        # keep clear of the white border
    strip = np.median(mean[r0:r1, c0:c1], axis=1)
    strip = np.maximum.accumulate(strip[::-1])[::-1]
    pos = np.linspace(1.0, 0.0, len(strip))
    order = np.argsort(strip, kind="stable")
    lum, p = strip[order], pos[order]
    u, inv = np.unique(lum, return_inverse=True)
    pm = np.bincount(inv, weights=p) / np.bincount(inv)
    return np.interp(np.arange(256), u, pm)


def overlay_mask(std):
    """Pixels much quieter than their neighbourhood = static overlay."""
    loc = median_filter(std, size=15)
    m = std / (loc + 1e-9) < 0.55
    return binary_dilation(m, iterations=1)


def make_interpolator(mask):
    """Row-wise periodic linear interpolation across masked pixels."""
    nx = mask.shape[1]
    x = np.arange(nx)
    plan = []
    for r in np.where(mask.any(axis=1))[0]:
        bad = mask[r]
        if bad.mean() > 0.5:
            continue
        plan.append((r, x[bad], x[~bad]))

    def apply(F):
        for r, xb, xg in plan:
            F[r, xb] = np.interp(xb, xg, F[r, xg], period=nx)
        return F
    return apply, len(plan)


def row_spectra(F):
    """|rFFT_x|^2 per row after removing each row's mean. Drops k = 0."""
    F = F - F.mean(axis=1, keepdims=True)
    P = np.abs(np.fft.rfft(F, axis=1)) ** 2 / F.shape[1]
    return P[:, 1:]


def run(tag):
    st = np.load(f"{CACHE}/frame_stats_{tag}.npz")
    cbar = np.load(f"{CACHE}/colour_bar_{tag}.npz")["cbar"]
    lut = build_lut(st["mean"], cbar)

    lr0, lr1, lc0, lc1 = LEGACY[tag]
    r0, r1, c0, c1 = DYN[tag]
    r0, r1, c0, c1 = r0 + INSET, r1 - INSET, c0 + INSET, c1 - INSET

    mask = overlay_mask(st["std"][r0:r1, c0:c1])
    interp, nrows_fixed = make_interpolator(mask)
    nr = r1 - r0
    lowz_from = int(np.ceil(LOWZ_CUT * nr))        # image row 0 is z = 1
    band_edges = np.linspace(0, nr, NBANDS + 1).astype(int)

    out = {k: [] for k in ("legacy", "tight", "clean", "lowz", "bands", "ptype")}
    t0 = time.time()
    c = av.open(FILES[tag])
    for i, fr in enumerate(c.decode(video=0)):
        Y = fr.to_ndarray(format="yuv420p")[:fr.height, :fr.width]
        out["ptype"].append(str(fr.pict_type)[-1])

        out["legacy"].append(row_spectra(lut[Y[lr0:lr1, lc0:lc1]]).mean(0))

        F = lut[Y[r0:r1, c0:c1]]
        out["tight"].append(row_spectra(F).mean(0))

        Fc = interp(F.copy())
        P = row_spectra(Fc)
        out["clean"].append(P.mean(0))
        out["lowz"].append(P[lowz_from:].mean(0))
        out["bands"].append(np.stack([P[band_edges[b]:band_edges[b + 1]].mean(0)
                                      for b in range(NBANDS)]))
    c.close()

    arrs = {k: np.asarray(v, dtype=np.float32) for k, v in out.items() if k != "ptype"}
    np.savez_compressed(f"{CACHE}/x_spectra_{tag}.npz", **arrs,
                        ptype=np.array(out["ptype"]), mask=mask,
                        band_edges=band_edges, crop=np.array([r0, r1, c0, c1]),
                        lut=lut, lowz_from=lowz_from)
    n = len(out["ptype"])
    print(f"{tag}: {n} frames, crop {nr}x{c1 - c0}, masked px {int(mask.sum())}, "
          f"rows interpolated {nrows_fixed}, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    for tag in (sys.argv[1:] or FILES):
        run(tag)
