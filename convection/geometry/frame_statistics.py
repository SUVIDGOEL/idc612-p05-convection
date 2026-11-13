"""Step 0: per-pixel temporal statistics of the luma plane, and the colour-bar boxes.

Convection pixels change from frame to frame; borders, tick marks, the zoom-box overlay, text and colour bars
do not. So the temporal std separates the animated panels from the static frame.

Writes  cache/frame_stats_<tag>.npz  (mean, std, n; every 5th frame)   for the three 2D movies and the 3D movie
        cache/colour_bar_<tag>.npz (cbar = row0, row1, col0, col1 of the temperature colour bar, 2D movies)
The colour-bar boxes were read once from the frame_stats images (the bar is static, bright and next to the T panel).
usage: python -m convection.geometry.frame_statistics
"""
import time
import av
import numpy as np
from convection.config import CACHE, DATA

FILES = {"Ra1e9": f"{DATA}/2d_Ra1e9.mp4", "Ra1e12": f"{DATA}/2d_Ra1e12.mp4", "Ra1e15": f"{DATA}/2d_Ra1e15.mp4",
         "3d": f"{DATA}/3d_Ra1e8.mp4"}
COLOUR_BAR = {"Ra1e9": (120, 439, 1425, 1442), "Ra1e12": (358, 887, 2277, 2311), "Ra1e15": (184, 450, 1138, 1154)}


def temporal_stats(path, every=5):
    c = av.open(path)
    n = 0; mean = None; m2 = None
    for i, fr in enumerate(c.decode(video=0)):
        if i % every:
            continue
        Y = fr.to_ndarray(format="yuv420p")[:fr.height, :fr.width].astype(np.float64)
        n += 1
        if mean is None:
            mean = np.zeros_like(Y); m2 = np.zeros_like(Y)
        d = Y - mean; mean += d / n; m2 += d * (Y - mean)      # Welford running variance
    c.close()
    return mean, np.sqrt(m2 / (n - 1)), n


if __name__ == "__main__":
    for tag, p in FILES.items():
        t0 = time.time()
        mean, std, n = temporal_stats(p)
        np.savez_compressed(f"{CACHE}/frame_stats_{tag}.npz", mean=mean, std=std, n=n)
        print(f"{tag}: {n} frames sampled, image {mean.shape}, std range {std.min():.2f}-{std.max():.2f}, {time.time()-t0:.0f}s")
    for tag, box in COLOUR_BAR.items():
        np.savez(f"{CACHE}/colour_bar_{tag}.npz", cbar=np.array(box))
