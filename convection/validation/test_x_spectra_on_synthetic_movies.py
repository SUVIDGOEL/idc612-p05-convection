"""End-to-end test of the extraction method on movies with a KNOWN spectrum.

1. Make a slowly evolving random field whose x-spectrum is a power law k^(-alpha).
2. Paint it with the inferno colormap in the same frame layout as 2d_Ra1e9.mp4
   (panel, white border, colour bar), and encode it with H.264 yuv420p at the same
   bit rate and GOP length as the real movie.
3. Run the same pipeline: temporal statistics -> crop -> colour-bar LUT -> x-spectra.
4. Compare the recovered slope with the slope of the true (unpainted) field.
"""
from convection.config import FFMPEG, SYNTH
import sys, subprocess, os
import numpy as np, matplotlib, av
from convection.spectrum_1d.measure_x_spectra import build_lut, row_spectra
from convection.common.fits_and_errors import slope

W, H = 1600, 912
PR0, PR1, PC0, PC1 = 120, 439, 147, 1410          # panel = dynamic region of Ra1e9
CB = (120, 439, 1425, 1443)                         # colour bar columns
NF = 300
INFERNO = matplotlib.colormaps["inferno"]


def make_fields(alpha, nf, seed):
    nz, nx = PR1 - PR0, PC1 - PC0
    rng = np.random.default_rng(seed)
    kx = np.fft.rfftfreq(nx) * nx
    kz = np.fft.fftfreq(nz) * nz
    K = np.hypot(kx[None, :], kz[:, None]); K[0, 0] = 1.0
    amp = K ** (-(alpha + 1) / 2)                   # 2D isotropic -> 1D x-spectrum ~ k^-alpha
    amp[0, 0] = 0.0
    phi0 = rng.uniform(0, 2 * np.pi, amp.shape)
    omega = rng.normal(0, 0.04, amp.shape)          # slow drift -> correlated frames
    scale = None
    for t in range(nf):
        f = np.fft.irfft2(amp * np.exp(1j * (phi0 + omega * t)), s=(nz, nx))
        if scale is None:
            scale = f.std()
        yield f / scale


def render(fields, path, bitrate="1063k"):
    """Paint fields with inferno in the Ra1e9 layout and encode with x264."""
    cmd = [FFMPEG, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", "30", "-i", "-", "-c:v", "libx264", "-b:v", bitrate,
           "-g", "250", "-bf", "3", "-pix_fmt", "yuv420p", path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    bar = np.linspace(1, 0, CB[1] - CB[0])[:, None] * np.ones((1, CB[3] - CB[2]))
    truth = []
    for f in fields:
        pos = np.clip(0.62 + 0.11 * f, 0, 1)        # mostly orange, dark plumes, like the data
        truth.append(row_spectra(pos[2:-2, 2:-2]).mean(0))
        img = np.zeros((H, W, 3), np.uint8)
        img[PR0:PR1, PC0:PC1] = (INFERNO(pos)[..., :3] * 255).astype(np.uint8)
        img[PR0 - 1, PC0 - 1:PC1 + 1] = img[PR1, PC0 - 1:PC1 + 1] = 230   # white border
        img[PR0 - 1:PR1 + 1, PC0 - 1] = img[PR0 - 1:PR1 + 1, PC1] = 230
        img[CB[0]:CB[1], CB[2]:CB[3]] = (INFERNO(bar)[..., :3] * 255).astype(np.uint8)
        p.stdin.write(img.tobytes())
    p.stdin.close(); p.wait()
    return np.array(truth)


def recover(path):
    """The real pipeline: temporal mean for the colour bar, then per-frame x-spectra."""
    c = av.open(path); n = 0; mean = None
    for fr in c.decode(video=0):
        Y = fr.to_ndarray(format="yuv420p")[:H, :W].astype(np.float64)
        n += 1; mean = Y if mean is None else mean + (Y - mean) / n
    c.close()
    lut = build_lut(mean, np.array([CB[0], CB[1], CB[2] + 2, CB[3] - 1]))
    out = []
    c = av.open(path)
    for fr in c.decode(video=0):
        Y = fr.to_ndarray(format="yuv420p")[:H, :W]
        out.append(row_spectra(lut[Y[PR0 + 2:PR1 - 2, PC0 + 2:PC1 - 2]]).mean(0))
    c.close()
    return np.array(out)


if __name__ == "__main__":
    os.makedirs(f"{SYNTH}", exist_ok=True)
    print("%-7s %-9s %-10s %-10s %-10s" % ("alpha", "bitrate", "true", "recovered", "error"))
    rows = []
    for alpha in (1.4, 1.67, 2.0, 2.4):
        for br in ("1063k", "300k"):
            path = f"{SYNTH}/alpha{alpha}_{br}.mp4"
            truth = render(make_fields(alpha, NF, seed=int(alpha * 100)), path, br)
            rec = recover(path)
            st, sr = slope(truth.mean(0), 10, 30), slope(rec.mean(0), 10, 30)
            rows.append((alpha, br, st, sr))
            print("%-7.2f %-9s %-10.3f %-10.3f %+.3f" % (alpha, br, st, sr, sr - st), flush=True)
    np.save(f"{SYNTH}/x_spectra_tests.npy", np.array(rows, dtype=object), allow_pickle=True)
