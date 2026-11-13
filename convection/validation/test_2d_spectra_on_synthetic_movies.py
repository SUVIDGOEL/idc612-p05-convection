"""End-to-end test of the vertical (mirror / DST) method on movies with a KNOWN answer.

Truth: theta'(x, z, t) = mean profile with two boundary layers (known sine modes)
                         + fluctuations with a known 2D spectrum, shell E(K) ~ K^-alpha, zero at the plates.
Render: inferno colours of theta = theta' - eps z in the Ra1e9 layout (white axis lines, colour bar),
        H.264 yuv420p at the Ra1e9 bit rate (the worst bits-per-pixel of the three movies) and at 300k.
Recover: the same steps as mirror_modes_2d.py -- line centres (plate_positions), colour-bar LUT, data rows, wall padding,
         DST-I in z x rFFT in x, shells.
Compare recovered and true: shell slope, z-slope, x-slope, mean-profile modes, upper-branch slope.
"""
from convection.config import FFMPEG, SYNTH
import sys, os, subprocess
import numpy as np, matplotlib, av
from scipy.fft import irfft, idst, dst, rfft
from convection.common import mirror_transform as V
from convection.geometry.plate_positions import measure, plates
from convection.spectrum_1d.measure_x_spectra import build_lut

W, H = 1600, 912
TOP, BOT, LEFT, RIGHT = 118, 439, 146, 1410            # 1-px white lines (centres)
Nz, Nx = BOT - TOP, RIGHT - LEFT                        # 321, 1264
CB = (120, 439, 1425, 1443)
EPS = 0.1
NF = 200
INF = matplotlib.colormaps["inferno"]
OUT = f"{SYNTH}"


def mean_profile(z, db=0.01, dt=0.02):
    th = -0.025 - 0.005 * z + 0.025 * np.exp(-z / db) - 0.07 * np.exp(-(1 - z) / dt)
    th = th - (th[0] - 0.0) * (1 - z) - (th[-1] + 0.1) * z      # exact plate values 0 and -eps
    return th + EPS * z                                         # theta', zero at both plates


def fields(alpha, seed, amp=0.006, db=0.01, dt=0.02):
    rng = np.random.default_rng(seed)
    n = np.arange(1, Nz)[:, None]; m = np.arange(Nx // 2 + 1)[None, :]
    K = np.hypot(m / 2.0, n)                                    # K / pi
    a = K ** (-(alpha + 1) / 2) * (m >= 1)
    a[:, m[0] > Nx // 2 - 2] = 0
    phi = rng.uniform(0, 2 * np.pi, a.shape)
    om = rng.normal(0, 0.05, a.shape)
    z = np.arange(Nz + 1) / Nz
    mp = mean_profile(z, db, dt)
    scale = None
    for t in range(NF):
        A = a * np.exp(1j * (phi + om * t))
        Fz = idst(A.real, type=1, axis=0) + 1j * idst(A.imag, type=1, axis=0)     # n -> interior z
        f = irfft(Fz, n=Nx, axis=1)
        if scale is None:
            scale = amp / f.std()
        G = np.zeros((Nz + 1, Nx)); G[1:-1] = f * scale
        yield G + mp[:, None]


def render(gen, path, br, fps=30):
    cmd = [FFMPEG, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(fps), "-i", "-", "-c:v", "libx264", "-b:v", br, "-g", "250", "-bf", "3", "-pix_fmt", "yuv420p", path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    z = np.arange(Nz + 1) / Nz
    bar = np.linspace(1, 0, CB[1] - CB[0])[:, None] * np.ones((1, CB[3] - CB[2]))
    truth = []
    for G in gen:
        truth.append(G.astype(np.float32))
        th = G - EPS * z[:, None]
        pos = np.clip(th / 0.1 + 1, 0, 1)
        img = np.zeros((H, W, 3), np.uint8)
        # grid row j (z = j/Nz) is image row BOT - j; column i is image column LEFT + i
        blk = (INF(pos[::-1])[..., :3] * 255).astype(np.uint8)         # rows from z=1 (TOP) to z=0 (BOT)
        img[TOP:BOT + 1, LEFT:RIGHT] = blk
        img[TOP:BOT + 1, RIGHT] = blk[:, 0]                            # periodic copy under the right line
        img[TOP, LEFT - 1:RIGHT + 2] = 235; img[BOT, LEFT - 1:RIGHT + 2] = 235
        img[TOP:BOT + 1, LEFT] = 235; img[TOP:BOT + 1, RIGHT] = 235
        img[CB[0]:CB[1], CB[2]:CB[3]] = (INF(bar)[..., :3] * 255).astype(np.uint8)
        p.stdin.write(img.tobytes())
    p.stdin.close(); p.wait()
    return np.array(truth)


def analyse(G, Ssin):
    A = rfft(G[1:-1], axis=1)
    C = dst(A.real, type=1, axis=0) + 1j * dst(A.imag, type=1, axis=0)
    nz, nx = G.shape[0] - 1, G.shape[1]
    E = np.abs(C) ** 2 * V.xw(nx)[None, :] / (2.0 * nx ** 2 * nz ** 2)
    return dict(shell=Ssin.shell(E), mean=E[:, 0], z=E[:, 1:].sum(1), x=E[:, 1:].sum(0))


def recover(path):
    c = av.open(path); n = 0; mean = None; m2 = None
    for fr in c.decode(video=0):
        Y = fr.to_ndarray(format="yuv420p")[:H, :W].astype(np.float64)
        n += 1
        if mean is None: mean = np.zeros_like(Y); m2 = np.zeros_like(Y)
        d = Y - mean; mean += d / n; m2 += d * (Y - mean)
    c.close()
    g = measure(mean, TOP + 2, BOT, LEFT + 1, RIGHT)
    y1, y0, w, _ = plates(mean, g)
    lut = build_lut(mean, np.array([CB[0], CB[1], CB[2] + 2, CB[3] - 1]))
    Hp = y0 - y1
    nz, nx = int(round(Hp)), int(round(g["W"]))
    r0 = int(np.ceil(y1 + w / 2 + 1)); r1 = int(np.floor(y0 - w / 2 - 1))
    c0 = int(np.ceil(g["x_left"] + w / 2 + 1)); c1 = int(np.floor(g["x_right"] - w / 2 - 1))
    zr = (y0 - np.arange(r0, r1 + 1)) / Hp
    M, _, _ = V.zinterp_matrix(zr, nz, walls=(0.0, 0.0))
    cref = int(round(g["x_left"]))
    gidx, wl, wf = V.gap_plan(c0 - cref, c1 - cref, nx)
    S = V.Shells(nz, nx, "sin")
    res = []
    c = av.open(path)
    for fr in c.decode(video=0):
        Y = fr.to_ndarray(format="yuv420p")[:H, :W]
        th = -0.1 + 0.1 * lut[Y[r0:r1 + 1, c0:c1 + 1]] + EPS * zr[:, None]
        F = np.empty((len(zr), nx)); F[:, c0 - cref:c1 - cref + 1] = th
        F[:, gidx] = th[:, -1:] * wl + th[:, :1] * wf
        res.append(analyse(M @ F, S))
    c.close()
    geo = dict(z1=y1, z0=y0, x_left=g["x_left"], x_right=g["x_right"], nz=nz, nx=nx, rows=(r0, r1), cols=(c0, c1))
    return {k: np.array([r[k] for r in res]) for k in res[0]}, geo


def sl(E, lo, hi, k=None):
    k = np.arange(len(E)) if k is None else k
    m = (k >= lo) & (k <= hi) & (E > 0)
    return np.polyfit(np.log10(k[m]), np.log10(E[m]), 1)[0]


CONFIGS = {"set1": [(a, br, 30, 0.01, 0.02) for a in (5 / 3, 2.4) for br in ("1063k", "300k")],
           # bits per pixel of the Ra1e12 movie (2.36M here) and Ra1e15 movie (2758k at 10 fps); sharp sub-pixel layers
           "set2": [(5 / 3, "2360k", 30, 0.01, 0.02), (5 / 3, "2758k", 10, 0.01, 0.02),
                    (5 / 3, "1063k", 30, 0.0005, 0.0005), (5 / 3, "2360k", 30, 0.0005, 0.0005),
                    (2.4, "2360k", 30, 0.01, 0.02)]}

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    S = V.Shells(Nz, Nx, "sin")
    rows = []
    which = sys.argv[1] if len(sys.argv) > 1 else "set1"
    for alpha, br, fps, db, dt in CONFIGS[which]:
        if True:
            path = f"{OUT}/vert_a{alpha:.2f}_{br}_f{fps}_d{db}.mp4"
            truth = render(fields(alpha, int(alpha * 100), db=db, dt=dt), path, br, fps)
            tr = [analyse(G.astype(float), S) for G in truth]
            tr = {k: np.array([r[k] for r in tr]) for k in tr[0]}
            rec, geo = recover(path)
            kz = np.arange(1, Nz)
            line = dict(alpha=alpha, br=br, fps=fps, db=db, dt=dt, geo=geo)
            for nm, E_t, E_r, k, ranges in (
                    ("shell", tr["shell"].mean(0), rec["shell"].mean(0), None, ((5, 15), (10, 40), (40, 100))),
                    ("z", tr["z"].mean(0), rec["z"].mean(0)[:Nz - 1], kz, ((5, 15), (10, 40), (40, 100))),
                    ("x", tr["x"].mean(0), rec["x"].mean(0)[:Nx // 2], np.arange(1, Nx // 2 + 1), ((10, 30), (20, 80), (80, 200)))):
                for lo, hi in ranges:
                    st_, sr_ = sl(E_t, lo, hi, k), sl(E_r, lo, hi, k)
                    line[f"{nm}_{lo}_{hi}"] = (st_, sr_)
            mt, mr = tr["mean"].mean(0), rec["mean"].mean(0)[:Nz - 1]
            line["mean_amp_ratio_n1_10"] = float(np.sqrt(mr[:10] / mt[:10]).mean())
            for lo, hi in ((2, 20), (20, 60)):
                line[f"mean_{lo}_{hi}"] = (sl(mt, lo, hi, kz), sl(mr, lo, hi, kz))
            rows.append(line)
            print(f"alpha {alpha:.2f} br {br} fps {fps} BL {db}/{dt}: plates found at rows {geo['z1']:.2f} / {geo['z0']:.2f} (true {TOP}/{BOT}), "
                  f"x lines {geo['x_left']:.2f}/{geo['x_right']:.2f} (true {LEFT}/{RIGHT}); grid {geo['nz']}x{geo['nx']}", flush=True)
            for k_, v in line.items():
                if isinstance(v, tuple) and len(v) == 2 and not isinstance(v[0], int):
                    print(f"    {k_:16s} true {v[0]:+.3f}  recovered {v[1]:+.3f}  error {v[1] - v[0]:+.3f}")
            print(f"    mean-profile amplitude ratio (recovered/true, n=1..10): {line['mean_amp_ratio_n1_10']:.3f}", flush=True)
    np.save(f"{OUT}/spectra_2d_tests_{which}.npy", np.array(rows, dtype=object), allow_pickle=True)
