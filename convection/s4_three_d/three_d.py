"""The 3D movie (Ra = 1e8): straighten each face of the temperature box and measure its spectra.

The movie draws the box at an angle, with lighting. Each face is a slightly perspective quadrilateral, and
its colours are the inferno colours multiplied by one constant lighting factor s (one s per face).
  1. Fit s for each face (RGB of face pixels against s * colour-bar RGB).
  2. Invert brightness with the unlit colour bar: Y_unlit = 16 + (Y - 16) / s  (limited-range luma is linear in RGB).
  3. Straighten each face with a 4-corner perspective transform (cv2.warpPerspective).
  4. Walls (vertical faces): FFT along the horizontal direction, which is periodic.
     Floor and lid (horizontal faces): periodic in both directions, so a full 2D FFT is valid.
"""
from convection.config import CACHE, DATA
import sys, time
import av, cv2
import numpy as np

PATH = f"{DATA}/3d_Ra1e8.mp4"
TLIM = (-0.1, 0.0)
# corners in the full-resolution frame (x, y), found with a Hough transform of the static white edges
# order: top-left, top-right, bottom-right, bottom-left of the straightened face
FACES = {
    "wall_xz":   [(208, 517), (724, 702), (724, 888), (215, 688)],      # top view, front-left wall
    "wall_yz":   [(724, 702), (1234, 516), (1234, 687), (724, 888)],    # top view, front-right wall
    "wall_back1": [(214, 1362), (724, 1163), (724, 1347), (209, 1533)],  # bottom view, back-left wall
    "wall_back2": [(724, 1163), (1233, 1362), (1234, 1532), (724, 1347)],  # bottom view, back-right wall
    "lid":       [(208, 517), (724, 360), (1234, 516), (724, 702)],     # top face, z = 1
    "floor":     [(209, 1533), (724, 1347), (1234, 1532), (724, 1689)],  # bottom face, z = 0
}
SIZE = {"wall": (544, 184), "lid": (544, 544), "floor": (544, 544)}
INSET = 8
CBAR_COLS = (1326, 1362)


def out_size(name):
    return SIZE["wall"] if name.startswith("wall") else SIZE[name]


def homographies():
    H = {}
    for name, q in FACES.items():
        W, Hh = out_size(name)
        dst = np.float32([[0, 0], [W - 1, 0], [W - 1, Hh - 1], [0, Hh - 1]])
        H[name] = cv2.getPerspectiveTransform(np.float32(q), dst)
    return H


def face_mask(name, shape):
    m = np.zeros(shape, np.uint8)
    cv2.fillConvexPoly(m, np.int32(FACES[name]), 1)
    m = cv2.erode(m, np.ones((2 * INSET + 1, 2 * INSET + 1), np.uint8))
    return m.astype(bool)


def colour_bar(rgb, Y, mean_std):
    """Rows of the unlit colour bar, its RGB and luma profiles (top = value 0, bottom = -0.1)."""
    mean, std = mean_std
    col = (CBAR_COLS[0] + CBAR_COLS[1]) // 2
    rows = np.nonzero((mean[300:900, col] > 20) & (mean[300:900, col] < 250))[0] + 300
    runs = np.split(rows, np.where(np.diff(rows) > 1)[0] + 1)
    r = max(runs, key=len)
    r0, r1 = r[0] + 3, r[-1] - 2
    c0, c1 = CBAR_COLS
    prgb = np.median(rgb[r0:r1, c0:c1].astype(float), axis=1)
    py = np.median(mean[r0:r1, c0:c1], axis=1)
    return (r0, r1), prgb, py


def fit_light(rgb, mask, prgb):
    P = rgb[mask].astype(float)[::37]
    best = None
    for s in np.linspace(0.25, 1.3, 106):
        d = np.sqrt(((P[:, None, :] - s * prgb[None]) ** 2).sum(-1)).min(1)
        m = np.median(d)
        if best is None or m < best[0]:
            best = (m, s)
    return best[1], best[0]


def luma_lut(py):
    """Luma (float) -> value, from the colour-bar profile (monotone enforced)."""
    py = np.maximum.accumulate(py[::-1])[::-1]
    pos = np.linspace(1.0, 0.0, len(py))
    o = np.argsort(py, kind="stable")
    u, inv = np.unique(py[o], return_inverse=True)
    pm = np.bincount(inv, weights=pos[o]) / np.bincount(inv)
    return u, TLIM[0] + (TLIM[1] - TLIM[0]) * pm


def shell_spectrum(F):
    """Shell-averaged 2D spectrum of a doubly periodic field (k = whole waves across the box)."""
    F = F - F.mean()
    P = np.abs(np.fft.fft2(F)) ** 2 / F.size
    ny, nx = F.shape
    kx = np.fft.fftfreq(nx) * nx
    ky = np.fft.fftfreq(ny) * ny
    K = np.rint(np.hypot(kx[None, :], ky[:, None])).astype(int)
    kmax = min(nx, ny) // 2
    E = np.bincount(K.ravel(), weights=P.ravel(), minlength=kmax + 1)[1:kmax + 1]
    return E


def rows_spectrum(F):
    F = F - F.mean(axis=1, keepdims=True)
    return (np.abs(np.fft.rfft(F, axis=1)) ** 2 / F.shape[1])[:, 1:]


def run():
    st = np.load(f"{CACHE}/tstats_3d.npz")
    mean, std = st["mean"], st["std"]
    Hm = homographies()
    # calibration from frame 500
    c = av.open(PATH)
    for i, fr in enumerate(c.decode(video=0)):
        if i == 500:
            rgb = fr.to_ndarray(format="rgb24")
            break
    c.close()
    (cr0, cr1), prgb, py = colour_bar(rgb, None, (mean, std))
    u, val = luma_lut(py)
    light = {}
    masks = {}
    for name in FACES:
        masks[name] = face_mask(name, mean.shape)
        s, d = fit_light(rgb, masks[name], prgb)
        light[name] = s
        print(f"{name:11s} light factor {s:.3f}  (median RGB distance {d:.1f})", flush=True)

    out = {k: [] for k in FACES}
    samples = {}
    t0 = time.time()
    c = av.open(PATH)
    for i, fr in enumerate(c.decode(video=0)):
        Y = fr.to_ndarray(format="yuv420p")[:fr.height, :fr.width].astype(np.float32)
        for name in FACES:
            s = light[name]
            Yu = 16 + (Y - 16) / s
            T = np.interp(Yu, u, val).astype(np.float32)
            W, Hh = out_size(name)
            F = cv2.warpPerspective(T, Hm[name], (W, Hh), flags=cv2.INTER_LINEAR)
            F = F[INSET:-INSET, INSET:-INSET].astype(np.float64)
            if name.startswith("wall"):
                out[name].append(rows_spectrum(F))           # rows x k  (row 0 = z = 1)
            else:
                Ex = rows_spectrum(F).mean(0)
                Ey = rows_spectrum(F.T).mean(0)
                out[name].append(np.concatenate([shell_spectrum(F), [np.nan], Ex, [np.nan], Ey]))
            if i in (0, 500, 1000):
                samples[f"{name}_{i}"] = F.astype(np.float32)
    c.close()
    arr = {}
    for name, v in out.items():
        a = np.asarray(v, dtype=np.float32)
        if name.startswith("wall"):
            nb = 5
            e = np.linspace(0, a.shape[1], nb + 1).astype(int)
            arr[name] = a.mean(1)                                          # frames x k
            arr[name + "_bands"] = np.stack([a[:, e[b]:e[b + 1]].mean(1) for b in range(nb)], 1)
        else:
            arr[name] = a
    np.savez_compressed(f"{CACHE}/spec3d.npz", **arr, **samples,
                        light=np.array([light[k] for k in FACES]), faces=np.array(list(FACES)))
    print(f"done, {len(out['lid'])} frames, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    run()
