"""3D movie: invert the DENSITY faces and measure T-rho coherence on every face.

The density box is drawn with lighting that also greys the colours. Per face we fit a colour model
    colour(v) = (1 - a) * s * seismic(v) + a * G          (s = light factor, a = grey mix, G = grey level)
to the face pixels, then invert each pixel to the nearest colour on that curve (RGB nearest neighbour,
through a 64^3 lookup table). The T faces use the method of three_d.py. Both are straightened onto the same grid.
"""
from convection.config import CACHE
import sys, time
import av, cv2
import numpy as np
import matplotlib
from scipy.spatial import cKDTree
from convection.s4_three_d import three_d as T3
PATH = T3.PATH
RFACES = {
    "wall_xz":    [(1725, 517), (2239, 702), (2239, 887), (1730, 688)],
    "wall_yz":    [(2239, 702), (2749, 516), (2748, 687), (2239, 887)],
    "wall_back1": [(1729, 1362), (2239, 1163), (2239, 1348), (1724, 1533)],
    "wall_back2": [(2239, 1163), (2748, 1362), (2749, 1533), (2239, 1348)],
    "lid":        [(1725, 517), (2238, 361), (2749, 516), (2239, 702)],
    "floor":      [(1724, 1533), (2239, 1348), (2749, 1533), (2238, 1688)],
}
SEIS = matplotlib.colormaps["seismic"](np.linspace(0, 1, 512))[:, :3] * 255
VALS = np.linspace(-0.1, 0.1, 512)
Q = 64                                    # RGB quantisation for the lookup table


def homog(q, name):
    W, H = T3.out_size(name)
    return cv2.getPerspectiveTransform(np.float32(q), np.float32([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]]))


def fit_model(P):
    best = None
    for G in (96, 128, 160, 192, 224):
        for a in np.linspace(0, 0.8, 17):
            for s in np.linspace(0.3, 1.4, 45):
                M = (1 - a) * s * SEIS + a * G
                d = np.sqrt(((P[:, None, :] - M[None]) ** 2).sum(-1)).min(1)
                m = np.median(d)
                if best is None or m < best[0]:
                    best = (m, s, a, G)
    return best


def lut_for(s, a, G):
    curve = np.clip((1 - a) * s * SEIS + a * G, 0, 255)
    tree = cKDTree(curve)
    g = (np.arange(Q) + 0.5) * 256 / Q
    grid = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    _, idx = tree.query(grid)
    return VALS[idx].reshape(Q, Q, Q).astype(np.float32)


def rows_fft(F):
    F = F - F.mean(1, keepdims=True)
    return np.fft.rfft(F, axis=1)[:, 1:]


def run(step=2):
    st = np.load(f"{CACHE}/tstats_3d.npz"); mean, std = st["mean"], st["std"]
    HT = T3.homographies(); HR = {n: homog(q, n) for n, q in RFACES.items()}
    c = av.open(PATH)
    for i, fr in enumerate(c.decode(video=0)):
        if i == 500:
            rgb = fr.to_ndarray(format="rgb24"); break
    c.close()
    _, prgb, py = T3.colour_bar(rgb, None, (mean, std)); u, val = T3.luma_lut(py)
    light, luts = {}, {}
    for name in T3.FACES:
        light[name] = T3.fit_light(rgb, T3.face_mask(name, mean.shape), prgb)[0]
        m = np.zeros(mean.shape, np.uint8); cv2.fillConvexPoly(m, np.int32(RFACES[name]), 1)
        m = cv2.erode(m, np.ones((17, 17), np.uint8)).astype(bool)
        P = rgb[m].astype(float)[::97]
        d, s, a, G = fit_model(P)
        luts[name] = lut_for(s, a, G)
        print(f"{name:11s} T light {light[name]:.2f} | rho model s={s:.2f} a={a:.2f} G={G} (median RGB distance {d:.1f})", flush=True)
    acc = {n: {"ET": 0, "ER": 0, "CO": 0, "n": 0} for n in T3.FACES}
    bands = {n: [] for n in T3.FACES}
    samples = {}
    t0 = time.time()
    c = av.open(PATH)
    for i, fr in enumerate(c.decode(video=0)):
        if i % step:
            continue
        Y = fr.to_ndarray(format="yuv420p")[:fr.height, :fr.width].astype(np.float32)
        RGB = fr.to_ndarray(format="rgb24")
        for name in T3.FACES:
            W, H = T3.out_size(name); s_ = T3.INSET
            Tf = cv2.warpPerspective(np.interp(16 + (Y - 16) / light[name], u, val).astype(np.float32), HT[name], (W, H))[s_:-s_, s_:-s_]
            Rw = cv2.warpPerspective(RGB, HR[name], (W, H))[s_:-s_, s_:-s_]
            qi = (Rw.astype(np.int32) * Q) // 256
            Rf = luts[name][qi[..., 0], qi[..., 1], qi[..., 2]]
            a_, b_ = rows_fft(Tf.astype(np.float64)), rows_fft(Rf.astype(np.float64))
            ET, ER, CO = np.abs(a_) ** 2, np.abs(b_) ** 2, (a_ * np.conj(b_)).real
            A = acc[name]; A["ET"] = A["ET"] + ET.mean(0); A["ER"] = A["ER"] + ER.mean(0); A["CO"] = A["CO"] + CO.mean(0); A["n"] += 1
            if name.startswith("wall"):
                e = np.linspace(0, ET.shape[0], 6).astype(int)
                bands[name].append(np.stack([[X[e[b]:e[b + 1]].mean(0) for X in (ET, ER, CO)] for b in range(5)]))
            if i in (500,):
                samples[f"T_{name}"] = Tf; samples[f"R_{name}"] = Rf
    c.close()
    out = {}
    for name, A in acc.items():
        for k in ("ET", "ER", "CO"):
            out[f"{name}_{k}"] = A[k] / A["n"]
        if bands[name]:
            out[f"{name}_bands"] = np.array(bands[name], dtype=np.float32)          # frames x 5 x 3 x k
    np.savez_compressed(f"{CACHE}/rho3d.npz", **out, **samples)
    print(f"done, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    run()
