"""Sub-pixel geometry of the 2D panels: the centres of the four white axis lines (spines).

The axis line is drawn centred on the axis edge, so its centre is the exact position of
x = 0, x = 4, z = 0 (bottom plate) and z = 1 (top plate). For each edge we take the temporal-mean
luma across the line, convert each pixel to a white coverage fraction (outer side against the black
background, inner side against the data next to it), and put the edge at the middle of the covered span.

Output: cache/geom2d.json with, per movie and panel (T, rho):
  y_top, y_bot, x_left, x_right  (pixel coordinates, pixel i covers [i-0.5, i+0.5])
  rows, cols                     first/last data pixel that the line does not touch
"""
from convection.config import CACHE
import json, sys
import numpy as np
DYN = {"Ra1e9": (120, 439, 147, 1410), "Ra1e12": (358, 887, 142, 2252), "Ra1e15": (184, 450, 73, 1125)}


def edge(p, i_out, step):
    """p: 1D luma profile, i_out: an index on the black outside, step: +1 if the inside is at larger index."""
    idx = np.arange(len(p))
    w = int(np.argmax(p if step > 0 else p[::-1]))
    peak = w if step > 0 else len(p) - 1 - w
    white = p[peak]
    black = np.median(p[[min(max(i_out + k, 0), len(p) - 1) for k in range(-2, 3)]])
    inner = np.median(p[[peak + step * k for k in range(4, 9)]])
    cov = np.zeros(len(p))
    # walk outward from the peak
    j = peak
    while 0 <= j < len(p):
        f = (p[j] - black) / (white - black)
        if f <= 0.02:
            break
        cov[j] = min(f, 1.0); j -= step
    j = peak + step
    while 0 <= j < len(p):
        f = (p[j] - inner) / (white - inner)
        if f <= 0.02:
            break
        cov[j] = min(f, 1.0); j += step
    on = np.nonzero(cov)[0]
    width = cov.sum()
    lo, hi = on.min(), on.max()
    if step > 0:          # outside is at low index: outer boundary = lo - 0.5 + (1 - cov[lo])
        outer = lo - 0.5 + (1 - cov[lo]); centre = outer + width / 2
    else:
        outer = hi + 0.5 - (1 - cov[hi]); centre = outer - width / 2
    return float(centre), float(width), int(hi if step > 0 else lo)


def measure(mean, r0, r1, c0, c1, pad=14):
    rr = slice(r0 + 40, r1 - 40); cc = slice(c0 + 60, c1 - 60)
    top = np.median(mean[r0 - pad:r0 + pad, cc], axis=1)
    bot = np.median(mean[r1 - pad:r1 + pad, cc], axis=1)
    lef = np.median(mean[rr, c0 - pad:c0 + pad], axis=0)
    rig = np.median(mean[rr, c1 - pad:c1 + pad], axis=0)
    yt, wt, last_t = edge(top, 0, +1)
    yb, wb, first_b = edge(bot, 2 * pad - 3, -1)
    xl, wl, last_l = edge(lef, 0, +1)
    xr, wr, first_r = edge(rig, 2 * pad - 3, -1)
    g = dict(y_top=r0 - pad + yt, y_bot=r1 - pad + yb, x_left=c0 - pad + xl, x_right=c1 - pad + xr,
             widths=[wt, wb, wl, wr])
    g["rows"] = [r0 - pad + last_t + 1, r1 - pad + first_b - 1]      # inclusive data rows
    g["cols"] = [c0 - pad + last_l + 1, c1 - pad + first_r - 1]
    g["H"] = g["y_bot"] - g["y_top"]; g["W"] = g["x_right"] - g["x_left"]
    g["aspect"] = g["W"] / g["H"]
    return g


if __name__ == "__main__":
    out = {}
    for t, (r0, r1, c0, c1) in DYN.items():
        mean = np.load(f"{CACHE}/tstats_{t}.npz")["mean"]
        p2 = np.load(f"{CACHE}/pass2_{t}.npz")
        rs, cs = int(p2["rshift"]), int(p2["cshift"])
        gT = measure(mean, r0, r1, c0, c1)
        gR = measure(mean, r0 + rs, r1 + rs, c0 + cs, c1 + cs)
        out[t] = {"T": gT, "rho": gR, "rshift": rs, "cshift": cs}
        for nm, g in (("T", gT), ("rho", gR)):
            print(f"{t:6s} {nm:3s} top {g['y_top']:.2f} bot {g['y_bot']:.2f} left {g['x_left']:.2f} right {g['x_right']:.2f}"
                  f" | H {g['H']:.2f} W {g['W']:.2f} aspect {g['aspect']:.3f} | line widths "
                  + " ".join(f"{w:.2f}" for w in g["widths"]) + f" | data rows {g['rows']} cols {g['cols']}")
        print(f"        rho - T offset: rows {gR['y_top'] - gT['y_top']:.2f} / {gR['y_bot'] - gT['y_bot']:.2f}, "
              f"cols {gR['x_left'] - gT['x_left']:.2f} / {gR['x_right'] - gT['x_right']:.2f}  (pass2 alignment {rs}, {cs})")
    json.dump(out, open(f"{CACHE}/geom2d.json", "w"), indent=1)



def centroid_from_black(p, w, white):
    """p runs from the black outside inward. Walk inward and give each pixel its white coverage
    (against black) until the line width w is used up. Returns the centroid (index units of p)."""
    black = np.median(p[:2])
    cum, num = 0.0, 0.0
    for i, v in enumerate(p):
        f = float(np.clip((v - black) / (white - black), 0, 1))
        f = min(f, w - cum)
        cum += f; num += i * f
        if cum >= w - 1e-9:
            break
    return num / cum


def plates(mean, g, pad=6):
    """z = 1 and z = 0 line centres. w = mean width of the two clean x-lines, white = their peak luma."""
    c0, c1 = int(g["x_left"]), int(g["x_right"])
    rr = slice(int(g["y_top"]) + 40, int(g["y_bot"]) - 40)
    w = 0.5 * (g["widths"][2] + g["widths"][3])
    white = 0.5 * (np.median(mean[rr, c0 - 3:c0 + 4], 0).max() + np.median(mean[rr, c1 - 3:c1 + 4], 0).max())
    cc = slice(c0 + 60, c1 - 60)
    yt0, yb0 = int(round(g["y_top"])), int(round(g["y_bot"]))
    top = np.median(mean[yt0 - pad:yt0 + pad, cc], axis=1)
    bot = np.median(mean[yb0 - pad:yb0 + pad, cc], axis=1)[::-1]
    y_top = yt0 - pad + centroid_from_black(top, w, white)
    y_bot = yb0 + pad - 1 - centroid_from_black(bot, w, white)
    return y_top, y_bot, w, white


if __name__ == "__main__":
    print("\nPlate line centres (centroid from the black side, width of the x-lines):")
    for t in DYN:
        mean = np.load(f"{CACHE}/tstats_{t}.npz")["mean"]
        for nm in ("T", "rho"):
            g = out[t][nm]
            yt, yb, w, wh = plates(mean, g)
            g.update(z1=yt, z0=yb, Hz=yb - yt, aspect_z=g["W"] / (yb - yt))
            print(f"{t:6s} {nm:3s} z=1 at {yt:.2f}  z=0 at {yb:.2f}  H {yb - yt:.2f}  W/H {g['W'] / (yb - yt):.3f}  (w {w:.2f}, white {wh:.0f})")
        a, b = out[t]["T"], out[t]["rho"]
        print(f"        rho-T offset: top {b['z1'] - a['z1']:.2f}, bottom {b['z0'] - a['z0']:.2f}, left {b['x_left'] - a['x_left']:.2f}, right {b['x_right'] - a['x_right']:.2f}")
    json.dump(out, open(f"{CACHE}/geom2d.json", "w"), indent=1)
