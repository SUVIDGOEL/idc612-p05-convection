"""Core of the vertical analysis (the mirror trick), shared by the 2D pass, the 3D pass and the tests.

The box is periodic in x and has plates at z = 0 and z = 1. A plain FFT in z would see a jump.
Mirror the column about a plate (z -> -z): for a field that is zero at both plates (odd mirror) the
column becomes periodic with period 2 and its Fourier series is a sine series,
    f(z) = sum_n a_n sin(n pi z),      k_z = n pi                    (DST-I on the grid z_j = j/Nz)
For a field that is not zero at the plates we use the even mirror -> cosine series (DCT-I).

theta' = (T - T_A) + eps z is zero at both plates (T fixed: T - T_A = 0 at z = 0, -eps at z = 1).
Its 2D transform C(m, n) = DST_z( rFFT_x ) gives
    m = 0  : the x-averaged profile (the theta(0, n) modes, "upper branch")
    m >= 1 : the fluctuations
Energies are normalised so that the sum over all modes = domain mean of f^2 (trapezoid in z).
Physical wavenumbers: k_x = 2 pi m / Lx (Lx = 4), k_z = n pi.  Shells in units of pi: K/pi = sqrt((m Lx'/... )).
"""
import numpy as np
from scipy.fft import rfft, dst, dct

LX = 4.0


# ---------------------------------------------------------------- grids and filling
def zinterp_matrix(z_src, Nz, walls=None):
    """Linear interpolation matrix from source rows (z_src, any order) to the grid z_j = j/Nz, j = 0..Nz.
    walls = (v0, v1) adds exact plate values at z = 0 and z = 1; otherwise the end values are held."""
    zs = np.asarray(z_src, float)
    order = np.argsort(zs)
    zs = zs[order]
    nsrc = len(zs)
    if walls is not None:
        zs = np.concatenate([[0.0], zs, [1.0]])
    zg = np.arange(Nz + 1) / Nz
    M = np.zeros((Nz + 1, len(zs)))
    k = np.clip(np.searchsorted(zs, zg) - 1, 0, len(zs) - 2)
    t = np.clip((zg - zs[k]) / (zs[k + 1] - zs[k]), 0, 1)
    M[np.arange(Nz + 1), k] = 1 - t
    M[np.arange(Nz + 1), k + 1] += t
    # map back to source order; wall columns (if any) returned separately
    if walls is not None:
        Mw0, Mw1, Ms = M[:, 0], M[:, -1], M[:, 1:-1]
    else:
        Mw0 = Mw1 = np.zeros(Nz + 1); Ms = M
    Mo = np.zeros((Nz + 1, nsrc)); Mo[:, order] = Ms
    return Mo, Mw0, Mw1


def gap_plan(i_first, i_last, Nx):
    """Columns i_first..i_last hold data on a periodic grid of Nx columns; the rest (the axis-line gap)
    is filled by linear interpolation across the wrap. Returns (gap_idx, w_last, w_first)."""
    gap = [(i_last + k) % Nx for k in range(1, Nx - (i_last - i_first))]
    L = len(gap) + 1
    w_first = np.array([(k + 1) / L for k in range(len(gap))])
    return np.array(gap, int), 1 - w_first, w_first


def fill_plan(mask):
    """Static overlay pixels (dashed zoom box, ticks): fill each by linear interpolation along the
    direction (x or z) in which its masked run is shorter. Returns flat index arrays for apply_fill."""
    nz, nx = mask.shape
    idx, n0, n1, w1 = [], [], [], []
    for r, c in zip(*np.nonzero(mask)):
        # run along x (periodic)
        a = c
        while mask[r, a % nx] and c - a < nx: a -= 1
        b = c
        while mask[r, b % nx] and b - c < nx: b += 1
        lx = b - a
        # run along z
        u = r
        while u >= 0 and mask[u, c]: u -= 1
        d = r
        while d < nz and mask[d, c]: d += 1
        okz = u >= 0 and d < nz
        if okz and (d - u) <= lx:
            idx.append(r * nx + c); n0.append(u * nx + c); n1.append(d * nx + c); w1.append((r - u) / (d - u))
        else:
            idx.append(r * nx + c); n0.append(r * nx + a % nx); n1.append(r * nx + b % nx); w1.append((c - a) / lx)
    return tuple(np.array(v) for v in (idx, n0, n1, w1))


def apply_fill(F, plan):
    if len(plan[0]) == 0:
        return F
    f = F.reshape(-1)
    idx, n0, n1, w1 = plan
    f[idx] = (1 - w1) * f[n0] + w1 * f[n1]
    return F


# ---------------------------------------------------------------- transforms
def xw(Nx):
    w = np.full(Nx // 2 + 1, 2.0); w[0] = 1.0
    if Nx % 2 == 0: w[-1] = 1.0
    return w


def spec2d_sine(G):
    """G: (Nz+1, Nx) on the full grid, zero at j = 0 and j = Nz. Returns complex C (Nz-1, Nx//2+1)
    for n = 1..Nz-1, m = 0..Nx//2, and energies E(n, m) that sum to the domain mean of G^2."""
    Nz, Nx = G.shape[0] - 1, G.shape[1]
    A = rfft(G[1:-1], axis=1)
    C = dst(A.real, type=1, axis=0) + 1j * dst(A.imag, type=1, axis=0)
    E = (np.abs(C) ** 2) * xw(Nx)[None, :] / (2.0 * Nx ** 2 * Nz ** 2)
    return C, E


def spec2d_cos(G):
    """Even mirror (DCT-I) in z. G: (Nz+1, Nx). n = 0..Nz. Energies sum to the trapezoid mean of G^2."""
    Nz, Nx = G.shape[0] - 1, G.shape[1]
    A = rfft(G, axis=1)
    C = dct(A.real, type=1, axis=0) + 1j * dct(A.imag, type=1, axis=0)
    wn = np.ones(Nz + 1); wn[[0, -1]] = 0.5
    E = (np.abs(C) ** 2) * xw(Nx)[None, :] * wn[:, None] / (2.0 * Nx ** 2 * Nz ** 2)
    return C, E


def cross2d(Ca, Cb, Nz, Nx, kind):
    """Co-spectrum Re(Ca Cb*) with the same normalisation as the energies."""
    X = (Ca * np.conj(Cb)).real * xw(Nx)[None, :]
    if kind == "cos":
        wn = np.ones(Nz + 1); wn[[0, -1]] = 0.5
        X = X * wn[:, None]
    return X / (2.0 * Nx ** 2 * Nz ** 2)


class Shells:
    """Binning of (n, m) modes into shells of physical K, width pi; and sectors by angle."""

    def __init__(self, Nz, Nx, kind, lx=LX, nsect=3):
        n = np.arange(1, Nz) if kind == "sin" else np.arange(0, Nz + 1)
        m = np.arange(Nx // 2 + 1)
        self.n, self.m = n, m
        kx = 2 * np.pi * m / lx
        kz = np.pi * n
        K = np.hypot(kx[None, :], kz[:, None]) / np.pi
        self.b = np.rint(K).astype(int)
        self.nb = int(min((Nx // 2) * 2 / lx, Nz)) + 1           # shells fully inside the grid
        ang = np.degrees(np.arctan2(kz[:, None] * np.ones_like(kx)[None, :], kx[None, :] * np.ones_like(kz)[:, None]))
        self.sect = np.minimum((ang / (90.0 / nsect)).astype(int), nsect - 1)   # 0 = k mostly along x
        self.nsect = nsect
        self.fluct = (m[None, :] >= 1) & np.ones((len(n), 1), bool)
        self.count = np.bincount(self.b[self.fluct], minlength=self.nb + 1)[:self.nb + 1]

    def shell(self, E, fluct_only=True):
        sel = self.fluct if fluct_only else np.ones_like(self.fluct)
        return np.bincount(self.b[sel], weights=E[sel], minlength=self.nb + 1)[:self.nb + 1]

    def sectors(self, E):
        out = np.zeros((self.nsect, self.nb + 1))
        for s in range(self.nsect):
            sel = self.fluct & (self.sect == s)
            out[s] = np.bincount(self.b[sel], weights=E[sel], minlength=self.nb + 1)[:self.nb + 1]
        return out

    def sector_counts(self):
        return np.stack([np.bincount(self.b[self.fluct & (self.sect == s)], minlength=self.nb + 1)[:self.nb + 1]
                         for s in range(self.nsect)])


# ---------------------------------------------------------------- 1D helpers
def zspec_hann(F, j0, j1):
    """Robustness variant: Hann-windowed FFT along z of rows j0..j1 (bulk only), per column, averaged.
    Returns energy per wavenumber index q (k_z = 2 pi q / Lz_window)."""
    seg = F[j0:j1 + 1]
    seg = seg - seg.mean(0, keepdims=True)
    w = np.hanning(seg.shape[0])[:, None]
    P = np.abs(np.fft.rfft(seg * w, axis=0)) ** 2 / (w ** 2).sum()
    P[1:] *= 2
    return P.mean(1) / seg.shape[0]


def increments(F, rs, axis, p_max=6, rows=None):
    """Moments of increments along x (axis=1, periodic) or z (axis=0, pairs inside rows[0]..rows[1]).
    Returns (len(rs), p_max + 1): columns 0..p_max-1 = <|d|^p> for p = 1..p_max, last = <d^3>."""
    out = np.full((len(rs), p_max + 1), np.nan)
    j0, j1 = rows
    for i, r in enumerate(rs):
        if axis == 1:
            d = np.roll(F[j0:j1 + 1], -r, axis=1) - F[j0:j1 + 1]
        else:
            if j1 - j0 - r < 1:
                continue
            d = F[j0 + r:j1 + 1] - F[j0:j1 + 1 - r]
        a = np.abs(d).astype(np.float64).ravel()
        ap = a.copy()
        for p in range(p_max):
            out[i, p] = ap.mean()
            ap *= a
        out[i, p_max] = (d.astype(np.float64) ** 3).mean()
    return out


def clean_overlay_mask(m, box=None):
    """Keep only thin line-like parts of a static-pixel mask (dashed zoom box, tick marks).
    Drops round blobs (slow vortices are real flow) and plate-hugging horizontal runs (boundary layers)."""
    from scipy import ndimage as ndi
    lab, n = ndi.label(m)
    keep = np.zeros_like(m, bool)
    nz = m.shape[0]
    for i, s in enumerate(ndi.find_objects(lab), 1):
        h, w = s[0].stop - s[0].start, s[1].stop - s[1].start
        comp = lab[s] == i
        thin = min(h, w) <= 6 and max(h, w) >= 2.5 * min(h, w)
        tick = h <= 12 and w <= 4 and (s[0].start <= 3 or s[0].stop >= nz - 3)   # short tick at a plate
        at_plate = s[0].start <= 3 or s[0].stop >= nz - 3
        if at_plate and w > h:
            continue
        near_box = False
        if box is not None:
            b0, b1, a0, a1 = box                          # outline rows b0..b1, cols a0..a1 (data coords)
            rr = (s[0].start - 8, s[0].stop + 8); cc = (s[1].start - 8, s[1].stop + 8)
            on_v = (rr[0] < b1 and rr[1] > b0) and any(cc[0] <= a <= cc[1] for a in (a0, a1))
            on_h = (cc[0] < a1 and cc[1] > a0) and any(rr[0] <= b <= rr[1] for b in (b0, b1))
            near_box = on_v or on_h
        if (thin and near_box) or tick:
            keep[s] |= comp
    return keep
