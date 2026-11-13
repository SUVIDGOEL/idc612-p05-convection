"""Compare the outputs of this codebase with a reference folder (e.g. the original p05/cache and p05/slides_figs).

usage: python tools/compare_outputs.py REF_CACHE REF_FIGS
For every .npy/.npz/.json in cache/ that also exists in REF_CACHE: arrays must agree to rtol 1e-9.
For every .png in figures/ that also exists in REF_FIGS: pixels must be identical (reports the max difference).
"""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def same(a, b, path=""):
    if isinstance(a, dict):
        if set(a) != set(b):
            return [f"{path}: keys differ"]
        return [m for k in a for m in same(a[k], b[k], f"{path}/{k}")]
    if isinstance(a, (list, tuple)) and not np.isscalar(a):
        if len(a) != len(b):
            return [f"{path}: lengths differ"]
        return [m for i, (x, y) in enumerate(zip(a, b)) for m in same(x, y, f"{path}[{i}]")]
    try:
        A, B = np.asarray(a), np.asarray(b)
        if A.dtype == object:
            return same(A.tolist(), B.tolist(), path)
        if A.shape != B.shape:
            return [f"{path}: shape {A.shape} vs {B.shape}"]
        if A.dtype.kind in "fc":
            ok = np.allclose(A, B, rtol=1e-9, atol=0, equal_nan=True)
        else:
            ok = np.array_equal(A, B)
        return [] if ok else [f"{path}: values differ (max abs diff {np.nanmax(np.abs(A.astype(float) - B.astype(float))):.3g})"]
    except Exception as e:                                    # noqa: BLE001
        return [] if a == b else [f"{path}: differ ({e})"]


def load(p):
    if p.suffix == ".npy":
        x = np.load(p, allow_pickle=True)
        return x.item() if x.dtype == object and x.shape == () else x
    if p.suffix == ".npz":
        d = np.load(p, allow_pickle=True); return {k: d[k] for k in d.files}
    return json.loads(p.read_text())


if __name__ == "__main__":
    ref_cache, ref_figs = Path(sys.argv[1]), Path(sys.argv[2])
    since = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0      # only files written after this time
    bad = 0; n = 0
    for p in sorted((ROOT / "cache").iterdir()):
        q = ref_cache / p.name
        if p.suffix in (".npy", ".npz", ".json") and q.exists() and p.stat().st_mtime > since:
            n += 1; msgs = same(load(p), load(q))
            print(("OK   " if not msgs else "DIFF ") + p.name + ("" if not msgs else "  " + "; ".join(msgs[:3])))
            bad += bool(msgs)
    from PIL import Image
    for p in sorted((ROOT / "figures").glob("*.png")):
        q = ref_figs / p.name
        if q.exists() and p.stat().st_mtime > since:
            n += 1
            A = np.asarray(Image.open(p).convert("RGB"), dtype=int); B = np.asarray(Image.open(q).convert("RGB"), dtype=int)
            if A.shape != B.shape:
                print(f"DIFF {p.name}  size {A.shape} vs {B.shape}"); bad += 1; continue
            d = np.abs(A - B); frac = (d.max(-1) > 0).mean()
            print(("OK   " if frac == 0 else "DIFF ") + f"{p.name}  pixels changed {100 * frac:.3f}%")
            bad += frac > 0
    print(f"\n{n} outputs compared, {bad} differ")
