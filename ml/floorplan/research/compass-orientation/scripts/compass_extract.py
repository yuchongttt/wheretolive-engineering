#!/usr/bin/env python3
"""Automatic compass-angle extractor v1 -- an ensemble of geometric methods, picked by a validity score
(does not trust VLM style labels). Part of the frozen v2.2 research pipeline.

extract_north_multi(path, windows) -> best {"north_deg", "clock", "conf"} | None
windows: [(cx, cy, r_hint)] in 1288-px coordinates; converted back to native resolution for cropping.
Angle: 0 = page up, clockwise; clock = deg / 30.
"""
from __future__ import annotations

import math

import numpy as np
from PIL import Image
from scipy import ndimage


def _load_rgb(path) -> Image.Image:
    im = Image.open(path)
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[-1])
        return bg
    return im.convert("RGB")


def _crop_native(im: Image.Image, cx1288, cy1288, r1288) -> np.ndarray:
    """1288-px coordinates -> native-resolution square window on white, resampled to 320px."""
    W, H = im.size
    s = max(W, H) / min(1288, max(W, H))          # 1288 -> native scale factor (1 for small images)
    cx, cy, r = int(cx1288 * s), int(cy1288 * s), int(r1288 * s)
    r = max(40, min(r, 600))
    cx = min(max(cx, 0), W - 1)
    cy = min(max(cy, 0), H - 1)
    canvas = Image.new("RGB", (2 * r, 2 * r), (255, 255, 255))
    src = im.crop((max(0, cx - r), max(0, cy - r), min(W, cx + r), min(H, cy + r)))
    canvas.paste(src, (max(0, cx - r) - (cx - r), max(0, cy - r) - (cy - r)))
    return np.asarray(canvas.convert("L").resize((320, 320), Image.LANCZOS),
                      dtype=np.float64)


def _mask(arr: np.ndarray) -> np.ndarray:
    lo, med = float(arr.min()), float(np.median(arr))
    th = min(128.0, lo + 0.45 * (med - lo)) if med > lo else 128.0
    m = arr < th
    m[:4, :] = m[-4:, :] = False
    m[:, :4] = m[:, -4:] = False
    return m


def _comps(mask: np.ndarray):
    lab, n = ndimage.label(mask)
    out = []
    for i in range(1, n + 1):
        ys, xs = np.nonzero(lab == i)
        out.append(np.stack([xs, ys], axis=1))
    out.sort(key=len, reverse=True)
    h, w = mask.shape
    keep = []
    for c in out:
        touches = (c[:, 0].min() <= 4 or c[:, 1].min() <= 4
                   or c[:, 0].max() >= w - 5 or c[:, 1].max() >= h - 5)
        if touches and len(c) > 0.05 * w * h:
            continue                          # wall / large block spilling into the window
        span = max(np.ptp(c[:, 0]), np.ptp(c[:, 1]))
        if touches and span > 0.9 * w:
            continue                          # long bar crossing the window (wall line)
        keep.append(c)
    return keep


def _ang(cx, cy, x, y) -> float:
    return math.degrees(math.atan2(x - cx, -(y - cy))) % 360


def _rose_method(comps) -> dict | None:
    """Ring fit + angular median of the cluster protruding beyond the ring. Validity = ring point count and
    residual + protrusion."""
    if not comps:
        return None
    allpix = np.concatenate(comps)
    cx0 = (allpix[:, 0].min() + allpix[:, 0].max()) / 2
    cy0 = (allpix[:, 1].min() + allpix[:, 1].max()) / 2
    rad = np.hypot(allpix[:, 0] - cx0, allpix[:, 1] - cy0)
    rmax = float(rad.max())
    if rmax < 20:
        return None
    hist, edges = np.histogram(rad, bins=24, range=(0, rmax))
    band = int(np.argmax(hist[4:17])) + 4
    bw = edges[1] - edges[0]
    ring = allpix[(rad >= edges[band]) & (rad < edges[band] + 1.6 * bw)]
    if len(ring) < 60:
        return None
    x, y = ring[:, 0].astype(float), ring[:, 1].astype(float)
    A = np.array([[np.sum(x * x), np.sum(x * y), x.sum()],
                  [np.sum(x * y), np.sum(y * y), y.sum()],
                  [x.sum(), y.sum(), len(x)]])
    z = x * x + y * y
    try:
        sol = np.linalg.solve(A, np.array([np.sum(x * z), np.sum(y * z), z.sum()]))
    except np.linalg.LinAlgError:
        return None
    cx, cy = sol[0] / 2, sol[1] / 2
    rfit = math.sqrt(max(sol[2] + cx * cx + cy * cy, 1e-9))
    resid = float(np.abs(np.hypot(x - cx, y - cy) - rfit).mean() / rfit)
    if resid > 0.08 or rfit < 18:
        return None
    # protruding cluster: dark pixels with r > 1.12*rfit, clustered by angle; keep the largest cluster
    rr = np.hypot(allpix[:, 0] - cx, allpix[:, 1] - cy)
    out = allpix[rr > 1.12 * rfit]
    if len(out) < 12:
        return None
    angs = np.array([_ang(cx, cy, px, py) for px, py in out])
    best_center, best_n = None, 0
    for a0 in range(0, 360, 10):
        d = np.minimum(np.abs(angs - a0), 360 - np.abs(angs - a0))
        n = int((d < 18).sum())
        if n > best_n:
            best_n, best_center = n, a0
    d = np.minimum(np.abs(angs - best_center), 360 - np.abs(angs - best_center))
    sel = angs[d < 18]
    # circular mean
    rad_sel = np.radians(sel)
    mang = math.degrees(math.atan2(np.sin(rad_sel).mean(), np.cos(rad_sel).mean())) % 360
    purity = best_n / len(out)
    validity = purity * min(1.0, len(ring) / 200) * (1 - resid * 6)
    return {"north_deg": mang, "validity": round(float(validity), 3),
            "conf": {"method": "rose", "resid": round(resid, 3),
                     "purity": round(purity, 2), "n_out": int(len(out))}}


def _star_method(comps) -> dict | None:
    """Four-point star: direction of the filled point (largest solid component) centroid relative to the
    whole symbol's centre. Validity = clear share of the filled point + eccentricity."""
    if len(comps) < 1:
        return None
    allpix = np.concatenate(comps)
    scx = (allpix[:, 0].min() + allpix[:, 0].max()) / 2
    scy = (allpix[:, 1].min() + allpix[:, 1].max()) / 2
    span = max(np.ptp(allpix[:, 0]), np.ptp(allpix[:, 1]))
    big = comps[0]
    if len(big) < 40:
        return None
    bcx, bcy = big.mean(axis=0)
    off = math.hypot(bcx - scx, bcy - scy)
    ecc = off / max(span / 2, 1e-9)
    if ecc < 0.18:                        # solid block sits at the symbol centre -> not a direction point
        return None
    dom = len(big) / len(allpix)
    validity = min(ecc, 0.9) * min(dom * 2, 1.0)
    return {"north_deg": _ang(scx, scy, bcx, bcy),
            "validity": round(float(validity), 3),
            "conf": {"method": "star", "ecc": round(ecc, 2), "dom": round(dom, 2)}}


def _needle_method(comps) -> dict | None:
    """Thin needle / dart: PCA main axis, narrow end = head. Validity = elongation x width difference of
    the two ends."""
    if not comps:
        return None
    comp = comps[0]
    n = len(comp)
    if n < 30:
        return None
    m = comp.mean(axis=0)
    d = comp - m
    cov = d.T @ d / n
    evals, evecs = np.linalg.eigh(cov)
    elong = float(evals[1] / max(evals[0], 1e-9))
    if elong < 2.2:
        return None
    u = evecs[:, 1]
    proj = d @ u
    perp = d @ evecs[:, 0]
    order = np.argsort(proj)
    k = max(6, n // 8)
    s_lo = float(np.abs(perp[order[:k]]).mean())
    s_hi = float(np.abs(perp[order[-k:]]).mean())
    margin = abs(s_lo - s_hi) / max(s_lo + s_hi, 1e-9)
    if margin < 0.18:
        return None
    tip = -u if s_lo < s_hi else u
    validity = min(elong / 8, 1.0) * min(margin * 2.5, 1.0)
    return {"north_deg": math.degrees(math.atan2(tip[0], -tip[1])) % 360,
            "validity": round(float(validity), 3),
            "conf": {"method": "needle", "elong": round(elong, 1),
                     "margin": round(margin, 2), "n": n}}


def extract_from_window(im: Image.Image, cx, cy, r) -> dict | None:
    arr = _crop_native(im, cx, cy, r)
    comps = _comps(_mask(arr))
    cands = [f(comps) for f in (_rose_method, _star_method, _needle_method)]
    cands = [c for c in cands if c]
    if not cands:
        return None
    return max(cands, key=lambda c: c["validity"])


def extract_north_multi(path, windows) -> dict | None:
    im = _load_rgb(path)
    best = None
    for cx, cy, r in windows:
        res = extract_from_window(im, cx, cy, r)
        if res and (best is None or res["validity"] > best["validity"]):
            best = res
    if best:
        best["clock"] = round((best["north_deg"] / 30.0) % 12, 2)
    return best


def letter_side_check(im, cx1288, cy1288, r1288, theta_deg):
    """Single-letter head check: find the only letter-like small component next to the symbol and see
    whether it sits at the theta end or the theta+180 end.
    -> "confirm" / "flip" / "unknown". Several letters (NESW rose) or none both return unknown.
    Basis: the winning rule from two batches of manual labelling -- the N letter always sits on the tip side."""
    import math as _m
    arr = _crop_native(im, cx1288, cy1288, r1288)
    mask = _mask(arr)
    comps = _comps(mask)
    if not comps:
        return "unknown"
    big = comps[0]
    bcx, bcy = big.mean(axis=0)
    ext = max(np.ptp(big[:, 0]), np.ptp(big[:, 1])) / 2 + 1e-6
    letters = []
    for c in comps[1:]:
        w = np.ptp(c[:, 0]) + 1
        h = np.ptp(c[:, 1]) + 1
        if not (5 <= max(w, h) <= 46):
            continue
        if len(c) < 10 or len(c) / (w * h) < 0.15:
            continue
        lcx, lcy = c.mean(axis=0)
        rho = _m.hypot(lcx - bcx, lcy - bcy)
        if not (0.4 * ext <= rho <= 3.2 * ext):
            continue
        phi = _ang(bcx, bcy, lcx, lcy)
        letters.append(phi)
    if len(letters) != 1:
        return "unknown"
    phi = letters[0]
    d_same = min(abs(phi - theta_deg) % 360, 360 - abs(phi - theta_deg) % 360)
    d_opp = min(abs(phi - (theta_deg + 180)) % 360, 360 - abs(phi - (theta_deg + 180)) % 360)
    if d_same <= 55 and d_opp > 55:
        return "confirm"
    if d_opp <= 55 and d_same > 55:
        return "flip"
    return "unknown"
