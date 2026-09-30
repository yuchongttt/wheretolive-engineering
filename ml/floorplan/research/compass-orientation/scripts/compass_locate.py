#!/usr/bin/env python3
"""Compass candidate-window locator v2 (classical CV).

Criterion: a compass = a medium-sized ink cluster (symbol + letters; remove long structures first, then
merge near neighbours) with an almost blank ring around it (unlike text paragraphs or plan detail).
locate_candidates(im_1288) -> [(score, cx, cy, r), ...] sorted by score, in 1288-px coordinates.
"""
from __future__ import annotations

import numpy as np
from PIL import Image
from scipy import ndimage


def to_1288(im: Image.Image) -> Image.Image:
    w, h = im.size
    s = 1288 / max(w, h)
    return im.resize((int(w * s), int(h * s)), Image.LANCZOS) if s < 1 else im


def locate_candidates(im: Image.Image, max_cands: int = 4):
    g = np.asarray(im.convert("L"), dtype=np.float64)
    H, W = g.shape
    ink = g < 160

    # 1) drop long structures (walls / frames / long banners): remove any connected component whose bbox long side > 200px
    lab, n = ndimage.label(ink)
    if n == 0:
        return []
    sl = ndimage.find_objects(lab)
    small = np.zeros_like(ink)
    for i, s in enumerate(sl, 1):
        hh, ww = s[0].stop - s[0].start, s[1].stop - s[1].start
        if max(hh, ww) <= 200:
            small[s][lab[s] == i] = True

    # 2) merge near neighbours (join the symbol with its N/S/E/W letters into one candidate cluster)
    merged = ndimage.binary_dilation(small, iterations=9)
    clab, cn = ndimage.label(merged)
    csl = ndimage.find_objects(clab)

    cands = []
    for i, s in enumerate(csl, 1):
        y0, y1 = s[0].start, s[0].stop
        x0, x1 = s[1].start, s[1].stop
        bh, bw = y1 - y0, x1 - x0
        size = max(bh, bw)
        if not (22 <= size <= 175):
            continue
        if size / max(min(bh, bw), 1) > 2.6:
            continue
        sub = small[s] & (clab[s] == i)
        n_ink = int(sub.sum())
        if not (60 <= n_ink <= 7000):
            continue
        dens = n_ink / (bh * bw)
        if not (0.03 <= dens <= 0.65):
            continue
        # 3) isolation: ink (walls included) in a ring 0.65x the bbox size around it must be sparse
        pad_y, pad_x = int(bh * 0.65), int(bw * 0.65)
        Y0, Y1 = max(0, y0 - pad_y), min(H, y1 + pad_y)
        X0, X1 = max(0, x0 - pad_x), min(W, x1 + pad_x)
        outer = ink[Y0:Y1, X0:X1].sum() - ink[y0:y1, x0:x1].sum()
        outer_area = (Y1 - Y0) * (X1 - X0) - bh * bw
        ann = outer / max(outer_area, 1)
        if ann > 0.06:
            continue
        score = (1 - ann * 10) * np.sqrt(n_ink)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        cands.append((float(score), cx, cy, max(70, size * 0.85)))
    cands.sort(reverse=True)
    return cands[:max_cands]
