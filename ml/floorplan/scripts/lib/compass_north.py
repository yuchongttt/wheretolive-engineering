"""Geometry, gates and schema for the floorplan north-angle lane (pure functions, no network).

- Qwen `bbox_2d` is in 0-1000 normalised coordinates (confirmed 2026-09-02; the research pipeline had
  been treating it as 1288-px pixel coordinates).
- Three gates: VLM detection (present) + compass-ness classifier (presence_prob) + 4x90-degree
  equivariance gate (spread <= SPREAD_TAU). Four-fold, 233 real crops: spread<=10 -> 62.2% yield @ 100%.
- 8-way binning: within +-BIN_MARGIN of a 45-degree boundary, return the clockwise-adjacent pair "A|B"
  (inside the gate, exact 8-way match was 94%, and every miss was a small error near a boundary);
  theta itself is unaffected, and consumers must be able to parse "A|B".
- Property level: emit theta only if every pair of emitted readings is within PROPERTY_TOL.
Design: room-aspect pipeline spec (2026-09-02, not in this extract) section 3 S1; research notes in
research/compass-orientation/README.md.
"""
from __future__ import annotations

import math
import sqlite3
from pathlib import Path

from PIL import Image

MODEL_VERSION = "cnn-v1"
SPREAD_TAU = 10.0
BIN_MARGIN = 7.0
PROPERTY_TOL = 15.0
_BINS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
READER_WEIGHTS = Path(__file__).resolve().parents[2] / "data" / "models" / "compass_reader_v1.pt"
PRESENCE_WEIGHTS = Path(__file__).resolve().parents[2] / "data" / "models" / "compass_presence_v1.pt"


def flatten_on_white(img: Image.Image) -> Image.Image:
    """Alpha-composite onto white (same as scripts/floorplan-vlm/dispatcher.flatten_on_white in the
    private repo; that path contains a hyphen and cannot be imported, so it is copied here)."""
    has_alpha = img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info)
    if not has_alpha:
        return img if img.mode == "RGB" else img.convert("RGB")
    rgba = img.convert("RGBA")
    bg = Image.new("RGB", rgba.size, (255, 255, 255))
    bg.paste(rgba, mask=rgba.split()[-1])
    return bg


def bbox1000_to_window(bbox, w: int, h: int):
    x0, y0, x1, y1 = bbox
    cx, cy = (x0 + x1) / 2000 * w, (y0 + y1) / 2000 * h
    r = 0.75 * max((x1 - x0) / 1000 * w, (y1 - y0) / 1000 * h)
    return cx, cy, max(40.0, min(r, 600.0))


def crop_window(im: Image.Image, cx: float, cy: float, r: float, size: int = 320) -> Image.Image:
    """Square window at native resolution on a white canvas (out-of-bounds padded white), resampled
    to `size`, greyscale."""
    W, H = im.size
    cx, cy, r = int(cx), int(cy), int(r)
    cx = min(max(cx, 0), W - 1)
    cy = min(max(cy, 0), H - 1)
    canvas = Image.new("RGB", (2 * r, 2 * r), (255, 255, 255))
    src = im.crop((max(0, cx - r), max(0, cy - r), min(W, cx + r), min(H, cy + r)))
    canvas.paste(src, (max(0, cx - r) - (cx - r), max(0, cy - r) - (cy - r)))
    return canvas.convert("L").resize((size, size), Image.LANCZOS)


PRESENCE_MIN = 0.4   # held-out batch4: a 0.5 threshold would drop one real compass that reads correctly (prob 0.488); the other FNs are already blocked by the equivariance gate


def gate_decision(present, spread, presence_prob, tau: float = SPREAD_TAU, presence_min: float = PRESENCE_MIN):
    if present is None:
        return False, "detect_failed"
    if not present:
        return False, "absent"
    if presence_prob is not None and presence_prob < presence_min:
        return False, "presence"
    if spread is None or (isinstance(spread, float) and math.isnan(spread)):
        return False, "unread"
    if spread > tau:
        return False, "spread"
    return True, "pass"


def bin8_with_margin(deg: float, margin: float = BIN_MARGIN):
    deg = deg % 360
    off = (deg - 22.5) % 45          # distance to the nearest 45-degree boundary (22.5 + 45k)
    if min(off, 45 - off) <= margin:
        k = round((deg - 22.5) / 45) % 8     # lower bin of the nearest boundary; within +-margin, (deg-22.5)/45 is <0.16 from an integer, so round() is unambiguous
        return f"{_BINS[k]}|{_BINS[(k + 1) % 8]}"
    return _BINS[int(((deg + 22.5) % 360) // 45)]


def combine_property(readings, tol: float = PROPERTY_TOL):
    vals = [float(v) for v in readings]
    if not vals:
        return None, False
    for i in range(len(vals)):
        for j in range(i + 1, len(vals)):
            d = abs(vals[i] - vals[j]) % 360
            if min(d, 360 - d) > tol:
                return None, False
    r = [math.radians(v) for v in vals]
    mean = math.degrees(math.atan2(sum(map(math.sin, r)) / len(r), sum(map(math.cos, r)) / len(r))) % 360
    return (0.0 if mean >= 360.0 - 1e-9 else mean), True


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute("""CREATE TABLE IF NOT EXISTS floorplan_north (
        property_id TEXT NOT NULL, idx INTEGER NOT NULL, url TEXT, model_version TEXT NOT NULL,
        ok INTEGER NOT NULL DEFAULT 0, error TEXT,
        compass_present INTEGER, bbox_json TEXT,
        north_deg REAL, spread_deg REAL, preds_json TEXT, presence_prob REAL,
        gate TEXT, emitted INTEGER NOT NULL DEFAULT 0, north_8way TEXT,
        image_source TEXT, inference_ms INTEGER, checked_at TEXT NOT NULL,
        PRIMARY KEY (property_id, idx, model_version))""")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_floorplan_north_pid ON floorplan_north(property_id)")
    conn.execute("""CREATE TABLE IF NOT EXISTS property_north (
        property_id TEXT PRIMARY KEY, north_deg REAL, north_8way TEXT,
        n_floorplans INTEGER NOT NULL, n_emitted INTEGER NOT NULL, agree INTEGER NOT NULL,
        model_version TEXT NOT NULL, computed_at TEXT NOT NULL)""")
    conn.commit()
