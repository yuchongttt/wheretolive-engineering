#!/usr/bin/env python3
"""Floorplan VLM dispatcher v6.2 — production.

V6.2 = v6 schema (room_labels OCR + bathroom 3-way split + bedroom audit) +
       max_tokens=2200 (mansion outputs no longer truncated) +
       slim room_counts (flat int per type, not {count, total_sqft}) +
       room_labels.floor field dropped (not used downstream).

Apples-to-apples on 600 v4 sample vs prior versions:
  bedrooms vs listing 91.9% (v6: 92.1%, v5: 90.8%)
  bathrooms either-match 89.0% (v6: 90.8%, v5: ~82.4% strict-only)
  multi-fp bedrooms 94.4% (matches v6 high, +8.3pp vs Sonnet)
  parse fail rate 0.5% (v6: 2.8%, v5: 0%)
  latency 36s/property (v6: 62s, v5: 33s — v6.2 trades 3s for accuracy)

Env vars:
  WTL_VLM_URL          base URL of vllm serve (default http://vlm-box:8200)
  WTL_VLM_BATCH        properties per tick (default 4)
  WTL_VLM_CONCURRENCY  parallel HTTP requests (default 4)
  WTL_VLM_MODEL        served-model-name (default 'qwen3.6-27b')
  WTL_VLM_MODEL_VER    db tag (default 'qwen3.6-27b-autoround-v6.2')
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import sqlite3
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "data" / "evaluations.db"
# rm_sold_floorplans split out 2026-05-27; ATTACH onto each DB conn below.
SOLD_DB = ROOT / "data" / "sold.db"

VLM_URL = os.environ.get("WTL_VLM_URL", "http://vlm-box:8200").rstrip("/")
BATCH = int(os.environ.get("WTL_VLM_BATCH", "4"))
CONCURRENCY = int(os.environ.get("WTL_VLM_CONCURRENCY", "4"))
MODEL = os.environ.get("WTL_VLM_MODEL", "qwen3.6-27b")
# v6.6b (2026-06-05): room_labels enum += balcony|terrace — outdoor labels were
# all landing in type='other' (3.8k of them), making balcony stats/conditions
# rely on label-text matching. Suffix keeps "LIKE '%v6.6%'" consumers matching,
# and claim queries use LIKE 'qwen3.6-27b-autoround-v6.6%' so the v6.6 backlog
# is NOT re-claimed (no mass rerun).
MODEL_VERSION = os.environ.get("WTL_VLM_MODEL_VER", "qwen3.6-27b-autoround-v6.6b")
REQUEST_TIMEOUT = int(os.environ.get("WTL_VLM_TIMEOUT", "300"))
MAX_TOKENS = int(os.environ.get("WTL_VLM_MAX_TOKENS", "2200"))
IMG_SHORT_EDGE = int(os.environ.get("WTL_VLM_IMG_EDGE", "1280"))
MAX_IMGS = int(os.environ.get("WTL_VLM_MAX_IMGS", "4"))

# Local-first floorplan images (A-4): when enabled, read mirrored originals from
# the VLM box's local image server over the LAN instead of fetching the image
# URL again (lower latency, no dependency on the URL staying valid). Off by default →
# identical behaviour. Any local miss/error falls back to the source URL.
# (The mirror and its image server are not included in this repo.)
FP_LOCAL_IMAGES = os.environ.get("WTL_FP_LOCAL_IMAGES", "0") == "1"
FP_LOCAL_URL = os.environ.get("WTL_FP_LOCAL_URL", "http://vlm-box:8203").rstrip("/")

AGGREGATED_IDX = -1

PROMPT_INTRO_SINGLE = "You are a UK property floorplan analyser. Parse the floorplan and output ONLY a single JSON object. Do NOT output markdown bullets, asterisks, prose, or thinking. Your response MUST start with `{` immediately."
PROMPT_INTRO_MULTI = (
    "You are a UK property floorplan analyser. You are given {n} floorplan images of the SAME property. "
    "WARNING: Property listings often include MULTIPLE versions of the same floor — labeled + unlabeled, "
    "large + thumbnail. THESE ARE DUPLICATES — count each unique floor ONLY ONCE. "
    "First identify which images are unique floors and which are duplicates. Typical UK property has 1-3 distinct floors. "
    "Then AGGREGATE rooms across the unique floors and output ONE single JSON describing the WHOLE property. "
    "If an image is a site/block plan (shows footprint within a plot, no rooms) — ignore room counts in it, only use it for outdoor features. "
    # NOTE: this string is passed through str.format(n=...), so the literal brace
    # below MUST be doubled ({{) or .format() raises "expected '}' before end of
    # string" and every multi-image property crashes before the VLM call.
    "Output ONLY a single JSON object. Do NOT output markdown bullets, asterisks, prose, or thinking. Your response MUST start with `{{` immediately."
)

PROMPT_SCHEMA = """
THINK CAREFULLY and enumerate small rooms — common MISSES:
- Ground-floor WC / cloakroom (small, often near entrance/hallway)
- Upstairs en-suite shower room (small, attached to a bedroom)
- Box room / nursery (small bedroom 5-7 sqm)
- Utility / boot room (small, often by kitchen)

ABSOLUTELY CRITICAL — DO NOT STRIP DIMENSIONS FROM LABELS:
Each label MUST contain the room's VERBATIM OCR text INCLUDING dimensions, qualifiers, and printed area. Do NOT shorten "Bedroom 1 13'6 x 8'6 (4.14m x 2.57m)" to just "Bedroom 1". Do NOT drop "max" or "into bay" qualifiers — they tell us the room is irregular. Examples:
  CORRECT: "Bedroom 1 13'6 x 8'6 (4.14m x 2.57m)"
  CORRECT: "Lounge 21'2 into bay x 18'6 max"
  CORRECT: "Bedroom 15.93 m² (4.06 x 3.93)"
  WRONG:   "Bedroom 1" (stripped dimensions)

PROCESS (you MUST follow this 3-step order before outputting JSON):

STEP 1: ENUMERATE — silently list EVERY visible text label on EVERY floorplan image, even tiny ones. INCLUDE the dimension text printed BELOW or BESIDE each room name. En-suite labels are often abbreviated as "EN" or "EN/S".

STEP 2: CLASSIFY — for each label decide type (bedroom/bathroom/shower_room/wc/kitchen/living/dining/reception/study/utility/storage/hallway/other), floor (ground/first/second/third/lower-ground/loft/basement/other), fixtures.

STEP 3: AGGREGATE — room_counts.X = labels with type=X.

OUTDOOR SPACES: label "Balcony" → type=balcony, "Terrace"/"Roof Terrace"/"Roof Garden" → type=terrace. Do NOT type them "other". They are NOT in room_counts (indoor rooms only) and NEVER count toward total_sqft.

Schema:
{
  "room_labels": [
    {"label": "<VERBATIM OCR text including dimensions, qualifiers, and any m² printed area>",
     "type":  "bedroom|bathroom|shower_room|wc|kitchen|living|dining|reception|study|utility|storage|hallway|balcony|terrace|other",
     "floor": "ground|first|second|third|lower-ground|loft|basement|other",
     "fixtures": [<list: "bed", "bath", "shower", "toilet", "basin", "sink", "stove", "fireplace", or empty>]}
  ],
  "total_sqft": <int or null>,
  "floors": <int>,
  "layout_class": <"studio"|"one-bed-flat"|"two-bed-flat"|"three-bed-flat"|"four-plus-bed-flat"|"maisonette"|"house-mid-terraced"|"house-end-terraced"|"house-semi"|"house-detached"|"unknown">,
  "room_counts": {
    "bedroom":<int>,"bathroom":<int>,"shower_room":<int>,"wc":<int>,"kitchen":<int>,"living":<int>,"dining":<int>,"reception":<int>,"study":<int>,"utility":<int>,"storage":<int>
  },
  "bathroom_layout": {
    "ensuite_bath":<int>,"ensuite_shower":<int>,"shared_bath":<int>,"shared_shower":<int>,"wc_only":<int>
  },
  "features": {
    "garden":<bool>,"garage":<bool>,"balcony":<bool>,"terrace":<bool>,"conservatory":<bool>,"fireplace":<bool>
  },
  "confidence": <"high"|"medium"|"low">
}

KEY BATHROOM TYPES:
  - bathroom = has BATH (with or without shower over it)
  - shower_room = shower only, NO bath
  - wc = toilet only (cloakroom, powder room, "W.C.", "Cloak")

KEY BEDROOM RULES:
  - type=bedroom requires label containing "Bed" / "Master" / "Br" OR fixtures including "bed"
  - "Bedroom 4 / Study" → type=bedroom (keep "/ Study" in label)
  - Living Room / Reception / Lounge ≠ bedroom even if large
  - STUDIO: open-plan single room with bed+kitchen+living combined → bedrooms=0, layout_class="studio"

KEY FLOOR RULES:
  - Read image headers ("Ground Floor", "First Floor", "Lower Ground Floor")
  - Single-floor flat: ALL rooms → same floor (usually "ground")
  - Multi-image property: each image is typically one floor — assign accordingly

total_sqft = SINGLE "Total Floor Area" / "Gross Internal Floor Area" / "Approx. Total Area" label on the plan. NEVER sum per-floor (double-counts). m² × 10.764 if m². null if no total label.

Examples (indented JSON, copy this exact style):

Example 1 — 1-bed flat (labels preserve dimensions verbatim):
{"room_labels":[{"label":"Living Room 17'3 x 10'6 (5.2m x 3.2m)","type":"living","floor":"ground","fixtures":[]},{"label":"Kitchen 11'5\" x 6'4\" (3.47m x 1.92m)","type":"kitchen","floor":"ground","fixtures":["stove","sink"]},{"label":"Bedroom 13'6 x 8'6 (4.14m x 2.57m)","type":"bedroom","floor":"ground","fixtures":["bed"]},{"label":"Bathroom 7'10 x 5'8 (2.40m x 1.72m)","type":"bathroom","floor":"ground","fixtures":["bath","toilet","basin"]}],"total_sqft":425,"floors":1,"layout_class":"one-bed-flat","room_counts":{"bedroom":1,"bathroom":1,"shower_room":0,"wc":0,"kitchen":1,"living":1,"dining":0,"reception":0,"study":0,"utility":0,"storage":0},"bathroom_layout":{"ensuite_bath":0,"ensuite_shower":0,"shared_bath":1,"shared_shower":0,"wc_only":0},"features":{"garden":false,"garage":false,"balcony":true,"terrace":false,"conservatory":false,"fireplace":false},"confidence":"high"}

Example 2 — 3-bed with cloakroom + irregular lounge (note "into bay" and "max" preserved):
{"room_labels":[{"label":"Lounge 18'6 max x 12'4 into bay (5.64m x 3.76m)","type":"living","floor":"ground","fixtures":["fireplace"]},{"label":"Kitchen/Diner 14'2 x 9'8 (4.32m x 2.95m)","type":"kitchen","floor":"ground","fixtures":["stove","sink"]},{"label":"Cloakroom","type":"wc","floor":"ground","fixtures":["toilet","basin"]},{"label":"Bedroom 1 13'6 x 11'2 (4.14m x 3.40m)","type":"bedroom","floor":"first","fixtures":["bed"]},{"label":"Bedroom 2 11'8 x 9'4 (3.56m x 2.85m)","type":"bedroom","floor":"first","fixtures":["bed"]},{"label":"Bedroom 3 9'0 x 6'6 (2.74m x 1.98m)","type":"bedroom","floor":"first","fixtures":["bed"]},{"label":"Bathroom 7'10 x 5'8 (2.40m x 1.72m)","type":"bathroom","floor":"first","fixtures":["bath","toilet","basin"]}],"total_sqft":920,"floors":2,"layout_class":"house-mid-terraced","room_counts":{"bedroom":3,"bathroom":1,"shower_room":0,"wc":1,"kitchen":1,"living":1,"dining":0,"reception":0,"study":0,"utility":0,"storage":0},"bathroom_layout":{"ensuite_bath":0,"ensuite_shower":0,"shared_bath":1,"shared_shower":0,"wc_only":1},"features":{"garden":true,"garage":false,"balcony":false,"terrace":false,"conservatory":false,"fireplace":true},"confidence":"high"}

Example 3 — Strict bedroom rule (Living Room is LARGE but NOT a bedroom):
{"room_labels":[{"label":"Bedroom","type":"bedroom","floor":"ground","fixtures":["bed"]},{"label":"Living Room","type":"living","floor":"ground","fixtures":[]},{"label":"Kitchen","type":"kitchen","floor":"ground","fixtures":["stove","sink"]},{"label":"Bathroom","type":"bathroom","floor":"ground","fixtures":["bath","toilet","basin"]}],"total_sqft":560,"floors":1,"layout_class":"one-bed-flat","room_counts":{"bedroom":1,"bathroom":1,"shower_room":0,"wc":0,"kitchen":1,"living":1,"dining":0,"reception":0,"study":0,"utility":0,"storage":0},"bathroom_layout":{"ensuite_bath":0,"ensuite_shower":0,"shared_bath":1,"shared_shower":0,"wc_only":0},"features":{"garden":false,"garage":false,"balcony":false,"terrace":false,"conservatory":false,"fireplace":false},"confidence":"high"}

Example 4 — Studio (open-plan, no enclosed bedroom):
{"room_labels":[{"label":"","type":"living","floor":"ground","fixtures":["bed","sink","stove"]},{"label":"Bathroom","type":"bathroom","floor":"ground","fixtures":["bath","toilet","basin"]}],"total_sqft":350,"floors":1,"layout_class":"studio","room_counts":{"bedroom":0,"bathroom":1,"shower_room":0,"wc":0,"kitchen":1,"living":1,"dining":0,"reception":0,"study":0,"utility":0,"storage":0},"bathroom_layout":{"ensuite_bath":0,"ensuite_shower":0,"shared_bath":1,"shared_shower":0,"wc_only":0},"features":{"garden":false,"garage":false,"balcony":false,"terrace":false,"conservatory":false,"fireplace":false},"confidence":"high"}

Example 5 — 4-bed detached with ensuite + family bathroom + cloakroom (multi-floor):
{"room_labels":[{"label":"Living Room","type":"living","floor":"ground","fixtures":["fireplace"]},{"label":"Kitchen/Diner","type":"kitchen","floor":"ground","fixtures":["stove","sink"]},{"label":"Study","type":"study","floor":"ground","fixtures":[]},{"label":"Cloakroom","type":"wc","floor":"ground","fixtures":["toilet","basin"]},{"label":"Master Bedroom","type":"bedroom","floor":"first","fixtures":["bed"]},{"label":"En-suite","type":"shower_room","floor":"first","fixtures":["shower","toilet","basin"]},{"label":"Bedroom 2","type":"bedroom","floor":"first","fixtures":["bed"]},{"label":"Bedroom 3","type":"bedroom","floor":"first","fixtures":["bed"]},{"label":"Bedroom 4","type":"bedroom","floor":"first","fixtures":["bed"]},{"label":"Bathroom","type":"bathroom","floor":"first","fixtures":["bath","toilet","basin"]}],"total_sqft":1500,"floors":2,"layout_class":"house-detached","room_counts":{"bedroom":4,"bathroom":1,"shower_room":1,"wc":1,"kitchen":1,"living":1,"dining":0,"reception":0,"study":1,"utility":0,"storage":0},"bathroom_layout":{"ensuite_bath":0,"ensuite_shower":1,"shared_bath":1,"shared_shower":0,"wc_only":1},"features":{"garden":true,"garage":false,"balcony":false,"terrace":false,"conservatory":false,"fireplace":true},"confidence":"high"}

Output ONE single JSON object — start with `{` immediately, no markdown fence, no prose, no bullets.
"""

# Negative filter: a room_labels entry tagged type=bedroom but whose label
# contains a non-bedroom keyword gets dropped from the bedroom count.
_NON_BEDROOM_RE = re.compile(
    r"\b(living|lounge|sitting|reception|family|snug|playroom|study|office|den|"
    r"kitchen|dining|breakfast|hall|hallway|landing|entrance|porch|"
    r"bath|shower|wc|cloak|toilet|powder|utility|laundry|"
    r"storage|store|cupboard|larder|pantry|conservatory|garage|garden)\b",
    re.IGNORECASE,
)


def build_prompt(n_images: int) -> str:
    intro = PROMPT_INTRO_MULTI.format(n=n_images) if n_images > 1 else PROMPT_INTRO_SINGLE
    suffix = (
        "\n\nNow analyse the above {n} floorplan images and output ONE aggregated JSON for the whole property."
        if n_images > 1 else
        "\n\nNow parse the floorplan and output the JSON for THIS image."
    ).format(n=n_images)
    return intro + PROMPT_SCHEMA + suffix


def flatten_on_white(img):
    """Return an RGB copy, compositing any alpha channel onto WHITE.

    Dropping alpha with a bare .convert("RGB") fills transparent pixels with
    BLACK. Listing floorplans are routinely dark line art on a transparent
    background (real sample: mode 'LA', alpha 0..255), so the naive convert
    turns the entire plan into a black rectangle — mean luma 3.5 vs 231.6 when
    composited on white. The VLM then sees a black canvas and answers
    layout_class=unknown / confidence=low / zero rooms, which is recorded as
    ok=1 and therefore NEVER retried: a silent, permanent write-off.

    Measured at discovery (v6.6 series): PNG-sourced listings had a 24.4%
    empty-parse rate (2,465 of 10,085) against 0.1% for non-PNG (30 of 49,096).

    Modes with alpha are RGBA / LA / PA, plus palette images carrying a
    'transparency' key — that last one is the easy miss, since its mode is
    just 'P'. Everything else converts straight through unchanged.
    """
    from PIL import Image

    has_alpha = img.mode in ("RGBA", "LA", "PA") or (
        img.mode == "P" and "transparency" in img.info)
    if not has_alpha:
        return img if img.mode == "RGB" else img.convert("RGB")
    rgba = img.convert("RGBA")
    bg = Image.new("RGB", rgba.size, (255, 255, 255))
    bg.paste(rgba, mask=rgba.split()[-1])
    return bg


def fetch_and_resize(url: str, short_edge: int = None) -> str:
    """Download image and resize. Raises on non-image content (HTML redirect).

    Retries once with 3s delay on transient errors (5xx, timeout, connection
    refused). 2026-05-27 v6.4 probe showed 2/4 'all URLs dead' failures were
    actually transient — URLs returned 200 OK minutes later.
    """
    target = short_edge if short_edge is not None else IMG_SHORT_EDGE
    last_err = None
    for attempt in (1, 2):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                raw = r.read()
                content_type = (r.headers.get("content-type", "") or "").lower()
            break
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as e:
            last_err = e
            # Don't retry permanent 4xx (404 = dead listing, 410 = sold/removed).
            if isinstance(e, urllib.error.HTTPError) and 400 <= e.code < 500 and e.code != 408:
                raise
            if attempt == 1:
                time.sleep(3)
            else:
                raise
    if not content_type.startswith("image/"):
        raise ValueError(f"non-image content-type: {content_type!r} for {url[:80]}")
    from PIL import Image
    # Flatten FIRST: resizing an alpha image and only dropping the channel at
    # JPEG-save time still ends up black, so the composite has to happen before
    # anything else touches the pixels.
    img = flatten_on_white(Image.open(io.BytesIO(raw)))
    w, h = img.size
    short = min(w, h)
    if short > target:
        scale = target / short
        new_size = (int(w * scale), int(h * scale))
        img = img.resize(new_size, Image.LANCZOS)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=88)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


def resolve_and_resize(image_url: str, pid: str | None, idx: int | None, edge: int) -> str:
    """Local-first image load: when FP_LOCAL_IMAGES is on and we know the
    (pid, idx), try the VLM box's local mirror server (LAN, no second fetch).
    Any miss/error transparently falls back to the source URL — so the worst
    case is identical to the old behaviour.
    """
    if FP_LOCAL_IMAGES and pid is not None and idx is not None:
        try:
            return fetch_and_resize(f"{FP_LOCAL_URL}/fp/{pid}/{idx}?edge={edge}", short_edge=edge)
        except Exception:
            pass  # not mirrored / server down → source-URL fallback below
    return fetch_and_resize(image_url, short_edge=edge)


def claim_property_batch(conn: sqlite3.Connection, n_properties: int) -> list[tuple[str, list[tuple[int, str]]]]:
    """Claim N properties not yet processed at THIS model_version AND not already
    processed successfully at ANY earlier model_version (v3/v4/v5/v6).

    Rationale: v6.2 has richer schema (shower_room split, bathroom_layout 5-way,
    room_labels OCR) but the core fields (bedrooms / bathrooms / total_sqft /
    layout_class) are all populated by v3+. For first-pass backfill we prefer
    coverage over schema uniformity — skip already-processed properties to ~halve
    the remaining work. A future pass can upgrade old v3 rows to v6.2 schema.
    """
    rows = conn.execute(
        """
        SELECT f.rm_uuid, f.idx, f.url
        FROM sold.rm_sold_floorplans f
        WHERE f.url IS NOT NULL
          AND f.rm_uuid IN (
            SELECT f2.rm_uuid FROM sold.rm_sold_floorplans f2
            LEFT JOIN sold.rm_sold_properties p ON p.rm_uuid = f2.rm_uuid
            WHERE f2.url IS NOT NULL
              -- Skip properties already successfully processed at ANY model_version.
              AND NOT EXISTS (
                SELECT 1 FROM floorplan_vlm_results r2
                WHERE r2.rm_uuid = f2.rm_uuid AND r2.ok = 1
              )
              -- Skip properties already ATTEMPTED at THIS model_version (incl. failed
              -- rows). Each property gets one shot per version; an un-processable
              -- property is recorded once then drops out of the queue instead of being
              -- re-claimed forever and starving the rest of the backlog behind it.
              -- To retry a failure, delete its ok=0 row at this model_version.
              AND NOT EXISTS (
                SELECT 1 FROM floorplan_vlm_results r3
                WHERE r3.rm_uuid = f2.rm_uuid AND r3.model_version = ?
              )
            GROUP BY f2.rm_uuid
            -- Newest listings first: more user-facing value, fresher comparables.
            -- NULL list_scraped_at sorts last (legacy rows without metadata).
            ORDER BY p.list_scraped_at DESC NULLS LAST
            LIMIT ?
          )
        ORDER BY f.rm_uuid, f.idx
        """,
        (MODEL_VERSION, n_properties),
    ).fetchall()
    grouped: dict[str, list[tuple[int, str]]] = {}
    for uid, idx, url in rows:
        grouped.setdefault(uid, []).append((idx, url))
    return list(grouped.items())


# ---- v6.6 area extraction (3-tier regex over verbatim label text) ----
# Tier 1 (exact): label contains "XX m²" / "XX sqm" — UK agent's measured area
# Tier 2 (rectangular): plain dimensions "WxH" without qualifiers
# Tier 3 (irregular): dimensions with "max" / "into bay" — apply 0.82 L-shape factor
# Calibrated on 50-property probe 2026-05-27: 73.7% capture, median ratio 0.74
# (room sum / total_sqft), consistent with UK floorplan convention where walls
# and hallways absorb ~25% of total floor area.

_AREA_TIER1_M2_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:m\s*²|m2|sq\s*m|sqm)", re.IGNORECASE)
_AREA_METRIC_NUM_RE = re.compile(r"(?<![\d.])(\d+\.\d{1,2})(?![\d.])")
_AREA_IMPERIAL_NUM_RE = re.compile(r"(\d+)\s*['′]\s*(\d*)\s*[\"″]?")
_AREA_IRREGULAR_RE = re.compile(r"\b(max|maximum|into\s*bay|into\s*recess|excluding|irregular)\b", re.IGNORECASE)


def _ft_in_to_ft(ft_str: str, in_str: str) -> float:
    return float(ft_str) + (float(in_str) if in_str else 0) / 12.0


def extract_room_area(label: str) -> dict | None:
    """Parse a single room_label text → {area_sqft, tier, shape, method, long_edge_m}.
    long_edge_m = the room's longest printed side in metres — a LINEAR measure that
    stays valid even when the bbox AREA is an over-estimate (L/angled rooms) and
    even when area_sqft is later NULLed. Used by the room max-edge search filter."""
    if not label: return None
    text = label.strip()
    m = _AREA_TIER1_M2_RE.search(text)
    if m:
        sqm = float(m.group(1))
        if 1 <= sqm <= 200:
            return {"area_sqft": int(round(sqm * 10.764)), "tier": 1, "shape": "exact",
                    "method": "printed_m2", "long_edge_m": None}  # printed m² has no side dims
    is_irr = bool(_AREA_IRREGULAR_RE.search(text))
    factor = 0.82 if is_irr else 1.0
    metric_nums = [float(m.group(1)) for m in _AREA_METRIC_NUM_RE.finditer(text)]
    for i in range(len(metric_nums) - 1):
        w, h = metric_nums[i], metric_nums[i+1]
        if 0.8 <= w <= 30 and 0.8 <= h <= 30:
            return {"area_sqft": int(round(w * h * 10.764 * factor)),
                    "tier": 3 if is_irr else 2,
                    "shape": "irregular" if is_irr else "rectangular",
                    "method": "metric_dims", "long_edge_m": round(max(w, h), 2)}
    imp = _AREA_IMPERIAL_NUM_RE.findall(text)
    if len(imp) >= 2:
        feet = [_ft_in_to_ft(ft, inch) for ft, inch in imp]
        for i in range(len(feet) - 1):
            w, h = feet[i], feet[i+1]
            if 2 <= w <= 80 and 2 <= h <= 80:
                return {"area_sqft": int(round(w * h * factor)),
                        "tier": 3 if is_irr else 2,
                        "shape": "irregular" if is_irr else "rectangular",
                        "method": "imperial_dims", "long_edge_m": round(max(w, h) * 0.3048, 2)}
    return None


def _img_edge_for_n(n: int) -> int:
    """Dynamic image short-edge — v6.4 bumped to catch small "EN" / "W.C." labels.
    Single 1600 (was 1280), double 1280 (was 960), triple 960, quad 768."""
    if n <= 1: return IMG_SHORT_EDGE  # default 1600 in plist
    if n == 2: return 1280
    if n == 3: return 960
    return 768


def call_vlm_multi(urls: list[str], refs: list[tuple[str, int]] | None = None) -> dict:
    images_to_send = urls[:MAX_IMGS]
    # refs[i] = (pid, idx) parallel to urls[i], enabling local-first reads.
    refs = (refs or [None] * len(urls))[:MAX_IMGS]
    edge = _img_edge_for_n(len(images_to_send))
    img_payloads = []
    skipped = []
    for u, ref in zip(images_to_send, refs):
        pid, idx = ref if ref else (None, None)
        try:
            img_payloads.append(resolve_and_resize(u, pid, idx, edge))
        except Exception as e:
            skipped.append((u, str(e)[:80]))
    if not img_payloads:
        raise RuntimeError(f"all {len(images_to_send)} URLs dead: {skipped[:3]}")
    if len(img_payloads) != len(images_to_send):
        edge = _img_edge_for_n(len(img_payloads))
    content = [{"type": "image_url", "image_url": {"url": p}} for p in img_payloads]
    content.append({"type": "text", "text": build_prompt(len(img_payloads))})
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(
        f"{VLM_URL}/v1/chat/completions",
        data=body, headers={"content-type": "application/json"}, method="POST",
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as r:
        resp = json.loads(r.read().decode())
    inference_ms = int((time.time() - t0) * 1000)
    raw = resp["choices"][0]["message"]["content"]
    usage = resp.get("usage", {})
    parsed = parse_json_from_text(raw)

    # Site-plan / multi-unit brochure detector — mark confidence=low when the
    # floorplan is actually a dev brochure showing N identical unit layouts.
    # Heuristic: kitchen >= 3 OR living >= 4 (normal property has 1-2 of each).
    # 2026-05-27 calibrated on 102 v6.4 probe properties: 2/102 = 2% false-fire
    # rate, both genuine multi-unit cases.
    if parsed:
        rc = parsed.get("room_counts") or {}
        def _ct(k):
            v = rc.get(k); return v.get("count") if isinstance(v, dict) else (v or 0)
        if _ct("kitchen") >= 3 or _ct("living") >= 4:
            parsed["confidence"] = "low"
            parsed["_site_plan_detected"] = True

    # v6.6 area extraction post-process — parse dimensions from each room_label
    # text using 3-tier regex (printed m² > rectangular dims > irregular w/ "max"
    # qualifier × 0.82 factor). Stores per-room area in _room_areas array.
    # See `extract_room_area()` for tier logic.
    if parsed:
        areas = []
        for l in parsed.get("room_labels") or []:
            if not isinstance(l, dict): continue
            ai = extract_room_area(l.get("label") or "")
            entry = {"label": l.get("label",""), "type": l.get("type"),
                     "floor": l.get("floor")}
            entry.update(ai or {"area_sqft": None, "tier": None, "shape": None, "method": None, "long_edge_m": None})
            areas.append(entry)
        if areas:
            parsed["_room_areas"] = areas

    return {
        "raw": raw,
        "parsed": parsed,
        "n_images_used": len(img_payloads),
        "n_images_skipped": len(skipped),
        "inference_ms": inference_ms,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
    }


def parse_json_from_text(text: str) -> dict | None:
    txt = text.strip()
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", txt, re.DOTALL)
    if m:
        try: return json.loads(m.group(1))
        except json.JSONDecodeError: pass
    try: return json.loads(txt)
    except json.JSONDecodeError: pass
    first, last = txt.find("{"), txt.rfind("}")
    if first >= 0 and last > first:
        try: return json.loads(txt[first:last + 1])
        except json.JSONDecodeError: pass
    return None


def _as_int(v):
    if v is None: return None
    if isinstance(v, bool): return int(v)
    try: return int(v)
    except (ValueError, TypeError): return None


def _as_bool_int(v):
    if v is None: return None
    if isinstance(v, bool): return 1 if v else 0
    if isinstance(v, (int, float)): return 1 if v else 0
    if isinstance(v, str): return 1 if v.lower() in ("true", "yes", "1") else 0
    return None


def _room_count(rc: dict, t: str) -> int:
    """v6.2 stores int directly; older v5/v6 used {count, total_sqft}."""
    v = rc.get(t)
    if isinstance(v, dict):
        v = v.get("count")
    try:
        return int(v) if v is not None else 0
    except (TypeError, ValueError):
        return 0


def _audit_bedrooms_via_labels(p: dict) -> int | None:
    """Negative filter: count type=bedroom entries UNLESS label clearly says otherwise."""
    labels = p.get("room_labels")
    if not isinstance(labels, list) or not labels:
        return None
    audited = 0
    for r in labels:
        if not isinstance(r, dict): continue
        if r.get("type") != "bedroom": continue
        label = (r.get("label") or "").strip()
        if _NON_BEDROOM_RE.search(label):
            continue
        audited += 1
    return audited


def _derive_bedrooms(p: dict) -> int | None:
    audited = _audit_bedrooms_via_labels(p)
    if audited is not None:
        return audited
    rc = p.get("room_counts") or {}
    return _room_count(rc, "bedroom")


def _derive_bathrooms(p: dict) -> int | None:
    """v6+: bathrooms = bathroom (with bath) + shower_room (shower only).
    Excludes wc / cloakroom (strict listing convention)."""
    rc = p.get("room_counts") or {}
    bath = _room_count(rc, "bathroom")
    shower = _room_count(rc, "shower_room")
    return bath + shower


def _derive_receptions(p: dict) -> int:
    rc = p.get("room_counts") or {}
    return sum(_room_count(rc, k) for k in ("living", "dining", "reception"))


def _derive_room_count(p: dict) -> int:
    rc = p.get("room_counts") or {}
    return sum(_room_count(rc, k) for k in (
        "bedroom","bathroom","shower_room","wc","kitchen",
        "living","dining","reception","study","utility","storage",
    ))


def insert_result(conn, rm_uuid, idx, url, vlm, error=None):
    if vlm is None or vlm.get("parsed") is None:
        conn.execute("""
            INSERT OR REPLACE INTO floorplan_vlm_results
              (rm_uuid, floorplan_idx, floorplan_url, model_version,
               raw_response, inference_ms, ok, error)
            VALUES (?, ?, ?, ?, ?, ?, 0, ?)
        """, (rm_uuid, idx, url, MODEL_VERSION,
              ((vlm or {}).get("raw") or "")[:2500], (vlm or {}).get("inference_ms"),
              error or "json_parse_failed"))
        return

    p = vlm["parsed"]
    feats = p.get("features") or {}
    bath_layout = p.get("bathroom_layout") or {}
    rc = p.get("room_counts") or {}

    # ensuite_count compat: v6+ uses ensuite_bath + ensuite_shower; older v5 used "ensuite"
    if "ensuite_bath" in bath_layout or "ensuite_shower" in bath_layout:
        ensuite = (_as_int(bath_layout.get("ensuite_bath")) or 0) + (_as_int(bath_layout.get("ensuite_shower")) or 0)
    else:
        ensuite = _as_int(bath_layout.get("ensuite"))

    conn.execute("""
        INSERT OR REPLACE INTO floorplan_vlm_results
          (rm_uuid, floorplan_idx, floorplan_url, model_version,
           total_sqft, bedrooms, bathrooms, reception_rooms,
           ensuite_count, floors, room_count,
           layout_class, confidence,
           has_garden, has_garage, has_balcony, has_terrace, has_fireplace, has_conservatory,
           room_counts_json, bathroom_layout_json, per_floor_json, features_json,
           raw_extraction_json, raw_response,
           inference_ms, ok, error)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, NULL)
    """, (
        rm_uuid, idx, url, MODEL_VERSION,
        _as_int(p.get("total_sqft")),
        _derive_bedrooms(p), _derive_bathrooms(p), _derive_receptions(p),
        ensuite, _as_int(p.get("floors")),
        _derive_room_count(p),
        p.get("layout_class"), p.get("confidence"),
        _as_bool_int(feats.get("garden")), _as_bool_int(feats.get("garage")),
        _as_bool_int(feats.get("balcony")), _as_bool_int(feats.get("terrace")),
        _as_bool_int(feats.get("fireplace")), _as_bool_int(feats.get("conservatory")),
        json.dumps(rc),
        json.dumps(bath_layout),
        json.dumps([]),  # per_floor dropped in v6.2 schema
        json.dumps(feats),
        json.dumps(p),
        vlm["raw"][:2500],
        vlm["inference_ms"],
    ))
    sync_room_areas(conn, rm_uuid, p)


def ensure_room_areas_schema(conn) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS floorplan_room_areas (
          rm_uuid       TEXT NOT NULL,
          seq           INTEGER NOT NULL,      -- label order within the property
          room_type     TEXT,
          label         TEXT,
          floor         TEXT,
          area_sqft     INTEGER,               -- NULL when the label has no parsable size
          tier          INTEGER,               -- 1 printed m² | 2 rect dims | 3 irregular dims (x0.82)
          shape         TEXT,
          method        TEXT,
          long_edge_m   REAL,                  -- longest printed side (m); valid even when area NULL
          model_version TEXT NOT NULL,
          created_at    TEXT NOT NULL DEFAULT (datetime('now')),
          PRIMARY KEY (rm_uuid, seq)
        )""")
    # add long_edge_m to pre-existing tables (CREATE IF NOT EXISTS won't alter them)
    try:
        conn.execute("ALTER TABLE floorplan_room_areas ADD COLUMN long_edge_m REAL")
    except sqlite3.OperationalError:
        pass  # already present
    # rm_uuid-leading covering index — the radar/chat EXISTS subqueries seek by
    # property. A (room_type, area_sqft)-leading index here made the planner
    # scan ~30k rows per outer row (67k × 30k, query never returned).
    conn.execute("""CREATE INDEX IF NOT EXISTS idx_room_areas_lookup
                    ON floorplan_room_areas(rm_uuid, room_type, area_sqft)""")


# Outdoor spaces are excluded from total_sqft (GIA) and from the cap below.
_OUTDOOR_TYPES = {"balcony", "terrace", "garden", "patio", "outdoor", "roof_terrace", "veranda"}

# The prompt's type enum has no garden/patio, so the VLM types a plain "Garden" /
# "Patio" / "Driveway" (and ~1k balconies) as 'other' — ~25k rows in production.
# Typed 'other', they were summed as indoor rooms and squeezed by the GIA cap
# (2026-09-29: "Garden 9.75 x 3.05m" = 320 sqft stored as 305, the whole flat ×0.955).
# For rows the VLM left untyped we read the label head (text before the first
# dimension): an outdoor noun with no indoor word. "Garden Room/Office/Shed…" and
# "Winter Garden" are buildings/indoor, and "Living Room opening to patio" is a room.
_OUTDOOR_NOUN = re.compile(
    r"\b(gardens?|patios?|courtyards?|terraces?|balcon(?:y|ies)|decking|deck|yards?|"
    r"verandah?s?|driveways?|drive|drive way|forecourts?|decked|parking|lawns?|flat roof)\b")
_INDOOR_WORD = re.compile(
    r"\b(rooms?|office|studio|house|store|storage|shed|cabin|bar|gym|pod|kitchen|living|"
    r"lounge|bedroom|reception|dining|study|loft|landing|hall|hallway|bathroom|utility|"
    r"conservatory|orangery|winter|garage)\b")


def is_outdoor_label(label) -> bool:
    head = re.split(r"\d", (label or "").lower(), maxsplit=1)[0]
    return bool(_OUTDOOR_NOUN.search(head)) and not _INDOOR_WORD.search(head)


def _outdoor_kind(a: dict):
    """The outdoor type of a room row, or None if it is indoor. Typed rows keep
    their type; for rows the VLM left 'other' the label head decides, and only an
    unmixed head maps to balcony / terrace / patio ("Patio / Garden" → 'outdoor') —
    the balcony and shape-audit guards below key on it, so a balcony typed 'other'
    no longer slips past them (3 wrap-around bboxes larger than the whole flat did)."""
    t = a.get("type")
    if t in _OUTDOOR_TYPES:
        return t
    if t not in (None, "", "other") or not is_outdoor_label(a.get("label")):
        return None
    head = re.split(r"\d", (a.get("label") or "").lower(), maxsplit=1)[0]
    nouns = {m.group(1) for m in _OUTDOOR_NOUN.finditer(head)}
    if all(n.startswith("balcon") for n in nouns):
        return "balcony"
    if all(n.startswith(("terrace", "patio")) for n in nouns):
        return "terrace" if any(n.startswith("terrace") for n in nouns) else "patio"
    return "outdoor"


def _is_outdoor(a: dict) -> bool:
    return _outdoor_kind(a) is not None


def apply_gia_cap(areas: list[dict], total_sqft) -> bool:
    """Reliability guard: indoor per-room areas must never sum ABOVE the trusted
    total floor area (GIA). The tier-2/3 dims overshoot on ~21% of properties
    (indoor room sum > total — physically impossible: rooms ⊆ GIA). When that
    happens, scale indoor rooms down proportionally to the total so downstream
    (check cards / radar / chat) never sees an over-stated room. Outdoor spaces
    (balcony/terrace/…) are not part of GIA → untouched. Returns True if capped."""
    if not total_sqft or total_sqft <= 0:
        return False
    indoor = [a for a in areas if a.get("area_sqft") and not _is_outdoor(a)]
    s = sum(a["area_sqft"] for a in indoor)
    if s <= total_sqft:
        return False
    scale = total_sqft / s
    for a in indoor:
        a["area_sqft"] = int(round(a["area_sqft"] * scale))
    return True


def flag_implausible_balcony(areas: list[dict], total_sqft) -> int:
    """A BALCONY (not terrace — terraces can legitimately be large) whose
    dims-derived area exceeds the whole flat's GIA is physically impossible —
    it's a bounding-box over-estimate of an L-shaped / wrap-around / angled
    balcony (the printed 'WxH' is the extent, not the area). We can't recover
    the true area from text, so NULL it (mark unreliable) rather than emit a
    wrong number — downstream balcony-area conditions then won't false-match.
    Returns count nulled. (Partial guard: only catches the egregious >GIA cases;
    moderate over-estimates need visual shape detection — see v6.7 work.)"""
    if not total_sqft or total_sqft <= 0:
        return 0
    n = 0
    for a in areas:
        if _outdoor_kind(a) == "balcony" and a.get("area_sqft") and a["area_sqft"] > total_sqft:
            a["area_sqft"] = None
            a["method"] = "suspect_oversized"
            a["shape"] = "irregular"
            n += 1
    return n


def compute_room_areas(conn, rm_uuid: str, parsed: dict) -> list[dict]:
    """The rows sync_room_areas would write, without writing them (the GIA cap,
    balcony and outdoor-shape guards applied). Mutates parsed['_room_areas'] in
    place, as sync always has. Split out so a recompute can dry-run the exact
    same logic."""
    areas = parsed.get("_room_areas")
    if areas is None:
        areas = []
        for l in parsed.get("room_labels") or []:
            entry = {"label": l.get("label"), "type": l.get("type"), "floor": l.get("floor")}
            entry.update(extract_room_area(l.get("label") or "")
                         or {"area_sqft": None, "tier": None, "shape": None, "method": None, "long_edge_m": None})
            areas.append(entry)
    # Pre-existing _room_areas blobs lack long_edge_m (added later) — re-derive it
    # from each label so backfill populates the new column. Area guards below only
    # touch area_sqft, so long_edge_m survives even when the area is NULLed.
    for a in areas:
        if a.get("long_edge_m") is None and a.get("label"):
            _ai = extract_room_area(a["label"])
            if _ai and _ai.get("long_edge_m") is not None:
                a["long_edge_m"] = _ai["long_edge_m"]
    _gia = _as_int(parsed.get("total_sqft"))
    apply_gia_cap(areas, _gia)
    flag_implausible_balcony(areas, _gia)
    # Consult the VLM-vision outdoor-shape audits: if the property's BALCONY
    # (balcony_shape_audit.py → irregular_outdoor) or TERRACE/patio
    # (terrace_shape_audit.py → irregular_terrace) was judged a real notch/L/U/
    # wrap from the drawing, its dims (tier 2/3) bbox area is an over-estimate →
    # NULL it. Per-room-TYPE flags so a genuine rectangular terrace following an
    # angled wall is KEPT (judged 'rectangular'/'angled', not flagged). Size
    # guard ≥200 sqft (small ones' bbox≈true area, VLM over-flags shape).
    # tier-1 printed_m² kept (accurate). Persists across re-sync.
    _aud = None
    try:
        _aud = conn.execute("SELECT irregular_outdoor, COALESCE(irregular_terrace,0) FROM outdoor_shape_audit WHERE rm_uuid=?", (rm_uuid,)).fetchone()
    except sqlite3.OperationalError:
        try:
            _aud = conn.execute("SELECT irregular_outdoor, 0 FROM outdoor_shape_audit WHERE rm_uuid=?", (rm_uuid,)).fetchone()
        except sqlite3.OperationalError:
            _aud = None
    if _aud:
        irr_bal, irr_terr = _aud[0], _aud[1]
        for a in areas:
            if not (a.get("area_sqft") and a["area_sqft"] >= 200 and a.get("tier") in (2, 3)):
                continue
            kind = _outdoor_kind(a)
            if (irr_bal and kind == "balcony") or (irr_terr and kind in ("terrace", "patio")):
                a["area_sqft"] = None
                a["method"] = "suspect_irregular_shape"
                a["shape"] = "irregular"
    return areas


def sync_room_areas(conn, rm_uuid: str, parsed: dict) -> None:
    """Mirror parsed['_room_areas'] (or recompute from room_labels) into the
    queryable floorplan_room_areas table. Replaces the property's rows so
    re-analysis never leaves stale labels behind."""
    ensure_room_areas_schema(conn)
    areas = compute_room_areas(conn, rm_uuid, parsed)
    conn.execute("DELETE FROM floorplan_room_areas WHERE rm_uuid=?", (rm_uuid,))
    conn.executemany(
        """INSERT INTO floorplan_room_areas
             (rm_uuid, seq, room_type, label, floor, area_sqft, tier, shape, method, long_edge_m, model_version)
           VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        [(rm_uuid, i, a.get("type"), a.get("label"), a.get("floor"),
          a.get("area_sqft"), a.get("tier"), a.get("shape"), a.get("method"), a.get("long_edge_m"), MODEL_VERSION)
         for i, a in enumerate(areas)],
    )


def process_property(uid: str, floorplans: list[tuple[int, str]]) -> tuple[str, dict, str]:
    ordered = sorted(floorplans, key=lambda x: x[0])
    urls = [u for _, u in ordered]
    refs = [(uid, idx) for idx, _ in ordered]   # (pid, idx) for local-first reads
    joined_url = "; ".join(urls[:MAX_IMGS])
    vlm = call_vlm_multi(urls, refs=refs)
    return uid, vlm, joined_url


def process_one_tick(conn: sqlite3.Connection) -> tuple[int, int]:
    """Process one batch. Returns (ok_count, err_count). Returns (0, 0) when no
    work is left (caller should sleep + retry)."""
    batch = claim_property_batch(conn, BATCH)
    if not batch:
        return 0, 0
    total_imgs = sum(len(fps) for _, fps in batch)
    print(f"[claim] {len(batch)} properties ({total_imgs} images) | model={MODEL_VERSION} -> {VLM_URL} (conc={CONCURRENCY})", flush=True)
    t0 = time.time()
    ok_count = err_count = total_tokens_out = 0

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        futures = {pool.submit(process_property, uid, fps): uid for uid, fps in batch}
        for fut in as_completed(futures):
            uid = futures[fut]
            try:
                uid, vlm, joined_url = fut.result()
                n_img = vlm.get("n_images_used", 1)
                insert_result(conn, uid, AGGREGATED_IDX, joined_url, vlm)
                if vlm.get("parsed"):
                    p = vlm["parsed"]
                    ok_count += 1
                    total_tokens_out += vlm.get("completion_tokens") or 0
                    bd = _derive_bedrooms(p) or "-"
                    ba = _derive_bathrooms(p) or "-"
                    rc = _derive_receptions(p) or "-"
                    print(f"  [ok ] {uid[:8]} imgs={n_img} {p.get('total_sqft','-')}sf "
                          f"{bd}bd/{ba}ba/{rc}rec floors={p.get('floors','-')} "
                          f"{p.get('layout_class','-')} {p.get('confidence','-')} "
                          f"({vlm['inference_ms']}ms, {vlm.get('completion_tokens','-')}tok)", flush=True)
                else:
                    err_count += 1
                    print(f"  [err] {uid[:8]} (imgs={n_img}): parse fail. raw: {(vlm.get('raw') or '')[:80]}", flush=True)
            except Exception as e:
                err_count += 1
                err_msg = str(e)[:200]
                print(f"  [exc] {uid[:8]}: {err_msg}", flush=True)
                transient = any(s in err_msg for s in (
                    "HTTP Error 404", "HTTP Error 500", "HTTP Error 502", "HTTP Error 503",
                    "ConnectionError", "Connection refused", "timed out", "Read timeout",
                    "URLError",
                ))
                if not transient:
                    insert_result(conn, uid, AGGREGATED_IDX, "", None, f"dispatcher_exception: {err_msg}")

    conn.commit()

    elapsed = time.time() - t0
    avg_tok = total_tokens_out / max(ok_count, 1)
    rate = ok_count / max(elapsed, 0.1)
    print(f"[done] {ok_count} ok + {err_count} err in {elapsed:.1f}s "
          f"({rate:.2f} property/s, avg {avg_tok:.0f} tok out)", flush=True)
    return ok_count, err_count


def main() -> int:
    if not DB.exists():
        print(f"[err] DB missing: {DB}", file=sys.stderr); return 1

    conn = sqlite3.connect(DB, timeout=60)
    conn.execute("PRAGMA busy_timeout = 60000")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute(f"ATTACH DATABASE '{SOLD_DB}' AS sold")

    # Long-running loop: keep vLLM saturated by feeding batches back-to-back.
    # No inter-tick idle gap (was ~10-30s under launchd StartInterval=60 model).
    # When queue is empty (caught up), sleep IDLE_SLEEP then retry.
    IDLE_SLEEP = int(os.environ.get("WTL_VLM_IDLE_SLEEP", "30"))
    consecutive_empty = 0
    while True:
        try:
            ok, err = process_one_tick(conn)
            if ok == 0 and err == 0:
                consecutive_empty += 1
                if consecutive_empty == 1:
                    print(f"[idle] queue empty, sleeping {IDLE_SLEEP}s...", flush=True)
                time.sleep(IDLE_SLEEP)
            else:
                consecutive_empty = 0
        except KeyboardInterrupt:
            print("[stop] received interrupt", flush=True)
            break
        except Exception as e:
            print(f"[err] tick exception: {e}", flush=True)
            time.sleep(5)
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
