"""VLM compass detection: one question per image, asking only "is there one + normalised bbox".

Measured (2026-09-02, Qwen3.6-27B, 1288px long edge): ~2 s/image, bboxes match the research-phase
results pixel for pixel; detection rate 56-67% across four batches, false detections ~2/64 (entrance
arrows, sunrise/sunset diagrams) -- which is why a downstream "is this a compass" classifier gate exists.
"""
from __future__ import annotations

import base64
import io
import json
import os
import time
import urllib.request

from PIL import Image

# OpenAI-compatible VLM endpoint on the GPU box, e.g. "http://gpu-box:8200".
VLM_URL_DEFAULT = os.environ.get("VLM_URL", "http://gpu-box:8200")
MODEL_DEFAULT = "qwen3.6-27b"
EDGE = 1288

DETECT_PROMPT = (
    "You are analysing a UK property floorplan image. Your ONLY task: find the compass indicator "
    "(north arrow / compass rose / an 'N' with a pointer). It is often tiny, in a corner, rotated, or "
    "part of an agency logo. Ignore 'To Garden' style arrows, entrance arrows and sunrise/sunset "
    "diagrams — they are not compasses.\n"
    "Output ONLY a JSON object starting immediately with `{`:\n"
    '{"compass_present": true|false, "bbox_2d": [x1, y1, x2, y2] or null}\n'
    "bbox_2d = tight box around the whole compass symbol including its letters."
)


def image_to_data_uri(im: Image.Image, edge: int = EDGE) -> str:
    im = im.convert("RGB")
    w, h = im.size
    s = edge / max(w, h)
    if s < 1:
        im = im.resize((int(w * s), int(h * s)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def parse_detect(raw: str):
    if not raw:
        return None, None
    s, e = raw.find("{"), raw.rfind("}")
    if s < 0 or e <= s:
        return None, None
    try:
        obj = json.loads(raw[s:e + 1])
    except json.JSONDecodeError:
        return None, None
    present = obj.get("compass_present")
    if not isinstance(present, bool):
        return None, None
    bbox = obj.get("bbox_2d")
    ok = (isinstance(bbox, list) and len(bbox) == 4
          and all(isinstance(v, (int, float)) and 0 <= v <= 1000 for v in bbox)
          and bbox[2] > bbox[0] and bbox[3] > bbox[1])
    return present, ([int(v) for v in bbox] if ok else None)


def detect_compass(data_uri: str, vlm_url: str = VLM_URL_DEFAULT, model: str = MODEL_DEFAULT,
                   timeout: int = 120) -> dict:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": data_uri}},
            {"type": "text", "text": DETECT_PROMPT}]}],
        "max_tokens": 80, "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(f"{vlm_url.rstrip('/')}/v1/chat/completions", data=body,
                                 headers={"content-type": "application/json"}, method="POST")
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        resp = json.loads(r.read().decode())
    raw = resp["choices"][0]["message"]["content"]
    present, bbox = parse_detect(raw)
    return {"present": present, "bbox": bbox, "raw": raw, "inference_ms": int((time.time() - t0) * 1000)}
