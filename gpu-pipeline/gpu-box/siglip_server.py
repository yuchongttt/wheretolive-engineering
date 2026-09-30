#!/usr/bin/env python3
"""SigLIP-2 luxury scoring HTTP server. Sister to dino_server_v3.py.

Endpoints:
  GET  /health         → {"status": "ok", "model": ...}
  POST /score          → multipart/form-data with image file → JSON {score, luxury_cos, budget_cos}

Loaded once on startup; image inference ~50ms on RTX 3060.
"""
from __future__ import annotations
import io, os, sys, time, threading
from http.server import HTTPServer, BaseHTTPRequestHandler

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor
from PIL import Image
import numpy as np
import json

PORT = 8101
CACHE = "/data/ml/hf-cache"
SIGLIP = "google/siglip2-base-patch16-256"
DEVICE = "cuda"

LUXURY_PROMPTS = [
    "a luxurious high-end interior with marble surfaces, gold fixtures, crystal chandelier",
    "an expensive designer interior with premium materials and elegant decor",
    "a luxury upscale apartment interior with high-end finishes and chic furniture",
]
BUDGET_PROMPTS = [
    "a worn-down basic interior with old furniture and dated decor",
    "a cheap simple interior with low-quality materials and bare walls",
    "an outdated budget interior with cluttered shabby furniture",
]

os.environ["HF_HOME"] = CACHE

print("[init] loading SigLIP", flush=True)
proc = AutoProcessor.from_pretrained(SIGLIP, cache_dir=CACHE)
model = AutoModel.from_pretrained(SIGLIP, cache_dir=CACHE).to(DEVICE).eval()

with torch.no_grad():
    txt_in = proc(text=LUXURY_PROMPTS + BUDGET_PROMPTS, return_tensors="pt", padding="max_length").to(DEVICE)
    out = model.get_text_features(**txt_in)
    txt_feats = out.pooler_output if hasattr(out, "pooler_output") and out.pooler_output is not None else out
    txt_feats = F.normalize(txt_feats, dim=-1)
LUX_FEATS = txt_feats[:len(LUXURY_PROMPTS)]
BUD_FEATS = txt_feats[len(LUXURY_PROMPTS):]
print(f"[init] text feats ready · luxury {LUX_FEATS.shape} budget {BUD_FEATS.shape}", flush=True)

LOCK = threading.Lock()

@torch.no_grad()
def score_image(img: Image.Image) -> dict:
    with LOCK:
        img_in = proc(images=img, return_tensors="pt").to(DEVICE)
        out = model.get_image_features(**img_in)
        feats = out.pooler_output if hasattr(out, "pooler_output") and out.pooler_output is not None else out
        feats = F.normalize(feats, dim=-1)
        lux_cos = float((feats @ LUX_FEATS.T).mean())
        bud_cos = float((feats @ BUD_FEATS.T).mean())
    raw = lux_cos - bud_cos
    score = 1.0 / (1.0 + np.exp(-raw * 20.0))
    return {"score": float(score), "luxury_cos": lux_cos, "budget_cos": bud_cos}


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, code: int, payload: dict):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            return self._send_json(200, {"status": "ok", "model": SIGLIP, "device": DEVICE})
        self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/score":
            return self._send_json(404, {"error": "use POST /score"})
        ctype = self.headers.get("Content-Type", "")
        clen = int(self.headers.get("Content-Length", "0"))
        if clen <= 0 or clen > 20_000_000:
            return self._send_json(400, {"error": "image required, max 20MB"})

        raw = self.rfile.read(clen)

        # Accept either raw image bytes or multipart/form-data
        img_bytes = raw
        if ctype.startswith("multipart/form-data"):
            # Cheap multipart parse: find first \r\n\r\n then last boundary
            idx = raw.find(b"\r\n\r\n")
            if idx == -1:
                return self._send_json(400, {"error": "malformed multipart"})
            body = raw[idx + 4:]
            # Strip trailing boundary
            tail = body.rfind(b"\r\n--")
            if tail != -1:
                body = body[:tail]
            img_bytes = body

        try:
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        except Exception as e:
            return self._send_json(400, {"error": f"cannot decode image: {e}"})

        t0 = time.time()
        result = score_image(img)
        result["latency_ms"] = int((time.time() - t0) * 1000)
        result["image_size"] = list(img.size)
        self._send_json(200, result)

    def log_message(self, format, *args):
        # quiet default logging
        sys.stderr.write(f"[{time.strftime('%H:%M:%S')}] {format % args}\n")


if __name__ == "__main__":
    server = HTTPServer(("0.0.0.0", PORT), Handler)
    print(f"[ready] siglip_server on :{PORT}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()
