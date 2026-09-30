#!/usr/bin/env python3
"""Route under test: ask the VLM once "which side of the building is the main outdoor area on + its
bbox", with a 4x90-degree rotation-equivariance gate.

One call evaluates two routes at once:
  G1 direct side word -- the model itself says top/right/bottom/left
  G2 bbox + geometry  -- side from the vector between the bbox centre and the building's ink centroid
                         (the compass-lane approach: the VLM only detects, geometry decides)
Gate: map each of the 4 rotated frames' readings back to the original frame; emit only if all agree,
otherwise abstain. The compass experiments showed four times over that same-source self-consistency
cannot catch correlated errors, and rotation equivariance is the one check that does.

Writes predictions_g.jsonl (one row per image, with the raw readings of all four frames so a different
gate can be re-scored later).
"""
import json, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

B = Path(__file__).parent
REPO = B.parents[2]
sys.path.insert(0, str(REPO / "scripts"))
from lib import compass_detect as cd  # noqa: E402

SIDES = ["top", "right", "bottom", "left"]      # clockwise; image rotated clockwise by k*90 -> observed index = (original index + k) % 4

PROMPT = """This is a UK estate-agent floorplan. Some floorplans draw the outdoor space (garden, terrace, balcony, patio) as an area outside the building's walls; many do not draw it at all.

First, in one or two short sentences, describe (a) where the building's rooms sit on the page, and (b) whether any outdoor area is actually drawn, and where it sits relative to the building.

Then output ONE JSON line, nothing after it:
{"outdoor_drawn": true or false, "type": "garden" or "terrace" or "balcony" or "patio" or "other", "side": "top" or "right" or "bottom" or "left", "bbox_2d": [x0,y0,x1,y1], "confidence": "high" or "low"}

- "side" = which side of the BUILDING the main outdoor area sits on, as seen on the page right now.
- "bbox_2d" = that outdoor area's bounding box in 0-1000 normalised page coordinates.
- Text that merely mentions a garden is NOT a drawn outdoor area; set outdoor_drawn false.
- A scale bar, key, logo, parking bay or roof plan is NOT the main outdoor area.
- If nothing is drawn, set outdoor_drawn false and omit "side" and "bbox_2d"."""


def ask(im: Image.Image) -> dict:
    body = json.dumps({"model": cd.MODEL_DEFAULT, "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": cd.image_to_data_uri(im, 1024)}},
        {"type": "text", "text": PROMPT}]}],
        "max_tokens": 500, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(f"{cd.VLM_URL_DEFAULT}/v1/chat/completions", data=body,
                                 headers={"content-type": "application/json"})
    raw = json.loads(urllib.request.urlopen(req, timeout=300).read())["choices"][0]["message"]["content"]
    e = raw.rfind("}")                          # description first, JSON last: take the last {...} (truncation lesson from P2)
    s = raw.rfind("{", 0, e) if e >= 0 else -1
    try:
        return json.loads(raw[s:e + 1]) if s >= 0 else {"_raw": raw[:200]}
    except json.JSONDecodeError:
        return {"_raw": raw[:200]}


def geometric_side(im: Image.Image, bbox) -> str | None:
    """Side of the bbox centre relative to the ink centroid computed outside the bbox. If the vector is
    too short (the two nearly coincide) the side is undecidable -> None."""
    g = im.convert("L").resize((256, 256), Image.LANCZOS)
    px = g.load()
    x0, y0, x1, y1 = [v / 1000 * 256 for v in bbox]
    sx = sy = n = 0
    for y in range(256):
        for x in range(256):
            if px[x, y] < 200 and not (x0 <= x <= x1 and y0 <= y <= y1):   # ink, and not inside the outdoor block
                sx += x; sy += y; n += 1
    if n < 100:
        return None
    bx, by = sx / n, sy / n
    dx, dy = (x0 + x1) / 2 - bx, (y0 + y1) / 2 - by
    if (dx * dx + dy * dy) ** 0.5 < 12:         # outdoor block overlaps the building; side is meaningless
        return None
    return ("right" if dx > 0 else "left") if abs(dx) > abs(dy) else ("bottom" if dy > 0 else "top")


def rotate_bbox_back(bbox, k):
    """Map a 0-1000 bbox from the frame rotated clockwise by k*90 back to the original frame."""
    for _ in range(k % 4):                       # each step rotates 90 counter-clockwise: (x,y) -> (y, 1000-x)
        x0, y0, x1, y1 = bbox
        bbox = [y0, 1000 - x1, y1, 1000 - x0]
    return bbox


def one_image(pid: str):
    im0 = Image.open(B / "imgs" / f"{pid}.png").convert("RGB")
    frames = []
    for k in range(4):
        im = im0.rotate(-90 * k, expand=True)    # clockwise by k*90
        r = ask(im)
        drawn = bool(r.get("outdoor_drawn"))
        side_obs = r.get("side") if r.get("side") in SIDES else None
        bbox = r.get("bbox_2d") if isinstance(r.get("bbox_2d"), list) and len(r.get("bbox_2d")) == 4 else None
        # back to the original frame
        side_g1 = SIDES[(SIDES.index(side_obs) - k) % 4] if side_obs else None
        side_g2 = None
        if bbox:
            gs = geometric_side(im, bbox)
            side_g2 = SIDES[(SIDES.index(gs) - k) % 4] if gs else None
        frames.append({"k": k, "drawn": drawn, "type": r.get("type"), "conf": r.get("confidence"),
                       "side_g1": side_g1, "side_g2": side_g2,
                       "bbox_orig": rotate_bbox_back(bbox, k) if bbox else None,
                       "raw": r.get("_raw")})
    return {"pid": pid, "frames": frames}


if __name__ == "__main__":
    pids = [l.split("\t")[0] for l in open(B / "sample.tsv").read().splitlines()[1:] if l.strip()]
    out = B / "predictions_g.jsonl"
    done = {json.loads(l)["pid"] for l in open(out)} if out.exists() else set()      # resume from checkpoint
    todo = [p for p in pids if p not in done]
    print(f"todo {len(todo)}/{len(pids)}")
    with open(out, "a") as f, ThreadPoolExecutor(4) as ex:
        for i, res in enumerate(ex.map(one_image, todo), 1):
            f.write(json.dumps(res) + "\n"); f.flush()
            print(f"[{i}/{len(todo)}] {res['pid']} " +
                  " ".join(str(fr["side_g1"]) for fr in res["frames"]), flush=True)
