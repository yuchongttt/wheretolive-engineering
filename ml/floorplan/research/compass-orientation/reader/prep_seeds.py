#!/usr/bin/env python3
"""Seed crops for the compass-reader spike: truth pid -> 320px white-bg square crop around the compass.
Window source: VLM bbox (detect_results/bbox_results, 1288 space) else CV locator top-1.
Writes seeds/<pid>.png + seeds.tsv (pid, batch, clock, deg, style, source, cx, cy, r) + contact sheet with truth arrow overlay.
Extract note: inputs (fp_images/, batch*/ truth and detection files) are data and are not included."""
import csv, glob, json, os, sys, math
from PIL import Image, ImageDraw
import numpy as np
SP = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SP)  # research/compass-orientation
sys.path.insert(0, f'{ROOT}/scripts')
from compass_extract import _load_rgb
from compass_locate import locate_candidates, to_1288
IMG = f'{SP}/fp_images'; OUT = f'{SP}/seeds'; os.makedirs(OUT, exist_ok=True)

def crop_native(im, cx1288, cy1288, r1288, size=320):
    W, H = im.size
    s = max(W, H) / min(1288, max(W, H))
    cx, cy, r = int(cx1288 * s), int(cy1288 * s), int(r1288 * s)
    r = max(40, min(r, 600)); cx = min(max(cx, 0), W - 1); cy = min(max(cy, 0), H - 1)
    canvas = Image.new('RGB', (2 * r, 2 * r), (255, 255, 255))
    src = im.crop((max(0, cx - r), max(0, cy - r), min(W, cx + r), min(H, cy + r)))
    canvas.paste(src, (max(0, cx - r) - (cx - r), max(0, cy - r) - (cy - r)))
    return canvas.resize((size, size), Image.LANCZOS)

rows = []
for b in (1, 2, 3, 4):
    bb = {}
    if b == 1:
        bb = {k: v for k, v in json.load(open(f'{ROOT}/batch1/bbox_results.json')).items()}
    else:
        bb = {k: v.get('bbox_2d') for k, v in json.load(open(f'{ROOT}/batch{b}/detect_results.json')).items()}
    for row in csv.reader(open(f'{ROOT}/batch{b}/truth_angles.tsv'), delimiter='\t'):
        if not row or row[0] == 'pid': continue
        pid = row[0]
        try: clock = float(row[1])
        except ValueError: print('skip non-numeric', pid, row[1:3]); continue
        style = row[2] if len(row) > 2 else ''
        paths = glob.glob(f'{IMG}/{pid}.*')
        if not paths: print('missing image', pid); continue
        im = _load_rgb(paths[0]); im1288 = to_1288(im)
        box = bb.get(pid)
        if box and len(box) == 4 and box[2] > box[0]:
            # Qwen bbox_2d is NORMALISED 0-1000 (all 236 boxes <=1000, several exceed native width) -> map to 1288 space
            w1, h1 = im1288.size
            cx, cy = (box[0] + box[2]) / 2000 * w1, (box[1] + box[3]) / 2000 * h1
            r = max(70, max((box[2] - box[0]) / 1000 * w1, (box[3] - box[1]) / 1000 * h1) * 0.75); src = 'vlm'
        else:
            c = locate_candidates(im1288, max_cands=3)
            if not c: print('no candidate', pid); continue
            _, cx, cy, r = c[0]; src = 'cv'
        crop = crop_native(im, cx, cy, r)
        crop.save(f'{OUT}/{pid}.png')
        rows.append(dict(pid=pid, batch=b, clock=clock, deg=(clock * 30) % 360, style=style, source=src, cx=round(cx), cy=round(cy), r=round(r)))
with open(f'{SP}/seeds.tsv', 'w') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter='\t'); w.writeheader(); w.writerows(rows)
print('seeds', len(rows), 'vlm', sum(r['source'] == 'vlm' for r in rows), 'cv', sum(r['source'] == 'cv' for r in rows))
# contact sheets, 8 cols, 160px tiles, truth arrow overlay (red = north per truth)
T, C = 170, 8
for b in (1, 2, 3, 4):
    rs = [r for r in rows if r['batch'] == b]
    R = math.ceil(len(rs) / C)
    sheet = Image.new('RGB', (C * T, R * T), (230, 230, 230)); d = ImageDraw.Draw(sheet)
    for i, r in enumerate(rs):
        x0, y0 = (i % C) * T, (i // C) * T
        tile = Image.open(f'{OUT}/{r["pid"]}.png').resize((T - 10, T - 10))
        sheet.paste(tile, (x0 + 5, y0 + 5))
        cx, cy, L = x0 + T / 2, y0 + T / 2, 60
        a = math.radians(r['deg']); ex, ey = cx + L * math.sin(a), cy - L * math.cos(a)
        d.line([(cx, cy), (ex, ey)], fill=(255, 0, 0), width=3); d.ellipse([ex - 5, ey - 5, ex + 5, ey + 5], fill=(255, 0, 0))
        d.rectangle([x0 + 5, y0 + 5, x0 + T - 5, y0 + 22], fill=(255, 255, 200))
        d.text((x0 + 8, y0 + 7), f"{i}:{r['pid']} {r['clock']} {r['source']}", fill=(0, 0, 0))
    sheet.save(f'{SP}/sheet_b{b}.png')
print('sheets written')
