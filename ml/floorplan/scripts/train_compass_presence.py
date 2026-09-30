#!/usr/bin/env python3
"""Train the compass-ness classifier. Negative mining: run the CV locator (research compass_locate) on the
236 ground-truth floorplans, keep candidate-window crops (320px) that do not overlap the true compass
window, plus 2 random text / wall patches per image.
--eval-batch N: hold out that batch (positives and negatives) for evaluation and print FN/FP; without it,
train on everything and write the production weights.
Source images default to research/compass-orientation/reader/fp_images (not included in this extract;
the image fetcher is out of scope). Seeds listed in $COMPASS_EXCLUDE_SEEDS are dropped, as in
train_compass_reader.py."""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "research/compass-orientation/scripts"))
from lib import compass_north as cn, compass_reader_model as m, compass_presence as cp  # noqa: E402
from compass_locate import locate_candidates, to_1288  # noqa: E402

READER = REPO / "research/compass-orientation/reader"


def mine_negatives(rows, img_dir: Path, rng: random.Random):
    """Return [(pid, batch, PIL.L 320)]. cx/cy/r in seeds.tsv are in 1288-px coordinates."""
    out = []
    for r in rows:
        paths = glob.glob(str(img_dir / f"{r['pid']}.*"))
        if not paths:
            continue
        im = cn.flatten_on_white(Image.open(paths[0]))
        W, H = im.size
        im1288 = to_1288(im)
        s = max(W, H) / max(im1288.size)  # 1288 → native
        cx, cy, rr = float(r["cx"]), float(r["cy"]), float(r["r"])
        for _, x, y, rad in locate_candidates(im1288, max_cands=6):
            # Exclude by centre distance: a candidate that could contain any part of the true compass
            # is not a negative (an IoU<0.05 rule once let through a crop with a compass at its edge,
            # 1 in 80 sampled).
            if math.hypot(x - cx, y - cy) > rr * 1.5 + rad:
                out.append((r["pid"], int(r["batch"]), cn.crop_window(im, x * s, y * s, rad * s)))
        for _ in range(2):  # random patches (mostly text / wall lines)
            x, y = rng.uniform(0.1, 0.9) * W, rng.uniform(0.1, 0.9) * H
            crop = cn.crop_window(im, x, y, rng.uniform(60, 160))
            if np.asarray(crop).mean() < 250:  # keep only if not plain white
                out.append((r["pid"], int(r["batch"]), crop))
    return out


class DS(torch.utils.data.Dataset):
    def __init__(self, items, per, seed):
        self.items, self.per, self.seed = items, per, seed

    def __len__(self):
        return len(self.items) * self.per

    def __getitem__(self, i):
        img, y = self.items[i % len(self.items)]
        t, _ = m.synth(img, 0.0, m.INPUT, True, random.Random(self.seed * 7919 + i))
        return t, torch.tensor([y], dtype=torch.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-dir", default=str(READER / "fp_images"))
    ap.add_argument("--eval-batch", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(cn.PRESENCE_WEIGHTS))
    a = ap.parse_args()
    rng = random.Random(a.seed); torch.manual_seed(a.seed)
    excl = {s for s in os.environ.get("COMPASS_EXCLUDE_SEEDS", "").split(",") if s}
    rows = [r for r in csv.DictReader(open(READER / "seeds.tsv"), delimiter="\t") if r["pid"] not in excl]
    pos = [(r["pid"], int(r["batch"]), Image.open(READER / "seeds" / f"{r['pid']}.png").convert("L")) for r in rows]
    neg = mine_negatives(rows, Path(a.img_dir), rng)
    print(f"pos {len(pos)} neg {len(neg)}", flush=True)
    tr = [(im, 1.0) for p, b, im in pos if b != a.eval_batch] + [(im, 0.0) for p, b, im in neg if b != a.eval_batch]
    ev = [(im, 1.0) for p, b, im in pos if b == a.eval_batch] + [(im, 0.0) for p, b, im in neg if b == a.eval_batch]
    ev_pids = [p for p, b, im in pos if b == a.eval_batch] + [f"neg:{p}" for p, b, im in neg if b == a.eval_batch]
    dev = m.pick_device(); net = cp.build_presence_model(True).to(dev)
    per = max(1, int(48 * len(pos) / max(1, len(tr))))  # ~48 samples per seed-equivalent for positives and negatives
    dl = torch.utils.data.DataLoader(DS(tr, per, a.seed), batch_size=64, shuffle=True, num_workers=0)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([len(neg) / max(1, len(pos))], device=dev))
    t0 = time.time()
    for ep in range(a.epochs):
        net.train(); tot = n = 0
        for x, y in dl:
            x, y = x.to(dev), y.to(dev)
            loss = lossf(net(x), y); opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item(); n += 1
        print(f"ep {ep+1}/{a.epochs} loss {tot/n:.4f} {time.time()-t0:.0f}s", flush=True)
    if ev:
        probs = [(cp.presence_prob(net, im, dev), y) for im, y in ev]
        fn = sum(1 for p, y in probs if y == 1 and p < 0.5)
        fp = sum(1 for p, y in probs if y == 0 and p >= 0.5)
        print(f"EVAL batch{a.eval_batch}: pos {sum(1 for _, y in ev if y == 1)} FN {fn} | "
              f"neg {sum(1 for _, y in ev if y == 0)} FP {fp}")
        print("FN (pid, prob):", [(pid, round(p, 3)) for pid, (p, y) in zip(ev_pids, probs) if y == 1 and p < 0.5])
        print("FP (pid, prob):", [(pid, round(p, 3)) for pid, (p, y) in zip(ev_pids, probs) if y == 0 and p >= 0.5])
        print("pos probs sorted low→high:", sorted(round(p, 3) for p, y in probs if y == 1)[:8])
        return
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.cpu() for k, v in net.state_dict().items()}, out)
    meta = {"weights": out.name, "sha256": hashlib.sha256(out.read_bytes()).hexdigest(), "n_pos": len(pos),
            "n_neg": len(neg), "epochs": a.epochs, "seed": a.seed,
            "git": subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                                  cwd=REPO).stdout.strip()}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1)); print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
