#!/usr/bin/env python3
"""Train the production compass reader: all seed crops (research/compass-orientation/reader/seeds) with
synthetic rotations, ResNet18.
Deterministic (fixed seed); writes data/models/compass_reader_v1.pt + a same-name .json (sha256,
n_seeds, epochs, git commit, train-real self-check). The held-out estimate is the four-fold
leave-one-batch-out result (see research/compass-orientation/README.md); this script does no hold-out.
Usage: .venv/bin/python3 scripts/train_compass_reader.py [--epochs 25] [--exclude <seed id>,...]
Seeds to exclude default to $COMPASS_EXCLUDE_SEEDS (comma-separated). In the private run one seed was
excluded: its VLM box had landed on a text banner, not a compass.
"""
from __future__ import annotations

import argparse
import csv
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
from lib import compass_reader_model as m  # noqa: E402
from lib.compass_north import READER_WEIGHTS  # noqa: E402

SEEDS = REPO / "research/compass-orientation/reader"


class DS(torch.utils.data.Dataset):
    def __init__(self, rows, per_seed, seed):
        self.rows, self.per_seed, self.seed = rows, per_seed, seed
        self.cache = {r["pid"]: Image.open(SEEDS / "seeds" / f"{r['pid']}.png").convert("L") for r in rows}

    def __len__(self):
        return len(self.rows) * self.per_seed

    def __getitem__(self, i):
        r = self.rows[i % len(self.rows)]
        t, d = m.synth(self.cache[r["pid"]], float(r["deg"]), m.INPUT, True,
                       random.Random(self.seed * 1_000_003 + i))
        a = math.radians(d)
        return t, torch.tensor([math.cos(a), math.sin(a)], dtype=torch.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--per-seed", type=int, default=48)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exclude", default=os.environ.get("COMPASS_EXCLUDE_SEEDS", ""),
                    help="comma-separated seed ids to drop (e.g. a seed whose VLM box landed on a text banner)")
    ap.add_argument("--out", default=str(READER_WEIGHTS))
    a = ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    excl = {s for s in a.exclude.split(",") if s}
    rows = [r for r in csv.DictReader(open(SEEDS / "seeds.tsv"), delimiter="\t") if r["pid"] not in excl]
    dev = m.pick_device()
    model = m.build_model(True).to(dev)
    dl = torch.utils.data.DataLoader(DS(rows, a.per_seed, a.seed), batch_size=a.bs, shuffle=True, num_workers=0)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, total_steps=a.epochs * len(dl), pct_start=0.1)
    t0 = time.time()
    print(f"seeds {len(rows)} device {dev}", flush=True)
    for ep in range(a.epochs):
        model.train(); tot = n = 0
        for x, y in dl:
            x, y = x.to(dev), y.to(dev)
            v = model(x); v = v / (v.norm(dim=1, keepdim=True) + 1e-6)
            loss = (1 - (v * y).sum(1)).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += loss.item(); n += 1
        print(f"ep {ep+1}/{a.epochs} loss {tot/n:.4f} {time.time()-t0:.0f}s", flush=True)
    # train-real self-check (not an accuracy estimate; only proves the weights are not broken)
    errs = []
    for r in rows:
        res = m.predict_tta(model, Image.open(SEEDS / "seeds" / f"{r['pid']}.png").convert("L"), dev)
        errs.append(m.circ_diff(res["deg"], float(r["deg"])))
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, out)
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=REPO).stdout.strip()
    meta = {"weights": out.name, "sha256": sha, "n_seeds": len(rows), "epochs": a.epochs, "per_seed": a.per_seed,
            "seed": a.seed, "excluded": sorted(excl), "git": commit, "device": str(dev),
            "train_real_acc22": float(np.mean(np.array(errs) <= 22.5)), "train_real_median_err": float(np.median(errs)),
            "held_out_estimate": "LOBO 4-fold 233 crops: acc22 94.8%, gate<=10 62.2%@100% (research README)"}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
