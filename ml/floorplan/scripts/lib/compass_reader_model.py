"""Compass-reader CNN: model definition, synthetic-rotation samples, 4x90-degree TTA.

Angle convention (matches the research/compass-orientation ground truth): deg = clockwise angle from
page "up" to north (clock hour x 30). PIL Image.rotate(theta) is counter-clockwise, so the label after
rotation is (deg - theta) mod 360.
Training uses "real seed crops x random synthetic rotations"; inference uses 4x90-degree rotation TTA:
the circular mean of the four un-rotated readings is deg, and the largest pairwise circular distance is
the spread (input to the equivariance gate). Research log: research/compass-orientation/README.md,
section "Reader spike".
"""
from __future__ import annotations

import math
import random

import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image, ImageDraw, ImageFilter

MEAN, STD = 0.449, 0.226      # ImageNet greyscale mean / std
CROP = 320                    # seed crop side (px)
INPUT = 160                   # network input side (px)
_CANVAS = 480                 # rotation canvas (white margin so rotation doesn't clip corners)


def circ_diff(a: float, b: float) -> float:
    d = abs(a - b) % 360
    return min(d, 360 - d)


def circ_mean(vals) -> float:
    r = np.radians(list(vals))
    ang = math.degrees(math.atan2(np.sin(r).mean(), np.cos(r).mean())) % 360
    return 0.0 if ang >= 360.0 - 1e-9 else ang   # -1e-17 deg % 360 floats up to 360.0


def angle_of(v: torch.Tensor) -> torch.Tensor:
    """[N,2] (cos, sin) -> degrees [N], 0-360."""
    return torch.rad2deg(torch.atan2(v[:, 1], v[:, 0])) % 360


def _to_tensor(a: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(((a - MEAN) / STD).astype(np.float32))[None].repeat(3, 1, 1)


def synth(img: Image.Image, deg: float, size: int, train: bool, rng: random.Random):
    """Seed crop (L, 320) -> network input. With train=True: random rotation / scale / offset /
    clutter lines / photometric jitter; returns (tensor[3,size,size], deg after rotation).
    train=False is a deterministic centre crop."""
    S = _CANVAS
    canvas = Image.new("L", (S, S), 255)
    canvas.paste(img, ((S - CROP) // 2, (S - CROP) // 2))
    if train:
        theta = rng.uniform(0, 360)
        canvas = canvas.rotate(theta, resample=Image.BICUBIC, fillcolor=255)
        deg = (deg - theta) % 360
        d = ImageDraw.Draw(canvas)
        for _ in range(rng.randint(0, 5)):          # floorplan clutter: lines / bars away from the centre
            x0, y0 = rng.randint(0, S), rng.randint(0, S)
            if math.hypot(x0 - S / 2, y0 - S / 2) < 110:
                continue
            if rng.random() < 0.6:
                d.line([(x0, y0), (x0 + rng.randint(-200, 200), y0 + rng.randint(-200, 200))],
                       fill=rng.randint(0, 90), width=rng.randint(1, 4))
            else:
                d.rectangle([x0, y0, x0 + rng.randint(10, 80), y0 + rng.randint(4, 12)],
                            fill=rng.randint(0, 120))
        side = rng.uniform(210, 400)
        ox, oy = rng.uniform(-30, 30), rng.uniform(-30, 30)
    else:
        side, ox, oy = CROP, 0.0, 0.0
    cx, cy = S / 2 + ox, S / 2 + oy
    crop = canvas.crop((int(cx - side / 2), int(cy - side / 2),
                        int(cx + side / 2), int(cy + side / 2))).resize((size, size), Image.BICUBIC)
    a = np.asarray(crop, dtype=np.float32) / 255.0
    if train:
        if rng.random() < 0.3:
            crop = crop.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.2)))
            a = np.asarray(crop, dtype=np.float32) / 255.0
        a = np.clip((a - 0.5) * rng.uniform(0.7, 1.3) + 0.5 + rng.uniform(-0.1, 0.1), 0, 1)
        if rng.random() < 0.3:
            a = np.clip(a + np.random.default_rng(rng.randrange(1 << 30)).normal(0, 0.03, a.shape), 0, 1)
    return _to_tensor(a), deg


def build_model(pretrained: bool = True) -> nn.Module:
    w = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    m = torchvision.models.resnet18(weights=w)
    m.fc = nn.Linear(512, 2)
    return m


def pick_device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def load_reader(path: str, device: torch.device | None = None):
    device = device or pick_device()
    m = build_model(pretrained=False)
    m.load_state_dict(torch.load(path, map_location="cpu"))
    return m.eval().to(device), device


@torch.no_grad()
def predict_tta(model: nn.Module, img: Image.Image, device: torch.device, size: int = INPUT) -> dict:
    """4x90-degree rotation TTA. rotate(90k) is counter-clockwise -> add 90k to un-rotate the reading;
    circular mean = deg, largest pairwise circular distance = spread."""
    model.eval()
    preds = []
    for k in range(4):
        rot = img.rotate(90 * k, resample=Image.BICUBIC, fillcolor=255)
        t, _ = synth(rot, 0.0, size, train=False, rng=random.Random(0))
        v = model(t[None].to(device)).cpu()
        preds.append((angle_of(v)[0].item() + 90 * k) % 360)
    return {"deg": circ_mean(preds),
            "spread": max(circ_diff(a, b) for a in preds for b in preds),
            "preds": [round(p, 1) for p in preds]}
