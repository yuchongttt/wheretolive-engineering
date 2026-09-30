"""Compass-ness binary classifier (is this crop a compass?): blocks VLM false detections such as
entrance arrows, sunrise/sunset diagrams and text blocks.
Positives = the 234 seed crops (synthetic random rotations); negatives = isolated ink clusters from the
CV locator on the same floorplans that do not overlap the true compass box (logos / text / icons /
entrance arrows) + random text blocks. Inference: 4x90-degree TTA, mean of the sigmoids."""
from __future__ import annotations

import random

import torch
import torch.nn as nn
import torchvision
from PIL import Image

from . import compass_reader_model as m


def build_presence_model(pretrained: bool = True) -> nn.Module:
    w = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    net = torchvision.models.resnet18(weights=w)
    net.fc = nn.Linear(512, 1)
    return net


def load_presence(path: str, device: torch.device | None = None):
    device = device or m.pick_device()
    net = build_presence_model(pretrained=False)
    net.load_state_dict(torch.load(path, map_location="cpu"))
    return net.eval().to(device), device


@torch.no_grad()
def presence_prob(model: nn.Module, img: Image.Image, device: torch.device) -> float:
    model.eval()
    ps = []
    for k in range(4):
        rot = img.rotate(90 * k, resample=Image.BICUBIC, fillcolor=255)
        t, _ = m.synth(rot, 0.0, m.INPUT, train=False, rng=random.Random(0))
        ps.append(torch.sigmoid(model(t[None].to(device)))[0, 0].item())
    return float(sum(ps) / len(ps))
