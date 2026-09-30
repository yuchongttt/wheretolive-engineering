"""compass_reader_model contract: label direction under synthetic rotation, TTA un-rotation and
circular statistics. Convention: deg = clockwise angle from page "up" to north; PIL rotate(theta) is
counter-clockwise -> label = (deg - theta) mod 360."""
import math, random, sys
from pathlib import Path
import pytest
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
torch = pytest.importorskip("torch")
from lib import compass_reader_model as m  # noqa: E402


def _arrow(deg: float) -> Image.Image:
    """320px white canvas, thick black arrow from the centre pointing at deg (clockwise from up)."""
    im = Image.new("L", (m.CROP, m.CROP), 255); d = ImageDraw.Draw(im)
    a = math.radians(deg); cx = cy = m.CROP / 2
    ex, ey = cx + 100 * math.sin(a), cy - 100 * math.cos(a)
    d.line([(cx, cy), (ex, ey)], fill=0, width=12)
    d.ellipse([ex - 14, ey - 14, ex + 14, ey + 14], fill=0)
    return im


def test_circ_diff_and_mean():
    assert m.circ_diff(350, 10) == 20
    assert m.circ_diff(0, 180) == 180
    assert abs(m.circ_mean([350, 10]) - 0) < 1e-6
    assert abs(m.circ_mean([90, 90]) - 90) < 1e-6


def test_angle_of_unit_vectors():
    v = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    assert [round(x) for x in m.angle_of(v).tolist()] == [0, 90, 180, 270]


def test_synth_eval_is_deterministic_and_shaped():
    im = _arrow(0)
    t1, d1 = m.synth(im, 0.0, m.INPUT, train=False, rng=random.Random(0))
    t2, d2 = m.synth(im, 0.0, m.INPUT, train=False, rng=random.Random(1))
    assert tuple(t1.shape) == (3, m.INPUT, m.INPUT) and t1.dtype == torch.float32
    assert d1 == d2 == 0.0 and torch.equal(t1, t2)


def test_synth_train_rotates_label_ccw_convention():
    """rotate(theta) is counter-clockwise: an up-pointing (0) arrow rotated by 90 points left (270). Pin
    theta=90 through the rng and check that pixels and label agree."""
    im = _arrow(0)

    class R(random.Random):
        def uniform(self, a, b):
            return 90.0 if (a, b) == (0, 360) else super().uniform(a, b)

        def random(self):
            return 1.0  # disable every probabilistic augmentation

        def randint(self, a, b):
            return 0

    t, d = m.synth(im, 0.0, m.INPUT, train=True, rng=R(0))
    assert d == 270.0
    # after de-normalising, the left half should hold clearly more ink than the right (arrow points left)
    g = t[0] * m.STD + m.MEAN
    left, right = (1 - g[:, : m.INPUT // 2]).sum().item(), (1 - g[:, m.INPUT // 2:]).sum().item()
    assert left > right * 1.5


def test_predict_tta_unrotates_consistently():
    """Replace the model with a stub that reads the black arrow's geometric direction; after TTA's four
    un-rotations the spread should be ~0."""
    class Stub(torch.nn.Module):
        def forward(self, x):
            g = 1 - (x[:, 0] * m.STD + m.MEAN)  # ink amount [N,H,W]
            N, H, W = g.shape
            ys, xs = torch.meshgrid(torch.arange(H).float(), torch.arange(W).float(), indexing="ij")
            out = []
            for i in range(N):
                w = g[i]; tot = w.sum()
                mx, my = (w * xs).sum() / tot - W / 2, (w * ys).sum() / tot - H / 2
                ang = math.atan2(mx.item(), -my.item())  # clockwise from up
                out.append([math.cos(ang), math.sin(ang)])
            return torch.tensor(out)
    res = m.predict_tta(Stub(), _arrow(45), torch.device("cpu"))
    assert m.circ_diff(res["deg"], 45) < 8 and res["spread"] < 8 and len(res["preds"]) == 4
