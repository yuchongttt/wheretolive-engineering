"""Compass-ness classifier contract: model output shape.

(Extract note: the source file also has a weights smoke test -- known seed crops must score >= 0.5 and a
pure text block < 0.5. It needs the trained weights and the seed crops, neither of which is in git, so it
is not included here.)"""
import sys
from pathlib import Path
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
torch = pytest.importorskip("torch")
from lib import compass_presence as cp  # noqa: E402


def test_presence_model_shape():
    mdl = cp.build_presence_model(pretrained=False)
    assert tuple(mdl(torch.zeros(2, 3, 160, 160)).shape) == (2, 1)
