"""Floorplans with an alpha channel must be flattened onto white, not squashed
into a black sheet.

Background (2026-08-14, found 70 properties into the backfill): a sizeable share
of listing floorplans are **line art drawn on a transparent-background PNG**
(measured: original mode=`LA`, greyscale + alpha). dispatcher.fetch_and_resize
had no alpha compositing at all:

    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")          # LA/P here: transparent -> black
    ...
    img.convert("RGB").save(buf, "JPEG")  # RGBA here: transparent -> black

When PIL simply drops alpha, transparent areas become **black**, and a
floorplan's wall lines/text are themselves dark — black lines on a black
background = nothing visible. Measured on the same image:
  plain convert("RGB")        mean luma 3.5   (all black)
  composited onto white       mean luma 231.6 (normal white image)

The VLM then receives a black sheet and returns
`layout_class=unknown / confidence=low / all zeros`. The fatal part is that this
is **recorded as ok=1**, and the queue skips every ok=1 row — so it is never
retried, a silent permanent write-off.

Scale (v6.6 series, at discovery):
  PNG source      10,085 properties, 2,465 empty parses -> **24.4%**
  non-PNG source  49,096 properties,    30 empty parses -> **0.1%**
A 244x gap. 2,498 rows had already been ruined; the backlog held another 3,012
PNG properties, which would have ruined ~735 more.

The ruined ones were not bad images: re-played on a white background, one is a
clean high-resolution floorplan with "APPROXIMATE GROSS INTERNAL AREA 569 SQ FT
/ 52.9 SQ M" printed at the bottom plus full room dimensions — exactly what the
whole pipeline wants.

The criterion is **mean luma** rather than a per-pixel comparison: line art on
white is necessarily mostly white (high luma), squashed onto black it is
necessarily mostly black (luma near 0); they differ by an order of magnitude,
so there is no misjudgement.
"""
import io
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mac" / "floorplan_vlm"))

Image = pytest.importorskip("PIL.Image", reason="Pillow not installed")
from PIL import Image as PILImage  # noqa: E402

from dispatcher import flatten_on_white  # noqa: E402


def _mean_luma(img) -> float:
    px = list(img.convert("L").getdata())
    return sum(px) / len(px)


def _line_art_with_alpha(mode: str):
    """Build a "dark line art + transparent background" image, reproducing the
    real shape of these listing floorplans.

    Background alpha=0 (fully transparent), lines alpha=255 and very dark —
    exactly the kind of image that vanishes entirely when squashed onto black."""
    w = h = 64
    base = PILImage.new("RGBA", (w, h), (0, 0, 0, 0))       # fully transparent
    px = base.load()
    for x in range(10, 54):                                  # draw two dark solid lines
        px[x, 20] = (20, 20, 20, 255)
        px[x, 40] = (20, 20, 20, 255)
    if mode == "RGBA":
        return base
    if mode == "LA":
        return base.convert("LA")
    if mode == "P":
        # palette + transparency metadata — the easiest one to miss in PIL
        p = base.convert("P", palette=PILImage.Palette.ADAPTIVE)
        p.info["transparency"] = 0
        return p
    raise ValueError(mode)


class TestAlphaIsFlattenedOntoWhite:
    @pytest.mark.parametrize("mode", ["RGBA", "LA", "P"])
    def test_transparent_line_art_survives(self, mode):
        """Core regression: transparent-background line art must not become a black sheet."""
        out = flatten_on_white(_line_art_with_alpha(mode))
        assert out.mode == "RGB", out.mode
        luma = _mean_luma(out)
        assert luma > 200, (
            f"mode={mode} mean luma after flattening {luma:.1f} — transparent areas were squashed to black; "
            f"the VLM would see a black sheet, return unknown/low/all zeros, record ok=1 and never retry")

    @pytest.mark.parametrize("mode", ["RGBA", "LA", "P"])
    def test_the_lines_themselves_are_still_dark(self, mode):
        """White alone is not enough — the lines must still be there, or we've
        just washed the image into blank paper."""
        out = flatten_on_white(_line_art_with_alpha(mode))
        assert min(out.convert("L").getdata()) < 100, (
            f"mode={mode} line art was washed out, only white background left")

    def test_naive_convert_would_have_failed(self):
        """Pin the bug itself: the old approach (plain convert) is necessarily
        black on the same image. This shows the thresholds above aren't
        arbitrary — the two results differ by an order of magnitude."""
        bad = _mean_luma(_line_art_with_alpha("LA").convert("RGB"))
        good = _mean_luma(flatten_on_white(_line_art_with_alpha("LA")))
        assert bad < 50 and good > 200, f"naive={bad:.1f} fixed={good:.1f}"


class TestNonAlphaImagesUnchanged:
    def test_rgb_passes_through(self):
        src = PILImage.new("RGB", (32, 32), (255, 255, 255))
        out = flatten_on_white(src)
        assert out.mode == "RGB"
        assert _mean_luma(out) > 250

    def test_grayscale_without_alpha_keeps_its_content(self):
        """Mode L has no alpha; it must not be treated as transparent, content kept as-is."""
        src = PILImage.new("L", (32, 32), 128)
        out = flatten_on_white(src)
        assert out.mode == "RGB"
        assert 120 < _mean_luma(out) < 136, _mean_luma(out)

    def test_palette_without_transparency_keeps_its_content(self):
        """A plain P-mode image (no transparency metadata) must not be mistaken
        for a transparent one and washed white."""
        src = PILImage.new("RGB", (32, 32), (0, 0, 0)).convert(
            "P", palette=PILImage.Palette.ADAPTIVE)
        out = flatten_on_white(src)
        assert out.mode == "RGB"
        assert _mean_luma(out) < 50, f"pure black image was washed white: {_mean_luma(out)}"


class TestPipelineUsesIt:
    def test_fetch_and_resize_flattens(self, monkeypatch):
        """Guard the wiring: a helper that isn't wired into fetch_and_resize
        fixes nothing."""
        import base64
        import dispatcher as d

        buf = io.BytesIO()
        _line_art_with_alpha("LA").save(buf, format="PNG")
        payload = buf.getvalue()

        class _Resp:
            headers = {"content-type": "image/png"}

            def read(self):
                return payload

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        monkeypatch.setattr(d.urllib.request, "urlopen", lambda *a, **k: _Resp())
        data_uri = d.fetch_and_resize("https://example.com/plan.png", short_edge=64)
        jpeg = base64.b64decode(data_uri.split(",", 1)[1])
        assert _mean_luma(PILImage.open(io.BytesIO(jpeg))) > 200, (
            "fetch_and_resize still produces a black image — the helper is not wired into the pipeline")
