"""Contract for parsing VLM detection output (offline, never calls the VLM)."""
import base64, io, sys
from pathlib import Path
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from lib import compass_detect as cd  # noqa: E402


def test_parse_fenced_json():
    raw = '```json\n{"compass_present": true, "bbox_2d": [899, 32, 969, 75]}\n```'
    assert cd.parse_detect(raw) == (True, [899, 32, 969, 75])


def test_parse_bare_json_absent():
    assert cd.parse_detect('{"compass_present": false, "bbox_2d": null}') == (False, None)


def test_parse_invalid_bbox_is_dropped_but_presence_kept():
    assert cd.parse_detect('{"compass_present": true, "bbox_2d": [900, 32, 899, 75]}') == (True, None)
    assert cd.parse_detect('{"compass_present": true, "bbox_2d": [0, 0, 1200, 10]}') == (True, None)
    assert cd.parse_detect('{"compass_present": true, "bbox_2d": [1, 2, 3]}') == (True, None)


def test_parse_garbage_is_unknown():
    assert cd.parse_detect("I cannot see the image") == (None, None)
    assert cd.parse_detect("") == (None, None)


def test_image_to_data_uri_caps_long_edge_and_is_jpeg():
    im = Image.new("RGB", (4000, 2000), (255, 255, 255))
    uri = cd.image_to_data_uri(im, edge=1288)
    assert uri.startswith("data:image/jpeg;base64,")
    out = Image.open(io.BytesIO(base64.b64decode(uri.split(",", 1)[1])))
    assert max(out.size) == 1288
