"""Web interface tests: every exported endpoint answers, the static build is complete, and the JS engine tests pass.

Run: python -m pytest -q tests/test_web.py
"""
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.web.app import STATIC, STATIC_ENDPOINTS, app  # noqa: E402

client = TestClient(app)


def test_index_served():
    r = client.get("/")
    assert r.status_code == 200 and "static/main.js" in r.text


@pytest.mark.parametrize("path", STATIC_ENDPOINTS)
def test_endpoint(path):
    r = client.get("/api" + path)
    assert r.status_code == 200, r.text
    assert r.json() is not None


def test_static_build(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    import build_static
    build_static.build(tmp_path / "site")
    assert (tmp_path / "site" / "index.html").read_text().count("APP_STATIC") == 1
    for path in STATIC_ENDPOINTS:
        assert (tmp_path / "site" / "api" / (path.strip("/") + ".json")).exists()


def test_static_assets_referenced_exist():
    html = (STATIC / "index.html").read_text()
    for name in ("style.css", "main.js"):
        assert f"static/{name}" in html and (STATIC / name).exists()


@pytest.mark.skipif(not shutil.which("node"), reason="node not installed")
def test_js_engine():
    tests = sorted(STATIC.glob("*.test.mjs"))
    for t in tests:
        r = subprocess.run(["node", str(t)], capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr


# --- project-specific -------------------------------------------------------------------------------------------
import io  # noqa: E402
import json  # noqa: E402

import numpy as np  # noqa: E402
import tifffile  # noqa: E402

from src.web import demo_data  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures"


def _close(a, b, path=""):
    """Recursive comparison with a tolerance (libm differs between macOS and Linux in the last digit)."""
    if isinstance(b, dict):
        assert set(a) == set(b), path
        for k in b:
            _close(a[k], b[k], f"{path}.{k}")
    elif isinstance(b, list):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            _close(x, y, f"{path}[{i}]")
    elif isinstance(b, float):
        assert a == pytest.approx(b, rel=1e-6, abs=1e-9), path
    else:
        assert a == b, path


def test_engine_fixture_is_current():
    sys.path.insert(0, str(ROOT / "scripts"))
    import make_engine_fixture
    _close(make_engine_fixture.build(), json.loads((FIXTURES / "engine_cases.json").read_text()))


def test_demo_rle_round_trip():
    masks = demo_data.demo_masks()
    assert masks.shape == (demo_data.N_FRAMES, demo_data.SIZE, demo_data.SIZE)
    for m in masks:
        assert np.array_equal(demo_data.rle_decode(demo_data.rle_encode(m), m.shape), m)


def test_demo_python_matches_full_analysis():
    r = client.get("/api/demo/python").json()
    assert r["engine"] == "python" and r["optical_flow"] is False
    assert len(r["frames"]) == demo_data.N_FRAMES and r["failed"] == []
    assert r["frames"][0]["motion"] is None and r["frames"][1]["correlation"]["r"] is not None


def _tif(arr, **kw):
    buf = io.BytesIO()
    tifffile.imwrite(buf, arr, **kw)
    return buf.getvalue()


def test_analyse_upload_reads_scale_and_runs():
    masks = demo_data.demo_masks()[:4]
    files = {"masks": ("cell.tif", _tif(masks, imagej=True, resolution=(10, 10), metadata={"unit": "um"}), "image/tiff")}
    r = client.post("/api/analyse", data={"source": "upload", "um_per_px": "0.1", "frame_s": "2"}, files=files)
    assert r.status_code == 200, r.text
    j = r.json()
    assert len(j["frames"]) == 4 and j["params"]["pixel_size_nm"] == pytest.approx(100)
    assert j["frames"][1]["motion"]["classes"]


def test_analyse_demo_with_optical_flow():
    r = client.post("/api/analyse", data={"source": "demo", "use_demo_fluorescence": "true", "frame_s": "2", "end": "3"})
    assert r.status_code == 200, r.text
    assert r.json()["optical_flow"] is True and len(r.json()["frames"]) == 4


@pytest.mark.parametrize("payload,msg", [
    ({"source": "upload"}, "Choose a mask"),
    ({"source": "demo", "um_per_px": "0"}, "positive"),
    ({"source": "demo", "start": "5", "end": "2"}, "after the last"),
])
def test_analyse_rejects_bad_input(payload, msg):
    r = client.post("/api/analyse", data=payload)
    assert r.status_code == 400 and msg in r.text


def test_analyse_rejects_non_tiff():
    r = client.post("/api/analyse", data={"source": "upload"}, files={"masks": ("x.tif", b"not a tiff", "image/tiff")})
    assert r.status_code == 400 and "TIFF" in r.text


@pytest.mark.skipif(not shutil.which("node"), reason="node not installed")
def test_js_tiff_writer_is_read_by_tifffile(tmp_path):
    out = tmp_path / "js.tif"
    r = subprocess.run(["node", str(STATIC / "tiff_io.test.mjs"), str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    with tifffile.TiffFile(out) as tif:
        arr = tif.asarray()
        assert arr.shape == (2, 10, 12) and arr.dtype == np.uint8
        assert tif.imagej_metadata["unit"] == "micron" and tif.imagej_metadata["finterval"] == 2
        num, den = tif.pages[0].tags["XResolution"].value
        assert den / num == pytest.approx(0.1)
