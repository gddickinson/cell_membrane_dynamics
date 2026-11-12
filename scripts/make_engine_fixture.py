"""Write tests/fixtures/engine_cases.json: inputs and the repo's Python results that the JS engines must reproduce.

Cases: seeded random blob masks and the synthetic demo stack (src/web/demo_data.py) through EdgeDetector, the
finite-difference curvature, contour metrics and normals; the demo stack through pipeline.analyze_sequence with blank
fluorescence frames (no optical flow, the part the browser runs); scipy.stats.pearsonr on random data.
Run: python scripts/make_engine_fixture.py   (tests/test_web.py checks the committed fixture is current)
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore", category=FutureWarning)

from src.analysis.edge_detection import EdgeDetector  # noqa: E402
from src.analysis.pipeline import analyze_sequence, contour_normals, finite_difference_curvature  # noqa: E402
from src.analysis.curvature_analyzer import CurvatureAnalyzer  # noqa: E402
from src.web.demo_data import demo_masks, rle_encode  # noqa: E402

OUT = ROOT / "tests" / "fixtures" / "engine_cases.json"
PIXEL_NM = 100.0


def blob_mask(seed, size=150):
    """Union of 1-4 random ellipses, some touching the border margin or each other (diagonal contacts)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size]
    m = np.zeros((size, size), bool)
    for _ in range(rng.integers(1, 5)):
        cx, cy = rng.uniform(20, size - 20, 2)
        a, b = rng.uniform(4, 40, 2)
        t = rng.uniform(0, np.pi)
        u = (xx - cx) * np.cos(t) + (yy - cy) * np.sin(t)
        v = -(xx - cx) * np.sin(t) + (yy - cy) * np.cos(t)
        m |= (u / a) ** 2 + (v / b) ** 2 < 1
    m &= rng.random((size, size)) > 0.02       # pepper holes and ragged edges
    return m.astype(np.uint8) * 255


def edge_case(mask):
    h, w = mask.shape
    raw = EdgeDetector(smoothing_sigma=0).detect_edge(mask)
    out = {"w": w, "h": h, "rle": rle_encode(mask), "raw": None if raw is None else raw.tolist()}
    for sigma in (1.0, 2.5):
        c = EdgeDetector(smoothing_sigma=sigma).detect_edge(mask)
        if c is None:
            out[f"s{sigma}"] = None
            continue
        k = finite_difference_curvature(c, PIXEL_NM)
        met = EdgeDetector().analyze_contour(c)
        out[f"s{sigma}"] = {"contour": c.tolist(), "curvature": k.tolist(), "normals": contour_normals(c).tolist(),
                            "metrics": {k2: (list(v) if isinstance(v, tuple) else float(v)) for k2, v in met.items()},
                            "stats": {k2: float(v) for k2, v in CurvatureAnalyzer().get_curvature_statistics(k).items()}}
    return out


def sequence_case(dt=2.0):
    masks = demo_masks()
    res = analyze_sequence(list(masks), list(np.zeros_like(masks)), pixel_size=PIXEL_NM, time_delta=dt)
    frames = []
    corr = {c.frame_index: c for c in res["correlation_data"]}
    for fid, c, m in zip(res["frame_ids"], res["contours"], res["motion_data"]):
        fr = {"frame": fid, "n": len(c)}
        if m is not None:
            fr.update(velocities=m.velocities.astype(float).tolist(), classes=list(m.classifications),
                      r=float(corr[fid].correlation), p=float(corr[fid].p_value))
        frames.append(fr)
    kymo = res["tracker"].create_kymograph(5, 10)[:, :, :3]
    return {"dt": dt, "pixel_nm": PIXEL_NM, "shape": list(masks.shape[1:]),
            "rle": [rle_encode(m) for m in masks], "frames": frames, "kymograph_p5_w10": kymo.tolist()}


def pearson_cases():
    rng = np.random.default_rng(7)
    cases = []
    for n, rho in [(5, 0.3), (30, 0.0), (120, 0.5), (400, -0.2), (1000, 0.9)]:
        x = rng.normal(size=n)
        y = rho * x + rng.normal(size=n)
        r, p = stats.pearsonr(x, y)
        cases.append({"x": x.tolist(), "y": y.tolist(), "r": float(r), "p": float(p)})
    return cases


def build():
    demo = demo_masks()
    return {"pixel_nm": PIXEL_NM,
            "edges": [edge_case(blob_mask(s)) for s in range(12)] + [edge_case(demo[i]) for i in (0, 11)],
            "sequence": sequence_case(), "pearson": pearson_cases()}


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(build()))
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.2f} MB)")
