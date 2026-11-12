"""Run the repo's analysis (src/analysis/pipeline.py) for the web interface and shape the result as JSON.

The JSON uses the same keys and units as the browser engine (dynamics_engine.js analyseSequence), so the pages draw
both the same way: positions in px, curvature in nm^-1, velocities in px/s. The pages convert to µm, µm^-1 and µm/s
with the pixel size that is stored in `params` next to the numbers.
"""
import io
import warnings
from typing import List, Optional

import numpy as np

from ..analysis.curvature_analyzer import CurvatureAnalyzer
from ..analysis.dynamics_tracker import DynamicsTracker
from ..analysis.edge_detection import EdgeDetector
from ..analysis.pipeline import analyze_sequence

CLASSES = ["expanding", "retracting", "flowing", "stationary"]
MAX_FRAMES = 500
MAX_PIXELS = 2048 * 2048


class AnalysisError(ValueError):
    """Input the analysis cannot use (shown to the user as a 400)."""


def read_stack(data: bytes, what: str = "file"):
    """2D planes of a TIFF plus the pixel size (µm/px) and frame interval (s) the file declares (None if absent)."""
    import tifffile
    try:
        with tifffile.TiffFile(io.BytesIO(data)) as tif:
            series = tif.series[0]
            arr = series.asarray()
            axes = series.axes
            um, dt = None, None
            ij = tif.imagej_metadata or {}
            page = tif.pages[0]
            res = page.tags.get("XResolution")
            if ij:
                if ij.get("finterval"):
                    dt = float(ij["finterval"])
                unit = str(ij.get("unit", "")).replace("\\u00B5", "µ")
                if res and unit in ("um", "µm", "micron", "microns"):
                    num, den = res.value
                    um = den / num if num else None
            elif tif.is_ome:
                import re
                m = re.search(r'PhysicalSizeX="([0-9.eE+-]+)"', tif.ome_metadata or "")
                um = float(m.group(1)) if m else None
                m = re.search(r'TimeIncrement="([0-9.eE+-]+)"', tif.ome_metadata or "")
                dt = float(m.group(1)) if m else None
    except Exception as e:  # tifffile raises many types for bad input
        raise AnalysisError(f"Could not read the {what} as a TIFF: {e}") from e
    if axes.endswith("S"):
        raise AnalysisError(f"The {what} is an RGB image: save a single-channel (grey) stack.")
    if arr.ndim < 2:
        raise AnalysisError(f"The {what} has no 2D image.")
    planes = [arr] if arr.ndim == 2 else list(arr.reshape(-1, arr.shape[-2], arr.shape[-1]))
    if len(planes) > MAX_FRAMES or planes[0].size > MAX_PIXELS:
        raise AnalysisError(f"The {what} is too large for the web app (max {MAX_FRAMES} frames of 2048 x 2048).")
    return planes, um, dt


def _num(x):
    """float, or None for NaN / infinity (JSON has no NaN; a constant input gives an undefined correlation)."""
    x = float(x)
    return x if np.isfinite(x) else None


def _motion_stats(motion) -> dict:
    s = DynamicsTracker().get_motion_statistics(motion)
    return {"mean_speed": float(s["mean_speed"]), "max_speed": float(s["max_speed"]),
            "median_speed": float(s["median_speed"]), "std_speed": float(s["std_speed"]),
            "counts": {c: int(s["movement_classifications"][c]) for c in CLASSES},
            "percent": {c: float(s[f"{c}_percent"]) for c in CLASSES}}


def summarise(result: dict) -> dict:
    """The result without the per-point arrays: per-frame metrics, statistics and r / p (for comparisons)."""
    frames = []
    for fr in result["frames"]:
        frames.append({"frame": fr["frame"], "n_points": len(fr["contour"]), "metrics": fr["metrics"],
                       "curvature_mean_all": float(np.mean(fr["curvature"])),
                       "curvature_stats": fr["curvature_stats"],
                       "motion": fr["motion"] and {"stats": fr["motion"]["stats"], "n": len(fr["motion"]["classes"])},
                       "correlation": fr["correlation"] and {k: fr["correlation"][k] for k in ("r", "p")}})
    return {**{k: v for k, v in result.items() if k != "frames"}, "frames": frames}


def analyse(masks: List[np.ndarray], frames: Optional[List[np.ndarray]], um_per_px: float, frame_s: float,
            sigma: float = 1.0, start: int = 0, end: Optional[int] = None) -> dict:
    """Run pipeline.analyze_sequence. Without fluorescence frames the optical-flow step sees blank images and finds
    no flow, so velocities come from contour tracking alone (what the browser engine computes)."""
    if not masks:
        raise AnalysisError("No mask frames.")
    if not (um_per_px > 0 and frame_s > 0):
        raise AnalysisError("Pixel size and frame interval must be positive.")
    flow = frames is not None
    if flow:
        if len(frames) != len(masks) or frames[0].shape != masks[0].shape:
            raise AnalysisError(f"The PIEZO1 stack ({len(frames)} x {frames[0].shape}) does not match the masks "
                                f"({len(masks)} x {masks[0].shape}).")
    else:
        frames = [np.zeros(masks[0].shape, np.uint8)] * len(masks)
    end = len(masks) - 1 if end is None else min(int(end), len(masks) - 1)
    start = max(0, int(start))
    if start > end:
        raise AnalysisError("The first frame is after the last frame.")
    pixel_nm = um_per_px * 1000.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        res = analyze_sequence(list(masks), list(frames), pixel_size=pixel_nm, time_delta=frame_s, smoothing=sigma,
                               start=start, end=end, edge_detector=EdgeDetector(smoothing_sigma=sigma))
    corr = {c.frame_index: c for c in res["correlation_data"]}
    curv_an = CurvatureAnalyzer(pixel_size=pixel_nm)
    out = []
    for fid, contour, k, motion in zip(res["frame_ids"], res["contours"], res["curvatures"], res["motion_data"]):
        met = EdgeDetector().analyze_contour(contour)
        fr = {"frame": fid, "contour": contour.tolist(), "curvature": k.tolist(),
              "metrics": {key: (list(v) if isinstance(v, tuple) else float(v)) for key, v in met.items()},
              "curvature_stats": {key: float(v) for key, v in curv_an.get_curvature_statistics(k).items()},
              "motion": None, "correlation": None}
        if motion is not None:
            fr["motion"] = {"velocities": motion.velocities.astype(float).tolist(),
                            "magnitudes": motion.magnitudes.astype(float).tolist(),
                            "classes": list(motion.classifications), "stats": _motion_stats(motion)}
        c = corr.get(fid)
        if c is not None:
            fr["correlation"] = {"curvatures": c.curvatures.tolist(), "normal": c.normal_velocities.tolist(),
                                 "tangential": c.tangential_velocities.tolist(), "classes": list(c.classifications),
                                 "r": _num(c.correlation), "p": _num(c.p_value)}
        out.append(fr)
    return {"engine": "python", "optical_flow": flow, "failed": res["failed"], "frames": out,
            "params": {"um_per_px": um_per_px, "pixel_size_nm": pixel_nm, "frame_s": frame_s, "sigma": sigma,
                       "start": start, "end": end, "n_frames": len(masks), "shape": list(masks[0].shape)}}
