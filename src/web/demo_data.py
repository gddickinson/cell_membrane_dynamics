"""Synthetic demo recording for the web interface (NOT real microscopy data).

A deterministic cell outline, r(theta, t) = R0 + A sin(3 theta - w t) + a protrusion that grows over time, drawn as
binary masks on a 256 x 256 grid, plus a matching 8-bit "membrane label" image (the mask edge blurred). It exists
only so the analysis can be tried without a recording; numbers from it say nothing about PIEZO1 or real cells.
"""
import numpy as np
from scipy.ndimage import gaussian_filter

SIZE = 256
N_FRAMES = 12
R0, A, W = 62.0, 7.0, 0.35            # px, px, rad per frame
PROTRUSION_ANGLE, PROTRUSION_WIDTH, PROTRUSION_GROWTH = 0.8, 0.35, 1.6   # rad, rad, px per frame
PIXEL_SIZE_UM = 0.1                    # suggested pixel size for the demo (the desktop app's default, 100 nm)
FRAME_INTERVAL_S = 2.0                 # suggested frame interval for the demo


def radius(theta: np.ndarray, t: int) -> np.ndarray:
    """Outline radius (px) at angle theta (rad) in frame t."""
    d = np.angle(np.exp(1j * (theta - PROTRUSION_ANGLE)))
    return R0 + A * np.sin(3 * theta - W * t) + PROTRUSION_GROWTH * t * np.exp(-(d / PROTRUSION_WIDTH) ** 2)


def demo_masks() -> np.ndarray:
    """uint8 array (N_FRAMES, SIZE, SIZE), 255 inside the synthetic cell, 0 outside."""
    yy, xx = np.mgrid[:SIZE, :SIZE].astype(float)
    dx, dy = xx - SIZE / 2 + 0.5, yy - SIZE / 2 + 0.5
    r, th = np.hypot(dx, dy), np.arctan2(dy, dx)
    return np.stack([(r < radius(th, t)).astype(np.uint8) * 255 for t in range(N_FRAMES)])


def demo_fluorescence(masks: np.ndarray = None) -> np.ndarray:
    """uint8 synthetic membrane label: the mask boundary band, Gaussian-blurred (sigma 1.5 px), plus a dim interior."""
    masks = demo_masks() if masks is None else masks
    out = []
    for m in masks:
        inside = (m > 0).astype(float)
        edge = np.abs(np.gradient(inside)[0]) + np.abs(np.gradient(inside)[1])
        img = 20 + 40 * gaussian_filter(inside, 2.0) + 400 * gaussian_filter(edge, 1.5)
        out.append(np.clip(img, 0, 255).astype(np.uint8))
    return np.stack(out)


def rle_encode(mask: np.ndarray) -> list:
    """Row-major run lengths of a binary mask, starting with a run of zeros (may be 0)."""
    flat = (mask.ravel() > 0).astype(np.int8)
    change = np.flatnonzero(np.diff(flat)) + 1
    bounds = np.concatenate([[0], change, [flat.size]])
    runs = np.diff(bounds).tolist()
    return runs if flat[0] == 0 else [0] + runs


def rle_decode(runs: list, shape) -> np.ndarray:
    """Inverse of rle_encode (uint8 0/255)."""
    flat = np.zeros(int(np.prod(shape)), np.uint8)
    pos, val = 0, 0
    for n in runs:
        if val:
            flat[pos:pos + n] = 255
        pos += n
        val ^= 1
    return flat.reshape(shape)
