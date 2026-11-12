"""Write small TIFF test files to tests/fixtures/tiff/ for the browser reader test (src/web/static/tiff_io.test.mjs).

Each file holds the same 3-frame 16 x 20 ramp (value = 1000 t + 20 y + x) in a different layout; meta.json lists the
pixel size / frame interval each file declares. Run: python scripts/make_tiff_fixtures.py
"""
import json
from pathlib import Path

import numpy as np
import tifffile

OUT = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "tiff"


def ramp(dtype):
    t, y, x = np.mgrid[:3, :16, :20]
    return (1000 * t + 20 * y + x).astype(dtype) if np.dtype(dtype).itemsize > 1 else ((20 * y + x + 7 * t) % 256).astype(dtype)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meta = {}
    tifffile.imwrite(OUT / "shaped_u16.tif", ramp(np.uint16), photometric="minisblack")
    meta["shaped_u16.tif"] = {"dtype": "uint16", "umPerPx": None, "frameS": None}
    tifffile.imwrite(OUT / "planar_u16.tif", ramp(np.uint16))   # tifffile's default guess: 3 planes = planar RGB
    meta["planar_u16.tif"] = {"dtype": "uint16", "umPerPx": None, "frameS": None}
    tifffile.imwrite(OUT / "imagej_u16.tif", ramp(np.uint16), imagej=True, resolution=(1 / 0.16, 1 / 0.16),
                     metadata={"unit": "um", "finterval": 2.5, "axes": "TYX"})
    meta["imagej_u16.tif"] = {"dtype": "uint16", "umPerPx": 0.16, "frameS": 2.5}
    tifffile.imwrite(OUT / "ome_u8.ome.tif", ramp(np.uint8), metadata={"axes": "TYX", "PhysicalSizeX": 0.108,
                                                                          "PhysicalSizeY": 0.108, "TimeIncrement": 0.5})
    meta["ome_u8.ome.tif"] = {"dtype": "uint8", "umPerPx": 0.108, "frameS": 0.5}
    tifffile.imwrite(OUT / "bigendian_f32.tif", ramp(np.float32), byteorder=">", photometric="minisblack")
    meta["bigendian_f32.tif"] = {"dtype": "float32", "umPerPx": None, "frameS": None}
    tifffile.imwrite(OUT / "zlib_u16.tif", ramp(np.uint16), compression="zlib", photometric="minisblack")
    meta["zlib_u16.tif"] = {"error": "compressed"}
    (OUT / "meta.json").write_text(json.dumps(meta, indent=1))
    print("wrote", ", ".join(sorted(meta)))


if __name__ == "__main__":
    main()
