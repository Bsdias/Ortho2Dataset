"""Utilities for tiling a georeferenced orthomosaic raster into fixed-size chunks."""
import subprocess
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window, bounds as window_bounds
from shapely.geometry import box


def resample_raster(input_tif, output_dir, target_res, resampling="cubic"):
    """Resamples a raster to `target_res` (CRS units/pixel) using gdalwarp.

    `target_res=0` skips resampling entirely and returns `input_tif` unchanged,
    keeping the orthomosaic's native resolution.

    Reuses an existing output file if one is already present, since gdalwarp
    can be slow on large orthomosaics.
    """
    input_tif = Path(input_tif)
    if not target_res:
        return input_tif

    output_dir = Path(output_dir)
    out_tif = output_dir / f"{input_tif.stem}_{int(target_res * 100)}cm.tif"
    if not out_tif.exists():
        print(f"Resampling raster to {target_res}m/px...")
        cmd = [
            "gdalwarp", "-tr", str(target_res), str(target_res),
            "-r", resampling, "-co", "COMPRESS=LZW", "-co", "TILED=YES",
            str(input_tif), str(out_tif)
        ]
        subprocess.run(cmd, check=True)
    return out_tif


def iter_tile_windows(width, height, tile_size, min_coverage=0.8):
    """Yields rasterio Windows covering a raster in tile_size x tile_size chunks.

    Edge tiles covering less than `min_coverage` of tile_size on either axis
    are skipped rather than yielded as undersized tiles.
    """
    for row_off in range(0, height, tile_size):
        for col_off in range(0, width, tile_size):
            w = min(tile_size, width - col_off)
            h = min(tile_size, height - row_off)
            if w < tile_size * min_coverage or h < tile_size * min_coverage:
                continue
            yield Window(col_off, row_off, w, h)


def tile_bbox(window, transform):
    """Returns the shapely geometry of a tile's geographic bounding box."""
    return box(*window_bounds(window, transform))


def read_tile_image(src, window, bands=(1, 2, 3)):
    """Reads `window` from an open rasterio dataset as an HWC uint8 array."""
    data = src.read(list(bands), window=window)
    img = np.transpose(data, (1, 2, 0))
    if img.dtype != np.uint8:
        img = (img / img.max() * 255).astype(np.uint8)
    return img
