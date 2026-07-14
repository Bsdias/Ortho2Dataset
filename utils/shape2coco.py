"""Utilities to build COCO-format annotations from tile-pixel polygons."""
import numpy as np


def build_categories(names):
    """Builds the COCO `categories` list, assigning sequential ids starting at 1."""
    return [{"id": i + 1, "name": name} for i, name in enumerate(names)]


def build_annotation(polygons, area, ann_id, img_id, cat_id):
    """Builds a COCO annotation dict from already-clipped, pixel-space polygons.

    `area` is the annotation's real-world area already converted to pixel units
    by the caller (e.g. CRS-units area / pixel_size**2).
    """
    if not polygons:
        return None

    all_points = np.concatenate([np.array(p).reshape(-1, 2) for p in polygons])
    xmin, ymin = all_points.min(axis=0)
    xmax, ymax = all_points.max(axis=0)

    return {
        "id": ann_id,
        "image_id": img_id,
        "category_id": cat_id,
        "segmentation": polygons,
        "area": float(area),
        "bbox": [float(xmin), float(ymin), float(xmax - xmin), float(ymax - ymin)],
        "iscrowd": 0
    }
