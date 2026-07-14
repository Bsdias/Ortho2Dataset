"""Utilities to convert georeferenced shapes into COCO-format annotations."""
import numpy as np


def geom_to_coco_polygons(geom, transform, window):
    """Converts a Shapely geometry into COCO polygons (pixel coords relative to `window`)."""
    if geom.geom_type == 'Polygon':
        polys = [geom]
    elif geom.geom_type == 'MultiPolygon':
        polys = list(geom.geoms)
    elif geom.geom_type == 'GeometryCollection':
        # Filters out residual lines or points left over from clipping
        polys = [g for g in geom.geoms if g.geom_type == 'Polygon']
    else:
        return []

    coco_polys = []
    for poly in polys:
        pixel_coords = []
        for coord in poly.exterior.coords:
            x, y = coord[:2]  # Ignore Z if present
            px, py = ~transform * (x, y)
            pixel_coords.extend([float(px - window.col_off), float(py - window.row_off)])
        coco_polys.append(pixel_coords)
    return coco_polys


def build_categories(names):
    """Builds the COCO `categories` list, assigning sequential ids starting at 1."""
    return [{"id": i + 1, "name": name} for i, name in enumerate(names)]


def build_annotation(geom, ann_id, img_id, cat_id, tile_bbox, transform, window, pixel_area_scale=1.0):
    """Clips `geom` to `tile_bbox` and builds a COCO annotation dict.

    Returns None if the clipped geometry is empty or too small to form a polygon
    (fewer than 3 points).
    """
    clipped = geom.intersection(tile_bbox)
    if clipped.is_empty:
        return None

    segmentation = geom_to_coco_polygons(clipped, transform, window)
    if not segmentation or len(segmentation[0]) < 6:
        return None

    points = np.array(segmentation[0]).reshape(-1, 2)
    xmin, ymin = points.min(axis=0)
    xmax, ymax = points.max(axis=0)

    return {
        "id": ann_id,
        "image_id": img_id,
        "category_id": cat_id,
        "segmentation": segmentation,
        "area": float(clipped.area * pixel_area_scale),
        "bbox": [float(xmin), float(ymin), float(xmax - xmin), float(ymax - ymin)],
        "iscrowd": 0
    }
