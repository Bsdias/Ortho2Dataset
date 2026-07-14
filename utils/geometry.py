"""Generic helpers to clip a geometry to a tile and project it into the tile's pixel space.

This is deliberately format-agnostic: COCO and YOLO builders both consume the
same pixel-space polygons produced here, instead of one format being derived
from the other.
"""


def clip_to_tile(geom, tile_bbox):
    """Clips `geom` to `tile_bbox`. Returns None if the result is empty."""
    clipped = geom.intersection(tile_bbox)
    return None if clipped.is_empty else clipped


def geom_to_pixel_polygons(geom, transform, window, min_points=3):
    """Converts a Shapely geometry into polygon point lists (flat [x0, y0, x1, y1, ...],
    pixel coords relative to `window`), dropping any part with fewer than `min_points` vertices.

    A geometry may expand into several disjoint polygons (e.g. a MultiPolygon,
    or a Polygon split into parts after clipping to a tile boundary).
    """
    if geom.geom_type == 'Polygon':
        polys = [geom]
    elif geom.geom_type == 'MultiPolygon':
        polys = list(geom.geoms)
    elif geom.geom_type == 'GeometryCollection':
        # Filters out residual lines or points left over from clipping
        polys = [g for g in geom.geoms if g.geom_type == 'Polygon']
    else:
        return []

    pixel_polys = []
    for poly in polys:
        pixel_coords = []
        for coord in poly.exterior.coords:
            x, y = coord[:2]  # Ignore Z if present
            px, py = ~transform * (x, y)
            pixel_coords.extend([float(px - window.col_off), float(py - window.row_off)])
        if len(pixel_coords) >= min_points * 2:
            pixel_polys.append(pixel_coords)
    return pixel_polys
