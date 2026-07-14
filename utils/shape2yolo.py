"""Utilities to build YOLO segmentation label lines from tile-pixel polygons."""


def build_category_id_map(categories):
    """Maps COCO-style category ids to sequential 0-indexed YOLO class ids, ordered by id."""
    ordered = sorted(categories, key=lambda c: c["id"])
    return {cat["id"]: i for i, cat in enumerate(ordered)}


def build_yolo_lines(polygons, yolo_class, img_width, img_height):
    """Builds YOLO segmentation lines from already-clipped, pixel-space polygons
    (`class_id x1 y1 x2 y2 ...`, coordinates normalized to [0, 1]).

    A geometry may hold several disjoint polygons (e.g. split by a tile
    boundary); each becomes its own line, sharing the same class id.
    """
    lines = []
    for polygon in polygons:
        points = []
        for i in range(0, len(polygon), 2):
            x = min(1.0, max(0.0, polygon[i] / img_width))
            y = min(1.0, max(0.0, polygon[i + 1] / img_height))
            points.append(f"{x:.6f}")
            points.append(f"{y:.6f}")
        lines.append(f"{yolo_class} " + " ".join(points))
    return lines


def write_label_file(path, lines):
    """Writes a YOLO `.txt` label file (empty file if `lines` is empty)."""
    with open(path, "w") as f:
        f.write("\n".join(lines))


def write_classes_file(path, names_by_yolo_id):
    """Writes a `classes.txt` listing class names in YOLO id order (one per line)."""
    with open(path, "w") as f:
        f.write("\n".join(names_by_yolo_id))
