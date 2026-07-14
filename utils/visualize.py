"""Draws dataset annotations on top of tile images for quick visual QA,
so correctness can be eyeballed without opening a labeling tool."""
from PIL import Image, ImageDraw

# Small fixed palette so each category gets a consistent, distinguishable color
# across preview images.
_PALETTE = [
    (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200),
    (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230),
]

# Out of 255. Fills are drawn on a shared transparent layer before being
# composited onto the image, so overlapping annotations visibly stack into a
# darker/blended shade instead of just one color hiding the other.
_FILL_ALPHA = 90


def _color_for(index):
    return _PALETTE[index % len(_PALETTE)]


def _render(image_path, polygons, out_path):
    """Renders `polygons` (a list of `(points, color, label)`) on top of the
    image at `image_path` and saves the result to `out_path`.

    Each polygon gets a semi-transparent fill (so overlapping annotations —
    e.g. one shape entirely inside another — are visible as a shade change)
    plus a crisp, fully opaque outline and label, drawn last so boundaries
    stay legible no matter how much fill overlap there is underneath.
    """
    base = Image.open(image_path).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    for points, color, _ in polygons:
        # Each fill is composited onto the shared overlay individually — drawing
        # them directly on one overlay would just overwrite pixels instead of
        # blending, silently hiding overlapping annotations.
        layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
        ImageDraw.Draw(layer).polygon(points, fill=(*color, _FILL_ALPHA))
        overlay = Image.alpha_composite(overlay, layer)

    composited = Image.alpha_composite(base, overlay).convert("RGB")
    outline_draw = ImageDraw.Draw(composited)
    for points, color, label in polygons:
        outline_draw.polygon(points, outline=color, width=3)
        if label:
            outline_draw.text((points[0][0], max(0, points[0][1] - 12)), label, fill=color)

    composited.save(out_path, quality=90)


def draw_coco_preview(image_path, annotations, names_by_cat_id, out_path):
    """Draws COCO segmentation polygons for one tile and saves the result."""
    color_index = {cat_id: i for i, cat_id in enumerate(sorted(names_by_cat_id))}

    polygons = []
    for ann in annotations:
        color = _color_for(color_index[ann["category_id"]])
        label = names_by_cat_id[ann["category_id"]]
        for polygon in ann.get("segmentation", []):
            points = list(zip(polygon[0::2], polygon[1::2]))
            if len(points) >= 3:
                polygons.append((points, color, label))

    _render(image_path, polygons, out_path)


def draw_yolo_preview(image_path, label_path, names_by_yolo_id, out_path):
    """Draws YOLO segmentation polygons (read back from the .txt label file,
    denormalized to pixel space) for one tile and saves the result."""
    w, h = Image.open(image_path).size

    polygons = []
    if label_path.exists():
        for line in label_path.read_text().splitlines():
            if not line.strip():
                continue
            parts = line.split()
            cls_id = int(parts[0])
            coords = [float(v) for v in parts[1:]]
            points = [(coords[i] * w, coords[i + 1] * h) for i in range(0, len(coords), 2)]
            if len(points) >= 3:
                color = _color_for(cls_id)
                label = names_by_yolo_id[cls_id] if cls_id < len(names_by_yolo_id) else str(cls_id)
                polygons.append((points, color, label))

    _render(image_path, polygons, out_path)
