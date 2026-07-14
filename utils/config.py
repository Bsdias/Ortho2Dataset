"""Loads pipeline settings from a YAML config file, so common runs don't need
a long list of CLI flags every time."""
import yaml

DEFAULTS = {
    "res": 0.20,
    "tile_size": 1024,
    "format": "coco",
}


def load_config(path):
    """Reads a YAML config file into a dict. Returns {} if `path` is None."""
    if path is None:
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def normalize_shapes(shapes_config):
    """Converts the YAML `shapes` mapping into `{name: (path, layer)}`, the same
    shape `parse_shapes` produces for `--shapes` CLI entries.

    Accepts either a plain path string (single-layer source) or a
    `{path, layer}` mapping per category:

        shapes:
          lot: lots.gpkg
          block:
            path: data.gpkg
            layer: blocks
    """
    shapes = {}
    for name, spec in (shapes_config or {}).items():
        if isinstance(spec, str):
            shapes[name] = (spec, None)
        else:
            shapes[name] = (spec["path"], spec.get("layer"))
    return shapes
