"""Loads pipeline settings from a YAML config file, so common runs don't need
a long list of CLI flags every time."""
import yaml

DEFAULTS = {
    "res": 0,
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


def parse_preview_config(preview_config):
    """Normalizes the `preview` config value into `{"ratio": float, "seed": int}`.

    Accepts either a bare fraction (e.g. `preview: 0.05`) or a
    `{ratio, seed}` mapping. Returns None if `preview_config` is falsy
    (no preview requested).
    """
    if not preview_config:
        return None
    if isinstance(preview_config, dict):
        ratio = float(preview_config.get("ratio", 0))
        seed = preview_config.get("seed", 42)
    else:
        ratio = float(preview_config)
        seed = 42

    if not (0 < ratio <= 1):
        raise ValueError(f"preview ratio must be between 0 and 1, got {ratio}")

    return {"ratio": ratio, "seed": seed}
