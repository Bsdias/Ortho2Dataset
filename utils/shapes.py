"""Utilities to resolve and load shape sources, including multi-layer vector files
(e.g. a single GeoPackage holding several layers)."""
import argparse
from pathlib import Path

import geopandas as gpd


def parse_shape_spec(value):
    """Parses a `name=path[:layer]` CLI entry.

    The `:layer` suffix is only recognized if the part before the colon
    resolves to an existing file, so paths are not confused with a trailing
    layer name (e.g. `lot=data.gpkg:lots` -> path=data.gpkg, layer=lots).
    """
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            f"Invalid shape entry '{value}', expected format name=path[:layer]"
        )
    name, rest = value.split("=", 1)
    path, layer = rest, None
    if ":" in rest:
        candidate_path, candidate_layer = rest.rsplit(":", 1)
        if Path(candidate_path).exists():
            path, layer = candidate_path, candidate_layer
    return name, path, layer


def list_layers(path):
    """Lists the layer names available in a vector source (e.g. a GeoPackage)."""
    try:
        import pyogrio
        return [row[0] for row in pyogrio.list_layers(path)]
    except ImportError:
        import fiona
        return list(fiona.listlayers(path))


def load_shape(path, layer=None):
    """Reads a vector source as a GeoDataFrame, optionally selecting a specific layer.

    Uses `on_invalid="ignore"` because some source files contain malformed
    geometries (e.g. a LinearRing with only 2 points), which otherwise raise
    a GEOSException during reading; such geometries are read as null/empty
    instead of crashing, then dropped.
    """
    gdf = gpd.read_file(path, layer=layer, engine="pyogrio", on_invalid="ignore")
    gdf = gdf[gdf.geometry.notna()]
    return gdf[~gdf.geometry.is_empty]
