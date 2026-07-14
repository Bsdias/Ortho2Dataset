#!/usr/bin/env python3
"""
Generates a COCO dataset from an orthomosaic and any number of georeferenced shapefiles.

Usage:
    # Discover layer names inside a multi-layer GeoPackage first, if needed:
    python ortho2dataset.py --list-layers /path/to/data.gpkg

    python ortho2dataset.py \
        --tif /media/bruno/HDD/DEV/PedroAfonso_Orto_and_Shapes/pedro_afonso__prepared.tif \
        --shapes lot=/media/bruno/HDD/DEV/PedroAfonso_Orto_and_Shapes/data.gpkg:lots \
                 block=/media/bruno/HDD/DEV/PedroAfonso_Orto_and_Shapes/data.gpkg:blocks \
        --out /media/bruno/HDD/DEV/PedroAfonso_Orto_and_Shapes/dataset_qd_lt_pedroafonso \
        --res 0.20
"""
import json
import argparse
from pathlib import Path

import rasterio
from PIL import Image
from tqdm import tqdm

from utils.ortho_tiling import resample_raster, iter_tile_windows, tile_bbox, read_tile_image
from utils.shape2coco import build_categories, build_annotation
from utils.shapes import parse_shape_spec, list_layers, load_shape


class CocoDatasetGenerator:
    def __init__(self, input_tif, shapes, output_dir, tile_size=1024, res=0.20):
        """
        shapes: dict mapping category name -> (path, layer) to a georeferenced
                shapefile/GeoPackage. `layer` may be None for single-layer sources.
        res: output resolution of the dataset in meters/pixel. Since tile_size is fixed
             in pixels, this sets how much ground area each tile covers.
        """
        self.input_tif = Path(input_tif)
        self.output_dir = Path(output_dir)
        self.tile_size = tile_size
        self.target_res = res

        self.images_dir = self.output_dir / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)

        print("Reading shapefiles...")
        self.category_names = list(shapes.keys())
        self.gdfs = {name: load_shape(path, layer) for name, (path, layer) in shapes.items()}
        self.cat_ids = {name: i + 1 for i, name in enumerate(self.category_names)}

        self.coco = {
            "images": [],
            "annotations": [],
            "categories": build_categories(self.category_names)
        }
        self.ann_id_counter = 1

    def run(self):
        tif_path = resample_raster(self.input_tif, self.output_dir, self.target_res)

        with rasterio.open(tif_path) as src:
            transform = src.transform
            width, height, crs = src.width, src.height, src.crs

            # Ensure every shape is in the same CRS as the raster
            for name in self.category_names:
                self.gdfs[name] = self.gdfs[name].to_crs(crs)

            windows = list(iter_tile_windows(width, height, self.tile_size))
            print(f"Starting tile extraction ({len(windows)} candidate tiles)...")

            img_id = 1
            for window in tqdm(windows):
                bbox = tile_bbox(window, transform)

                matches = {
                    name: gdf[gdf.intersects(bbox)]
                    for name, gdf in self.gdfs.items()
                }
                if all(gdf_in.empty for gdf_in in matches.values()):
                    continue  # Skip tiles with no annotations of interest

                tile_filename = f"tile_{img_id:05d}.jpg"
                img_array = read_tile_image(src, window)
                Image.fromarray(img_array).save(self.images_dir / tile_filename, quality=95)

                self.coco["images"].append({
                    "id": img_id,
                    "file_name": tile_filename,
                    "width": window.width,
                    "height": window.height
                })

                for name, gdf_in in matches.items():
                    cat_id = self.cat_ids[name]
                    for _, row in gdf_in.iterrows():
                        ann = build_annotation(
                            row.geometry, self.ann_id_counter, img_id, cat_id,
                            bbox, transform, window,
                            pixel_area_scale=1.0 / (self.target_res ** 2)
                        )
                        if ann is not None:
                            self.coco["annotations"].append(ann)
                            self.ann_id_counter += 1

                img_id += 1

        with open(self.output_dir / "annotations.json", "w") as f:
            json.dump(self.coco, f, indent=4)
        print(f"COCO dataset successfully created at {self.output_dir}")


def parse_shapes(values):
    """Parses `name=path[:layer]` CLI entries into an ordered dict of category -> (path, layer)."""
    shapes = {}
    for value in values:
        name, path, layer = parse_shape_spec(value)
        shapes[name] = (path, layer)
    return shapes


if __name__ == "__main__":
    # Handled separately so `--list-layers` can be used on its own, without the
    # other arguments (--tif, --shapes, --out) being required.
    list_layers_parser = argparse.ArgumentParser(add_help=False)
    list_layers_parser.add_argument("--list-layers")
    list_layers_args, _ = list_layers_parser.parse_known_args()
    if list_layers_args.list_layers:
        for layer_name in list_layers(list_layers_args.list_layers):
            print(layer_name)
        raise SystemExit(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("--tif", required=True, help="Path to the input orthomosaic (GeoTIFF)")
    parser.add_argument(
        "--shapes", required=True, nargs="+",
        help="One or more category=path[:layer] entries, e.g. "
             "--shapes lot=data.gpkg:lots block=data.gpkg:blocks"
    )
    parser.add_argument("--out", required=True, help="Output directory for the dataset")
    parser.add_argument(
        "--res", type=float, default=0.20,
        help="Output resolution of the dataset, in meters/pixel. Since tile size is fixed "
             "in pixels, this controls how much ground area each tile covers "
             "(tile_size * res meters per side)."
    )
    parser.add_argument("--tile-size", type=int, default=1024, help="Tile size in pixels")
    parser.add_argument(
        "--list-layers", metavar="PATH",
        help="List the layer names available in PATH (e.g. a multi-layer GeoPackage) and exit, "
             "without processing anything."
    )
    args = parser.parse_args()

    shapes = parse_shapes(args.shapes)
    gen = CocoDatasetGenerator(args.tif, shapes, args.out, tile_size=args.tile_size, res=args.res)
    gen.run()
