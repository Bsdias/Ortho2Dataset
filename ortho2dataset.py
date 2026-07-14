#!/usr/bin/env python3
"""
Generates a COCO dataset from an orthomosaic and any number of georeferenced shapefiles.

Usage:
    # Discover layer names inside a multi-layer GeoPackage first, if needed:
    python ortho2dataset.py --list-layers /path/to/data.gpkg

    # Run from a config file (copy config.example.yaml -> config.yaml and edit it):
    python ortho2dataset.py --config config.yaml

    # Or pass everything as flags (any of these override the matching config.yaml value):
    python ortho2dataset.py \
        --tif /path/to/orthomosaic.tif \
        --shapes class1=shape/data.gpkg:layer1 \
        --out dataset/my_dataset/\
        --res 0.20
"""
import json
import argparse
from pathlib import Path

import rasterio
from PIL import Image
from tqdm import tqdm

from utils.ortho_tiling import resample_raster, iter_tile_windows, tile_bbox, read_tile_image
from utils.geometry import clip_to_tile, geom_to_pixel_polygons
from utils.shape2coco import build_categories, build_annotation
from utils.shapes import parse_shape_spec, list_layers, load_shape
from utils.shape2yolo import build_category_id_map, build_yolo_lines, write_label_file, write_classes_file
from utils.config import load_config, normalize_shapes, DEFAULTS

OUTPUT_FORMATS = ("coco", "yolo", "both")


class DatasetGenerator:
    def __init__(self, input_tif, shapes, output_dir, tile_size=1024, res=0.20, output_format="coco"):
        """
        shapes: dict mapping category name -> (path, layer) to a georeferenced
                shapefile/GeoPackage. `layer` may be None for single-layer sources.
        res: output resolution of the dataset in meters/pixel. Since tile_size is fixed
             in pixels, this sets how much ground area each tile covers.
        output_format: "coco", "yolo", or "both" — which annotation format(s) to write.
        """
        self.input_tif = Path(input_tif)
        self.output_dir = Path(output_dir)
        self.tile_size = tile_size
        self.target_res = res
        self.output_format = output_format

        self.images_dir = self.output_dir / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        if self.output_format in ("yolo", "both"):
            self.labels_dir = self.output_dir / "labels"
            self.labels_dir.mkdir(parents=True, exist_ok=True)

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
        self.cat_id_to_yolo = build_category_id_map(self.coco["categories"])

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

                if self.output_format in ("coco", "both"):
                    self.coco["images"].append({
                        "id": img_id,
                        "file_name": tile_filename,
                        "width": window.width,
                        "height": window.height
                    })

                yolo_lines = [] if self.output_format in ("yolo", "both") else None

                for name, gdf_in in matches.items():
                    cat_id = self.cat_ids[name]
                    yolo_class = self.cat_id_to_yolo[cat_id]
                    for _, row in gdf_in.iterrows():
                        clipped = clip_to_tile(row.geometry, bbox)
                        if clipped is None:
                            continue

                        polygons = geom_to_pixel_polygons(clipped, transform, window)
                        if not polygons:
                            continue

                        if self.output_format in ("coco", "both"):
                            ann = build_annotation(
                                polygons, clipped.area / (self.target_res ** 2),
                                self.ann_id_counter, img_id, cat_id
                            )
                            self.coco["annotations"].append(ann)
                            self.ann_id_counter += 1

                        if yolo_lines is not None:
                            yolo_lines.extend(
                                build_yolo_lines(polygons, yolo_class, window.width, window.height)
                            )

                if yolo_lines is not None:
                    label_filename = Path(tile_filename).stem + ".txt"
                    write_label_file(self.labels_dir / label_filename, yolo_lines)

                img_id += 1

        if self.output_format in ("yolo", "both"):
            names_by_yolo_id = [None] * len(self.cat_id_to_yolo)
            for cat in self.coco["categories"]:
                names_by_yolo_id[self.cat_id_to_yolo[cat["id"]]] = cat["name"]
            write_classes_file(self.output_dir / "classes.txt", names_by_yolo_id)
            print(f"YOLO labels written to {self.labels_dir}")

        if self.output_format in ("coco", "both"):
            with open(self.output_dir / "annotations.json", "w") as f:
                json.dump(self.coco, f, indent=4)
            print(f"COCO annotations written to {self.output_dir / 'annotations.json'}")

        print(f"Dataset successfully created at {self.output_dir}")


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
    parser.add_argument(
        "--config", metavar="PATH",
        help="Path to a YAML config file (see config.example.yaml) providing defaults for "
             "any of the other flags below. Flags passed on the command line take precedence."
    )
    parser.add_argument("--tif", help="Path to the input orthomosaic (GeoTIFF)")
    parser.add_argument(
        "--shapes", nargs="+",
        help="One or more category=path[:layer] entries, e.g. "
             "--shapes lot=data.gpkg:lots block=data.gpkg:blocks"
    )
    parser.add_argument("--out", help="Output directory for the dataset")
    parser.add_argument(
        "--res", type=float,
        help="Output resolution of the dataset, in meters/pixel. Since tile size is fixed "
             "in pixels, this controls how much ground area each tile covers "
             f"(tile_size * res meters per side). Default: {DEFAULTS['res']}."
    )
    parser.add_argument(
        "--tile-size", type=int,
        help=f"Tile size in pixels. Default: {DEFAULTS['tile_size']}."
    )
    parser.add_argument(
        "--format", choices=OUTPUT_FORMATS,
        help="Annotation format(s) to generate: 'coco' (annotations.json), "
             f"'yolo' (labels/*.txt + classes.txt), or 'both'. Default: {DEFAULTS['format']}."
    )
    parser.add_argument(
        "--list-layers", metavar="PATH",
        help="List the layer names available in PATH (e.g. a multi-layer GeoPackage) and exit, "
             "without processing anything."
    )
    args = parser.parse_args()

    config = load_config(args.config)

    tif = args.tif or config.get("tif")
    out = args.out or config.get("out")
    res = args.res if args.res is not None else config.get("res", DEFAULTS["res"])
    tile_size = args.tile_size if args.tile_size is not None else config.get("tile_size", DEFAULTS["tile_size"])
    output_format = args.format or config.get("format", DEFAULTS["format"])
    shapes = parse_shapes(args.shapes) if args.shapes else normalize_shapes(config.get("shapes"))

    if output_format not in OUTPUT_FORMATS:
        parser.error(f"format must be one of {OUTPUT_FORMATS}, got '{output_format}'")
    missing = [flag for flag, val in (("--tif", tif), ("--out", out)) if not val]
    if not shapes:
        missing.append("--shapes")
    if missing:
        parser.error(
            f"missing required settings: {', '.join(missing)} "
            "(pass as flags or set them in --config)"
        )

    gen = DatasetGenerator(tif, shapes, out, tile_size=tile_size, res=res, output_format=output_format)
    gen.run()
