#!/usr/bin/env python3
"""
Generates a COCO and/or YOLO dataset from an orthomosaic and any number of
georeferenced shapefiles, optionally split into train/val/test and/or with a
visual QA preview sample rendered (see the `split`/`preview` sections in
config.example.yaml; only configurable via --config).

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
import shutil
import random
import argparse
from pathlib import Path
from collections import defaultdict

import yaml
import rasterio
from PIL import Image
from tqdm import tqdm

from utils.ortho_tiling import resample_raster, iter_tile_windows, tile_bbox, read_tile_image
from utils.geometry import clip_to_tile, geom_to_pixel_polygons
from utils.shape2coco import build_categories, build_annotation
from utils.shapes import parse_shape_spec, list_layers, load_shape
from utils.shape2yolo import build_category_id_map, build_yolo_lines, write_label_file, write_classes_file
from utils.config import load_config, normalize_shapes, parse_preview_config, DEFAULTS
from utils.split import parse_split_config, assign_splits
from utils.visualize import draw_coco_preview, draw_yolo_preview

OUTPUT_FORMATS = ("coco", "yolo", "both")


class DatasetGenerator:
    def __init__(
        self, input_tif, shapes, output_dir, tile_size=1024, res=0.20, output_format="coco",
        split=None, preview=None
    ):
        """
        shapes: dict mapping category name -> (path, layer) to a georeferenced
                shapefile/GeoPackage. `layer` may be None for single-layer sources.
        res: output resolution of the dataset in meters/pixel. Since tile_size is fixed
             in pixels, this sets how much ground area each tile covers.
        output_format: "coco", "yolo", or "both" — which annotation format(s) to write.
        split: parsed split config (see utils.split.parse_split_config), or None
               to keep a single flat dataset with no train/val/test partitioning.
        preview: parsed preview config (see utils.config.parse_preview_config), or
                 None to skip generating a visual QA sample.
        """
        self.input_tif = Path(input_tif)
        self.output_dir = Path(output_dir)
        self.tile_size = tile_size
        self.target_res = res
        self.output_format = output_format
        self.split = split
        self.preview = preview
        self.image_split = {}

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
            # Derived from the raster actually opened (native or resampled), so
            # area-in-pixels is correct even with res=0 (native resolution).
            pixel_area = abs(transform.a * transform.e)

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

                # Tracked regardless of output_format: needed to assign tiles to
                # a train/val/test split and to build data.yaml/classes.txt.
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
                                polygons, clipped.area / pixel_area,
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

        if self.split:
            self.image_split = assign_splits([img["id"] for img in self.coco["images"]], self.split)
            self._write_split_outputs(self.image_split)
        else:
            self._write_flat_outputs()

        if self.preview:
            self._write_previews()

        print(f"Dataset successfully created at {self.output_dir}")

    def _names_by_yolo_id(self):
        names = [None] * len(self.cat_id_to_yolo)
        for cat in self.coco["categories"]:
            names[self.cat_id_to_yolo[cat["id"]]] = cat["name"]
        return names

    def _write_flat_outputs(self):
        if self.output_format in ("yolo", "both"):
            write_classes_file(self.output_dir / "classes.txt", self._names_by_yolo_id())
            print(f"YOLO labels written to {self.labels_dir}")

        if self.output_format in ("coco", "both"):
            with open(self.output_dir / "annotations.json", "w") as f:
                json.dump(self.coco, f, indent=4)
            print(f"COCO annotations written to {self.output_dir / 'annotations.json'}")

    def _write_split_outputs(self, assignment):
        split_names = sorted(set(assignment.values()))
        images_by_split = {name: [] for name in split_names}

        for name in split_names:
            (self.images_dir / name).mkdir(parents=True, exist_ok=True)
            if self.output_format in ("yolo", "both"):
                (self.labels_dir / name).mkdir(parents=True, exist_ok=True)

        for img in self.coco["images"]:
            split_name = assignment[img["id"]]
            images_by_split[split_name].append(img)
            shutil.move(self.images_dir / img["file_name"], self.images_dir / split_name / img["file_name"])
            if self.output_format in ("yolo", "both"):
                label_filename = Path(img["file_name"]).stem + ".txt"
                shutil.move(self.labels_dir / label_filename, self.labels_dir / split_name / label_filename)

        if self.output_format in ("yolo", "both"):
            write_classes_file(self.output_dir / "classes.txt", self._names_by_yolo_id())
            self._write_data_yaml(split_names)
            print(f"YOLO labels split into {self.labels_dir}/{{{', '.join(split_names)}}}")

        if self.output_format in ("coco", "both"):
            annotations_dir = self.output_dir / "annotations"
            annotations_dir.mkdir(parents=True, exist_ok=True)
            for name in split_names:
                image_ids = {img["id"] for img in images_by_split[name]}
                split_coco = {
                    "images": images_by_split[name],
                    "annotations": [a for a in self.coco["annotations"] if a["image_id"] in image_ids],
                    "categories": self.coco["categories"],
                }
                with open(annotations_dir / f"instances_{name}.json", "w") as f:
                    json.dump(split_coco, f, indent=4)
            print(f"COCO annotations split into {annotations_dir}/instances_{{{', '.join(split_names)}}}.json")

    def _write_data_yaml(self, split_names):
        data = {
            "path": str(self.output_dir.resolve()),
            "names": {i: name for i, name in enumerate(self._names_by_yolo_id())},
        }
        for name in ("train", "val", "test"):
            if name in split_names:
                data[name] = f"images/{name}"

        with open(self.output_dir / "data.yaml", "w") as f:
            yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
        print(f"data.yaml written to {self.output_dir / 'data.yaml'}")

    def _write_previews(self):
        """Renders a random sample of tiles with their annotations drawn on top,
        read back from the actual persisted files (not recomputed in-memory), so
        the preview reflects exactly what ended up on disk for each format.
        """
        all_images = self.coco["images"]
        sample_size = max(1, round(len(all_images) * self.preview["ratio"]))
        sampled = random.Random(self.preview["seed"]).sample(all_images, min(sample_size, len(all_images)))

        if self.output_format in ("coco", "both"):
            anns_by_image = defaultdict(list)
            for ann in self.coco["annotations"]:
                anns_by_image[ann["image_id"]].append(ann)
            names_by_cat_id = {cat["id"]: cat["name"] for cat in self.coco["categories"]}

            preview_dir = self.output_dir / "preview" / "coco"
            for img in sampled:
                split_name = self.image_split.get(img["id"], "")
                out_dir = preview_dir / split_name
                out_dir.mkdir(parents=True, exist_ok=True)
                draw_coco_preview(
                    self.images_dir / split_name / img["file_name"],
                    anns_by_image.get(img["id"], []), names_by_cat_id,
                    out_dir / img["file_name"]
                )
            print(f"COCO preview written to {preview_dir} ({len(sampled)} images)")

        if self.output_format in ("yolo", "both"):
            names_by_yolo_id = self._names_by_yolo_id()

            preview_dir = self.output_dir / "preview" / "yolo"
            for img in sampled:
                split_name = self.image_split.get(img["id"], "")
                out_dir = preview_dir / split_name
                out_dir.mkdir(parents=True, exist_ok=True)
                label_filename = Path(img["file_name"]).stem + ".txt"
                draw_yolo_preview(
                    self.images_dir / split_name / img["file_name"],
                    self.labels_dir / split_name / label_filename, names_by_yolo_id,
                    out_dir / img["file_name"]
                )
            print(f"YOLO preview written to {preview_dir} ({len(sampled)} images)")


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
             f"(tile_size * res meters per side). Use 0 to keep the orthomosaic's native "
             f"resolution (no resampling). Default: {DEFAULTS['res']}."
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

    try:
        split = parse_split_config(config.get("split"))
        preview = parse_preview_config(config.get("preview"))
    except ValueError as e:
        parser.error(str(e))

    gen = DatasetGenerator(
        tif, shapes, out, tile_size=tile_size, res=res, output_format=output_format,
        split=split, preview=preview
    )
    gen.run()
