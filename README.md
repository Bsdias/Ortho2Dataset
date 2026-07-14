# Ortho2Dataset

A Python pipeline that turns a georeferenced orthomosaic (GeoTIFF) plus any
number of vector shapefiles (shapefile, GeoJSON, GeoPackage, etc.) into a
computer vision dataset ready for training: the orthomosaic is cut into fixed
size tiles, and every shape that falls inside each tile is converted into
COCO and/or YOLO segmentation annotations.

## How it works

1. The input orthomosaic is optionally resampled to a target ground
   resolution (meters/pixel) using `gdalwarp`.
2. The raster is scanned in fixed-size windows. This is just a cheap
   bounding-box check per window, windows with no shape inside them are
   discarded before any pixel is read, so no tile image is extracted for
   empty areas.
3. Tiling happens for every window that does contain at least one shape: the
   pixels within that window are cropped from the raster and saved as a JPEG
   tile image. Every shape geometry that intersects the tile is also clipped
   to the tile boundary and converted into pixel-space polygons.
4. Those polygons are turned into COCO annotations, YOLO label lines, or
   both, depending on the requested output format.
5. Optionally, the generated tiles are split into train/val/test partitions,
   and/or a random sample is rendered with its annotations drawn on top for a
   quick visual sanity check.

## Requirements

Install the Python dependencies:

```
pip install -r requirements.txt
```

This project also shells out to `gdalwarp`, so the GDAL command-line tools
must be installed separately on the system — they are not installable via
pip alone:

```
# Debian/Ubuntu
sudo apt install gdal-bin

# conda
conda install -c conda-forge gdal
```

Verify it's available with `gdalwarp --version`.

## Quick start

The simplest way to run the pipeline is through a config file:

```
cp config.example.yaml config.yaml
# edit config.yaml with your paths, resolution, classes, etc.
python ortho2dataset.py --config config.yaml
```

Any value in `config.yaml` can still be overridden with a matching CLI flag,
for a one-off change without editing the file:

```
python ortho2dataset.py --config config.yaml --res 0.10 --format yolo
```

The pipeline can also be run purely from CLI flags, without any config file:

```
python ortho2dataset.py \
    --tif orthomosaic.tif \
    --shapes lot=lots.gpkg block=blocks.gpkg \
    --out dataset/my_dataset/ \
    --res 0.20
```

If a GeoPackage has more than one layer, discover the layer names first:

```
python ortho2dataset.py --list-layers data.gpkg
```

## Configuration reference (`config.yaml`)

See `config.example.yaml` for a commented, ready-to-copy template. Fields:

| Key | Required | Description |
|---|---|---|
| `tif` | yes | Path to the input orthomosaic (GeoTIFF). |
| `out` | yes | Output directory for the generated dataset. |
| `shapes` | yes | Mapping of class name to shape source (see below). |
| `res` | no (default `0`) | Output resolution in meters/pixel. Since tile images always have a fixed pixel size, this is what controls how much ground area each tile covers (`tile_size * res` meters per side). `0` keeps the orthomosaic's native resolution (skips resampling). |
| `tile_size` | no (default `1024`) | Tile size in pixels (tiles are always square). |
| `format` | no (default `coco`) | `coco`, `yolo`, or `both` — which annotation format(s) to generate. |
| `split` | no | Splits the dataset into train/val/test. Omit to keep a single flat dataset. |
| `preview` | no | Fraction of images (0-1) to render with annotations drawn on top, for visual QA. `0` (or omitted) skips it. |

### `shapes`

Each key becomes a class/category in the output dataset — it does not need
to match the source file or layer name. The value is either a plain path
(for a single-layer source, e.g. a shapefile), or a `{path, layer}` mapping
for a specific layer inside a multi-layer source (e.g. a GeoPackage). The
same file can supply more than one class, one per layer. See
`config.example.yaml` for both forms.

### `split`

Ratios must add up to 1.0. `test` may be omitted (or set to `0`) to skip a
test partition. The split is a random, seeded shuffle over the generated
tiles — not tied to raster position.

### `preview`

Renders a random sample of tiles with their annotations drawn on top of the
image, so you can visually confirm the pipeline worked correctly without
opening a full annotation tool (CVAT, LabelImg, etc.). The sample is read
back from the actual files written to disk (not recomputed in memory), so it
reflects exactly what ended up in the dataset. If `format: both`, two
separate preview folders are generated (one per annotation format) so you
can cross-check that COCO and YOLO outputs render identically.

## CLI reference

| Flag | Description |
|---|---|
| `--config PATH` | YAML config file providing defaults for any flag below. |
| `--tif PATH` | Path to the input orthomosaic. |
| `--shapes NAME=PATH[:LAYER] ...` | One or more class definitions, e.g. `--shapes lot=lots.gpkg block=data.gpkg:blocks`. |
| `--out PATH` | Output directory. |
| `--res FLOAT` | Output resolution in meters/pixel (`0` = native resolution). |
| `--tile-size INT` | Tile size in pixels. |
| `--format {coco,yolo,both}` | Annotation format(s) to generate. |
| `--list-layers PATH` | Lists the layer names in a vector file and exits, without processing anything. |

Flags passed on the command line always take precedence over the matching
`config.yaml` value. `split` and `preview` are only configurable through
`config.yaml` (they are structured, multi-value settings that don't map
cleanly onto a single flag).

## Output structure

Without a `split` configured:

```
dataset/
  images/tile_00001.jpg ...
  annotations.json              # if format includes coco
  labels/tile_00001.txt ...     # if format includes yolo
  classes.txt                   # if format includes yolo
  preview/coco/ ...              # if preview is set and format includes coco
  preview/yolo/ ...              # if preview is set and format includes yolo
```

With a `split` configured, images and labels are grouped into `train/`,
`val/`, `test/` subfolders, COCO annotations are split into one JSON file per
partition, and a `data.yaml` (Ultralytics-style) is written when the format
includes yolo:

```
dataset/
  images/{train,val,test}/tile_00001.jpg ...
  labels/{train,val,test}/tile_00001.txt ...   # if format includes yolo
  annotations/instances_{train,val,test}.json  # if format includes coco
  classes.txt                                  # if format includes yolo
  data.yaml                                    # if format includes yolo
  preview/{coco,yolo}/{train,val,test}/ ...    # if preview is set
```

`data.yaml` follows the Ultralytics convention (`path`, `train`, `val`,
`test`, `names`) and can be passed directly to `yolo train data=data.yaml ...`.
COCO-consuming frameworks (Detectron2, mmdetection, torchvision, etc.) don't
use a `data.yaml` — they take the annotation JSON path and the image folder
path directly.

## Project structure

```
ortho2dataset.py          CLI entry point and the DatasetGenerator orchestrator
config.example.yaml       Commented config template (copy to config.yaml)
requirements.txt          Python dependencies

utils/
  config.py               Loads config.yaml, normalizes shapes/preview settings
  shapes.py               Parses --shapes entries, lists/loads vector layers
  ortho_tiling.py          Raster resampling and tile window generation
  geometry.py             Clips a geometry to a tile and projects it to pixel space
  shape2coco.py           Builds COCO annotation dicts from pixel-space polygons
  shape2yolo.py           Builds YOLO label lines from pixel-space polygons
  split.py                Validates split ratios and assigns tiles to train/val/test
  visualize.py            Draws annotations on top of tiles for the preview feature
```

`geometry.py` is intentionally the only module that touches raw shapely
geometry — `shape2coco.py` and `shape2yolo.py` both consume the same
pixel-space polygons independently, so neither annotation format depends on
the other.
