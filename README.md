# Ortho2Dataset
A Python pipeline to automate the generation of computer vision datasets from high-resolution orthomosaics and shapefiles.

## Requirements

Install the Python dependencies:

```
pip install -r requirements.txt
```

This project also shells out to `gdalwarp` (used to resample the orthomosaic to the target output resolution before tiling), so the GDAL command-line tools must be installed separately on the system — they are not installable via pip alone:

```
# Debian/Ubuntu
sudo apt install gdal-bin


Verify it's available with `gdalwarp --version`.
