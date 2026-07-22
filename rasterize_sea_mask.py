# 1. Create this preprocessing script (save as preprocess_sea_mask.py):
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize as rio_rasterize
import os

# Load reference raster
temp_tiff = r"data\chelsa_bio01_rasters\historical\CHELSA_bio01_1981-2010_V.2.1.tif"
with rasterio.open(temp_tiff) as src:
    ref_meta = src.meta.copy()
    ref_transform = src.transform
    ref_crs = src.crs
    height, width = src.height, src.width

print("Reference raster loaded:")
# Rasterize seas once
seas_gdf = gpd.read_file("GOaS_v1_20211214_gpkg/goas_v01.gpkg").to_crs(ref_crs)
sea_shapes = [(geom, 1) for geom in seas_gdf.geometry]

print(f"Rasterizing sea polygons to create sea mask...")
sea_mask = rio_rasterize(
    sea_shapes,
    out_shape=(height, width),
    transform=ref_transform,
    fill=0,
    dtype=np.uint8
)

print("Sea mask rasterized. Saving to GeoTIFF...")
# Create output directory if it doesn't exist
os.makedirs('cleaned_env_data', exist_ok=True)
# Save as GeoTIFF
out_meta = ref_meta.copy()
out_meta.update({'dtype': 'uint8', 'count': 1, 'nodata': 0})
with rasterio.open('cleaned_env_data/sea_mask.tif', 'w', **out_meta) as dst:
    dst.write(sea_mask, 1)

print("Sea mask saved to cleaned_env_data/sea_mask.tif")