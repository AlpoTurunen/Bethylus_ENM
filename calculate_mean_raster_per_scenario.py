import rasterio
import numpy as np
from pathlib import Path
from osgeo import gdal
from concurrent.futures import ThreadPoolExecutor, as_completed


raster_names = ["bio01", "bio02", "bio03", "bio04",
                "bio05", "bio06", "bio07", "bio08",
                "bio09", "bio10", "bio11", "bio12",
                "bio13", "bio14", "bio15", "bio16",
                "bio17", "bio18", "bio19", "gdd5"]

scenarios = ["ssp126", "ssp585"]


def process_raster(scenario, raster_name):
    folder = f'data/chelsa_{raster_name}_rasters/{scenario}'
    tif_files = sorted(Path(folder).glob('*.tif'))

    with rasterio.open(tif_files[0]) as src:
        profile = src.profile
        original_dtype = src.dtypes[0]
        scale = src.scales[0]
        offset = src.offsets[0]
        data_list = [src.read(1).astype(np.float32)]

    for tif_file in tif_files[1:]:
        with rasterio.open(tif_file) as src:
            data_list.append(src.read(1).astype(np.float32))

    print(f"Calculating mean raster for {len(tif_files)} files in {folder}")
    mean_data = np.mean(data_list, axis=0).astype(original_dtype)

    profile.update(dtype=original_dtype)
    output_path = f'mean_rasters/{scenario}_{raster_name}_mean.tif'

    with rasterio.open(output_path, 'w', **profile) as dst:
        print(f"Writing mean raster to {output_path}")
        dst.write(mean_data, 1)

    ds = gdal.Open(str(output_path), gdal.GA_Update)
    ds.GetRasterBand(1).SetScale(scale)
    ds.GetRasterBand(1).SetOffset(offset)
    ds.FlushCache()
    ds = None


tasks = [(s, r) for s in scenarios for r in raster_names]
max_workers = 2

print(f"Processing {len(tasks)} tasks with {max_workers} workers...")
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = {executor.submit(process_raster, s, r): (s, r) for s, r in tasks}
    for future in as_completed(futures):
        future.result()