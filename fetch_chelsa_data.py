import requests
from pathlib import Path



raster_names = ["bio14", "bio15", "bio16",
                "bio17", "bio18", "bio19", "gdd5"]

for raster_name in raster_names:
    OUTPUT_DIR = Path(f"data/chelsa_{raster_name}_rasters")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    models = [("UKESM1-0-LL", "ukesm1-0-ll"), ("MPI-ESM1-2-HR", "mpi-esm1-2-hr"), ("IPSL-CM6A-LR", "ipsl-cm6a-lr")]
    scenarios = ["ssp126", "ssp585"]

    (OUTPUT_DIR / "historical").mkdir(exist_ok=True)
    response = requests.get(f"https://os.unil.cloud.switch.ch/chelsa02/chelsa/global/bioclim/{raster_name}/1981-2010/CHELSA_{raster_name}_1981-2010_V.2.1.tif", stream=True)
    with open(OUTPUT_DIR / "historical" / f"CHELSA_{raster_name}_1981-2010_V.2.1.tif", 'wb') as f:
        for chunk in response.iter_content(8192):
            if chunk:
                f.write(chunk)
        print(f"Downloaded {OUTPUT_DIR / 'historical' / f'CHELSA_{raster_name}_1981-2010_V.2.1.tif'}")

    for scenario in scenarios:
        (OUTPUT_DIR / scenario).mkdir(exist_ok=True)
        for full, lower in models:
            url = f"https://os.unil.cloud.switch.ch/chelsa02/chelsa/global/bioclim/{raster_name}/2071-2100/{full}/{scenario}/CHELSA_{lower}_{scenario}_{raster_name}_2071-2100_V.2.1.tif"
            filepath = OUTPUT_DIR / scenario / f"CHELSA_{lower}_{scenario}_{raster_name}_2071-2100_V.2.1.tif"
            response = requests.get(url, stream=True)
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(8192):
                    if chunk:
                        f.write(chunk)
                print(f"Downloaded {filepath}")
