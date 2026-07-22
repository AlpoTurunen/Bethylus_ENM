# Ecological Niche Modeling (SDM) Maps

This repository includes a script (`run_models.py`) that builds simple Ecological Niche Models (ENM) using presence data and environmental rasters, then generates habitat suitability maps.

## What it does
- Loads presence records (GBIF + local cleaned data).
- Reads environmental raster layers (temperature, precipitation, elevation, human footprint).
 - Optionally clips predictions to a European bounding box for faster runs.
 - Samples background points from land pixels only (using a precomputed sea mask).
 - Trains multiple algorithms and produces a weighted ensemble of predictions.
 - Predicts habitat suitability across the target area and exports:
  - GeoTIFFs of predicted suitability
  - PNG maps
  - JSON metrics (AUC, accuracy, coefficients)

## Key settings
- `EUROPE`: when `True`, predictions are restricted to a European bounding box.
- `SCALE`: downscaling factor for raster resolution (higher = less memory / faster).

## Inputs (expected paths)
- `data/bethylus_fuscicornis_gbif.gpkg` and `data/bethylus_fuscicornis_local.gpkg` (occurrence records)
- CHELSA rasters organized under `data/chelsa_<variable>_rasters/` with subfolders `historical/`, `ssp126/`, `ssp585/` (see `fetch_chelsa_data.py` and `calculate_mean_raster_per_scenario.py`)
- Mean per-scenario rasters created in `mean_rasters/`
- Sea/land mask: `cleaned_env_data/sea_mask.tif` (used to restrict predictions to land)

## Outputs
- Model bundles (calibrated models, scalers, metadata): `models/`
- Prediction maps, ensemble means, uncertainty and metrics: `output/{REGION}/{period}/{scenario}/` (see `run_models.py` for exact filenames)

## Running
```bash
# Calibrate models (train on 1981-2010 using recent occurrences)
python run_models.py --mode calibrate

# Predict using pre-calibrated models for a scenario (e.g., ssp585)
python run_models.py --mode predict --scenario ssp585
```

## Notes
- The script assumes the raster layers all share the same grid and CRS.
- If you need a larger or smaller spatial extent, adjust `EUROPE` / `EUROPE_BBOX`.
- If you run out of memory, increase `SCALE` to reduce the raster size.
