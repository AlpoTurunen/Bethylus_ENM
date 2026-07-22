# Ecological Niche Modelling Workflow Report

## 1. Data Acquisition

The workflow begins by downloading CHELSA climate data for a historical baseline period and future projections. The script `fetch_chelsa_data.py` automates retrieval of bioclimatic variables `bio01` through `bio19`, plus `gdd5`, from the CHELSA v2.1 repository (Brun et. al. 2022).

- Historical baseline: 1981–2010.
- Future projection period: 2071–2100.
- Future scenarios: `ssp126` and `ssp585`.
- General models: `UKESM1-0-LL`, `MPI-ESM1-2-HR`, and `IPSL-CM6A-LR`, which are all European models.

This download stage produces one raster file per variable, scenario, and GCM (Global Climate Model) realization, ensuring that the same predictor set is available for both calibration and projection. This phase can take up to tens of gigabytes of storage temporarily.


## 2. Raster Processing

### 2.1 Mean raster creation

The script `calculate_mean_raster_per_scenario.py` calculates per-scenario mean raster layers from the downloaded GCM-specific TIFF files. For each climate variable and scenario, the workflow:

1. loads all available model-specific TIFFs for that variable,
2. converts the raster values to floating-point arrays,
3. computes the pixel-wise mean across models,
4. writes a new mean raster to `mean_rasters/`.
5. deletes temporary rasters, which were downloaded earlier

This step reduces inter-model variability and creates a single representative predictor layer for each variable under each scenario.

### 2.2 Sea mask rasterization

The script `rasterize_sea_mask.py` creates a sea mask aligned to the CHELSA reference grid using GOaS (Global Oceans and Seas) coastline polygons (Flanders Marine Institute, 2021). The sea mask is rasterized at the same resolution, transform, and CRS as the reference CHELSA raster, then saved as `cleaned_env_data/sea_mask.tif`.

The sea mask is used to exclude marine pixels from the modelling grid, so that predictions are constrained to land and terrestrial habitat only.

### 2.3 Predictor alignment

All environmental rasters are aligned to the reference CHELSA grid. This ensures that the spatial extent, coordinate reference system, and raster metadata are consistent across all predictor layers, which is critical for stacking variables and mapping predictions back to geographic space.

## 3. Ecological Niche Modelling Pipeline

The main modelling workflow is implemented in `sdm_maps.py`, which contains two distinct operation modes:

- `calibrate_mode(args)`
- `predict_mode(args)`

### 3.1 Calibration phase

In calibration mode, models are trained using contemporary species observations and baseline environmental conditions.

#### Observation data

- Presence records are loaded from `data/bethylus_fuscicornis_gbif.gpkg` and `data/bethylus_fuscicornis_local.gpkg`. First one is downloaded from Global Biodiversity Information Facility (GBIF) and another one was the local Excel datasheet converted to a GeoPackage. Both files have been cleaned to contain similar structure and columns.
- Records are merged into a single GeoDataFrame of recent observations.

#### Environmental predictors

- Predictor variables include bioclimatic variables `bio01`–`bio19` plus `gdd5`.
- Predictor layers are loaded into a stack and filtered for collinearity.
- Variable selection uses correlation and variance inflation factor (VIF) thresholds to remove redundant predictors.

#### Model training

- The workflow performs three independent calibration runs with different random seeds (`42`, `123`, `456`).
- Each run includes spatial thinning of presence records to reduce spatial bias and adaptive background sampling to produce pseudo-absence points.
- Data are split into training and test sets using an approximate 80% training / 20% test split.
- Multiple algorithms (LogisticRegression, RandomForestClassifier, SVC (Support Vector Machine) and GradientBoostingClassifier) are trained and evaluated using stratified 5-fold cross-validation on the training set.

#### Ensemble creation

- Model outputs are combined into an ensemble using a weighted average of algorithm predictions.
- Ensemble weights are derived from model performance metrics such as TSS.
- The pipeline computes mean and standard deviation suitability maps across the three calibration runs.

#### Performance metrics

The calibration phase reports the following metrics:

- True Skill Statistic (TSS)
- Area Under the Receiver Operating Characteristic Curve (AUC)
- Area Under the Precision-Recall Curve (AUC-PR)

These metrics are computed for cross-validation folds and for held-out test data.

#### Outputs

Calibration outputs include:

- ensemble mean suitability raster
- ensemble standard deviation raster
- JSON summary of metrics
- PNG figure of ensemble mean suitability

Output files are written to `output/{REGION}/1981-2010/historical/ensemble_mean/`.

### 3.2 Prediction phase

In prediction mode, the workflow loads calibrated models and applies them to future climate scenarios.

#### Projection data

- Future environmental rasters for 2071–2100 are loaded from mean scenario-specific layers.
- The same predictor variables and variable order used during calibration are preserved.
- The previously saved `StandardScaler` is reused to scale predictor values consistently.

#### Model application

- The calibrated model ensemble is loaded from disk.
- Predictions are made only for land pixels defined by the sea/land mask.
- No retraining occurs during projection; the model coefficients remain fixed.

#### Outputs

Prediction outputs are written to `output/{REGION}/2071-2100/{scenario}/` and include suitability maps and associated raster products.

## 4. Methodological principles

### 4.1 Separation of calibration and projection

The workflow follows a best-practice principle: calibrate models on contemporaneous observations and baseline climate, then project to future conditions without retraining. This avoids information leakage and ensures that projected suitability reflects the model learned from observed species-environment relationships.

### 4.2 Ensemble and uncertainty handling

- Multiple GCMs per scenario are averaged to reduce noise in future predictor fields.
- Multiple model runs quantify variability in the calibration phase.
- Ensemble mean and standard deviation maps provide measures of predicted suitability and projection uncertainty.

### 4.3 Reproducibility

- Random seeds are explicitly set for repeated runs.
- Raster metadata are preserved when writing outputs.
- Model artifacts and metric summaries are stored in structured output directories.

## 5. Summary

This workflow provides a complete pipeline from raw CHELSA climate data through raster preprocessing, spatial masking, model calibration, and future projection. It is suitable for research that requires reproducible species ecological niche models constrained by historical calibration and scenario-based climate projections. It is possible to use this same model for different species.


## References:
Flanders Marine Institute (2021). Global Oceans and Seas, version 1. Available online at https://www.marineregions.org/. https://doi.org/10.14284/542.

Brun, P., Zimmermann, N. E., Hari, C., Pellissier, L., Karger, D. N. (2022). CHELSA-BIOCLIM+ A novel set of global climate-related predictors at kilometre-resolution. EnviDat. https://www.doi.org/10.16904/envidat.332.