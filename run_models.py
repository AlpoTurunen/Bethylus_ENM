import argparse
import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio
from rasterio.transform import array_bounds
from shapely.geometry import box
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import os
import json
import warnings
from config import EUROPE, EUROPE_BBOX, REGION
from helpers import (
    spatial_thin, extract_env_at_points, train_ensemble_models, weighted_ensemble_predict, calculate_tss, calculate_auprc, create_buffer_mask,
    calculate_adaptive_background_size, save_calibrated_models, load_calibrated_models, filter_collinear_variables, build_prediction_grid
)
warnings.filterwarnings('ignore')

def load_observation_data():
    """Load merged observation data (recent, all species)."""
    print("Loading observation data...")
    gbif_data = gpd.read_file('data/bethylus_fuscicornis_gbif.gpkg')
    local_data = gpd.read_file('data/bethylus_fuscicornis_local.gpkg')
    
    merged_gdf = gpd.GeoDataFrame(
        pd.concat([gbif_data, local_data], ignore_index=True), 
        crs=gbif_data.crs
    )
    print(f"Total presence points after merging: {len(merged_gdf)}")
    print(f"CRS of merged data: {merged_gdf.crs}")
    return merged_gdf


def calibrate_mode(args):
    """
    CALIBRATION PHASE: Train models on 1981-2010 environment + recent observations.
    This is the only time models are trained. Models are saved to disk.
    
    Principle: "Current records cannot be modelled directly against past or future 
    environmental data. Models should be built using contemporaneous data."
    
    In this case:
    - Observations: Recent (all records are from the last 10-20 years, contemporary)
    - Environment: 1981-2010 (the baseline/reference period we calibrate to)
    
    This ensures models capture niche relationships for the contemporaneous period,
    which can then be projected to other time periods.
    """
    print("\n" + "="*80)
    print("CALIBRATION MODE")
    print("="*80)
    print("Training models on: Recent observations + 1981-2010 environment")
    print("Models will be saved to: models/{species}_{region}_calibrated.pkl")
    print("Note: Models trained only once. Use --mode predict to project to other periods.\n")
    
    threshold = args.threshold
    print(f"Using decision threshold: {threshold:.2f}")
    
    # HARDCODED: Training always uses 1981-2010 environment
    calibration_years = '1981-2010'
    data_scenario = 'historical'
    
    # Load environmental raster files from variable-specific folders
    print(f"Loading environmental data for {calibration_years} ({data_scenario})...")
    raster_files = {}
    
    # List of bio variables and gdd5
    bio_vars = [f'bio{i:02d}' for i in range(1, 20)]  # bio01 to bio19
    all_vars = bio_vars + ['gdd5']
    
    for var_name in all_vars:
        raster_dir = f'data/chelsa_{var_name}_rasters/{data_scenario}/'
        
        if not os.path.exists(raster_dir):
            print(f"  Warning: Directory not found: {raster_dir}")
            continue
        
        # Find .tif file in this directory
        for filename in os.listdir(raster_dir):
            if filename.endswith('.tif') and not filename.endswith('.tif.aux.xml'):
                file_path = os.path.join(raster_dir, filename)
                raster_files[var_name] = file_path
                break
    
    print(f"Found {len(raster_files)} environmental variables: {sorted(raster_files.keys())}")
    
    if not raster_files:
        raise FileNotFoundError(f"No raster files found for {calibration_years} ({data_scenario})")
    
    env_vars = sorted(raster_files.keys())
    
    # Load observation data
    merged_gdf = load_observation_data()
    species = merged_gdf['species'].unique()[0]  # Single species dataset
    print(f"Processing species: {species}\n")
    
    # Build prediction grid and load environmental data
    grid_info = build_prediction_grid(raster_files)
    height, width = grid_info['height'], grid_info['width']
    ref_transform = grid_info['ref_transform']
    ref_meta = grid_info['ref_meta']
    band_arrays = grid_info['band_arrays']
    land_mask = grid_info['land_mask']
    
    # Filter collinear variables and remove variables with excessive VIF
    env_vars, band_arrays = filter_collinear_variables(
        band_arrays, land_mask, env_vars, corr_threshold=0.7, vif_threshold=10.0
    )
    
    # Stack environmental data
    X_stack = np.stack([band_arrays[v] for v in env_vars], axis=-1)
    X_flat = X_stack.reshape(-1, len(env_vars))
    land_flat = land_mask.reshape(-1)
    X_land = X_flat[land_flat]
    
    # Lists to store suitability maps from each run for ensemble calculation
    run_suitability_maps = []
    run_ensemble_metrics = []
    
    # Process 3 independent runs with different random seeds
    run_seeds = [42, 123, 456]
    print(f"\n{'='*80}")
    print(f"Starting 3 independent calibration runs")
    print(f"{'='*80}")
    
    for run_num, seed in enumerate(run_seeds, 1):
        print(f"\n{'#'*80}")
        print(f"CALIBRATION RUN {run_num}/3 (seed={seed})")
        print(f"{'#'*80}")
        
        run_out_dir = f'output/{REGION}/{calibration_years}/{data_scenario}/run{run_num}/'
        os.makedirs(run_out_dir, exist_ok=True)
        
        rng = np.random.default_rng(seed)
        
        print(f"\n{'='*80}")
        print(f"Run {run_num}: Calibrating {species}")
        
        species_data = merged_gdf[merged_gdf['species'] == species].copy()
            
        if EUROPE:
            lon_min, lat_min, lon_max, lat_max = EUROPE_BBOX
            bbox_poly = box(lon_min, lat_min, lon_max, lat_max)
            species_data = species_data[species_data.geometry.intersects(bbox_poly)]
        
        n_presence_before = len(species_data)
        print(f"  Presence points before thinning: {n_presence_before}")
        species_data = spatial_thin(species_data, grid_size_deg=0.05)
        
        n_presence = len(species_data)
        print(f"  Presence points after thinning: {n_presence}")
            
        if n_presence < 15:
            print("  Skipping: too few presence points after thinning (<15)")
            raise ValueError(f"Not enough presence points for {species}")
            
        # Extract env values at presence locations
        print("  Extracting env values at presence points...")
        X_pres = extract_env_at_points(species_data, env_vars, ref_transform, height, width, band_arrays)
        valid_pres = ~np.any(np.isnan(X_pres), axis=1)
        X_pres = X_pres[valid_pres]
        species_data_valid = species_data[valid_pres]
        print(f"  Valid presence points: {len(X_pres)}")
        
        if len(X_pres) < 15:
            print("  Skipping: too few valid presence points with env data")
            raise ValueError(f"Not enough valid presence points for {species}")
            
        # Create buffer mask
        print("  Creating buffer around presence points...")
        presence_coords = np.column_stack([species_data_valid.geometry.x, species_data_valid.geometry.y])
        buffer_mask = create_buffer_mask(presence_coords, ref_transform, height, width, buffer_deg=0.1)
        print(f"    Pixels excluded by buffer: {buffer_mask.sum():,}")
        
        valid_bg_mask = land_mask & ~buffer_mask
        valid_bg_indices = np.where(valid_bg_mask.reshape(-1))[0]
        
        n_presence = len(X_pres)
        n_background = 5000 # Adjust based on presence points
        n_background = min(n_background, len(valid_bg_indices))
        
        bg_idx = rng.choice(valid_bg_indices, size=n_background, replace=False)
        X_bg = X_flat[bg_idx]
        print(f"  Background points: {len(X_bg)} (adaptive ratio ~1:{n_background//max(1, n_presence)})")
        
        # Assemble training data
        X = np.vstack([X_pres, X_bg])
        y = np.array([1] * len(X_pres) + [0] * len(X_bg))
        
        # Two-way data split: training (80%) and test (20%)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y
        )
        
        # Scale features on training set only, then apply to test set
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        print(f"  Data split: Train={len(X_train)} ({len(X_train)/len(X)*100:.0f}%), "
              f"Test={len(X_test)} ({len(X_test)/len(X)*100:.0f}%)")
        
        # Stratified k-fold cross-validation on TRAINING set only for robust evaluation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        fold_metrics = []
        
        print(f"  Performing 5-fold cross-validation on training set...")
        
        for fold_num, (train_idx, cv_test_idx) in enumerate(skf.split(X_train_scaled, y_train), 1):
            X_cv_train, X_cv_test = X_train_scaled[train_idx], X_train_scaled[cv_test_idx]
            y_cv_train, y_cv_test = y_train[train_idx], y_train[cv_test_idx]
            
            # Train 4 model types
            model_results = train_ensemble_models(X_cv_train, y_cv_train, X_cv_test, y_cv_test, random_state=seed)
            
            top_models = [m[0] for m in model_results[:4]]
            top_tss_scores = [m[2] for m in model_results[:4]]
            top_auc_scores = [m[3] for m in model_results[:4]]
            top_auprc_scores = [m[4] for m in model_results[:4]]
            top_aic_scores = [m[5] for m in model_results[:4]]
            
            # Compute fold ensemble metrics at the configured threshold
            ensemble_probs_test = weighted_ensemble_predict(top_models, top_tss_scores, X_cv_test)
            ensemble_pred_test = (ensemble_probs_test > threshold).astype(int)
            ensemble_tss = calculate_tss(y_cv_test, ensemble_pred_test)
            ensemble_auc = roc_auc_score(y_cv_test, ensemble_probs_test)
            ensemble_auprc = calculate_auprc(y_cv_test, ensemble_probs_test)
            
            fold_metrics.append({
                'fold': fold_num,
                'tss': ensemble_tss,
                'auc': ensemble_auc,
                'auprc': ensemble_auprc,
                'n_test': len(y_cv_test)
            })
            print(f"    Fold {fold_num}/5: TSS={ensemble_tss:.3f}  AUC={ensemble_auc:.3f}  AUC-PR={ensemble_auprc:.3f}")
        
        # Compute mean cross-validation metrics
        cv_mean_tss = np.mean([m['tss'] for m in fold_metrics])
        cv_std_tss = np.std([m['tss'] for m in fold_metrics])
        cv_mean_auc = np.mean([m['auc'] for m in fold_metrics])
        cv_std_auc = np.std([m['auc'] for m in fold_metrics])
        cv_mean_auprc = np.mean([m['auprc'] for m in fold_metrics])
        cv_std_auprc = np.std([m['auprc'] for m in fold_metrics])
        
        print(f"  Cross-validation results (5-fold on training set):")
        print(f"    Mean TSS: {cv_mean_tss:.3f} ± {cv_std_tss:.3f}")
        print(f"    Mean AUC: {cv_mean_auc:.3f} ± {cv_std_auc:.3f}")
        print(f"    Mean AUC-PR: {cv_mean_auprc:.3f} ± {cv_std_auprc:.3f}")
        
        # Train final models on FULL TRAINING set for suitability prediction
        print(f"  Training final ensemble on full training set...")
        model_results_final = train_ensemble_models(X_train_scaled, y_train, X_test_scaled, y_test, random_state=seed)
        
        top_models = [m[0] for m in model_results_final[:4]]
        top_tss_scores = [m[2] for m in model_results_final[:4]]
        top_auc_scores = [m[3] for m in model_results_final[:4]]
        top_auprc_scores = [m[4] for m in model_results_final[:4]]
        top_aic_scores = [m[5] for m in model_results_final[:4]]
        
        print(f"  Model ranks by TSS:")
        for i, (model, name, tss, auc, auprc, aic) in enumerate(model_results_final, 1):
            in_ensemble = "✓" if i <= 4 else "✗"
            print(f"    {i}. {name:20s} TSS={tss:.3f} AUC={auc:.3f} AUC-PR={auprc:.3f} AIC={aic:.1f} {in_ensemble}")
        
        # Evaluate on TEST set with the configured threshold
        print(f"  Evaluating on test set with threshold {threshold:.2f}...")
        ensemble_probs_test = weighted_ensemble_predict(top_models, top_tss_scores, X_test_scaled)
        ensemble_pred_test = (ensemble_probs_test > threshold).astype(int)
        ensemble_tss = calculate_tss(y_test, ensemble_pred_test)
        ensemble_auc = roc_auc_score(y_test, ensemble_probs_test)
        ensemble_auprc = calculate_auprc(y_test, ensemble_probs_test)
        
        best_threshold = threshold
        print(f"  Test set results: TSS={ensemble_tss:.3f}  AUC={ensemble_auc:.3f}  AUC-PR={ensemble_auprc:.3f}")
        
        print(f"  Ensemble TSS: {ensemble_tss:.3f}  |  Ensemble AUC: {ensemble_auc:.3f}  |  Ensemble AUC-PR: {ensemble_auprc:.3f}")
        
        # Predict suitability on all valid land pixels
        print("  Predicting calibration suitability...")
        X_land_scaled = scaler.transform(X_land)
        suitability_land = weighted_ensemble_predict(top_models, top_tss_scores, X_land_scaled)
        
        suitability_flat = np.full(height * width, np.nan, dtype=np.float32)
        suitability_flat[land_flat] = suitability_land
        suitability_map = suitability_flat.reshape(height, width)
        
        # Store for ensemble calculation
        run_suitability_maps.append(suitability_map.copy())
        run_ensemble_metrics.append({
            'run': run_num,
            'seed': seed,
            'cv_mean_tss': cv_mean_tss,
            'cv_std_tss': cv_std_tss,
            'cv_mean_auc': cv_mean_auc,
            'cv_std_auc': cv_std_auc,
            'cv_mean_auprc': cv_mean_auprc,
            'cv_std_auprc': cv_std_auprc,
            'test_threshold': best_threshold,
            'test_tss': ensemble_tss,
            'test_auc': ensemble_auc,
            'test_auprc': ensemble_auprc,
            'n_presence': len(X_pres),
            'n_background': len(X_bg),
            'n_training': len(X_train),
            'n_test': len(X_test)
        })
        
        species_safe = species.replace(' ', '_')
        
        # Save calibration suitability GeoTIFF
        out_tiff = f"{run_out_dir}/{species_safe}_calibration_suitability.tif"
        out_meta = ref_meta.copy()
        out_meta.update({'dtype': 'float32', 'count': 1, 'nodata': -9999.0})
        suitability_out = np.where(land_mask, suitability_map, -9999.0).astype(np.float32)
        with rasterio.open(out_tiff, 'w', **out_meta) as dst:
            dst.write(suitability_out, 1)
        print(f"  Saved: {out_tiff}")
        
        # Save calibration metrics JSON
        ensemble_model_names = [m[1] for m in model_results_final[:4]]
        results_dict = {
            'species': species,
            'calibration_period': '1981-2010',
            'run': run_num,
            'seed': seed,
            'presence_points_original': int(n_presence_before),
            'presence_points_after_thinning': int(len(X_pres)),
            'background_points': int(len(X_bg)),
            'data_split': {
                'training': int(len(X_train)),
                'test': int(len(X_test))
            },
            'ensemble_method': 'tss_weighted_mean',
            'ensemble_models': ensemble_model_names,
            'cross_validation': {
                'method': '5-fold stratified k-fold on training set',
                'cv_mean_tss': round(float(cv_mean_tss), 4),
                'cv_std_tss': round(float(cv_std_tss), 4),
                'cv_mean_auc': round(float(cv_mean_auc), 4),
                'cv_std_auc': round(float(cv_std_auc), 4),
                'cv_mean_auprc': round(float(cv_mean_auprc), 4),
                'cv_std_auprc': round(float(cv_std_auprc), 4),
                'fold_metrics': fold_metrics
            },
            'threshold_selection': {
                'method': f'fixed threshold {threshold:.2f}',
                'threshold': round(float(best_threshold), 2)
            },
            'final_test_results': {
                'threshold': round(float(best_threshold), 2),
                'ensemble_tss': round(float(ensemble_tss), 4),
                'ensemble_auc': round(float(ensemble_auc), 4),
                'ensemble_auprc': round(float(ensemble_auprc), 4),
                'note': 'Test set used for final evaluation; no separate validation set'
            },
            'individual_model_scores': [
                {'model': name, 'tss': round(float(tss), 4), 'auc': round(float(auc), 4), 'auprc': round(float(auprc), 4), 'aic': round(float(aic), 4)}
                for _, name, tss, auc, auprc, aic in model_results_final
            ],
            'variables': env_vars
        }
        with open(f"{run_out_dir}/{species_safe}_calibration_results.json", 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        # Plot calibration suitability map
        left, bottom, right, top = array_bounds(height, width, ref_transform)
        fig, ax = plt.subplots(figsize=(18, 10))
        masked_suit = np.ma.masked_where(~land_mask, suitability_map)
        im = ax.imshow(
            masked_suit, cmap='YlOrRd', vmin=0, vmax=1,
            extent=(left, right, bottom, top),
            aspect='auto', interpolation='nearest', origin='upper'
        )
        ax.scatter(
            species_data.geometry.x, species_data.geometry.y,
            c='dodgerblue', s=14, alpha=0.7,
            edgecolors='white', linewidths=0.3,
            label=f'Presence (n={len(species_data)})', zorder=5
        )
        cbar = plt.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
        cbar.set_label('Habitat Suitability', fontsize=11)
        region_label = 'European' if EUROPE else 'Global'
        ax.set_title(
            f'Run {run_num}: {species} — {region_label} Calibration Suitability '
            f'(1981-2010, TSS={ensemble_tss:.3f}, AUC={ensemble_auc:.3f})',
            fontsize=14
        )
        ax.set_xlabel('Longitude', fontsize=10)
        ax.set_ylabel('Latitude', fontsize=10)
        ax.legend(loc='lower left', fontsize=9)
        plt.tight_layout()
        out_png = f"{run_out_dir}/{species_safe}_calibration_map.png"
        plt.savefig(out_png, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {out_png}")
        
        # ===== SAVE CALIBRATED MODELS TO DISK =====
        if run_num == 1:  # Save only on first run
            print("  SAVING CALIBRATED MODELS...")
            models_dict = {
                'models': top_models,
                'tss_scores': top_tss_scores,
                'auc_scores': top_auc_scores,
                'auprc_scores': top_auprc_scores,
                'aic_scores': top_aic_scores
            }
            save_calibrated_models(
                species, REGION, models_dict, scaler, env_vars, ensemble_model_names
            )
    
    # Compute ensemble mean across 3 runs
    print(f"\n{'='*80}")
    print(f"Computing ensemble mean suitability maps across 3 calibration runs for {species}...")
    print(f"{'='*80}")
    
    ensemble_out_dir = f'output/{REGION}/{calibration_years}/{data_scenario}/ensemble_mean/'
    os.makedirs(ensemble_out_dir, exist_ok=True)
    
    species_safe = species.replace(' ', '_')
    print(f"\n{species}:")
    
    suit_stack = np.stack(run_suitability_maps, axis=0)
    mean_suitability = np.nanmean(suit_stack, axis=0).astype(np.float32)
    std_suitability = np.nanstd(suit_stack, axis=0).astype(np.float32)
    
    suitable_areas = [(suit >= threshold).sum() for suit in run_suitability_maps]
    mean_suitable_area = np.mean(suitable_areas)
    std_suitable_area = np.std(suitable_areas)
    
    print(f"  Mean suitable area (pixels): {mean_suitable_area:.0f} ± {std_suitable_area:.0f}")
    print(f"  Cross-validation (training set, threshold={threshold:.2f}):")
    print(f"    Mean TSS: {np.mean([m['cv_mean_tss'] for m in run_ensemble_metrics]):.3f} ± {np.mean([m['cv_std_tss'] for m in run_ensemble_metrics]):.3f}")
    print(f"    Mean AUC: {np.mean([m['cv_mean_auc'] for m in run_ensemble_metrics]):.3f} ± {np.mean([m['cv_std_auc'] for m in run_ensemble_metrics]):.3f}")
    print(f"    Mean AUC-PR: {np.mean([m['cv_mean_auprc'] for m in run_ensemble_metrics]):.3f} ± {np.mean([m['cv_std_auprc'] for m in run_ensemble_metrics]):.3f}")
    print(f"  Test set (threshold={threshold:.2f}):")
    print(f"    Mean TSS: {np.mean([m['test_tss'] for m in run_ensemble_metrics]):.3f}")
    print(f"    Mean AUC: {np.mean([m['test_auc'] for m in run_ensemble_metrics]):.3f}")
    print(f"    Mean AUC-PR: {np.mean([m['test_auprc'] for m in run_ensemble_metrics]):.3f}")
    
    # Save ensemble mean suitability
    out_tiff = f"{ensemble_out_dir}/{species_safe}_calibration_ensemble_mean.tif"
    out_meta = ref_meta.copy()
    out_meta.update({'dtype': 'float32', 'count': 1, 'nodata': -9999.0})
    suitability_out = np.where(land_mask, mean_suitability, -9999.0)
    with rasterio.open(out_tiff, 'w', **out_meta) as dst:
        dst.write(suitability_out, 1)
    print(f"  Saved: {out_tiff}")
    
    # Save ensemble std dev
    out_std = f"{ensemble_out_dir}/{species_safe}_calibration_ensemble_std.tif"
    suitability_std_out = np.where(land_mask, std_suitability, -9999.0)
    with rasterio.open(out_std, 'w', **out_meta) as dst:
        dst.write(suitability_std_out, 1)
    print(f"  Saved: {out_std}")
    
    # Save ensemble metrics JSON
    ensemble_metrics_dict = {
        'species': species,
        'calibration_period': '1981-2010',
        'ensemble_type': 'mean_across_3_calibration_runs',
        'data_split_strategy': 'training 80% / test 20%',
        'mean_suitable_area_pixels': float(mean_suitable_area),
        'std_suitable_area_pixels': float(std_suitable_area),
        'cross_validation_metrics': {
            'method': '5-fold stratified k-fold on training set only',
            'threshold': round(float(threshold), 4),
            'mean_tss': round(float(np.mean([m['cv_mean_tss'] for m in run_ensemble_metrics])), 4),
            'std_tss': round(float(np.mean([m['cv_std_tss'] for m in run_ensemble_metrics])), 4),
            'mean_auc': round(float(np.mean([m['cv_mean_auc'] for m in run_ensemble_metrics])), 4),
            'std_auc': round(float(np.mean([m['cv_std_auc'] for m in run_ensemble_metrics])), 4),
            'mean_auprc': round(float(np.mean([m['cv_mean_auprc'] for m in run_ensemble_metrics])), 4),
            'std_auprc': round(float(np.mean([m['cv_std_auprc'] for m in run_ensemble_metrics])), 4)
        },
        'test_set_metrics': {
            'method': f'Fixed threshold {threshold:.2f} on unseen test data',
            'mean_tss': round(float(np.mean([m['test_tss'] for m in run_ensemble_metrics])), 4),
            'std_tss': round(float(np.std([m['test_tss'] for m in run_ensemble_metrics])), 4),
            'mean_auc': round(float(np.mean([m['test_auc'] for m in run_ensemble_metrics])), 4),
            'std_auc': round(float(np.std([m['test_auc'] for m in run_ensemble_metrics])), 4),
            'mean_auprc': round(float(np.mean([m['test_auprc'] for m in run_ensemble_metrics])), 4),
            'std_auprc': round(float(np.std([m['test_auprc'] for m in run_ensemble_metrics])), 4)
        },
        'per_run_metrics': run_ensemble_metrics
    }
    out_json = f"{ensemble_out_dir}/{species_safe}_calibration_ensemble_metrics.json"
    with open(out_json, 'w') as f:
        json.dump(ensemble_metrics_dict, f, indent=2)
    print(f"  Saved: {out_json}")
    
    # Plot ensemble mean
    left, bottom, right, top = array_bounds(height, width, ref_transform)
    fig, ax = plt.subplots(figsize=(18, 10))
    masked_suit = np.ma.masked_where(~land_mask, mean_suitability)
    im = ax.imshow(
        masked_suit, cmap='YlOrRd', vmin=0, vmax=1,
        extent=(left, right, bottom, top),
        aspect='auto', interpolation='nearest', origin='upper'
    )
    cbar = plt.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label('Mean Habitat Suitability', fontsize=11)
    mean_test_tss = np.mean([m['test_tss'] for m in run_ensemble_metrics])
    mean_test_auc = np.mean([m['test_auc'] for m in run_ensemble_metrics])
    mean_test_auprc = np.mean([m['test_auprc'] for m in run_ensemble_metrics])
    ax.set_title(
        f'{species} Calibration Ensemble Mean Suitability '
        f'(1981-2010, 3 runs, threshold={threshold:.2f}, Test TSS={mean_test_tss:.3f}, Test AUC={mean_test_auc:.3f}, Test AUC-PR={mean_test_auprc:.3f})',
        fontsize=14
    )
    ax.set_xlabel('Longitude', fontsize=10)
    ax.set_ylabel('Latitude', fontsize=10)
    plt.tight_layout()
    out_png = f"{ensemble_out_dir}/{species_safe}_calibration_ensemble_mean_map.png"
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_png}")
    
    print("\n" + "="*80)
    print("Calibration complete! Models saved to models/ directory.")
    print(f"Results saved in: output/{REGION}/{calibration_years}/{data_scenario}/")
    print("="*80)


def predict_mode(args):
    """
    PREDICTION PHASE: Load pre-calibrated models and generate suitability maps
    for 2071-2100 by applying the trained models to different environmental scenarios.
    
    This respects the principle that models should be calibrated on contemporaneous
    data, then projected to other periods without retraining.
    """
    # Prediction always uses 2071-2100
    pred_years = '2071-2100'
    data_scenario = args.scenario  # e.g., 'no_scenario', 'ssp126', 'ssp585'
    
    print("\n" + "="*80)
    print("PREDICTION MODE")
    print("="*80)
    print(f"Using pre-calibrated models to predict suitability for:")
    print(f"  Time period: {pred_years}")
    print(f"  Scenario: {data_scenario}")
    print(f"Models loaded from: models/{REGION}_{{species}}_calibrated.pkl\n")
    
    
    # Load environmental raster files from variable-specific folders
    print(f"Loading environmental data for {pred_years} ({data_scenario})...")
    raster_files = {}
    
    # List of bio variables and gdd5
    bio_vars = [f'bio{i:02d}' for i in range(1, 20)]  # bio01 to bio19
    all_vars = bio_vars + ['gdd5']
    
    for var_name in all_vars:
        raster_dir = f'data/chelsa_{var_name}_rasters/{data_scenario}/'
        
        if not os.path.exists(raster_dir):
            print(f"  Warning: Directory not found: {raster_dir}")
            continue
        
        # Find .tif file in this directory
        for filename in os.listdir(raster_dir):
            if filename.endswith('.tif') and not filename.endswith('.tif.aux.xml'):
                file_path = os.path.join(raster_dir, filename)
                raster_files[var_name] = file_path
                break
    
    print(f"Found {len(raster_files)} environmental variables: {sorted(raster_files.keys())}")
    
    if not raster_files:
        raise FileNotFoundError(f"No raster files found for {pred_years} ({data_scenario})")
    
    # Load observation data to get the single species
    merged_gdf = load_observation_data()
    species = merged_gdf['species'].unique()[0]  # Single species dataset
    print(f"Generating predictions for: {species}\n")
    
    # Build prediction grid and load environmental data
    grid_info = build_prediction_grid(raster_files)
    height, width = grid_info['height'], grid_info['width']
    ref_transform = grid_info['ref_transform']
    ref_meta = grid_info['ref_meta']
    band_arrays = grid_info['band_arrays']
    land_mask = grid_info['land_mask']
    
    # Create output directory
    out_dir = f'output/{REGION}/{pred_years}/{data_scenario}/'
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"{'='*80}")
    print(f"Generating predictions for {pred_years} ({data_scenario})")
    print(f"{'='*80}\n")
    
    print(f"\n{'='*80}")
    print(f"Predicting {species}")
    
    # Load pre-calibrated model
    model_bundle = load_calibrated_models(species, REGION)
    
    if model_bundle is None:
        print(f"  Error: calibrated model not found (run --mode calibrate first)")
        raise FileNotFoundError(f"Model file for {species} not found")
        
    # Extract model components
    models = model_bundle['models']
    tss_scores = model_bundle['tss_scores']
    auc_scores = model_bundle['auc_scores']
    auprc_scores = model_bundle['auprc_scores']
    aic_scores = model_bundle['aic_scores']
    scaler = model_bundle['scaler']
    env_vars_calibrated = model_bundle['env_vars']
    model_names = model_bundle['model_names']
    
    print(f"  Loaded models: {', '.join(model_names)}")
    print(f"  Calibration environment variables: {env_vars_calibrated}")
    
    # Extract environment data for prediction using CALIBRATED variable order
    band_arrays_pred = {v: band_arrays[v] for v in env_vars_calibrated if v in band_arrays}
    
    if len(band_arrays_pred) != len(env_vars_calibrated):
        print(f"  Warning: Prediction env missing some variables. Expected: {env_vars_calibrated}")
        print(f"  Available: {list(band_arrays_pred.keys())}")
    
    # Stack environmental data for prediction (using calibrated variable order)
    land_flat = land_mask.reshape(-1)
    X_stack = np.stack([band_arrays_pred[v] for v in env_vars_calibrated], axis=-1)
    X_flat = X_stack.reshape(-1, len(env_vars_calibrated))
    X_land = X_flat[land_flat]
    
    # Predict suitability using pre-calibrated models (NO NEW TRAINING)
    print(f"  Predicting suitability on {land_flat.sum():,} valid land pixels...")
    X_land_scaled = scaler.transform(X_land)  # Use calibrated scaler
    suitability_land = weighted_ensemble_predict(models, tss_scores, X_land_scaled)
    
    # Map back to 2D grid
    suitability_flat = np.full(height * width, np.nan, dtype=np.float32)
    suitability_flat[land_flat] = suitability_land
    suitability_map = suitability_flat.reshape(height, width)
    
    species_safe = species.replace(' ', '_')
    scenario_str = '' if data_scenario == 'no_scenario' else f'{data_scenario}_'
    
    # Save prediction GeoTIFF
    out_tiff = f"{out_dir}/{species_safe}_{pred_years}_{scenario_str}suitability.tif"
    out_meta = ref_meta.copy()
    out_meta.update({'dtype': 'float32', 'count': 1, 'nodata': -9999.0})
    suitability_out = np.where(land_mask, suitability_map, -9999.0).astype(np.float32)
    with rasterio.open(out_tiff, 'w', **out_meta) as dst:
        dst.write(suitability_out, 1)
    print(f"  Saved: {out_tiff}")
    
    # Save prediction metadata JSON
    results_dict = {
        'species': species,
        'prediction_period': pred_years,
        'scenario': data_scenario,
        'calibrated_on': '1981-2010 environment + recent observations',
        'ensemble_models': model_names,
        'ensemble_tss_scores': [round(float(t), 4) for t in tss_scores],
        'ensemble_auc_scores': [round(float(a), 4) for a in auc_scores],
        'ensemble_auprc_scores': [round(float(a), 4) for a in auprc_scores],
        'ensemble_aic_scores': [round(float(a), 4) for a in aic_scores],
        'prediction_variables': env_vars_calibrated,
        'note': 'Predictions generated using pre-trained models. No new training occurred.'
    }
    out_json = f"{out_dir}/{species_safe}_{pred_years}_{scenario_str}results.json"
    with open(out_json, 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f"  Saved: {out_json}")
    
    # Plot prediction map
    left, bottom, right, top = array_bounds(height, width, ref_transform)
    fig, ax = plt.subplots(figsize=(18, 10))
    masked_suit = np.ma.masked_where(~land_mask, suitability_map)
    im = ax.imshow(
        masked_suit, cmap='YlOrRd', vmin=0, vmax=1,
        extent=(left, right, bottom, top),
        aspect='auto', interpolation='nearest', origin='upper'
    )
    cbar = plt.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label('Habitat Suitability', fontsize=11)
    region_label = 'European' if EUROPE else 'Global'
    ax.set_title(
        f'{species} — {region_label} Predicted Suitability '
        f'({pred_years} {scenario_str}| Calibrated on 1981-2010)',
        fontsize=14
    )
    ax.set_xlabel('Longitude', fontsize=10)
    ax.set_ylabel('Latitude', fontsize=10)
    plt.tight_layout()
    out_png = f"{out_dir}/{species_safe}_{pred_years}_{scenario_str}suitability_map.png"
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_png}")
    
    print("\n" + "="*80)
    print(f"Predictions complete!")
    print(f"Results saved in: {out_dir}")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Species Distribution Modeling (SDM) with proper temporal calibration/projection workflow.'
    )
    parser.add_argument(
        '--mode',
        choices=['calibrate', 'predict'],
        required=True,
        help='Run mode: "calibrate" trains models on 1981-2010 data; "predict" applies trained models to 2071-2100'
    )
    parser.add_argument(
        '--scenario',
        default='ssp126',
        choices=['no_scenario', 'ssp126', 'ssp585'],
        help='Climate scenario for PREDICTION mode (ignored in calibrate mode). Default: ssp126'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.2,
        help='Probability threshold for converting suitability to binary predictions. Default: 0.5'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'calibrate':
        calibrate_mode(args)
    else:  # predict
        predict_mode(args)


if __name__ == '__main__':
    main()
