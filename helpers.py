import numpy as np
from rasterio.transform import rowcol as raster_rowcol
from scipy.spatial.distance import cdist
from rasterio.windows import from_bounds as window_from_bounds
from rasterio.transform import array_bounds, Affine
from rasterio.enums import Resampling
import rasterio
from sklearn.metrics import roc_auc_score, average_precision_score
from config import EUROPE, EUROPE_BBOX, SCALE

rng = np.random.default_rng(42)


def build_prediction_grid(raster_files):
    """
    Build prediction grid from reference raster. Returns grid metadata and loaded band arrays.
    """
    print("\nBuilding prediction grid...")
    
    with rasterio.open(list(raster_files.values())[0]) as src:
        if EUROPE:
            window = window_from_bounds(*EUROPE_BBOX, transform=src.transform).round_lengths().round_offsets()
        else:
            window = rasterio.windows.Window(0, 0, src.width, src.height)
        win_h, win_w = int(window.height), int(window.width)
        height, width = win_h // SCALE, win_w // SCALE
        win_transform = src.window_transform(window)
        ref_transform = win_transform * Affine.scale(SCALE, SCALE)
        ref_meta = src.meta.copy()
        ref_meta.update({'height': height, 'width': width, 'transform': ref_transform})
        
        print(f"  Prediction grid size: {width} x {height} pixels (scaled by factor {SCALE})")
        print(f"  Prediction grid bounds: {array_bounds(height, width, ref_transform)}")
        print(f"  Prediction grid CRS: {src.crs}")
        print(f"  Prediction grid resolution: {ref_transform.a:.6f} x {abs(ref_transform.e):.6f} degrees/pixel")
    
    # Read all env bands into memory
    print("  Reading environmental rasters...")
    band_arrays = {}
    nodata_vals = {}
    for name, path in raster_files.items():
        with rasterio.open(path) as src:
            band_arrays[name] = src.read(1, window=window, out_shape=(height, width), resampling=Resampling.average).astype(np.float32)
            nodata_vals[name] = src.nodata
    
    # Build masks
    print("  Building invalid pixel mask...")
    invalid_mask = np.zeros((height, width), dtype=bool)
    for arr in band_arrays.values():
        invalid_mask |= ~np.isfinite(arr)
    
    print("  Loading pre-computed sea mask...")
    with rasterio.open('cleaned_env_data/sea_mask.tif') as src:
        sea_mask = src.read(1, window=window, out_shape=(height, width), resampling=Resampling.nearest).astype(bool)
    
    print(f"  Sea pixels: {sea_mask.sum():,}")
    land_mask = ~sea_mask & ~invalid_mask
    print(f"  Valid land pixels after masking: {land_mask.sum():,}")
    
    return {
        'height': height,
        'width': width,
        'ref_transform': ref_transform,
        'ref_meta': ref_meta,
        'band_arrays': band_arrays,
        'land_mask': land_mask,
        'window': window
    }


def spatial_thin(gdf, grid_size_deg=0.05):
    """
    Thin observations by grid cells (1 observation per grid cell).
    Works entirely in WGS84.
    
    Args:
        gdf: GeoDataFrame with points in WGS84 (EPSG:4326)
        grid_size_deg: Grid cell size in degrees (default 0.05° ≈ 5 km at equator)
    
    Returns:
        Thinned GeoDataFrame with one point per grid cell
    """
    if len(gdf) == 0:
        return gdf
    
    x = gdf.geometry.x.values
    y = gdf.geometry.y.values
    
    # Create grid cell identifiers
    cell_x = (x // grid_size_deg).astype(int)
    cell_y = (y // grid_size_deg).astype(int)
    
    gdf = gdf.copy()
    gdf['_cell_id'] = [f"{cx}_{cy}" for cx, cy in zip(cell_x, cell_y)]
    
    # For each cell, randomly select one point
    thinned = gdf.groupby('_cell_id').apply(
        lambda group: group.iloc[rng.integers(0, len(group))],
        include_groups=False
    ).reset_index(drop=False)
    
    # Drop the _cell_id column
    thinned = thinned.drop(columns=['_cell_id'])
    
    return thinned


def create_buffer_mask(presence_coords, ref_transform, height, width, buffer_deg=0.1):
    """
    Create a binary mask of pixels to EXCLUDE as background (within buffer of presence points).
    
    This prevents data leakage by excluding background pixels too close to presence observations.
    Background pixels near presence points don't represent environmental conditions the species
    genuinely avoids—they're just nearby land. Using them biases the model.
    
    Args:
        presence_coords: Array of (lon, lat) coordinates of thinned presence points, shape (n_pres, 2)
        ref_transform: Rasterio transform of the prediction grid
        height: Height of prediction grid in pixels
        width: Width of prediction grid in pixels
        buffer_deg: Buffer radius in degrees (default 0.1° ≈ 11 km at equator = ~10 km)
    
    Returns:
        Boolean array of shape (height, width) where True = exclude from background sampling
    """
    # Create grid of all pixel coordinates in (lon, lat)
    cols = np.arange(width)
    rows = np.arange(height)
    col_grid, row_grid = np.meshgrid(cols, rows)
    
    # Convert row/col to lon/lat using rasterio transform
    lons = ref_transform.c + col_grid * ref_transform.a
    lats = ref_transform.f + row_grid * ref_transform.e
    
    pixel_coords = np.column_stack([lons.ravel(), lats.ravel()])  # shape (n_pixels, 2)
    
    # Compute distances from each pixel to nearest presence point (in degrees)
    distances = cdist(pixel_coords, presence_coords, metric='euclidean').min(axis=1)  # shape (n_pixels,)
    
    # Create exclusion mask: True where distance < buffer
    buffer_mask = (distances < buffer_deg).reshape(height, width)
    
    return buffer_mask


def filter_collinear_variables(band_arrays, land_mask, env_vars, corr_threshold=0.7, vif_threshold=10.0):
    """Remove highly correlated variables and variables with excessive VIF."""
    print(f"  Checking collinearity (Pearson r > {corr_threshold}) and VIF > {vif_threshold}...")
    land_flat = land_mask.reshape(-1)
    X_temp = np.stack([band_arrays[v] for v in env_vars], axis=-1).reshape(-1, len(env_vars))
    X_temp_valid = X_temp[land_flat]

    if X_temp_valid.shape[0] < 2:
        print("  Not enough valid pixels for collinearity checks; keeping all variables.")
        return env_vars, band_arrays

    corr_matrix = np.corrcoef(X_temp_valid.T)
    vars_to_remove = set()

    for i in range(len(env_vars)):
        if env_vars[i] in vars_to_remove:
            continue
        for j in range(i + 1, len(env_vars)):
            if env_vars[j] in vars_to_remove:
                continue
            if abs(corr_matrix[i, j]) > corr_threshold:
                vars_to_remove.add(env_vars[j])
                print(f"    Removing '{env_vars[j]}' (corr={corr_matrix[i, j]:.3f} with '{env_vars[i]}')")

    remaining_vars = [v for v in env_vars if v not in vars_to_remove]

    if len(remaining_vars) > 1:
        current_vars = list(remaining_vars)
        while len(current_vars) > 1:
            X_current = np.stack([band_arrays[v] for v in current_vars], axis=-1).reshape(-1, len(current_vars))
            X_current_valid = X_current[land_flat]

            vifs = []
            for idx in range(len(current_vars)):
                target = X_current_valid[:, idx]
                others = np.delete(X_current_valid, idx, axis=1)
                if others.shape[1] == 0:
                    vifs.append(0.0)
                    continue

                design = np.column_stack([np.ones(len(target)), others])
                coeffs, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
                y_pred = design @ coeffs
                ss_res = np.sum((target - y_pred) ** 2)
                ss_tot = np.sum((target - np.mean(target)) ** 2)
                if ss_tot <= 0:
                    r2 = 0.0
                else:
                    r2 = 1 - (ss_res / ss_tot)
                vif = np.inf if r2 >= 1 else 1 / (1 - r2)
                vifs.append(vif)

            max_vif = float(np.max(vifs)) if len(vifs) else 0.0
            max_idx = int(np.argmax(vifs)) if len(vifs) else -1
            if not np.isfinite(max_vif) or max_vif <= vif_threshold:
                break

            var_to_remove = current_vars[max_idx]
            print(f"    Removing '{var_to_remove}' (VIF={max_vif:.2f} > {vif_threshold})")
            current_vars.pop(max_idx)

        remaining_vars = current_vars

    for var in env_vars:
        if var not in remaining_vars:
            del band_arrays[var]

    env_vars = remaining_vars
    print(f"  Remaining variables: {env_vars}")
    return env_vars, band_arrays

def calculate_adaptive_background_size(n_presence):
    """
    Calculate adaptive background sample size based on presence point count.
    
    Rationale: Large datasets (n_presence > 100) benefit from many background points for
    stable model estimation. Small datasets (n_presence < 30) produce unstable models when
    background >> presence; reducing background improves class balance and model stability.
    
    Args:
        n_presence: Number of presence points after thinning
    
    Returns:
        Recommended number of background points to sample (int)
    """
    if n_presence < 10:
        # Minimal data: 1:10 presence:background ratio
        return min(100, n_presence * 10)
    elif n_presence < 30:
        # Small data: 1:20 ratio
        return min(600, n_presence * 20)
    elif n_presence < 100:
        # Medium data: 1:50 ratio
        return min(5000, n_presence * 50)
    else:
        # Large data: use full 10K
        return 10_000


def extract_env_at_points(gdf, env_vars, ref_transform, height, width, band_arrays):
    xs = gdf.geometry.x.values
    ys = gdf.geometry.y.values
    results = np.full((len(gdf), len(env_vars)), np.nan, dtype=np.float32)
    valid = np.isfinite(xs) & np.isfinite(ys)
    valid_idx = np.where(valid)[0]
    rows, cols = raster_rowcol(ref_transform, xs[valid], ys[valid])
    for i, (r, c) in enumerate(zip(rows, cols)):
        r, c = int(r), int(c)
        if 0 <= r < height and 0 <= c < width:
            for j, var in enumerate(env_vars):
                results[valid_idx[i], j] = band_arrays[var][r, c]
    return results


def calculate_tss(y_true, y_pred_binary):
    """
    Calculate True Skill Statistic (TSS).
    TSS = (TP*TN - FP*FN) / ((TP+FN)*(FP+TN))
    Ranges from -1 to 1; values > 0.4 indicate good model performance.
    
    Args:
        y_true: Binary true labels (0 or 1)
        y_pred_binary: Binary predictions (0 or 1)
    
    Returns:
        TSS score (float)
    """
    TP = np.sum((y_pred_binary == 1) & (y_true == 1))
    TN = np.sum((y_pred_binary == 0) & (y_true == 0))
    FP = np.sum((y_pred_binary == 1) & (y_true == 0))
    FN = np.sum((y_pred_binary == 0) & (y_true == 1))
    
    denominator = (TP + FN) * (FP + TN)
    if denominator == 0:
        return 0.0
    
    tss = (TP * TN - FP * FN) / denominator
    return tss


def calculate_auprc(y_true, y_score):
    """
    Calculate Area Under the Precision-Recall Curve (AUC-PR).

    Args:
        y_true: Binary true labels (0 or 1)
        y_score: Continuous predicted scores/probabilities

    Returns:
        AUC-PR score (float)
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return np.nan

    return float(average_precision_score(y_true, y_score))


def calculate_aic(y_true, y_score, n_params):
    """
    Calculate Akaike Information Criterion (AIC) from binomial log-likelihood.

    This uses the fitted class probabilities as the likelihood contribution and a
    simple parameter-count approximation for the model complexity term.

    Args:
        y_true: Binary true labels (0 or 1)
        y_score: Continuous predicted scores/probabilities
        n_params: Approximate number of fitted parameters/complexity term

    Returns:
        AIC score (float)
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return np.nan

    y_score = np.clip(y_score, 1e-15, 1 - 1e-15)
    log_likelihood = np.sum(y_true * np.log(y_score) + (1 - y_true) * np.log(1 - y_score))
    return float(2 * max(int(n_params), 1) - 2 * log_likelihood)


def count_model_parameters(model):
    """Estimate the effective parameter count for AIC calculation."""
    if hasattr(model, 'coef_') and getattr(model, 'coef_', None) is not None:
        return int(np.size(model.coef_)) + int(np.size(getattr(model, 'intercept_', [])))

    if hasattr(model, 'estimators_'):
        try:
            return int(sum(getattr(tree.tree_, 'node_count', 0) for tree in model.estimators_.ravel()))
        except Exception:
            pass

    if hasattr(model, 'support_'):
        return int(np.size(model.support_)) + 1

    return int(getattr(model, 'n_features_in_', 1)) + 1


def train_ensemble_models(X_train, y_train, X_test, y_test, random_state=42):
    """
    Train 4 different model types and rank by TSS score.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        random_state: Random seed for reproducibility
    
    Returns:
        List of tuples: [(model, model_name, tss, auc, auprc, aic), ...] sorted by TSS descending
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC
    
    results = []
    
    # 1. Logistic Regression
    lr = LogisticRegression(max_iter=1000, C=1.0, class_weight='balanced', random_state=random_state)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    y_prob_lr = lr.predict_proba(X_test)[:, 1]
    tss_lr = calculate_tss(y_test, y_pred_lr)
    auc_lr = roc_auc_score(y_test, y_prob_lr)
    auprc_lr = calculate_auprc(y_test, y_prob_lr)
    aic_lr = calculate_aic(y_test, y_prob_lr, count_model_parameters(lr))
    results.append((lr, 'LogisticRegression', tss_lr, auc_lr, auprc_lr, aic_lr))
    
    # 2. Random Forest
    rf = RandomForestClassifier(n_estimators=100, max_depth=15, min_samples_split=5,
                                class_weight='balanced', random_state=random_state, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    y_prob_rf = rf.predict_proba(X_test)[:, 1]
    tss_rf = calculate_tss(y_test, y_pred_rf)
    auc_rf = roc_auc_score(y_test, y_prob_rf)
    auprc_rf = calculate_auprc(y_test, y_prob_rf)
    aic_rf = calculate_aic(y_test, y_prob_rf, count_model_parameters(rf))
    results.append((rf, 'RandomForest', tss_rf, auc_rf, auprc_rf, aic_rf))
    
    # 3. Support Vector Machine (RBF kernel)
    svm = SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=random_state)
    svm.fit(X_train, y_train)
    y_pred_svm = svm.predict(X_test)
    y_prob_svm = svm.predict_proba(X_test)[:, 1]
    tss_svm = calculate_tss(y_test, y_pred_svm)
    auc_svm = roc_auc_score(y_test, y_prob_svm)
    auprc_svm = calculate_auprc(y_test, y_prob_svm)
    aic_svm = calculate_aic(y_test, y_prob_svm, count_model_parameters(svm))
    results.append((svm, 'SVM', tss_svm, auc_svm, auprc_svm, aic_svm))
    
    # 4. Gradient Boosting
    gb = GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.1,
                                    random_state=random_state)
    gb.fit(X_train, y_train)
    y_pred_gb = gb.predict(X_test)
    y_prob_gb = gb.predict_proba(X_test)[:, 1]
    tss_gb = calculate_tss(y_test, y_pred_gb)
    auc_gb = roc_auc_score(y_test, y_prob_gb)
    auprc_gb = calculate_auprc(y_test, y_prob_gb)
    aic_gb = calculate_aic(y_test, y_prob_gb, count_model_parameters(gb))
    results.append((gb, 'GradientBoosting', tss_gb, auc_gb, auprc_gb, aic_gb))
    
    # Sort by TSS score (descending)
    results.sort(key=lambda x: x[2], reverse=True)
    
    return results


def ensemble_predict(models, X):
    """
    Generate ensemble predictions by averaging probabilities across models.
    
    Args:
        models: List of trained scikit-learn models
        X: Features to predict on
    
    Returns:
        Array of ensemble probabilities (mean across models)
    """
    probs = []
    for model in models:
        y_prob = model.predict_proba(X)[:, 1]
        probs.append(y_prob)
    
    ensemble_prob = np.mean(probs, axis=0).astype(np.float32)
    return ensemble_prob


def weighted_ensemble_predict(models, tss_scores, X):
    """
    Generate ensemble predictions using TSS-weighted averaging.
    
    Models with higher TSS (better individual performance) get higher weight in the ensemble.
    This respects the fact that some models perform better than others on the validation set,
    and those better-performing models should have more influence on final predictions.
    
    Particularly valuable for rare species (< 20 presence points) where random variation
    in model performance is high: weighting prevents poor models from degrading ensemble quality.
    
    Args:
        models: List of trained scikit-learn models (same order as tss_scores)
        tss_scores: List of TSS scores corresponding to each model (float)
        X: Features to predict on (n_samples, n_features)
    
    Returns:
        Array of ensemble probabilities (weighted average across models) shape (n_samples,)
    """
    tss_array = np.array(tss_scores, dtype=np.float32)
    
    # Shift TSS scores by minimum to ensure all weights are positive
    # (handles cases where some models have negative TSS, which is valid in SDM)
    tss_shifted = tss_array - np.min(tss_array) + 1e-6
    
    # Normalize weights to sum to 1 for interpretability
    weights = tss_shifted / np.sum(tss_shifted)
    
    # Get class probabilities from each model
    probs = []
    for model in models:
        y_prob = model.predict_proba(X)[:, 1]
        probs.append(y_prob)
    
    # Compute weighted average of probabilities across models
    probs_array = np.array(probs, dtype=np.float32)  # shape: (n_models, n_samples)
    ensemble_prob = np.average(probs_array, axis=0, weights=weights).astype(np.float32)
    
    return ensemble_prob


# ============ MODEL PERSISTENCE (JOBLIB) ============

def save_calibrated_models(species, region, models_dict, scaler, env_vars, model_names):
    """
    Save calibrated models, scaler, and metadata to disk using joblib.
    
    The models_dict is a collection of fitted models from train_ensemble_models(),
    along with preprocessing (scaler) and configuration (env_vars, model names).
    This allows later runs to load and reuse the models without retraining.
    
    Args:
        species: Species name (string)
        region: Region name (e.g., 'europe', 'global')
        models_dict: Dict with keys 'models' (list), 'tss_scores' (list), 'auc_scores' (list), 'auprc_scores' (list), 'aic_scores' (list)
        scaler: Fitted StandardScaler object
        env_vars: List of environmental variable names used in training
        model_names: List of model class names (e.g., ['LogisticRegression', 'RandomForest', ...])
    
    Returns:
        Path to saved model file
    """
    import joblib
    import os
    
    os.makedirs('models', exist_ok=True)
    
    species_safe = species.replace(' ', '_')
    model_path = f'models/{species_safe}_{region}_calibrated.pkl'
    
    # Bundle all necessary objects for prediction
    model_bundle = {
        'species': species,
        'region': region,
        'models': models_dict['models'],
        'tss_scores': models_dict['tss_scores'],
        'auc_scores': models_dict['auc_scores'],
        'auprc_scores': models_dict['auprc_scores'],
        'aic_scores': models_dict['aic_scores'],
        'scaler': scaler,
        'env_vars': env_vars,
        'model_names': model_names,
        'calibration_note': 'Calibrated on 1981-2010 environment + recent observations'
    }
    
    joblib.dump(model_bundle, model_path, compress=3)
    print(f"    Model saved: {model_path}")
    return model_path


def load_calibrated_models(species, region):
    """
    Load calibrated models, scaler, and metadata from disk using joblib.
    
    Args:
        species: Species name (string)
        region: Region name (e.g., 'europe', 'global')
    
    Returns:
        Dict with keys: 'models', 'tss_scores', 'auc_scores', 'auprc_scores', 'aic_scores', 'scaler', 'env_vars', 'model_names'
        OR None if model file not found
    """
    import joblib
    import os
    
    species_safe = species.replace(' ', '_')
    model_path = f'models/{species_safe}_{region}_calibrated.pkl'
    
    if not os.path.exists(model_path):
        print(f"    Model file not found: {model_path}")
        return None
    
    model_bundle = joblib.load(model_path)
    print(f"    Model loaded: {model_path}")
    return model_bundle