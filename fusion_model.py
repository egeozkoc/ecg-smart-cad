import numpy as np
import pandas as pd
import joblib
import torch
from torch.utils.data import DataLoader, TensorDataset
from scipy import signal
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score, 
                              precision_score, recall_score, f1_score, roc_curve, confusion_matrix)


def get_rf_data(num_features=None):
    """Load feature-based data for RF model.
    
    Args:
        num_features: Number of top features to use. If None, uses all features from features.txt
    
    Returns:
        x_val: validation features
        y_val: validation labels
        x_test: test features
        y_test: test labels
    """
    # Use the same features file as training
    features = pd.read_csv('results/features.csv')

    # Load feature names based on num_features parameter
    if num_features is None:
        # Load all features from features.txt
        feature_names = pd.read_csv('features.txt', header=None)[0].tolist()
    else:
        # Load top N features from feature importance ranking
        feature_names = pd.read_csv('rf_feature_importance_cross_validation.csv')['feature'].tolist()[:num_features]

    val_df = pd.read_csv('val_set.csv')
    test_df = pd.read_csv('test_set.csv')
    y_val = val_df['label'].to_numpy()
    y_test = test_df['label'].to_numpy()
    val_ids = val_df['ID'].to_list()
    test_ids = test_df['ID'].to_list()

    # Get validation data
    x_val = []
    for id in val_ids:
        features_id = features[features['Unnamed: 0'] == id]
        features_id = features_id.drop(columns=['Unnamed: 0'])
        # Keep only the desired features and enforce ordering
        features_id = features_id[feature_names]
        # Convert to numeric, coercing errors to NaN, then fill NaN with 0
        features_id = features_id.apply(pd.to_numeric, errors='coerce').fillna(0)
        x_val.append(features_id.to_numpy())
    x_val = np.concatenate(x_val, axis=0).astype(float)  # Ensure float dtype
    print(f"RF - Val set: {len(x_val)} samples loaded")

    # Get test data
    x_test = []
    for id in test_ids:
        features_id = features[features['Unnamed: 0'] == id]
        features_id = features_id.drop(columns=['Unnamed: 0'])
        # Keep only the desired features and enforce ordering
        features_id = features_id[feature_names]
        # Convert to numeric, coercing errors to NaN, then fill NaN with 0
        features_id = features_id.apply(pd.to_numeric, errors='coerce').fillna(0)
        x_test.append(features_id.to_numpy())
    x_test = np.concatenate(x_test, axis=0).astype(float)  # Ensure float dtype
    print(f"RF - Test set: {len(x_test)} samples loaded")

    return x_val, y_val, x_test, y_test


def get_dl_data(path='cad_dataset_preprocessed/'):
    """Load ECG data for deep learning model.
    
    Args:
        path: path to preprocessed data
        
    Returns:
        x_val: validation ECG data
        y_val: validation labels
        x_test: test ECG data
        y_test: test labels
    """
    val_df = pd.read_csv('val_set.csv')
    test_df = pd.read_csv('test_set.csv')
    val_outcomes = val_df['label'].to_numpy()
    test_outcomes = test_df['label'].to_numpy()
    val_ids = val_df['ID'].to_list()
    test_ids = test_df['ID'].to_list()

    # Get validation data
    val_data = []
    for id in val_ids:
        ecg = np.load(path + id + '.npy', allow_pickle=True).item()
        ecg = ecg['waveforms']['ecg_median']
        ecg = ecg[:, 150:-50]
        ecg = signal.resample(ecg, 200, axis=1)
        max_val = np.max(np.abs(ecg), axis=1, keepdims=True)
        
        # Replace zeros with 1 to avoid division by zero
        max_val = np.where(max_val == 0, 1, max_val)
        ecg = ecg / max_val
        
        val_data.append(ecg)
    
    val_data = np.array(val_data)
    print(f"DL - Val set: {len(val_data)} files loaded")

    # Get test data
    test_data = []
    for id in test_ids:
        ecg = np.load(path + id + '.npy', allow_pickle=True).item()
        ecg = ecg['waveforms']['ecg_median']
        ecg = ecg[:, 150:-50]
        ecg = signal.resample(ecg, 200, axis=1)
        max_val = np.max(np.abs(ecg), axis=1, keepdims=True)
        
        # Replace zeros with 1 to avoid division by zero
        max_val = np.where(max_val == 0, 1, max_val)
        ecg = ecg / max_val
        
        test_data.append(ecg)
    
    test_data = np.array(test_data)
    print(f"DL - Test set: {len(test_data)} files loaded")

    return val_data, val_outcomes, test_data, test_outcomes


def get_dl_predictions(model, device, dataloader):
    """Get predictions from deep learning model."""
    model.eval()
    all_pred_probs = []

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(dataloader):
            x = x.to(device)
            x = torch.unsqueeze(x, 1)

            with torch.amp.autocast(device.type, enabled=True):
                logits = model(x)
                probs = torch.softmax(logits, dim=-1)

            all_pred_probs.append(probs[:, 1].detach().cpu().to(torch.float32).numpy())
            
            # Print progress every 5 batches
            if (batch_idx + 1) % 5 == 0:
                processed = (batch_idx + 1) * dataloader.batch_size
                print(f"  Processed {processed} samples...", flush=True)

    y_prob_pos = np.concatenate(all_pred_probs, axis=0)
    
    # Check for NaN/Inf in predictions
    if np.any(np.isnan(y_prob_pos)) or np.any(np.isinf(y_prob_pos)):
        print("WARNING: NaN or Inf detected in DL predictions!")
        print(f"  NaN count: {np.sum(np.isnan(y_prob_pos))}")
        print(f"  Inf count: {np.sum(np.isinf(y_prob_pos))}")
        print("  Replacing NaN/Inf with 0.5 (neutral prediction)")
        y_prob_pos = np.nan_to_num(y_prob_pos, nan=0.5, posinf=1.0, neginf=0.0)
    
    return y_prob_pos


def find_thresholds_from_val(y_true, y_prob_pos):
    """Compute thresholds based on validation data.
    - Rule-out: highest threshold achieving sensitivity >= 0.90
    - Rule-in: lowest threshold achieving PPV >= 0.85
    - F1: threshold maximizing F1
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob_pos)

    # Rule-out threshold (maximize threshold while sens >= 0.90)
    rule_out_idx = np.where(tpr >= 0.90)[0]
    rule_out_thresh = thresholds[rule_out_idx[0]] if len(rule_out_idx) > 0 else None

    # Rule-in threshold (first threshold with PPV >= 0.85)
    ppv_values = []
    for t in thresholds:
        preds_t = (y_prob_pos >= t).astype(int)
        if preds_t.sum() == 0:
            ppv_values.append(0.0)
        else:
            ppv_values.append(precision_score(y_true, preds_t, zero_division=0))
    ppv_values = np.array(ppv_values)
    rule_in_idx = np.where(ppv_values >= 0.85)[0]
    rule_in_thresh = thresholds[rule_in_idx[-1]] if len(rule_in_idx) > 0 else None

    # F1 threshold
    f1_values = []
    for t in thresholds:
        preds_t = (y_prob_pos >= t).astype(int)
        f1_values.append(f1_score(y_true, preds_t, zero_division=0))
    f1_values = np.array(f1_values)
    f1_thresh = thresholds[np.argmax(f1_values)]

    return rule_out_thresh, rule_in_thresh, f1_thresh


def compute_metrics(y_true, y_prob_pos, threshold):
    """Compute classification metrics at a given threshold."""
    preds = (y_prob_pos >= threshold).astype(int)
    sens = recall_score(y_true, preds, zero_division=0)
    spec = recall_score(y_true, preds, pos_label=0, zero_division=0)
    acc = accuracy_score(y_true, preds)
    ppv = precision_score(y_true, preds, pos_label=1, zero_division=0)
    npv = precision_score(y_true, preds, pos_label=0, zero_division=0)
    f1 = f1_score(y_true, preds, zero_division=0)
    auc = roc_auc_score(y_true, y_prob_pos)
    ap = average_precision_score(y_true, y_prob_pos)
    return {
        'sens': sens,
        'spec': spec,
        'acc': acc,
        'ppv': ppv,
        'npv': npv,
        'f1': f1,
        'auc': auc,
        'ap': ap,
    }


def bootstrap_ci_val(y_true, y_prob_pos, n_boot=200, seed=42):
    """Bootstrap 95% CIs for validation metrics, recomputing F1 threshold per resample."""
    rng = np.random.default_rng(seed)
    metrics = {'auc': [], 'ap': [], 'f1': [], 'sens': [], 'spec': [], 'acc': [], 'ppv': [], 'npv': []}
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        y_b = y_true[idx]
        p_b = y_prob_pos[idx]
        _, _, f1_t = find_thresholds_from_val(y_b, p_b)
        m = compute_metrics(y_b, p_b, f1_t)
        for k in metrics.keys():
            metrics[k].append(m[k])
    ci = {}
    for k, vals in metrics.items():
        vals = np.array(vals)
        ci[k] = (np.percentile(vals, 2.5), np.percentile(vals, 97.5))
    return ci


def bootstrap_ci_test(y_true, y_prob_pos, threshold=0.5, n_boot=200, seed=42):
    """Bootstrap 95% CIs for test metrics at a fixed threshold."""
    rng = np.random.default_rng(seed)
    metrics = {'auc': [], 'ap': [], 'f1': [], 'sens': [], 'spec': [], 'acc': [], 'ppv': [], 'npv': []}
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        y_b = y_true[idx]
        p_b = y_prob_pos[idx]
        m = compute_metrics(y_b, p_b, threshold)
        for k in metrics.keys():
            metrics[k].append(m[k])
    ci = {}
    for k, vals in metrics.items():
        vals = np.array(vals)
        ci[k] = (np.percentile(vals, 2.5), np.percentile(vals, 97.5))
    return ci


def format_with_ci(value, ci_tuple):
    """Format like 0.945 [0.913,0.972]."""
    low, high = ci_tuple
    return f"{value:.3f} [{low:.3f},{high:.3f}]"


def r3(x):
    """Round to 3 decimals or return None."""
    return None if x is None else float(f"{x:.3f}")


def plot_val_test_roc(y_val_true, y_val_prob, y_test_true, y_test_prob, model_name, save_path):
    """Plot validation and test ROC curves on the same figure and save."""
    fpr_v, tpr_v, _ = roc_curve(y_val_true, y_val_prob)
    auc_v = roc_auc_score(y_val_true, y_val_prob)

    fpr_t, tpr_t, _ = roc_curve(y_test_true, y_test_prob)
    auc_t = roc_auc_score(y_test_true, y_test_prob)

    plt.figure(figsize=(10, 8))
    plt.plot(fpr_t, tpr_t, color='tab:blue', lw=2, label=f'Test ROC (AUC = {auc_t:.3f})')
    plt.plot(fpr_v, tpr_v, color='tab:orange', lw=2, linestyle='--', label=f'Val ROC (AUC = {auc_v:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle=':')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves - {model_name}')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Val+Test ROC curves saved to: {save_path}")


def plot_confusion_matrix(cm, model_name, save_path):
    """Plot confusion matrix and save it."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No CAD', 'CAD'], 
                yticklabels=['No CAD', 'CAD'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to: {save_path}")


def optimize_fusion_weights(y_val, rf_val_probs, dl_val_probs):
    """Find optimal fusion weights using grid search on validation set.
    
    Args:
        y_val: validation labels
        rf_val_probs: RF model probabilities
        dl_val_probs: DL model probabilities
        
    Returns:
        best_weight_rf: optimal weight for RF model
        best_auc: best AUC score
    """
    best_auc = 0
    best_weight_rf = 0.5
    
    # Grid search over weights
    for weight_rf in np.arange(0, 1.05, 0.05):
        weight_dl = 1 - weight_rf
        fused_probs = weight_rf * rf_val_probs + weight_dl * dl_val_probs
        auc = roc_auc_score(y_val, fused_probs)
        
        if auc > best_auc:
            best_auc = auc
            best_weight_rf = weight_rf
    
    return best_weight_rf, best_auc


def main():
    print('=' * 80)
    print('FUSION MODEL EVALUATION')
    print('Combining Random Forest and Deep Learning Models')
    print('=' * 80)
    
    # ============ Configuration ============
    # Set num_features to match your RF model:
    # - None: uses all features (for models trained on all features)
    # - Integer (e.g., 100): uses top N features (for models trained on selected features)
    num_features = 100  # Using the best performing RF model with 100 features
    
    # ============ Define Model Paths ============
    if num_features is None:
        rf_model_path = 'rf_models/best_rf_all_features.pkl'
    else:
        rf_model_path = f'rf_models/best_rf_selected_features_{num_features}.pkl'
    
    dl_model_path = 'models/transfer_learning_CAD__2025-11-17-16-51-30.pt'
    
    print(f'\nConfiguration:')
    print(f'  RF Model: {rf_model_path}')
    print(f'  DL Model: {dl_model_path}')
    print(f'  Number of features: {num_features if num_features else "all"}')
    
    # ============ Load Data ============
    print('\n[1/6] Loading data...')
    print('\nLoading RF data:')
    x_val_rf, y_val, x_test_rf, y_test = get_rf_data(num_features)
    
    print('\nLoading DL data:')
    x_val_dl, y_val_dl, x_test_dl, y_test_dl = get_dl_data()
    
    # Verify labels match
    assert np.all(y_val == y_val_dl), "Validation labels mismatch between RF and DL!"
    assert np.all(y_test == y_test_dl), "Test labels mismatch between RF and DL!"
    print('✓ Data loaded successfully')
    
    # ============ Load Models ============
    print('\n[2/6] Loading models...')
    
    # Load RF model
    print(f'Loading RF model from: {rf_model_path}')
    rf_model = joblib.load(rf_model_path)
    print('✓ RF model loaded')
    
    # Load DL model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    print(f'Loading DL model from: {dl_model_path}')
    dl_model = torch.load(dl_model_path, map_location=device, weights_only=False)
    dl_model.eval()
    print('✓ DL model loaded')
    
    # ============ Get Predictions ============
    print('\n[3/6] Getting predictions from individual models...')
    
    # Check for NaN in input data
    print('\nChecking input data for NaN/Inf...')
    print(f'  RF Val data - NaN: {np.sum(np.isnan(x_val_rf))}, Inf: {np.sum(np.isinf(x_val_rf))}')
    print(f'  RF Test data - NaN: {np.sum(np.isnan(x_test_rf))}, Inf: {np.sum(np.isinf(x_test_rf))}')
    print(f'  DL Val data - NaN: {np.sum(np.isnan(x_val_dl))}, Inf: {np.sum(np.isinf(x_val_dl))}')
    print(f'  DL Test data - NaN: {np.sum(np.isnan(x_test_dl))}, Inf: {np.sum(np.isinf(x_test_dl))}')
    
    # RF predictions
    print('\nRF predictions:')
    print('  Validation set...')
    rf_val_probs = rf_model.predict_proba(x_val_rf)[:, 1]
    print('  Test set...')
    rf_test_probs = rf_model.predict_proba(x_test_rf)[:, 1]
    
    # Check RF predictions for NaN
    if np.any(np.isnan(rf_val_probs)) or np.any(np.isinf(rf_val_probs)):
        print(f'  WARNING: RF Val predictions - NaN: {np.sum(np.isnan(rf_val_probs))}, Inf: {np.sum(np.isinf(rf_val_probs))}')
    if np.any(np.isnan(rf_test_probs)) or np.any(np.isinf(rf_test_probs)):
        print(f'  WARNING: RF Test predictions - NaN: {np.sum(np.isnan(rf_test_probs))}, Inf: {np.sum(np.isinf(rf_test_probs))}')
    print('✓ RF predictions complete')
    
    # DL predictions
    print('\nDL predictions:')
    x_val_dl_tensor = torch.tensor(x_val_dl, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)
    val_loader = DataLoader(TensorDataset(x_val_dl_tensor, y_val_tensor), 
                            batch_size=64, shuffle=False)
    
    x_test_dl_tensor = torch.tensor(x_test_dl, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    test_loader = DataLoader(TensorDataset(x_test_dl_tensor, y_test_tensor), 
                             batch_size=64, shuffle=False)
    
    print('  Validation set...')
    dl_val_probs = get_dl_predictions(dl_model, device, val_loader)
    print('  Test set...')
    dl_test_probs = get_dl_predictions(dl_model, device, test_loader)
    print('✓ DL predictions complete')
    
    # ============ Optimize Fusion Weights ============
    print('\n[4/6] Optimizing fusion weights on validation set...')
    best_weight_rf, best_val_auc = optimize_fusion_weights(y_val, rf_val_probs, dl_val_probs)
    weight_dl = 1 - best_weight_rf
    
    print(f'  Optimal weights: RF={best_weight_rf:.3f}, DL={weight_dl:.3f}')
    print(f'  Best validation AUC: {best_val_auc:.3f}')
    print('✓ Fusion weights optimized')
    
    # ============ Create Fused Predictions ============
    print('\n[5/6] Creating fused predictions...')
    
    # Simple average fusion
    fusion_val_probs_avg = 0.5 * rf_val_probs + 0.5 * dl_val_probs
    fusion_test_probs_avg = 0.5 * rf_test_probs + 0.5 * dl_test_probs
    
    # Optimized weighted fusion
    fusion_val_probs = best_weight_rf * rf_val_probs + weight_dl * dl_val_probs
    fusion_test_probs = best_weight_rf * rf_test_probs + weight_dl * dl_test_probs
    
    print('✓ Fused predictions created')
    
    # ============ Evaluate Fusion Model ============
    print('\n[6/6] Evaluating fusion model...')
    
    # Use optimized weighted fusion for evaluation
    print('\nFinding optimal thresholds...')
    rule_out_thresh, rule_in_thresh, f1_thresh = find_thresholds_from_val(y_val, fusion_val_probs)
    print(f'✓ Thresholds found (F1 thresh: {f1_thresh:.3f})')
    
    # Compute validation metrics at F1 threshold
    print('\nComputing validation metrics...')
    val_metrics = compute_metrics(y_val, fusion_val_probs, f1_thresh)
    print('Computing validation bootstrap CIs (200 iterations)...')
    val_ci = bootstrap_ci_val(y_val, fusion_val_probs)
    print('✓ Validation metrics complete')

    # Test metrics at F1-optimal threshold (from validation set)
    print('\nComputing test metrics...')
    test_metrics = compute_metrics(y_test, fusion_test_probs, f1_thresh)
    print('Computing test bootstrap CIs (200 iterations)...')
    test_ci = bootstrap_ci_test(y_test, fusion_test_probs, f1_thresh)
    print('✓ Test metrics complete')

    # Confusion matrix at F1-optimal threshold
    test_preds = (fusion_test_probs >= f1_thresh).astype(int)
    cm = confusion_matrix(y_test, test_preds)
    
    # ============ Print Results ============
    print('\n' + '=' * 80)
    print('FUSION MODEL RESULTS')
    feature_info = f'{num_features} features' if num_features else 'all features'
    print(f'Fusion Strategy: Weighted Average (RF={best_weight_rf:.3f}, DL={weight_dl:.3f})')
    print(f'RF Features Used: {feature_info}')
    print('-' * 80)
    print(f"Rule Out Thresh\nSens > 0.90\n{r3(rule_out_thresh) if rule_out_thresh is not None else 'N/A'}")
    print(f"Rule In Thresh\nPPV > 0.85\n{r3(rule_in_thresh) if rule_in_thresh is not None else 'N/A'}")
    print(f"F1 Thresh\n{r3(f1_thresh)}")

    print(f"Val AUC\n{format_with_ci(val_metrics['auc'], val_ci['auc'])}")
    print(f"Val AP\n{format_with_ci(val_metrics['ap'], val_ci['ap'])}")
    print(f"Val F1\n{format_with_ci(val_metrics['f1'], val_ci['f1'])}")
    print(f"Val Sens\n{format_with_ci(val_metrics['sens'], val_ci['sens'])}")
    print(f"Val Spec\n{format_with_ci(val_metrics['spec'], val_ci['spec'])}")
    print(f"Val Acc\n{format_with_ci(val_metrics['acc'], val_ci['acc'])}")
    print(f"Val PPV\n{format_with_ci(val_metrics['ppv'], val_ci['ppv'])}")
    print(f"Val NPV\n{format_with_ci(val_metrics['npv'], val_ci['npv'])}")

    print(f"Test AUC\n{format_with_ci(roc_auc_score(y_test, fusion_test_probs), test_ci['auc'])}")
    print(f"Test AP\n{format_with_ci(average_precision_score(y_test, fusion_test_probs), test_ci['ap'])}")
    print(f"Test F1\n{format_with_ci(test_metrics['f1'], test_ci['f1'])}")
    print(f"Test Sens\n{format_with_ci(test_metrics['sens'], test_ci['sens'])}")
    print(f"Test Spec\n{format_with_ci(test_metrics['spec'], test_ci['spec'])}")
    print(f"Test Acc\n{format_with_ci(test_metrics['acc'], test_ci['acc'])}")
    print(f"Test PPV\n{format_with_ci(test_metrics['ppv'], test_ci['ppv'])}")
    print(f"Test NPV\n{format_with_ci(test_metrics['npv'], test_ci['npv'])}")
    print('=' * 80)
    
    # ============ Save Results ============
    print('\nSaving results...')
    os.makedirs('test_results', exist_ok=True)
    if num_features is None:
        model_name = 'fusion_model_all_features'
    else:
        model_name = f'fusion_model_{num_features}_features'

    # Save metrics table to CSV in the displayed order, with CI columns (rounded to 3 decimals)
    feature_info = f'{num_features} features' if num_features else 'all features'
    rows = [
        ['Fusion Strategy', f'Weighted (RF={best_weight_rf:.3f}, DL={weight_dl:.3f})', None, None],
        ['RF Features Used', feature_info, None, None],
        ['Rule Out Thresh (Sens > 0.90)', r3(rule_out_thresh), None, None],
        ['Rule In Thresh (PPV > 0.85)', r3(rule_in_thresh), None, None],
        ['F1 Thresh', r3(f1_thresh), None, None],
        ['Val AUC', r3(val_metrics['auc']), r3(val_ci['auc'][0]), r3(val_ci['auc'][1])],
        ['Val AP', r3(val_metrics['ap']), r3(val_ci['ap'][0]), r3(val_ci['ap'][1])],
        ['Val F1', r3(val_metrics['f1']), r3(val_ci['f1'][0]), r3(val_ci['f1'][1])],
        ['Val Sens', r3(val_metrics['sens']), r3(val_ci['sens'][0]), r3(val_ci['sens'][1])],
        ['Val Spec', r3(val_metrics['spec']), r3(val_ci['spec'][0]), r3(val_ci['spec'][1])],
        ['Val Acc', r3(val_metrics['acc']), r3(val_ci['acc'][0]), r3(val_ci['acc'][1])],
        ['Val PPV', r3(val_metrics['ppv']), r3(val_ci['ppv'][0]), r3(val_ci['ppv'][1])],
        ['Val NPV', r3(val_metrics['npv']), r3(val_ci['npv'][0]), r3(val_ci['npv'][1])],
        ['Test AUC', r3(roc_auc_score(y_test, fusion_test_probs)), r3(test_ci['auc'][0]), r3(test_ci['auc'][1])],
        ['Test AP', r3(average_precision_score(y_test, fusion_test_probs)), r3(test_ci['ap'][0]), r3(test_ci['ap'][1])],
        ['Test F1', r3(test_metrics['f1']), r3(test_ci['f1'][0]), r3(test_ci['f1'][1])],
        ['Test Sens', r3(test_metrics['sens']), r3(test_ci['sens'][0]), r3(test_ci['sens'][1])],
        ['Test Spec', r3(test_metrics['spec']), r3(test_ci['spec'][0]), r3(test_ci['spec'][1])],
        ['Test Acc', r3(test_metrics['acc']), r3(test_ci['acc'][0]), r3(test_ci['acc'][1])],
        ['Test PPV', r3(test_metrics['ppv']), r3(test_ci['ppv'][0]), r3(test_ci['ppv'][1])],
        ['Test NPV', r3(test_metrics['npv']), r3(test_ci['npv'][0]), r3(test_ci['npv'][1])],
    ]
    metrics_df = pd.DataFrame(rows, columns=['Metric', 'Value', 'CI_low', 'CI_high'])
    metrics_df.to_csv(f'test_results/{model_name}_results.csv', index=False)
    print(f"✓ Results CSV saved to: test_results/{model_name}_results.csv")

    # Save probabilities as .npy files
    np.save(f'test_results/{model_name}_val_probabilities.npy', fusion_val_probs)
    np.save(f'test_results/{model_name}_test_probabilities.npy', fusion_test_probs)
    
    # Also save individual model probabilities for comparison
    np.save(f'test_results/{model_name}_rf_val_probabilities.npy', rf_val_probs)
    np.save(f'test_results/{model_name}_rf_test_probabilities.npy', rf_test_probs)
    np.save(f'test_results/{model_name}_dl_val_probabilities.npy', dl_val_probs)
    np.save(f'test_results/{model_name}_dl_test_probabilities.npy', dl_test_probs)
    print(f"✓ Probabilities saved to test_results/{model_name}_*_probabilities.npy")

    # Save plots (combined val+test ROC and confusion matrix)
    plot_val_test_roc(
        y_val,
        fusion_val_probs,
        y_test,
        fusion_test_probs,
        model_name,
        f'test_results/{model_name}_roc.png'
    )
    plot_confusion_matrix(cm, model_name, f'test_results/{model_name}_cm.png')
    
    # ============ Print Individual Model Results for Comparison ============
    print('\n' + '=' * 80)
    print('INDIVIDUAL MODEL COMPARISON')
    print('=' * 80)
    
    # RF Model
    print('\n--- Random Forest Model ---')
    rf_val_auc = roc_auc_score(y_val, rf_val_probs)
    rf_val_ap = average_precision_score(y_val, rf_val_probs)
    rf_test_auc = roc_auc_score(y_test, rf_test_probs)
    rf_test_ap = average_precision_score(y_test, rf_test_probs)
    print(f'Val AUC: {rf_val_auc:.3f}')
    print(f'Val AP:  {rf_val_ap:.3f}')
    print(f'Test AUC: {rf_test_auc:.3f}')
    print(f'Test AP:  {rf_test_ap:.3f}')
    
    # DL Model
    print('\n--- Deep Learning Model ---')
    dl_val_auc = roc_auc_score(y_val, dl_val_probs)
    dl_val_ap = average_precision_score(y_val, dl_val_probs)
    dl_test_auc = roc_auc_score(y_test, dl_test_probs)
    dl_test_ap = average_precision_score(y_test, dl_test_probs)
    print(f'Val AUC: {dl_val_auc:.3f}')
    print(f'Val AP:  {dl_val_ap:.3f}')
    print(f'Test AUC: {dl_test_auc:.3f}')
    print(f'Test AP:  {dl_test_ap:.3f}')
    
    # Fusion Model
    print('\n--- Fusion Model ---')
    fusion_val_auc = roc_auc_score(y_val, fusion_val_probs)
    fusion_val_ap = average_precision_score(y_val, fusion_val_probs)
    fusion_test_auc = roc_auc_score(y_test, fusion_test_probs)
    fusion_test_ap = average_precision_score(y_test, fusion_test_probs)
    print(f'Val AUC: {fusion_val_auc:.3f}')
    print(f'Val AP:  {fusion_val_ap:.3f}')
    print(f'Test AUC: {fusion_test_auc:.3f}')
    print(f'Test AP:  {fusion_test_ap:.3f}')
    
    print('\n' + '=' * 80)
    print('EVALUATION COMPLETE!')
    print('=' * 80)


if __name__ == '__main__':
    main()

