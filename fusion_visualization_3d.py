import numpy as np
import pandas as pd
import joblib
import torch
from torch.utils.data import DataLoader, TensorDataset
from scipy import signal
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import roc_auc_score, average_precision_score
import os


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
    x_val = np.concatenate(x_val, axis=0).astype(float)
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
    x_test = np.concatenate(x_test, axis=0).astype(float)
    print(f"RF - Test set: {len(x_test)} samples loaded")

    return x_val, y_val, x_test, y_test


def get_clinical_feature(feature_name='age'):
    """Load a specific clinical feature for visualization.
    
    Args:
        feature_name: Name of the feature to load (e.g., 'age', 'sex', 'bmi')
    
    Returns:
        val_feature: feature values for validation set
        test_feature: feature values for test set
    """
    # Load the full feature dataset
    features_df = pd.read_csv('results/features.csv')
    
    # Check if feature exists
    if feature_name not in features_df.columns:
        available_features = [col for col in features_df.columns if col != 'Unnamed: 0']
        print(f"Warning: Feature '{feature_name}' not found!")
        print(f"Available features: {', '.join(available_features[:20])}...")
        raise ValueError(f"Feature '{feature_name}' not found in features.csv")
    
    # Load validation and test IDs
    val_df = pd.read_csv('val_set.csv')
    test_df = pd.read_csv('test_set.csv')
    val_ids = val_df['ID'].to_list()
    test_ids = test_df['ID'].to_list()
    
    # Extract feature values for validation set
    val_feature = []
    for id in val_ids:
        feature_val = features_df[features_df['Unnamed: 0'] == id][feature_name].values[0]
        val_feature.append(float(feature_val))
    val_feature = np.array(val_feature)
    
    # Extract feature values for test set
    test_feature = []
    for id in test_ids:
        feature_val = features_df[features_df['Unnamed: 0'] == id][feature_name].values[0]
        test_feature.append(float(feature_val))
    test_feature = np.array(test_feature)
    
    print(f"Feature '{feature_name}' loaded:")
    print(f"  Val set - mean: {np.mean(val_feature):.2f}, std: {np.std(val_feature):.2f}, range: [{np.min(val_feature):.2f}, {np.max(val_feature):.2f}]")
    print(f"  Test set - mean: {np.mean(test_feature):.2f}, std: {np.std(test_feature):.2f}, range: [{np.min(test_feature):.2f}, {np.max(test_feature):.2f}]")
    
    return val_feature, test_feature


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


def find_optimal_f1_threshold(y_true, y_probs):
    """Find the threshold that maximizes F1 score.
    
    Args:
        y_true: true labels
        y_probs: predicted probabilities
        
    Returns:
        optimal_threshold: threshold that maximizes F1
        max_f1: maximum F1 score
    """
    from sklearn.metrics import f1_score, roc_curve
    
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    
    f1_scores = []
    for thresh in thresholds:
        preds = (y_probs >= thresh).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        f1_scores.append(f1)
    
    f1_scores = np.array(f1_scores)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    max_f1 = f1_scores[optimal_idx]
    
    return optimal_threshold, max_f1


def plot_3d_scatter_interactive(rf_probs, dl_probs, feature_vals, y_true, feature_name, 
                                title, save_path, rf_threshold=None, dl_threshold=None,
                                show_interactive=True):
    """Create interactive 3D scatter plot of RF vs DL probabilities vs clinical feature.
    
    Args:
        rf_probs: RF model probabilities
        dl_probs: DL model probabilities
        feature_vals: clinical feature values
        y_true: true labels
        feature_name: name of the clinical feature
        title: plot title
        save_path: path to save the plot (if saving)
        rf_threshold: optimal F1 threshold for RF model
        dl_threshold: optimal F1 threshold for DL model
        show_interactive: if True, display interactive plot; if False, just save
    """
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Separate samples by class
    no_cad_mask = (y_true == 0)
    cad_mask = (y_true == 1)
    
    # Plot samples by class with different colors (opaque)
    ax.scatter(rf_probs[no_cad_mask], dl_probs[no_cad_mask], feature_vals[no_cad_mask],
               c='#3498db', alpha=1.0, s=50, label=f'No CAD (n={np.sum(no_cad_mask)})',
               edgecolors='white', linewidth=0.5, depthshade=True)
    ax.scatter(rf_probs[cad_mask], dl_probs[cad_mask], feature_vals[cad_mask],
               c='#e74c3c', alpha=1.0, s=50, label=f'CAD (n={np.sum(cad_mask)})',
               edgecolors='white', linewidth=0.5, depthshade=True)
    
    # Add threshold planes
    if rf_threshold is not None:
        # Vertical plane at RF threshold
        feature_range = [np.min(feature_vals), np.max(feature_vals)]
        dl_range = [0, 1]
        z_grid, y_grid = np.meshgrid(feature_range, dl_range)
        x_grid = np.full_like(z_grid, rf_threshold)
        ax.plot_surface(x_grid, y_grid, z_grid, alpha=0.2, color='#9b59b6', 
                       edgecolor='none', label='RF Threshold')
    
    if dl_threshold is not None:
        # Horizontal plane at DL threshold
        feature_range = [np.min(feature_vals), np.max(feature_vals)]
        rf_range = [0, 1]
        z_grid, x_grid = np.meshgrid(feature_range, rf_range)
        y_grid = np.full_like(z_grid, dl_threshold)
        ax.plot_surface(x_grid, y_grid, z_grid, alpha=0.2, color='#e67e22', 
                       edgecolor='none', label='DL Threshold')
    
    ax.set_xlabel('Random Forest Probability', fontsize=11, fontweight='bold', labelpad=10)
    ax.set_ylabel('ECG-Smart-Net Probability', fontsize=11, fontweight='bold', labelpad=10)
    ax.set_zlabel(f'{feature_name}', fontsize=11, fontweight='bold', labelpad=10)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=20)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"3D scatter plot saved to: {save_path}")
    
    # Show interactive plot
    if show_interactive:
        print(f"Displaying interactive plot: {title}")
        print("  → Use your mouse to rotate and zoom the 3D plot")
        print("  → Close the window to continue...")
        plt.show()
    else:
        plt.close()




def plot_feature_distribution_by_quadrant(rf_probs, dl_probs, feature_vals, y_true, 
                                          feature_name, title, save_path,
                                          rf_threshold=0.5, dl_threshold=0.5):
    """Plot feature distribution for different prediction quadrants.
    
    Args:
        rf_probs: RF model probabilities
        dl_probs: DL model probabilities
        feature_vals: clinical feature values
        y_true: true labels
        feature_name: name of the clinical feature
        title: plot title
        save_path: path to save the plot
        rf_threshold: RF threshold for quadrant division
        dl_threshold: DL threshold for quadrant division
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Define quadrants based on thresholds
    # Q1: Both predict positive (upper-right)
    q1_mask = (rf_probs >= rf_threshold) & (dl_probs >= dl_threshold)
    # Q2: RF negative, DL positive (upper-left)
    q2_mask = (rf_probs < rf_threshold) & (dl_probs >= dl_threshold)
    # Q3: Both predict negative (lower-left)
    q3_mask = (rf_probs < rf_threshold) & (dl_probs < dl_threshold)
    # Q4: RF positive, DL negative (lower-right)
    q4_mask = (rf_probs >= rf_threshold) & (dl_probs < dl_threshold)
    
    quadrants = [
        (q1_mask, 'Both Predict CAD', axes[0, 1]),
        (q2_mask, 'Only DL Predicts CAD', axes[0, 0]),
        (q3_mask, 'Both Predict No CAD', axes[1, 0]),
        (q4_mask, 'Only RF Predicts CAD', axes[1, 1])
    ]
    
    for mask, label, ax in quadrants:
        if np.sum(mask) > 0:
            # Separate by true class
            no_cad_in_quad = mask & (y_true == 0)
            cad_in_quad = mask & (y_true == 1)
            
            ax.hist(feature_vals[no_cad_in_quad], bins=20, alpha=0.6, color='#3498db', 
                   label=f'No CAD (n={np.sum(no_cad_in_quad)})', edgecolor='black', linewidth=0.5)
            ax.hist(feature_vals[cad_in_quad], bins=20, alpha=0.6, color='#e74c3c', 
                   label=f'CAD (n={np.sum(cad_in_quad)})', edgecolor='black', linewidth=0.5)
            
            ax.set_xlabel(feature_name, fontsize=10, fontweight='bold')
            ax.set_ylabel('Frequency', fontsize=10, fontweight='bold')
            ax.set_title(f'{label}\n(Total: {np.sum(mask)} samples)', fontsize=11, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.2, axis='y')
        else:
            ax.text(0.5, 0.5, 'No samples in this quadrant', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(label, fontsize=11, fontweight='bold')
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Feature distribution by quadrant saved to: {save_path}")


def main():
    print('=' * 80)
    print('FUSION MODEL 3D VISUALIZATION')
    print('Comparing Random Forest and Deep Learning Model Predictions with Clinical Features')
    print('=' * 80)
    
    # ============ Configuration (matching fusion_model.py) ============
    num_features = 100  # Using the best performing RF model with 100 features
    
    # Clinical feature to visualize (change this to explore different features)
    clinical_feature = 'age'  # Options: 'age', 'sex', 'bmi', etc.
    
    # Display mode: show interactive plots (True) or only save images (False)
    show_interactive = True  # Set to False to only save static images without displaying
    
    # ============ Define Model Paths (matching fusion_model.py) ============
    if num_features is None:
        rf_model_path = 'rf_models/best_rf_all_features.pkl'
    else:
        rf_model_path = f'rf_models/best_rf_selected_features_{num_features}.pkl'
    
    dl_model_path = 'models/transfer_learning_CAD__2025-11-17-16-51-30.pt'
    
    print(f'\nConfiguration:')
    print(f'  RF Model: {rf_model_path}')
    print(f'  DL Model: {dl_model_path}')
    print(f'  Number of features: {num_features if num_features else "all"}')
    print(f'  Clinical feature: {clinical_feature}')
    print(f'  Interactive mode: {show_interactive}')
    
    # ============ Load Data ============
    print('\n[1/5] Loading data...')
    print('\nLoading RF data:')
    x_val_rf, y_val, x_test_rf, y_test = get_rf_data(num_features)
    
    print('\nLoading DL data:')
    x_val_dl, y_val_dl, x_test_dl, y_test_dl = get_dl_data()
    
    # Verify labels match
    assert np.all(y_val == y_val_dl), "Validation labels mismatch between RF and DL!"
    assert np.all(y_test == y_test_dl), "Test labels mismatch between RF and DL!"
    print('✓ Data loaded successfully')
    
    # ============ Load Clinical Feature ============
    print(f'\n[2/5] Loading clinical feature: {clinical_feature}...')
    val_feature, test_feature = get_clinical_feature(clinical_feature)
    print('✓ Feature loaded successfully')
    
    # ============ Load Models ============
    print('\n[3/5] Loading models...')
    
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
    print('\n[4/5] Getting predictions from both models...')
    
    # RF predictions
    print('\nRF predictions:')
    print('  Validation set...')
    rf_val_probs = rf_model.predict_proba(x_val_rf)[:, 1]
    print('  Test set...')
    rf_test_probs = rf_model.predict_proba(x_test_rf)[:, 1]
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
    
    # ============ Find Optimal F1 Thresholds from Validation Set ============
    print('\nFinding optimal F1 thresholds from validation set...')
    rf_val_f1_threshold, rf_val_f1_score = find_optimal_f1_threshold(y_val, rf_val_probs)
    dl_val_f1_threshold, dl_val_f1_score = find_optimal_f1_threshold(y_val, dl_val_probs)
    
    print(f'  RF optimal F1 threshold: {rf_val_f1_threshold:.3f} (F1={rf_val_f1_score:.3f})')
    print(f'  DL optimal F1 threshold: {dl_val_f1_threshold:.3f} (F1={dl_val_f1_score:.3f})')
    print('✓ Thresholds computed')
    
    # ============ Create 3D Visualizations ============
    print('\n[5/5] Creating 3D visualizations...')
    os.makedirs('visualizations_3d', exist_ok=True)
    
    if show_interactive:
        print('\n' + '=' * 80)
        print('INTERACTIVE MODE')
        print('Each 3D plot will open in a separate window.')
        print('Use your mouse to:')
        print('  - Left-click + drag: Rotate the 3D view')
        print('  - Right-click + drag: Zoom in/out')
        print('  - Middle-click + drag: Pan the view')
        print('Close each window to proceed to the next plot.')
        print('=' * 80)
    
    # Validation set 3D scatter (interactive)
    print('\n[1/3] Validation set 3D scatter plot...')
    plot_3d_scatter_interactive(
        rf_val_probs, dl_val_probs, val_feature, y_val, clinical_feature,
        f'Validation Set: 3D View (RF vs DL vs {clinical_feature})',
        f'visualizations_3d/val_3d_scatter_{clinical_feature}.png',
        rf_threshold=rf_val_f1_threshold,
        dl_threshold=dl_val_f1_threshold,
        show_interactive=show_interactive
    )
    
    # Test set 3D scatter (interactive)
    print('\n[2/3] Test set 3D scatter plot...')
    plot_3d_scatter_interactive(
        rf_test_probs, dl_test_probs, test_feature, y_test, clinical_feature,
        f'Test Set: 3D View (RF vs DL vs {clinical_feature})',
        f'visualizations_3d/test_3d_scatter_{clinical_feature}.png',
        rf_threshold=rf_val_f1_threshold,
        dl_threshold=dl_val_f1_threshold,
        show_interactive=show_interactive
    )
    
    # Feature distribution by prediction quadrants (always non-interactive)
    print('\n[3/3] Feature distribution by quadrant plots...')
    print('  Validation set...')
    plot_feature_distribution_by_quadrant(
        rf_val_probs, dl_val_probs, val_feature, y_val, clinical_feature,
        f'Validation Set: {clinical_feature} Distribution by Prediction Quadrants',
        f'visualizations_3d/val_feature_by_quadrant_{clinical_feature}.png',
        rf_threshold=rf_val_f1_threshold,
        dl_threshold=dl_val_f1_threshold
    )
    
    print('  Test set...')
    plot_feature_distribution_by_quadrant(
        rf_test_probs, dl_test_probs, test_feature, y_test, clinical_feature,
        f'Test Set: {clinical_feature} Distribution by Prediction Quadrants',
        f'visualizations_3d/test_feature_by_quadrant_{clinical_feature}.png',
        rf_threshold=rf_val_f1_threshold,
        dl_threshold=dl_val_f1_threshold
    )
    
    # Calculate summary statistics
    print('\n' + '=' * 80)
    print('SUMMARY STATISTICS')
    print('=' * 80)
    
    print(f'\nFeature: {clinical_feature}')
    print('\nValidation Set:')
    print(f'  No CAD - mean: {np.mean(val_feature[y_val==0]):.2f}, std: {np.std(val_feature[y_val==0]):.2f}')
    print(f'  CAD    - mean: {np.mean(val_feature[y_val==1]):.2f}, std: {np.std(val_feature[y_val==1]):.2f}')
    print(f'  RF AUC:  {roc_auc_score(y_val, rf_val_probs):.3f}')
    print(f'  DL AUC:  {roc_auc_score(y_val, dl_val_probs):.3f}')
    
    print('\nTest Set:')
    print(f'  No CAD - mean: {np.mean(test_feature[y_test==0]):.2f}, std: {np.std(test_feature[y_test==0]):.2f}')
    print(f'  CAD    - mean: {np.mean(test_feature[y_test==1]):.2f}, std: {np.std(test_feature[y_test==1]):.2f}')
    print(f'  RF AUC:  {roc_auc_score(y_test, rf_test_probs):.3f}')
    print(f'  DL AUC:  {roc_auc_score(y_test, dl_test_probs):.3f}')
    
    print('\n' + '=' * 80)
    print('3D VISUALIZATION COMPLETE!')
    print('All plots saved to: visualizations_3d/')
    print('=' * 80)


if __name__ == '__main__':
    main()

