import numpy as np
import pandas as pd
import joblib
import torch
from torch.utils.data import DataLoader, TensorDataset
from scipy import signal
import matplotlib.pyplot as plt
import seaborn as sns
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


def plot_probability_scatter(rf_probs, dl_probs, y_true, title, save_path, 
                             rf_threshold=None, dl_threshold=None):
    """Create scatter plot of RF vs DL probabilities.
    
    Args:
        rf_probs: RF model probabilities
        dl_probs: DL model probabilities
        y_true: true labels
        title: plot title
        save_path: path to save the plot
        rf_threshold: optimal F1 threshold for RF model (vertical line)
        dl_threshold: optimal F1 threshold for DL model (horizontal line)
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Separate samples by class
    no_cad_mask = (y_true == 0)
    cad_mask = (y_true == 1)
    
    # Plot samples by class with different colors (opaque)
    ax.scatter(rf_probs[no_cad_mask], dl_probs[no_cad_mask], 
               c='#3498db', alpha=1.0, s=50, label=f'No CAD (n={np.sum(no_cad_mask)})',
               edgecolors='white', linewidth=0.5)
    ax.scatter(rf_probs[cad_mask], dl_probs[cad_mask], 
               c='#e74c3c', alpha=1.0, s=50, label=f'CAD (n={np.sum(cad_mask)})',
               edgecolors='white', linewidth=0.5)
    
    # Add optimal threshold lines
    if rf_threshold is not None:
        ax.axvline(x=rf_threshold, color='#9b59b6', linestyle=':', linewidth=2.5, 
                   label=f'RF F1 Threshold = {rf_threshold:.3f}', alpha=0.8)
    
    if dl_threshold is not None:
        ax.axhline(y=dl_threshold, color='#e67e22', linestyle=':', linewidth=2.5, 
                   label=f'DL F1 Threshold = {dl_threshold:.3f}', alpha=0.8)
    
    ax.set_xlabel('Random Forest Probability', fontsize=12, fontweight='bold')
    ax.set_ylabel('ECG-Smart-Net Probability', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
    
    # Make plot square
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Scatter plot saved to: {save_path}")


def plot_probability_hexbin(rf_probs, dl_probs, y_true, title, save_path):
    """Create hexbin density plot of RF vs DL probabilities (for large datasets).
    
    Args:
        rf_probs: RF model probabilities
        dl_probs: DL model probabilities
        y_true: true labels
        title: plot title
        save_path: path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # Separate samples by class
    no_cad_mask = (y_true == 0)
    cad_mask = (y_true == 1)
    
    # Plot for No CAD class
    hb0 = axes[0].hexbin(rf_probs[no_cad_mask], dl_probs[no_cad_mask], 
                         gridsize=30, cmap='Blues', mincnt=1, alpha=0.8)
    axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1, label='RF = DL')
    axes[0].set_xlabel('Random Forest Probability', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('ECG-Smart-Net Probability', fontsize=12, fontweight='bold')
    axes[0].set_title(f'No CAD Samples (n={np.sum(no_cad_mask)})', fontsize=13, fontweight='bold')
    axes[0].set_xlim(-0.05, 1.05)
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].grid(True, alpha=0.2)
    axes[0].set_aspect('equal')
    axes[0].legend()
    plt.colorbar(hb0, ax=axes[0], label='Count')
    
    # Plot for CAD class
    hb1 = axes[1].hexbin(rf_probs[cad_mask], dl_probs[cad_mask], 
                         gridsize=30, cmap='Reds', mincnt=1, alpha=0.8)
    axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1, label='RF = DL')
    axes[1].set_xlabel('Random Forest Probability', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('ECG-Smart-Net Probability', fontsize=12, fontweight='bold')
    axes[1].set_title(f'CAD Samples (n={np.sum(cad_mask)})', fontsize=13, fontweight='bold')
    axes[1].set_xlim(-0.05, 1.05)
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(True, alpha=0.2)
    axes[1].set_aspect('equal')
    axes[1].legend()
    plt.colorbar(hb1, ax=axes[1], label='Count')
    
    fig.suptitle(title, fontsize=15, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Hexbin plot saved to: {save_path}")


def plot_probability_distribution(rf_probs, dl_probs, y_true, title, save_path):
    """Create histogram/distribution plot comparing RF and DL probabilities.
    
    Args:
        rf_probs: RF model probabilities
        dl_probs: DL model probabilities
        y_true: true labels
        title: plot title
        save_path: path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Separate samples by class
    no_cad_mask = (y_true == 0)
    cad_mask = (y_true == 1)
    
    # RF probabilities distribution
    axes[0, 0].hist(rf_probs[no_cad_mask], bins=30, alpha=0.6, color='#3498db', 
                    label=f'No CAD (n={np.sum(no_cad_mask)})', edgecolor='black', linewidth=0.5)
    axes[0, 0].hist(rf_probs[cad_mask], bins=30, alpha=0.6, color='#e74c3c', 
                    label=f'CAD (n={np.sum(cad_mask)})', edgecolor='black', linewidth=0.5)
    axes[0, 0].set_xlabel('Random Forest Probability', fontsize=11, fontweight='bold')
    axes[0, 0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
    axes[0, 0].set_title('RF Model Predictions', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.2, axis='y')
    
    # DL probabilities distribution
    axes[0, 1].hist(dl_probs[no_cad_mask], bins=30, alpha=0.6, color='#3498db', 
                    label=f'No CAD (n={np.sum(no_cad_mask)})', edgecolor='black', linewidth=0.5)
    axes[0, 1].hist(dl_probs[cad_mask], bins=30, alpha=0.6, color='#e74c3c', 
                    label=f'CAD (n={np.sum(cad_mask)})', edgecolor='black', linewidth=0.5)
    axes[0, 1].set_xlabel('ECG-Smart-Net Probability', fontsize=11, fontweight='bold')
    axes[0, 1].set_ylabel('Frequency', fontsize=11, fontweight='bold')
    axes[0, 1].set_title('DL Model Predictions', fontsize=12, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.2, axis='y')
    
    # KDE plots
    from scipy.stats import gaussian_kde
    
    # RF KDE
    if len(rf_probs[no_cad_mask]) > 1:
        kde_rf_no_cad = gaussian_kde(rf_probs[no_cad_mask])
        x_range = np.linspace(0, 1, 200)
        axes[1, 0].plot(x_range, kde_rf_no_cad(x_range), color='#3498db', 
                       linewidth=2, label='No CAD')
    if len(rf_probs[cad_mask]) > 1:
        kde_rf_cad = gaussian_kde(rf_probs[cad_mask])
        axes[1, 0].plot(x_range, kde_rf_cad(x_range), color='#e74c3c', 
                       linewidth=2, label='CAD')
    axes[1, 0].set_xlabel('Random Forest Probability', fontsize=11, fontweight='bold')
    axes[1, 0].set_ylabel('Density', fontsize=11, fontweight='bold')
    axes[1, 0].set_title('RF Model Probability Density', fontsize=12, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.2)
    axes[1, 0].set_xlim(0, 1)
    
    # DL KDE
    if len(dl_probs[no_cad_mask]) > 1:
        kde_dl_no_cad = gaussian_kde(dl_probs[no_cad_mask])
        axes[1, 1].plot(x_range, kde_dl_no_cad(x_range), color='#3498db', 
                       linewidth=2, label='No CAD')
    if len(dl_probs[cad_mask]) > 1:
        kde_dl_cad = gaussian_kde(dl_probs[cad_mask])
        axes[1, 1].plot(x_range, kde_dl_cad(x_range), color='#e74c3c', 
                       linewidth=2, label='CAD')
    axes[1, 1].set_xlabel('ECG-Smart-Net Probability', fontsize=11, fontweight='bold')
    axes[1, 1].set_ylabel('Density', fontsize=11, fontweight='bold')
    axes[1, 1].set_title('DL Model Probability Density', fontsize=12, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.2)
    axes[1, 1].set_xlim(0, 1)
    
    fig.suptitle(title, fontsize=15, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Distribution plot saved to: {save_path}")


def main():
    print('=' * 80)
    print('FUSION MODEL VISUALIZATION')
    print('Comparing Random Forest and Deep Learning Model Predictions')
    print('=' * 80)
    
    # ============ Configuration (matching fusion_model.py) ============
    num_features = 100  # Using the best performing RF model with 100 features
    
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
    
    # ============ Load Data ============
    print('\n[1/4] Loading data...')
    print('\nLoading RF data:')
    x_val_rf, y_val, x_test_rf, y_test = get_rf_data(num_features)
    
    print('\nLoading DL data:')
    x_val_dl, y_val_dl, x_test_dl, y_test_dl = get_dl_data()
    
    # Verify labels match
    assert np.all(y_val == y_val_dl), "Validation labels mismatch between RF and DL!"
    assert np.all(y_test == y_test_dl), "Test labels mismatch between RF and DL!"
    print('✓ Data loaded successfully')
    
    # ============ Load Models ============
    print('\n[2/4] Loading models...')
    
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
    print('\n[3/4] Getting predictions from both models...')
    
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
    
    # ============ Create Visualizations ============
    print('\n[4/4] Creating visualizations...')
    os.makedirs('visualizations', exist_ok=True)
    
    # Validation set scatter plot (with validation-derived thresholds)
    print('\nGenerating validation set scatter plot...')
    plot_probability_scatter(
        rf_val_probs, dl_val_probs, y_val,
        'Validation Set: RF vs DL Model Predictions',
        'visualizations/val_rf_vs_dl_scatter.png',
        rf_threshold=rf_val_f1_threshold,
        dl_threshold=dl_val_f1_threshold
    )
    
    # Test set scatter plot (using same thresholds from validation)
    print('Generating test set scatter plot...')
    plot_probability_scatter(
        rf_test_probs, dl_test_probs, y_test,
        'Test Set: RF vs DL Model Predictions',
        'visualizations/test_rf_vs_dl_scatter.png',
        rf_threshold=rf_val_f1_threshold,
        dl_threshold=dl_val_f1_threshold
    )
    
    # Validation set hexbin plot (density)
    print('Generating validation set hexbin plot...')
    plot_probability_hexbin(
        rf_val_probs, dl_val_probs, y_val,
        'Validation Set: Probability Density Comparison',
        'visualizations/val_rf_vs_dl_hexbin.png'
    )
    
    # Test set hexbin plot (density)
    print('Generating test set hexbin plot...')
    plot_probability_hexbin(
        rf_test_probs, dl_test_probs, y_test,
        'Test Set: Probability Density Comparison',
        'visualizations/test_rf_vs_dl_hexbin.png'
    )
    
    # Validation set distribution plot
    print('Generating validation set distribution plot...')
    plot_probability_distribution(
        rf_val_probs, dl_val_probs, y_val,
        'Validation Set: Probability Distributions',
        'visualizations/val_probability_distributions.png'
    )
    
    # Test set distribution plot
    print('Generating test set distribution plot...')
    plot_probability_distribution(
        rf_test_probs, dl_test_probs, y_test,
        'Test Set: Probability Distributions',
        'visualizations/test_probability_distributions.png'
    )
    
    # Calculate and print summary statistics
    print('\n' + '=' * 80)
    print('SUMMARY STATISTICS')
    print('=' * 80)
    
    print('\nValidation Set:')
    print(f'  RF AUC:  {roc_auc_score(y_val, rf_val_probs):.3f}')
    print(f'  DL AUC:  {roc_auc_score(y_val, dl_val_probs):.3f}')
    print(f'  RF AP:   {average_precision_score(y_val, rf_val_probs):.3f}')
    print(f'  DL AP:   {average_precision_score(y_val, dl_val_probs):.3f}')
    print(f'  Correlation (RF vs DL): {np.corrcoef(rf_val_probs, dl_val_probs)[0, 1]:.3f}')
    
    print('\nTest Set:')
    print(f'  RF AUC:  {roc_auc_score(y_test, rf_test_probs):.3f}')
    print(f'  DL AUC:  {roc_auc_score(y_test, dl_test_probs):.3f}')
    print(f'  RF AP:   {average_precision_score(y_test, rf_test_probs):.3f}')
    print(f'  DL AP:   {average_precision_score(y_test, dl_test_probs):.3f}')
    print(f'  Correlation (RF vs DL): {np.corrcoef(rf_test_probs, dl_test_probs)[0, 1]:.3f}')
    
    print('\n' + '=' * 80)
    print('VISUALIZATION COMPLETE!')
    print('All plots saved to: visualizations/')
    print('=' * 80)


if __name__ == '__main__':
    main()

