"""
Transfer Learning for CAD Detection using Pretrained OMI Model

This script loads a pretrained ECGSmartNet model with SE (Squeeze-and-Excitation) 
blocks trained on OMI detection and fine-tunes layer4 (last residual block) and 
the final FC layer for CAD detection. This provides more adaptation capacity than 
freezing all except FC, while still preserving most of the pretrained features.

Training strategy matches train_models_random_search.py:
- 200 epochs with early stopping after 10 epochs without improvement
- Two learning rates: lr0 for epoch 0, lr for epochs 1+
- Random hyperparameter search with 100 iterations
"""

import os
from ecg_models import *
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, f1_score, roc_curve, precision_recall_curve
from scipy import signal
import pandas as pd
import wandb
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def find_optimal_f1_threshold(y_true, y_pred_proba):
    """
    Find the threshold that maximizes F1 score.
    
    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities for positive class
    
    Returns:
        optimal_threshold: Threshold that maximizes F1
        optimal_f1: Maximum F1 score achieved
    """
    # Get precision, recall, and thresholds
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)
    
    # Calculate F1 score for each threshold
    f1_scores = np.zeros(len(precisions))
    for i in range(len(precisions)):
        if precisions[i] + recalls[i] > 0:
            f1_scores[i] = 2 * (precisions[i] * recalls[i]) / (precisions[i] + recalls[i])
        else:
            f1_scores[i] = 0
    
    # Find the threshold that maximizes F1
    optimal_idx = np.argmax(f1_scores)
    optimal_f1 = f1_scores[optimal_idx]
    
    # Handle edge case
    if optimal_idx < len(thresholds):
        optimal_threshold = thresholds[optimal_idx]
    else:
        optimal_threshold = thresholds[-1] if len(thresholds) > 0 else 0.5
    
    return optimal_threshold, optimal_f1


def train_epoch(model, device, train_dataloader, criterion, optimizer, scaler, use_amp=True):
    train_loss = 0
    total_samples = 0
    model.train()

    ys = []
    y_preds = []

    for (x, y) in train_dataloader:

        # undersample the No CAD class
        indices0 = np.where(y == 0)[0]
        indices1 = np.where(y == 1)[0]
        num_samples = np.min([len(indices1), len(indices0)])

        
        if num_samples > 0:

            indices0 = np.random.choice(indices0, num_samples, replace=False)
            indices = np.concatenate([indices0, indices1], axis=0)
            np.random.shuffle(indices)
            x = x[indices]
            y = y[indices]
            x = torch.unsqueeze(x, 1)

            x = x.to(device)
            y = y.to(device)

            with torch.amp.autocast(device_type='cuda', enabled=use_amp):
                y_pred = model(x)
                loss = criterion(y_pred, y)
                y_pred = torch.softmax(y_pred, dim=-1)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            
            # Unscale gradients and clip them to prevent exploding gradients
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()

            # Check for NaN in loss
            if torch.isnan(loss):
                print("WARNING: NaN detected in training loss! Skipping batch.")
                continue

            batch_size = y.size(0)
            train_loss += loss.item() * batch_size
            total_samples += batch_size

            y_pred = y_pred.cpu().detach().numpy()
            y = y.cpu().detach().numpy()
            ys.append(y)
            y_preds.append(y_pred)

    # Safeguard against empty lists
    if len(ys) == 0 or len(y_preds) == 0:
        print("ERROR: All training batches were skipped due to NaN! Returning dummy metrics.")
        return np.inf, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    y = np.concatenate(ys, axis=0)
    y_pred = np.concatenate(y_preds, axis=0)
    train_loss /= total_samples

    y_pred = y_pred[:, 1]
    
    # Check for NaN/Inf in predictions
    if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
        print("ERROR: NaN or Inf detected in predictions!")
        print(f"  NaN count: {np.sum(np.isnan(y_pred))}")
        print(f"  Inf count: {np.sum(np.isinf(y_pred))}")
        y_pred = np.nan_to_num(y_pred, nan=0.5, posinf=1.0, neginf=0.0)

    auc = roc_auc_score(y, y_pred)
    ap = average_precision_score(y, y_pred)
    acc = accuracy_score(y, y_pred > 0.5)
    prec = precision_score(y, y_pred > 0.5)
    rec = recall_score(y, y_pred > 0.5)
    spec = recall_score(y, y_pred > 0.5, pos_label=0)
    f1 = f1_score(y, y_pred > 0.5)

    return train_loss, auc, acc, prec, rec, spec, f1, ap


def val_epoch(model, device, val_dataloader, criterion, use_amp=True):
    val_loss = 0
    total_samples = 0
    model.eval()

    ys = []
    y_preds = []

    with torch.no_grad():
        for (x,y) in val_dataloader:
            x = x.to(device)
            y = y.to(device)
            x = torch.unsqueeze(x, 1)

            with torch.amp.autocast(device_type='cuda', enabled=use_amp):
                y_pred = model(x)
                loss = criterion(y_pred, y)
                y_pred = torch.softmax(y_pred, dim=-1)

            batch_size = y.size(0)
            val_loss += loss.item() * batch_size
            total_samples += batch_size

            y_pred = y_pred.cpu().detach().numpy()
            y = y.cpu().detach().numpy()

            ys.append(y)
            y_preds.append(y_pred)

    # Safeguard against empty lists
    if len(ys) == 0 or len(y_preds) == 0:
        print("ERROR: All validation batches were empty! Returning dummy metrics.")
        return np.inf, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, np.array([]), np.array([])

    y = np.concatenate(ys, axis=0)
    y_pred = np.concatenate(y_preds, axis=0)
    val_loss /= total_samples

    y_pred = y_pred[:, 1]
    
    # Check for NaN/Inf in validation predictions
    if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
        print("ERROR: NaN or Inf detected in VALIDATION predictions!")
        print(f"  NaN count: {np.sum(np.isnan(y_pred))}")
        print(f"  Inf count: {np.sum(np.isinf(y_pred))}")
        y_pred = np.nan_to_num(y_pred, nan=0.5, posinf=1.0, neginf=0.0)

    auc = roc_auc_score(y, y_pred)
    ap = average_precision_score(y, y_pred)
    acc = accuracy_score(y, y_pred > 0.5)
    prec = precision_score(y, y_pred > 0.5)
    rec = recall_score(y, y_pred > 0.5)
    spec = recall_score(y, y_pred > 0.5, pos_label=0)
    f1 = f1_score(y, y_pred > 0.5)

    return val_loss, auc, acc, prec, rec, spec, f1, ap, y, y_pred


def test_epoch(model, device, test_dataloader, criterion, use_amp=True):
    test_loss = 0
    total_samples = 0
    model.eval()

    ys = []
    y_preds = []

    with torch.no_grad():
        for (x,y) in test_dataloader:
            x = x.to(device)
            y = y.to(device)
            x = torch.unsqueeze(x, 1)

            with torch.amp.autocast(device_type='cuda', enabled=use_amp):
                y_pred = model(x)
                loss = criterion(y_pred, y)
                y_pred = torch.softmax(y_pred, dim=-1)

            batch_size = y.size(0)
            test_loss += loss.item() * batch_size
            total_samples += batch_size

            y_pred = y_pred.cpu().detach().numpy()
            y = y.cpu().detach().numpy()

            ys.append(y)
            y_preds.append(y_pred)

    # Safeguard against empty lists
    if len(ys) == 0 or len(y_preds) == 0:
        print("ERROR: All test batches were empty! Returning dummy metrics.")
        return np.inf, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, np.array([]), np.array([])

    y = np.concatenate(ys, axis=0)
    y_pred = np.concatenate(y_preds, axis=0)
    test_loss /= total_samples

    y_pred = y_pred[:, 1]
    
    # Check for NaN/Inf in test predictions
    if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
        print("ERROR: NaN or Inf detected in TEST predictions!")
        print(f"  NaN count: {np.sum(np.isnan(y_pred))}")
        print(f"  Inf count: {np.sum(np.isinf(y_pred))}")
        y_pred = np.nan_to_num(y_pred, nan=0.5, posinf=1.0, neginf=0.0)

    auc = roc_auc_score(y, y_pred)
    ap = average_precision_score(y, y_pred)
    acc = accuracy_score(y, y_pred > 0.5)
    prec = precision_score(y, y_pred > 0.5)
    rec = recall_score(y, y_pred > 0.5)
    spec = recall_score(y, y_pred > 0.5, pos_label=0)
    f1 = f1_score(y, y_pred > 0.5)

    return test_loss, auc, acc, prec, rec, spec, f1, ap, y, y_pred


def get_data(path):
    """Load and preprocess ECG data."""
    train_df = pd.read_csv('train_set.csv')
    val_df = pd.read_csv('val_set.csv')
    test_df = pd.read_csv('test_set.csv')
    train_outcomes = train_df['label'].to_numpy()
    val_outcomes = val_df['label'].to_numpy()
    test_outcomes = test_df['label'].to_numpy()
    train_ids = train_df['ID'].to_list()
    val_ids = val_df['ID'].to_list()
    test_ids = test_df['ID'].to_list()

    # get train data
    train_data = []
    for id in train_ids:
        ecg = np.load(path + id + '.npy', allow_pickle=True).item()
        ecg = ecg['waveforms']['ecg_median']
        ecg = ecg[:,150:-50]
        ecg = signal.resample(ecg, 200, axis=1)
        max_val = np.max(np.abs(ecg), axis=1, keepdims=True)
        
        # Replace zeros with 1 to avoid division by zero
        max_val = np.where(max_val == 0, 1, max_val)
        ecg = ecg / max_val
        
        train_data.append(ecg)
    
    train_data = np.array(train_data)
    print(f"Train set: {len(train_data)} files loaded")

    val_data = []
    for id in val_ids:
        ecg = np.load(path + id + '.npy', allow_pickle=True).item()
        ecg = ecg['waveforms']['ecg_median']
        ecg = ecg[:,150:-50]
        ecg = signal.resample(ecg, 200, axis=1)
        max_val = np.max(np.abs(ecg), axis=1, keepdims=True)
        
        # Replace zeros with 1 to avoid division by zero
        max_val = np.where(max_val == 0, 1, max_val)
        ecg = ecg / max_val
        
        val_data.append(ecg)
    
    val_data = np.array(val_data)
    print(f"Val set: {len(val_data)} files loaded")

    test_data = []
    for id in test_ids:
        ecg = np.load(path + id + '.npy', allow_pickle=True).item()
        ecg = ecg['waveforms']['ecg_median']
        ecg = ecg[:,150:-50]
        ecg = signal.resample(ecg, 200, axis=1)
        max_val = np.max(np.abs(ecg), axis=1, keepdims=True)
        
        # Replace zeros with 1 to avoid division by zero
        max_val = np.where(max_val == 0, 1, max_val)
        ecg = ecg / max_val
        
        test_data.append(ecg)
    
    test_data = np.array(test_data)
    print(f"Test set: {len(test_data)} files loaded")

    return train_data, train_outcomes, val_data, val_outcomes, test_data, test_outcomes


def plot_roc_curve(y_val_true, y_val_pred, y_test_true, y_test_pred, save_path):
    """Plot ROC curves for validation and test sets."""
    fpr_val, tpr_val, _ = roc_curve(y_val_true, y_val_pred)
    fpr_test, tpr_test, _ = roc_curve(y_test_true, y_test_pred)
    
    auc_val = roc_auc_score(y_val_true, y_val_pred)
    auc_test = roc_auc_score(y_test_true, y_test_pred)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr_test, tpr_test, color='tab:blue', lw=2, 
             label=f'Test ROC (AUC = {auc_test:.3f})')
    plt.plot(fpr_val, tpr_val, color='tab:orange', lw=2, linestyle='--',
             label=f'Val ROC (AUC = {auc_val:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle=':')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Transfer Learning (ECGSMARTNET_Attention SE)', fontsize=14)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"ROC curve saved to: {save_path}")


def plot_confusion_matrix(y_true, y_pred, threshold, save_path):
    """Plot confusion matrix."""
    y_pred_binary = (y_pred >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred_binary)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No CAD', 'CAD'], 
                yticklabels=['No CAD', 'CAD'])
    plt.title(f'Confusion Matrix - Transfer Learning (ECGSMARTNET_Attention SE)\n(Threshold = {threshold:.3f})', fontsize=14)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to: {save_path}")


def load_pretrained_model(pretrained_path, device, num_classes=2):
    """
    Load pretrained model and prepare for transfer learning (Option 1 Modified).
    Freezes all layers except layer4 (last residual block) and final FC layer.
    Uses ECGSMARTNET_Attention with SE (Squeeze-and-Excitation) blocks.
    
    Args:
        pretrained_path: Path to pretrained .pt file
        device: torch device
        num_classes: Number of output classes (default: 2 for binary CAD)
    
    Returns:
        model: Model with frozen layers (trainable: layer4 + FC)
        trainable_params: Number of trainable parameters
        total_params: Total number of parameters
    """
    print(f'\n{"="*80}')
    print('LOADING PRETRAINED MODEL')
    print(f'{"="*80}')
    print(f'Path: {pretrained_path}')
    
    # Load pretrained model
    pretrained_model = torch.load(pretrained_path, map_location=device, weights_only=False)
    
    # Get the model architecture details
    print(f'Pretrained model loaded successfully')
    print(f'Model type: {type(pretrained_model).__name__}')
    
    # Create new model with same architecture (with SE attention blocks)
    model = ECGSMARTNET_Attention(num_classes=num_classes, attention='se').to(device)
    
    # Copy pretrained weights (except final layer if num_classes differs)
    pretrained_dict = pretrained_model.state_dict()
    model_dict = model.state_dict()
    
    # Filter out fc layer (will be randomly initialized for new task)
    pretrained_dict_filtered = {k: v for k, v in pretrained_dict.items() 
                                 if k in model_dict and 'fc' not in k}
    
    # Update model dict with pretrained weights
    model_dict.update(pretrained_dict_filtered)
    model.load_state_dict(model_dict)
    
    print(f'✓ Transferred {len(pretrained_dict_filtered)} layers from pretrained model')
    print(f'✓ Final FC layer initialized randomly for CAD detection')
    
    # FREEZE all layers except layer4 and final FC layer
    print(f'\n{"="*80}')
    print('FREEZING LAYERS (Feature Extraction + Last Residual Block)')
    print(f'{"="*80}')
    
    for name, param in model.named_parameters():
        if 'layer4' in name or 'fc' in name:
            param.requires_grad = True
            print(f'  ✓ Trainable: {name}')
        else:
            param.requires_grad = False
            print(f'  Frozen: {name}')
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    print(f'\n{"="*80}')
    print('MODEL PARAMETER SUMMARY')
    print(f'{"="*80}')
    print(f'Total parameters:     {total_params:,}')
    print(f'Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)')
    print(f'Frozen parameters:    {frozen_params:,} ({frozen_params/total_params*100:.2f}%)')
    print(f'{"="*80}\n')
    
    return model, trainable_params, total_params


if __name__ == '__main__':
    # ============================================================
    # CONFIGURATION
    # ============================================================
    
    # Path to pretrained OMI model (USER WILL PROVIDE THIS)
    PRETRAINED_MODEL_PATH = 'models/ecgsmartnet_base_omi_2025-10-22-16-26-34.pt'  # CHANGE THIS
    
    # Data directory
    DATA_DIR = 'cad_dataset_preprocessed/'
    
    # Training parameters
    NUM_EPOCHS = 200           # Match training from scratch
    EARLY_STOP_PATIENCE = 10   # Stop if no improvement for 10 epochs
    
    # Random seed for data loading
    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    # ============================================================
    # HYPERPARAMETER SEARCH CONFIGURATION
    # ============================================================
    
    # Define hyperparameter search space with continuous distributions
    # Learning rates: sample from log-uniform distribution
    LR0_MIN, LR0_MAX = 1e-4, 1e-1  # Initial learning rate range (epoch 0)
    LR_MIN, LR_MAX = 1e-6, 1e-3    # Regular learning rate range (epochs 1+)
    WD_MIN, WD_MAX = 1e-5, 1e-1    # Weight decay range
    
    # Batch sizes: sample from discrete powers of 2
    BS_CHOICES = [16, 32, 64, 128, 256]
    
    # Number of random search iterations
    N_RANDOM_SEARCH = 100
    
    # ============= CHANGE THIS TO RESUME FROM SPECIFIC ITERATION =============
    START_ITERATION = 1  # Set to 1 to start from beginning, or higher to resume
    # ==========================================================================
    
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # ============================================================
    # SETUP
    # ============================================================
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'\n{"="*80}')
    print(f'TRANSFER LEARNING - ECGSMARTNET_ATTENTION (SE)')
    print(f'LAYER4 + FC FINE-TUNING WITH RANDOM SEARCH')
    print(f'{"="*80}')
    print(f'Using device: {device}')
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')
        print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')
    else:
        print('WARNING: No GPU detected, training will be SLOW on CPU!')
    print(f'{"="*80}\n')
    
    # ============================================================
    # LOAD DATA
    # ============================================================
    
    print(f'{"="*80}')
    print('LOADING DATA')
    print(f'{"="*80}')
    x_train, y_train, x_val, y_val, x_test, y_test = get_data(DATA_DIR)
    
    # Convert to tensors
    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    x_val = torch.tensor(x_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.long)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)
    
    # Create datasets
    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)
    
    print(f'\nClass distribution:')
    print(f'  Train: {torch.sum(y_train==0).item()} No CAD, {torch.sum(y_train==1).item()} CAD')
    print(f'  Val:   {torch.sum(y_val==0).item()} No CAD, {torch.sum(y_val==1).item()} CAD')
    print(f'  Test:  {torch.sum(y_test==0).item()} No CAD, {torch.sum(y_test==1).item()} CAD')
    print(f'{"="*80}\n')
    
    # ============================================================
    # RANDOM HYPERPARAMETER SEARCH
    # ============================================================
    
    # Track best model across all hyperparameter configurations
    global_best_val_loss = np.inf
    global_best_model_path = None
    global_best_hyperparams = None
    
    print(f"Starting random search with {N_RANDOM_SEARCH} iterations")
    print(f"Starting from iteration: {START_ITERATION}")
    print(f"Hyperparameter ranges:")
    print(f"  lr0: [{LR0_MIN:.1e}, {LR0_MAX:.1e}] (log-uniform)")
    print(f"  lr:  [{LR_MIN:.1e}, {LR_MAX:.1e}] (log-uniform)")
    print(f"  wd:  [{WD_MIN:.1e}, {WD_MAX:.1e}] (log-uniform)")
    print(f"  bs:  {BS_CHOICES} (uniform choice)")
    print()
    
    for count_search in range(1, N_RANDOM_SEARCH + 1):
        # Skip iterations before start_iteration
        if count_search < START_ITERATION:
            continue
        
        # Set seed for reproducible hyperparameter sampling
        np.random.seed(count_search * 42)
        
        # Sample hyperparameters from distributions
        lr0 = np.exp(np.random.uniform(np.log(LR0_MIN), np.log(LR0_MAX)))
        lr = np.exp(np.random.uniform(np.log(LR_MIN), np.log(LR_MAX)))
        wd = np.exp(np.random.uniform(np.log(WD_MIN), np.log(WD_MAX)))
        bs = int(np.random.choice(BS_CHOICES))
        
        print(f'\n{"="*80}')
        print(f'RANDOM SEARCH ITERATION: {count_search}/{N_RANDOM_SEARCH}')
        print(f'Sampled hyperparameters:')
        print(f'  lr0 (initial): {lr0:.6e}')
        print(f'  lr (main):     {lr:.6e}')
        print(f'  batch size:    {bs}')
        print(f'  weight decay:  {wd:.6e}')
        print(f'{"="*80}\n')
        
        try:
            # Set random seeds for this run
            torch.random.manual_seed(count_search)
            np.random.seed(count_search)
            
            current_time = time.strftime('%Y-%m-%d-%H-%M-%S')
            
            # Load pretrained model and setup transfer learning
            model, trainable_params, total_params = load_pretrained_model(
                PRETRAINED_MODEL_PATH, device, num_classes=2
            )
            
            # Setup optimizer (only for trainable parameters)
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=lr,
                weight_decay=wd
            )
            
            criterion = torch.nn.CrossEntropyLoss()
            
            # Weighted loss for validation
            pos_weight = torch.sum(y_val == 0) / torch.sum(y_val == 1)
            val_criterion = torch.nn.CrossEntropyLoss(
                weight=torch.tensor([1, pos_weight], dtype=torch.float32).to(device)
            )
            
            # Mixed precision training (only on GPU)
            use_amp = device.type == 'cuda'
            scaler = torch.amp.GradScaler(enabled=use_amp)
            
            # Data loaders with current batch size
            train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, 
                                      num_workers=4, pin_memory=True, persistent_workers=True)
            val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False,
                                    num_workers=2, pin_memory=True, persistent_workers=True)
            
            # Initialize wandb for this iteration
            wandb.init(
                project='ecgsmartnet-cad-transfer-learning-random-search',
                config={
                    'method': 'Transfer Learning - Layer4 + FC Fine-tuning',
                    'pretrained_model': PRETRAINED_MODEL_PATH,
                    'model': 'ECGSMARTNET_Attention (SE)',
                    'task': 'CAD Detection',
                    'attention_type': 'se',
                    'trainable_layers': 'layer4 + fc',
                    'optimizer': 'AdamW',
                    'num_epochs': NUM_EPOCHS,
                    'lr epoch0': lr0,
                    'lr': lr,
                    'batch_size': bs,
                    'weight_decay': wd,
                    'total_params': total_params,
                    'trainable_params': trainable_params,
                    'frozen_params': total_params - trainable_params,
                    'early_stop_patience': EARLY_STOP_PATIENCE,
                    'time': current_time,
                    'iteration': count_search
                }
            )
            
            # Training loop
            best_val_loss = np.inf
            count = 0
            
            for epoch in range(NUM_EPOCHS):
                # Set learning rate schedule (lr0 for epoch 0, lr for rest)
                if epoch == 0:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr0
                else:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr
                
                print(f'Epoch {epoch+1}/{NUM_EPOCHS}')
                
                # Train
                train_loss, train_auc, train_acc, train_prec, train_rec, train_spec, train_f1, train_ap = train_epoch(
                    model, device, train_loader, criterion, optimizer, scaler, use_amp
                )
                
                # Validate
                val_loss, val_auc, val_acc, val_prec, val_rec, val_spec, val_f1, val_ap, _, _ = val_epoch(
                    model, device, val_loader, val_criterion, use_amp
                )
                
                # Log to wandb
                wandb.log({
                    'Loss/Train': train_loss,
                    'AUC/Train': train_auc,
                    'AP/Train': train_ap,
                    'Loss/Validation': val_loss,
                    'AUC/Validation': val_auc,
                    'AP/Validation': val_ap,
                }, step=epoch)
                
                # Print metrics
                print(f'  Train Loss: {train_loss:.3f}, AUC: {train_auc:.3f}, AP: {train_ap:.3f}, F1: {train_f1:.3f}')
                print(f'  Val   Loss: {val_loss:.3f}, AUC: {val_auc:.3f}, AP: {val_ap:.3f}, F1: {val_f1:.3f}')
                
                # Save best model for this iteration
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    model_filename = f'models/transfer_learning_CAD_random_iter{count_search:03d}_{current_time}.pt'
                    torch.save(model, model_filename)
                    wandb.run.summary['best_val_loss'] = val_loss
                    wandb.run.summary['best_val_auc'] = val_auc
                    wandb.run.summary['best_val_ap'] = val_ap
                    wandb.run.summary['best_epoch'] = epoch
                    count = 0
                else:
                    count += 1
                
                # Early stopping
                if count == EARLY_STOP_PATIENCE:
                    print(f'  Early stopping at epoch {epoch+1}')
                    break
            
            # Check if this is the best model overall
            if best_val_loss < global_best_val_loss:
                global_best_val_loss = best_val_loss
                global_best_model_path = model_filename
                global_best_hyperparams = {
                    'lr0': lr0,
                    'lr': lr,
                    'bs': bs,
                    'wd': wd,
                    'val_loss': best_val_loss,
                    'val_auc': wandb.run.summary['best_val_auc'],
                    'val_ap': wandb.run.summary['best_val_ap'],
                    'time': current_time,
                    'iteration': count_search
                }
                print(f'\n*** NEW BEST MODEL FOUND! ***')
                print(f'  Iteration: {count_search}/{N_RANDOM_SEARCH}')
                print(f'  Validation Loss: {best_val_loss:.4f}')
                print(f'  Model saved to: {global_best_model_path}')
            
            wandb.finish()
            
            # Memory cleanup
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f'\n!!! ERROR in iteration {count_search} !!!')
            print(f'Error type: {type(e).__name__}')
            print(f'Error message: {str(e)}')
            print('Skipping this configuration and continuing search...\n')
            
            # Try to clean up
            try:
                wandb.finish()
            except:
                pass
            
            try:
                if 'model' in locals():
                    del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except:
                pass
            
            continue
    
    # ============================================================
    # EVALUATE BEST MODEL ON TEST SET
    # ============================================================
    
    print(f'\n{"="*80}')
    print('HYPERPARAMETER SEARCH COMPLETE')
    print(f'{"="*80}')
    
    # Check if any model was successfully trained
    if global_best_model_path is None or global_best_hyperparams is None:
        print('\n!!! ERROR: No models were successfully trained during the search !!!')
        print('All iterations may have failed. Check the error messages above.')
        print('Exiting without test evaluation.')
        exit(1)
    
    print(f'\nBest model details:')
    print(f'  Validation Loss: {global_best_hyperparams["val_loss"]:.4f}')
    print(f'  Validation AUC:  {global_best_hyperparams["val_auc"]:.3f}')
    print(f'  Validation AP:   {global_best_hyperparams["val_ap"]:.3f}')
    print(f'  lr0: {global_best_hyperparams["lr0"]:.6e}')
    print(f'  lr:  {global_best_hyperparams["lr"]:.6e}')
    print(f'  bs:  {global_best_hyperparams["bs"]}')
    print(f'  wd:  {global_best_hyperparams["wd"]:.6e}')
    print(f'  Model path: {global_best_model_path}')
    print(f'\n{"="*80}')
    print('Evaluating best model on test set...')
    print(f'{"="*80}')
    
    # Load best model
    best_model = torch.load(global_best_model_path)
    best_model.eval()
    
    # Create test dataset and loaders
    test_dataset = TensorDataset(x_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=global_best_hyperparams['bs'], shuffle=False,
                            num_workers=2, pin_memory=True)
    val_loader_final = DataLoader(val_dataset, batch_size=global_best_hyperparams['bs'], shuffle=False,
                                  num_workers=2, pin_memory=True)
    
    # Setup criterion
    criterion_final = torch.nn.CrossEntropyLoss()
    pos_weight = torch.sum(y_val == 0) / torch.sum(y_val == 1)
    val_criterion_final = torch.nn.CrossEntropyLoss(
        weight=torch.tensor([1, pos_weight], dtype=torch.float32).to(device)
    )
    
    # Evaluate on test set
    use_amp = device.type == 'cuda'
    test_loss, test_auc, test_acc, test_prec, test_rec, test_spec, test_f1, test_ap, y_test_true, y_test_pred = test_epoch(
        best_model, device, test_loader, criterion_final, use_amp
    )
    
    # Get validation predictions from best model
    _, val_auc_final, _, _, _, _, _, val_ap_final, y_val_true, y_val_pred = val_epoch(
        best_model, device, val_loader_final, val_criterion_final, use_amp
    )
    
    # Find optimal F1 threshold on validation set
    optimal_threshold, optimal_f1_val = find_optimal_f1_threshold(y_val_true, y_val_pred)
    
    print(f'\n{"="*80}')
    print('OPTIMAL THRESHOLD SELECTION (on Validation Set)')
    print(f'{"="*80}')
    print(f'  Optimal F1 Threshold: {optimal_threshold:.4f}')
    print(f'  Validation F1 at optimal threshold: {optimal_f1_val:.3f}')
    
    # Recompute validation metrics at optimal threshold
    val_acc_opt = accuracy_score(y_val_true, y_val_pred >= optimal_threshold)
    val_prec_opt = precision_score(y_val_true, y_val_pred >= optimal_threshold)
    val_rec_opt = recall_score(y_val_true, y_val_pred >= optimal_threshold)
    val_spec_opt = recall_score(y_val_true, y_val_pred >= optimal_threshold, pos_label=0)
    
    print(f'  Validation Acc:  {val_acc_opt:.3f}')
    print(f'  Validation Prec: {val_prec_opt:.3f}')
    print(f'  Validation Rec:  {val_rec_opt:.3f}')
    print(f'  Validation Spec: {val_spec_opt:.3f}')
    
    # Apply optimal threshold to test set
    test_acc_opt = accuracy_score(y_test_true, y_test_pred >= optimal_threshold)
    test_prec_opt = precision_score(y_test_true, y_test_pred >= optimal_threshold)
    test_rec_opt = recall_score(y_test_true, y_test_pred >= optimal_threshold)
    test_spec_opt = recall_score(y_test_true, y_test_pred >= optimal_threshold, pos_label=0)
    test_f1_opt = f1_score(y_test_true, y_test_pred >= optimal_threshold)
    
    # Print test metrics with optimal threshold
    print(f'\n{"="*80}')
    print('TEST SET RESULTS (using optimal F1 threshold)')
    print(f'{"="*80}')
    print(f'  Threshold used: {optimal_threshold:.4f}')
    print(f'  Loss: {test_loss:.3f}')
    print(f'  AUC:  {test_auc:.3f}')
    print(f'  AP:   {test_ap:.3f}')
    print(f'  Acc:  {test_acc_opt:.3f}')
    print(f'  Prec: {test_prec_opt:.3f}')
    print(f'  Rec:  {test_rec_opt:.3f}')
    print(f'  Spec: {test_spec_opt:.3f}')
    print(f'  F1:   {test_f1_opt:.3f}')
    
    # Create final wandb run for test evaluation
    wandb.init(
        project='ecgsmartnet-cad-transfer-learning-random-search', 
        name='FINAL_TEST_EVALUATION',
        config={
            'model': 'ECGSMARTNET_Attention (SE) - Transfer Learning', 
            'outcome': 'CAD',
            'attention_type': 'se',
            'optimizer': 'AdamW',
            'phase': 'test_evaluation',
            **global_best_hyperparams
        }
    )
    
    # Log test metrics to wandb (using optimal threshold)
    wandb.log({'Test/Loss': test_loss})
    wandb.log({'Test/AUC': test_auc})
    wandb.log({'Test/AP': test_ap})
    wandb.log({'Test/Optimal_Threshold': optimal_threshold})
    wandb.log({'Test/Accuracy': test_acc_opt})
    wandb.log({'Test/Precision': test_prec_opt})
    wandb.log({'Test/Recall': test_rec_opt})
    wandb.log({'Test/Specificity': test_spec_opt})
    wandb.log({'Test/F1': test_f1_opt})
    
    # Log validation metrics at optimal threshold
    wandb.log({'Validation/Optimal_F1': optimal_f1_val})
    wandb.log({'Validation/Accuracy_at_optimal': val_acc_opt})
    wandb.log({'Validation/Precision_at_optimal': val_prec_opt})
    wandb.log({'Validation/Recall_at_optimal': val_rec_opt})
    wandb.log({'Validation/Specificity_at_optimal': val_spec_opt})
    
    wandb.run.summary['test_auc'] = test_auc
    wandb.run.summary['test_ap'] = test_ap
    wandb.run.summary['optimal_threshold'] = optimal_threshold
    wandb.run.summary['test_f1_optimal'] = test_f1_opt
    
    # ============================================================
    # SAVE VISUALIZATIONS
    # ============================================================
    
    print(f'\n{"="*80}')
    print('GENERATING VISUALIZATIONS')
    print(f'{"="*80}')
    
    # Plot ROC and PR curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # ROC Curve
    fpr_val, tpr_val, thresholds_roc_val = roc_curve(y_val_true, y_val_pred)
    fpr_test, tpr_test, thresholds_roc_test = roc_curve(y_test_true, y_test_pred)
    
    ax1.plot(fpr_val, tpr_val, 'b-', linewidth=2, label=f'Validation (AUC={val_auc_final:.3f})')
    ax1.plot(fpr_test, tpr_test, 'r-', linewidth=2, label=f'Test (AUC={test_auc:.3f})')
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    
    # Mark the optimal threshold point on validation curve
    idx_val = np.argmin(np.abs(thresholds_roc_val - optimal_threshold))
    ax1.plot(fpr_val[idx_val], tpr_val[idx_val], 'b*', markersize=15, 
             label=f'Optimal Threshold (Val)', markeredgecolor='black', markeredgewidth=1)
    
    # Mark the optimal threshold point on test curve
    idx_test = np.argmin(np.abs(thresholds_roc_test - optimal_threshold))
    ax1.plot(fpr_test[idx_test], tpr_test[idx_test], 'r*', markersize=15, 
             label=f'Optimal Threshold (Test)', markeredgecolor='black', markeredgewidth=1)
    
    ax1.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax1.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax1.set_title('ROC Curve', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Precision-Recall Curve
    prec_val, rec_val, thresholds_pr_val = precision_recall_curve(y_val_true, y_val_pred)
    prec_test, rec_test, thresholds_pr_test = precision_recall_curve(y_test_true, y_test_pred)
    
    ax2.plot(rec_val, prec_val, 'b-', linewidth=2, label=f'Validation (AP={val_ap_final:.3f})')
    ax2.plot(rec_test, prec_test, 'r-', linewidth=2, label=f'Test (AP={test_ap:.3f})')
    
    # Mark the optimal threshold point on validation PR curve
    if len(thresholds_pr_val) > 0:
        idx_val_pr = np.argmin(np.abs(thresholds_pr_val - optimal_threshold))
        ax2.plot(rec_val[idx_val_pr], prec_val[idx_val_pr], 'b*', markersize=15,
                 label=f'Optimal Threshold (Val)', markeredgecolor='black', markeredgewidth=1)
    
    # Mark the optimal threshold point on test PR curve
    if len(thresholds_pr_test) > 0:
        idx_test_pr = np.argmin(np.abs(thresholds_pr_test - optimal_threshold))
        ax2.plot(rec_test[idx_test_pr], prec_test[idx_test_pr], 'r*', markersize=15,
                 label=f'Optimal Threshold (Test)', markeredgecolor='black', markeredgewidth=1)
    
    ax2.set_xlabel('Recall', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax2.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f'Best Model Performance (Iteration {global_best_hyperparams["iteration"]}, Optimal Threshold={optimal_threshold:.3f})\n' + 
                 f'Transfer Learning (ECGSMARTNET_Attention SE): Layer4+FC Fine-tuning (lr0={global_best_hyperparams["lr0"]:.2e}, lr={global_best_hyperparams["lr"]:.2e}, bs={global_best_hyperparams["bs"]}, wd={global_best_hyperparams["wd"]:.2e})', 
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    plot_path = f'models/transfer_learning_CAD_BEST_MODEL_iter{global_best_hyperparams["iteration"]:03d}_{global_best_hyperparams["time"]}_curves.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f'  ROC and PR curves saved to: {plot_path}')
    
    # Log plot to wandb
    wandb.log({"ROC and PR Curves": wandb.Image(plot_path)})
    
    plt.close()
    
    # Save confusion matrix
    cm_path = f'models/transfer_learning_CAD_BEST_MODEL_iter{global_best_hyperparams["iteration"]:03d}_{global_best_hyperparams["time"]}_cm.png'
    plot_confusion_matrix(y_test_true, y_test_pred, optimal_threshold, cm_path)
    print(f'  Confusion matrix saved to: {cm_path}')
    wandb.log({"Confusion Matrix": wandb.Image(cm_path)})
    
    # Save predictions
    pred_path = f'models/transfer_learning_CAD_BEST_MODEL_iter{global_best_hyperparams["iteration"]:03d}_{global_best_hyperparams["time"]}'
    np.save(f'{pred_path}_val_predictions.npy', y_val_pred)
    np.save(f'{pred_path}_test_predictions.npy', y_test_pred)
    print(f'  Predictions saved to: {pred_path}_val/test_predictions.npy')
    
    wandb.finish()
    
    print(f'\n{"="*80}')
    print('TRANSFER LEARNING WITH RANDOM SEARCH COMPLETE!')
    print(f'{"="*80}')
    print(f'\nBest model: {global_best_model_path}')
    print(f'Test AUC: {test_auc:.4f}')
    print(f'Test F1 (optimal threshold): {test_f1_opt:.4f}')
    print(f'{"="*80}\n')


