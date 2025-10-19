import os
# Fix OpenMP library conflict on Windows
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from ecg_models import *
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, f1_score, roc_curve, precision_recall_curve
import torchvision.models as models
from scipy import signal
import pandas as pd
import wandb
import time
import matplotlib.pyplot as plt
from tkinter.filedialog import askdirectory

def train_epoch(model, device, train_dataloader, criterion, optimizer, scaler, use_amp=True):
    train_loss = 0
    total_samples = 0
    model.train()

    ys = []
    y_preds = []

    for (x, y) in train_dataloader:

        # undersample the No ACS class
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
            
            # Unscale gradients and clip them to prevent exploding gradients (NaN issue)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()

            batch_size = y.size(0)
            train_loss += loss.item() * batch_size
            total_samples += batch_size

            # Check for NaN in loss
            if torch.isnan(loss):
                print("WARNING: NaN detected in training loss! Skipping batch.")
                continue

            y_pred = y_pred.cpu().detach().numpy()
            y = y.cpu().detach().numpy()
            ys.append(y)
            y_preds.append(y_pred)

    y = np.concatenate(ys, axis=0)
    y_pred = np.concatenate(y_preds, axis=0)
    train_loss /= total_samples

    y_pred = y_pred[:, 1]
    
    # Check for NaN/Inf in predictions
    if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
        print("ERROR: NaN or Inf detected in predictions!")
        print(f"  NaN count: {np.sum(np.isnan(y_pred))}")
        print(f"  Inf count: {np.sum(np.isinf(y_pred))}")
        # Replace NaN/Inf with 0.5 (neutral prediction) to allow training to continue
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

    y = np.concatenate(ys, axis=0)
    y_pred = np.concatenate(y_preds, axis=0)
    val_loss /= total_samples

    y_pred = y_pred[:, 1]
    
    # Check for NaN/Inf in validation predictions
    if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
        print("ERROR: NaN or Inf detected in VALIDATION predictions!")
        print(f"  NaN count: {np.sum(np.isnan(y_pred))}")
        print(f"  Inf count: {np.sum(np.isinf(y_pred))}")
        # Replace NaN/Inf with 0.5 (neutral prediction)
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
        # Disconnected leads (max=0) will remain zero after division (0/1=0)
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
        # Disconnected leads (max=0) will remain zero after division (0/1=0)
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
        # Disconnected leads (max=0) will remain zero after division (0/1=0)
        max_val = np.where(max_val == 0, 1, max_val)
        ecg = ecg / max_val
        
        test_data.append(ecg)
    
    test_data = np.array(test_data)
    print(f"Test set: {len(test_data)} files loaded")

    return train_data, train_outcomes, val_data, val_outcomes, test_data, test_outcomes

if __name__ == '__main__':
    # prompt user for path using browse
    path = 'cad_dataset_preprocessed/'
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'\n{"="*80}')
    print(f'Using device: {device}')
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')
        print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')
    else:
        print('WARNING: No GPU detected, training will be SLOW on CPU!')
    print(f'{"="*80}\n')
    # model = Temporal().to(device)

    x_train, y_train, x_val, y_val, x_test, y_test = get_data(path)

    # model is pretrained ResNet18 ############################################################################################################
    # model = models.resnet18(weights='DEFAULT')
    # model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    # model.fc = nn.Linear(model.fc.in_features, 2)
    # model = model.to(device)
    ############################################################################################################################################

    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    x_val = torch.tensor(x_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.long)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)
    num_epochs = 200

    # Define hyperparameter search space with continuous distributions
    # Learning rates: sample from log-uniform distribution (better for orders of magnitude)
    lr0_min, lr0_max = 1e-4, 1e-1  # Initial learning rate range
    lr_min, lr_max = 1e-6, 1e-3    # Regular learning rate range
    wd_min, wd_max = 1e-5, 1e-1    # Weight decay range
    
    # Batch sizes: sample from discrete powers of 2
    bs_choices = [16, 32, 64, 128, 256]
    
    # Number of random search iterations
    n_random_search = 100
    
    # ============= CHANGE THIS TO RESUME FROM SPECIFIC ITERATION =============
    start_iteration = 9  # Set to 1 to start from beginning, or higher to resume
    # ==========================================================================
    
    # Track best model across all hyperparameter configurations
    global_best_val_loss = np.inf
    global_best_model_path = None
    global_best_hyperparams = None
    
    print(f"Starting random search with {n_random_search} iterations")
    print(f"Starting from iteration: {start_iteration}")
    print(f"Hyperparameter ranges:")
    print(f"  lr0: [{lr0_min:.1e}, {lr0_max:.1e}] (log-uniform)")
    print(f"  lr:  [{lr_min:.1e}, {lr_max:.1e}] (log-uniform)")
    print(f"  wd:  [{wd_min:.1e}, {wd_max:.1e}] (log-uniform)")
    print(f"  bs:  {bs_choices} (uniform choice)")
    print()
    
    for count_search in range(1, n_random_search + 1):
        # Skip iterations before start_iteration
        if count_search < start_iteration:
                        continue
        
        # Sample hyperparameters from continuous distributions
        # Log-uniform sampling: better for hyperparameters spanning orders of magnitude
        # This ensures equal probability in log-space (e.g., 1e-5 to 1e-4 has same 
        # probability as 1e-2 to 1e-1, which makes sense for learning rates)
        lr0 = np.exp(np.random.uniform(np.log(lr0_min), np.log(lr0_max)))
        lr = np.exp(np.random.uniform(np.log(lr_min), np.log(lr_max)))
        wd = np.exp(np.random.uniform(np.log(wd_min), np.log(wd_max)))
        
        # Discrete uniform sampling for batch size (powers of 2)
        bs = int(np.random.choice(bs_choices))
        
        print(f'\n{"="*80}')
        print(f'RANDOM SEARCH ITERATION: {count_search}/{n_random_search}')
        print(f'Sampled hyperparameters:')
        print(f'  lr0 (initial): {lr0:.6e}')
        print(f'  lr (main):     {lr:.6e}')
        print(f'  batch size:    {bs}')
        print(f'  weight decay:  {wd:.6e}')
        print(f'{"="*80}\n')
        
        # Set random seeds for reproducibility of this specific run
        torch.random.manual_seed(count_search)
        np.random.seed(count_search)

        current_time = time.strftime('%Y-%m-%d-%H-%M-%S')
        model = ECGSMARTNET().to(device)
        wandb.init(project='ecgsmartnet-cad-random-search', 
                   config={'model': 'ECGSMARTNET', 
                           'outcome': 'CAD', 
                           'optimizer': 'AdamW',
                           'num_epochs': 200,
                           'lr epoch0': lr0,
                           'lr': lr,
                           'bs': bs,
                           'weight decay': wd,
                           'time': current_time
                    }
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        criterion = torch.nn.CrossEntropyLoss()
        pos_weight = torch.sum(y_val == 0) / torch.sum(y_val == 1)
        val_criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([1, pos_weight], dtype=torch.float32).to(device))
        
        # Only enable mixed precision on GPU
        use_amp = device.type == 'cuda'
        scaler = torch.amp.GradScaler(enabled=use_amp)

        train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, 
                                  num_workers=4, pin_memory=True, persistent_workers=True)
        val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False,
                                num_workers=2, pin_memory=True, persistent_workers=True)

        best_val_loss = np.inf
        count = 0
        for epoch in range(num_epochs):
            if epoch == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr0
            else:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr


            print(f'Epoch {epoch+1}/{num_epochs}')
            train_loss, train_auc, train_acc, train_prec, train_rec, train_spec, train_f1, train_ap = train_epoch(model, device, train_loader, criterion, optimizer, scaler, use_amp)
            val_loss, val_auc, val_acc, val_prec, val_rec, val_spec, val_f1, val_ap, _, _ = val_epoch(model, device, val_loader, val_criterion, use_amp)

            wandb.log({'Loss/Train': train_loss}, step=epoch)
            wandb.log({'AUC/Train': train_auc}, step=epoch)
            wandb.log({'AP/Train': train_ap}, step=epoch)
            wandb.log({'Loss/Validation': val_loss}, step=epoch)
            wandb.log({'AUC/Validation': val_auc}, step=epoch)
            wandb.log({'AP/Validation': val_ap}, step=epoch)

            print('Train Loss: {:.3f}, Train AUC: {:.3f}, Train AP: {:.3f}, Train Acc: {:.3f}, Train Prec: {:.3f}, Train Rec: {:.3f}, Train Spec: {:.3f}, Train F1: {:.3f}'.format(train_loss, train_auc, train_ap, train_acc, train_prec, train_rec, train_spec, train_f1))
            print('Val Loss: {:.3f}, Val AUC: {:.3f}, Val AP: {:.3f}, Val Acc: {:.3f}, Val Prec: {:.3f}, Val Rec: {:.3f}, Val Spec: {:.3f}, Val F1: {:.3f}'.format(val_loss, val_auc, val_ap, val_acc, val_prec, val_rec, val_spec, val_f1))

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model, 'models/ecgsmartnet_CAD_random_{}.pt'.format(current_time))
                wandb.run.summary['best_val_loss'] = val_loss
                wandb.run.summary['best_val_auc'] = val_auc
                wandb.run.summary['best_val_ap'] = val_ap
                wandb.run.summary['best_epoch'] = epoch
                count = 0
            else:
                count +=1
            
            if count == 10:
                break
        
        # Check if this is the best model overall across all hyperparameter configurations
        if best_val_loss < global_best_val_loss:
            global_best_val_loss = best_val_loss
            global_best_model_path = 'models/ecgsmartnet_CAD_random_{}.pt'.format(current_time)
            global_best_hyperparams = {
                'lr0': lr0,
                'lr': lr,
                'bs': bs,
                'wd': wd,
                'val_loss': best_val_loss,
                'val_auc': wandb.run.summary['best_val_auc'],
                'val_ap': wandb.run.summary['best_val_ap'],
                'time': current_time
            }
            print(f'\n*** NEW BEST MODEL FOUND! ***')
            print(f'  Validation Loss: {best_val_loss:.4f}')
            print(f'  Model saved to: {global_best_model_path}')
        
        wandb.finish()
    
    # After all hyperparameter search iterations, evaluate ONLY the best model on test set
    print(f'\n{"="*80}')
    print('HYPERPARAMETER SEARCH COMPLETE')
    print(f'{"="*80}')
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
    
    # Create test dataloader
    test_dataset = TensorDataset(x_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=global_best_hyperparams['bs'], shuffle=False,
                            num_workers=2, pin_memory=True)
    
    # Create validation dataloader with best batch size
    val_loader_final = DataLoader(val_dataset, batch_size=global_best_hyperparams['bs'], shuffle=False,
                                  num_workers=2, pin_memory=True)
    
    # Recreate criterion with best hyperparameters
    criterion_final = torch.nn.CrossEntropyLoss()
    pos_weight = torch.sum(y_val == 0) / torch.sum(y_val == 1)
    val_criterion_final = torch.nn.CrossEntropyLoss(weight=torch.tensor([1, pos_weight], dtype=torch.float32).to(device))
    
    # Evaluate test set
    use_amp = device.type == 'cuda'
    test_loss, test_auc, test_acc, test_prec, test_rec, test_spec, test_f1, test_ap, y_test_true, y_test_pred = test_epoch(best_model, device, test_loader, criterion_final, use_amp)
    
    # Get validation predictions from best model
    _, val_auc_final, _, _, _, _, _, val_ap_final, y_val_true, y_val_pred = val_epoch(best_model, device, val_loader_final, val_criterion_final, use_amp)
    
    # Print test metrics
    print('\nTest Set Results:')
    print(f'  Loss: {test_loss:.3f}')
    print(f'  AUC:  {test_auc:.3f}')
    print(f'  AP:   {test_ap:.3f}')
    print(f'  Acc:  {test_acc:.3f}')
    print(f'  Prec: {test_prec:.3f}')
    print(f'  Rec:  {test_rec:.3f}')
    print(f'  Spec: {test_spec:.3f}')
    print(f'  F1:   {test_f1:.3f}')
    
    # Create final wandb run for test evaluation
    wandb.init(project='ecgsmartnet-cad-random-search', 
               name='FINAL_TEST_EVALUATION',
               config={
                   'model': 'ECGSMARTNET', 
                   'outcome': 'CAD',
                   'optimizer': 'AdamW',
                   'phase': 'test_evaluation',
                   **global_best_hyperparams
               })
    
    # Log test metrics to wandb
    wandb.log({'Test/Loss': test_loss})
    wandb.log({'Test/AUC': test_auc})
    wandb.log({'Test/AP': test_ap})
    wandb.log({'Test/Accuracy': test_acc})
    wandb.log({'Test/Precision': test_prec})
    wandb.log({'Test/Recall': test_rec})
    wandb.log({'Test/Specificity': test_spec})
    wandb.log({'Test/F1': test_f1})
    
    wandb.run.summary['test_auc'] = test_auc
    wandb.run.summary['test_ap'] = test_ap
    
    # Plot ROC and PR curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # ROC Curve
    fpr_val, tpr_val, _ = roc_curve(y_val_true, y_val_pred)
    fpr_test, tpr_test, _ = roc_curve(y_test_true, y_test_pred)
    
    ax1.plot(fpr_val, tpr_val, 'b-', linewidth=2, label=f'Validation (AUC={val_auc_final:.3f})')
    ax1.plot(fpr_test, tpr_test, 'r-', linewidth=2, label=f'Test (AUC={test_auc:.3f})')
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    ax1.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax1.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax1.set_title('ROC Curve', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Precision-Recall Curve
    prec_val, rec_val, _ = precision_recall_curve(y_val_true, y_val_pred)
    prec_test, rec_test, _ = precision_recall_curve(y_test_true, y_test_pred)
    
    ax2.plot(rec_val, prec_val, 'b-', linewidth=2, label=f'Validation (AP={val_ap_final:.3f})')
    ax2.plot(rec_test, prec_test, 'r-', linewidth=2, label=f'Test (AP={test_ap:.3f})')
    ax2.set_xlabel('Recall', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax2.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f'Best Model Performance (lr0={global_best_hyperparams["lr0"]:.2e}, lr={global_best_hyperparams["lr"]:.2e}, bs={global_best_hyperparams["bs"]}, wd={global_best_hyperparams["wd"]:.2e})', 
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    plot_path = f'models/ecgsmartnet_CAD_random_BEST_MODEL_{global_best_hyperparams["time"]}_curves.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f'\nPlots saved to: {plot_path}')
    
    # Log plot to wandb
    wandb.log({"ROC and PR Curves": wandb.Image(plot_path)})
    
    plt.close()
    
    wandb.finish()
    
    print(f'\n{"="*80}')
    print('TEST EVALUATION COMPLETE')
    print(f'{"="*80}')
