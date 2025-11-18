import os
from ecg_models import *
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, f1_score
from scipy import signal
import pandas as pd
import wandb
import time


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

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
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

            y_pred = y_pred.cpu().detach().numpy()
            y = y.cpu().detach().numpy()
            ys.append(y)
            y_preds.append(y_pred)


    y = np.concatenate(ys, axis=0)
    y_pred = np.concatenate(y_preds, axis=0)
    train_loss /= total_samples

    y_pred = y_pred[:, 1]
    y_pred_binary = y_pred > 0.5

    auc = roc_auc_score(y, y_pred)
    ap = average_precision_score(y, y_pred)
    acc = accuracy_score(y, y_pred_binary)
    prec = precision_score(y, y_pred_binary)
    rec = recall_score(y, y_pred_binary)
    spec = recall_score(y, y_pred_binary, pos_label=0)
    f1 = f1_score(y, y_pred_binary)

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

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
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
    y_pred_binary = y_pred > 0.5

    auc = roc_auc_score(y, y_pred)
    ap = average_precision_score(y, y_pred)
    acc = accuracy_score(y, y_pred_binary)
    prec = precision_score(y, y_pred_binary)
    rec = recall_score(y, y_pred_binary)
    spec = recall_score(y, y_pred_binary, pos_label=0)
    f1 = f1_score(y, y_pred_binary)

    return val_loss, auc, acc, prec, rec, spec, f1, ap, y, y_pred


def get_data(path):
 
    train_df = pd.read_csv('train_set.csv')
    val_df = pd.read_csv('val_set.csv')
    train_outcomes = train_df['label'].to_numpy()
    val_outcomes = val_df['label'].to_numpy()
    train_ids = train_df['ID'].to_list()
    val_ids = val_df['ID'].to_list()

    # get train data
    train_data = []
    for id in train_ids:

        ecg = np.load(os.path.join(path, id + '.npy'), allow_pickle=True).item()
        ecg = ecg['waveforms']['ecg_median']
        ecg = ecg[:,150:-50]
        ecg = signal.resample(ecg, 200, axis=1)
        max_val = np.max(np.abs(ecg), axis=1)
        if np.sum(max_val) > 0:
            ecg = ecg / max_val[:, None]
        train_data.append(ecg)    
    train_data = np.array(train_data)
    print(f"Train set: {len(train_data)} files loaded")

    val_data = []
    for id in val_ids:

        ecg = np.load(os.path.join(path, id + '.npy'), allow_pickle=True).item()
        ecg = ecg['waveforms']['ecg_median']
        ecg = ecg[:,150:-50]
        ecg = signal.resample(ecg, 200, axis=1)
        max_val = np.max(np.abs(ecg), axis=1)
        if np.sum(max_val) > 0:
            ecg = ecg / max_val[:, None]
        val_data.append(ecg)    
    val_data = np.array(val_data)
    print(f"Val set: {len(val_data)} files loaded")

    return train_data, train_outcomes, val_data, val_outcomes


def freeze_all_except_fc(model, verbose=True):
    """
    Freeze all layers in the model except the final fully connected layer (fc).
    """
    # First, freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Then, unfreeze only the fc layer
    for param in model.fc.parameters():
        param.requires_grad = True
    
    if verbose:
        # Print trainable parameters summary
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
        
        # Print detailed layer-by-layer status
        print("\n" + "="*80)
        print("Model layers and their freeze status:")
        print("="*80)
        for name, param in model.named_parameters():
            status = "Trainable" if param.requires_grad else "Frozen"
            num_params = param.numel()
            print(f"{status:12} | {num_params:>10,} params | {name}")
        print("="*80 + "\n")
    

if __name__ == '__main__':

    # Path to preprocessed CAD data
    path = 'cad_dataset_preprocessed/'
    
    # Path to pretrained model
    pretrained_model_path = 'models/ecgsmartnet_attention_se_acs_2025-09-23-12-03-20.pt'

    os.makedirs('models', exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load CAD data
    x_train, y_train, x_val, y_val = get_data(path)

    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    x_val = torch.tensor(x_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.long)

    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)
    num_epochs = 200
    
    # Calculate pos_weight (used for validation criterion)
    pos_weight = torch.sum(y_val == 0) / torch.sum(y_val == 1)

    # Random search hyperparameter ranges
    lr0_min, lr0_max = 1e-4, 1e-1  # Initial learning rate range
    lr_min, lr_max = 1e-6, 1e-3    # Regular learning rate range
    wd_min, wd_max = 1e-5, 1e-1    # Weight decay range
    bs_choices = [16, 32, 64, 128, 256] # batch size choices

    n_random_search = 100
    
    start_iteration = 1  # Set to 1 to start from beginning, or higher to resume

    
    for count_search in range(1, n_random_search + 1):
        # Skip iterations before start_iteration
        if count_search < start_iteration:
            continue
        
        # Set random seeds for reproducibility of this specific run
        torch.manual_seed(count_search)
        np.random.seed(count_search * 42)
        
        # Sample hyperparameters from continuous distributions
        lr0 = np.exp(np.random.uniform(np.log(lr0_min), np.log(lr0_max)))
        lr = np.exp(np.random.uniform(np.log(lr_min), np.log(lr_max)))
        wd = np.exp(np.random.uniform(np.log(wd_min), np.log(wd_max)))
        # Discrete uniform sampling for batch size
        bs = int(np.random.choice(bs_choices))

        print(f'\n{"="*80}')
        print(f'Random search iteration: {count_search}/{n_random_search}')
        print(f'Sampled hyperparameters:')
        print(f'  lr0 (initial): {lr0:.6e}')
        print(f'  lr (main):     {lr:.6e}')
        print(f'  batch size:    {bs}')
        print(f'  weight decay:  {wd:.6e}')
        print(f'{"="*80}')
        
        try:

            # Reload the pretrained model for each iteration (to reset weights)
            model = torch.load(pretrained_model_path, map_location=device, weights_only=False)
            # Print layer status only on first iteration
            freeze_all_except_fc(model, verbose=(count_search == start_iteration))

            # Get model name from the loaded model class
            model_name = model.__class__.__name__
            
            # Detect attention mechanism if using ECGSMARTNET_Attention
            has_attention = hasattr(model, 'layer2') and hasattr(model.layer2[0], 'attn')
            attention_type = 'none'
            if has_attention and model.layer2[0].attn is not None:
                if isinstance(model.layer2[0].attn, SEBlock):
                    attention_type = 'SE'
                elif isinstance(model.layer2[0].attn, CBAM):
                    attention_type = 'CBAM'
            
            if count_search == start_iteration:
                print(f'Model: {model_name}')
                print(f'Attention mechanism: {attention_type}')
            
            current_time = time.strftime('%Y-%m-%d-%H-%M-%S')
            wandb.init(project='ecgsmartnet-cad-transfer-learning', 
                       config={'model': model_name, 
                               'outcome': 'CAD', 
                               'optimizer': 'AdamW',
                               'num_epochs': num_epochs,
                               'lr epoch0': lr0,
                               'lr': lr,
                               'bs': bs,
                               'weight decay': wd,
                               'pretrained_model': pretrained_model_path,
                               'transfer_learning': True,
                               'frozen_layers': 'all except fc',
                               'attention': attention_type,
                               'time': current_time
                        }
            )

            # Only optimize the unfrozen parameters (fc layer)
            # Initialize with lr0 for first epoch, will be changed to lr after epoch 0
            optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                                         lr=lr0, weight_decay=wd)
            criterion = torch.nn.CrossEntropyLoss()
            val_criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([1, pos_weight], dtype=torch.float32).to(device))
            

            use_amp = device.type == 'cuda'
            scaler = torch.amp.GradScaler(enabled=use_amp)

            # Set num_workers based on platform (Windows can have issues with multiprocessing)
            num_workers = 0 if os.name == 'nt' else 4
            train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, 
                                      num_workers=num_workers, pin_memory=True, 
                                      persistent_workers=(num_workers > 0))
            val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False,
                                    num_workers=num_workers, pin_memory=True, 
                                    persistent_workers=(num_workers > 0))

            best_val_loss = np.inf
            patience_counter = 0
            for epoch in range(num_epochs):
                # Change learning rate from lr0 to lr after first epoch
                if epoch == 1:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr


                print(f'Epoch {epoch+1}/{num_epochs}')
                train_loss, train_auc, train_acc, train_prec, train_rec, train_spec, train_f1, train_ap = train_epoch(model, device, train_loader, criterion, optimizer, scaler, use_amp)
                val_loss, val_auc, val_acc, val_prec, val_rec, val_spec, val_f1, val_ap, _, _ = val_epoch(model, device, val_loader, val_criterion, use_amp)

                # Log all metrics in a single call for efficiency
                wandb.log({
                    'Loss/Train': train_loss,
                    'AUC/Train': train_auc,
                    'AP/Train': train_ap,
                    'Loss/Validation': val_loss,
                    'AUC/Validation': val_auc,
                    'AP/Validation': val_ap
                }, step=epoch)

                print('Train Loss: {:.3f}, Train AUC: {:.3f}, Train AP: {:.3f}, Train Acc: {:.3f}, Train Prec: {:.3f}, Train Rec: {:.3f}, Train Spec: {:.3f}, Train F1: {:.3f}'.format(train_loss, train_auc, train_ap, train_acc, train_prec, train_rec, train_spec, train_f1))
                print('Val Loss: {:.3f}, Val AUC: {:.3f}, Val AP: {:.3f}, Val Acc: {:.3f}, Val Prec: {:.3f}, Val Rec: {:.3f}, Val Spec: {:.3f}, Val F1: {:.3f}'.format(val_loss, val_auc, val_ap, val_acc, val_prec, val_rec, val_spec, val_f1))

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    model_filename = f'models/transfer_learning_CAD__{current_time}.pt'
                    torch.save(model, model_filename)
                    wandb.run.summary['best_val_loss'] = val_loss
                    wandb.run.summary['best_val_auc'] = val_auc
                    wandb.run.summary['best_val_ap'] = val_ap
                    wandb.run.summary['best_epoch'] = epoch
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter == 10:
                    print("Early stopping triggered (no improvement for 10 epochs)")
                    break
            
            wandb.finish()
            
            # Free memory
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f'\nError in iteration {count_search}')
            print(f'Error type: {type(e).__name__}')
            print(f'Error message: {str(e)}')
            
            # Finish wandb
            try:
                wandb.finish()
            except:
                pass
            
            # Free memory
            try:
                if 'model' in locals():
                    del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except:
                pass
            
            continue

