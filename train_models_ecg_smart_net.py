import os
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

            y_pred = y_pred.cpu().detach().numpy()
            y = y.cpu().detach().numpy()
            ys.append(y)
            y_preds.append(y_pred)


    y = np.concatenate(ys, axis=0)
    y_pred = np.concatenate(y_preds, axis=0)
    train_loss /= total_samples

    y_pred = y_pred[:, 1]

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
    

    auc = roc_auc_score(y, y_pred)
    ap = average_precision_score(y, y_pred)
    acc = accuracy_score(y, y_pred > 0.5)
    prec = precision_score(y, y_pred > 0.5)
    rec = recall_score(y, y_pred > 0.5)
    spec = recall_score(y, y_pred > 0.5, pos_label=0)
    f1 = f1_score(y, y_pred > 0.5)

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
        ecg = np.load(path + id + '.npy', allow_pickle=True).item()
        ecg = ecg['waveforms']['ecg_median']
        ecg = ecg[:,150:-50]
        ecg = signal.resample(ecg, 200, axis=1)
        max_val = np.max(np.abs(ecg), axis=1, keepdims=True)
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
        ecg = ecg / max_val
        val_data.append(ecg)
    
    val_data = np.array(val_data)
    print(f"Val set: {len(val_data)} files loaded")

    return train_data, train_outcomes, val_data, val_outcomes

if __name__ == '__main__':

    path = 'cad_dataset_preprocessed/'
    
    os.makedirs('models', exist_ok=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    x_train, y_train, x_val, y_val = get_data(path)


    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    x_val = torch.tensor(x_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.long)

    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)
    num_epochs = 200


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
        
        np.random.seed(count_search * 42)  # Unique seed per iteration
        
        # Sample hyperparameters from continuous distributions
        lr0 = np.exp(np.random.uniform(np.log(lr0_min), np.log(lr0_max)))
        lr = np.exp(np.random.uniform(np.log(lr_min), np.log(lr_max)))
        wd = np.exp(np.random.uniform(np.log(wd_min), np.log(wd_max)))
        
        # Discrete uniform sampling for batch size
        bs = int(np.random.choice(bs_choices))

        print(f'Random search iteration: {count_search}/{n_random_search}')
        print(f'Sampled hyperparameters:')
        print(f'  lr0 (initial): {lr0:.6e}')
        print(f'  lr (main):     {lr:.6e}')
        print(f'  batch size:    {bs}')
        print(f'  weight decay:  {wd:.6e}')
        
        try:
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
                    model_filename = f'models/ecgsmartnet_CAD__{current_time}.pt'
                    torch.save(model, model_filename)
                    wandb.run.summary['best_val_loss'] = val_loss
                    wandb.run.summary['best_val_auc'] = val_auc
                    wandb.run.summary['best_val_ap'] = val_ap
                    wandb.run.summary['best_epoch'] = epoch
                    count = 0
                else:
                    count +=1
                
                if count == 10:
                    break
            
            wandb.finish()
            
            # Memory cleanup - free GPU memory
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f'\nError in iteration {count_search}')
            print(f'Error type: {type(e).__name__}')
            print(f'Error message: {str(e)}')
            
            # Try to clean up and finish wandb
            try:
                wandb.finish()
            except:
                pass
            
            # Try to free memory
            try:
                if 'model' in locals():
                    del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except:
                pass
            
            continue
