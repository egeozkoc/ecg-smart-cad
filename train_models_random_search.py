from ecg_models import *
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import wandb
import time
import os
import random


def train_epoch(model, device, train_dataloader, criterion, optimizer, scaler):
    train_loss = 0
    total_samples = 0
    model.train()
    ys = []
    y_preds = []

    for (x, y) in train_dataloader:

        # undersample the No ACS class (keep all positives, sample negatives to match)
        with torch.no_grad():
            idx0 = (y == 0).nonzero(as_tuple=False).squeeze(1)
            idx1 = (y == 1).nonzero(as_tuple=False).squeeze(1)
            num_samples = int(min(idx0.numel(), idx1.numel()))

        if num_samples > 0:
            sampled_neg = idx0[torch.randperm(idx0.numel())[:num_samples]]
            indices = torch.cat([sampled_neg, idx1], dim=0)
            indices = indices[torch.randperm(indices.numel())]

            x = x.index_select(0, indices)
            y = y.index_select(0, indices)
            x = torch.unsqueeze(x, 1)

            x = x.to(device)
            y = y.to(device)

            with torch.amp.autocast(device_type=device.type, enabled=torch.cuda.is_available()):
                y_pred = model(x)
                loss = criterion(y_pred, y)
                y_pred = torch.softmax(y_pred, dim=-1)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_size = y.size(0)
            train_loss += loss.item() * batch_size
            total_samples += batch_size

            y_pred = y_pred.cpu().detach().to(torch.float32).numpy()
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

def val_epoch(model, device, val_dataloader, criterion):
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

            with torch.amp.autocast(device_type=device.type, enabled=torch.cuda.is_available()):
                y_pred = model(x)
                loss = criterion(y_pred, y)
                y_pred = torch.softmax(y_pred, dim=-1)

            batch_size = y.size(0)
            val_loss += loss.item() * batch_size
            total_samples += batch_size

            y_pred = y_pred.cpu().detach().to(torch.float32).numpy()
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

    return val_loss, auc, acc, prec, rec, spec, f1, ap

def get_data(base_path, selected_outcome):
    # Load CSVs for IDs and labels
    train_df = pd.read_csv(f'{base_path}/train_data.csv')
    val_df = pd.read_csv(f'{base_path}/val_data.csv')
    test_df = pd.read_csv(f'{base_path}/test_data.csv')

    train_ids = train_df['id'].to_list()
    val_ids = val_df['id'].to_list()
    test_ids = test_df['id'].to_list()
    train_outcomes = train_df[selected_outcome].to_numpy()
    val_outcomes = val_df[selected_outcome].to_numpy()
    test_outcomes = test_df[selected_outcome].to_numpy()

    # Load preprocessed .npy files (no further preprocessing)
    def load_data(ids, split):
        data = []
        for id in ids:
            arr = np.load(f'{base_path}/{split}_data/{id}.npy', allow_pickle=True)
            data.append(arr)
        return np.array(data)

    train_data = load_data(train_ids, 'train')
    val_data = load_data(val_ids, 'val')
    test_data = load_data(test_ids, 'test')

    return train_data, train_outcomes, val_data, val_outcomes, test_data, test_outcomes

if __name__ == '__main__':
    # Use preprocessed_median_beats as base path
    base_path = 'preprocessed_median_beats'
    selected_outcome = 'acs' # 'acs' or 'omi'

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # if torch.backends.mps.is_available():
    #     device = torch.device("mps")
    # else:
    #     device = torch.device("cpu")

    # print(f"Using device: {device}")


    x_train, y_train, x_val, y_val, x_test, y_test = get_data(base_path, selected_outcome)

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
    test_dataset = TensorDataset(x_test, y_test)
    num_epochs = 200

    torch.random.manual_seed(0)
    np.random.seed(0)

    search_space = {
        "lr0":  lambda: 10**np.random.uniform(-4, -1),
        "lr":  lambda: 10**np.random.uniform(-7, -4),
        "bs":  lambda: random.choice([32, 64, 128, 256]),
        "wd":  lambda: 10**np.random.uniform(-4, 0),
    }

    attention = ['se', 'cbam']
    reduction = [16, 8]

    for red in reduction:
        for attn in attention:
            n_trials = 100
            for t in range(0, n_trials):
                cfg = {k: v() for k, v in search_space.items()}
                print(f"Trial {t+1}/{n_trials}: {cfg}")

                lr0 = cfg['lr0']
                lr = cfg['lr']
                bs = cfg['bs']
                wd = cfg['wd']
                
                
                current_time = time.strftime('%Y-%m-%d-%H-%M-%S')
                model = ECGSMARTNET_Attention(reduction=red, attention=attn, dropout=True).to(device)
                wandb.init(project='ecgsmartnet-attention-dropout-0.5-V4',
                            config={'model': 'ECGSMARTNET_Attention', 
                                    'outcome': selected_outcome,
                                    'num_epochs': 200,
                                    'lr0': lr0,
                                    'lr': lr,
                                    'bs': bs,
                                    'weight decay': wd,
                                    'reduction': red,
                                    'attention': attn,
                                    'time': current_time
                            }
                )

                optimizer = torch.optim.AdamW(model.parameters(), lr=lr0, weight_decay=wd)
                criterion = torch.nn.CrossEntropyLoss()
                neg_count = int(torch.sum(y_val == 0).item())
                pos_count = int(torch.sum(y_val == 1).item())
                pos_weight_ratio = float(neg_count / max(pos_count, 1))
                class_weights = torch.tensor([1.0, pos_weight_ratio], dtype=torch.float32, device=device)
                val_criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
                
                scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

                train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False)
                test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False)

                best_val_loss = np.inf
                best_model_path = None
                count = 0
                for epoch in range(num_epochs):
                    lr_current = lr0 if epoch == 0 else lr
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_current


                    print(f'Epoch {epoch+1}/{num_epochs}')
                    train_loss, train_auc, train_acc, train_prec, train_rec, train_spec, train_f1, train_ap = train_epoch(model, device, train_loader, criterion, optimizer, scaler)
                    val_loss, val_auc, val_acc, val_prec, val_rec, val_spec, val_f1, val_ap = val_epoch(model, device, val_loader, val_criterion)

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
                        os.makedirs('models', exist_ok=True)
                        save_path = f"models/ecgsmartnet_attention_{selected_outcome}_{current_time}.pt"
                        torch.save(model, save_path)
                        best_model_path = save_path
                        wandb.run.summary['best_val_loss'] = val_loss
                        wandb.run.summary['best_val_auc'] = val_auc
                        wandb.run.summary['best_val_ap'] = val_ap
                        wandb.run.summary['best_epoch'] = epoch
                        count = 0
                    else:
                        count +=1
                    
                    if count == 10:
                        break
                # Evaluate the best saved model on the test set and log to W&B
                if best_model_path is not None:
                    best_model = torch.load(best_model_path, map_location=device)
                    best_model = best_model.to(device)
                    test_loss, test_auc, test_acc, test_prec, test_rec, test_spec, test_f1, test_ap = val_epoch(best_model, device, test_loader, val_criterion)
                    wandb.log({'Loss/Test': test_loss})
                    wandb.log({'AUC/Test': test_auc})
                    wandb.log({'Acc/Test': test_acc})
                    wandb.log({'Prec/Test': test_prec})
                    wandb.log({'Rec/Test': test_rec})
                    wandb.log({'Spec/Test': test_spec})
                    wandb.log({'F1/Test': test_f1})
                    wandb.log({'AP/Test': test_ap})
                    wandb.run.summary['test_loss'] = test_loss
                    wandb.run.summary['test_auc'] = test_auc
                    wandb.run.summary['test_acc'] = test_acc
                    wandb.run.summary['test_prec'] = test_prec
                    wandb.run.summary['test_rec'] = test_rec
                    wandb.run.summary['test_spec'] = test_spec
                    wandb.run.summary['test_f1'] = test_f1
                    wandb.run.summary['test_ap'] = test_ap
                wandb.finish()