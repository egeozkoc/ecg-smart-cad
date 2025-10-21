from ecg_models import *
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, f1_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from scipy import signal


def load_split_data(split):
    """Load a split's data (train/val/test) for evaluation.
    
    Args:
        split: 'train', 'val', or 'test'
    
    Returns:
        data: numpy array of preprocessed ECG signals
        outcomes: numpy array of labels
    """
    # Read CSV for labels
    split_csv = f'{split}_set.csv'
    df = pd.read_csv(split_csv)
    ids = df['ID'].to_list()
    outcomes = df['label'].to_numpy()

    # Load and preprocess data
    data_list = []
    for sample_id in ids:
        # Load from the respective folder
        ecg = np.load(f'cad_dataset_preprocessed/{split}_set/{sample_id}.npy', allow_pickle=True).item()
        # Extract ECG median from dictionary
        ecg = ecg['waveforms']['ecg_median']
        # Slice to remove edges
        ecg = ecg[:, 150:-50]
        # Resample to 200 samples
        ecg = signal.resample(ecg, 200, axis=1)
        # Normalize by max value (handle division by zero for disconnected leads)
        max_val = np.max(np.abs(ecg), axis=1, keepdims=True)
        max_val = np.where(max_val == 0, 1, max_val)  # Replace zeros with 1
        ecg = ecg / max_val
        
        data_list.append(ecg)
    
    data = np.array(data_list)
    print(f"{split.capitalize()} set: {len(data)} files loaded")
    return data, outcomes


def evaluate_split(model, device, dataloader, criterion):
    """Evaluate a split and return base outputs for metric computation."""
    model.eval()
    total_loss = 0
    total_samples = 0

    all_y_true = []
    all_pred_probs = []

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(dataloader):
            x = x.to(device)
            y = y.to(device)
            x = torch.unsqueeze(x, 1)

            with torch.amp.autocast(device.type, enabled=True):
                logits = model(x)
                loss = criterion(logits, y)
                probs = torch.softmax(logits, dim=-1)

            batch_size = y.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            all_pred_probs.append(probs[:, 1].detach().cpu().to(torch.float32).numpy())
            all_y_true.append(y.detach().cpu().numpy())
            
            # Print progress every 5 batches
            if (batch_idx + 1) % 5 == 0:
                print(f"  Processed {total_samples} samples...", flush=True)

    print(f"  Completed processing {total_samples} samples. Aggregating results...", flush=True)
    y_true = np.concatenate(all_y_true, axis=0)
    y_prob_pos = np.concatenate(all_pred_probs, axis=0)
    avg_loss = total_loss / max(total_samples, 1)
    return avg_loss, y_true, y_prob_pos


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
    """Plot confusion matrix and save it"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No CAD', 'CAD'], 
                yticklabels=['No CAD', 'CAD'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to: {save_path}")



def main():
    # Define model path in a single line
    model_path = 'models/ecgsmartnet_CAD_random_2025-10-18-19-56-03.pt'  # Update this path

    # Device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Load splits (data is already preprocessed)
    print('Loading validation data...')
    x_val_np, y_val_np = load_split_data('val')
    print('Loading test data...')
    x_test_np, y_test_np = load_split_data('test')

    # Build dataloaders
    x_val = torch.tensor(x_val_np, dtype=torch.float32)
    y_val = torch.tensor(y_val_np, dtype=torch.long)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=64, shuffle=False)

    x_test = torch.tensor(x_test_np, dtype=torch.float32)
    y_test = torch.tensor(y_test_np, dtype=torch.long)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=64, shuffle=False)

    # Load model
    print(f'Loading model from: {model_path}')
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()

    # Evaluate validation and test splits
    print('Evaluating validation set...')
    _, y_val_true, y_val_prob = evaluate_split(model, device, val_loader, criterion)
    print('✓ Validation evaluation complete')
    
    print('Evaluating test set...')
    _, y_test_true, y_test_prob = evaluate_split(model, device, test_loader, criterion)
    print('✓ Test evaluation complete')
    
    # Derive thresholds from validation split
    print('Finding optimal thresholds...')
    rule_out_thresh, rule_in_thresh, f1_thresh = find_thresholds_from_val(y_val_true, y_val_prob)
    print(f'✓ Thresholds found (F1 thresh: {f1_thresh:.3f})')
    
    # Compute validation metrics at F1 threshold
    print('Computing validation metrics...')
    val_metrics = compute_metrics(y_val_true, y_val_prob, f1_thresh)
    print('Computing validation bootstrap CIs (200 iterations, ~30-60 sec)...')
    val_ci = bootstrap_ci_val(y_val_true, y_val_prob)
    print('✓ Validation metrics complete')

    # Test metrics at F1-optimal threshold (from validation set)
    print('Computing test metrics...')
    test_metrics = compute_metrics(y_test_true, y_test_prob, f1_thresh)
    print('Computing test bootstrap CIs (200 iterations, ~30-60 sec)...')
    test_ci = bootstrap_ci_test(y_test_true, y_test_prob, f1_thresh)
    print('✓ Test metrics complete')

    # Confusion matrix at F1-optimal threshold
    test_preds = (y_test_prob >= f1_thresh).astype(int)
    cm = confusion_matrix(y_test_true, test_preds)

    # Print results in requested format (single-column model)
    print('=' * 80)
    print('Results for {}'.format(model_path))
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


    print(f"Test AUC\n{format_with_ci(roc_auc_score(y_test_true, y_test_prob), test_ci['auc'])}")
    print(f"Test AP\n{format_with_ci(average_precision_score(y_test_true, y_test_prob), test_ci['ap'])}")
    print(f"Test F1\n{format_with_ci(test_metrics['f1'], test_ci['f1'])}")
    print(f"Test Sens\n{format_with_ci(test_metrics['sens'], test_ci['sens'])}")
    print(f"Test Spec\n{format_with_ci(test_metrics['spec'], test_ci['spec'])}")
    print(f"Test Acc\n{format_with_ci(test_metrics['acc'], test_ci['acc'])}")
    print(f"Test PPV\n{format_with_ci(test_metrics['ppv'], test_ci['ppv'])}")
    print(f"Test NPV\n{format_with_ci(test_metrics['npv'], test_ci['npv'])}")
    print('=' * 80)

    # Save results and plots
    os.makedirs('test_results', exist_ok=True)
    model_name = os.path.basename(model_path).replace('.pt', '')

    # Save metrics table to CSV in the displayed order, with CI columns (rounded to 3 decimals)
    rows = [
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
        ['Test AUC', r3(roc_auc_score(y_test_true, y_test_prob)), r3(test_ci['auc'][0]), r3(test_ci['auc'][1])],
        ['Test AP', r3(average_precision_score(y_test_true, y_test_prob)), r3(test_ci['ap'][0]), r3(test_ci['ap'][1])],
        ['Test F1', r3(test_metrics['f1']), r3(test_ci['f1'][0]), r3(test_ci['f1'][1])],
        ['Test Sens', r3(test_metrics['sens']), r3(test_ci['sens'][0]), r3(test_ci['sens'][1])],
        ['Test Spec', r3(test_metrics['spec']), r3(test_ci['spec'][0]), r3(test_ci['spec'][1])],
        ['Test Acc', r3(test_metrics['acc']), r3(test_ci['acc'][0]), r3(test_ci['acc'][1])],
        ['Test PPV', r3(test_metrics['ppv']), r3(test_ci['ppv'][0]), r3(test_ci['ppv'][1])],
        ['Test NPV', r3(test_metrics['npv']), r3(test_ci['npv'][0]), r3(test_ci['npv'][1])],
    ]
    metrics_df = pd.DataFrame(rows, columns=['Metric', 'Value', 'CI_low', 'CI_high'])
    metrics_df.to_csv(f'test_results/{model_name}_results.csv', index=False)
    print(f"Results CSV saved to: test_results/{model_name}_results.csv")

    # Save probabilities as .npy files
    np.save(f'test_results/{model_name}_val_probabilities.npy', y_val_prob)
    np.save(f'test_results/{model_name}_test_probabilities.npy', y_test_prob)
    print(f"Probabilities saved to test_results/{model_name}_*_probabilities.npy")

    # Save plots (combined val+test ROC and confusion matrix)
    plot_val_test_roc(
        y_val_true,
        y_val_prob,
        y_test_true,
        y_test_prob,
        model_name,
        f'test_results/{model_name}_roc.png'
    )
    plot_confusion_matrix(cm, model_name, f'test_results/{model_name}_cm.png')

if __name__ == '__main__':
    main()
