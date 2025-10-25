import numpy as np
import pandas as pd
import joblib
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score, 
                              precision_score, recall_score, f1_score, roc_curve, confusion_matrix)


def get_data(num_features):
    """Load feature-based data for RF model evaluation.
    
    Returns:
        x_val: validation features
        y_val: validation labels
        x_test: test features
        y_test: test labels
    """
    
    feature_importance = pd.read_csv('rf_feature_importance.csv')
    feature_names = feature_importance['feature'].tolist()[:num_features]
    # Use the same features file as training
    features = pd.read_csv('results/features.csv')


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
        x_val.append(features_id.to_numpy())
    x_val = np.concatenate(x_val, axis=0)
    print(f"Val set: {len(x_val)} samples loaded")

    # Get test data
    x_test = []
    for id in test_ids:
        features_id = features[features['Unnamed: 0'] == id]
        features_id = features_id.drop(columns=['Unnamed: 0'])
        # Keep only the desired features and enforce ordering
        features_id = features_id[feature_names]
        x_test.append(features_id.to_numpy())
    x_test = np.concatenate(x_test, axis=0)
    print(f"Test set: {len(x_test)} samples loaded")

    return x_val, y_val, x_test, y_test


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


def main():

    folder_path = 'rf_models/'
    for file in Path(folder_path).glob('*.pkl'):
        model_path = file

        # Define model path
        # model_path = 'best_rf.pkl'  # Update this path if needed
        
        print('=' * 80)
        print('Random Forest Model Evaluation')
        print('=' * 80)
        
        # Load data
        print('\nLoading validation and test data...')
        num_features = file.stem.split('_')[-1]
        num_features = int(num_features)
        x_val, y_val, x_test, y_test = get_data(num_features)
        print('✓ Data loaded successfully')
        
        # Load RF model
        print(f'\nLoading RF model from: {model_path}')
        clf = joblib.load(model_path)
        print('✓ Model loaded successfully')
        
        # Get predictions
        print('\nGenerating predictions...')
        y_val_prob = clf.predict_proba(x_val)[:, 1]
        y_test_prob = clf.predict_proba(x_test)[:, 1]
        print('✓ Predictions generated')
        
        # Derive thresholds from validation split
        print('\nFinding optimal thresholds...')
        rule_out_thresh, rule_in_thresh, f1_thresh = find_thresholds_from_val(y_val, y_val_prob)
        print(f'✓ Thresholds found (F1 thresh: {f1_thresh:.3f})')
        
        # Compute validation metrics at F1 threshold
        print('\nComputing validation metrics...')
        val_metrics = compute_metrics(y_val, y_val_prob, f1_thresh)
        print('Computing validation bootstrap CIs (200 iterations)...')
        val_ci = bootstrap_ci_val(y_val, y_val_prob)
        print('✓ Validation metrics complete')

        # Test metrics at F1-optimal threshold (from validation set)
        print('\nComputing test metrics...')
        test_metrics = compute_metrics(y_test, y_test_prob, f1_thresh)
        print('Computing test bootstrap CIs (200 iterations)...')
        test_ci = bootstrap_ci_test(y_test, y_test_prob, f1_thresh)
        print('✓ Test metrics complete')

        # Confusion matrix at F1-optimal threshold
        test_preds = (y_test_prob >= f1_thresh).astype(int)
        cm = confusion_matrix(y_test, test_preds)

        # Print results in requested format
        print('\n' + '=' * 80)
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

        print(f"Test AUC\n{format_with_ci(roc_auc_score(y_test, y_test_prob), test_ci['auc'])}")
        print(f"Test AP\n{format_with_ci(average_precision_score(y_test, y_test_prob), test_ci['ap'])}")
        print(f"Test F1\n{format_with_ci(test_metrics['f1'], test_ci['f1'])}")
        print(f"Test Sens\n{format_with_ci(test_metrics['sens'], test_ci['sens'])}")
        print(f"Test Spec\n{format_with_ci(test_metrics['spec'], test_ci['spec'])}")
        print(f"Test Acc\n{format_with_ci(test_metrics['acc'], test_ci['acc'])}")
        print(f"Test PPV\n{format_with_ci(test_metrics['ppv'], test_ci['ppv'])}")
        print(f"Test NPV\n{format_with_ci(test_metrics['npv'], test_ci['npv'])}")
        print('=' * 80)

        # Save results and plots
        print('\nSaving results...')
        os.makedirs('rf_results', exist_ok=True)
        model_name = os.path.basename(model_path).replace('.pkl', '')

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
            ['Test AUC', r3(roc_auc_score(y_test, y_test_prob)), r3(test_ci['auc'][0]), r3(test_ci['auc'][1])],
            ['Test AP', r3(average_precision_score(y_test, y_test_prob)), r3(test_ci['ap'][0]), r3(test_ci['ap'][1])],
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
        np.save(f'test_results/{model_name}_val_probabilities.npy', y_val_prob)
        np.save(f'test_results/{model_name}_test_probabilities.npy', y_test_prob)
        print(f"✓ Probabilities saved to test_results/{model_name}_*_probabilities.npy")

        # Save plots (combined val+test ROC and confusion matrix)
        plot_val_test_roc(
            y_val,
            y_val_prob,
            y_test,
            y_test_prob,
            model_name,
            f'test_results/{model_name}_roc.png'
        )
        plot_confusion_matrix(cm, model_name, f'test_results/{model_name}_cm.png')
        
        print('\n' + '=' * 80)
        print('Evaluation Complete!')
        print('=' * 80)


if __name__ == '__main__':
    main()

