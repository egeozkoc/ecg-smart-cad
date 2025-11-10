import numpy as np
import pandas as pd
import joblib
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score, 
                              precision_score, recall_score, f1_score, roc_curve, confusion_matrix)
from sklearn.model_selection import StratifiedKFold


def get_data(num_features):
    """Load feature-based data for RF model training and evaluation.
    
    Args:
        num_features: Number of top features to use
    
    Returns:
        x_train_val: combined train+val features
        y_train_val: combined train+val labels
        x_test: test features
        y_test: test labels
    """
    
    feature_importance = pd.read_csv('rf_feature_importance_cross_validation.csv')
    feature_names = feature_importance['feature'].tolist()[:num_features]
    
    # Use the same features file as training
    features = pd.read_csv('results/features.csv')

    train_df = pd.read_csv('train_set.csv')
    val_df = pd.read_csv('val_set.csv')
    test_df = pd.read_csv('test_set.csv')
    
    y_train = train_df['label'].to_numpy()
    y_val = val_df['label'].to_numpy()
    y_test = test_df['label'].to_numpy()
    
    train_ids = train_df['ID'].to_list()
    val_ids = val_df['ID'].to_list()
    test_ids = test_df['ID'].to_list()

    # Get training data
    x_train = []
    for id in train_ids:
        features_id = features[features['Unnamed: 0'] == id]
        features_id = features_id.drop(columns=['Unnamed: 0'])
        features_id = features_id[feature_names]
        x_train.append(features_id.to_numpy())
    x_train = np.concatenate(x_train, axis=0)

    # Get validation data
    x_val = []
    for id in val_ids:
        features_id = features[features['Unnamed: 0'] == id]
        features_id = features_id.drop(columns=['Unnamed: 0'])
        features_id = features_id[feature_names]
        x_val.append(features_id.to_numpy())
    x_val = np.concatenate(x_val, axis=0)

    # Get test data
    x_test = []
    for id in test_ids:
        features_id = features[features['Unnamed: 0'] == id]
        features_id = features_id.drop(columns=['Unnamed: 0'])
        features_id = features_id[feature_names]
        x_test.append(features_id.to_numpy())
    x_test = np.concatenate(x_test, axis=0)

    # Combine train and val for cross-validation
    x_train_val = np.vstack([x_train, x_val])
    y_train_val = np.concatenate([y_train, y_val])

    return x_train_val, y_train_val, x_test, y_test


def get_cv_predictions(clf, x_train_val, y_train_val, n_splits=5):
    """Generate out-of-fold predictions using stratified k-fold CV.
    
    This mimics the cross-validation approach used during training with GridSearchCV.
    
    Args:
        clf: Trained classifier
        x_train_val: Combined train+val features
        y_train_val: Combined train+val labels
        n_splits: Number of CV folds (default: 5, matching training)
    
    Returns:
        y_cv_prob: Out-of-fold probability predictions for entire train+val set
    """
    stratified_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    y_cv_prob = np.zeros(len(y_train_val))
    
    print(f'Performing {n_splits}-fold stratified CV to get validation predictions...')
    for fold_idx, (train_idx, val_idx) in enumerate(stratified_cv.split(x_train_val, y_train_val)):
        x_fold_train, x_fold_val = x_train_val[train_idx], x_train_val[val_idx]
        y_fold_train = y_train_val[train_idx]
        
        # Clone and retrain the model on this fold
        fold_clf = clf.__class__(**clf.get_params())
        fold_clf.fit(x_fold_train, y_fold_train)
        
        # Get predictions for validation fold
        y_cv_prob[val_idx] = fold_clf.predict_proba(x_fold_val)[:, 1]
        print(f'  Fold {fold_idx + 1}/{n_splits} complete')
    
    print('✓ CV predictions generated')
    return y_cv_prob


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
    # ========================================================================
    # HARDCODED HYPERPARAMETERS - MODIFY THESE TO TEST DIFFERENT MODELS
    # ========================================================================
    num_features = 50  # Number of top features to use
    
    hyperparameters = {
        'n_estimators': 50,
        'class_weight': 'balanced',
        'criterion': 'gini',
        'max_features': 'log2',
        'min_samples_split': 0.001,
        'min_samples_leaf': 0.005,
        'min_impurity_decrease': 0.0,
        'ccp_alpha': 0.005,
        'max_samples': 1.0,
        'n_jobs': 1,
        'random_state': 42,
        'bootstrap': True,
        'oob_score': True
    }
    # ========================================================================
    
    model_name = f'rf_custom_{num_features}_features'
    
    print('=' * 80)
    print('Random Forest Model Training and Evaluation')
    print('=' * 80)
    print(f'\nModel: {model_name}')
    print(f'Number of features: {num_features}')
    print('\nHyperparameters:')
    for key, value in hyperparameters.items():
        print(f'  {key}: {value}')
    
    # Load data
    print('\n' + '=' * 80)
    print('STEP 1: Loading Data')
    print('=' * 80)
    x_train_val, y_train_val, x_test, y_test = get_data(num_features)
    print(f'Train+Val samples: {len(x_train_val)}')
    print(f'Test samples: {len(x_test)}')
    print(f'Features: {x_train_val.shape[1]}')
    print('✓ Data loaded successfully')
    
    # Train model
    print('\n' + '=' * 80)
    print('STEP 2: Training Model')
    print('=' * 80)
    clf = RandomForestClassifier(**hyperparameters)
    print('Training on full train+val set...')
    clf.fit(x_train_val, y_train_val)
    print('✓ Model trained successfully')
    
    # Quick evaluation on test set (like in rf_train_models.py)
    y_pred_test = clf.predict_proba(x_test)[:, 1]
    auc_test = roc_auc_score(y_test, y_pred_test)
    ap_test = average_precision_score(y_test, y_pred_test)
    print(f'\nQuick Test Evaluation:')
    print(f'  Test AUC: {auc_test:.3f}')
    print(f'  Test AP: {ap_test:.3f}')
    
    # Save model
    print('\nSaving model...')
    os.makedirs('rf_models', exist_ok=True)
    model_path = f'rf_models/{model_name}.pkl'
    joblib.dump(clf, model_path)
    print(f'✓ Model saved to: {model_path}')
    
    # ========================================================================
    # EVALUATION (same as rf_test_models.py)
    # ========================================================================
    
    print('\n' + '=' * 80)
    print('STEP 3: Comprehensive Evaluation')
    print('=' * 80)
    
    # Get CV predictions for validation
    print('\nGenerating validation predictions using 5-fold stratified CV...')
    y_val_prob = get_cv_predictions(clf, x_train_val, y_train_val, n_splits=5)
    
    # Get test predictions
    print('\nGenerating test predictions...')
    y_test_prob = clf.predict_proba(x_test)[:, 1]
    print('✓ Test predictions generated')
    
    # Derive thresholds from CV validation predictions
    print('\nFinding optimal thresholds from CV predictions...')
    rule_out_thresh, rule_in_thresh, f1_thresh = find_thresholds_from_val(y_train_val, y_val_prob)
    print(f'  Rule-out threshold (Sens >= 0.90): {r3(rule_out_thresh) if rule_out_thresh is not None else "N/A"}')
    print(f'  Rule-in threshold (PPV >= 0.85): {r3(rule_in_thresh) if rule_in_thresh is not None else "N/A"}')
    print(f'  F1-optimal threshold: {r3(f1_thresh)}')
    print('✓ Thresholds found')
    
    # Compute validation metrics
    print('\nComputing validation metrics from CV predictions...')
    val_metrics = compute_metrics(y_train_val, y_val_prob, f1_thresh)
    print('Computing validation bootstrap CIs (200 iterations)...')
    val_ci = bootstrap_ci_val(y_train_val, y_val_prob)
    print('✓ Validation metrics complete')

    # Compute test metrics
    print('\nComputing test metrics...')
    test_metrics = compute_metrics(y_test, y_test_prob, f1_thresh)
    print('Computing test bootstrap CIs (200 iterations)...')
    test_ci = bootstrap_ci_test(y_test, y_test_prob, f1_thresh)
    print('✓ Test metrics complete')

    # Confusion matrix
    test_preds = (y_test_prob >= f1_thresh).astype(int)
    cm = confusion_matrix(y_test, test_preds)

    # Print results
    print('\n' + '=' * 80)
    print('RESULTS')
    print('=' * 80)
    print(f"Rule Out Thresh\nSens > 0.90\n{r3(rule_out_thresh) if rule_out_thresh is not None else 'N/A'}")
    print(f"Rule In Thresh\nPPV > 0.85\n{r3(rule_in_thresh) if rule_in_thresh is not None else 'N/A'}")
    print(f"F1 Thresh\n{r3(f1_thresh)}")

    print(f"\nVal AUC\n{format_with_ci(val_metrics['auc'], val_ci['auc'])}")
    print(f"Val AP\n{format_with_ci(val_metrics['ap'], val_ci['ap'])}")
    print(f"Val F1\n{format_with_ci(val_metrics['f1'], val_ci['f1'])}")
    print(f"Val Sens\n{format_with_ci(val_metrics['sens'], val_ci['sens'])}")
    print(f"Val Spec\n{format_with_ci(val_metrics['spec'], val_ci['spec'])}")
    print(f"Val Acc\n{format_with_ci(val_metrics['acc'], val_ci['acc'])}")
    print(f"Val PPV\n{format_with_ci(val_metrics['ppv'], val_ci['ppv'])}")
    print(f"Val NPV\n{format_with_ci(val_metrics['npv'], val_ci['npv'])}")

    print(f"\nTest AUC\n{format_with_ci(roc_auc_score(y_test, y_test_prob), test_ci['auc'])}")
    print(f"Test AP\n{format_with_ci(average_precision_score(y_test, y_test_prob), test_ci['ap'])}")
    print(f"Test F1\n{format_with_ci(test_metrics['f1'], test_ci['f1'])}")
    print(f"Test Sens\n{format_with_ci(test_metrics['sens'], test_ci['sens'])}")
    print(f"Test Spec\n{format_with_ci(test_metrics['spec'], test_ci['spec'])}")
    print(f"Test Acc\n{format_with_ci(test_metrics['acc'], test_ci['acc'])}")
    print(f"Test PPV\n{format_with_ci(test_metrics['ppv'], test_ci['ppv'])}")
    print(f"Test NPV\n{format_with_ci(test_metrics['npv'], test_ci['npv'])}")

    # Save results
    print('\n' + '=' * 80)
    print('STEP 4: Saving Results')
    print('=' * 80)
    os.makedirs('rf_results', exist_ok=True)

    # Save metrics table to CSV
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
    metrics_df.to_csv(f'rf_results/{model_name}_results.csv', index=False)
    print(f"✓ Results CSV saved to: rf_results/{model_name}_results.csv")

    # Save probabilities
    np.save(f'rf_results/{model_name}_val_probabilities.npy', y_val_prob)
    np.save(f'rf_results/{model_name}_test_probabilities.npy', y_test_prob)
    print(f"✓ Probabilities saved to rf_results/{model_name}_*_probabilities.npy")

    # Save plots
    plot_val_test_roc(
        y_train_val,
        y_val_prob,
        y_test,
        y_test_prob,
        model_name,
        f'rf_results/{model_name}_roc.png'
    )
    plot_confusion_matrix(cm, model_name, f'rf_results/{model_name}_cm.png')
    
    print('\n' + '=' * 80)
    print('COMPLETE!')
    print('=' * 80)
    print(f'\nModel saved to: {model_path}')
    print(f'Results saved to: rf_results/{model_name}_*')
    print('\nTo test a different configuration, modify the hyperparameters')
    print('at the top of main() and run again.')
    print('=' * 80)


if __name__ == '__main__':
    main()

