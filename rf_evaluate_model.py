import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
import joblib
import matplotlib.pyplot as plt

def get_feature_importance(num_features):
    # Load the best model
    clf = joblib.load('best_rf.pkl')
    
    # Get feature importance
    importances = clf.feature_importances_
    importance_idx = np.argsort(importances)
    importance_idx = importance_idx[::-1]
    important_features = feature_names[importance_idx]
    return important_features[:num_features]


def get_data():
    # Use a single features file for all runs
    features = pd.read_csv('results/features.csv')

    # Load feature names from single text file (one feature per line)
    feature_names = pd.read_csv('features.txt', header=None)[0].tolist()

    train_df = pd.read_csv('train_set.csv')
    val_df = pd.read_csv('val_set.csv')
    test_df = pd.read_csv('test_set.csv')
    y_train = train_df['label'].to_numpy()
    y_val = val_df['label'].to_numpy()
    y_test = test_df['label'].to_numpy()
    train_ids = train_df['ID'].to_list()
    val_ids = val_df['ID'].to_list()
    test_ids = test_df['ID'].to_list()

    x_train = []
    for id in train_ids:
        features_id = features[features['Unnamed: 0'] == id]
        features_id = features_id.drop(columns=['Unnamed: 0'])
        # Keep only the desired features and enforce ordering
        features_id = features_id[feature_names]
        x_train.append(features_id.to_numpy())
    x_train = np.concatenate(x_train, axis=0)

    x_val = []
    for id in val_ids:
        features_id = features[features['Unnamed: 0'] == id]
        features_id = features_id.drop(columns=['Unnamed: 0'])
        # Keep only the desired features and enforce ordering
        features_id = features_id[feature_names]
        x_val.append(features_id.to_numpy())
    x_val = np.concatenate(x_val, axis=0)

    x_test = []
    for id in test_ids:
        features_id = features[features['Unnamed: 0'] == id]
        features_id = features_id.drop(columns=['Unnamed: 0'])
        # Keep only the desired features and enforce ordering
        features_id = features_id[feature_names]
        x_test.append(features_id.to_numpy())
    x_test = np.concatenate(x_test, axis=0)

    return x_train, y_train, x_val, y_val, x_test, y_test

def evaluate_model():
    # Load the best model
    clf = joblib.load('best_rf.pkl')
    
    # Load the best parameters
    best_params = np.load('best_params.npy', allow_pickle=True).item()
    
    # Get data
    x_train, y_train, x_val, y_val, x_test, y_test = get_data()
    
    print('='*60)
    print('Best Model Hyperparameters:')
    print('='*60)
    for key, value in best_params.items():
        print(f'{key}: {value}')
    print('='*60)
    print()
    
    # Compute train metrics
    y_pred_train = clf.predict_proba(x_train)[:,1]
    auc_train = roc_auc_score(y_train, y_pred_train)
    ap_train = average_precision_score(y_train, y_pred_train)
    # harmonic mean of auc and ap
    train_score = 2 * (auc_train * ap_train) / (auc_train + ap_train)
    
    # Compute validation metrics
    y_pred_val = clf.predict_proba(x_val)[:,1]
    auc_val = roc_auc_score(y_val, y_pred_val)
    ap_val = average_precision_score(y_val, y_pred_val)
    # harmonic mean of auc and ap
    val_score = 2 * (auc_val * ap_val) / (auc_val + ap_val)
    
    # Compute test metrics
    y_pred_test = clf.predict_proba(x_test)[:,1]
    auc_test = roc_auc_score(y_test, y_pred_test)
    ap_test = average_precision_score(y_test, y_pred_test)
    # harmonic mean of auc and ap
    test_score = 2 * (auc_test * ap_test) / (auc_test + ap_test)
    
    # Print results
    print('='*60)
    print('TRAIN SET PERFORMANCE')
    print('='*60)
    print('Train Score (Harmonic Mean):', train_score)
    print('Train AUC:', auc_train)
    print('Train AP:', ap_train)
    print()
    
    print('='*60)
    print('VALIDATION SET PERFORMANCE')
    print('='*60)
    print('Val Score (Harmonic Mean):', val_score)
    print('Val AUC:', auc_val)
    print('Val AP:', ap_val)
    print()
    
    print('='*60)
    print('TEST SET PERFORMANCE')
    print('='*60)
    print('Test Score (Harmonic Mean):', test_score)
    print('Test AUC:', auc_test)
    print('Test AP:', ap_test)
    print('='*60)
    
    # Plot ROC curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # ROC Curve
    fpr_train, tpr_train, _ = roc_curve(y_train, y_pred_train)
    fpr_val, tpr_val, _ = roc_curve(y_val, y_pred_val)
    fpr_test, tpr_test, _ = roc_curve(y_test, y_pred_test)
    
    axes[0].plot(fpr_train, tpr_train, label=f'Train (AUC = {auc_train:.3f})', linewidth=2)
    axes[0].plot(fpr_val, tpr_val, label=f'Validation (AUC = {auc_val:.3f})', linewidth=2)
    axes[0].plot(fpr_test, tpr_test, label=f'Test (AUC = {auc_test:.3f})', linewidth=2)
    axes[0].plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
    axes[0].set_xlabel('False Positive Rate', fontsize=12)
    axes[0].set_ylabel('True Positive Rate', fontsize=12)
    axes[0].set_title('ROC Curve', fontsize=14, fontweight='bold')
    axes[0].legend(loc='lower right', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Precision-Recall Curve
    precision_train, recall_train, _ = precision_recall_curve(y_train, y_pred_train)
    precision_val, recall_val, _ = precision_recall_curve(y_val, y_pred_val)
    precision_test, recall_test, _ = precision_recall_curve(y_test, y_pred_test)
    
    axes[1].plot(recall_train, precision_train, label=f'Train (AP = {ap_train:.3f})', linewidth=2)
    axes[1].plot(recall_val, precision_val, label=f'Validation (AP = {ap_val:.3f})', linewidth=2)
    axes[1].plot(recall_test, precision_test, label=f'Test (AP = {ap_test:.3f})', linewidth=2)
    
    # Add baseline (random classifier line based on class distribution)
    baseline_train = np.sum(y_train) / len(y_train)
    baseline_val = np.sum(y_val) / len(y_val)
    baseline_test = np.sum(y_test) / len(y_test)
    axes[1].axhline(y=baseline_test, color='k', linestyle='--', label=f'Random Classifier (Test baseline = {baseline_test:.3f})', linewidth=1)
    
    axes[1].set_xlabel('Recall', fontsize=12)
    axes[1].set_ylabel('Precision', fontsize=12)
    axes[1].set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    axes[1].legend(loc='best', fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    
    # Evaluate the model
    evaluate_model()
    for num_features in [25, 50, 75, 100, 125, 150]:
        feature_importance = get_feature_importance(num_features)
        pd.DataFrame({'feature': feature_importance}).to_csv(f'feature_importance_{num_features}.csv', index=False)

    

