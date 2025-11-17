import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, make_scorer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import os
import joblib

def get_feature_importance():

    feature_names = pd.read_csv('features.txt', header=None)[0].tolist()
    # Load the best model
    clf = joblib.load('rf_models/best_rf_all_features.pkl')
    
    # Get feature importance
    importances = clf.feature_importances_
    importance_idx = np.argsort(importances)
    importance_idx = importance_idx[::-1]
    sorted_importance = importances[importance_idx]
    
    # Convert feature_names to numpy array for fancy indexing
    feature_names_array = np.array(feature_names)
    return feature_names_array[importance_idx], sorted_importance


def harmonic_mean_scorer(y_true, y_pred_proba):
    auc = roc_auc_score(y_true, y_pred_proba)
    ap = average_precision_score(y_true, y_pred_proba)
    return 2 * (auc * ap) / (auc + ap)

def get_data(num_features=None):
    # Use a single features file for all runs
    features = pd.read_csv('results/features.csv')
    if num_features is None:
        # For all-features model: use original feature list
        feature_names = pd.read_csv('features.txt', header=None)[0].tolist()
    else:
        # For selected-features models: use importance-ranked features
        feature_names = pd.read_csv('rf_feature_importance_cross_validation.csv')['feature'].tolist()[:num_features]

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

def train_model_all_features():

    # Get Data
    x_train, y_train, x_val, y_val, x_test, y_test = get_data()

    # Combine train and val for cross-validation
    x_train_val = np.vstack([x_train, x_val])
    y_train_val = np.concatenate([y_train, y_val])


    # Define parameter grid
    param_grid = {
        'n_estimators': [50, 75, 100, 250],
        'class_weight': ['balanced_subsample'],
        'criterion': ['entropy', 'gini'],
        'max_features': ['sqrt', 'log2', None],
        'min_samples_split': [0.001, 0.005, 0.01],
        'min_samples_leaf': [0.001, 0.005, 0.01],
        'min_impurity_decrease': [0.0, 0.001, 0.005],
        'ccp_alpha': [0.0, 0.001, 0.005],
        'max_samples': [0.25, 0.5, 0.75, 1.0]
    }

    # Base classifier
    rf = RandomForestClassifier(
        n_jobs=1,
        random_state=42,
        bootstrap=True,
        oob_score=True
    )

    stratified_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    harmonic_scorer = make_scorer(harmonic_mean_scorer)


    grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    scoring=harmonic_scorer,
    cv=stratified_cv,
    n_jobs=-1, 
    verbose=2,
    refit=True,
    return_train_score=True
    )

    grid_search.fit(x_train_val, y_train_val)

    best_clf = grid_search.best_estimator_


    # Evaluate on test set
    y_pred_test = best_clf.predict_proba(x_test)[:, 1]
    auc_test = roc_auc_score(y_test, y_pred_test)
    ap_test = average_precision_score(y_test, y_pred_test)

    print('Best Params:', grid_search.best_params_)
    print('Best CV Score:', grid_search.best_score_)
    print('Test AUC:', auc_test)
    print('Test AP:', ap_test)

    # Save results
    results_df = pd.DataFrame(grid_search.cv_results_)
    results_df.to_csv('results/rf_all_features_grid_search_results.csv', index=False)

    # Save best model
    os.makedirs('rf_models', exist_ok=True)
    joblib.dump(best_clf, 'rf_models/best_rf_all_features.pkl')
    
    return grid_search


    
def train_models_selected_features(num_features):

    # Get Data
    x_train, y_train, x_val, y_val, x_test, y_test = get_data(num_features)

    # Combine train and val for cross-validation
    x_train_val = np.vstack([x_train, x_val])
    y_train_val = np.concatenate([y_train, y_val])


    # Define parameter grid
    param_grid = {
        'n_estimators': [50, 75, 100, 250],
        'class_weight': ['balanced_subsample'],
        'criterion': ['entropy', 'gini'],
        'max_features': ['sqrt', 'log2', None],
        'min_samples_split': [0.001, 0.005, 0.01],
        'min_samples_leaf': [0.001, 0.005, 0.01],
        'min_impurity_decrease': [0.0, 0.001, 0.005],
        'ccp_alpha': [0.0, 0.001, 0.005],
        'max_samples': [0.25, 0.5, 0.75, 1.0]
    }

    # Base classifier
    rf = RandomForestClassifier(
        n_jobs=1,
        random_state=42,
        bootstrap=True,
        oob_score=True
    )

    stratified_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    harmonic_scorer = make_scorer(harmonic_mean_scorer)


    grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    scoring=harmonic_scorer,
    cv=stratified_cv,
    n_jobs=-1, 
    verbose=2,
    refit=True,
    return_train_score=True
    )

    grid_search.fit(x_train_val, y_train_val)

    best_clf = grid_search.best_estimator_


    # Evaluate on test set
    y_pred_test = best_clf.predict_proba(x_test)[:, 1]
    auc_test = roc_auc_score(y_test, y_pred_test)
    ap_test = average_precision_score(y_test, y_pred_test)

    print('Best Params:', grid_search.best_params_)
    print('Best CV Score:', grid_search.best_score_)
    print('Test AUC:', auc_test)
    print('Test AP:', ap_test)

    # Save results
    results_df = pd.DataFrame(grid_search.cv_results_)
    results_df.to_csv(f'results/rf_selected_features_{num_features}_grid_search_results.csv', index=False)

    # Save best model
    os.makedirs('rf_models', exist_ok=True)
    joblib.dump(best_clf, f'rf_models/best_rf_selected_features_{num_features}.pkl')
    
    return grid_search




if __name__ == '__main__':

    # train_models_all_features()

    # important_features, importance_scores = get_feature_importance()
    # pd.DataFrame({'feature': important_features, 'importance': importance_scores}).to_csv(f'rf_feature_importance_cross_validation.csv', index=False)
    
    for num_features in [125]:
        train_models_selected_features(num_features)