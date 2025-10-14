import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, SequentialFeatureSelector, RFE, chi2
import shap

np.random.seed(42)
features = pd.read_csv('../outcomes/features.csv')
feature_names = features.columns.to_numpy()
feature_names = feature_names[2:]

def get_feature_ranks(data):
    x_train = data['train_features']
    y_train = data['train_outcomes']

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)

    # # RFE #########################################################################
    print('RFE')
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rfe = RFE(estimator=rf, n_features_to_select=1, step=1, verbose=1)
    rfe.fit(x_train, y_train)
    importances = rfe.ranking_
    importance_idx = np.argsort(importances)
    important_features_rfe = feature_names[importance_idx]

    # Random Forest ###############################################################
    print('Random Forest')
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(x_train, y_train)
    # get feature importance
    importances = rf.feature_importances_
    importance_idx = np.argsort(importances)
    importance_idx = importance_idx[::-1]
    important_features_rf = feature_names[importance_idx]

    # Lasso #######################################################################
    print('Lasso')
    lasso = Lasso(random_state=42)
    lasso.fit(x_train, y_train)
    importances = np.abs(lasso.coef_)
    importance_idx = np.argsort(importances)
    importance_idx = importance_idx[::-1]
    important_features_lasso = feature_names[importance_idx]

    # SVM #########################################################################
    print('SVM')
    svc = SVC(kernel='linear', random_state=42)
    svc.fit(x_train, y_train)
    importances = np.abs(svc.coef_[0])
    importance_idx = np.argsort(importances)
    importance_idx = importance_idx[::-1]
    important_features_svm = feature_names[importance_idx]

    # SelectKBest #################################################################
    print('F Statistic')
    selector = SelectKBest(k=417)
    selector.fit(x_train, y_train)
    importances = selector.scores_
    importance_idx = np.argsort(importances)
    importance_idx = importance_idx[::-1]
    important_features_kbest = feature_names[importance_idx]

    # # SHAP #########################################################################
    print('SHAP')
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(x_train, y_train)

    explainer = shap.TreeExplainer(rf, check_additivity=False)
    shap_values = explainer.shap_values(x_train, check_additivity=False)[1]
    importance = np.mean(np.abs(shap_values),axis=0)
    importance_idx = np.argsort(importance)
    importance_idx = importance_idx[::-1]
    important_features_shap = feature_names[importance_idx]

    # Chi2 #########################################################################
    print('Chi2')
    scaler = MinMaxScaler()
    X_train= scaler.fit_transform(x_train)
    chi2_selector = SelectKBest(chi2, k='all')
    chi2_selector.fit(X_train, y_train)
    importances = chi2_selector.scores_
    importance_idx = np.argsort(importances)
    importance_idx = importance_idx[::-1]
    important_features_chi2 = feature_names[importance_idx]


    return important_features_rf, important_features_lasso, important_features_svm, important_features_kbest, important_features_chi2, important_features_shap, important_features_rfe

data = np.load('data_acs.npy', allow_pickle=True).item()
important_features_rf_acs, important_features_lasso_acs, important_features_svm_acs, important_features_kbest_acs, important_features_chi2_acs, important_features_shap_acs, important_features_rfe_acs = get_feature_ranks(data)
data = np.load('data_omi.npy', allow_pickle=True).item()
important_features_rf_omi, important_features_lasso_omi, important_features_svm_omi, important_features_kbest_omi, important_features_chi2_omi, important_features_shap_omi, important_features_rfe_omi = get_feature_ranks(data)

average_rank_omi = np.zeros(len(feature_names))
for i in range(len(feature_names)):
    average_rank_omi[i] = np.median([np.where(important_features_rf_omi == feature_names[i])[0][0], np.where(important_features_lasso_omi == feature_names[i])[0][0], np.where(important_features_svm_omi == feature_names[i])[0][0], np.where(important_features_kbest_omi == feature_names[i])[0][0], np.where(important_features_chi2_omi == feature_names[i])[0][0], np.where(important_features_shap_omi == feature_names[i])[0][0], np.where(important_features_rfe_omi == feature_names[i])[0][0]])

average_rank_acs = np.zeros(len(feature_names))
for i in range(len(feature_names)):
    average_rank_acs[i] = np.median([np.where(important_features_rf_acs == feature_names[i])[0][0], np.where(important_features_lasso_acs == feature_names[i])[0][0], np.where(important_features_svm_acs == feature_names[i])[0][0], np.where(important_features_kbest_acs == feature_names[i])[0][0], np.where(important_features_chi2_acs == feature_names[i])[0][0], np.where(important_features_shap_acs == feature_names[i])[0][0], np.where(important_features_rfe_acs == feature_names[i])[0][0]])

average_rank = np.zeros(len(feature_names))
for i in range(len(feature_names)):
    average_rank[i] = np.median([np.where(important_features_rf_acs == feature_names[i])[0][0], np.where(important_features_lasso_acs == feature_names[i])[0][0], np.where(important_features_svm_acs == feature_names[i])[0][0], np.where(important_features_kbest_acs == feature_names[i])[0][0], np.where(important_features_chi2_acs == feature_names[i])[0][0], np.where(important_features_shap_acs == feature_names[i])[0][0], np.where(important_features_rfe_acs == feature_names[i])[0][0], np.where(important_features_rf_omi == feature_names[i])[0][0], np.where(important_features_lasso_omi == feature_names[i])[0][0], np.where(important_features_svm_omi == feature_names[i])[0][0], np.where(important_features_kbest_omi == feature_names[i])[0][0], np.where(important_features_chi2_omi == feature_names[i])[0][0], np.where(important_features_shap_omi == feature_names[i])[0][0], np.where(important_features_rfe_omi == feature_names[i])[0][0]])


importance_idx_omi = np.argsort(average_rank_omi)
importance_omi = average_rank_omi[importance_idx_omi]
important_features_omi = feature_names[importance_idx_omi]

importance_idx_acs = np.argsort(average_rank_acs)
importance_acs = average_rank_acs[importance_idx_acs]
important_features_acs = feature_names[importance_idx_acs]

importance_idx = np.argsort(average_rank)
importance = average_rank[importance_idx]
important_features = feature_names[importance_idx]

df = pd.DataFrame({'feature': important_features, 'importance': importance, 'feature_omi': important_features_omi, 'importance_omi': importance_omi, 'feature_acs': important_features_acs, 'importance_acs': importance_acs})
df.to_csv('feature_importance_final.csv', index=False)

print('done')