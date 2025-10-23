import numpy as np
import pandas as pd
from glob import glob
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, roc_curve, precision_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.feature_selection import SelectKBest, f_classif, chi2
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
from sklearn.svm import SVC

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

    # do not include serial ECGs in validation or test sets
    
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


if __name__ == '__main__':
    x_train, y_train, x_val, y_val, x_test, y_test = get_data()
    print(x_train.shape)
    print(y_train.shape)
    print(x_val.shape)
    print(y_val.shape)
    print(x_test.shape)
    print(y_test.shape)