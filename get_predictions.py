from glob import glob
import pandas as pd
import joblib
import numpy as np
from scipy import signal
import torch
from tkinter.filedialog import askdirectory, askopenfilename

def get_data(path, device):

    ecg = np.load(path, allow_pickle=True).item()
    ecg = ecg['waveforms']['ecg_median']
    ecg = ecg[:,150:-50]
    ecg = signal.resample(ecg, 200, axis=1)
    max_val = np.max(np.abs(ecg), axis=1)
    max_val[max_val == 0] = 1
    ecg = ecg / max_val[:, None]
    ecg = torch.tensor(ecg).float().to(device)
    ecg = ecg.unsqueeze(0).unsqueeze(0)

    return ecg


def getPredictions(features_filename, ecg_folder):
    # RF predictions
    features = pd.read_csv(features_filename, index_col=0)
    ids = features.index.to_list()
    features = features.to_numpy()

    rf_acs = joblib.load('models/rf_all_acs.pkl')
    # rf_omi = joblib.load('models/rf_all_omi.pkl')

    predictions = pd.DataFrame()
    predictions['id'] = ids
    predictions['rf_acs'] = rf_acs.predict_proba(features)[:, 1]
    # predictions['rf_omi'] = rf_omi.predict_proba(features)[:, 1]

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ecgsmartnet_acs = torch.load('models/ecgsmartnet_acs_2025-01-27-18-39-20.pt', weights_only=False).to(device)
    ecgsmartnet_acs.eval()

    filenames = glob(ecg_folder + '/*.npy')
    for filename in filenames:
        ecg = get_data(filename, device)

        # get prediction
        with torch.no_grad():
            output = ecgsmartnet_acs(ecg)
            output = torch.softmax(output, dim=-1)
            output = output.cpu().numpy()[0,-1]
            id = filename.split('\\')[-1].split('.')[0]
            predictions.loc[predictions['id'] == id, 'ecgsmartnet_acs'] = output

    logreg_acs = joblib.load('models/logreg_acs.pkl')
    fusion_input = predictions[['ecgsmartnet_acs', 'rf_acs']].to_numpy()
    predictions['fusion_acs'] = logreg_acs.predict_proba(fusion_input)[:, 1]
    
    predictions_filename = features_filename.replace('features.csv', 'predictions.csv')
    predictions.to_csv(predictions_filename, index = False)
        
if __name__ == '__main__':
    print('Select features file')
    features_filename = askopenfilename()
    print('Select folder containing processed ECGs')
    ecg_folder = askdirectory()
    getPredictions(features_filename, ecg_folder)
    


