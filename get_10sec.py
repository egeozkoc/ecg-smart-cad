import numpy as np
import xmltodict
import scipy.io as io
import scipy.signal as signal
import json
import base64
import struct
import wfdb
from sierraecg import read_file
import h5py
import pandas as pd

ptbxl_df = pd.read_csv('ptbxl_database.csv')
ptbxl_df['filename_hr'] = ptbxl_df['filename_hr'].apply(lambda x: x.split('/')[-1])

def decode_ekg_muse_to_array(raw_wave, downsample = 1):
    """
    Ingest the base64 encoded waveforms and transform to numeric

    downsample: 0.5 takes every other value in the array. Muse samples at 500/s and the sample model requires 250/s. So take every other.
    """
    try:
        dwnsmpl = int(1//downsample)
    except ZeroDivisionError:
        print("You must downsample by more than 0")
    # covert the waveform from base64 to byte array
    arr = base64.b64decode(bytes(raw_wave, 'utf-8'))

    # unpack every 2 bytes, little endian (16 bit encoding)
    unpack_symbols = ''.join([char*int(len(arr)/2) for char in 'h'])
    byte_array = struct.unpack(unpack_symbols,  arr)
    return np.array(byte_array)

def get10sec(filename, lpf=100):
    global ptbxl_df
    print(filename)
    leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    leads = np.array(leads)
############################################################################################################
# Get Raw ECG data
############################################################################################################
    
    if filename.split('.')[-1] == 'xml':
        dic = xmltodict.parse(open(filename, 'rb'))

        # Philips XML
        if 'restingecgdata' in dic:
            fs = int(dic['restingecgdata']['dataacquisition']['signalcharacteristics']['samplingrate'])

            # XLI encoding
            if dic['restingecgdata']['waveforms']['parsedwaveforms']['@dataencoding'] == 'Base64':
                f = read_file(filename)
                ecg = np.empty([12, 5000], dtype=float)
                for i in range(12):
                    ecg[i, :] = f.leads[i].samples[0:5000]
            
            # No encoding
            else:
                raw_ecg = dic['restingecgdata']['waveforms']['parsedwaveforms']['#text'].split()
                ecg = np.empty([len(leads), 10 * fs])
                for i in range(len(leads)):
                    ecg[i, :] = raw_ecg[i * fs * 11:i * fs * 11 + fs * 10]

            if 'signalresolution' in dic['restingecgdata']['dataacquisition']['signalcharacteristics']:
                signal_res = int(dic['restingecgdata']['dataacquisition']['signalcharacteristics']['signalresolution'])
            else:
                signal_res = int(dic['restingecgdata']['dataacquisition']['signalcharacteristics']['resolution'])
            ecg *= signal_res # scale ECG signal to uV

            try:
                age = int(dic['restingecgdata']['patient']['generalpatientdata']['age']['years'])
            except:
                print('Error: Age not found')
                age = 60
            try:
                sex = dic['restingecgdata']['patient']['generalpatientdata']['sex'].lower()
                if sex == 'female' or sex == 'f':
                    sex = 1
                else:
                    sex = 0
            except:
                print('Error: Sex not found')
                sex = 0

        elif 'root' in dic:
            fs = int(dic['root']['ECGRecord']['Record']['RecordData'][0]['Waveforms']['XValues']['SampleRate']['#text'])
            raw_ecg = dic['root']['ECGRecord']['Record']['RecordData']
            ecg = np.empty([12, 5000], dtype=float)
            leadnames = []
            for i in range(len(raw_ecg)):
                leadnames.append(raw_ecg[i]['Channel'])
            for i in range(len(leads)):
                j = leadnames.index(leads[i])
                scale = float(raw_ecg[j]['Waveforms']['YValues']['RealValue']['Scale']) * 1000
                data = raw_ecg[j]['Waveforms']['YValues']['RealValue']['Data'].split(',')
                data = ['0' if x == '' else x for x in data]
                ecg[i, :] = np.array(data)
                ecg[i, :] *= scale
            try:
                age = int(dic['root']['ECGRecord']['PatientDemographics']['Age']['#text'])
            except:
                print('Error: Age not found')
                age = 60
            try:
                sex = dic['root']['ECGRecord']['PatientDemographics']['Sex'].lower()
                if sex == 'female':
                    sex = 1
                else:
                    sex = 0
            except:
                print('Error: Sex not found')
                sex = 0

        # Physio-Control XML
        elif 'ECGRecord' in dic:
            fs = int(dic['ECGRecord']['Record']['RecordData'][0]['Waveforms']['XValues']['SampleRate']['#text'])
            raw_ecg = dic['ECGRecord']['Record']['RecordData']
            ecg = np.empty([12, 5000], dtype=float)
            leadnames = []
            for i in range(len(raw_ecg)):
                leadnames.append(raw_ecg[i]['Channel'])
            for i in range(len(leads)):
                j = leadnames.index(leads[i])
                scale = float(raw_ecg[j]['Waveforms']['YValues']['RealValue']['Scale']) * 1000
                data = raw_ecg[j]['Waveforms']['YValues']['RealValue']['Data'].split(',')
                data = ['0' if x == '' else x for x in data]
                ecg[i, :] = np.array(data)
                ecg[i, :] *= scale
            try:
                age = int(dic['ECGRecord']['PatientDemographics']['Age']['#text'])
            except:
                print('Error: Age not found')
                age = 60
            try:
                sex = dic['ECGRecord']['PatientDemographics']['Sex'].lower()
                if sex == 'female':
                    sex = 1
                else:
                    sex = 0
            except:
                print('Error: Sex not found')
                sex = 0

        # Mortara XML
        elif 'ECG' in dic:
            try:
                age = int(dic['ECG']['@AGE'])
            except:
                print('Error: Age not found')
                age = 60
            try:
                sex = dic['ECG']['SUBJECT']['@GENDER'].lower()
                if sex == 'female':
                    sex = 1
                else:
                    sex = 0
            except:
                print('Error: Sex not found')
                sex = 0
            fs = int(dic['ECG']['TYPICAL_CYCLE']['@SAMPLE_FREQ'])
            scale = int(dic['ECG']['TYPICAL_CYCLE']['@UNITS_PER_MV'])

            ecg = np.empty([12, 10000], dtype=float)
            for i in range(12):
                lead_data = dic['ECG']['CHANNEL'][i]['@DATA']
                # convert data from base64 to floats
                arr = base64.b64decode(bytes(lead_data, 'utf-8'))
                # unpack every 2 bytes, little endian (16 bit encoding)
                unpack_symbols = ''.join([char*int(len(arr)/2) for char in 'h'])
                byte_array = struct.unpack(unpack_symbols,  arr)
                lead_data = np.array(byte_array)
                ecg[i, :] = lead_data / scale * 1000

        # GE MUSE XML
        elif 'RestingECG' in dic:
            fs = int(dic['RestingECG']['Waveform'][1]['SampleBase'])
            try:
                age = int(dic['RestingECG']['PatientDemographics']['PatientAge'])
            except:
                print('Error: Age not found')
                age = 60
            try:
                sex = dic['RestingECG']['PatientDemographics']['Gender'].lower()
                if sex == 'female':
                    sex = 1
                else:
                    sex = 0
            except:
                print('Error: Sex not found')
                sex = 0
        
            lead_order = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
            lead_data = dict.fromkeys(lead_order)
            for waveform in dic['RestingECG']['Waveform']:
                if waveform['WaveformType'] == 'Rhythm':
                    for leadid in range(len(waveform['LeadData'])):
                        sample_length = len(decode_ekg_muse_to_array(waveform['LeadData'][leadid]['WaveFormData']))
                        # sample_length is equivalent to dic['RestingECG']['Waveform']['LeadData']['LeadSampleCountTotal']
                        if sample_length == 5000:
                            lead_data[waveform['LeadData'][leadid]['LeadID']] = decode_ekg_muse_to_array(
                                waveform['LeadData'][leadid]['WaveFormData'], downsample=1)
                        elif sample_length == 2500:
                            lead_data[waveform['LeadData'][leadid]['LeadID']] = decode_ekg_muse_to_array(
                                waveform['LeadData'][leadid]['WaveFormData'], downsample=2)
                        else:
                            continue
                # ensures all leads have 2500 samples and also passes over the 3 second waveform

            lead_data['III'] = (np.array(lead_data["II"]) - np.array(lead_data["I"]))
            lead_data['aVR'] = -(np.array(lead_data["I"]) + np.array(lead_data["II"])) / 2
            lead_data['aVF'] = (np.array(lead_data["II"]) + np.array(lead_data["III"])) / 2
            lead_data['aVL'] = (np.array(lead_data["I"]) - np.array(lead_data["III"])) / 2
            lead_data = {k: lead_data[k] for k in lead_order}
            temp = []
            for key, value in lead_data.items():
                temp.append(value)
            ecg = np.array(temp)
            ecg *= 4.88 # convert to uV

    # ZOLL XML
    elif filename.split('.')[-1] == 'json':
        with open(filename) as f:
            dic = json.load(f)
        f.close()

        try:
            age = int(dic['ZOLL']['Report12Lead'][0]['Ecg12LeadRec']['AnalysisResult']['Age'])
        except:
            print('Error: Age not found')
            age = 60
        try:
            sex = int(dic['ZOLL']['Report12Lead'][0]['Ecg12LeadRec']['AnalysisResult']['Gender'])
            if sex == 2:
                sex = 1
            else:
                sex = 0
        except:
            print('Error: Sex not found')
            sex = 0
        
        leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        ecg = np.empty([12, 5000], dtype=float)
        ecg[0] = dic['ZOLL']['Report12Lead'][0]['Ecg12LeadRec']['LeadData']['LeadI']
        ecg[1] = dic['ZOLL']['Report12Lead'][0]['Ecg12LeadRec']['LeadData']['LeadII']
        ecg[2] = ecg[1] - ecg[0]
        ecg[3] = -(ecg[0] + ecg[1]) / 2
        ecg[4] = ecg[0] - ecg[1] / 2
        ecg[5] = ecg[1] - ecg[0] / 2
        ecg[6] = dic['ZOLL']['Report12Lead'][0]['Ecg12LeadRec']['LeadData']['LeadV1']
        ecg[7] = dic['ZOLL']['Report12Lead'][0]['Ecg12LeadRec']['LeadData']['LeadV2']
        ecg[8] = dic['ZOLL']['Report12Lead'][0]['Ecg12LeadRec']['LeadData']['LeadV3']
        ecg[9] = dic['ZOLL']['Report12Lead'][0]['Ecg12LeadRec']['LeadData']['LeadV4']
        ecg[10] = dic['ZOLL']['Report12Lead'][0]['Ecg12LeadRec']['LeadData']['LeadV5']
        ecg[11] = dic['ZOLL']['Report12Lead'][0]['Ecg12LeadRec']['LeadData']['LeadV6']
        fs = dic['ZOLL']['Report12Lead'][0]['Ecg12LeadRec']['SampleRate']
        ecg *= 2.5 # convert to uV

    # PTB-XL Files
    elif filename.split('.')[-1] == 'hea':
        ecg = wfdb.rdsamp(filename.split('.')[0])[0]
        ecg = ecg.T
        ecg *= 1000 # convert to uV
        fs = 500
        id = filename.split('/')[-1].split('.')[0]
        try:
            age = int(ptbxl_df[ptbxl_df['filename_hr'] == id]['age'].values[0])
        except:
            print('Error: Age not found')
            age = 60
        try:
            sex = int(ptbxl_df[ptbxl_df['filename_hr'] ==id]['sex'].values[0])
        except:
            sex = 0
            print('Error: Sex not found')

    # Basel H5 files
    elif filename.split('.')[-1] == 'h5':
        h5=h5py.File(filename,'r+')
        fs=int(h5["/rhythms/"].attrs.get('fs'))	#sampling rate

        #check if resolution is uV/Bit
        resolutionExp=h5["/rhythms/"].attrs.get('resolutionExp') #uV = -6
        resolutionFactor=h5["/rhythms/"].attrs.get('resolutionFactor') # 1
        if(resolutionExp !=-6 or resolutionFactor !=1):
            raise Exception('Unknow amplitude resolution')

        ecg = np.empty((12,5000))
        #ecg fs,uV/Bit
        for i,lead in enumerate(leads):
            data = h5["/rhythms/ECG_"+lead][:,0] * resolutionFactor
            if fs == 1000:
                data = signal.resample(data, int(len(data)/2))
                fs = 500
            ecg[i,:] = data[0:5000]

        try:
            age = h5['/patient/'].attrs.get('age')[0]
            if np.isnan(age):
                age = 60
        except:
            age = 60

        try:
            sex = h5['/patient/'].attrs.get('sex')[0]
            if np.isnan(sex):
                sex = 0
        except:
            sex = 0

    if fs != 500:
        sig_len = int(ecg.shape[1] / fs * 500)
        ecg = signal.resample(ecg, sig_len, axis=1)
        fs = 500

############################################################################################################
# Remove Baseline Wander
############################################################################################################

    b_baseline = io.loadmat('filters/baseline_filt.mat')['Num'][0]
    ecg = signal.filtfilt(b_baseline, 1, ecg, axis=1, padtype='even', padlen=1000)
    ecg = ecg - np.median(ecg,axis=1)[:,None]

############################################################################################################
# Remove Pacing Spikes
############################################################################################################

    b_pace, a_pace = signal.butter(2, 80, btype='highpass', fs=500)
    ecg_pace = signal.filtfilt(b_pace, a_pace, ecg, axis=-1, padtype='even', padlen=1000)

    rms_pace = np.sqrt(np.sum(ecg_pace**2,axis=0)) / np.sqrt(12)
    rms_ecg = np.sqrt(np.sum(ecg**2,axis=0)) / np.sqrt(12)
    peaks_pace = signal.find_peaks(rms_pace, height=250, distance=2)[0]
    peaks_pace = peaks_pace[(peaks_pace < 4995) & (peaks_pace > 5)]

    if (len(peaks_pace) > 0) and (len(peaks_pace) < 50):

        peaks_pace1 = np.empty(len(peaks_pace), dtype=int)
        for j in range(len(peaks_pace)):
            peaks_pace1[j] = np.argmax(rms_ecg[peaks_pace[j]-2:peaks_pace[j]+2])+peaks_pace[j]-2

        # get width of each peak
        for j in range(len(peaks_pace1)):
            if ((rms_ecg[peaks_pace1[j] - 2] < 1/2 * rms_ecg[peaks_pace1[j]]) or (rms_ecg[peaks_pace1[j] - 1] < 1/2 * rms_ecg[peaks_pace1[j]])) and ((rms_ecg[peaks_pace1[j] + 2] < 1/2 * rms_ecg[peaks_pace1[j]]) or (rms_ecg[peaks_pace1[j] + 1] < 1/2 * rms_ecg[peaks_pace1[j]])):
                for lead in range(12):
                    start = np.argmin(np.abs(ecg[lead,peaks_pace1[j]-5:peaks_pace1[j]])) + peaks_pace1[j]-5
                    end = np.argmin(np.abs(ecg[lead,peaks_pace1[j]+1:peaks_pace1[j]+6])) + peaks_pace1[j]+1
                    ecg[:, start:end] = np.nan

        # interpolate over removed pacing spikes
        for lead in range(12):
            nan_idx = np.where(np.isnan(ecg[lead,:]))[0]
            if len(nan_idx) > 0:
                ecg[lead,nan_idx] = np.interp(nan_idx, np.where(~np.isnan(ecg[lead,:]))[0], ecg[lead,~np.isnan(ecg[lead,:])])
    else:
        peaks_pace1 = []
############################################################################################################
# Low Pass Filter & PLI Filters
############################################################################################################

    if lpf == 100:
        # low pass hamming window filter
        b_low = io.loadmat('filters/lowpass100_filt.mat')['Num'][0]
        ecg = signal.filtfilt(b_low, 1, ecg, axis=1, padtype='even', padlen=1000)

        # pli filter
        b_pli50 = io.loadmat('filters/pli50_filt.mat')['Num'][0]
        b_pli60 = io.loadmat('filters/pli60_filt.mat')['Num'][0]

        ecg = signal.filtfilt(b_pli50, 1, ecg, axis=1, padtype='even', padlen=1000)
        ecg = signal.filtfilt(b_pli60, 1, ecg, axis=1, padtype='even', padlen=1000)

    elif lpf == 150:
        # low pass hamming window filter
        b_low = io.loadmat('filters/lowpass150_filt.mat')['Num'][0]
        ecg = signal.filtfilt(b_low, 1, ecg, axis=1, padtype='even', padlen=1000)

        # pli filter
        b_pli50 = io.loadmat('filters/pli50_filt.mat')['Num'][0]
        b_pli60 = io.loadmat('filters/pli60_filt.mat')['Num'][0]
        b_pli100 = io.loadmat('filters/pli100_filt.mat')['Num'][0]
        b_pli120 = io.loadmat('filters/pli120_filt.mat')['Num'][0]

        ecg = signal.filtfilt(b_pli50, 1, ecg, axis=1, padtype='even', padlen=1000)
        ecg = signal.filtfilt(b_pli60, 1, ecg, axis=1, padtype='even', padlen=1000)
        ecg = signal.filtfilt(b_pli100, 1, ecg, axis=1, padtype='even', padlen=1000)
        ecg = signal.filtfilt(b_pli120, 1, ecg, axis=1, padtype='even', padlen=1000)

    ecg = ecg - np.median(ecg,axis=1)[:,None]

############################################################################################################
# Remove artifacts within lead
############################################################################################################

    ecg = ecg.reshape(12,-1,625)
    num_leads = ecg.shape[0]
    num_windows = ecg.shape[1]
    
    bad_windows = np.zeros([num_leads,num_windows])
    ecg1 = np.abs(ecg)**2
    for i in range(num_leads):
        pwr = np.sum(ecg1[i,:,:], axis=-1)
        rng = np.max(ecg[i,:,:], axis=-1) - np.min(ecg[i,:,:],axis=-1)
        avg_rng = np.median(rng)
        avg_pwr = np.median(pwr)
        bad_windows[i,:] = (pwr > avg_pwr*10) | (rng > avg_rng*5) | (rng > 10000)
    ecg[bad_windows == 1] = 0

############################################################################################################
# Remove uncorrelated leads
############################################################################################################

    ecg = ecg.reshape(12,-1)
    nonzero_leads = np.where(np.sum(np.abs(ecg),axis=1) != 0)[0]
    # If fewer than 2 non-zero leads remain, skip correlation filtering to avoid AxisError
    if len(nonzero_leads) > 1:
        corrs = np.abs(np.corrcoef(ecg[nonzero_leads]))
        corrs = np.sum(corrs,axis=1) - 1
        corrs /= (len(nonzero_leads) - 1)
        bad_leads = np.zeros(12)
        bad_leads[nonzero_leads[corrs < 0.05]] = 1
        bad_leads[[2,3,4,5]] = 0

        ecg[bad_leads == 1] = 0

############################################################################################################
# Remove missing leads
############################################################################################################

    ecg = ecg.reshape(12,-1,625)
    missing_leads = np.max(np.abs(ecg), axis=2)
    missing_leads = np.median(missing_leads, axis=1) < 100
    ecg = ecg.reshape(12,-1)
    ecg[missing_leads == 1] = 0
    if missing_leads[2]: #lead III = lead II - lead I
        ecg[2] = ecg[1] - ecg[0]
    if missing_leads[3]: #lead aVR = -(lead I + lead II)/2
        ecg[3] = -(ecg[0] + ecg[1])/2
    if missing_leads[4]: #lead aVL = lead I - lead II/2
        ecg[4] = ecg[0] - ecg[1]/2
    if missing_leads[5]: #lead aVF = lead II - lead I/2
        ecg[5] = ecg[1] - ecg[0]/2
    missing_leads[[2,3,4,5]] = 0

    if np.sum(missing_leads) > 1 or missing_leads[0] or missing_leads[1] or missing_leads[6] or missing_leads[11]:
        poor_quality = True
    else:
        poor_quality = False
        if missing_leads[7]:
            ecg[7] = (ecg[6] + ecg[8])/2
        elif missing_leads[8]:
            ecg[8] = (ecg[7] + ecg[9])/2
        elif missing_leads[9]:
            ecg[9] = (ecg[8] + ecg[10])/2
        elif missing_leads[10]:
            ecg[10] = (ecg[9] + ecg[11])/2


    ecg_clean = ecg.copy()

    return {'ecg_clean': ecg_clean, 'poor_quality': poor_quality, 'fs': fs, 'leads': leads, 'pacing_spikes': peaks_pace1, 'age': age, 'sex': sex}
    
