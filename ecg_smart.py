import tkinter as tk
from glob import glob
from tkinter import filedialog
import h5py
import numpy as np
import requests
from datetime import datetime
from scipy import signal, integrate
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform, cdist, euclidean
import itertools
import joblib
import pandas as pd
import os
import shap
from cryptography.fernet import Fernet
import scipy.io as io
from io import BytesIO
import xmltodict
import scipy
import sklearn
import base64
import struct
import json
import ast

# ECG Class ####################################################################
class ECG:
    def __init__(self, filename):
        self.filename = filename
        self.id = filename.split('\\')[-1].split('.')[0]
        self.waveforms = {'ecg_10sec_raw': None, 'ecg_10sec_filtered': None, 'ecg_10sec_clean': None, 'ecg_median': None, 'beats': None}
        self.fiducials = {'local': None, 'global': None, 'pacing_spikes': None, 'r_peaks': None}
        self.features = {}
        self.demographics = {'age': None, 'sex': None}
        self.leads = None
        self.fs = None
        self.poor_quality = None

    def processRawData(self, b_baseline, b_low100, b_low150, b_pli50, b_pli60, b_pli100, b_pli120):
        data = get10sec(self.filename, b_baseline, b_low100, b_low150, b_pli50, b_pli60, b_pli100, b_pli120, lpf=100)
        self.waveforms['ecg_10sec_raw'] = data['ecg_raw']
        self.waveforms['ecg_10sec_filtered'] = data['ecg_filtered']
        self.waveforms['ecg_10sec_clean'] = data['ecg_clean']
        self.fs = data['fs']
        self.leads = data['leads']
        self.poor_quality = data['poor_quality']
        self.fiducials['pacing_spikes'] = data['pacing_spikes']
        self.demographics['age'] = data['age']
        self.demographics['sex'] = data['sex']

    def processMedian(self):
        data = getMedianBeat(self.waveforms['ecg_10sec_clean'])
        self.waveforms['ecg_median'] = data['median_beat']
        self.waveforms['beats'] = data['beats']
        self.features['rrint'] = data['rrint']
        self.fiducials['rpeaks'] = data['rpeaks']

    def segmentMedian(self):
        data = getFiducials(self.waveforms['ecg_median'], self.features['rrint'], self.fs)
        self.fiducials['local'] = data['local']
        self.fiducials['global'] = data['global']

    def processFeatures(self):
        getFeatures(self)

################################################################################

# ECG Functions ################################################################
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

def get10sec(filename, b_baseline, b_low100, b_low150, b_pli50, b_pli60, b_pli100, b_pli120, lpf=100):
    print(filename)
    leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    leads = np.array(leads)
############################################################################################################
# Get Raw ECG data

    if filename.split('.')[-1] == 'xml':
        dic = xmltodict.parse(open(filename, 'rb'))

        # Philips XML
        if 'restingecgdata' in dic:
            fs = int(dic['restingecgdata']['dataacquisition']['signalcharacteristics']['samplingrate'])
            raw_ecg = dic['restingecgdata']['waveforms']['parsedwaveforms']['#text'].split()
            if 'signalresolution' in dic['restingecgdata']['dataacquisition']['signalcharacteristics']:
                signal_res = int(dic['restingecgdata']['dataacquisition']['signalcharacteristics']['signalresolution'])
            else:
                signal_res = int(dic['restingecgdata']['dataacquisition']['signalcharacteristics']['resolution'])
            ecg = np.empty([len(leads), 10 * fs])
            for i in range(len(leads)):
                ecg[i, :] = raw_ecg[i * fs * 11:i * fs * 11 + fs * 10]
            ecg *= signal_res # scale ECG signal to uV

            try:
                age = int(dic['restingecgdata']['patient']['generalpatientdata']['age']['years'])
            except:
                print('Error: Age not found')
                age = 60
            try:
                sex = dic['restingecgdata']['patient']['generalpatientdata']['sex'].lower()
                if sex == 'female':
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
        print('ZOLL json file amplitudes are incorrect. Do not use until fixed.')
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

    
    sig_len = int(ecg.shape[1] / fs * 500)
    ecg = signal.resample(ecg, sig_len, axis=1)
    fs = 500
    ecg_raw = ecg.copy()

############################################################################################################
# Remove Baseline Wander
############################################################################################################
    ecg = np.concatenate((np.flip(ecg[:,:1000],axis=1), ecg, np.flip(ecg[:,-1000:],axis=1)), axis=1)
    ecg = signal.filtfilt(b_baseline, 1, ecg, axis=1)
    ecg = ecg - np.median(ecg,axis=1)[:,None]

############################################################################################################
# Remove Pacing Spikes
############################################################################################################

    b_pace, a_pace = signal.butter(2, 80, btype='highpass', fs=500)
    ecg_pace = signal.filtfilt(b_pace, a_pace, ecg, axis=-1)
    ecg = ecg[:,1000:-1000]
    ecg_pace = ecg_pace[:,1000:-1000]

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

    ecg = np.concatenate((np.flip(ecg[:,:1000],axis=1), ecg, np.flip(ecg[:,-1000:],axis=1)), axis=1)

    if lpf == 100:
        ecg = signal.filtfilt(b_low100, 1, ecg, axis=1)
        ecg = signal.filtfilt(b_pli50, 1, ecg, axis=1)
        ecg = signal.filtfilt(b_pli60, 1, ecg, axis=1)

    elif lpf == 150:
        ecg = signal.filtfilt(b_low150, 1, ecg, axis=1)
        ecg = signal.filtfilt(b_pli50, 1, ecg, axis=1)
        ecg = signal.filtfilt(b_pli60, 1, ecg, axis=1)
        ecg = signal.filtfilt(b_pli100, 1, ecg, axis=1)
        ecg = signal.filtfilt(b_pli120, 1, ecg, axis=1)

    ecg = ecg[:,1000:-1000]
    ecg = ecg - np.median(ecg,axis=1)[:,None]
    ecg_filtered = ecg.copy()

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

    return {'ecg_raw': ecg_raw, 'ecg_filtered': ecg_filtered, 'ecg_clean': ecg_clean, 'poor_quality': poor_quality, 'fs': fs, 'leads': leads, 'pacing_spikes': peaks_pace1, 'age': age, 'sex': sex}
    
use_corr = False

def getTemplateBeat(beats):
    beats1 = np.empty((beats.shape[0], beats.shape[1], 200))
    for i in range(beats.shape[0]):
        for j in range(beats.shape[1]):
            peak = np.argmax(np.abs(beats[i, j, 285:315])) + 285
            beats1[i, j, :] = beats[i, j, peak - 100:peak + 100]
    
    beats1 -= np.median(beats1[:,:,0:75],axis=2)[:,:,None]
    if use_corr:
        corrs = np.ones([12, beats1.shape[0], beats1.shape[0]])
    else:
        corrs = np.zeros([12, beats1.shape[0], beats1.shape[0]])
    best_beat = np.empty(12, dtype=int)


    mags = np.zeros(12)
    for i in range(12):
        for beat in range(beats1.shape[0]):
            mags[i] += np.linalg.norm(beats1[beat,i,:])**2
        mags[i] = np.sqrt(mags[i])


    for i in range(12):
        for j in range(beats1.shape[0]):
            for k in range(j+1,beats1.shape[0]):
                beat1 = beats1[j,i,:]
                beat2 = beats1[k,i,:]
                beat1 = np.reshape(beat1, [25,-1])
                beat2 = np.reshape(beat2, [25,-1])
                beat1_stds = np.std(beat1, axis=1)
                beat2_stds = np.std(beat2, axis=1)
                if np.any(beat1_stds == 0) or np.any(beat2_stds == 0):
                    if use_corr:
                        corrs[i,j,k] = 0
                        corrs[i,k,j] = 0
                    else:
                        corrs[i,j,k] = 1000
                        corrs[i,k,j] = 1000
                else:
                    if use_corr:
                        corrs[i,j,k] = np.corrcoef(beats1[j,i,:], beats1[k,i,:])[0,1]
                    else:
                        mag = np.sqrt(np.linalg.norm(beats1[j,i,:])**2 + np.linalg.norm(beats1[k,i,:])**2)

                        corrs[i,j,k] = np.linalg.norm(beats1[j,i,:] - beats1[k,i,:]) / mags[i]
                    corrs[i,k,j] = corrs[i,j,k]
        
        if use_corr:
            best_beat[i] = np.argmax(np.sum(corrs[i,:,:], axis=-1))
        else:
            best_beat[i] = np.argmin(np.sum(corrs[i,:,:], axis=-1))

    best_corrs = np.zeros([12,beats1.shape[0]])
    for i in range(12):
        best_corrs[i] = corrs[i,best_beat[i],:]
    
    if use_corr:
        best_overall_beat = np.argmax(np.sum(best_corrs, axis=0))
    else:
        best_overall_beat = np.argmin(np.sum(best_corrs, axis=0))

    return best_overall_beat

def getValidBeats(beats, template_beat):
    beats = beats[:,:,200:400]
    if use_corr:
        corrs = np.ones([12, beats.shape[0], beats.shape[0]])
    else:
        corrs = np.zeros([12, beats.shape[0], beats.shape[0]])
    
    mags = np.linalg.norm(template_beat, axis=1)

    best_beat = np.empty(12, dtype=int)
    for i in range(12):
        for j in range(beats.shape[0]):
            for k in range(j+1,beats.shape[0]):
                beat1 = beats[j,i,:]
                beat2 = beats[k,i,:]
                beat1 = np.reshape(beat1, [25,-1])
                beat2 = np.reshape(beat2, [25,-1])
                beat1_stds = np.std(beat1, axis=1)
                beat2_stds = np.std(beat2, axis=1)
                if np.any(beat1_stds == 0) or np.any(beat2_stds == 0):
                    if use_corr:
                        corrs[i,j,k] = 0
                        corrs[i,k,j] = 0
                    else:
                        corrs[i,j,k] = 1000
                        corrs[i,k,j] = 1000
                else:
                    if use_corr:
                        corrs[i,j,k] = np.corrcoef(beats[j,i,:], beats[k,i,:])[0,1]
                    else:
                        mag = np.sqrt(np.linalg.norm(beats[j,i,:])**2 + np.linalg.norm(beats[k,i,:])**2)
                        corrs[i,j,k] = np.linalg.norm(beats[j,i,:] - beats[k,i,:]) / mags[i]
                    corrs[i,k,j] = corrs[i,j,k]

        if use_corr:
            best_beat[i] = np.argmax(np.sum(corrs[i,:,:], axis=-1))
        else:
            best_beat[i] = np.argmin(np.sum(corrs[i,:,:], axis=-1))

    best_corrs = np.zeros([12,beats.shape[0]])
    for i in range(12):
        best_corrs[i] = corrs[i,best_beat[i],:]

    if use_corr:
        best_corrs = best_corrs > 0.8
    else:
        best_corrs = best_corrs < .8

    return best_corrs
       
def getMedianBeat(ecg):    

    if np.std(ecg) == 0:
        beats = []
        for i in range(12):
            beats.append(np.zeros([1,600]))
        return {'median_beat': np.zeros([12, 600]), 'rrint': 450, 'beats': beats, 'rpeaks': np.zeros(1)}

############################################################################################################
# Get R-Peaks
############################################################################################################
    # Apply QRS Filter
    # ecg = np.clip(ecg, -2500, 2500)
    b_qrs, a_qrs = signal.butter(2, [8, 40], btype='bandpass', fs=500)
    ecg = np.concatenate((np.flip(ecg[:,:1000],axis=1), ecg, np.flip(ecg[:,-1000:],axis=1)), axis=1)
    ecg_qrs = signal.filtfilt(b_qrs, a_qrs, ecg, axis=1)
    ecg_qrs = ecg_qrs[:,1000:-1000]
    ecg = ecg[:,1000:-1000]

    rms_ecg = np.sqrt(np.sum(ecg**2,axis=0)) / np.sqrt(12)
    rms_qrs = np.sqrt(np.sum(ecg_qrs**2,axis=0)) / np.sqrt(12)

    # Find R-Peaks in QRS signal
    rpeaks = signal.find_peaks(rms_qrs, distance=100, height=[np.min([300,np.max(rms_qrs[300:4700])/2]),np.min([2500,np.max(rms_qrs[300:4700])])])[0]
    if len(rpeaks) < 5:
        rpeaks = signal.find_peaks(rms_qrs, distance=100, height=[np.min([75,np.max(rms_qrs[300:4700])/2]),np.min([2500,np.max(rms_qrs[300:4700])])])[0]

    thresh = np.median(rms_qrs[rpeaks])
    rpeaks = signal.find_peaks(rms_qrs, distance=100, height=[thresh/2,np.min([thresh*2,2500])])[0]
    rpeaks = rpeaks[(rpeaks < 4700) & (rpeaks > 300)]  

    # Find R-Peaks in RMS signal
    rpeaks1 = np.empty(len(rpeaks), dtype=int)
    for j in range(len(rpeaks)):
        rpeaks1[j] = np.argmax(rms_ecg[rpeaks[j]-30:rpeaks[j]+30])+rpeaks[j]-30
    rpeaks1 = rpeaks1[(rpeaks1 < 4675) & (rpeaks1 > 325)]

    # Remove low amplitude R-Peaks
    thresh = np.median(rms_ecg[rpeaks1]) * 0.8
    good = rpeaks1[rms_ecg[rpeaks1] > thresh]
    if len(good) > 5:
        rpeaks1 = rpeaks1[rms_ecg[rpeaks1] > thresh]

############################################################################################################
# Get Median Beat
############################################################################################################
    # get all beats
    beats = np.empty([len(rpeaks1), 12, 600])
    for j in range(len(rpeaks1)):
        beats[j,:,:] = ecg[:,rpeaks1[j]-300:rpeaks1[j]+300]
    beats -= np.median(beats[:,:,200:275],axis=2)[:,:,None]
    template_beat_idx = getTemplateBeat(beats)
    template_beat = beats[template_beat_idx,:,:]

    # time shift
    for j in range(len(rpeaks1)):
        corrs = np.zeros(50)
        for idx,shift in enumerate(range(-25,25)):
            beat = ecg[:,rpeaks1[j]-100+shift:rpeaks1[j]+100+shift]
            corrs[idx] = np.linalg.norm(beat - template_beat[:,200:400])
        corrs[np.isnan(corrs)] = 100000
        shift = np.argmin(corrs) - 25
        beats[j,:,:] = ecg[:,rpeaks1[j]-300+shift:rpeaks1[j]+300+shift]
    beats -= np.median(beats[:,:,200:275],axis=2)[:,:,None]

    # amplitude shift
    for j in range(len(rpeaks1)):
        for lead in range(12):
            corrs = np.zeros(500)
            for idx,shift in enumerate(range(-250,250)):
                beat = beats[j,lead,:]+shift
                if np.linalg.norm(template_beat[lead,250:300]) != 0:
                    corrs[idx] = np.linalg.norm(beat[250:300] - template_beat[lead,250:300])
                else:
                    corrs[idx] = 0
            shift = np.argmin(corrs) - 250
            beats[j,lead,:] += shift

    template_beat = beats[template_beat_idx,:,:]
    beat_corrs = getValidBeats(beats, template_beat)

    median_beat = np.empty([12, 600])
    for i in range(12):
        median_beat[i,:] = np.median(beats[beat_corrs[i],i,:],axis=0)

    beats1 = []
    for j in range(12):
        beats1.append(beats[beat_corrs[j,:],j,:])

    beat_corrs = np.sum(beat_corrs, axis=0) >= 4
    rpeaks1 = rpeaks1[beat_corrs]

    # RR-Interval
    diffs = np.diff(rpeaks1)
    diffs = np.sort(diffs)
    if len(diffs) == 1:
        rr_int = diffs[0]
    elif len(diffs) == 0:
        rr_int = 450
    else:
        rr_int = np.median(diffs)

############################################################################################################
# Baseline Correction
############################################################################################################
    keep = np.zeros(12, dtype=int)
    max_vals = np.max(np.abs(median_beat),axis=1)
    max_vals[max_vals == 0] = 1
    stds = np.zeros([12, 75])

    #divide each lead by its max value
    sig = median_beat / max_vals[:,None]
    for j in range(12):
        sig[j,:] = np.convolve(sig[j,:],np.ones(5),'same')/5

    # Find first window before R-Peak with STD < 0.005
    for j in range(12):
        for k in range(0,len(stds[j,:])):
            stds[j,k] = np.std(sig[j,k+225-7:k+225])
        
        pt = np.where(stds[j,:] < .005)[0]
        thresh1 = .005
        while len(pt) == 0:
            thresh1 += .001
            pt = np.where(stds[j,:] < thresh1)[0]
        keep[j] = pt[-1] + 225

    # Set baseline at median time-point across leads
    if np.max(keep) == 0:
        keep += 250
        thresh2 = np.median(keep)
    else:
        thresh2 = np.median(keep[keep>0])

    if thresh2 < 240:
        thresh2 = 240
    elif thresh2 > 285:
        thresh2 = 285

    for j in range(12):
        if np.abs(keep[j] - thresh2) > 5:
            keep[j] = np.argmin(stds[j,int(thresh2)-225-5:int(thresh2)-225+15])+int(thresh2)-5

        shift = np.median(median_beat[j,keep[j]-7:keep[j]])
        median_beat[j,:] -= shift

    return {'median_beat': median_beat, 'rrint': rr_int, 'beats': beats1, 'rpeaks': rpeaks1}

def getFiducials(ecg, rr_int, fs):
    ecg[3,:] = -ecg[3,:] # invert aVR
    ecg = signal.resample(ecg, int(1000/fs * ecg.shape[-1]), axis=1)
    fs = 1000
    std_sig = np.std(ecg,axis=0)
    rms_median = np.sqrt(np.sum(ecg**2,axis=0))

############################################################################################################
# Onset QRS
############################################################################################################

    onset_thresh = np.min(rms_median[450:600])
    onset_qrs = np.where(rms_median[450:600] <= onset_thresh+50)[0][-1]+450
    onset_qrs = np.argmin(std_sig[onset_qrs-5:onset_qrs+5])+onset_qrs-5
    for j in range(12):
        ecg[j] -= np.median(ecg[j,onset_qrs-int(0.02*fs):onset_qrs])

############################################################################################################
# Offset QRS
############################################################################################################
    keep = np.zeros(12, dtype=int)
    for j in range(12):
        if np.max(np.abs(ecg[j,:])) > 0:
            sig = ecg[j,:]/np.max(np.abs(ecg[j,550:650]))
        else:
            sig = ecg[j,:]

        sig = np.convolve(sig,np.ones(10),'same')/10

        stds = np.zeros(len(sig))
        for k in range(20,len(stds)):
            stds[k] = np.std(sig[k:k+20])
        
        pt = np.where(stds[610:750] < .012)[0]
        thresh1 = .012
        while len(pt) == 0:
            thresh1 += .001
            pt = np.where(stds[610:750] < thresh1)[0]
        keep[j] = pt[0] + 610
    
    thresh2 = np.median(keep[keep>0])
    for j in range(12):
        if np.max(np.abs(ecg[j,:])) > 0:
            sig = ecg[j,:]/np.max(np.abs(ecg[j,550:650]))
        else:
            sig = ecg[j,:]
        stds = np.zeros(len(sig))
        for k in range(20,len(stds)):
            stds[k] = np.std(sig[k:k+20]) 
        if np.abs(keep[j] - thresh2) > 20:
            keep[j] = np.argmin(stds[int(thresh2)-20:int(thresh2)+20])+int(thresh2)-20

    offset_qrs = np.median(keep).astype(int)

    dmin = np.Inf
    offset_qrs1 = 610
    for j in range(610, 750, 1):
        dsum = np.sum(np.abs(rms_median[j - 3:j+3]))
        if (dsum < dmin-.1*rms_median[600]) or (dsum < dmin-.05*rms_median[600] and j - offset_qrs < 10):
            dmin = dsum
            offset_qrs1 = j

    if rms_median[offset_qrs]/rms_median[offset_qrs1] > 5:
        offset_qrs = offset_qrs1

    offset_qrs = np.argmin(std_sig[offset_qrs-10:offset_qrs+10])+offset_qrs-10
    offset_qrs2 = np.argmin(rms_median[600:800])+600
    offset_qrs = np.min([offset_qrs,offset_qrs2])

    rms_median1 = np.convolve(rms_median,np.ones(20),'same')/20

############################################################################################################
# T peak
############################################################################################################
    median_beat_t = np.copy(ecg)
    for j in range(12):
        median_beat_t[j,:] = np.convolve(median_beat_t[j,:],np.ones(40),'same')/40
        # median_beat_t[j,:] -= median_beat_t[j,offset_qrs+10]
    rms_median_t = np.sqrt(np.sum(median_beat_t**2,axis=0))

    beat_end = np.min([onset_qrs + rr_int*2, onset_qrs + 450, offset_qrs+300])
    beat_end = np.round(beat_end).astype(int)

    if onset_qrs - 200 + rr_int*2 > 1200:
        beat_end = 1200

    peaks = signal.find_peaks(rms_median_t[offset_qrs+40:beat_end])[0]+offset_qrs+40
    if len(peaks) > 0:
        tpeak = np.argmax(rms_median_t[peaks])
        tpeak = peaks[tpeak]
    else:
        tpeak = 0

############################################################################################################
# T Onset
############################################################################################################
    if tpeak == 0:
        onset_t = 0
    else:
        drms = np.diff(rms_median_t)
        max_loc = np.argmax(drms[tpeak-100:tpeak]) + tpeak - 100
        max_slope = np.max(drms[tpeak-100:tpeak])
        if max_slope != 0:
            amp = np.min(rms_median_t[offset_qrs+20:tpeak])
            onset_t = max_loc - int((rms_median_t[max_loc]-amp)/max_slope)
            if onset_t < 10:
                onset_t = 0
            else:
                onset_t = np.argmin(rms_median[onset_t-10:onset_t+10])+onset_t-10
        else:
            onset_t = 0

    if (onset_t >= tpeak) and (tpeak > 0):
        onset_t = np.argmin(rms_median_t[tpeak-50:tpeak])+tpeak-50

############################################################################################################
# T Offset
############################################################################################################
    if tpeak == 0:
        offset_t = 0
    else:
        min_loc = np.argmin(drms[tpeak:tpeak+100]) + tpeak
        min_slope = np.min(drms[tpeak:tpeak+100])
        if min_slope != 0:
            amp = np.min(rms_median_t[tpeak:tpeak+int(3/2*(tpeak-onset_t))])
            offset_t = min_loc - int((rms_median_t[min_loc]-amp)/min_slope)
            if offset_t > 1190:
                offset_t = 1190
            offset_t = np.argmin(rms_median[offset_t-10:offset_t+10])+offset_t-10
        else:
            offset_t = 0

    if offset_t < tpeak:
        offset_t = np.argmin(rms_median_t[tpeak:tpeak+50])+tpeak

    if offset_t > beat_end:
        offset_t = beat_end
   
############################################################################################################
# P Peak, Onset, Offset
############################################################################################################
    beat_start = np.max([300, offset_t - rr_int*2])
    beat_start = np.round(beat_start).astype(int)

    if onset_qrs - beat_start < 100:
        onset_p = 0
        offset_p = 0
        ppeak = 0
    else:
        ppeak = np.argmax(rms_median1[beat_start+20:onset_qrs-20]) + beat_start+20

        rng_onset = rms_median1[ppeak] - np.min(rms_median1[beat_start:ppeak])
        rng_offset = rms_median1[ppeak] - np.min(rms_median1[ppeak:onset_qrs])

        start = np.max([ppeak-100, beat_start])
        stop = np.min([ppeak+100, onset_qrs])

        onset_p = np.where(rms_median1[start:ppeak] - np.min(rms_median1[start:ppeak]) < 0.1*rng_onset)[0]
        if len(onset_p) > 0:
            onset_p = onset_p[-1]+start
        else:
            onset_p = 0

        offset_p = np.where(rms_median1[ppeak:stop] - np.min(rms_median1[ppeak:stop]) < 0.1*rng_offset)[0]
        if len(offset_p) > 0:
            offset_p = offset_p[0]+ppeak
        else:
            offset_p = 0

        if onset_p == 0 or offset_p == 0:
            onset_p = 0
            offset_p = 0

############################################################################################################
# Estimate Missing Fiducials
############################################################################################################
    # Estimate T if missing
    if (onset_qrs > 0) & (offset_qrs > 0) & (offset_t < offset_qrs):
        offset_t = onset_qrs + 400
        onset_t = offset_t - 160

    if onset_t <= offset_qrs:
        onset_t = offset_qrs + 10
        if offset_t - onset_t < 100:
            offset_t = onset_t + 160

############################################################################################################
# QRS Onset Local
############################################################################################################
    qrs_onsets_local = np.empty(12, dtype=int)
    fs = 1000
    for j in range(12):
        qrs_onsets_local[j] = np.argmin(np.abs(ecg[j,onset_qrs-int(0.005*fs):onset_qrs+int(0.005*fs)]))+onset_qrs-int(0.005*fs)

############################################################################################################
# QRS Offset Local
############################################################################################################
    qrs_offsets_local = np.empty(12, dtype=int)
    for j in range(12):
        best_window = np.Inf
        d1 = np.gradient(ecg[j,:])
        for k in range(offset_qrs - 5, offset_qrs + 5):
            window = np.mean(np.abs(d1[k-5:k+5]))
            if window < best_window:
                best_window = window
                qrs_offsets_local[j] = k

############################################################################################################
# QRS Segmenter
############################################################################################################

    q_peaks_local = np.zeros(12, dtype=int)
    s_peaks_local = np.zeros(12, dtype=int)
    r_peaks_local = np.zeros(12, dtype=int)
    rp_peaks_local = np.zeros(12, dtype=int)
    sp_peaks_local = np.zeros(12, dtype=int)

    q_onsets_local = np.zeros(12, dtype=int)
    q_offsets_local = np.zeros(12, dtype=int)
    s_onsets_local = np.zeros(12, dtype=int)
    s_offsets_local = np.zeros(12, dtype=int)
    r_onsets_local = np.zeros(12, dtype=int)
    r_offsets_local = np.zeros(12, dtype=int)
    rp_onsets_local = np.zeros(12, dtype=int)
    rp_offsets_local = np.zeros(12, dtype=int)
    sp_onsets_local = np.zeros(12, dtype=int)
    sp_offsets_local = np.zeros(12, dtype=int)

    for j in range(12):
        # zero crossings of ecg
        zero_crossings = np.where(np.diff(np.sign(ecg[j,:])))[0]
        prom_r = 20
        prom = 20
        rpeaks = signal.find_peaks(ecg[j,qrs_onsets_local[j]:qrs_offsets_local[j]], prominence=prom_r, distance=10, width=5)[0]+qrs_onsets_local[j]
        rpeaks = rpeaks[ecg[j,rpeaks] > 0]
        #keep the top two largest peaks
        if len(rpeaks) > 1:
            rpeaks = rpeaks[ecg[j,rpeaks] > 0.2*np.max(ecg[j,rpeaks])]
        if len(rpeaks) > 2:
            rpeaks = rpeaks[np.argsort(ecg[j,rpeaks])[-2:]]

        if len(rpeaks) > 1:
            if ecg[j,rpeaks[0]] < 0.2*ecg[j,rpeaks[1]]:
                rpeaks = np.delete(rpeaks,0)
                r_peaks_local[j] = rpeaks[0]
            elif ecg[j,rpeaks[1]] < 0.2*ecg[j,rpeaks[0]]:
                rpeaks = np.delete(rpeaks,1)
                r_peaks_local[j] = rpeaks[0]
            else:
                r_peaks_local[j] = np.min(rpeaks)
                rp_peaks_local[j] = np.max(rpeaks)
        elif len(rpeaks) > 0:
            r_peaks_local[j] = rpeaks[0]

        rpeaks = np.sort(rpeaks)

        if len(rpeaks) > 0:
            q_peaks = signal.find_peaks(-ecg[j,qrs_onsets_local[j]:rpeaks[0]], prominence=prom, width=5, distance=10)[0] + qrs_onsets_local[j]
            q_peaks = q_peaks[ecg[j,q_peaks] < -10]
            if len(q_peaks) > 0:
                q_peaks_local[j] = q_peaks[np.argsort(ecg[j,q_peaks])[0]]

            if len(rpeaks) == 1:
                s_peaks = signal.find_peaks(-ecg[j,rpeaks[-1]:qrs_offsets_local[j]], prominence=prom, width=5, distance=10)[0] + rpeaks[-1]
                s_peaks = s_peaks[ecg[j,s_peaks] < -10]
                if len(s_peaks) >= 2:
                    s_peaks = s_peaks[np.argsort(ecg[j,s_peaks])[0:2]]
                    s_peaks_local[j] = np.min(s_peaks)
                    sp_peaks_local[j] = np.max(s_peaks)
                elif len(s_peaks) == 1:
                    s_peaks_local[j] = s_peaks[np.argsort(ecg[j,s_peaks])[0]]

            elif len(rpeaks) == 2:
                s_peaks = signal.find_peaks(-ecg[j,rpeaks[0]:rpeaks[1]], prominence=prom, width=5, distance=10)[0] + rpeaks[0]
                s_peaks = s_peaks[ecg[j,s_peaks] < -10]
                if len(s_peaks) > 0:
                    s_peaks_local[j] = s_peaks[np.argsort(ecg[j,s_peaks])[0]]
                sp_peaks = signal.find_peaks(-ecg[j,rpeaks[1]:qrs_offsets_local[j]], prominence=prom, width=5, distance=10)[0] + rpeaks[1]
                sp_peaks = sp_peaks[ecg[j,sp_peaks] < -10]
                if len(sp_peaks) > 0 and len(s_peaks) > 0:
                    sp_peaks_local[j] = sp_peaks[np.argsort(ecg[j,sp_peaks])[0]]
                elif len(sp_peaks) > 0:
                    s_peaks_local[j] = sp_peaks[np.argsort(ecg[j,sp_peaks])[0]]
        else:
            qspeaks = signal.find_peaks(-ecg[j,qrs_onsets_local[j]:qrs_offsets_local[j]], prominence=prom, distance=10, width=5)[0]+qrs_onsets_local[j]
            qspeaks = qspeaks[ecg[j,qspeaks] < -10]

            if len(qspeaks) > 2:
                qspeaks = qspeaks[np.argsort(ecg[j,qspeaks])[0:2]]
                qspeaks = np.sort(qspeaks)

            if len(qspeaks) == 2:
                q_peaks_local[j] = qspeaks[0]
                s_peaks_local[j] = qspeaks[1]
            elif len(qspeaks) == 1:
                q_peaks_local[j] = qspeaks[0]

############################################################################################################
# Q Onset/Offset
############################################################################################################
        if q_peaks_local[j] > 0:
            onset1 = zero_crossings[zero_crossings < q_peaks_local[j]]
            if len(onset1) > 0:
                q_onsets_local[j] = np.max([onset1[-1], qrs_onsets_local[j]])

            offset1 = zero_crossings[zero_crossings > q_peaks_local[j]]
            if len(offset1) > 0:
                q_offsets_local[j] = offset1[0]

############################################################################################################
# R Onset/Offset
############################################################################################################
        if r_peaks_local[j] > 0:
            r_onsets_local[j] = np.max([zero_crossings[zero_crossings < r_peaks_local[j]][-1], q_offsets_local[j]])
            if len(rpeaks) == 2:
                rmin = np.argmin(ecg[j,rpeaks[0]:rpeaks[1]])+rpeaks[0]
                offset1 = zero_crossings[zero_crossings > r_peaks_local[j]]
                if len(offset1) > 0:
                    r_offsets_local[j] = np.min([offset1[0],rmin])
                else:
                    r_offsets_local[j] = rmin
            else:
                offset1 = zero_crossings[zero_crossings > r_peaks_local[j]]
                if len(offset1) > 0:
                    r_offsets_local[j] = np.min([offset1[0],qrs_offsets_local[j]])
                else:
                    r_offsets_local[j] = qrs_offsets_local[j]

############################################################################################################
# R' Onset/Offset
############################################################################################################
        if rp_peaks_local[j] > 0:
            rmin = np.argmin(ecg[j,rpeaks[0]:rpeaks[1]])+rpeaks[0]
            onset1 = zero_crossings[zero_crossings < rp_peaks_local[j]]
            if len(onset1) > 0:
                rp_onsets_local[j] = np.max([onset1[-1], rmin])
            else:
                rp_onsets_local[j] = rmin

            offset1 = zero_crossings[zero_crossings > rp_peaks_local[j]]
            if len(offset1) > 0:
                rp_offsets_local[j] = np.min([offset1[0],qrs_offsets_local[j]])
            else:
                rp_offsets_local[j] = qrs_offsets_local[j]

############################################################################################################
# S Onset/Offset
############################################################################################################
        if s_peaks_local[j] > 0:
            if q_peaks_local[j] > r_peaks_local[j]:
                smax = np.argmax(ecg[j,q_peaks_local[j]:s_peaks_local[j]])+q_peaks_local[j]
                onset1 = zero_crossings[zero_crossings < s_peaks_local[j]]
                if len(onset1) > 0:
                    s_onsets_local[j] = np.max([onset1[-1], smax])
                else:
                    s_onsets_local[j] = smax
            else:
                onset1 = zero_crossings[zero_crossings < s_peaks_local[j]]
                if len(onset1) > 0:
                    s_onsets_local[j] = onset1[-1]
                else:
                    s_onsets_local[j] = qrs_onsets_local[j]

            if sp_peaks_local[j] > 0:
                smax = np.argmax(ecg[j,s_peaks_local[j]:sp_peaks_local[j]])+s_peaks_local[j]
                offset1 = zero_crossings[zero_crossings > s_peaks_local[j]]
                if len(offset1) > 0:
                    s_offsets_local[j] = np.min([offset1[0], smax])
                else:
                    s_offsets_local[j] = smax
            else:
                offset1 = zero_crossings[zero_crossings > s_peaks_local[j]]
                if len(offset1) > 0:
                    s_offsets_local[j] = np.min([offset1[0],qrs_offsets_local[j]])
                else:
                    s_offsets_local[j] = qrs_offsets_local[j]
            if s_offsets_local[j] <= q_offsets_local[j]:
                q_offsets_local[j] = np.argmax(ecg[j,q_peaks_local[j]:s_peaks_local[j]])+q_peaks_local[j]

############################################################################################################
# S' Onset/Offset
############################################################################################################
        if sp_peaks_local[j] > 0:
            if s_peaks_local[j] > rp_peaks_local[j]:
                smax = np.argmax(ecg[j,s_peaks_local[j]:sp_peaks_local[j]])+s_peaks_local[j]
                onset1 = zero_crossings[zero_crossings < sp_peaks_local[j]]
                if len(onset1) > 0:
                    sp_onsets_local[j] = np.max([onset1[-1], smax])
                else:
                    sp_onsets_local[j] = smax
            else:
                onset1 = zero_crossings[zero_crossings < sp_peaks_local[j]]
                if len(onset1) > 0:
                    sp_onsets_local[j] = onset1[-1]
                else:
                    sp_onsets_local[j] = np.max([s_offsets_local[j]], r_offsets_local[j], rp_offsets_local[j])

            offset1 = zero_crossings[zero_crossings > sp_peaks_local[j]]
            if len(offset1) > 0:
                sp_offsets_local[j] = np.min([offset1[0],qrs_offsets_local[j]])
            else:
                sp_offsets_local[j] = qrs_offsets_local[j]

############################################################################################################
# Fix labeling of S wave if mislabeled as Q wave
############################################################################################################
        if (q_offsets_local[j] > onset_qrs + 100) | (q_offsets_local[j] > 600):
            if s_peaks_local[j] == 0:
                s_peaks_local[j] = q_peaks_local[j]
                s_onsets_local[j] = q_onsets_local[j]
                s_offsets_local[j] = q_offsets_local[j]
                q_peaks_local[j] = 0
                q_onsets_local[j] = 0
                q_offsets_local[j] = 0
            else:
                sp_peaks_local[j] = s_peaks_local[j]
                sp_onsets_local[j] = s_onsets_local[j]
                sp_offsets_local[j] = s_offsets_local[j]
                s_peaks_local[j] = q_peaks_local[j]
                s_onsets_local[j] = q_onsets_local[j]
                s_offsets_local[j] = q_offsets_local[j]
                q_peaks_local[j] = 0
                q_onsets_local[j] = 0
                q_offsets_local[j] = 0

            if r_peaks_local[j] > 0:
                rp_peaks_local[j] = r_peaks_local[j]
                rp_onsets_local[j] = r_onsets_local[j]
                rp_offsets_local[j] = r_offsets_local[j]
                r_peaks_local[j] = 0
                r_onsets_local[j] = 0
                r_offsets_local[j] = 0

############################################################################################################

    fiducials = np.array([onset_p,offset_p,onset_qrs,offset_qrs,onset_t,offset_t],dtype=int)
    fiducials = fiducials + 1e-10
    fiducials = np.rint(fiducials/2).astype(int)

    q_peaks_local = np.rint(q_peaks_local/2+1e-10).astype(int)
    s_peaks_local = np.rint(s_peaks_local/2+1e-10).astype(int)
    r_peaks_local = np.rint(r_peaks_local/2+1e-10).astype(int)
    rp_peaks_local = np.rint(rp_peaks_local/2+1e-10).astype(int)
    sp_peaks_local = np.rint(sp_peaks_local/2+1e-10).astype(int)
    qrs_onsets_local = np.rint(qrs_onsets_local/2+1e-10).astype(int)
    qrs_offsets_local = np.rint(qrs_offsets_local/2+1e-10).astype(int)
    q_onsets_local = np.rint(q_onsets_local/2+1e-10).astype(int)
    q_offsets_local = np.rint(q_offsets_local/2+1e-10).astype(int)
    r_onsets_local = np.rint(r_onsets_local/2+1e-10).astype(int)
    r_offsets_local = np.rint(r_offsets_local/2+1e-10).astype(int)
    s_onsets_local = np.rint(s_onsets_local/2+1e-10).astype(int)
    s_offsets_local = np.rint(s_offsets_local/2+1e-10).astype(int)
    rp_onsets_local = np.rint(rp_onsets_local/2+1e-10).astype(int)
    rp_offsets_local = np.rint(rp_offsets_local/2+1e-10).astype(int)
    sp_onsets_local = np.rint(sp_onsets_local/2+1e-10).astype(int)
    sp_offsets_local = np.rint(sp_offsets_local/2+1e-10).astype(int)

    fiducials_local = {'q_peaks':q_peaks_local, 's_peaks':s_peaks_local, 'r_peaks':r_peaks_local, 'rp_peaks':rp_peaks_local, 'sp_peaks': sp_peaks_local, 'qrs_onsets':qrs_onsets_local, 'qrs_offsets':qrs_offsets_local, 'q_onsets': q_onsets_local, 'q_offsets': q_offsets_local, 'r_onsets': r_onsets_local, 'r_offsets': r_offsets_local, 's_onsets': s_onsets_local, 's_offsets': s_offsets_local, 'rp_onsets': rp_onsets_local, 'rp_offsets': rp_offsets_local, 'sp_onsets': sp_onsets_local, 'sp_offsets': sp_offsets_local}

    return {'local': fiducials_local, 'global': fiducials}

# Inverse Dower Transform V1-V6 I II
DT = np.array([[-0.172, -0.074, 0.122, 0.231, 0.239, 0.194, 0.156, -0.010], 
               [0.057, -0.019, -0.106, -0.022, 0.041, 0.048, -0.227, 0.887],
               [-0.229, -0.310, -0.246, -0.063, 0.055, 0.108, 0.022, 0.102]])

# Kors Regression Transform V1-V6 I II
KT = np.array([[-0.130, 0.050, -0.010, 0.140, 0.060, 0.540, 0.380, -0.070],
               [0.060, -0.020, -0.050, 0.060, -0.170, 0.130, -0.070, 0.930],
               [-0.430, -0.060, -0.140, -0.200, -0.110, 0.310, 0.110, -0.230]])

# Local features #######################################################################################################
def get_p_area_local(fiducials, fiducials_local, ecg, fs):
    qrs_onsets = fiducials_local['qrs_onsets']
    p_areas_local = np.zeros(12)
    for j in range(12):
        if fiducials[0] > 0:
            dur = int(qrs_onsets[j]-200)
            p_areas_local[j] = integrate.simpson(y=np.abs(ecg[j,200:qrs_onsets[j]]), x=np.linspace(0,dur-1,dur)/fs)
    return p_areas_local

def get_qrs_area_local(fiducials_local, ecg,fs):
    qrs_onsets = fiducials_local['qrs_onsets']
    qrs_offsets = fiducials_local['qrs_offsets']
    qrs_areas_local = np.zeros(12)

    for j in range(12):
        dur = int(qrs_offsets[j]-qrs_onsets[j])
        if dur < 1:
            qrs_areas_local[j] = 0
        else:
            qrs_areas_local[j] = integrate.simpson(y=np.abs(ecg[j,qrs_onsets[j]:qrs_offsets[j]]), x=np.linspace(0,dur-1,dur)/fs)
    return qrs_areas_local

def get_t_area_local(fiducials, fiducials_local, ecg, fs):
    qrs_offsets = fiducials_local['qrs_offsets']
    t_areas_local = np.zeros(12)

    for j in range(12):
        dur = int(fiducials[5]-qrs_offsets[j])
        if dur > 0:
            t_areas_local[j] = integrate.simpson(y=np.abs(ecg[j,qrs_offsets[j]:int(fiducials[5])]),x=np.linspace(0,dur-1,dur)/fs)
    return t_areas_local

def get_st_peak_area_local(fiducials, fiducials_local, ecg, fs):
    st_areas_local = np.zeros(12)
    tpeaks = np.zeros(12, dtype=int)
    qrs_offsets = fiducials_local['qrs_offsets']
    for j in range(12):
        tpeaks[j] = np.argmax(np.abs(ecg[j, int(fiducials[4]):int(fiducials[5])])) + int(fiducials[4])
        dur = int(tpeaks[j]-qrs_offsets[j])
        if dur > 1:
            st_areas_local[j] = integrate.simpson(y=np.abs(ecg[j,qrs_offsets[j]:tpeaks[j]]), x=np.linspace(0,dur-1,dur)/fs)
    return st_areas_local

def get_p_amps_local(fiducials, ecg):
    p_amps_local = np.zeros(12)
    if fiducials[0] > 0 and fiducials[1] > 0:
        for j in range(12):
            max_loc = np.argmax(np.abs(ecg[j,int(fiducials[0]):int(fiducials[1])])) + int(fiducials[0])
            p_amps_local[j] = ecg[j,max_loc]
    return p_amps_local

def get_qrs_amps_local(fiducials_local, ecg):
    qrs_onsets = fiducials_local['qrs_onsets']
    qrs_offsets = fiducials_local['qrs_offsets']
    qrs_amps_local = np.zeros(12)
    for j in range(12):
        if qrs_onsets[j] < qrs_offsets[j]:
            max_loc = np.argmax(np.abs(ecg[j,qrs_onsets[j]:qrs_offsets[j]])) + qrs_onsets[j]
            qrs_amps_local[j] = ecg[j,max_loc]
    return qrs_amps_local

def get_t_amps_local(fiducials, ecg):
    t_amps_local = np.zeros(12)
    for j in range(12):
        max_loc = np.argmax(np.abs(ecg[j,int(fiducials[4]):int(fiducials[5])])) + int(fiducials[4])
        t_amps_local[j] = ecg[j,max_loc]
    return t_amps_local

def get_st_amp_local(fiducials_local, ecg, fs):
    qrs_offsets = fiducials_local['qrs_offsets']
    st_amps_local = np.zeros(12)
    for j in range(12):
        st_amps_local[j] = np.median(ecg[j,qrs_offsets[j]:qrs_offsets[j]+int(0.08*fs)])
    return st_amps_local

def get_t_qrs_ratios_local(fiducials, fiducials_local, ecg):
    t_qrs_ratios_local = np.zeros(12)
    t_amps = get_t_amps_local(fiducials, ecg)
    qrs_amps = get_qrs_amps_local(fiducials_local, ecg)
    for j in range(12):
        if qrs_amps[j] != 0:
            t_qrs_ratios_local[j] = np.abs(t_amps[j] / qrs_amps[j])
    return t_qrs_ratios_local

def get_st_slope(fiducials_local, ecg, fs):
    qrs_offsets = fiducials_local['qrs_offsets']
    st_slope_local = np.zeros(12)

    for j in range(12):
        start = np.median(ecg[j, qrs_offsets[j]:qrs_offsets[j]+int(0.04*fs)]) # J+20
        end = np.median(ecg[j, qrs_offsets[j]+int(0.06*fs):qrs_offsets[j]+int(0.1*fs)]) # J+80
        st_slope_local[j] = (end - start) / 60 # uV/msec = mV/sec

    return st_slope_local

def get_qrs_slope(fiducials_local, ecg, fs):
    qrs_onsets = fiducials_local['qrs_onsets']
    qrs_offsets = fiducials_local['qrs_offsets']
    qrs_slope_local = np.zeros(12)
    for j in range(12):
        start = np.median(ecg[j, qrs_onsets[j]-int(0.04*fs):qrs_onsets[j]])
        end = np.median(ecg[j, qrs_offsets[j]:qrs_offsets[j]+int(0.04*fs)])
        qrs_slope_local[j] = (end-start) / (qrs_offsets[j] - qrs_onsets[j] + int(0.04*fs)) * fs /1000 # uV/msec = mV/sec
    return qrs_slope_local

def get_beat_to_beat_std(segment): # segment is leads x beats x datapoints
    b2b_stds = np.zeros(len(segment))

    for i in range(len(segment)): # for each lead
        segment_lead = segment[i] # beats x datapoints
        a = np.std(segment_lead, axis=0) # datapoints
        b2b_stds[i] = np.mean(a) # mean of the std of each datapoint

    return b2b_stds # uV

def get_q_amps_local(fiducials_local, ecg):
    q_peaks = fiducials_local['q_peaks']
    q_amps_local = np.zeros(12)
    for j in range(12):
        if q_peaks[j] > 0:
            q_amps_local[j] = ecg[j,q_peaks[j]]
    return q_amps_local

def get_q_durations_local(fiducials_local, fs):
    q_onsets = fiducials_local['q_onsets']
    q_offsets = fiducials_local['q_offsets']
    q_durations_local = np.zeros(12)
    for j in range(12):
        q_durations_local[j] = (q_offsets[j] - q_onsets[j]) / fs
    return q_durations_local

def get_q_areas_local(fiducials_local, ecg, fs):
    q_onsets = fiducials_local['q_onsets']
    q_offsets = fiducials_local['q_offsets']
    q_areas_local = np.zeros(12)
    for j in range(12):
        if q_onsets[j] < q_offsets[j]:
            dur = int(q_offsets[j]-q_onsets[j])
            q_areas_local[j] = integrate.simpson(y=np.abs(ecg[j,q_onsets[j]:q_offsets[j]]), x=np.linspace(0,dur-1,dur)/fs)
        else:
            q_areas_local[j] = 0
    return q_areas_local

def get_s_amps_local(fiducials_local, ecg):
    s_peaks = fiducials_local['s_peaks']
    s_amps_local = np.zeros(12)
    for j in range(12):
        if s_peaks[j] > 0:
            s_amps_local[j] = ecg[j,s_peaks[j]]
    return s_amps_local

def get_s_durations_local(fiducials_local, fs):
    s_onsets = fiducials_local['s_onsets']
    s_offsets = fiducials_local['s_offsets']
    s_durations_local = np.zeros(12)
    for j in range(12):
        s_durations_local[j] = (s_offsets[j] - s_onsets[j]) / fs
    return s_durations_local

def get_s_areas_local(fiducials_local, ecg, fs):
    s_onsets = fiducials_local['s_onsets']
    s_offsets = fiducials_local['s_offsets']
    s_areas_local = np.zeros(12)
    for j in range(12):
        if s_onsets[j] < s_offsets[j]:
            dur = int(s_offsets[j]-s_onsets[j])
            s_areas_local[j] = integrate.simpson(y=np.abs(ecg[j,s_onsets[j]:s_offsets[j]]), x=np.linspace(0,dur-1,dur)/fs)
        else:
            s_areas_local[j] = 0
    return s_areas_local

def get_r_amps_local(fiducials_local, ecg):
    r_peaks = fiducials_local['r_peaks']
    r_amps_local = np.zeros(12)
    for j in range(12):
        if r_peaks[j] > 0:
            r_amps_local[j] = ecg[j,r_peaks[j]]
    return r_amps_local

def get_r_durations_local(fiducials_local, fs):
    r_onsets = fiducials_local['r_onsets']
    r_offsets = fiducials_local['r_offsets']
    r_durations_local = np.zeros(12)
    for j in range(12):
        r_durations_local[j] = (r_offsets[j] - r_onsets[j]) / fs
    return r_durations_local

def get_r_areas_local(fiducials_local, ecg, fs):
    r_onsets = fiducials_local['r_onsets']
    r_offsets = fiducials_local['r_offsets']
    r_areas_local = np.zeros(12)
    for j in range(12):
        if r_onsets[j] < r_offsets[j]:
            dur = int(r_offsets[j]-r_onsets[j])
            r_areas_local[j] = integrate.simpson(y=np.abs(ecg[j,r_onsets[j]:r_offsets[j]]), x=np.linspace(0,dur-1,dur)/fs)
        else:
            r_areas_local[j] = 0
    return r_areas_local

def get_rp_amps_local(fiducials_local, ecg):
    rp_peaks = fiducials_local['rp_peaks']
    rp_amps_local = np.zeros(12)
    for j in range(12):
        if rp_peaks[j] > 0:
            rp_amps_local[j] = ecg[j,rp_peaks[j]]
    return rp_amps_local

def get_rp_durations_local(fiducials_local, fs):
    rp_onsets = fiducials_local['rp_onsets']
    rp_offsets = fiducials_local['rp_offsets']
    rp_durations_local = np.zeros(12)
    for j in range(12):
        rp_durations_local[j] = (rp_offsets[j] - rp_onsets[j]) / fs
    return rp_durations_local

def get_rp_areas_local(fiducials_local, ecg, fs):
    rp_onsets = fiducials_local['rp_onsets']
    rp_offsets = fiducials_local['rp_offsets']
    rp_areas_local = np.zeros(12)
    for j in range(12):
        if rp_onsets[j] < rp_offsets[j]:
            dur = int(rp_offsets[j]-rp_onsets[j])
            rp_areas_local[j] = integrate.simpson(y=np.abs(ecg[j,rp_onsets[j]:rp_offsets[j]]), x=np.linspace(0,dur-1,dur)/fs)
        else:
            rp_areas_local[j] = 0
    return rp_areas_local

def get_sp_amps_local(fiducials_local, ecg):
    sp_peaks = fiducials_local['sp_peaks']
    sp_amps_local = np.zeros(12)
    for j in range(12):
        if sp_peaks[j] > 0:
            sp_amps_local[j] = ecg[j,sp_peaks[j]]
    return sp_amps_local

def get_sp_durations_local(fiducials_local, fs):
    sp_onsets = fiducials_local['sp_onsets']
    sp_offsets = fiducials_local['sp_offsets']
    sp_durations_local = np.zeros(12)
    for j in range(12):
        sp_durations_local[j] = (sp_offsets[j] - sp_onsets[j]) / fs
    return sp_durations_local

def get_sp_areas_local(fiducials_local, ecg, fs):
    sp_onsets = fiducials_local['sp_onsets']
    sp_offsets = fiducials_local['sp_offsets']
    sp_areas_local = np.zeros(12)
    for j in range(12):
        if sp_onsets[j] < sp_offsets[j]:
            dur = int(sp_offsets[j]-sp_onsets[j])
            sp_areas_local[j] = integrate.simpson(y=np.abs(ecg[j,sp_onsets[j]:sp_offsets[j]]), x=np.linspace(0,dur-1,dur)/fs)
        else:
            sp_areas_local[j] = 0
    return sp_areas_local

def get_vat_local(fiducials_local, fs, ecg):
    r_peaks = fiducials_local['r_peaks']
    s_peaks = fiducials_local['s_peaks']
    qrs_onsets = fiducials_local['qrs_onsets']
    vat_local = np.zeros(12)
    for j in range(12):
        if r_peaks[j] > 0 and s_peaks[j] > 0:
            if ecg[j,r_peaks[j]] > 0.9 * ecg[j,s_peaks[j]] and ecg[j,s_peaks[j]] > 0.9 * ecg[j,r_peaks[j]]:
                vat_local[j] = (np.mean([r_peaks[j], s_peaks[j]]) - qrs_onsets[j]) / fs
            elif ecg[j, r_peaks[j]] > ecg[j, s_peaks[j]]:
                vat_local[j] = (r_peaks[j] - qrs_onsets[j]) / fs
            elif ecg[j, s_peaks[j]] > ecg[j, r_peaks[j]]:
                vat_local[j] = (s_peaks[j] - qrs_onsets[j]) / fs
        elif r_peaks[j] > 0:
            vat_local[j] = (r_peaks[j] - qrs_onsets[j]) / fs
        elif s_peaks[j] > 0:
            vat_local[j] = (s_peaks[j] - qrs_onsets[j]) / fs
    return vat_local

def get_rpsp(fiducials_local):
    rp_peaks = fiducials_local['rp_peaks']
    sp_peaks = fiducials_local['sp_peaks']

    return rp_peaks > 0, sp_peaks > 0

# Global features ######################################################################################################
def get_p_area_global(fiducials, ecg_rms, fs):
    if fiducials[0] > 0 and fiducials[1] > 0:
        dur = int(fiducials[1]-fiducials[0])
        return integrate.simpson(y=ecg_rms[int(fiducials[0]):int(fiducials[1])], x=np.linspace(0,dur-1,dur)/fs)
    return 0

def get_qrs_area_global(fiducials, ecg_rms, fs):
    dur = int(fiducials[3]-fiducials[2])
    return integrate.simpson(y=ecg_rms[int(fiducials[2]):int(fiducials[3])], x=np.linspace(0,dur-1,dur)/fs)  

def get_t_area_global(fiducials, ecg_rms, fs):
    dur = int(fiducials[5]-fiducials[4])
    return integrate.simpson(y=ecg_rms[int(fiducials[4]):int(fiducials[5])], x=np.linspace(0,dur-1,dur)/fs)

def get_p_duration(fiducials, fs):
    return (fiducials[1] - fiducials[0]) / fs

def get_qrs_duration(fiducials, fs):
    return (fiducials[3] - fiducials[2]) / fs

def get_t_duration(fiducials, fs):
    return (fiducials[5] - fiducials[4]) / fs

def get_pr_interval(fiducials, fs):
    if fiducials[0] == 0:
        return 0
    return (fiducials[2] - fiducials[0]) / fs

def get_qt_interval(fiducials, fs):
    return (fiducials[5] - fiducials[2]) / fs

def get_tpte_global(fiducials, ecg_rms, fs):
    tpeak = np.argmax(ecg_rms[int(fiducials[4]):int(fiducials[5])]) + int(fiducials[4])
    return (fiducials[5] - tpeak) / fs

def get_st_peak_area_global(fiducials, ecg_rms, fs):
    tpeak = np.argmax(ecg_rms[int(fiducials[4]):int(fiducials[5])]) + int(fiducials[4])
    dur = int(tpeak - fiducials[3])
    return integrate.simpson(y=ecg_rms[int(fiducials[3]):tpeak], x=np.linspace(0,dur-1,dur)/fs)

def get_st_amp_global(fiducials, ecg_rms, fs):
    st_amp = np.median(ecg_rms[int(fiducials[3]):int(fiducials[3]+0.08*fs)])
    return st_amp

def get_p_amp_global(fiducials, ecg_rms):
    if fiducials[0] > 0 and fiducials[1] > 0:
        return np.max(ecg_rms[int(fiducials[0]):int(fiducials[1])])
    return 0

def get_qrs_amp_global(fiducials, ecg_rms):
    return np.max(ecg_rms[int(fiducials[2]):int(fiducials[3])])

def get_t_amp_global(fiducials, ecg_rms):
    return np.max(ecg_rms[int(fiducials[4]):int(fiducials[5])])

def get_t_qrs_ratio_global(fiducials, ecg_rms):
    return get_t_amp_global(fiducials, ecg_rms) / get_qrs_amp_global(fiducials, ecg_rms)

def get_hr_global(rrint, fs):
    if rrint > 0:
        return (60 / rrint) * fs
    else:
        return 0

def get_qtc_interval(fiducials, rrint, fs):
    if rrint > 0:
        return get_qt_interval(fiducials, fs) / np.sqrt(rrint / fs)
    else:
        return 0

def get_qtc_interval2(fiducials, rrint, fs):
    if rrint > 0:
        return get_qt_interval(fiducials, fs) / np.cbrt(rrint / fs)
    else:
        return 0

def get_jt_global(fiducials, fs):
    return (fiducials[5] - fiducials[3]) / fs

def get_jtc_global(fiducials, rrint, fs):
    if rrint > 0:
        return get_jt_global(fiducials, fs) / np.sqrt(rrint / fs)
    else:
        return 0

def get_jtc_global2(fiducials, rrint, fs):
    if rrint > 0:
        return get_jt_global(fiducials, fs) / np.cbrt(rrint / fs)
    else:
        return 0

def get_tpte_qt_ratio(fiducials, ecg_rms, fs):
    return get_tpte_global(fiducials, ecg_rms, fs) / get_qt_interval(fiducials, fs)

def get_rms_min(ecg_rms):
    return np.min(ecg_rms)

def get_rms_std(ecg_rms):
    return np.std(ecg_rms)

def get_rms_mean(ecg_rms):
    return np.mean(ecg_rms)

def get_rms_median(ecg_rms):
    return np.median(ecg_rms)

def get_vat_global(ecg_rms, fiducials, fs):
    rpeak = np.argmax(ecg_rms[int(fiducials[2]):int(fiducials[3])]) + int(fiducials[2])
    return (rpeak - fiducials[2]) / fs

def get_pca(ecg): # just plug in the data to use (can be QRS, T, ST, STT, TpTe)
    ecg = ecg[[0,1,6,7,8,9,10,11],:]
    if ecg.shape[1] >= 8:
        pca = PCA(n_components=2)
        pca.fit(ecg.T)
        vr = pca.explained_variance_
        return vr[1]/vr[0]
    else:
        return 0

def get_ndpv(ecg): # plug in data to use (QRS or STT)
    ecg = ecg[[0,1,6,7,8,9,10,11],:]
    if ecg.shape[1] >= 8:
        pca = PCA(n_components=8)
        pca.fit(ecg.T)
        vr = pca.explained_variance_
        return np.sum(vr[3:])/np.sum(vr)
    else:
        return 0

def get_rel_psds(ecg, fs): # use 10-second data
    f, psd = signal.welch(ecg, fs=fs, nperseg=int(1.25*fs))

    freq_idx = np.where((f >= 0.5) & (f < 10))[0]
    low_psd = np.trapz(psd[:,freq_idx], f[freq_idx], axis=1)

    freq_idx = np.where((f >= 10) & (f < 50))[0]
    medium_psd = np.trapz(psd[:,freq_idx], f[freq_idx], axis=1)

    freq_idx = np.where((f >= 50) & (f < 100))[0]
    high_psd = np.trapz(psd[:,freq_idx], f[freq_idx], axis=1)

    freq_idx = np.where((f >= 0.5) & (f < 100))[0]
    total_psd = np.trapz(psd[:,freq_idx], f[freq_idx], axis=1)

    total_psd[total_psd==0] = 1
    low_psd /= total_psd
    medium_psd /= total_psd
    high_psd /= total_psd

    low_psd = np.mean(low_psd[low_psd > 0])
    medium_psd = np.mean(medium_psd[medium_psd > 0])
    high_psd = np.mean(high_psd[high_psd > 0])

    return low_psd, medium_psd, high_psd

def get_concavity(ecg, fiducials, fiducials_local):
    qrs_offsets = fiducials_local['qrs_offsets']
    conc_dists = np.zeros(12)
    for j in range(12):
        t_peak = np.argmax(np.abs(ecg[j,int(fiducials[4]):int(fiducials[5])])) + int(fiducials[4])
        if t_peak > qrs_offsets[j]:
            # create a line from qrs offset to t_peak
            x = np.linspace(ecg[j,qrs_offsets[j]], ecg[j,t_peak], t_peak-qrs_offsets[j])
            concavity = x > ecg[j,qrs_offsets[j]:t_peak]
            concavity = np.sum(concavity) / (t_peak-qrs_offsets[j])
            if concavity >= 0.5:
                conc_dists[j] = np.max(x - ecg[j,qrs_offsets[j]:t_peak])
            else:
                conc_dists[j] = np.min(x - ecg[j,qrs_offsets[j]:t_peak])
    # I, aVL, V5, V6
    lat_conc_dists = np.mean(conc_dists[[0,4,10,11]])
    # V1, V2, V3, V4
    ant_conc_dists = np.mean(conc_dists[[6,7,8,9]])
    # II, III, aVF
    inf_conc_dists = np.mean(conc_dists[[1,2,5]])

    return lat_conc_dists, ant_conc_dists, inf_conc_dists

# VCG features #########################################################################################################
def get_vcg(ecg):
    vcg = ecg[[6,7,8,9,10,11,0,1],:]
    vcg_DT = np.matmul(DT, vcg)
    vcg_KT = np.matmul(KT, vcg)

    return vcg_KT

def get_qrs_peak_features(vcg, rpeak):
    qrs_peak_x = vcg[0,rpeak]
    qrs_peak_y = vcg[1,rpeak]
    qrs_peak_z = vcg[2,rpeak]

    qrs_elev = np.rad2deg(np.arctan(qrs_peak_z/qrs_peak_x)) # z/x
    qrs_elev1 = np.rad2deg(np.arctan(qrs_peak_z/np.sqrt(np.square(qrs_peak_x) + np.square(qrs_peak_y))))
    qrs_azim = np.rad2deg(np.arctan(qrs_peak_y/qrs_peak_x)) # y/x
    qrs_zen = np.rad2deg(np.arctan(qrs_peak_z/qrs_peak_y)) # z/y
    qrs_mag = np.sqrt(np.square(qrs_peak_x) + np.square(qrs_peak_y) + np.square(qrs_peak_z))

    return qrs_elev, qrs_elev1, qrs_azim, qrs_zen, qrs_mag

def get_t_peak_features(vcg, tpeak):
    if np.isnan(tpeak):
        return 0, 0, 0, 0, 0
    t_peak_x = vcg[0,tpeak]
    t_peak_y = vcg[1,tpeak]
    t_peak_z = vcg[2,tpeak]

    t_elev = np.rad2deg(np.arctan(t_peak_z/t_peak_x)) # z/x
    t_elev1 = np.rad2deg(np.arctan(t_peak_z/np.sqrt(np.square(t_peak_x) + np.square(t_peak_y))))
    t_azim = np.rad2deg(np.arctan(t_peak_y/t_peak_x)) # y/x
    t_zen = np.rad2deg(np.arctan(t_peak_z/t_peak_y)) # z/y
    t_mag = np.sqrt(np.square(t_peak_x) + np.square(t_peak_y) + np.square(t_peak_z))

    return t_elev, t_elev1, t_azim, t_zen, t_mag

def get_qrs_avg_features(vcg, fiducials):
    qrs_avg_x = np.mean(vcg[0,int(fiducials[2]):int(fiducials[3])])
    qrs_avg_y = np.mean(vcg[1,int(fiducials[2]):int(fiducials[3])])
    qrs_avg_z = np.mean(vcg[2,int(fiducials[2]):int(fiducials[3])])

    qrs_avg_elev = np.rad2deg(np.arctan(qrs_avg_z/qrs_avg_x)) # z/x
    qrs_avg_elev1 = np.rad2deg(np.arctan(qrs_avg_z/np.sqrt(np.square(qrs_avg_x) + np.square(qrs_avg_y))))
    qrs_avg_azim = np.rad2deg(np.arctan(qrs_avg_y/qrs_avg_x)) # y/x
    qrs_avg_zen = np.rad2deg(np.arctan(qrs_avg_z/qrs_avg_y))  # z/y
    qrs_avg_mag = np.sqrt(np.square(qrs_avg_x) + np.square(qrs_avg_y) + np.square(qrs_avg_z))

    return qrs_avg_elev, qrs_avg_elev1, qrs_avg_azim, qrs_avg_zen, qrs_avg_mag

def get_iqrs_avg_features(vcg, fiducials, fs):
    qrs_avg_x = np.mean(vcg[0,int(fiducials[2]):int(fiducials[2]) + int(0.04*fs)])
    qrs_avg_y = np.mean(vcg[1,int(fiducials[2]):int(fiducials[2]) + int(0.04*fs)])
    qrs_avg_z = np.mean(vcg[2,int(fiducials[2]):int(fiducials[2]) + int(0.04*fs)])

    qrs_avg_elev = np.rad2deg(np.arctan(qrs_avg_z/qrs_avg_x)) # z/x
    qrs_avg_elev1 = np.rad2deg(np.arctan(qrs_avg_z/np.sqrt(np.square(qrs_avg_x) + np.square(qrs_avg_y))))
    qrs_avg_azim = np.rad2deg(np.arctan(qrs_avg_y/qrs_avg_x)) # y/x
    qrs_avg_zen = np.rad2deg(np.arctan(qrs_avg_z/qrs_avg_y))  # z/y
    qrs_avg_mag = np.sqrt(np.square(qrs_avg_x) + np.square(qrs_avg_y) + np.square(qrs_avg_z))

    return qrs_avg_elev, qrs_avg_elev1, qrs_avg_azim, qrs_avg_zen, qrs_avg_mag

def get_tqrs_avg_features(vcg, fiducials, fs):
    qrs_avg_x = np.mean(vcg[0,int(fiducials[3]) - int(0.04*fs):int(fiducials[3])])
    qrs_avg_y = np.mean(vcg[1,int(fiducials[3]) - int(0.04*fs):int(fiducials[3])])
    qrs_avg_z = np.mean(vcg[2,int(fiducials[3]) - int(0.04*fs):int(fiducials[3])])

    qrs_avg_elev = np.rad2deg(np.arctan(qrs_avg_z/qrs_avg_x)) # z/x
    qrs_avg_elev1 = np.rad2deg(np.arctan(qrs_avg_z/np.sqrt(np.square(qrs_avg_x) + np.square(qrs_avg_y))))
    qrs_avg_azim = np.rad2deg(np.arctan(qrs_avg_y/qrs_avg_x)) # y/x
    qrs_avg_zen = np.rad2deg(np.arctan(qrs_avg_z/qrs_avg_y))  # z/y
    qrs_avg_mag = np.sqrt(np.square(qrs_avg_x) + np.square(qrs_avg_y) + np.square(qrs_avg_z))

    return qrs_avg_elev, qrs_avg_elev1, qrs_avg_azim, qrs_avg_zen, qrs_avg_mag

def get_t_avg_features(vcg, fiducials):
    if np.isnan(fiducials[5]):
        return 0, 0, 0, 0, 0
    t_avg_x = np.mean(vcg[0,int(fiducials[3]):int(fiducials[5])])
    t_avg_y = np.mean(vcg[1,int(fiducials[3]):int(fiducials[5])])
    t_avg_z = np.mean(vcg[2,int(fiducials[3]):int(fiducials[5])])

    t_avg_elev = np.rad2deg(np.arctan(t_avg_z/t_avg_x)) # z/x
    t_avg_elev1 = np.rad2deg(np.arctan(t_avg_z/np.sqrt(np.square(t_avg_x) + np.square(t_avg_y))))
    t_avg_azim = np.rad2deg(np.arctan(t_avg_y/t_avg_x)) # y/x
    t_avg_zen = np.rad2deg(np.arctan(t_avg_z/t_avg_y)) # z/y
    t_avg_mag = np.sqrt(np.square(t_avg_x) + np.square(t_avg_y) + np.square(t_avg_z))

    return t_avg_elev, t_avg_elev1, t_avg_azim, t_avg_zen, t_avg_mag

def get_svg_peak_features(vcg, rpeak, tpeak):
    if np.isnan(tpeak):
        return 0, 0, 0, 0, 0
    qrs_peak_x = vcg[0,rpeak]
    qrs_peak_y = vcg[1,rpeak]
    qrs_peak_z = vcg[2,rpeak]
    t_peak_x = vcg[0,tpeak]
    t_peak_y = vcg[1,tpeak]
    t_peak_z = vcg[2,tpeak]

    svg_x = qrs_peak_x + t_peak_x
    svg_y = qrs_peak_y + t_peak_y
    svg_z = qrs_peak_z + t_peak_z

    svg_elev = np.rad2deg(np.arctan(svg_z/svg_x)) # z/x
    svg_elev1 = np.rad2deg(np.arctan(svg_z/np.sqrt(np.square(svg_x) + np.square(svg_y))))
    svg_azim = np.rad2deg(np.arctan(svg_y/svg_x)) # y/x
    svg_zen = np.rad2deg(np.arctan(svg_z/svg_y)) # z/y
    svg_mag = np.sqrt(np.square(svg_x) + np.square(svg_y) + np.square(svg_z))

    return svg_elev, svg_elev1, svg_azim, svg_zen, svg_mag

def get_svg_avg_features(vcg, fiducials):
    if np.isnan(fiducials[5]):
        return 0, 0, 0, 0, 0
    svg_avg_x = np.mean(vcg[0,int(fiducials[2]):int(fiducials[5])])
    svg_avg_y = np.mean(vcg[1,int(fiducials[2]):int(fiducials[5])])
    svg_avg_z = np.mean(vcg[2,int(fiducials[2]):int(fiducials[5])])

    svg_avg_elev = np.rad2deg(np.arctan(svg_avg_z/svg_avg_x)) # z/x
    svg_avg_elev1 = np.rad2deg(np.arctan(svg_avg_z/np.sqrt(np.square(svg_avg_x) + np.square(svg_avg_y))))
    svg_avg_azim = np.rad2deg(np.arctan(svg_avg_y/svg_avg_x)) # y/x
    svg_avg_zen = np.rad2deg(np.arctan(svg_avg_z/svg_avg_y)) # z/y
    svg_avg_mag = np.sqrt(np.square(svg_avg_x) + np.square(svg_avg_y) + np.square(svg_avg_z))

    return svg_avg_elev, svg_avg_elev1, svg_avg_azim, svg_avg_zen, svg_avg_mag

def get_qrst_angle(vcg, rpeak, tpeak):
    if np.isnan(tpeak):
        return 0
    qrs_peak_x = vcg[0,rpeak]
    qrs_peak_y = vcg[1,rpeak]
    qrs_peak_z = vcg[2,rpeak]
    qrs_mag = np.sqrt(np.square(qrs_peak_x) + np.square(qrs_peak_y) + np.square(qrs_peak_z))

    t_peak_x = vcg[0,tpeak]
    t_peak_y = vcg[1,tpeak]
    t_peak_z = vcg[2,tpeak]
    t_mag = np.sqrt(np.square(t_peak_x) + np.square(t_peak_y) + np.square(t_peak_z))

    qrs_peak_norm = [qrs_peak_x/qrs_mag, qrs_peak_y/qrs_mag, qrs_peak_z/qrs_mag]
    t_peak_norm = [t_peak_x/t_mag, t_peak_y/t_mag, t_peak_z/t_mag]
    qrst_angle = np.rad2deg(np.arccos(np.dot(qrs_peak_norm, t_peak_norm)))

    return qrst_angle

def get_qrst_avg_angle(vcg, fiducials):
    if np.isnan(fiducials[5]):
        return 0

    qrs_avg_x = np.mean(vcg[0,int(fiducials[2]):int(fiducials[3])])
    qrs_avg_y = np.mean(vcg[1,int(fiducials[2]):int(fiducials[3])])
    qrs_avg_z = np.mean(vcg[2,int(fiducials[2]):int(fiducials[3])])
    qrs_avg_mag = np.sqrt(np.square(qrs_avg_x) + np.square(qrs_avg_y) + np.square(qrs_avg_z))

    t_avg_x = np.mean(vcg[0,int(fiducials[3]):int(fiducials[5])])
    t_avg_y = np.mean(vcg[1,int(fiducials[3]):int(fiducials[5])])
    t_avg_z = np.mean(vcg[2,int(fiducials[3]):int(fiducials[5])])
    t_avg_mag = np.sqrt(np.square(t_avg_x) + np.square(t_avg_y) + np.square(t_avg_z))

    qrs_avg_norm = [qrs_avg_x/qrs_avg_mag, qrs_avg_y/qrs_avg_mag, qrs_avg_z/qrs_avg_mag]
    t_avg_norm = [t_avg_x/t_avg_mag, t_avg_y/t_avg_mag, t_avg_z/t_avg_mag]
    qrst_avg_angle = np.rad2deg(np.arccos(np.dot(qrs_avg_norm, t_avg_norm)))

    return qrst_avg_angle

# Other Features #######################################################################################################
def get_KPD(fiducials, ecg, ecg_rms):
    if ~np.isnan(fiducials[4]) & ~np.isnan(fiducials[5]):
        tpeak = np.argmax(ecg_rms[int(fiducials[4]):int(fiducials[5])]) + int(fiducials[4])
        rpeak = np.argmax(ecg_rms[int(fiducials[2]):int(fiducials[3])]) + int(fiducials[2])
        return np.min(np.max(np.abs(ecg[:, rpeak:tpeak]), axis=0))
    else:
        return np.nan

def get_TWAC(fiducials, segment, fs, p=0.016): # segment is segments x leads x datapoints; p=0.016 seconds
    
    delta = int(p*fs)
    s = segment[:, :, int(fiducials[4]):int(fiducials[5])]
    n = np.size(s, 2)
    for i in range(n):
        #s[:, :, i] = s[:, :, i] - (np.sum(s[:, :, max(0, i-delta):min(i+delta+1, n)], axis=-1) / (2*delta+1))
        s[:, :, i] = s[:, :, i] - (np.sum(s[:, :, max(0, i-delta):min(i+delta+1, n)], axis=-1) / (min(i+delta+1, n) - max(0, i-delta)))
    return np.sum(s, axis=-1) # input: segments x leads x datapoints, output: segments x leads

def get_Laplacian_Eigenmaps(fiducials, segment, K=1000): # segment is segments x leads x datapoints

    # QUESTIONS    
    # Use global representative beat?
    # Overall amplitude = max amplitude?
    # Earliest response to ischemia is indicated by the time instant at which it passes 1/3 of the overall amplitude of the run-metric (time instant?, overall amplitude of the run-metric=max of maxes?)
    # We computed along with Ar, the maximum value of the metric for that episode, Am over all representative beats in the episode; Am global max across run-metrics/curves?

    # Preprocessing
    # P = [p1, ..., p1000], P represents one beat
    P = signal.resample(np.squeeze(segment[0, :, fiducials[2]:fiducials[5]]), K, axis=1).T    
    # Compute R (pairwise distances between all time points P): Rij = norm_2(pi - pj); set sigma = the largest element in R
    R = squareform(pdist(P, 'euclidean')) # input P: K x leads, output K x K distance matrix
    # Compute W: Wij = exp(-Rij^2/sigma^2)
    sigma_square = np.max(R)**2
    W = np.exp(-np.square(R)/sigma_square)
    # Compute a diagonal degree matrix D = diag(sum over i of Wi:); Wi: is the ith row of W
    D = np.diag(np.sum(W, axis=0))
    # Solve for the singular value decomposition of (D^-1)W = USV; row k of V contains the LE coordinates of the point pk
    D_inv = np.linalg.inv(D)
    U, S, Vh = np.linalg.svd(np.matmul(D_inv, W), full_matrices=False)
    S = np.diag(S)
    # Discard the coordinate associated with the largest eigenvalue, which represents the mean of the dataset; retain the next three dimensions
    LE_P = np.transpose(Vh)[:, 1:4] # V = Vh.H; LE_P: K x 3
    
    # New data to map into LE space
    Pnew = np.transpose(signal.resample(segment[1:, :, fiducials[2]:fiducials[5]], K, axis=-1), (0, 2, 1))
    n = np.size(Pnew, 0)
    LE_Pnew = np.zeros((n, np.size(Pnew, 1), 3)) # LE_Pnew: (segments-1) x K x 3
    for i in range(n):
        G = np.exp(-np.square(cdist(P, np.squeeze(Pnew[i, :, :]), metric='euclidean'))/sigma_square)
        Vhnew = np.matmul(np.linalg.pinv(S, hermitian=True), np.matmul(np.transpose(U), np.matmul(D_inv, G)))
        LE_Pnew[i, :, :] = np.transpose(Vhnew)[:, 1:4]
        
    # Create a set of run-metrics (1000 curves): K x (segments-1)
    rm = np.zeros((K, n))
    for i in range(n):
        for j in range(K):
            rm[j, i] = euclidean(np.squeeze(LE_P[j, :]), np.squeeze(LE_Pnew[i, j, :]))
            
    # Normalize the run-metrics
    rm = (np.max(rm, axis=1).reshape((-1, 1)) - rm) / np.max(rm)
    
    # Select the 20 curves (from 1000) with the largest overall amplitude
    rm_m = np.squeeze(np.max(rm, axis=1) - np.min(rm, axis=1))/2 # overall amplitude for each curve 
    indx = np.argpartition(rm_m, -20)[-20:]
    rm_m = rm_m[indx]
    mask = rm[indx, :] > (1/3)*rm_m.reshape((-1, 1))
    
    indx_sorted_m = np.argsort(rm_m) # ascending order
    rm_m = rm_m[indx_sorted_m]
    mask = mask[indx_sorted_m, :]
    indx = indx[indx_sorted_m]
    
    t1 = float('inf')
    for i in range(20):
        t2 = np.nonzero(np.squeeze(mask[i, :]))[0][0]
        if t2 <= t1:
            t1 = t2
            indx_f = indx[i]
            
    return rm[indx_f, :] #, LE_P, LE_Pnew

def get_AldrichSTscore(st_amps_local):
    leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    # QUESTIONS
    # Does the leads order above correspond to that of st_amps_local?
    
    st_amps_local = np.delete(st_amps_local, leads.index('aVR'))
    leads.remove('aVR')

    st_elevation_local = [el if el>0 else 0 for el in st_amps_local] #uV
    Aldrich_a = 3*(1.5*(sum(st_amps_local>=100)) - 0.4) #uV
    Aldrich_i = 3*(0.6*0.001*(st_elevation_local[leads.index('II')] + st_elevation_local[leads.index('III')] + st_elevation_local[leads.index('aVF')]) + 2)
        
    return Aldrich_a, Aldrich_i

def get_SelvesterQRSscore(fiducials_local, ecg, fs):
    leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    # QUESTIONS
    # Any limits for Selvester score (for sanity check)?
    
    rdurs = (fiducials_local['r_offsets'] - fiducials_local['r_onsets']) / fs
    qdurs = (fiducials_local['q_offsets'] - fiducials_local['q_onsets']) / fs
    
    r_q = np.zeros(12)
    r_s = np.zeros(12)
    for l, (i_r, i_q, i_s) in enumerate(zip(fiducials_local['r_peaks'], fiducials_local['q_peaks'], fiducials_local['s_peaks'])):
        if ecg[l, i_q] == 0:
            r_q[l] = np.inf
        else:
            r_q[l] = np.divide(ecg[l, i_r], ecg[l, i_q])
        if ecg[l, i_s] == 0:
            r_s[l] = np.inf
        else:    
            r_s[l] = np.divide(ecg[l, i_r], ecg[l, i_s])
    
    score = 0
    for el in leads:
        j = leads.index(el)
        if el == 'I':
            if qdurs[j]>=0.03:
                score += 1
            if r_q[j]<=1:
                score += 1
        if el == 'II':
            if qdurs[j]>=0.04:
                score += 2
            elif qdurs[j]>=0.03:
                score += 1
        if el == 'aVL':
            if qdurs[j]>=0.03:
                score += 1
            if r_q[j]<=1:
                score += 1
        if el == 'aVF':
            if qdurs[j]>=0.05:
                score += 3
            elif qdurs[j]>=0.04:
                score += 2
            elif qdurs[j]>=0.03:
                score += 1
            if r_q[j]<=1:
                score += 2
            elif r_q[j]<=2:
                score += 1
        if el == 'V1':
            if qdurs[j]>0:
                score += 1
            if rdurs[j]>=0.05:
                score += 2
            elif rdurs[j]>=0.04: 
                score += 1
            if r_s[j]>=1:
                score += 1
        if el == 'V2':
            if (qdurs[j]>0) or (rdurs[j]<=0.01):
                score += 1
            if rdurs[j]>=0.06:
                score += 2
            elif rdurs[j]>=0.05:
                score += 1
            if r_s[j]>=1.5:
                score += 1
        if el == 'V3':
            if (qdurs[j]>0) or (rdurs[j]<=0.02):
                score += 1
        if el == 'V4':
            if qdurs[j]>=0.02:
                score += 1
            if (r_q[j]<=0.5) or (r_s[j]<=0.5):
                score += 2
            elif (r_q[j]<=1) or (r_s[j]<=1):
                score += 1
        if el == 'V5':
            if qdurs[j]>=0.03:
                score += 1
            if (r_q[j]<=1) or (r_s[j]<=1):
                score += 2
            elif (r_q[j]<=2) or (r_s[j]<=2):
                score += 1
        if el == 'V6':
            if qdurs[j]>=0.03:
                score += 1
            if (r_q[j]<=1) or (r_s[j]<=1):
                score += 2
            elif (r_q[j]<=3) or (r_s[j]<=3):
                score += 1
        
    return score

def get_vpace(rpeaks, pace_spikes, fs):
    if len(pace_spikes) == 0:
        return False, False
    delays = np.zeros(len(rpeaks))
    for i in range(len(rpeaks)):
        spike = pace_spikes[pace_spikes < rpeaks[i]]
        if len(spike) > 0:
            delays[i] = rpeaks[i] - spike[-1]
    delay = np.median(delays)
    delay = delay / fs
    if delay > 0.12:
        return True, False
    else:
        return True, True
    
def get_bbb(fiducials, fiducials_local, fs, ecg, t_qrs_ratios_local):
    rbbb = False
    lbbb = False
    qrsd = (fiducials[3] - fiducials[2]) / fs
    r = fiducials_local['r_peaks'][6]
    s = fiducials_local['s_peaks'][6]
    rp = fiducials_local['rp_peaks'][6]
    q = fiducials_local['q_peaks'][6]
    discordance = t_qrs_ratios_local[6] < 0

    if r > 0:
        r_amp = ecg[6, r]
    else:
        r_amp = 0
    if s > 0:
        s_amp = np.abs(ecg[6, s])
    else:
        s_amp = 0
    if rp > 0:
        rp_amp = ecg[6, rp]
    else:
        rp_amp = 0

    if qrsd > 0.12:
        if ((r_amp > s_amp) | (rp_amp > s_amp)):
            rbbb = True
        else:
            lbbb = True
    return rbbb, lbbb

def get_vtach(hr, qrsd):
    if (hr > 100) & (qrsd > 0.12):
        return True
    return False

def get_tmd(ecg, fiducials):
    ecg = ecg[[0,1,6,7,8,9,10,11],:]

    if np.std(ecg) == 0:
        return 0,0,0

    U, Sigma, VT = np.linalg.svd(ecg)
    Sigma_matrix = np.zeros((U.shape[1], VT.shape[0]))
    Sigma_matrix[:min(U.shape[1], VT.shape[0]), :min(U.shape[1], VT.shape[0])] = np.diag(Sigma)
    M = U @ Sigma_matrix @ VT
    U_red = U[:,0:3]
    S = U_red.T @ M

    E3D = np.linalg.norm(S, axis=0)
    tte = fiducials[5]
    rpeak = np.argmax(E3D[280:320])+280

    if len(np.where(E3D[0:rpeak] < 0.7*E3D[rpeak])[0]) == 0:
        return 0, 0, 0
    trs = np.where(E3D[0:rpeak] < 0.7*E3D[rpeak])[0][-1] - 24

    if len(np.where(E3D[rpeak:] < 0.7*E3D[rpeak])[0]) == 0:
        tre = 600
    else:
        tre = np.where(E3D[rpeak:] < 0.7*E3D[rpeak])[0][0] + rpeak + 24
    if tre > 450 or tre >= tte:
        return 0, 0, 0
    tp = np.argmax(E3D[tre:tte])+tre
    step = int((tp - tre + 24)/3)
    tts = tp - step
    maxval = np.max(E3D)
    S = S/maxval
    dc = 0.25*(S[:,trs]+S[:,tre]+ S[:,tts] + S[:,tte-1])
    S = (S.T - dc).T

    tmds = []
    segs = ['pre', 'post', 'all']

    for seg in segs:
        if seg == 'pre':
            S_tseg = S[:,tts:tp]
        elif seg == 'post':
            S_tseg = S[:,tp:tte]
        elif seg == 'all':
            S_tseg = S[:,tts:tte]

        M_tseg = U_red @ S_tseg
        U_tseg, Sigma_tseg, VT_tseg = np.linalg.svd(M_tseg)
        U_tseg_red = U_tseg[:,0:2]
        W = U_tseg_red * Sigma_tseg[0:2]
        W = W.T

        W = np.delete(W, 2, axis=1)
        nums = np.arange(0, W.shape[1])
        pairs = list(itertools.combinations(nums, 2))

        tmd = 0
        
        for pair in pairs:
            wi = W[:,pair[0]]
            wj = W[:,pair[1]]
            if np.linalg.norm(wi)*np.linalg.norm(wj) != 0:
                ratio =  np.dot(wi, wj)/(np.linalg.norm(wi)*np.linalg.norm(wj))
                ratio = np.clip(ratio, -1, 1)
                tmd += np.arccos(ratio) * 180/np.pi

        tmd /= len(pairs)
        tmds.append(tmd)
    return tmds

def get_tcrt(ecg, fiducials):
    ecg = ecg[[0,1,6,7,8,9,10,11],:]

    if np.std(ecg) == 0:
        return 0

    U, Sigma, VT = np.linalg.svd(ecg)
    Sigma_matrix = np.zeros((U.shape[1], VT.shape[0]))
    Sigma_matrix[:min(U.shape[1], VT.shape[0]), :min(U.shape[1], VT.shape[0])] = np.diag(Sigma)
    M = U @ Sigma_matrix @ VT
    U_red = U[:,0:3]
    S = U_red.T @ M

    E3D = np.linalg.norm(S, axis=0)
    tte = fiducials[5]
    rpeak = np.argmax(E3D[280:320])+280

    if len(np.where(E3D[0:rpeak] < 0.7*E3D[rpeak])[0]) == 0:
        return 0
    trs = np.where(E3D[0:rpeak] < 0.7*E3D[rpeak])[0][-1] - 24

    if len(np.where(E3D[rpeak:] < 0.7*E3D[rpeak])[0]) == 0:
        tre = 600
    else:
        tre = np.where(E3D[rpeak:] < 0.7*E3D[rpeak])[0][0] + rpeak + 24
    if tre > 450 or tre >= tte:
        return 0
    tp = np.argmax(E3D[tre:tte])+tre
    step = int((tp - tre + 24)/3)
    tts = tp - step
    maxval = np.max(E3D)
    S = S/maxval
    dc = 0.25*(S[:,trs]+S[:,tre]+ S[:,tts] + S[:,tte-1])
    S = (S.T - dc).T

    S_tseg = S[:,tts:tte]

    norms = np.linalg.norm(S_tseg, axis=1)
    max1idx = np.argmax(S_tseg[np.argmax(norms),:])
    et1 = S_tseg[:,max1idx] / np.max(norms)

    S_qrs = S[:,trs:tre]

    TCRT = 0
    for j in range(S_qrs.shape[1]):
        a = S_qrs[:,j]
        b = et1
        TCRT += np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))

    TCRT /= S_qrs.shape[1]

    return TCRT

def get_morphology_scores(vcg, fiducials):
    vcg = vcg[:,fiducials[3]:]
    b, a = signal.butter(2, 20, 'low', fs=500)
    # get first 3 prinicpal components of the vcg using SVD
    U, _, _ = np.linalg.svd(vcg)
    princ_cmpnts = np.matmul(U[:,0:3].T, vcg)

    # mirror beginning and end of principal components then filter
    mirror_len = len(princ_cmpnts[0,:])
    princ_cmpnts = np.concatenate((princ_cmpnts[:,::-1], princ_cmpnts, princ_cmpnts[:,::-1]), axis=1)
    princ_cmpnts = signal.filtfilt(b, a, princ_cmpnts, axis=1)
    princ_cmpnts = princ_cmpnts[:, mirror_len:-mirror_len]

    sig1 = princ_cmpnts[0,:]
    onset = fiducials[4] - fiducials[3]
    offset = fiducials[5] - fiducials[3]
    max_idx = np.argmax(np.abs(sig1[onset:offset]))+onset
    if sig1[max_idx] < 0:
        sig1 = sig1*-1

    
    tpeak = np.argmax(np.abs(sig1[onset:offset]))+onset
    diff1 = np.diff(sig1)

    ascend_sig = diff1[:tpeak]

    ascend_len = tpeak
    descend_len = offset-tpeak

    sig_len = np.max([ascend_len, descend_len])
    ascend_sig = np.zeros(sig_len)
    descend_sig = np.zeros(sig_len)
    ascend_sig[-tpeak:] = diff1[:tpeak]
    descend_sig[0:offset-tpeak] = diff1[tpeak-1:offset-1]*-1
    descend_sig = descend_sig[::-1]

    #normalize signals between 0 and 1
    ascend_sig = (ascend_sig - np.min(ascend_sig))/(np.max(ascend_sig)-np.min(ascend_sig))
    descend_sig = (descend_sig - np.min(descend_sig))/(np.max(descend_sig)-np.min(descend_sig))
    asymm_score = np.sum((ascend_sig - descend_sig)**2) / len(ascend_sig)

    # notch score
    sig1 = princ_cmpnts[0,onset:offset]
    max_idx = np.argmax(np.abs(sig1))
    if sig1[max_idx] < 0:
        sig1 = sig1*-1

    diff1 = np.diff(sig1)
    diff2 = np.diff(diff1)
    den = (1+diff1**2)**(3/2)
    isroc = diff2/den[0:-1]
    isroc /= np.max(np.abs(isroc))

    peaks = signal.find_peaks(isroc, height=0.1)[0]
    peaks1 = signal.find_peaks(-isroc, height=0.1)[0]

    notch_score = 0
    if len(peaks) > 0 and len(peaks1) > 0:
        neg_peak = peaks1[-1]
        pos_peak = peaks[peaks < neg_peak]
        if (len(pos_peak) > 0):
            pos_peak = pos_peak[-1]
            notch_score = isroc[pos_peak]

    # flatness score
    sig1 = princ_cmpnts[0,onset:offset]
    max_idx = np.argmax(np.abs(sig1))
    if sig1[max_idx] < 0:
        sig1 = sig1*-1
    sig1 = (sig1 - np.min(sig1))/(np.max(sig1)-np.min(sig1))

    area_sig1 = np.trapz(sig1)
    sig1 /= area_sig1
    m1 = 0
    for i in range(len(sig1)):
        m1 += sig1[i]*(i)
    
    m2 = 0
    for i in range(len(sig1)):
        m2 += (((i-m1)**2) * sig1[i])
    m2 = np.sqrt(m2)

    m4 = 0
    for i in range(len(sig1)):
        m4 += (((i-m1)**4) * sig1[i])
    m4 = m4**0.25
    m4 /= m2**2

    flatness_score = 1-0.1*m4

    mcs = asymm_score + 1.9 * notch_score + 1.6 * flatness_score

    return asymm_score, notch_score, flatness_score, mcs

def getFeatures(data):

    leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    ecg_10sec = data.waveforms['ecg_10sec_clean']
    ecg_median = data.waveforms['ecg_median']
    fs = data.fs
    ecg_beats = data.waveforms['beats']
    rrint = data.features['rrint']
    pace_spikes = data.fiducials['pacing_spikes']
    rpeaks = data.fiducials['rpeaks']
    fiducials = data.fiducials['global']
    fiducials_local = data.fiducials['local']
    age = data.demographics['age']
    sex = data.demographics['sex']

    for j in range(12):
        ecg_median[j] -= np.median(ecg_median[j,fiducials[2]-int(0.02*fs):fiducials[2]])

    ecg_rms = np.sqrt(np.sum(ecg_median**2,axis=0)) / np.sqrt(12)
    tpeak = np.argmax(ecg_rms[int(fiducials[4]):int(fiducials[5])]) + int(fiducials[4])
    vcg = get_vcg(ecg_median)
    rpeak = np.argmax(ecg_rms[int(fiducials[2]):int(fiducials[3])]) + int(fiducials[2])

    # Local Features #######################################################################################################
    p_area_local = get_p_area_local(fiducials, fiducials_local, ecg_median, fs)
    for j in range(12):
        data.features['p_area_{}'.format(leads[j])] = p_area_local[j]

    qrs_area_local = get_qrs_area_local(fiducials_local, ecg_median, fs)
    for j in range(12):
        data.features['qrs_area_{}'.format(leads[j])] = qrs_area_local[j]

    t_area_local = get_t_area_local(fiducials, fiducials_local, ecg_median, fs)
    for j in range(12):
        data.features['t_area_{}'.format(leads[j])] = t_area_local[j]

    st_peak_area_local = get_st_peak_area_local(fiducials, fiducials_local, ecg_median, fs)
    for j in range(12):
        data.features['st_peak_area_{}'.format(leads[j])] = st_peak_area_local[j]

    p_amps_local = get_p_amps_local(fiducials, ecg_median)
    for j in range(12):
        data.features['p_amp_{}'.format(leads[j])] = p_amps_local[j]

    qrs_amps_local = get_qrs_amps_local(fiducials_local, ecg_median)
    for j in range(12):
        data.features['qrs_amp_{}'.format(leads[j])] = qrs_amps_local[j]

    t_amps_local = get_t_amps_local(fiducials, ecg_median)
    for j in range(12):
        data.features['t_amp_{}'.format(leads[j])] = t_amps_local[j]

    st_amps_local = get_st_amp_local(fiducials_local, ecg_median, fs)
    for j in range(12):
        data.features['st_amp_{}'.format(leads[j])] = st_amps_local[j]

    t_qrs_ratios_local = get_t_qrs_ratios_local(fiducials, fiducials_local, ecg_median)
    for j in range(12):
        data.features['t_qrs_ratio_{}'.format(leads[j])] = t_qrs_ratios_local[j]

    st_slope_local = get_st_slope(fiducials_local, ecg_median, fs)
    for j in range(12):
        data.features['st_slope_{}'.format(leads[j])] = st_slope_local[j]

    qrs_slope_local = get_qrs_slope(fiducials_local, ecg_median, fs)
    for j in range(12):
        data.features['qrs_slope_{}'.format(leads[j])] = qrs_slope_local[j]

    b2b_std = get_beat_to_beat_std(ecg_beats)
    for j in range(12):
        data.features['b2b_std_{}'.format(leads[j])] = b2b_std[j]

    qrs_beats = ecg_beats.copy()
    for j in range(12):
        qrs_beats[j] = qrs_beats[j][:,int(fiducials[2]):int(fiducials[3])]
    b2b_std_qrs = get_beat_to_beat_std(qrs_beats)
    for j in range(12):
        data.features['b2b_std_qrs_{}'.format(leads[j])] = b2b_std_qrs[j]

    stt_beats = ecg_beats.copy()
    for j in range(12):
        stt_beats[j] = stt_beats[j][:,int(fiducials[3]):int(fiducials[5])]
    b2b_std_stt = get_beat_to_beat_std(stt_beats)
    for j in range(12):
        data.features['b2b_std_stt_{}'.format(leads[j])] = b2b_std_stt[j]

    q_amps_local = get_q_amps_local(fiducials_local, ecg_median)
    for j in range(12):
        data.features['q_amp_{}'.format(leads[j])] = q_amps_local[j]

    q_duration_local = get_q_durations_local(fiducials_local, fs)
    for j in range(12):
        data.features['q_duration_{}'.format(leads[j])] = q_duration_local[j]

    q_area_local = get_q_areas_local(fiducials_local, ecg_median, fs)
    for j in range(12):
        data.features['q_area_{}'.format(leads[j])] = q_area_local[j]

    r_amps_local = get_r_amps_local(fiducials_local, ecg_median)
    for j in range(12):
        data.features['r_amp_{}'.format(leads[j])] = r_amps_local[j]

    r_duration_local = get_r_durations_local(fiducials_local, fs)
    for j in range(12):
        data.features['r_duration_{}'.format(leads[j])] = r_duration_local[j]

    r_area_local = get_r_areas_local(fiducials_local, ecg_median, fs)
    for j in range(12):
        data.features['r_area_{}'.format(leads[j])] = r_area_local[j]

    s_amps_local = get_s_amps_local(fiducials_local, ecg_median)
    for j in range(12):
        data.features['s_amp_{}'.format(leads[j])] = s_amps_local[j]

    s_duration_local = get_s_durations_local(fiducials_local, fs)
    for j in range(12):
        data.features['s_duration_{}'.format(leads[j])] = s_duration_local[j]

    s_area_local = get_s_areas_local(fiducials_local, ecg_median, fs)
    for j in range(12):
        data.features['s_area_{}'.format(leads[j])] = s_area_local[j]

    rp, sp = get_rpsp(fiducials_local)
    for j in range(12):
        data.features['rp_{}'.format(leads[j])] = rp[j]
        data.features['sp_{}'.format(leads[j])] = sp[j]

    vat_local = get_vat_local(fiducials_local, fs, ecg_median)
    for j in range(12):
        data.features['vat_{}'.format(leads[j])] = vat_local[j]

    # Global Features ######################################################################################################
    data.features['p_area'] = get_p_area_global(fiducials, ecg_rms, fs)
    data.features['qrs_area'] = get_qrs_area_global(fiducials, ecg_rms, fs)
    data.features['t_area'] = get_t_area_global(fiducials, ecg_rms, fs)
    data.features['p_duration'] = get_p_duration(fiducials, fs)
    data.features['qrs_duration'] = get_qrs_duration(fiducials, fs)
    data.features['t_duration'] = get_t_duration(fiducials, fs)
    data.features['pr_interval'] = get_pr_interval(fiducials, fs)
    data.features['qt_interval'] = get_qt_interval(fiducials, fs)
    data.features['tpte'] = get_tpte_global(fiducials, ecg_rms, fs)
    data.features['st_peak_area'] = get_st_peak_area_global(fiducials, ecg_rms, fs)
    data.features['st_amp'] = get_st_amp_global(fiducials, ecg_rms, fs)
    data.features['p_amp'] = get_p_amp_global(fiducials, ecg_rms)
    data.features['qrs_amp'] = get_qrs_amp_global(fiducials, ecg_rms)
    data.features['t_amp'] = get_t_amp_global(fiducials, ecg_rms)
    data.features['t_qrs_ratio'] = get_t_qrs_ratio_global(fiducials, ecg_rms)
    data.features['hr'] = get_hr_global(rrint, fs)
    data.features['qtc_bazett'] = get_qtc_interval(fiducials, rrint, fs)
    data.features['qtc_fridericia'] = get_qtc_interval2(fiducials, rrint, fs)
    data.features['jt'] = get_jt_global(fiducials, fs)
    data.features['jtc_bazett'] = get_jtc_global(fiducials, rrint, fs)
    data.features['jtc_fridericia'] = get_jtc_global2(fiducials, rrint, fs)
    data.features['tpte_qt_ratio'] = get_tpte_qt_ratio(fiducials, ecg_rms, fs)
    data.features['rms_min'] = get_rms_min(ecg_rms)
    data.features['rms_std'] = get_rms_std(ecg_rms)
    data.features['rms_mean'] = get_rms_mean(ecg_rms)
    data.features['rms_median'] = get_rms_median(ecg_rms)
    data.features['vat'] = get_vat_global(ecg_rms, fiducials, fs)
    data.features['pca'] = get_pca(ecg_median)
    data.features['pca_qrs'] = get_pca(ecg_median[:,int(fiducials[2]):int(fiducials[3])])
    data.features['pca_stt'] = get_pca(ecg_median[:,int(fiducials[3]):int(fiducials[5])])
    data.features['pca_t'] = get_pca(ecg_median[:,int(fiducials[4]):int(fiducials[5])])
    data.features['pca_st'] = get_pca(ecg_median[:,int(fiducials[3]):int(fiducials[3])+int(0.08*fs)])
    data.features['pca_tpte'] = get_pca(ecg_median[:,tpeak:int(fiducials[5])])
    data.features['ndpv'] = get_ndpv(ecg_median)
    data.features['ndpv_qrs'] = get_ndpv(ecg_median[:,int(fiducials[2]):int(fiducials[3])])
    data.features['ndpv_stt'] = get_ndpv(ecg_median[:,int(fiducials[3]):int(fiducials[5])])
    data.features['psd_low'], data.features['psd_medium'], data.features['psd_high'] = get_rel_psds(ecg_10sec, fs)
    data.features['lat_conc'], data.features['ant_conc'], data.features['inf_conc'] = get_concavity(ecg_median, fiducials, fiducials_local)

    # VCG Features ########################################################################################################
    data.features['qrs_elev'], data.features['qrs_elev1'], data.features['qrs_azim'], data.features['qrs_zen'], data.features['qrs_mag'] = get_qrs_peak_features(vcg, rpeak)
    data.features['t_elev'], data.features['t_elev1'], data.features['t_azim'], data.features['t_zen'], data.features['t_mag'] = get_t_peak_features(vcg, tpeak)
    data.features['qrs_avg_elev'], data.features['qrs_avg_elev1'], data.features['qrs_avg_azim'], data.features['qrs_avg_zen'], data.features['qrs_avg_mag'] = get_qrs_avg_features(vcg, fiducials)
    data.features['iqrs_avg_elev'], data.features['iqrs_avg_elev1'], data.features['iqrs_avg_azim'], data.features['iqrs_avg_zen'], data.features['iqrs_avg_mag'] = get_iqrs_avg_features(vcg, fiducials, fs)
    data.features['tqrs_avg_elev'], data.features['tqrs_avg_elev1'], data.features['tqrs_avg_azim'], data.features['tqrs_avg_zen'], data.features['tqrs_avg_mag'] = get_tqrs_avg_features(vcg, fiducials, fs)
    data.features['t_avg_elev'], data.features['t_avg_elev1'], data.features['t_avg_azim'], data.features['t_avg_zen'], data.features['t_avg_mag'] = get_t_avg_features(vcg, fiducials)
    data.features['svg_elev'], data.features['svg_elev1'], data.features['svg_azim'], data.features['svg_zen'], data.features['svg_mag'] = get_svg_peak_features(vcg, rpeak, tpeak)
    data.features['svg_avg_elev'], data.features['svg_avg_elev1'], data.features['svg_avg_azim'], data.features['svg_avg_zen'], data.features['svg_avg_mag'] = get_svg_avg_features(vcg, fiducials)
    data.features['qrst_angle'] = get_qrst_angle(vcg, rpeak, tpeak)
    data.features['qrst_avg_angle'] = get_qrst_avg_angle(vcg, fiducials)

    # Other Features
    data.features['kpd'] = get_KPD(fiducials, ecg_median, ecg_rms)
    data.features['aldrich_a'], data.features['aldrich_i'] = get_AldrichSTscore(st_amps_local)
    data.features['selvester_score'] = get_SelvesterQRSscore(fiducials_local, ecg_median, fs)
    data.features['pace'], data.features['vpace'] = get_vpace(rpeaks, pace_spikes, fs)
    data.features['rbbb'], data.features['lbbb'] = get_bbb(fiducials, fiducials_local, fs, ecg_median, t_qrs_ratios_local)
    data.features['vtach'] = get_vtach(data.features['hr'], data.features['qrs_duration'])
    data.features['rrint'] = rrint / fs
    rpeak_diffs = np.diff(rpeaks) / fs
    if len(rpeak_diffs) < 2:
        rrint_std = 0
    else:
        rrint_std = np.std(rpeak_diffs)
    data.features['rrint_std'] = rrint_std    
    data.features['tmd_pre'], data.features['tmd_post'], data.features['tmd'] = get_tmd(ecg_median, fiducials)
    data.features['tcrt'] = get_tcrt(ecg_median, fiducials)
    data.features['asymm_score'], data.features['notch_score'], data.features['flatness_score'], data.features['mcs'] = get_morphology_scores(vcg, fiducials)
    data.features['age'] = age
    data.features['sex'] = sex

    return
################################################################################

def process_ecgs(save_features, save_predictions):
    filenames = glob(folder + '/*.json') + glob(folder + '/*.xml')
    rf_148, rf_417, baseline_filt, lowpass100_filt, pli50_filt, pli60_filt, lowpass150_filt, pli100_filt, pli120_filt = load_models()
    selected_features = np.load('models/selected_features.npy')

    for filename in filenames:
        text1.delete('1.0', tk.END)
        text1.insert('end', 'Processing: ' + filename.split('\\')[-1])
        root.update()

        try:
            ecg = ECG(filename)
            ecg.processRawData(b_baseline=baseline_filt,b_low100=lowpass100_filt,b_pli50=pli50_filt,b_pli60=pli60_filt,b_low150=lowpass150_filt,b_pli100=pli100_filt,b_pli120=pli120_filt)
            ecg.processMedian()
            ecg.segmentMedian()
            ecg.processFeatures()

            features = []
            for feature in ecg.features:
                features.append(ecg.features[feature])
            features = np.array(features).reshape(1, -1)
            features_148 = features[:,selected_features == 1]

            if save_features:
                with open('results/' + filename.split('\\')[-1].split('.')[0] + '_features.txt', 'w+') as f:
                    f.write(f"{features}\n")
            f.close()

            if save_predictions:
                y_pred_148 = rf_148.predict_proba(features_148)[0,1]
                y_pred_417 = rf_417.predict_proba(features)[0,1]
                with open('results/' + filename.split('\\')[-1].split('.')[0] + '_predictions.txt', 'w+') as f:
                    f.write(f"{y_pred_148}\n"
                            f"{y_pred_417}\n")
            f.close()
            
        except Exception as e:
            print(e)
            with open('error.txt', 'a+') as f:
                f.write(filename.split('\\')[-1].split('.')[0] + '\n')
                f.write(str(e) + '\n\n')
            f.close()

    text1.delete('1.0', tk.END)
    text1.insert('end', 'Processing complete.')
    root.update()

def load_models():
    key = b'FVQOeE8kcHVNve0gGymW3cCfg7uHP2WGly6ualcDspA='
    cipher_suite = Fernet(key)

    with open('models/rf_148.enc', 'rb') as encrypted_file:
        rf_148 = encrypted_file.read()
    rf_148 = cipher_suite.decrypt(rf_148)
    rf_148 = joblib.load(BytesIO(rf_148))

    with open('models/rf_417.enc', 'rb') as encrypted_file:
        rf_417 = encrypted_file.read()
    rf_417 = cipher_suite.decrypt(rf_417)
    rf_417 = joblib.load(BytesIO(rf_417))

    with open('models/baseline_filt.enc', 'rb') as encrypted_file:
        baseline_filt = encrypted_file.read()
    baseline_filt = cipher_suite.decrypt(baseline_filt)
    baseline_filt = joblib.load(BytesIO(baseline_filt))

    with open('models/lowpass100_filt.enc', 'rb') as encrypted_file:
        lowpass100_filt = encrypted_file.read()
    lowpass100_filt = cipher_suite.decrypt(lowpass100_filt)
    lowpass100_filt = joblib.load(BytesIO(lowpass100_filt))

    with open('models/pli50_filt.enc', 'rb') as encrypted_file:
        pli50_filt = encrypted_file.read()
    pli50_filt = cipher_suite.decrypt(pli50_filt)
    pli50_filt = joblib.load(BytesIO(pli50_filt))

    with open('models/pli60_filt.enc', 'rb') as encrypted_file:
        pli60_filt = encrypted_file.read()
    pli60_filt = cipher_suite.decrypt(pli60_filt)
    pli60_filt = joblib.load(BytesIO(pli60_filt))

    with open('models/lowpass150_filt.enc', 'rb') as encrypted_file:
        lowpass150_filt = encrypted_file.read()
    lowpass150_filt = cipher_suite.decrypt(lowpass150_filt)
    lowpass150_filt = joblib.load(BytesIO(lowpass150_filt))

    with open('models/pli100_filt.enc', 'rb') as encrypted_file:
        pli100_filt = encrypted_file.read()
    pli100_filt = cipher_suite.decrypt(pli100_filt)
    pli100_filt = joblib.load(BytesIO(pli100_filt))

    with open('models/pli120_filt.enc', 'rb') as encrypted_file:
        pli120_filt = encrypted_file.read()
    pli120_filt = cipher_suite.decrypt(pli120_filt)
    pli120_filt = joblib.load(BytesIO(pli120_filt))

    return rf_148, rf_417, baseline_filt, lowpass100_filt, pli50_filt, pli60_filt, lowpass150_filt, pli100_filt, pli120_filt

def save_only_predictions():
    save_features = False
    save_predictions = True
    process_ecgs(save_features, save_predictions)

def save_only_features():
    save_features = True
    save_predictions = False
    process_ecgs(save_features, save_predictions)

def save_both():
    save_features = True
    save_predictions = True
    process_ecgs(save_features, save_predictions)

def browse():
    global folder
    folder = filedialog.askdirectory()
    os.makedirs('results', exist_ok=True)

    # Check if the folder contains .xml or .json files
    num_files = len(glob(folder + '/*.xml')) + len(glob(folder + '/*.json'))

    if num_files == 0:
        text1.delete('1.0', tk.END)
        text1.insert('end', 'No .xml or .json ECG files found in the selected folder.')
        root.update()
    else:
        text1.delete('1.0', tk.END)
        text1.insert('end', 'Folder: ' + folder)
        root.update()

    button_frame.pack(side="top", pady=10)
    
def save_csv():

    filenames = glob('results/*predictions.txt')
    preds1 = []
    preds2 = []

    for file in filenames:
        with open(file, 'r') as f:
            lines = f.readlines()
        
        preds1.append(float(lines[0]))
        preds2.append(float(lines[1]))

    if len(filenames) > 0:
        filenames = [file.split('\\')[-1].split('.')[0] for file in filenames]
        df = pd.DataFrame(columns=['File','rf_148', 'rf_417'])
        df['File'] = filenames
        df['rf_148'] = preds1
        df['rf_417'] = preds2
        df.to_csv('predictions.csv', index=False)


    filenames = glob('results/*features.txt')
    feature_names = pd.read_csv('models/feature_names.csv')['feature'].to_list()

    feature_values = []

    for file in filenames:
        with open(file, 'r') as f:
            data = f.read()
        data = data.replace('\n', '').replace('[', '').replace(']', '').split(' ')
        data = [float(i) for i in data if i != '']
        feature_values.append(data)

    feature_values = np.array(feature_values)
        
    if len(filenames) > 0:
        filenames = [file.split('\\')[-1].split('.')[0] for file in filenames]
        df = pd.DataFrame(columns=['File'] + feature_names)
        df['File'] = filenames
        for i in range(len(feature_names)):
            df[feature_names[i]] = feature_values[:,i]
        df.to_csv('features.csv', index=False)

    text1.delete('1.0', tk.END)
    text1.insert('end', 'CSV Generated')
    root.update()

#################################################################################
if __name__ == '__main__':
    root = tk.Tk()
    root.title('ECG SMART')
    root.geometry('800x250')
    root.resizable(False, False)

    try:
        response = requests.get("http://worldtimeapi.org/api/ip")

        if response.status_code == 200:
            # Parse the JSON response to get the current time
            data = response.json()
            current_time_str = data['datetime']
            # Convert the string representation to a datetime object
            current_time = datetime.fromisoformat(current_time_str)  # Remove the 'Z' at the end
            current_time = datetime(current_time.year, current_time.month, current_time.day)

        if current_time < datetime(2024, 11, 5):
            text = tk.Text(root, height=2)
            text.pack(side='top')
            text.insert('end', 'Program expires October 05 2024.')
            text.config(state='disabled')

            text2 = tk.Text(root, height=2)
            text2.pack(side='top')
            text2.insert('end', 'Select the ECG folder to be analyzed.')
            text2.config(state='disabled')

            button_frame1 = tk.Frame(root)
            button_frame1.pack(side="top", pady=10)

            btn = tk.Button(button_frame1, text='Select ECG Folder', command=browse)
            btn.pack(side='left', padx=5)

            btn4 = tk.Button(button_frame1, text='Save CSV', command=save_csv)
            btn4.pack(side='left', padx=5)

            text1 = tk.Text(root, height=5)
            text1.pack(side='top')

            button_frame = tk.Frame(root)
            button_frame.pack(side="top", pady=10)

            btn1 = tk.Button(button_frame, text="Generate Features & Predictions", command=save_both)
            btn1.pack(side="left", padx=5)

            btn2 = tk.Button(button_frame, text='Generate Features', command=save_only_features)
            btn2.pack(side='left', padx=5)

            btn3 = tk.Button(button_frame, text='Generate Predictions', command=save_only_predictions)
            btn3.pack(side='left', padx=5)

            # hide button frame
            button_frame.pack_forget()

        else:
            text = tk.Text(root, height=2)
            text.pack(side='top')
            text.insert('end', 'The program has expired')
            text.config(state='disabled')

    except Exception as e:
        text = tk.Text(root, height=2)
        text.pack(side='top')
        text.insert('end', 'ECG SMART needs an internet connection to check the time.')
        text.config(state='disabled')
        # terminate program

    root.mainloop()