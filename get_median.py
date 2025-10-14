# Calculate median beats and rr intervals

import numpy as np
from scipy import signal

def getTemplateBeat(beats):
    beats1 = np.empty((beats.shape[0], beats.shape[1], 200))
    for i in range(beats.shape[0]):
        for j in range(beats.shape[1]):
            peak = np.argmax(np.abs(beats[i, j, 285:315])) + 285
            beats1[i, j, :] = beats[i, j, peak - 100:peak + 100]
    
    beats1 -= np.median(beats1[:,:,0:75],axis=2)[:,:,None]
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
                    corrs[i,j,k] = 1000
                    corrs[i,k,j] = 1000
                else:
                    corrs[i,j,k] = np.linalg.norm(beats1[j,i,:] - beats1[k,i,:]) / mags[i]
                    corrs[i,k,j] = corrs[i,j,k]
        best_beat[i] = np.argmin(np.sum(corrs[i,:,:], axis=-1))

    best_corrs = np.zeros([12,beats1.shape[0]])
    for i in range(12):
        best_corrs[i] = corrs[i,best_beat[i],:]
    
    best_overall_beat = np.argmin(np.sum(best_corrs, axis=0))

    return best_overall_beat

def getValidBeats(beats, template_beat):
    beats = beats[:,:,200:400]
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
                    corrs[i,j,k] = 1000
                    corrs[i,k,j] = 1000
                else:
                    corrs[i,j,k] = np.linalg.norm(beats[j,i,:] - beats[k,i,:]) / mags[i]
                    corrs[i,k,j] = corrs[i,j,k]
        best_beat[i] = np.argmin(np.sum(corrs[i,:,:], axis=-1))

    best_corrs = np.zeros([12,beats.shape[0]])
    for i in range(12):
        best_corrs[i] = corrs[i,best_beat[i],:]
    
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
    b_qrs, a_qrs = signal.butter(2, [8, 40], btype='bandpass', fs=500)
    ecg_qrs = signal.filtfilt(b_qrs, a_qrs, ecg, axis=1, padtype='even', padlen=1000)

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