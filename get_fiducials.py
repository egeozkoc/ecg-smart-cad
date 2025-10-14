# calculate fiducials from median beats

import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from scipy import signal
import multiprocessing as mp

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

    dmin = np.inf
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
        best_window = np.inf
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