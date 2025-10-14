# calculate features from the median beats

import numpy as np
from scipy import signal, integrate
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform, cdist, euclidean
import itertools

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

########################################################################################################################

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
    if get_qrs_amp_global(fiducials, ecg_rms) != 0:
        return get_t_amp_global(fiducials, ecg_rms) / get_qrs_amp_global(fiducials, ecg_rms)
    return 0

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
    if np.var(ecg) == 0:
        return 0
    if ecg.shape[1] >= 8:
        pca = PCA(n_components=2)
        pca.fit(ecg.T)
        vr = pca.explained_variance_
        if vr[0] != 0:
            return vr[1]/vr[0]
    return 0

def get_ndpv(ecg): # plug in data to use (QRS or STT)
    ecg = ecg[[0,1,6,7,8,9,10,11],:]
    if np.var(ecg) == 0:
        return 0
    if ecg.shape[1] >= 8:
        pca = PCA(n_components=8)
        pca.fit(ecg.T)
        vr = pca.explained_variance_
        if np.sum(vr)!= 0:
            return np.sum(vr[3:])/np.sum(vr)
    return 0

def get_rel_psds(ecg, fs): # use 10-second data
    f, psd = signal.welch(ecg, fs=fs, nperseg=int(1.25*fs))

    freq_idx = np.where((f >= 0.5) & (f < 10))[0]
    low_psd = np.trapezoid(psd[:,freq_idx], f[freq_idx], axis=1)

    freq_idx = np.where((f >= 10) & (f < 50))[0]
    medium_psd = np.trapezoid(psd[:,freq_idx], f[freq_idx], axis=1)

    freq_idx = np.where((f >= 50) & (f < 100))[0]
    high_psd = np.trapezoid(psd[:,freq_idx], f[freq_idx], axis=1)

    freq_idx = np.where((f >= 0.5) & (f < 100))[0]
    total_psd = np.trapezoid(psd[:,freq_idx], f[freq_idx], axis=1)

    total_psd[total_psd==0] = 1
    low_psd /= total_psd
    medium_psd /= total_psd
    high_psd /= total_psd

    if np.sum(low_psd) == 0:
        low_psd = 0
    else:
        low_psd = np.mean(low_psd[low_psd > 0])
    if np.sum(medium_psd) == 0:
        medium_psd = 0
    else:
        medium_psd = np.mean(medium_psd[medium_psd > 0])
    if np.sum(high_psd) == 0:
        high_psd = 0
    else:
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
        
########################################################################################################################

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

    if qrs_peak_x == 0:
        qrs_elev = 0
        qrs_azim = 0
    else:
        qrs_elev = np.rad2deg(np.arctan(qrs_peak_z/qrs_peak_x)) # z/x
        qrs_azim = np.rad2deg(np.arctan(qrs_peak_y/qrs_peak_x)) # y/x

    if qrs_peak_y == 0:
        qrs_zen = 0
    else:
        qrs_zen = np.rad2deg(np.arctan(qrs_peak_z/qrs_peak_y)) # z/y

    if qrs_peak_x == 0 and qrs_peak_y == 0:
        qrs_elev1 = 0
    else:
        qrs_elev1 = np.rad2deg(np.arctan(qrs_peak_z/np.sqrt(np.square(qrs_peak_x) + np.square(qrs_peak_y))))
    
    qrs_mag = np.sqrt(np.square(qrs_peak_x) + np.square(qrs_peak_y) + np.square(qrs_peak_z))

    return qrs_elev, qrs_elev1, qrs_azim, qrs_zen, qrs_mag

def get_t_peak_features(vcg, tpeak):
    if np.isnan(tpeak):
        return 0, 0, 0, 0, 0
    t_peak_x = vcg[0,tpeak]
    t_peak_y = vcg[1,tpeak]
    t_peak_z = vcg[2,tpeak]

    if t_peak_x == 0:
        t_elev = 0
        t_azim = 0
    else:
        t_elev = np.rad2deg(np.arctan(t_peak_z/t_peak_x)) # z/x
        t_azim = np.rad2deg(np.arctan(t_peak_y/t_peak_x)) # y/x
    
    if t_peak_y == 0:
        t_zen = 0
    else:
        t_zen = np.rad2deg(np.arctan(t_peak_z/t_peak_y)) # z/y
    
    if t_peak_x == 0 and t_peak_y == 0:
        t_elev1 = 0
    else:
        t_elev1 = np.rad2deg(np.arctan(t_peak_z/np.sqrt(np.square(t_peak_x) + np.square(t_peak_y))))
    
    t_mag = np.sqrt(np.square(t_peak_x) + np.square(t_peak_y) + np.square(t_peak_z))

    return t_elev, t_elev1, t_azim, t_zen, t_mag

def get_qrs_avg_features(vcg, fiducials):
    qrs_avg_x = np.mean(vcg[0,int(fiducials[2]):int(fiducials[3])])
    qrs_avg_y = np.mean(vcg[1,int(fiducials[2]):int(fiducials[3])])
    qrs_avg_z = np.mean(vcg[2,int(fiducials[2]):int(fiducials[3])])

    if qrs_avg_x == 0:
        qrs_avg_elev = 0
        qrs_avg_azim = 0
    else:
        qrs_avg_elev = np.rad2deg(np.arctan(qrs_avg_z/qrs_avg_x)) # z/x
        qrs_avg_azim = np.rad2deg(np.arctan(qrs_avg_y/qrs_avg_x)) # y/x
    
    if qrs_avg_y == 0:
        qrs_avg_zen = 0
    else:
        qrs_avg_zen = np.rad2deg(np.arctan(qrs_avg_z/qrs_avg_y))  # z/y
    

    if qrs_avg_x == 0 and qrs_avg_y == 0:
        qrs_avg_elev1 = 0
    else:
        qrs_avg_elev1 = np.rad2deg(np.arctan(qrs_avg_z/np.sqrt(np.square(qrs_avg_x) + np.square(qrs_avg_y))))
    
    qrs_avg_mag = np.sqrt(np.square(qrs_avg_x) + np.square(qrs_avg_y) + np.square(qrs_avg_z))

    return qrs_avg_elev, qrs_avg_elev1, qrs_avg_azim, qrs_avg_zen, qrs_avg_mag

def get_iqrs_avg_features(vcg, fiducials, fs):
    qrs_avg_x = np.mean(vcg[0,int(fiducials[2]):int(fiducials[2]) + int(0.04*fs)])
    qrs_avg_y = np.mean(vcg[1,int(fiducials[2]):int(fiducials[2]) + int(0.04*fs)])
    qrs_avg_z = np.mean(vcg[2,int(fiducials[2]):int(fiducials[2]) + int(0.04*fs)])

    if qrs_avg_x == 0:
        qrs_avg_elev = 0
        qrs_avg_azim = 0
    else:
        qrs_avg_elev = np.rad2deg(np.arctan(qrs_avg_z/qrs_avg_x)) # z/x
        qrs_avg_azim = np.rad2deg(np.arctan(qrs_avg_y/qrs_avg_x)) # y/x
    
    if qrs_avg_y == 0:
        qrs_avg_zen = 0
    else:
        qrs_avg_zen = np.rad2deg(np.arctan(qrs_avg_z/qrs_avg_y))  # z/y
    
    if qrs_avg_x == 0 and qrs_avg_y == 0:
        qrs_avg_elev1 = 0
    else:
        qrs_avg_elev1 = np.rad2deg(np.arctan(qrs_avg_z/np.sqrt(np.square(qrs_avg_x) + np.square(qrs_avg_y))))
    
    qrs_avg_mag = np.sqrt(np.square(qrs_avg_x) + np.square(qrs_avg_y) + np.square(qrs_avg_z))

    return qrs_avg_elev, qrs_avg_elev1, qrs_avg_azim, qrs_avg_zen, qrs_avg_mag

def get_tqrs_avg_features(vcg, fiducials, fs):
    qrs_avg_x = np.mean(vcg[0,int(fiducials[3]) - int(0.04*fs):int(fiducials[3])])
    qrs_avg_y = np.mean(vcg[1,int(fiducials[3]) - int(0.04*fs):int(fiducials[3])])
    qrs_avg_z = np.mean(vcg[2,int(fiducials[3]) - int(0.04*fs):int(fiducials[3])])

    if qrs_avg_x == 0:
        qrs_avg_elev = 0
        qrs_avg_azim = 0
    else:
        qrs_avg_elev = np.rad2deg(np.arctan(qrs_avg_z/qrs_avg_x)) # z/x
        qrs_avg_azim = np.rad2deg(np.arctan(qrs_avg_y/qrs_avg_x)) # y/x
    
    if qrs_avg_y == 0:
        qrs_avg_zen = 0
    else:
        qrs_avg_zen = np.rad2deg(np.arctan(qrs_avg_z/qrs_avg_y))  # z/y

    if qrs_avg_x == 0 and qrs_avg_y == 0:
        qrs_avg_elev1 = 0
    else:
        qrs_avg_elev1 = np.rad2deg(np.arctan(qrs_avg_z/np.sqrt(np.square(qrs_avg_x) + np.square(qrs_avg_y))))
    
    qrs_avg_mag = np.sqrt(np.square(qrs_avg_x) + np.square(qrs_avg_y) + np.square(qrs_avg_z))

    return qrs_avg_elev, qrs_avg_elev1, qrs_avg_azim, qrs_avg_zen, qrs_avg_mag

def get_t_avg_features(vcg, fiducials):
    if np.isnan(fiducials[5]):
        return 0, 0, 0, 0, 0
    t_avg_x = np.mean(vcg[0,int(fiducials[3]):int(fiducials[5])])
    t_avg_y = np.mean(vcg[1,int(fiducials[3]):int(fiducials[5])])
    t_avg_z = np.mean(vcg[2,int(fiducials[3]):int(fiducials[5])])

    if t_avg_x == 0:
        t_avg_elev = 0
        t_avg_azim = 0
    else:
        t_avg_elev = np.rad2deg(np.arctan(t_avg_z/t_avg_x)) # z/x
        t_avg_azim = np.rad2deg(np.arctan(t_avg_y/t_avg_x)) # y/x
    
    if t_avg_y == 0:
        t_avg_zen = 0
    else:
        t_avg_zen = np.rad2deg(np.arctan(t_avg_z/t_avg_y)) # z/y
    
    if t_avg_x == 0 and t_avg_y == 0:
        t_avg_elev1 = 0
    else:
        t_avg_elev1 = np.rad2deg(np.arctan(t_avg_z/np.sqrt(np.square(t_avg_x) + np.square(t_avg_y))))
    
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

    if svg_x == 0:
        svg_elev = 0
        svg_azim = 0
    else:
        svg_elev = np.rad2deg(np.arctan(svg_z/svg_x)) # z/x
        svg_azim = np.rad2deg(np.arctan(svg_y/svg_x)) # y/x

    if svg_y == 0:
        svg_zen = 0
    else:
        svg_zen = np.rad2deg(np.arctan(svg_z/svg_y)) # z/y
    
    if svg_x == 0 and svg_y == 0:
        svg_elev1 = 0
    else:
        svg_elev1 = np.rad2deg(np.arctan(svg_z/np.sqrt(np.square(svg_x) + np.square(svg_y))))
    
    svg_mag = np.sqrt(np.square(svg_x) + np.square(svg_y) + np.square(svg_z))

    return svg_elev, svg_elev1, svg_azim, svg_zen, svg_mag

def get_svg_avg_features(vcg, fiducials):
    if np.isnan(fiducials[5]):
        return 0, 0, 0, 0, 0
    svg_avg_x = np.mean(vcg[0,int(fiducials[2]):int(fiducials[5])])
    svg_avg_y = np.mean(vcg[1,int(fiducials[2]):int(fiducials[5])])
    svg_avg_z = np.mean(vcg[2,int(fiducials[2]):int(fiducials[5])])

    if svg_avg_x == 0:
        svg_avg_elev = 0
        svg_avg_azim = 0
    else:
        svg_avg_elev = np.rad2deg(np.arctan(svg_avg_z/svg_avg_x)) # z/x
        svg_avg_azim = np.rad2deg(np.arctan(svg_avg_y/svg_avg_x)) # y/x
    
    if svg_avg_y == 0:
        svg_avg_zen = 0
    else:
        svg_avg_zen = np.rad2deg(np.arctan(svg_avg_z/svg_avg_y)) # z/y
    
    if svg_avg_x == 0 and svg_avg_y == 0:
        svg_avg_elev1 = 0
    else:
        svg_avg_elev1 = np.rad2deg(np.arctan(svg_avg_z/np.sqrt(np.square(svg_avg_x) + np.square(svg_avg_y))))
    
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

    if qrs_mag == 0 or t_mag == 0:
        return 0

    qrs_peak_norm = [qrs_peak_x/qrs_mag, qrs_peak_y/qrs_mag, qrs_peak_z/qrs_mag]
    t_peak_norm = [t_peak_x/t_mag, t_peak_y/t_mag, t_peak_z/t_mag]
    dotprod = np.dot(qrs_peak_norm, t_peak_norm)
    dotprod = np.clip(dotprod, -1, 1)
    qrst_angle = np.rad2deg(np.arccos(dotprod))

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

    if qrs_avg_mag == 0 or t_avg_mag == 0:
        return 0

    qrs_avg_norm = [qrs_avg_x/qrs_avg_mag, qrs_avg_y/qrs_avg_mag, qrs_avg_z/qrs_avg_mag]
    t_avg_norm = [t_avg_x/t_avg_mag, t_avg_y/t_avg_mag, t_avg_z/t_avg_mag]
    
    dotprod = np.dot(qrs_avg_norm, t_avg_norm)
    dotprod = np.clip(dotprod, -1, 1)

    qrst_avg_angle = np.rad2deg(np.arccos(dotprod))

    return qrst_avg_angle

########################################################################################################################

# Other Features #######################################################################################################
# K point deviation
def get_KPD(fiducials, ecg, ecg_rms):
    if ~np.isnan(fiducials[4]) & ~np.isnan(fiducials[5]):
        tpeak = np.argmax(ecg_rms[int(fiducials[4]):int(fiducials[5])]) + int(fiducials[4])
        rpeak = np.argmax(ecg_rms[int(fiducials[2]):int(fiducials[3])]) + int(fiducials[2])
        return np.min(np.max(np.abs(ecg[:, rpeak:tpeak]), axis=0))
    else:
        return np.nan

# T-Wave Area Curve
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

    area_sig1 = np.trapezoid(sig1)
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

    if np.isnan(asymm_score):
        asymm_score = 0
    if np.isnan(notch_score):
        notch_score = 0
    if np.isnan(flatness_score):
        flatness_score = 0
    if np.isnan(mcs):
        mcs = 0

    return asymm_score, notch_score, flatness_score, mcs

# Calculate the features #######################################################################################################
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

    